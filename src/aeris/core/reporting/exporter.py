"""
AERIS Safe Report Exporter & Pipeline Orchestrator Integration.

Provides safe, deterministic, atomic export of post-flight engineering reports
to GitHub-Flavored Markdown (.md, .markdown) and self-contained HTML (.html, .htm).

Key capabilities:
- Enforces strict path validation, directory containment, and traversal rejection.
- Safe, atomic file writing with automatic temporary file cleanup on failure.
- Accidental overwrite protection by default (overwrite=False).
- Direct extraction and export from completed DataPipelineResult instances.
- Structured, typed ReportExportResult with status, byte counts, and provenance.
- Absolute preservation of pipeline results and report domain models (no mutation).
- Zero external runtime server or browser dependencies (no FastAPI, Dash, or PDF binaries).
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, Field

from aeris.core.provenance.tracker import (
    EpistemicLevel,
    compute_file_checksum,
    create_provenance,
)
from aeris.core.reporting.extractor import ReportDataExtractor
from aeris.core.reporting.html_renderer import HtmlReportRenderer
from aeris.core.reporting.markdown_renderer import MarkdownReportRenderer
from aeris.core.reporting.models import EngineeringReport
from aeris.core.results.types import (
    BasePipelineResult,
    Issue,
    IssueSeverity,
    ResultStatus,
)

if TYPE_CHECKING:
    from aeris.core.results.types import DataPipelineResult

# ---------------------------------------------------------------------------
# Supported Formats & Extension Mappings
# ---------------------------------------------------------------------------


class ReportExportFormat(StrEnum):
    """Supported output formats for post-flight engineering reports."""

    MARKDOWN = "markdown"
    """GitHub-Flavored Markdown document (.md, .markdown)."""

    HTML = "html"
    """Self-contained, interactive HTML document (.html, .htm)."""

    @classmethod
    def from_str(cls, val: str | ReportExportFormat) -> ReportExportFormat:
        """Parse a string or enum instance into a valid ReportExportFormat."""
        if isinstance(val, cls):
            return val
        s = str(val).strip().lower()
        if s in ("markdown", "md"):
            return cls.MARKDOWN
        if s in ("html", "htm"):
            return cls.HTML
        raise ValueError(
            f"Unsupported report export format: {val!r}. Expected 'markdown' or 'html'."
        )


VALID_EXTENSIONS: dict[ReportExportFormat, tuple[str, ...]] = {
    ReportExportFormat.MARKDOWN: (".md", ".markdown"),
    ReportExportFormat.HTML: (".html", ".htm"),
}


# ---------------------------------------------------------------------------
# Export Result Domain Model
# ---------------------------------------------------------------------------


class ReportExportResult(BasePipelineResult):
    """
    Result container for post-flight engineering report export operations.

    Inherits from BasePipelineResult to maintain consistent status flags,
    issue diagnostics, and epistemic lineage across the AERIS platform.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    output_path: pathlib.Path | None = Field(
        default=None,
        description="Resolved absolute output path where report was written, or None if failed.",
    )
    format: ReportExportFormat | None = Field(
        default=None,
        description="Format of exported report, or None if format resolution failed.",
    )
    bytes_written: int = Field(
        default=0,
        ge=0,
        description="Total number of bytes written to the output file on disk.",
    )


# ---------------------------------------------------------------------------
# Path & Format Validation Helpers
# ---------------------------------------------------------------------------


def _validate_and_resolve_path(
    output_path: str | pathlib.Path,
    format: ReportExportFormat | str | None,
    base_dir: str | pathlib.Path | None = None,
) -> tuple[pathlib.Path, ReportExportFormat]:
    """
    Validate, resolve, and confine a report output path and export format.

    Parameters
    ----------
    output_path : str | pathlib.Path
        Target destination path for the report file.
    format : ReportExportFormat | str | None
        Explicit format or None to infer format from output_path extension.
    base_dir : str | pathlib.Path | None
        Optional base directory constraint to prevent directory traversal.

    Returns
    -------
    tuple[pathlib.Path, ReportExportFormat]
        The resolved absolute Path and the validated ReportExportFormat.

    Raises
    ------
    ValueError
        If the path is empty, contains directory traversal tokens escaping base_dir,
        has an unsupported/mismatched file extension, or points to an existing directory.
    """
    if output_path is None:
        raise ValueError("Output path cannot be None.")

    raw_path_str = str(output_path).strip()
    if not raw_path_str:
        raise ValueError("Output path cannot be empty or whitespace.")

    path_obj = pathlib.Path(raw_path_str)

    # 1. Traversal and containment check
    if base_dir is not None:
        base_path = pathlib.Path(base_dir).resolve()
        if path_obj.is_absolute():
            resolved = path_obj.resolve()
        else:
            resolved = (base_path / path_obj).resolve()

        try:
            resolved.relative_to(base_path)
        except ValueError as err:
            raise ValueError(
                f"Path traversal detected: {output_path} resolves outside base directory {base_dir}"
            ) from err
    else:
        # If no explicit base_dir is supplied, check for explicit '..' traversal parts
        if ".." in path_obj.parts:
            raise ValueError(
                f"Path traversal detected: traversal components ('..') are not permitted: "
                f"{output_path}"
            )
        resolved = path_obj.resolve()

    # 2. Directory check
    if resolved.exists() and resolved.is_dir():
        raise ValueError(
            f"Target path points to an existing directory, not a report file: {resolved}"
        )

    # 3. Format and extension validation
    suffix = resolved.suffix.lower()

    if format is not None:
        resolved_format = ReportExportFormat.from_str(format)
        valid_exts = VALID_EXTENSIONS[resolved_format]
        if suffix not in valid_exts:
            raise ValueError(
                f"File extension '{suffix}' does not match format '{resolved_format.value}'. "
                f"Expected one of: {valid_exts}"
            )
    else:
        # Infer format from extension
        if suffix in VALID_EXTENSIONS[ReportExportFormat.MARKDOWN]:
            resolved_format = ReportExportFormat.MARKDOWN
        elif suffix in VALID_EXTENSIONS[ReportExportFormat.HTML]:
            resolved_format = ReportExportFormat.HTML
        else:
            all_valid = (
                VALID_EXTENSIONS[ReportExportFormat.MARKDOWN]
                + VALID_EXTENSIONS[ReportExportFormat.HTML]
            )
            raise ValueError(
                f"Cannot infer report export format from file extension '{suffix}'. "
                f"Explicitly specify format or use an extension in: {all_valid}"
            )

    return resolved, resolved_format


def _atomic_write_file(
    target_path: pathlib.Path,
    content: str,
    overwrite: bool,
) -> int:
    """
    Safely and atomically write content to target_path using a temporary file.

    Parameters
    ----------
    target_path : pathlib.Path
        The destination absolute path.
    content : str
        The textual report content to write.
    overwrite : bool
        Whether to overwrite an existing file.

    Returns
    -------
    int
        The number of bytes written to disk.

    Raises
    ------
    FileExistsError
        If target_path already exists and overwrite is False.
    OSError
        If a filesystem write or replacement error occurs.
    """
    if target_path.exists() and not overwrite:
        raise FileExistsError(
            f"Target file already exists and overwrite is disabled: {target_path}"
        )

    parent_dir = target_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    temp_path: pathlib.Path | None = None
    try:
        # Use same directory to ensure atomic os.replace across filesystem boundaries
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent_dir,
            prefix=".aeris_export_tmp_",
            suffix=target_path.suffix,
            delete=False,
        ) as temp_file:
            temp_path = pathlib.Path(temp_file.name)
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        # Atomic replace (Windows-safe after file descriptor is closed)
        os.replace(temp_path, target_path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            with contextlib.suppress(OSError):
                temp_path.unlink()
        raise

    return target_path.stat().st_size


# ---------------------------------------------------------------------------
# Report Exporter Engine
# ---------------------------------------------------------------------------


class ReportExporter:
    """
    Safe post-flight engineering report exporter.

    Coordinates path validation, format rendering, and atomic file replacement
    to ensure reports are exported reliably without partial or corrupted files.
    """

    def export(
        self,
        report: EngineeringReport,
        output_path: str | pathlib.Path,
        format: ReportExportFormat | str | None = None,
        *,
        figures: Sequence[Any] | Mapping[str, Any] | None = None,
        include_plotlyjs: bool | str = False,
        overwrite: bool = False,
        base_dir: str | pathlib.Path | None = None,
        raise_on_error: bool = False,
    ) -> ReportExportResult:
        """
        Export an EngineeringReport domain model to disk.

        Parameters
        ----------
        report : EngineeringReport
            The authoritative post-flight engineering report domain model.
        output_path : str | pathlib.Path
            Target file destination on the filesystem.
        format : ReportExportFormat | str | None
            Format to export ('markdown' or 'html'). If None, inferred from extension.
        figures : Sequence[Any] | Mapping[str, Any] | None
            Optional Plotly figures or TimeSeriesPlotResult objects to embed (HTML format only).
        include_plotlyjs : bool | str
            Plotly.js embedding mode (False, 'inline', or 'cdn'). Default False for offline safety.
        overwrite : bool
            Whether to overwrite an existing file at output_path. Default False.
        base_dir : str | pathlib.Path | None
            Optional base directory constraint to prevent directory traversal.
        raise_on_error : bool
            If True, raise underlying validation or I/O exceptions instead of returning
            an error ReportExportResult. Default False.

        Returns
        -------
        ReportExportResult
            Structured result with execution status, resolved path, bytes, and provenance.
        """
        # Step 1: Validate and resolve path and format
        try:
            resolved_path, resolved_format = _validate_and_resolve_path(
                output_path=output_path,
                format=format,
                base_dir=base_dir,
            )
        except ValueError as err:
            if raise_on_error:
                raise
            issue = Issue(
                issue_code="ERR_PATH_VALIDATION",
                severity=IssueSeverity.ERROR,
                message=str(err),
                context={"output_path": str(output_path)},
            )
            return ReportExportResult(
                status=ResultStatus.ERROR,
                output_path=None,
                format=None,
                bytes_written=0,
                issues=[issue],
            )

        # Step 2: Render content deterministically
        try:
            if resolved_format == ReportExportFormat.MARKDOWN:
                content = MarkdownReportRenderer.render(report)
            elif resolved_format == ReportExportFormat.HTML:
                content = HtmlReportRenderer.render(
                    report=report,
                    figures=figures,
                    include_plotlyjs=include_plotlyjs,
                )
            else:
                raise ValueError(f"Unsupported format: {resolved_format}")
        except Exception as err:
            if raise_on_error:
                raise
            issue = Issue(
                issue_code="ERR_REPORT_RENDER",
                severity=IssueSeverity.ERROR,
                message=f"Failed to render {resolved_format.value} report: {err}",
                context={"format": resolved_format.value},
            )
            return ReportExportResult(
                status=ResultStatus.ERROR,
                output_path=None,
                format=resolved_format,
                bytes_written=0,
                issues=[issue],
            )

        # Step 3: Atomic file write
        try:
            bytes_written = _atomic_write_file(
                target_path=resolved_path,
                content=content,
                overwrite=overwrite,
            )
        except FileExistsError as err:
            if raise_on_error:
                raise
            issue = Issue(
                issue_code="ERR_FILE_EXISTS",
                severity=IssueSeverity.ERROR,
                message=str(err),
                context={"target_path": str(resolved_path)},
            )
            return ReportExportResult(
                status=ResultStatus.ERROR,
                output_path=None,
                format=resolved_format,
                bytes_written=0,
                issues=[issue],
            )
        except PermissionError as err:
            if raise_on_error:
                raise
            issue = Issue(
                issue_code="ERR_PERMISSION_DENIED",
                severity=IssueSeverity.ERROR,
                message=f"Permission denied writing report: {err}",
                context={"target_path": str(resolved_path)},
            )
            return ReportExportResult(
                status=ResultStatus.ERROR,
                output_path=None,
                format=resolved_format,
                bytes_written=0,
                issues=[issue],
            )
        except OSError as err:
            if raise_on_error:
                raise
            issue = Issue(
                issue_code="ERR_WRITE_FAILED",
                severity=IssueSeverity.ERROR,
                message=f"Filesystem write failed: {err}",
                context={"target_path": str(resolved_path)},
            )
            return ReportExportResult(
                status=ResultStatus.ERROR,
                output_path=None,
                format=resolved_format,
                bytes_written=0,
                issues=[issue],
            )

        # Step 4: Attach provenance metadata
        source_checksum = compute_file_checksum(resolved_path)
        parent_records: list[str] = []
        if report.provenance.root_provenance_id:
            parent_records.append(report.provenance.root_provenance_id)

        provenance = create_provenance(
            source_uri=str(resolved_path),
            activity_name="ReportExporter",
            epistemic_level=EpistemicLevel.CALCULATED,
            source_checksum=source_checksum,
            parameters={
                "format": resolved_format.value,
                "bytes_written": bytes_written,
                "overwrite": overwrite,
                "report_id": report.metadata.report_id,
            },
            parent_record_ids=parent_records,
        )

        return ReportExportResult(
            status=ResultStatus.SUCCESS,
            output_path=resolved_path,
            format=resolved_format,
            bytes_written=bytes_written,
            issues=[],
            provenance=provenance,
            metadata={
                "report_id": report.metadata.report_id,
                "format": resolved_format.value,
                "bytes_written": bytes_written,
            },
        )

    def export_pipeline_result(
        self,
        pipeline_result: DataPipelineResult,
        output_path: str | pathlib.Path,
        format: ReportExportFormat | str | None = None,
        *,
        figures: Sequence[Any] | Mapping[str, Any] | None = None,
        include_plotlyjs: bool | str = False,
        overwrite: bool = False,
        base_dir: str | pathlib.Path | None = None,
        report_title: str = "AERIS Flight Test Engineering Report",
        analyst_name: str | None = None,
        custom_metadata: dict[str, Any] | None = None,
        raise_on_error: bool = False,
    ) -> ReportExportResult:
        """
        Extract an EngineeringReport domain model from a DataPipelineResult and export it.

        Does NOT mutate or rerun the pipeline. Preserves all pipeline stage statuses,
        anomalies, and limitations.

        Parameters
        ----------
        pipeline_result : DataPipelineResult
            The completed or partially completed pipeline result.
        output_path : str | pathlib.Path
            Target file destination on the filesystem.
        format : ReportExportFormat | str | None
            Format to export ('markdown' or 'html'). If None, inferred from extension.
        figures : Sequence[Any] | Mapping[str, Any] | None
            Optional Plotly figures or TimeSeriesPlotResult objects to embed (HTML format only).
        include_plotlyjs : bool | str
            Plotly.js embedding mode (False, 'inline', or 'cdn'). Default False.
        overwrite : bool
            Whether to overwrite an existing file at output_path. Default False.
        base_dir : str | pathlib.Path | None
            Optional base directory constraint to prevent directory traversal.
        report_title : str
            Title for the extracted report. Default 'AERIS Flight Test Engineering Report'.
        analyst_name : str | None
            Name of responsible engineering analyst.
        custom_metadata : dict[str, Any] | None
            Additional operational metadata to embed in the report.
        raise_on_error : bool
            If True, raise underlying validation or I/O exceptions instead of returning
            an error ReportExportResult. Default False.

        Returns
        -------
        ReportExportResult
            Structured result with execution status, resolved path, bytes, and provenance.
        """
        # Step 1: Extract domain model without mutating pipeline_result
        try:
            report = ReportDataExtractor.extract(
                pipeline_result=pipeline_result,
                report_title=report_title,
                analyst_name=analyst_name,
                custom_metadata=custom_metadata,
            )
        except Exception as err:
            if raise_on_error:
                raise
            issue = Issue(
                issue_code="ERR_REPORT_EXTRACTION",
                severity=IssueSeverity.ERROR,
                message=f"Failed to extract report from pipeline result: {err}",
            )
            return ReportExportResult(
                status=ResultStatus.ERROR,
                output_path=None,
                format=None,
                bytes_written=0,
                issues=[issue],
            )

        # Step 2: Delegate to export
        return self.export(
            report=report,
            output_path=output_path,
            format=format,
            figures=figures,
            include_plotlyjs=include_plotlyjs,
            overwrite=overwrite,
            base_dir=base_dir,
            raise_on_error=raise_on_error,
        )


# ---------------------------------------------------------------------------
# Convenience Functional Entrypoints
# ---------------------------------------------------------------------------


def export_report(
    report: EngineeringReport,
    output_path: str | pathlib.Path,
    format: ReportExportFormat | str | None = None,
    *,
    figures: Sequence[Any] | Mapping[str, Any] | None = None,
    include_plotlyjs: bool | str = False,
    overwrite: bool = False,
    base_dir: str | pathlib.Path | None = None,
    raise_on_error: bool = False,
) -> ReportExportResult:
    """
    Convenience function to export an EngineeringReport domain model to Markdown or HTML.

    Parameters
    ----------
    report : EngineeringReport
        The post-flight engineering report domain model.
    output_path : str | pathlib.Path
        Target file path on disk.
    format : ReportExportFormat | str | None
        Export format ('markdown' or 'html'). Inferred from extension if None.
    figures : Sequence[Any] | Mapping[str, Any] | None
        Optional figures to embed in HTML output.
    include_plotlyjs : bool | str
        Plotly.js embedding mode (False, 'inline', 'cdn').
    overwrite : bool
        Whether to overwrite an existing target file. Default False.
    base_dir : str | pathlib.Path | None
        Optional base directory constraint to prevent directory traversal.
    raise_on_error : bool
        If True, raise exceptions on failure instead of returning an error result.

    Returns
    -------
    ReportExportResult
        Structured export result.
    """
    exporter = ReportExporter()
    return exporter.export(
        report=report,
        output_path=output_path,
        format=format,
        figures=figures,
        include_plotlyjs=include_plotlyjs,
        overwrite=overwrite,
        base_dir=base_dir,
        raise_on_error=raise_on_error,
    )


def export_pipeline_report(
    pipeline_result: DataPipelineResult,
    output_path: str | pathlib.Path,
    format: ReportExportFormat | str | None = None,
    *,
    figures: Sequence[Any] | Mapping[str, Any] | None = None,
    include_plotlyjs: bool | str = False,
    overwrite: bool = False,
    base_dir: str | pathlib.Path | None = None,
    report_title: str = "AERIS Flight Test Engineering Report",
    analyst_name: str | None = None,
    custom_metadata: dict[str, Any] | None = None,
    raise_on_error: bool = False,
) -> ReportExportResult:
    """
    Convenience function to extract and export an EngineeringReport from a DataPipelineResult.

    Parameters
    ----------
    pipeline_result : DataPipelineResult
        The pipeline result to extract from.
    output_path : str | pathlib.Path
        Target file path on disk.
    format : ReportExportFormat | str | None
        Export format ('markdown' or 'html'). Inferred from extension if None.
    figures : Sequence[Any] | Mapping[str, Any] | None
        Optional figures to embed in HTML output.
    include_plotlyjs : bool | str
        Plotly.js embedding mode (False, 'inline', 'cdn').
    overwrite : bool
        Whether to overwrite an existing target file. Default False.
    base_dir : str | pathlib.Path | None
        Optional base directory constraint to prevent directory traversal.
    report_title : str
        Report title. Default 'AERIS Flight Test Engineering Report'.
    analyst_name : str | None
        Name of responsible analyst.
    custom_metadata : dict[str, Any] | None
        Optional custom metadata dictionary.
    raise_on_error : bool
        If True, raise exceptions on failure instead of returning an error result.

    Returns
    -------
    ReportExportResult
        Structured export result.
    """
    exporter = ReportExporter()
    return exporter.export_pipeline_result(
        pipeline_result=pipeline_result,
        output_path=output_path,
        format=format,
        figures=figures,
        include_plotlyjs=include_plotlyjs,
        overwrite=overwrite,
        base_dir=base_dir,
        report_title=report_title,
        analyst_name=analyst_name,
        custom_metadata=custom_metadata,
        raise_on_error=raise_on_error,
    )


# Alias for explicit domain clarity
export_engineering_report = export_report
