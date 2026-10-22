"""
AERIS CSV Ingestion Engine.

Responsible for safely loading raw CSV flight-test and telemetry files into
the AERIS pipeline. Adheres strictly to the Raw Data Principle: reads source
bytes into a structured container without modifying, reordering, imputing, or
validating the underlying telemetry data.

Epistemic status of ingested output: OBSERVED.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pandas as pd

from aeris.core.provenance.tracker import (
    EpistemicLevel,
    compute_file_checksum,
    create_provenance,
)
from aeris.core.results.types import (
    IngestionResult,
    Issue,
    IssueSeverity,
    ResultStatus,
)


class CSVIngestor:
    """Ingests raw CSV flight-test and telemetry data without modification.

    Reads tabular CSV data from the filesystem, attaches execution provenance
    (including SHA-256 digest), and produces an IngestionResult.
    """

    def ingest(self, file_path: str | pathlib.Path, **kwargs: Any) -> IngestionResult:
        """Ingest a CSV file from file_path into an IngestionResult.

        Parameters
        ----------
        file_path : str | pathlib.Path
            Filesystem path to the CSV file to be ingested.
        **kwargs : Any
            Optional arguments passed directly to pd.read_csv if needed.

        Returns
        -------
        IngestionResult
            Structured result containing the loaded DataFrame (if successful),
            row/column counts, issues (if errors occurred), and provenance.
        """
        path = pathlib.Path(file_path)

        # 1. Error case: File does not exist
        if not path.exists():
            return IngestionResult(
                status=ResultStatus.ERROR,
                source_path=str(path),
                issues=[
                    Issue(
                        issue_code="ERR_FILE_NOT_FOUND",
                        severity=IssueSeverity.ERROR,
                        message=f"CSV file does not exist: {path}",
                        location=str(path),
                    )
                ],
            )

        # 2. Error case: Path is a directory
        if path.is_dir():
            return IngestionResult(
                status=ResultStatus.ERROR,
                source_path=str(path),
                issues=[
                    Issue(
                        issue_code="ERR_PATH_IS_DIRECTORY",
                        severity=IssueSeverity.ERROR,
                        message=f"Specified path is a directory, not a CSV file: {path}",
                        location=str(path),
                    )
                ],
            )

        # 3. Error case: Empty file (0 bytes)
        try:
            file_size = path.stat().st_size
        except OSError as exc:
            return IngestionResult(
                status=ResultStatus.ERROR,
                source_path=str(path),
                issues=[
                    Issue(
                        issue_code="ERR_FILE_ACCESS",
                        severity=IssueSeverity.ERROR,
                        message=f"Unable to access file stats: {exc}",
                        location=str(path),
                    )
                ],
            )

        if file_size == 0:
            return IngestionResult(
                status=ResultStatus.ERROR,
                source_path=str(path),
                issues=[
                    Issue(
                        issue_code="ERR_EMPTY_FILE",
                        severity=IssueSeverity.ERROR,
                        message=f"CSV file is completely empty (0 bytes): {path}",
                        location=str(path),
                    )
                ],
            )

        # 4. Compute file checksum for deterministic provenance
        try:
            checksum = compute_file_checksum(path)
        except (OSError, FileNotFoundError) as exc:
            return IngestionResult(
                status=ResultStatus.ERROR,
                source_path=str(path),
                issues=[
                    Issue(
                        issue_code="ERR_CHECKSUM_FAILED",
                        severity=IssueSeverity.ERROR,
                        message=f"Failed to compute file checksum: {exc}",
                        location=str(path),
                    )
                ],
            )

        # 5. Read data with pandas preserving raw content
        try:
            df = pd.read_csv(path, **kwargs)
        except pd.errors.EmptyDataError as exc:
            return IngestionResult(
                status=ResultStatus.ERROR,
                source_path=str(path),
                issues=[
                    Issue(
                        issue_code="ERR_EMPTY_FILE",
                        severity=IssueSeverity.ERROR,
                        message=f"CSV file contains no columns or data to parse: {exc}",
                        location=str(path),
                    )
                ],
            )
        except pd.errors.ParserError as exc:
            return IngestionResult(
                status=ResultStatus.ERROR,
                source_path=str(path),
                issues=[
                    Issue(
                        issue_code="ERR_CSV_PARSE_FAILED",
                        severity=IssueSeverity.ERROR,
                        message=f"CSV parser error encountered: {exc}",
                        location=str(path),
                    )
                ],
            )
        except (UnicodeDecodeError, OSError) as exc:
            return IngestionResult(
                status=ResultStatus.ERROR,
                source_path=str(path),
                issues=[
                    Issue(
                        issue_code="ERR_CSV_READ_FAILED",
                        severity=IssueSeverity.ERROR,
                        message=f"Failed to read CSV content: {exc}",
                        location=str(path),
                    )
                ],
            )

        # 6. Check for header-only or zero-column empty parse
        if len(df.columns) == 0:
            return IngestionResult(
                status=ResultStatus.ERROR,
                source_path=str(path),
                issues=[
                    Issue(
                        issue_code="ERR_EMPTY_FILE",
                        severity=IssueSeverity.ERROR,
                        message="CSV file parsed to 0 columns; no structured data found",
                        location=str(path),
                    )
                ],
            )

        # 7. Construct provenance record for raw ingestion
        provenance = create_provenance(
            source_uri=str(path.resolve()),
            activity_name="csv_ingestion",
            epistemic_level=EpistemicLevel.OBSERVED,
            source_checksum=checksum,
            parameters={"filename": path.name, "size_bytes": file_size},
        )

        # 8. Produce successful IngestionResult
        return IngestionResult(
            status=ResultStatus.SUCCESS,
            data=df,
            row_count=len(df),
            column_names=[str(c) for c in df.columns],
            source_path=str(path.resolve()),
            provenance=provenance,
            metadata={"file_size_bytes": file_size},
        )


def ingest_csv(file_path: str | pathlib.Path, **kwargs: Any) -> IngestionResult:
    """Convenience functional interface for CSVIngestor.ingest()."""
    return CSVIngestor().ingest(file_path, **kwargs)
