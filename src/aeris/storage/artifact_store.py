"""
AERIS Analysis Run Artifact Storage.

Manages persistent serialization, cryptographic checksum verification, atomic writing,
and round-trip reconstruction of DataPipelineResult artifacts in project workspaces.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
from typing import Any

from aeris.core.results.types import DataPipelineResult
from aeris.storage.exceptions import (
    ArtifactCorruptError,
    ArtifactIncompatibleError,
    ArtifactNotFoundError,
    EntityNotFoundError,
    ProjectMismatchError,
    ProjectNotFoundError,
)
from aeris.storage.models import AnalysisRunRecord
from aeris.storage.repository import SQLiteMetadataRepository
from aeris.storage.workspace import WorkspaceManager

ARTIFACT_FORMAT: str = "aeris_pipeline_result_json"
CURRENT_FORMAT_VERSION: str = "1.0"
DEFAULT_ARTIFACT_FILENAME: str = "pipeline_result.json"


def sanitize_pipeline_result_for_storage(result: DataPipelineResult) -> DataPipelineResult:
    """
    Produce a sanitized copy of DataPipelineResult suitable for persistent serialization.

    Strips raw in-memory telemetry DataFrames (which are safely stored in dataset files)
    and removes host-specific absolute paths to protect sensitive directory structures.

    Parameters
    ----------
    result : DataPipelineResult
        The in-memory pipeline execution result.

    Returns
    -------
    DataPipelineResult
        Sanitized result copy with DataFrames set to None and paths normalized.
    """
    updates: dict[str, Any] = {}

    # 1. Sanitize Ingestion stage (drop in-memory DataFrame and host paths)
    if result.ingestion is not None:
        safe_source_path = None
        if result.ingestion.source_path:
            safe_source_path = pathlib.Path(result.ingestion.source_path).name

        safe_ingestion = result.ingestion.model_copy(
            update={
                "data": None,
                "source_path": safe_source_path,
            }
        )
        updates["ingestion"] = safe_ingestion

    # 2. Sanitize Normalization stage (drop in-memory DataFrame)
    if result.normalization is not None:
        safe_normalization = result.normalization.model_copy(
            update={
                "data": None,
            }
        )
        updates["normalization"] = safe_normalization

    # 3. Sanitize Pipeline Root Provenance (strip absolute host paths from source_uri)
    if result.provenance is not None and result.provenance.source_uri:
        raw_uri = result.provenance.source_uri
        if "/" in raw_uri or "\\" in raw_uri:
            safe_uri = pathlib.Path(raw_uri).name
            safe_prov = result.provenance.model_copy(update={"source_uri": safe_uri})
            updates["provenance"] = safe_prov

    if updates:
        return result.model_copy(update=updates)
    return result


def serialize_pipeline_result(
    result: DataPipelineResult,
    analysis_id: str,
    project_id: str,
    dataset_id: str,
) -> tuple[bytes, str]:
    """
    Serialize a DataPipelineResult into canonical, deterministic UTF-8 JSON bytes with envelope.

    Parameters
    ----------
    result : DataPipelineResult
        The pipeline execution result.
    analysis_id : str
        Associated analysis run identifier.
    project_id : str
        Parent project identifier.
    dataset_id : str
        Source dataset identifier.

    Returns
    -------
    tuple[bytes, str]
        (raw_json_bytes, sha256_hex_digest)
    """
    sanitized = sanitize_pipeline_result_for_storage(result)

    envelope: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "format_version": CURRENT_FORMAT_VERSION,
        "analysis_id": analysis_id,
        "project_id": project_id,
        "dataset_id": dataset_id,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "result": sanitized.model_dump(mode="json"),
    }

    raw_json_str = json.dumps(envelope, indent=2, sort_keys=True)
    raw_bytes = raw_json_str.encode("utf-8")
    sha256_checksum = hashlib.sha256(raw_bytes).hexdigest()

    return raw_bytes, sha256_checksum


def deserialize_pipeline_result(
    content_bytes: bytes,
    expected_checksum: str | None = None,
    expected_analysis_id: str | None = None,
    expected_project_id: str | None = None,
) -> DataPipelineResult:
    """
    Deserialize and validate a DataPipelineResult from UTF-8 JSON bytes.

    Performs cryptographic checksum verification, envelope validation, version checking,
    project/run ownership checks, and Pydantic model reconstruction.

    Parameters
    ----------
    content_bytes : bytes
        The raw artifact bytes read from disk.
    expected_checksum : str | None
        Optional SHA-256 hex digest to verify against.
    expected_analysis_id : str | None
        Optional analysis run identifier to match with envelope.
    expected_project_id : str | None
        Optional project identifier to match with envelope.

    Returns
    -------
    DataPipelineResult
        Fully reconstructed pipeline result model.

    Raises
    ------
    ArtifactCorruptError
        If checksum mismatch occurs, JSON syntax is corrupt, or model validation fails.
    ArtifactIncompatibleError
        If format name or version is unsupported.
    ProjectMismatchError
        If artifact project_id does not match expected_project_id.
    """
    # 1. Cryptographic integrity check
    if expected_checksum is not None:
        actual_checksum = hashlib.sha256(content_bytes).hexdigest()
        if actual_checksum.lower() != expected_checksum.lower():
            raise ArtifactCorruptError(
                f"Artifact checksum verification failed: expected {expected_checksum}, "
                f"computed {actual_checksum}."
            )

    # 2. JSON Syntax decoding
    try:
        raw_text = content_bytes.decode("utf-8")
        envelope = json.loads(raw_text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactCorruptError(f"Artifact JSON parsing failed: {exc}") from exc

    if not isinstance(envelope, dict):
        raise ArtifactCorruptError("Artifact payload must be a JSON object.")

    # 3. Format and Version compatibility
    artifact_fmt = envelope.get("format")
    if artifact_fmt != ARTIFACT_FORMAT:
        raise ArtifactIncompatibleError(
            f"Unsupported artifact format '{artifact_fmt}'. Expected '{ARTIFACT_FORMAT}'."
        )

    fmt_version = envelope.get("format_version")
    if fmt_version != CURRENT_FORMAT_VERSION:
        raise ArtifactIncompatibleError(
            f"Unsupported artifact version '{fmt_version}'. "
            f"Supported version: '{CURRENT_FORMAT_VERSION}'."
        )

    # 4. Identity and ownership verification
    if expected_analysis_id is not None:
        actual_analysis_id = envelope.get("analysis_id")
        if actual_analysis_id != expected_analysis_id:
            raise ArtifactCorruptError(
                f"Artifact analysis_id mismatch: expected '{expected_analysis_id}', "
                f"found '{actual_analysis_id}'."
            )

    if expected_project_id is not None:
        actual_project_id = envelope.get("project_id")
        if actual_project_id != expected_project_id:
            raise ProjectMismatchError(
                f"Artifact project_id mismatch: expected '{expected_project_id}', "
                f"found '{actual_project_id}'."
            )

    # 5. Model reconstruction
    result_data = envelope.get("result")
    if not isinstance(result_data, dict):
        raise ArtifactCorruptError("Artifact envelope is missing valid 'result' object.")

    try:
        reconstructed = DataPipelineResult.model_validate(result_data)
        return reconstructed
    except Exception as exc:
        raise ArtifactCorruptError(
            f"Failed to reconstruct DataPipelineResult from artifact payload: {exc}"
        ) from exc


class AnalysisArtifactStore:
    """
    Manages physical file persistence, integrity verification, and metadata registration
    for DataPipelineResult execution artifacts.
    """

    def __init__(
        self,
        workspace: WorkspaceManager,
        repository: SQLiteMetadataRepository,
    ) -> None:
        """
        Initialize the artifact store.

        Parameters
        ----------
        workspace : WorkspaceManager
            Authoritative workspace manager.
        repository : SQLiteMetadataRepository
            Authoritative metadata repository.
        """
        self.workspace = workspace
        self.repository = repository

    def get_run_dir(self, project_id: str, analysis_id: str) -> pathlib.Path:
        """
        Return the managed runs directory for a specific analysis run.

        Parameters
        ----------
        project_id : str
            Parent project identifier.
        analysis_id : str
            Analysis run identifier.

        Returns
        -------
        pathlib.Path
            Resolved absolute directory path inside project runs.
        """
        valid_project_id = self.workspace.validate_identifier(project_id, "project_id")
        valid_analysis_id = self.workspace.validate_identifier(analysis_id, "analysis_id")
        runs_dir = self.workspace.get_runs_dir(valid_project_id)
        run_dir = runs_dir / valid_analysis_id
        project_dir = self.workspace.get_project_dir(valid_project_id)
        return self.workspace.resolve_safe_path(run_dir, project_dir)

    def store_artifact(
        self,
        analysis_id: str,
        result: DataPipelineResult,
        project_id: str | None = None,
        overwrite: bool = True,
    ) -> tuple[AnalysisRunRecord, pathlib.Path]:
        """
        Atomically persist a DataPipelineResult as an artifact and record metadata.

        Parameters
        ----------
        analysis_id : str
            Analysis run identifier. Must exist in metadata repository.
        result : DataPipelineResult
            The completed pipeline result to persist.
        project_id : str | None
            Optional expected project identifier. If provided, verified against database record.
        overwrite : bool
            Whether to overwrite an existing artifact file on disk.

        Returns
        -------
        tuple[AnalysisRunRecord, pathlib.Path]
            Updated AnalysisRunRecord and absolute path to the stored artifact file.

        Raises
        ------
        EntityNotFoundError
            If analysis_id does not exist in repository.
        ProjectMismatchError
            If analysis_id does not belong to project_id.
        ProjectNotFoundError
            If parent project directory does not exist.
        StorageError
            If path resolution or atomic writing fails.
        """
        # 1. Fetch analysis run from repository
        run_record = self.repository.get_analysis_run(analysis_id)
        if run_record is None:
            raise EntityNotFoundError(
                f"Analysis run '{analysis_id}' not found in metadata repository."
            )

        # 2. Verify project ownership
        target_project_id = run_record.project_id
        if project_id is not None and target_project_id != project_id:
            raise ProjectMismatchError(
                f"Analysis run '{analysis_id}' belongs to project '{target_project_id}', "
                f"not '{project_id}'."
            )

        # 3. Verify project exists
        project = self.repository.get_project(target_project_id)
        if project is None:
            raise ProjectNotFoundError(
                f"Project '{target_project_id}' not found in metadata repository."
            )

        # 4. Serialize with envelope and compute cryptographic checksum
        raw_bytes, sha256_checksum = serialize_pipeline_result(
            result=result,
            analysis_id=analysis_id,
            project_id=target_project_id,
            dataset_id=run_record.dataset_id,
        )

        # 5. Compute target relative and absolute paths
        storage_relative_path = f"runs/{analysis_id}/{DEFAULT_ARTIFACT_FILENAME}"
        proj_dir = self.workspace.get_project_dir(target_project_id)
        target_file_path = proj_dir / storage_relative_path

        # 6. Atomically write using safe file staging and path confinement
        dest_path = self.workspace.create_safe_file(
            target_path=target_file_path,
            content_bytes=raw_bytes,
            overwrite=overwrite,
            base_dir=proj_dir,
        )

        # 7. Update analysis run metadata in database with artifact linkage
        current_summary = dict(run_record.result_summary or {})
        current_summary["artifact"] = {
            "storage_relative_path": storage_relative_path,
            "sha256_checksum": sha256_checksum,
            "file_size_bytes": len(raw_bytes),
            "format": ARTIFACT_FORMAT,
            "format_version": CURRENT_FORMAT_VERSION,
            "persisted_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }

        updated_run = self.repository.update_analysis_run(
            analysis_id=analysis_id,
            run_status=run_record.run_status,
            completed_at=run_record.completed_at,
            execution_duration_ms=run_record.execution_duration_ms,
            total_issues_count=run_record.total_issues_count,
            error_count=run_record.error_count,
            warning_count=run_record.warning_count,
            result_summary=current_summary,
            provenance_id=run_record.provenance_id,
        )

        return updated_run, dest_path

    def load_artifact(
        self,
        analysis_id: str,
        project_id: str | None = None,
    ) -> DataPipelineResult:
        """
        Load, verify, and reconstruct a DataPipelineResult artifact by analysis_id.

        Parameters
        ----------
        analysis_id : str
            Analysis run identifier.
        project_id : str | None
            Optional project identifier to verify run ownership.

        Returns
        -------
        DataPipelineResult
            Reconstructed pipeline result with full structural fidelity.

        Raises
        ------
        EntityNotFoundError
            If analysis run does not exist in repository.
        ProjectMismatchError
            If analysis run does not belong to specified project_id.
        ArtifactNotFoundError
            If artifact metadata is not registered or file does not exist on disk.
        ArtifactCorruptError
            If file fails checksum verification or JSON parsing.
        ArtifactIncompatibleError
            If artifact format or version is unsupported.
        """
        # 1. Fetch analysis run from repository
        run_record = self.repository.get_analysis_run(analysis_id)
        if run_record is None:
            raise EntityNotFoundError(
                f"Analysis run '{analysis_id}' not found in metadata repository."
            )

        # 2. Verify project ownership
        if project_id is not None and run_record.project_id != project_id:
            raise ProjectMismatchError(
                f"Analysis run '{analysis_id}' belongs to project '{run_record.project_id}', "
                f"not '{project_id}'."
            )

        # 3. Retrieve artifact metadata from result_summary
        result_summary = run_record.result_summary or {}
        artifact_info = result_summary.get("artifact")
        if not artifact_info or not isinstance(artifact_info, dict):
            raise ArtifactNotFoundError(
                f"No artifact metadata registered for analysis run '{analysis_id}'."
            )

        storage_relative_path = artifact_info.get("storage_relative_path")
        if not storage_relative_path:
            raise ArtifactNotFoundError(
                f"No storage relative path found in artifact metadata for run '{analysis_id}'."
            )

        # 4. Resolve safe absolute file path
        proj_dir = self.workspace.get_project_dir(run_record.project_id)
        target_path = proj_dir / storage_relative_path
        resolved_path = self.workspace.resolve_safe_path(target_path, proj_dir)

        # 5. Check physical file existence
        if not resolved_path.exists() or not resolved_path.is_file():
            raise ArtifactNotFoundError(
                f"Artifact file not found on disk for analysis run '{analysis_id}' "
                f"at '{resolved_path}'."
            )

        # 6. Read raw bytes from disk
        content_bytes = resolved_path.read_bytes()

        # 7. Deserialize and verify integrity against recorded metadata
        return deserialize_pipeline_result(
            content_bytes=content_bytes,
            expected_checksum=artifact_info.get("sha256_checksum"),
            expected_analysis_id=analysis_id,
            expected_project_id=run_record.project_id,
        )

    def get_artifact_path(
        self,
        analysis_id: str,
        project_id: str | None = None,
    ) -> pathlib.Path:
        """
        Return the verified on-disk path to an analysis artifact.

        Parameters
        ----------
        analysis_id : str
            Analysis run identifier.
        project_id : str | None
            Optional project identifier for verification.

        Returns
        -------
        pathlib.Path
            Absolute verified path to the artifact file.
        """
        run_record = self.repository.get_analysis_run(analysis_id)
        if run_record is None:
            raise EntityNotFoundError(
                f"Analysis run '{analysis_id}' not found in metadata repository."
            )

        if project_id is not None and run_record.project_id != project_id:
            raise ProjectMismatchError(
                f"Analysis run '{analysis_id}' belongs to project '{run_record.project_id}', "
                f"not '{project_id}'."
            )

        artifact_info = (run_record.result_summary or {}).get("artifact")
        if not artifact_info or "storage_relative_path" not in artifact_info:
            raise ArtifactNotFoundError(
                f"No artifact metadata registered for analysis run '{analysis_id}'."
            )

        proj_dir = self.workspace.get_project_dir(run_record.project_id)
        resolved_path = self.workspace.resolve_safe_path(
            proj_dir / artifact_info["storage_relative_path"], proj_dir
        )

        if not resolved_path.exists():
            raise ArtifactNotFoundError(
                f"Artifact file not found at '{resolved_path}'."
            )

        return resolved_path
