"""
AERIS SQLite Metadata Repository.

Implements CRUD operations for projects, datasets, parameter mappings,
configuration profiles, analysis runs, background jobs, and exported reports.
Uses standard library sqlite3 with foreign key enforcement and transaction isolation.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import pathlib
import sqlite3
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aeris.core.results.types import DataPipelineResult
    from aeris.storage.workspace import WorkspaceManager

from aeris.storage.db import get_db_connection
from aeris.storage.exceptions import (
    DatabaseError,
    DuplicateEntityError,
    ProjectNotFoundError,
    WorkspaceError,
)
from aeris.storage.migrations import apply_migrations
from aeris.storage.models import (
    AnalysisRunRecord,
    BackgroundJobRecord,
    ConfigurationProfileRecord,
    DatasetRecord,
    ExportedReportRecord,
    ParameterMappingRecord,
    ProjectRecord,
    ProjectStatus,
)


class SQLiteMetadataRepository:
    """Authoritative metadata repository backed by SQLite."""

    def __init__(
        self,
        db_path: str | pathlib.Path,
        wal_mode: bool = True,
        auto_migrate: bool = True,
        workspace: WorkspaceManager | None = None,
    ) -> None:
        """
        Initialize the repository.

        Parameters
        ----------
        db_path : str | pathlib.Path
            Path to SQLite database file or ':memory:'.
        wal_mode : bool
            Whether to use WAL mode.
        auto_migrate : bool
            Whether to automatically run unapplied schema migrations upon initialization.
        workspace : WorkspaceManager | None
            Optional WorkspaceManager for artifact and file persistence.
        """
        self.db_path = db_path
        self.wal_mode = wal_mode
        self.workspace = workspace

        if auto_migrate:
            with self._connection() as conn:
                apply_migrations(conn)

    @contextlib.contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing a managed connection that is reliably closed on exit."""
        conn = get_db_connection(self.db_path, wal_mode=self.wal_mode)
        try:
            yield conn
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new managed connection with foreign keys and WAL mode configured."""
        return get_db_connection(self.db_path, wal_mode=self.wal_mode)

    # -----------------------------------------------------------------------
    # Projects CRUD
    # -----------------------------------------------------------------------

    def create_project(self, project: ProjectRecord) -> ProjectRecord:
        """Insert a new ProjectRecord."""
        with self._connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO projects (
                        project_id, name, description, analyst_name,
                        created_at, updated_at, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project.project_id,
                        project.name,
                        project.description,
                        project.analyst_name,
                        project.created_at.isoformat(),
                        project.updated_at.isoformat(),
                        project.status.value,
                    ),
                )
                conn.commit()
                return project
            except sqlite3.IntegrityError as exc:
                if "UNIQUE constraint failed" in str(exc) or "PRIMARY KEY" in str(exc):
                    raise DuplicateEntityError(
                        f"Project with ID '{project.project_id}' already exists."
                    ) from exc
                raise DatabaseError(f"Database error inserting project: {exc}") from exc

    def get_project(self, project_id: str) -> ProjectRecord | None:
        """Retrieve ProjectRecord by project_id, or None if not found."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT project_id, name, description, analyst_name,
                       created_at, updated_at, status
                FROM projects WHERE project_id = ?
                """,
                (project_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_project(row)

    def list_projects(self, status: ProjectStatus | None = None) -> list[ProjectRecord]:
        """List projects, optionally filtered by status."""
        with self._connection() as conn:
            if status is not None:
                cursor = conn.execute(
                    """
                    SELECT project_id, name, description, analyst_name,
                           created_at, updated_at, status
                    FROM projects WHERE status = ?
                    ORDER BY created_at DESC
                    """,
                    (status.value,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT project_id, name, description, analyst_name,
                           created_at, updated_at, status
                    FROM projects WHERE status != 'DELETED'
                    ORDER BY created_at DESC
                    """
                )
            return [self._row_to_project(row) for row in cursor.fetchall()]

    def update_project(
        self,
        project_id: str,
        name: str | None = None,
        description: str | None = None,
        analyst_name: str | None = None,
        status: ProjectStatus | None = None,
    ) -> ProjectRecord:
        """Update mutable fields of a project."""
        existing = self.get_project(project_id)
        if existing is None:
            raise ProjectNotFoundError(f"Project '{project_id}' does not exist.")

        new_name = name if name is not None else existing.name
        new_desc = description if description is not None else existing.description
        new_analyst = analyst_name if analyst_name is not None else existing.analyst_name
        new_status = status if status is not None else existing.status
        now_dt = datetime.datetime.now(datetime.UTC)

        with self._connection() as conn:
            conn.execute(
                """
                UPDATE projects
                SET name = ?, description = ?, analyst_name = ?,
                    status = ?, updated_at = ?
                WHERE project_id = ?
                """,
                (
                    new_name,
                    new_desc,
                    new_analyst,
                    new_status.value,
                    now_dt.isoformat(),
                    project_id,
                ),
            )
            conn.commit()

        return ProjectRecord(
            project_id=project_id,
            name=new_name,
            description=new_desc,
            analyst_name=new_analyst,
            created_at=existing.created_at,
            updated_at=now_dt,
            status=new_status,
        )

    def delete_project(self, project_id: str, hard_delete: bool = False) -> bool:
        """
        Delete a project.

        If hard_delete is False (default), marks status as DELETED.
        If hard_delete is True, removes the row and cascades to child tables.
        """
        existing = self.get_project(project_id)
        if existing is None:
            return False

        with self._connection() as conn:
            if hard_delete:
                conn.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))
            else:
                now_iso = datetime.datetime.now(datetime.UTC).isoformat()
                conn.execute(
                    "UPDATE projects SET status = 'DELETED', updated_at = ? WHERE project_id = ?",
                    (now_iso, project_id),
                )
            conn.commit()
        return True

    # -----------------------------------------------------------------------
    # Datasets CRUD
    # -----------------------------------------------------------------------

    def create_dataset(self, dataset: DatasetRecord) -> DatasetRecord:
        """Insert a new DatasetRecord."""
        project = self.get_project(dataset.project_id)
        if project is None:
            raise ProjectNotFoundError(
                f"Cannot create dataset; project '{dataset.project_id}' does not exist."
            )

        with self._connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO datasets (
                        dataset_id, project_id, original_filename,
                        storage_relative_path, sha256_checksum, file_size_bytes,
                        row_count, column_count, detected_columns_json, ingested_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dataset.dataset_id,
                        dataset.project_id,
                        dataset.original_filename,
                        dataset.storage_relative_path,
                        dataset.sha256_checksum,
                        dataset.file_size_bytes,
                        dataset.row_count,
                        dataset.column_count,
                        json.dumps(dataset.detected_columns),
                        dataset.ingested_at.isoformat(),
                    ),
                )
                conn.commit()
                return dataset
            except sqlite3.IntegrityError as exc:
                if "UNIQUE constraint failed" in str(exc) or "PRIMARY KEY" in str(exc):
                    raise DuplicateEntityError(
                        f"Dataset with ID '{dataset.dataset_id}' already exists."
                    ) from exc
                raise DatabaseError(f"Database error inserting dataset: {exc}") from exc

    def get_dataset(self, dataset_id: str) -> DatasetRecord | None:
        """Retrieve DatasetRecord by dataset_id, or None if not found."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT dataset_id, project_id, original_filename,
                       storage_relative_path, sha256_checksum, file_size_bytes,
                       row_count, column_count, detected_columns_json, ingested_at
                FROM datasets WHERE dataset_id = ?
                """,
                (dataset_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_dataset(row)

    def list_datasets(self, project_id: str) -> list[DatasetRecord]:
        """List all datasets in a project."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT dataset_id, project_id, original_filename,
                       storage_relative_path, sha256_checksum, file_size_bytes,
                       row_count, column_count, detected_columns_json, ingested_at
                FROM datasets WHERE project_id = ?
                ORDER BY ingested_at DESC
                """,
                (project_id,),
            )
            return [self._row_to_dataset(row) for row in cursor.fetchall()]

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset metadata record."""
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
            conn.commit()
            return cursor.rowcount > 0

    # -----------------------------------------------------------------------
    # Parameter Mappings
    # -----------------------------------------------------------------------

    def save_parameter_mappings(self, mappings: list[ParameterMappingRecord]) -> None:
        """Save parameter mappings for a dataset (replaces existing mappings)."""
        if not mappings:
            return

        dataset_id = mappings[0].dataset_id
        with self._connection() as conn, conn:
            conn.execute(
                "DELETE FROM parameter_mappings WHERE dataset_id = ?",
                (dataset_id,),
            )
            for m in mappings:
                conn.execute(
                    """
                        INSERT INTO parameter_mappings (
                            mapping_id, dataset_id, raw_column_name,
                            target_parameter_id, source_unit, target_unit,
                            data_type, is_timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                    (
                        m.mapping_id,
                        m.dataset_id,
                        m.raw_column_name,
                        m.target_parameter_id,
                        m.source_unit,
                        m.target_unit,
                        m.data_type,
                        1 if m.is_timestamp else 0,
                    ),
                )

    def get_parameter_mappings(self, dataset_id: str) -> list[ParameterMappingRecord]:
        """Retrieve all parameter mappings for a dataset."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT mapping_id, dataset_id, raw_column_name,
                       target_parameter_id, source_unit, target_unit,
                       data_type, is_timestamp
                FROM parameter_mappings WHERE dataset_id = ?
                """,
                (dataset_id,),
            )
            records: list[ParameterMappingRecord] = []
            for row in cursor.fetchall():
                records.append(
                    ParameterMappingRecord(
                        mapping_id=row["mapping_id"],
                        dataset_id=row["dataset_id"],
                        raw_column_name=row["raw_column_name"],
                        target_parameter_id=row["target_parameter_id"],
                        source_unit=row["source_unit"],
                        target_unit=row["target_unit"],
                        data_type=row["data_type"],
                        is_timestamp=bool(row["is_timestamp"]),
                    )
                )
            return records

    # -----------------------------------------------------------------------
    # Configuration Profiles
    # -----------------------------------------------------------------------

    def create_configuration_profile(
        self, config: ConfigurationProfileRecord
    ) -> ConfigurationProfileRecord:
        """Insert a new configuration profile record."""
        with self._connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO configuration_profiles (
                        config_id, project_id, profile_type,
                        profile_identifier, storage_relative_path,
                        sha256_checksum, loaded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        config.config_id,
                        config.project_id,
                        config.profile_type,
                        config.profile_identifier,
                        config.storage_relative_path,
                        config.sha256_checksum,
                        config.loaded_at.isoformat(),
                    ),
                )
                conn.commit()
                return config
            except sqlite3.IntegrityError as exc:
                raise DatabaseError(f"Failed inserting configuration profile: {exc}") from exc

    def list_configuration_profiles(
        self, project_id: str, profile_type: str | None = None
    ) -> list[ConfigurationProfileRecord]:
        """List configuration profiles for a project."""
        with self._connection() as conn:
            if profile_type:
                cursor = conn.execute(
                    """
                    SELECT config_id, project_id, profile_type,
                           profile_identifier, storage_relative_path,
                           sha256_checksum, loaded_at
                    FROM configuration_profiles
                    WHERE project_id = ? AND profile_type = ?
                    ORDER BY loaded_at DESC
                    """,
                    (project_id, profile_type),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT config_id, project_id, profile_type,
                           profile_identifier, storage_relative_path,
                           sha256_checksum, loaded_at
                    FROM configuration_profiles
                    WHERE project_id = ?
                    ORDER BY loaded_at DESC
                    """,
                    (project_id,),
                )
            return [
                ConfigurationProfileRecord(
                    config_id=row["config_id"],
                    project_id=row["project_id"],
                    profile_type=row["profile_type"],
                    profile_identifier=row["profile_identifier"],
                    storage_relative_path=row["storage_relative_path"],
                    sha256_checksum=row["sha256_checksum"],
                    loaded_at=datetime.datetime.fromisoformat(row["loaded_at"]),
                )
                for row in cursor.fetchall()
            ]

    # -----------------------------------------------------------------------
    # Analysis Runs & Background Jobs
    # -----------------------------------------------------------------------

    def create_analysis_run(self, run: AnalysisRunRecord) -> AnalysisRunRecord:
        """Insert a new analysis run record."""
        with self._connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO analysis_runs (
                        analysis_id, project_id, dataset_id, run_status,
                        triggered_at, completed_at, execution_duration_ms,
                        total_issues_count, error_count, warning_count,
                        result_summary_json, provenance_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run.analysis_id,
                        run.project_id,
                        run.dataset_id,
                        run.run_status,
                        run.triggered_at.isoformat(),
                        run.completed_at.isoformat() if run.completed_at else None,
                        run.execution_duration_ms,
                        run.total_issues_count,
                        run.error_count,
                        run.warning_count,
                        json.dumps(run.result_summary) if run.result_summary else None,
                        run.provenance_id,
                    ),
                )
                conn.commit()
                return run
            except sqlite3.IntegrityError as exc:
                raise DatabaseError(f"Failed inserting analysis run: {exc}") from exc

    def get_analysis_run(self, analysis_id: str) -> AnalysisRunRecord | None:
        """Retrieve AnalysisRunRecord by analysis_id."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT analysis_id, project_id, dataset_id, run_status,
                       triggered_at, completed_at, execution_duration_ms,
                       total_issues_count, error_count, warning_count,
                       result_summary_json, provenance_id
                FROM analysis_runs WHERE analysis_id = ?
                """,
                (analysis_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_analysis_run(row)

    def list_analysis_runs(self, project_id: str) -> list[AnalysisRunRecord]:
        """List all analysis runs in a project."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT analysis_id, project_id, dataset_id, run_status,
                       triggered_at, completed_at, execution_duration_ms,
                       total_issues_count, error_count, warning_count,
                       result_summary_json, provenance_id
                FROM analysis_runs WHERE project_id = ?
                ORDER BY triggered_at DESC
                """,
                (project_id,),
            )
            return [self._row_to_analysis_run(row) for row in cursor.fetchall()]

    def update_analysis_run(
        self,
        analysis_id: str,
        run_status: str,
        completed_at: datetime.datetime | None = None,
        execution_duration_ms: int | None = None,
        total_issues_count: int = 0,
        error_count: int = 0,
        warning_count: int = 0,
        result_summary: dict[str, Any] | None = None,
        provenance_id: str | None = None,
    ) -> AnalysisRunRecord:
        """Update terminal results of an analysis run."""
        existing = self.get_analysis_run(analysis_id)
        if existing is None:
            raise DatabaseError(f"Analysis run '{analysis_id}' not found.")

        with self._connection() as conn:
            conn.execute(
                """
                UPDATE analysis_runs
                SET run_status = ?, completed_at = ?, execution_duration_ms = ?,
                    total_issues_count = ?, error_count = ?, warning_count = ?,
                    result_summary_json = ?, provenance_id = ?
                WHERE analysis_id = ?
                """,
                (
                    run_status,
                    completed_at.isoformat() if completed_at else None,
                    execution_duration_ms,
                    total_issues_count,
                    error_count,
                    warning_count,
                    json.dumps(result_summary) if result_summary else None,
                    provenance_id or existing.provenance_id,
                    analysis_id,
                ),
            )
            conn.commit()

        updated = self.get_analysis_run(analysis_id)
        if updated is None:
            raise DatabaseError(f"Failed reloading updated analysis run '{analysis_id}'.")
        return updated

    def save_analysis_run_artifact(
        self,
        analysis_id: str,
        result: DataPipelineResult,
        project_id: str | None = None,
        workspace: WorkspaceManager | None = None,
        overwrite: bool = True,
    ) -> tuple[AnalysisRunRecord, pathlib.Path]:
        """
        Persist a DataPipelineResult artifact to workspace storage and register metadata.

        Parameters
        ----------
        analysis_id : str
            Analysis run identifier. Must exist in repository.
        result : DataPipelineResult
            The completed pipeline result to persist.
        project_id : str | None
            Optional expected project identifier. If provided, verified against database.
        workspace : WorkspaceManager | None
            Workspace directory manager (uses self.workspace if None).
        overwrite : bool
            Whether to overwrite an existing artifact file on disk.

        Returns
        -------
        tuple[AnalysisRunRecord, pathlib.Path]
            Updated AnalysisRunRecord and absolute path to the written artifact file.
        """
        ws = workspace or self.workspace
        if ws is None:
            raise WorkspaceError("WorkspaceManager is required to persist analysis artifacts.")
        from aeris.storage.artifact_store import AnalysisArtifactStore

        store = AnalysisArtifactStore(workspace=ws, repository=self)
        return store.store_artifact(
            analysis_id=analysis_id,
            result=result,
            project_id=project_id,
            overwrite=overwrite,
        )

    def get_analysis_run_artifact(
        self,
        analysis_id: str,
        project_id: str | None = None,
        workspace: WorkspaceManager | None = None,
    ) -> DataPipelineResult:
        """
        Retrieve and reconstruct a DataPipelineResult artifact from workspace storage.

        Validates project/run ownership and cryptographic checksum integrity.

        Parameters
        ----------
        analysis_id : str
            Analysis run identifier.
        project_id : str | None
            Optional project identifier to verify run ownership.
        workspace : WorkspaceManager | None
            Workspace directory manager (uses self.workspace if None).

        Returns
        -------
        DataPipelineResult
            Reconstructed pipeline result with full structural fidelity.
        """
        ws = workspace or self.workspace
        if ws is None:
            raise WorkspaceError("WorkspaceManager is required to retrieve analysis artifacts.")
        from aeris.storage.artifact_store import AnalysisArtifactStore

        store = AnalysisArtifactStore(workspace=ws, repository=self)
        return store.load_artifact(
            analysis_id=analysis_id,
            project_id=project_id,
        )

    def get_analysis_run_artifact_path(
        self,
        analysis_id: str,
        project_id: str | None = None,
        workspace: WorkspaceManager | None = None,
    ) -> pathlib.Path:
        """
        Return the verified on-disk path to an analysis artifact.

        Parameters
        ----------
        analysis_id : str
            Analysis run identifier.
        project_id : str | None
            Optional project identifier for verification.
        workspace : WorkspaceManager | None
            Workspace directory manager (uses self.workspace if None).

        Returns
        -------
        pathlib.Path
            Absolute verified path to the artifact file.
        """
        ws = workspace or self.workspace
        if ws is None:
            raise WorkspaceError("WorkspaceManager is required to locate analysis artifacts.")
        from aeris.storage.artifact_store import AnalysisArtifactStore

        store = AnalysisArtifactStore(workspace=ws, repository=self)
        return store.get_artifact_path(
            analysis_id=analysis_id,
            project_id=project_id,
        )

    def create_background_job(self, job: BackgroundJobRecord) -> BackgroundJobRecord:
        """Insert a new background job tracking record."""
        with self._connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO background_jobs (
                        job_id, analysis_id, state, current_stage,
                        completed_stages_count, total_stages_count,
                        progress_percent, error_message, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.job_id,
                        job.analysis_id,
                        job.state,
                        job.current_stage,
                        job.completed_stages_count,
                        job.total_stages_count,
                        job.progress_percent,
                        job.error_message,
                        job.created_at.isoformat(),
                        job.updated_at.isoformat(),
                    ),
                )
                conn.commit()
                return job
            except sqlite3.IntegrityError as exc:
                raise DatabaseError(f"Failed inserting background job: {exc}") from exc

    def get_background_job(self, job_id: str) -> BackgroundJobRecord | None:
        """Retrieve BackgroundJobRecord by job_id."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT job_id, analysis_id, state, current_stage,
                       completed_stages_count, total_stages_count,
                       progress_percent, error_message, created_at, updated_at
                FROM background_jobs WHERE job_id = ?
                """,
                (job_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_background_job(row)

    def update_background_job(
        self,
        job_id: str,
        state: str,
        current_stage: str | None = None,
        completed_stages_count: int | None = None,
        progress_percent: int | None = None,
        error_message: str | None = None,
    ) -> BackgroundJobRecord:
        """Update state and progress of a background job."""
        existing = self.get_background_job(job_id)
        if existing is None:
            raise DatabaseError(f"Background job '{job_id}' not found.")

        now_dt = datetime.datetime.now(datetime.UTC)
        new_stage = current_stage if current_stage is not None else existing.current_stage
        new_completed = (
            completed_stages_count
            if completed_stages_count is not None
            else existing.completed_stages_count
        )
        new_progress = (
            progress_percent if progress_percent is not None else existing.progress_percent
        )
        new_err = error_message if error_message is not None else existing.error_message

        with self._connection() as conn:
            conn.execute(
                """
                UPDATE background_jobs
                SET state = ?, current_stage = ?, completed_stages_count = ?,
                    progress_percent = ?, error_message = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    state,
                    new_stage,
                    new_completed,
                    new_progress,
                    new_err,
                    now_dt.isoformat(),
                    job_id,
                ),
            )
            conn.commit()

        updated = self.get_background_job(job_id)
        if updated is None:
            raise DatabaseError(f"Failed reloading updated background job '{job_id}'.")
        return updated

    # -----------------------------------------------------------------------
    # Exported Reports
    # -----------------------------------------------------------------------

    def create_exported_report(self, report: ExportedReportRecord) -> ExportedReportRecord:
        """Insert an exported report record."""
        with self._connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO exported_reports (
                        report_id, analysis_id, report_format, report_title,
                        storage_relative_path, sha256_checksum, file_size_bytes,
                        generated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        report.report_id,
                        report.analysis_id,
                        report.report_format,
                        report.report_title,
                        report.storage_relative_path,
                        report.sha256_checksum,
                        report.file_size_bytes,
                        report.generated_at.isoformat(),
                    ),
                )
                conn.commit()
                return report
            except sqlite3.IntegrityError as exc:
                raise DatabaseError(f"Failed inserting exported report: {exc}") from exc

    def get_exported_report(self, report_id: str) -> ExportedReportRecord | None:
        """Retrieve ExportedReportRecord by report_id."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT report_id, analysis_id, report_format, report_title,
                       storage_relative_path, sha256_checksum, file_size_bytes,
                       generated_at
                FROM exported_reports WHERE report_id = ?
                """,
                (report_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_exported_report(row)

    def list_exported_reports(
        self, analysis_id: str | None = None, project_id: str | None = None
    ) -> list[ExportedReportRecord]:
        """List exported reports, optionally filtered by analysis_id or project_id."""
        with self._connection() as conn:
            if analysis_id:
                cursor = conn.execute(
                    """
                    SELECT report_id, analysis_id, report_format, report_title,
                           storage_relative_path, sha256_checksum, file_size_bytes,
                           generated_at
                    FROM exported_reports WHERE analysis_id = ?
                    ORDER BY generated_at DESC
                    """,
                    (analysis_id,),
                )
            elif project_id:
                cursor = conn.execute(
                    """
                    SELECT r.report_id, r.analysis_id, r.report_format, r.report_title,
                           r.storage_relative_path, r.sha256_checksum, r.file_size_bytes,
                           r.generated_at
                    FROM exported_reports r
                    JOIN analysis_runs a ON r.analysis_id = a.analysis_id
                    WHERE a.project_id = ?
                    ORDER BY r.generated_at DESC
                    """,
                    (project_id,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT report_id, analysis_id, report_format, report_title,
                           storage_relative_path, sha256_checksum, file_size_bytes,
                           generated_at
                    FROM exported_reports
                    ORDER BY generated_at DESC
                    """
                )
            return [self._row_to_exported_report(row) for row in cursor.fetchall()]

    # -----------------------------------------------------------------------
    # Row Deserializers
    # -----------------------------------------------------------------------

    @staticmethod
    def _row_to_project(row: sqlite3.Row) -> ProjectRecord:
        return ProjectRecord(
            project_id=row["project_id"],
            name=row["name"],
            description=row["description"] or "",
            analyst_name=row["analyst_name"],
            created_at=datetime.datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.datetime.fromisoformat(row["updated_at"]),
            status=ProjectStatus(row["status"]),
        )

    @staticmethod
    def _row_to_dataset(row: sqlite3.Row) -> DatasetRecord:
        return DatasetRecord(
            dataset_id=row["dataset_id"],
            project_id=row["project_id"],
            original_filename=row["original_filename"],
            storage_relative_path=row["storage_relative_path"],
            sha256_checksum=row["sha256_checksum"],
            file_size_bytes=int(row["file_size_bytes"]),
            row_count=int(row["row_count"]),
            column_count=int(row["column_count"]),
            detected_columns=json.loads(row["detected_columns_json"]),
            ingested_at=datetime.datetime.fromisoformat(row["ingested_at"]),
        )

    @staticmethod
    def _row_to_analysis_run(row: sqlite3.Row) -> AnalysisRunRecord:
        return AnalysisRunRecord(
            analysis_id=row["analysis_id"],
            project_id=row["project_id"],
            dataset_id=row["dataset_id"],
            run_status=row["run_status"],
            triggered_at=datetime.datetime.fromisoformat(row["triggered_at"]),
            completed_at=datetime.datetime.fromisoformat(row["completed_at"])
            if row["completed_at"]
            else None,
            execution_duration_ms=row["execution_duration_ms"],
            total_issues_count=row["total_issues_count"] or 0,
            error_count=row["error_count"] or 0,
            warning_count=row["warning_count"] or 0,
            result_summary=json.loads(row["result_summary_json"])
            if row["result_summary_json"]
            else None,
            provenance_id=row["provenance_id"],
        )

    @staticmethod
    def _row_to_background_job(row: sqlite3.Row) -> BackgroundJobRecord:
        return BackgroundJobRecord(
            job_id=row["job_id"],
            analysis_id=row["analysis_id"],
            state=row["state"],
            current_stage=row["current_stage"],
            completed_stages_count=row["completed_stages_count"] or 0,
            total_stages_count=row["total_stages_count"] or 15,
            progress_percent=row["progress_percent"] or 0,
            error_message=row["error_message"],
            created_at=datetime.datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _row_to_exported_report(row: sqlite3.Row) -> ExportedReportRecord:
        return ExportedReportRecord(
            report_id=row["report_id"],
            analysis_id=row["analysis_id"],
            report_format=row["report_format"],
            report_title=row["report_title"],
            storage_relative_path=row["storage_relative_path"],
            sha256_checksum=row["sha256_checksum"],
            file_size_bytes=int(row["file_size_bytes"]),
            generated_at=datetime.datetime.fromisoformat(row["generated_at"]),
        )
