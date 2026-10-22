"""
AERIS Workspace Reconciliation and Crash Consistency Engine.

Detects and reports inconsistencies between SQLite metadata and physical workspace files:
- Missing files (DB record exists, physical file is absent)
- Unindexed files (file exists on disk, no DB record)
- Corrupted files (file hash differs from registered DB checksum)
- Stale temporary files (.aeris_tmp_*) left behind by crashes or interruptions.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import pathlib
import time

from aeris.storage.models import WorkspaceIntegrityReport
from aeris.storage.repository import SQLiteMetadataRepository
from aeris.storage.workspace import WorkspaceManager

CHUNK_SIZE = 64 * 1024


class WorkspaceReconciler:
    """Diagnoses and reconciles workspace filesystem and SQLite metadata consistency."""

    def __init__(
        self,
        workspace: WorkspaceManager,
        repository: SQLiteMetadataRepository,
    ) -> None:
        """
        Initialize WorkspaceReconciler.

        Parameters
        ----------
        workspace : WorkspaceManager
            The workspace manager.
        repository : SQLiteMetadataRepository
            The SQLite metadata repository.
        """
        self.workspace = workspace
        self.repository = repository

    def check_workspace_integrity(self, project_id: str | None = None) -> WorkspaceIntegrityReport:
        """
        Perform a non-destructive integrity audit comparing database records with disk files.

        Parameters
        ----------
        project_id : str | None
            Optional project identifier to limit check to a single project.
            If None, audits all active projects.

        Returns
        -------
        WorkspaceIntegrityReport
            Detailed diagnostic summary of clean, missing, corrupted, unindexed,
            and temporary files.
        """
        now_utc = datetime.datetime.now(datetime.UTC)
        clean_count = 0
        missing_files: list[str] = []
        corrupted_files: list[str] = []
        unindexed_files: list[str] = []
        stale_temp_files: list[str] = []

        if project_id:
            projects = [p for p in [self.repository.get_project(project_id)] if p is not None]
        else:
            projects = self.repository.list_projects()

        registered_relative_paths: set[str] = set()

        # 1. Audit all registered datasets in DB against physical disk
        for proj in projects:
            datasets = self.repository.list_datasets(proj.project_id)
            for ds in datasets:
                registered_relative_paths.add(ds.storage_relative_path.replace("\\", "/"))

                try:
                    file_path = self.workspace.resolve_safe_path(
                        ds.storage_relative_path, self.workspace.workspace_root
                    )
                except Exception:
                    missing_files.append(ds.dataset_id)
                    continue

                if not file_path.exists() or not file_path.is_file():
                    missing_files.append(ds.dataset_id)
                    continue

                # Verify checksum
                try:
                    hasher = hashlib.sha256()
                    with file_path.open("rb") as f:
                        while chunk := f.read(CHUNK_SIZE):
                            hasher.update(chunk)
                    computed_digest = hasher.hexdigest()

                    if computed_digest == ds.sha256_checksum:
                        clean_count += 1
                    else:
                        corrupted_files.append(ds.dataset_id)
                except OSError:
                    corrupted_files.append(ds.dataset_id)

        # 2. Audit physical filesystem for unindexed or temporary files
        projects_dir = self.workspace.projects_dir
        if projects_dir.exists():
            for p_path in projects_dir.iterdir():
                if not p_path.is_dir():
                    continue

                p_id = p_path.name
                if project_id and p_id != project_id:
                    continue

                dataset_dir = p_path / "datasets"
                if dataset_dir.exists() and dataset_dir.is_dir():
                    for item in dataset_dir.iterdir():
                        if not item.is_file():
                            continue

                        # Detect temporary staging files
                        if item.name.startswith(".aeris_tmp_"):
                            rel_tmp = str(item.relative_to(self.workspace.workspace_root))
                            stale_temp_files.append(rel_tmp)
                            continue

                        # Check for unindexed CSV files
                        if item.name.lower().endswith(".csv"):
                            rel = str(item.relative_to(self.workspace.workspace_root))
                            rel_clean = rel.replace("\\", "/")
                            if rel_clean not in registered_relative_paths:
                                unindexed_files.append(rel_clean)

        return WorkspaceIntegrityReport(
            checked_at=now_utc,
            clean_datasets_count=clean_count,
            missing_files=missing_files,
            unindexed_files=unindexed_files,
            corrupted_files=corrupted_files,
            stale_temporary_files=stale_temp_files,
        )

    def clean_stale_temporary_files(self, max_age_seconds: int = 3600) -> list[pathlib.Path]:
        """
        Safely remove orphaned .aeris_tmp_* files older than max_age_seconds.

        Never touches registered dataset files or valid project directories.

        Returns
        -------
        list[pathlib.Path]
            List of deleted temporary file paths.
        """
        now = time.time()
        deleted_paths: list[pathlib.Path] = []
        projects_dir = self.workspace.projects_dir

        if not projects_dir.exists():
            return deleted_paths

        for item in projects_dir.rglob(".aeris_tmp_*"):
            if item.is_file():
                try:
                    mtime = item.stat().st_mtime
                    if (now - mtime) >= max_age_seconds:
                        with contextlib.suppress(OSError):
                            item.unlink()
                            deleted_paths.append(item)
                except OSError:
                    pass

        return deleted_paths
