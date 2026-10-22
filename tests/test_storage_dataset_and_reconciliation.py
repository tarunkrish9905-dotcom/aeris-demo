"""
Tests for AERIS Managed Dataset Store and Workspace Reconciliation Engine.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import pathlib
import tempfile
from collections.abc import Generator
from unittest.mock import patch

import pytest

from aeris.storage.dataset_store import ManagedDatasetStore
from aeris.storage.exceptions import (
    DatasetNotFoundError,
    FileAlreadyExistsError,
    OversizedFileError,
    PathTraversalError,
    StorageError,
)
from aeris.storage.models import ProjectRecord, ProjectStatus
from aeris.storage.reconciliation import WorkspaceReconciler
from aeris.storage.repository import SQLiteMetadataRepository
from aeris.storage.workspace import WorkspaceManager


@pytest.fixture
def storage_environment() -> Generator[
    tuple[WorkspaceManager, SQLiteMetadataRepository, ManagedDatasetStore], None, None
]:
    """Provide an isolated WorkspaceManager, SQLiteMetadataRepository, and ManagedDatasetStore."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = WorkspaceManager(workspace_root=tmp_dir)
        workspace.init_workspace()
        repo = SQLiteMetadataRepository(db_path=workspace.db_path)
        store = ManagedDatasetStore(
            workspace=workspace, repository=repo, max_upload_size_bytes=1024 * 1024
        )

        # Create baseline project
        now = datetime.datetime.now(datetime.UTC)
        p = ProjectRecord(
            project_id="proj_test",
            name="Test Flight Project",
            description="Testing dataset storage",
            analyst_name="FTE Tester",
            created_at=now,
            updated_at=now,
            status=ProjectStatus.ACTIVE,
        )
        repo.create_project(p)
        workspace.create_project_workspace("proj_test")

        yield workspace, repo, store


SAMPLE_CSV_CONTENT = (
    b"timestamp,altitude_ft,airspeed_kts,pitch_deg\n"
    b"2026-01-01T12:00:00Z,5000.0,150.2,2.5\n"
    b"2026-01-01T12:00:01Z,5010.0,151.0,2.6\n"
    b"2026-01-01T12:00:02Z,5020.0,151.8,2.7\n"
)


# ===========================================================================
# 1. ManagedDatasetStore Tests
# ===========================================================================


class TestManagedDatasetStore:
    """Test suite for ManagedDatasetStore byte preservation, integrity, and safety."""

    def test_store_dataset_from_file_path(
        self,
        storage_environment: tuple[WorkspaceManager, SQLiteMetadataRepository, ManagedDatasetStore],
    ) -> None:
        """Verify storing dataset from external source file path."""
        _workspace, repo, store = storage_environment

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(SAMPLE_CSV_CONTENT)
            src_path = pathlib.Path(f.name)

        try:
            record, managed_path = store.store_dataset(
                project_id="proj_test",
                source=src_path,
                original_filename="flight_alpha.csv",
                dataset_id="ds_001",
            )

            assert managed_path.exists()
            assert managed_path.read_bytes() == SAMPLE_CSV_CONTENT
            assert record.dataset_id == "ds_001"
            assert record.row_count == 3
            assert record.column_count == 4
            assert record.detected_columns == [
                "timestamp",
                "altitude_ft",
                "airspeed_kts",
                "pitch_deg",
            ]
            expected_hash = hashlib.sha256(SAMPLE_CSV_CONTENT).hexdigest()
            assert record.sha256_checksum == expected_hash

            # Verify recorded in DB
            db_record = repo.get_dataset("ds_001")
            assert db_record is not None
            assert db_record.sha256_checksum == expected_hash
        finally:
            src_path.unlink(missing_ok=True)

    def test_store_dataset_from_bytes(
        self,
        storage_environment: tuple[WorkspaceManager, SQLiteMetadataRepository, ManagedDatasetStore],
    ) -> None:
        """Verify storing dataset directly from in-memory bytes."""
        _workspace, repo, store = storage_environment

        record, managed_path = store.store_dataset(
            project_id="proj_test",
            source=SAMPLE_CSV_CONTENT,
            original_filename="in_memory.csv",
        )

        assert managed_path.exists()
        assert managed_path.read_bytes() == SAMPLE_CSV_CONTENT
        assert repo.get_dataset(record.dataset_id) is not None

    def test_store_dataset_from_binary_stream(
        self,
        storage_environment: tuple[WorkspaceManager, SQLiteMetadataRepository, ManagedDatasetStore],
    ) -> None:
        """Verify storing dataset from file-like binary stream."""
        _workspace, repo, store = storage_environment
        stream = io.BytesIO(SAMPLE_CSV_CONTENT)

        record, managed_path = store.store_dataset(
            project_id="proj_test",
            source=stream,
            original_filename="streamed.csv",
        )

        assert managed_path.exists()
        assert managed_path.read_bytes() == SAMPLE_CSV_CONTENT
        assert repo.get_dataset(record.dataset_id) is not None

    def test_store_dataset_rejects_non_csv_and_unsafe_names(
        self,
        storage_environment: tuple[WorkspaceManager, SQLiteMetadataRepository, ManagedDatasetStore],
    ) -> None:
        """Verify rejecting non-CSV extensions and path traversal filenames."""
        _workspace, _repo, store = storage_environment

        with pytest.raises(StorageError):
            store.store_dataset(
                project_id="proj_test",
                source=SAMPLE_CSV_CONTENT,
                original_filename="telemetry.txt",
            )

        with pytest.raises(PathTraversalError):
            store.store_dataset(
                project_id="proj_test",
                source=SAMPLE_CSV_CONTENT,
                original_filename="../../telemetry.csv",
            )

    def test_store_dataset_enforces_upload_size_limit(
        self,
        storage_environment: tuple[WorkspaceManager, SQLiteMetadataRepository, ManagedDatasetStore],
    ) -> None:
        """Verify oversized uploads are rejected and temporary files cleaned up."""
        workspace, _repo, store = storage_environment
        # Limit is 1MB in fixture, supply 2MB of payload
        oversized_payload = b"col1,col2\n" + (b"1,2\n" * 500000)

        with pytest.raises(OversizedFileError):
            store.store_dataset(
                project_id="proj_test",
                source=oversized_payload,
                original_filename="huge.csv",
            )

        # Verify no temporary files remain in datasets directory
        dataset_dir = workspace.get_dataset_dir("proj_test")
        tmp_files = list(dataset_dir.glob(".aeris_tmp_*"))
        assert len(tmp_files) == 0

    def test_store_dataset_overwrite_policy(
        self,
        storage_environment: tuple[WorkspaceManager, SQLiteMetadataRepository, ManagedDatasetStore],
    ) -> None:
        """Verify overwrite=False raises FileAlreadyExistsError, overwrite=True succeeds."""
        _workspace, _repo, store = storage_environment

        store.store_dataset(
            project_id="proj_test",
            source=SAMPLE_CSV_CONTENT,
            original_filename="initial.csv",
            dataset_id="ds_fixed",
        )

        # Overwrite=False must fail
        with pytest.raises(FileAlreadyExistsError):
            store.store_dataset(
                project_id="proj_test",
                source=SAMPLE_CSV_CONTENT,
                original_filename="initial.csv",
                dataset_id="ds_fixed",
                overwrite=False,
            )

        # Overwrite=True must succeed
        new_content = b"timestamp,val\n2026-01-01T00:00:00Z,99\n"
        rec2, path2 = store.store_dataset(
            project_id="proj_test",
            source=new_content,
            original_filename="initial.csv",
            dataset_id="ds_fixed",
            overwrite=True,
        )
        assert path2.read_bytes() == new_content
        assert rec2.row_count == 1

    def test_store_dataset_cleans_target_file_if_db_insert_fails(
        self,
        storage_environment: tuple[WorkspaceManager, SQLiteMetadataRepository, ManagedDatasetStore],
    ) -> None:
        """Verify that if database insertion fails, the target file is removed."""
        workspace, repo, store = storage_environment

        # Force repository.create_dataset to raise an exception
        with (
            patch.object(repo, "create_dataset", side_effect=RuntimeError("Simulated DB Crash")),
            pytest.raises(StorageError),
        ):
            store.store_dataset(
                project_id="proj_test",
                source=SAMPLE_CSV_CONTENT,
                original_filename="fail.csv",
                dataset_id="ds_db_fail",
            )

        target = workspace.get_dataset_dir("proj_test") / "ds_db_fail.csv"
        assert not target.exists()

    def test_get_dataset_path_and_missing_file_error(
        self,
        storage_environment: tuple[WorkspaceManager, SQLiteMetadataRepository, ManagedDatasetStore],
    ) -> None:
        """Verify resolving managed path and detecting missing files."""
        _workspace, _repo, store = storage_environment

        _rec, path = store.store_dataset(
            project_id="proj_test",
            source=SAMPLE_CSV_CONTENT,
            original_filename="sample.csv",
            dataset_id="ds_get",
        )
        assert store.get_dataset_path("ds_get") == path

        # Non-existent dataset ID
        with pytest.raises(DatasetNotFoundError):
            store.get_dataset_path("ds_unknown")

        # Delete physical file behind database's back
        path.unlink()
        with pytest.raises(DatasetNotFoundError):
            store.get_dataset_path("ds_get")

    def test_verify_dataset_integrity(
        self,
        storage_environment: tuple[WorkspaceManager, SQLiteMetadataRepository, ManagedDatasetStore],
    ) -> None:
        """Verify checksum integrity check detects intact and corrupted files."""
        _workspace, _repo, store = storage_environment

        _rec, path = store.store_dataset(
            project_id="proj_test",
            source=SAMPLE_CSV_CONTENT,
            original_filename="verify.csv",
            dataset_id="ds_check",
        )
        assert store.verify_dataset_integrity("ds_check") is True

        # Tamper with file
        path.write_bytes(b"tampered,data\n1,2\n")
        assert store.verify_dataset_integrity("ds_check") is False

    def test_delete_dataset_with_file_cleanup(
        self,
        storage_environment: tuple[WorkspaceManager, SQLiteMetadataRepository, ManagedDatasetStore],
    ) -> None:
        """Verify deleting dataset removes both DB record and file on disk."""
        _workspace, repo, store = storage_environment

        _rec, path = store.store_dataset(
            project_id="proj_test",
            source=SAMPLE_CSV_CONTENT,
            original_filename="delete.csv",
            dataset_id="ds_del",
        )
        assert path.exists()

        assert store.delete_dataset("ds_del", remove_file=True) is True
        assert not path.exists()
        assert repo.get_dataset("ds_del") is None


# ===========================================================================
# 2. WorkspaceReconciler Tests
# ===========================================================================


class TestWorkspaceReconciler:
    """Test suite for WorkspaceReconciler consistency checks and crash recovery."""

    def test_reconcile_healthy_workspace(
        self,
        storage_environment: tuple[WorkspaceManager, SQLiteMetadataRepository, ManagedDatasetStore],
    ) -> None:
        """Verify healthy workspace produces clean report with is_healthy == True."""
        workspace, repo, store = storage_environment
        reconciler = WorkspaceReconciler(workspace=workspace, repository=repo)

        store.store_dataset(
            project_id="proj_test",
            source=SAMPLE_CSV_CONTENT,
            original_filename="clean.csv",
            dataset_id="ds_clean",
        )

        report = reconciler.check_workspace_integrity()
        assert report.is_healthy is True
        assert report.clean_datasets_count == 1
        assert len(report.missing_files) == 0
        assert len(report.corrupted_files) == 0
        assert len(report.unindexed_files) == 0

    def test_reconcile_detects_missing_file(
        self,
        storage_environment: tuple[WorkspaceManager, SQLiteMetadataRepository, ManagedDatasetStore],
    ) -> None:
        """Verify reconciler flags datasets whose physical file was deleted."""
        workspace, repo, store = storage_environment
        reconciler = WorkspaceReconciler(workspace=workspace, repository=repo)

        _rec, path = store.store_dataset(
            project_id="proj_test",
            source=SAMPLE_CSV_CONTENT,
            original_filename="missing.csv",
            dataset_id="ds_miss",
        )
        path.unlink()  # simulate external file deletion

        report = reconciler.check_workspace_integrity()
        assert report.is_healthy is False
        assert "ds_miss" in report.missing_files

    def test_reconcile_detects_corrupted_file(
        self,
        storage_environment: tuple[WorkspaceManager, SQLiteMetadataRepository, ManagedDatasetStore],
    ) -> None:
        """Verify reconciler flags datasets whose checksum does not match."""
        workspace, repo, store = storage_environment
        reconciler = WorkspaceReconciler(workspace=workspace, repository=repo)

        _rec, path = store.store_dataset(
            project_id="proj_test",
            source=SAMPLE_CSV_CONTENT,
            original_filename="corrupt.csv",
            dataset_id="ds_corrupt",
        )
        path.write_bytes(b"corrupted,content\n")

        report = reconciler.check_workspace_integrity()
        assert report.is_healthy is False
        assert "ds_corrupt" in report.corrupted_files

    def test_reconcile_detects_unindexed_file(
        self,
        storage_environment: tuple[WorkspaceManager, SQLiteMetadataRepository, ManagedDatasetStore],
    ) -> None:
        """Verify reconciler detects rogue CSV files in project datasets directory."""
        workspace, repo, _store = storage_environment
        reconciler = WorkspaceReconciler(workspace=workspace, repository=repo)

        unindexed_path = workspace.get_dataset_dir("proj_test") / "rogue_telemetry.csv"
        unindexed_path.write_text("a,b\n1,2\n")

        report = reconciler.check_workspace_integrity()
        assert report.is_healthy is False
        assert any("rogue_telemetry.csv" in f for f in report.unindexed_files)

    def test_reconcile_detects_and_cleans_stale_temp_files(
        self,
        storage_environment: tuple[WorkspaceManager, SQLiteMetadataRepository, ManagedDatasetStore],
    ) -> None:
        """Verify reconciler detects and safely deletes stale .aeris_tmp_* files."""
        workspace, repo, _store = storage_environment
        reconciler = WorkspaceReconciler(workspace=workspace, repository=repo)

        temp_staging = workspace.get_dataset_dir("proj_test") / ".aeris_tmp_stale_crash.csv"
        temp_staging.write_text("partial,data\n")

        report = reconciler.check_workspace_integrity()
        assert report.is_healthy is False
        assert any(".aeris_tmp_stale_crash.csv" in f for f in report.stale_temporary_files)

        # Cleanup with age 0 seconds (deletes immediately)
        deleted = reconciler.clean_stale_temporary_files(max_age_seconds=0)
        assert len(deleted) == 1
        assert not temp_staging.exists()
