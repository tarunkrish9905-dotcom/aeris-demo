"""
Tests for AERIS Storage Workspace Manager, SQLite Migrations, and Metadata Repository.
"""

from __future__ import annotations

import datetime
import os
import pathlib
import tempfile
from collections.abc import Generator

import pytest

from aeris.storage.db import get_db_connection, transaction
from aeris.storage.exceptions import (
    DatabaseError,
    DuplicateEntityError,
    FileAlreadyExistsError,
    InvalidIdentifierError,
    PathTraversalError,
    ProjectNotFoundError,
)
from aeris.storage.migrations import apply_migrations, get_current_schema_version
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
from aeris.storage.repository import SQLiteMetadataRepository
from aeris.storage.workspace import WorkspaceManager


@pytest.fixture
def temp_workspace() -> Generator[WorkspaceManager, None, None]:
    """Provide a WorkspaceManager backed by an isolated temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield WorkspaceManager(workspace_root=tmp_dir)


@pytest.fixture
def temp_repo() -> Generator[SQLiteMetadataRepository, None, None]:
    """Provide an initialized SQLiteMetadataRepository in an isolated temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = pathlib.Path(tmp_dir) / "test_aeris.db"
        yield SQLiteMetadataRepository(db_path=db_path)


# ===========================================================================
# 1. WorkspaceManager Tests
# ===========================================================================


class TestWorkspaceManager:
    """Test suite for WorkspaceManager filesystem boundaries and safety."""

    def test_init_workspace_creates_directories(self, temp_workspace: WorkspaceManager) -> None:
        """Verify init_workspace creates workspace_root and projects directories."""
        temp_workspace.init_workspace()
        assert temp_workspace.workspace_root.exists()
        assert temp_workspace.projects_dir.exists()

    def test_validate_identifier_valid(self) -> None:
        """Verify valid domain identifiers pass validation."""
        assert WorkspaceManager.validate_identifier("flight_001") == "flight_001"
        assert WorkspaceManager.validate_identifier("PROJ-2026_A") == "PROJ-2026_A"
        assert WorkspaceManager.validate_identifier("  test-id  ") == "test-id"

    def test_validate_identifier_invalid(self) -> None:
        """Verify invalid identifiers raise InvalidIdentifierError."""
        with pytest.raises(InvalidIdentifierError):
            WorkspaceManager.validate_identifier("")
        with pytest.raises(InvalidIdentifierError):
            WorkspaceManager.validate_identifier("   ")
        with pytest.raises(InvalidIdentifierError):
            WorkspaceManager.validate_identifier("proj/sub")
        with pytest.raises(InvalidIdentifierError):
            WorkspaceManager.validate_identifier("proj\\sub")
        with pytest.raises(InvalidIdentifierError):
            WorkspaceManager.validate_identifier("proj..id")
        with pytest.raises(InvalidIdentifierError):
            WorkspaceManager.validate_identifier("a" * 65)  # exceeds max length 64

    def test_resolve_safe_path_confinement(self, temp_workspace: WorkspaceManager) -> None:
        """Verify safe relative paths resolve within base_dir."""
        temp_workspace.init_workspace()
        resolved = temp_workspace.resolve_safe_path("projects/proj_1")
        assert resolved.is_relative_to(temp_workspace.workspace_root)

    def test_resolve_safe_path_traversal_detection(self, temp_workspace: WorkspaceManager) -> None:
        """Verify traversal tokens and escapes raise PathTraversalError."""
        temp_workspace.init_workspace()
        with pytest.raises(PathTraversalError):
            temp_workspace.resolve_safe_path("../outside.csv")

        with pytest.raises(PathTraversalError):
            temp_workspace.resolve_safe_path("projects/../../outside.csv")

        outside = temp_workspace.workspace_root.parent / "escape.csv"
        with pytest.raises(PathTraversalError):
            temp_workspace.resolve_safe_path(outside)

    def test_create_and_get_project_workspace(self, temp_workspace: WorkspaceManager) -> None:
        """Verify full directory tree creation for a project."""
        proj_dir = temp_workspace.create_project_workspace("proj_alpha")
        assert proj_dir.exists()
        assert temp_workspace.get_dataset_dir("proj_alpha").exists()
        assert temp_workspace.get_config_dir("proj_alpha").exists()
        assert temp_workspace.get_runs_dir("proj_alpha").exists()
        assert temp_workspace.get_reports_dir("proj_alpha").exists()

    def test_purge_project_workspace(self, temp_workspace: WorkspaceManager) -> None:
        """Verify purging safely removes project tree."""
        proj_dir = temp_workspace.create_project_workspace("proj_to_purge")
        dummy_file = proj_dir / "datasets" / "test.txt"
        dummy_file.write_text("sample")
        assert dummy_file.exists()

        temp_workspace.purge_project_workspace("proj_to_purge")
        assert not proj_dir.exists()

    def test_create_safe_file_atomic_and_overwrite(self, temp_workspace: WorkspaceManager) -> None:
        """Verify atomic file creation, content preservation, and overwrite policies."""
        temp_workspace.init_workspace()
        dest = temp_workspace.workspace_root / "safe_file.txt"

        # Initial write
        written = temp_workspace.create_safe_file(dest, b"hello world")
        assert written.exists()
        assert written.read_bytes() == b"hello world"

        # Overwrite=False raises FileAlreadyExistsError
        with pytest.raises(FileAlreadyExistsError):
            temp_workspace.create_safe_file(dest, b"new content", overwrite=False)

        # Overwrite=True updates content atomically
        temp_workspace.create_safe_file(dest, b"new content", overwrite=True)
        assert written.read_bytes() == b"new content"

    def test_symlink_escape_rejection(self, temp_workspace: WorkspaceManager) -> None:
        """Verify symlinks resolving outside the workspace are detected and rejected."""
        temp_workspace.init_workspace()
        outside_dir = temp_workspace.workspace_root.parent / "outside_secret"
        outside_dir.mkdir(parents=True, exist_ok=True)
        outside_file = outside_dir / "secret.csv"
        outside_file.write_text("secret,data\n1,2\n")

        link_target = temp_workspace.workspace_root / "link_outside"
        try:
            os.symlink(outside_dir, link_target, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("Creating directory symlinks requires elevated privileges on Windows.")

        # Resolving path through symlink that points outside must be blocked
        with pytest.raises(PathTraversalError):
            temp_workspace.resolve_safe_path("link_outside/secret.csv")


# ===========================================================================
# 2. SQLite Migrations & Connection Tests
# ===========================================================================


class TestSQLiteMigrationsAndDB:
    """Test suite for database migrations, foreign keys, and WAL mode."""

    def test_fresh_database_applies_all_eight_tables(self) -> None:
        """Verify schema_v1.sql creates all 8 expected tables."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = pathlib.Path(tmp_dir) / "test.db"
            conn = get_db_connection(db_path)
            version = apply_migrations(conn)
            assert version == 1
            assert get_current_schema_version(conn) == 1

            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = {row[0] for row in cursor.fetchall()}

            expected_tables = {
                "schema_migrations",
                "projects",
                "datasets",
                "parameter_mappings",
                "configuration_profiles",
                "analysis_runs",
                "background_jobs",
                "exported_reports",
            }
            assert expected_tables.issubset(tables)
            conn.close()

    def test_reopening_database_is_idempotent(self) -> None:
        """Verify reopening existing database preserves records without re-migration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = pathlib.Path(tmp_dir) / "test.db"
            repo1 = SQLiteMetadataRepository(db_path)
            now = datetime.datetime.now(datetime.UTC)
            p = ProjectRecord(
                project_id="p1",
                name="Flight Test 1",
                description="Desc",
                analyst_name="Alice",
                created_at=now,
                updated_at=now,
                status=ProjectStatus.ACTIVE,
            )
            repo1.create_project(p)

            # Reopen database with new repository instance
            repo2 = SQLiteMetadataRepository(db_path)
            loaded = repo2.get_project("p1")
            assert loaded is not None
            assert loaded.name == "Flight Test 1"
            assert loaded.analyst_name == "Alice"

    def test_foreign_key_enforcement(self) -> None:
        """Verify SQLite foreign key constraints prevent orphaned child records."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = pathlib.Path(tmp_dir) / "test.db"
            repo = SQLiteMetadataRepository(db_path)
            now = datetime.datetime.now(datetime.UTC)

            # Attempt creating dataset for non-existent project
            ds = DatasetRecord(
                dataset_id="ds_1",
                project_id="non_existent_project",
                original_filename="flight.csv",
                storage_relative_path="projects/non_existent_project/datasets/ds_1.csv",
                sha256_checksum="abc",
                file_size_bytes=100,
                row_count=10,
                column_count=2,
                detected_columns=["time", "alt"],
                ingested_at=now,
            )
            with pytest.raises(ProjectNotFoundError):
                repo.create_dataset(ds)

    def test_transaction_rollback_on_error(self) -> None:
        """Verify transaction context manager rolls back on exception."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = pathlib.Path(tmp_dir) / "test.db"
            conn = get_db_connection(db_path)
            apply_migrations(conn)

            with pytest.raises(DatabaseError), transaction(conn):
                conn.execute(
                    "INSERT INTO projects (project_id, name, created_at, updated_at, status) "
                    "VALUES ('p_fail', 'Fail Project', '2026-01-01', '2026-01-01', 'ACTIVE')"
                )
                # Intentionally trigger an error
                raise RuntimeError("Simulated failure inside transaction")

            cursor = conn.execute("SELECT * FROM projects WHERE project_id = 'p_fail'")
            assert cursor.fetchone() is None
            conn.close()


# ===========================================================================
# 3. SQLiteMetadataRepository CRUD Tests
# ===========================================================================


class TestSQLiteMetadataRepository:
    """Test suite for repository CRUD operations."""

    def test_project_crud_and_status(self, temp_repo: SQLiteMetadataRepository) -> None:
        """Verify project creation, retrieval, listing, update, and soft delete."""
        now = datetime.datetime.now(datetime.UTC)
        p = ProjectRecord(
            project_id="proj_001",
            name="Aero Test Alpha",
            description="Initial aerodynamic baseline",
            analyst_name="Test Engineer",
            created_at=now,
            updated_at=now,
            status=ProjectStatus.ACTIVE,
        )
        temp_repo.create_project(p)

        # Retrieve
        retrieved = temp_repo.get_project("proj_001")
        assert retrieved is not None
        assert retrieved.name == "Aero Test Alpha"
        assert retrieved.status == ProjectStatus.ACTIVE

        # Update
        updated = temp_repo.update_project(
            "proj_001", name="Aero Test Alpha Revised", analyst_name="Lead FTE"
        )
        assert updated.name == "Aero Test Alpha Revised"
        assert updated.analyst_name == "Lead FTE"

        # List
        projects = temp_repo.list_projects()
        assert len(projects) == 1
        assert projects[0].project_id == "proj_001"

        # Soft Delete
        assert temp_repo.delete_project("proj_001", hard_delete=False)
        soft_deleted = temp_repo.get_project("proj_001")
        assert soft_deleted is not None
        assert soft_deleted.status == ProjectStatus.DELETED
        assert len(temp_repo.list_projects()) == 0  # list_projects filters out DELETED by default

        # Hard Delete
        assert temp_repo.delete_project("proj_001", hard_delete=True)
        assert temp_repo.get_project("proj_001") is None

    def test_duplicate_project_id_raises(self, temp_repo: SQLiteMetadataRepository) -> None:
        """Verify creating a duplicate project_id raises DuplicateEntityError."""
        now = datetime.datetime.now(datetime.UTC)
        p = ProjectRecord(
            project_id="dup_proj",
            name="Name 1",
            description="",
            analyst_name=None,
            created_at=now,
            updated_at=now,
            status=ProjectStatus.ACTIVE,
        )
        temp_repo.create_project(p)
        with pytest.raises(DuplicateEntityError):
            temp_repo.create_project(p)

    def test_parameter_mappings_crud(self, temp_repo: SQLiteMetadataRepository) -> None:
        """Verify saving and retrieving parameter mappings."""
        now = datetime.datetime.now(datetime.UTC)
        p = ProjectRecord(
            project_id="p_map",
            name="Mapping Project",
            description="",
            analyst_name=None,
            created_at=now,
            updated_at=now,
            status=ProjectStatus.ACTIVE,
        )
        temp_repo.create_project(p)

        ds = DatasetRecord(
            dataset_id="ds_map",
            project_id="p_map",
            original_filename="flight.csv",
            storage_relative_path="projects/p_map/datasets/ds_map.csv",
            sha256_checksum="123456",
            file_size_bytes=500,
            row_count=50,
            column_count=2,
            detected_columns=["UTC_TIME", "ALT_FEET"],
            ingested_at=now,
        )
        temp_repo.create_dataset(ds)

        mappings = [
            ParameterMappingRecord(
                mapping_id="m1",
                dataset_id="ds_map",
                raw_column_name="UTC_TIME",
                target_parameter_id="timestamp",
                source_unit="s",
                target_unit="UTC",
                data_type="datetime",
                is_timestamp=True,
            ),
            ParameterMappingRecord(
                mapping_id="m2",
                dataset_id="ds_map",
                raw_column_name="ALT_FEET",
                target_parameter_id="altitude_ft",
                source_unit="ft",
                target_unit="ft",
                data_type="float",
                is_timestamp=False,
            ),
        ]
        temp_repo.save_parameter_mappings(mappings)

        loaded = temp_repo.get_parameter_mappings("ds_map")
        assert len(loaded) == 2
        assert loaded[0].raw_column_name == "UTC_TIME"
        assert loaded[0].is_timestamp is True
        assert loaded[1].raw_column_name == "ALT_FEET"

    def test_analysis_runs_and_background_jobs(self, temp_repo: SQLiteMetadataRepository) -> None:
        """Verify creating, querying, and updating analysis runs and background jobs."""
        now = datetime.datetime.now(datetime.UTC)
        p = ProjectRecord(
            project_id="p_job",
            name="Job Project",
            description="",
            analyst_name=None,
            created_at=now,
            updated_at=now,
            status=ProjectStatus.ACTIVE,
        )
        temp_repo.create_project(p)

        ds = DatasetRecord(
            dataset_id="ds_job",
            project_id="p_job",
            original_filename="flight.csv",
            storage_relative_path="projects/p_job/datasets/ds_job.csv",
            sha256_checksum="hash",
            file_size_bytes=1000,
            row_count=100,
            column_count=3,
            detected_columns=["time", "alt", "spd"],
            ingested_at=now,
        )
        temp_repo.create_dataset(ds)

        run = AnalysisRunRecord(
            analysis_id="run_01",
            project_id="p_job",
            dataset_id="ds_job",
            run_status="RUNNING",
            triggered_at=now,
        )
        temp_repo.create_analysis_run(run)

        job = BackgroundJobRecord(
            job_id="job_01",
            analysis_id="run_01",
            state="QUEUED",
            created_at=now,
            updated_at=now,
        )
        temp_repo.create_background_job(job)

        # Query
        fetched_job = temp_repo.get_background_job("job_01")
        assert fetched_job is not None
        assert fetched_job.state == "QUEUED"
        assert fetched_job.progress_percent == 0

        # Update job progress
        updated_job = temp_repo.update_background_job(
            "job_01",
            state="RUNNING",
            current_stage="Quality Assessment",
            completed_stages_count=3,
            progress_percent=20,
        )
        assert updated_job.state == "RUNNING"
        assert updated_job.current_stage == "Quality Assessment"
        assert updated_job.progress_percent == 20

        # Update run completion
        completed_run = temp_repo.update_analysis_run(
            "run_01",
            run_status="SUCCESS",
            completed_at=datetime.datetime.now(datetime.UTC),
            execution_duration_ms=450,
            total_issues_count=0,
            result_summary={"quality_score": 0.98},
        )
        assert completed_run.run_status == "SUCCESS"
        assert completed_run.execution_duration_ms == 450

    def test_exported_report_crud(self, temp_repo: SQLiteMetadataRepository) -> None:
        """Verify recording and listing exported reports."""
        now = datetime.datetime.now(datetime.UTC)
        p = ProjectRecord(
            project_id="p_rep",
            name="Report Project",
            description="",
            analyst_name=None,
            created_at=now,
            updated_at=now,
            status=ProjectStatus.ACTIVE,
        )
        temp_repo.create_project(p)

        ds = DatasetRecord(
            dataset_id="ds_rep",
            project_id="p_rep",
            original_filename="flight.csv",
            storage_relative_path="projects/p_rep/datasets/ds_rep.csv",
            sha256_checksum="hash",
            file_size_bytes=1000,
            row_count=100,
            column_count=3,
            detected_columns=["time", "alt", "spd"],
            ingested_at=now,
        )
        temp_repo.create_dataset(ds)

        run = AnalysisRunRecord(
            analysis_id="run_rep",
            project_id="p_rep",
            dataset_id="ds_rep",
            run_status="SUCCESS",
            triggered_at=now,
        )
        temp_repo.create_analysis_run(run)

        report = ExportedReportRecord(
            report_id="rep_01",
            analysis_id="run_rep",
            report_format="HTML",
            report_title="Flight Test 01 Engineering Report",
            storage_relative_path="projects/p_rep/reports/rep_01.html",
            sha256_checksum="rep_hash",
            file_size_bytes=45000,
            generated_at=now,
        )
        temp_repo.create_exported_report(report)

        fetched = temp_repo.get_exported_report("rep_01")
        assert fetched is not None
        assert fetched.report_format == "HTML"
        assert fetched.file_size_bytes == 45000

        reports_list = temp_repo.list_exported_reports(project_id="p_rep")
        assert len(reports_list) == 1
        assert reports_list[0].report_id == "rep_01"

    def test_configuration_profile_crud(self, temp_repo: SQLiteMetadataRepository) -> None:
        """Verify saving and listing configuration profile references."""
        now = datetime.datetime.now(datetime.UTC)
        p = ProjectRecord(
            project_id="p_cfg",
            name="Config Project",
            description="",
            analyst_name=None,
            created_at=now,
            updated_at=now,
            status=ProjectStatus.ACTIVE,
        )
        temp_repo.create_project(p)

        cfg = ConfigurationProfileRecord(
            config_id="cfg_01",
            project_id="p_cfg",
            profile_type="VEHICLE",
            profile_identifier="F16_AERO",
            storage_relative_path="projects/p_cfg/configs/vehicle.yaml",
            sha256_checksum="cfg_hash",
            loaded_at=now,
        )
        temp_repo.create_configuration_profile(cfg)

        loaded_all = temp_repo.list_configuration_profiles("p_cfg")
        assert len(loaded_all) == 1
        assert loaded_all[0].profile_identifier == "F16_AERO"

        loaded_vehicle = temp_repo.list_configuration_profiles("p_cfg", profile_type="VEHICLE")
        assert len(loaded_vehicle) == 1

        loaded_sop = temp_repo.list_configuration_profiles("p_cfg", profile_type="SOP")
        assert len(loaded_sop) == 0
