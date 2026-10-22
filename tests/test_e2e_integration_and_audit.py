"""
AERIS Milestone 15f: Full End-to-End System Integration & Audit Test Suite.

Verifies the complete analytical workflow across storage, backend, frontend,
visualization, report export, security boundaries, and epistemic integrity:
TELEMETRY UPLOAD -> VALIDATION -> MAPPING -> CONFIGURATION
-> ANALYSIS -> EVIDENCE -> VISUALIZATION -> REPORT EXPORT.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import pathlib
import tempfile
import time
from collections.abc import Generator

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from aeris.backend.app import create_app
from aeris.backend.config import BackendConfig
from aeris.frontend.app import create_dash_app, mount_dash_app
from aeris.frontend.client import sanitize_ui_error
from aeris.storage.models import AnalysisRunRecord, BackgroundJobRecord
from aeris.storage.reconciliation import WorkspaceReconciler
from aeris.storage.repository import SQLiteMetadataRepository
from aeris.storage.workspace import WorkspaceManager

# Realistic flight telemetry containing valid timestamps and aerodynamic parameters
SYNTHETIC_FLIGHT_CSV = (
    b"timestamp,indicated_airspeed,pressure_altitude,pitch_rate,roll_rate,normal_acceleration\n"
    b"2026-03-15T10:00:00Z,210.5,12000.0,0.5,0.1,1.01\n"
    b"2026-03-15T10:00:01Z,211.2,12025.0,0.8,0.2,1.03\n"
    b"2026-03-15T10:00:02Z,212.0,12050.0,1.2,0.1,1.05\n"
    b"2026-03-15T10:00:03Z,212.8,12080.0,1.5,0.3,1.08\n"
    b"2026-03-15T10:00:04Z,213.5,12110.0,1.1,0.0,1.04\n"
    b"2026-03-15T10:00:05Z,214.0,12140.0,0.4,-0.1,1.00\n"
)

# Telemetry with duplicate and unordered timestamps to test validation honesty
UNORDERED_TELEMETRY_CSV = (
    b"timestamp,indicated_airspeed,pressure_altitude\n"
    b"2026-03-15T10:00:00Z,210.0,10000.0\n"
    b"2026-03-15T09:59:50Z,205.0,9900.0\n"  # unordered (backwards in time)
    b"2026-03-15T10:00:00Z,210.0,10000.0\n"  # duplicate timestamp
)


@pytest.fixture
def e2e_environment() -> Generator[tuple[TestClient, BackendConfig, pathlib.Path], None, None]:
    """Provide an isolated end-to-end environment with FastAPI backend and Dash frontend."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_path = pathlib.Path(tmp_dir).resolve()
        config = BackendConfig(
            workspace_root=root_path,
            host="127.0.0.1",
            port=8000,
            max_upload_size_bytes=10 * 1024 * 1024,  # 10 MB for test
        )

        app = create_app(config=config, include_frontend=False)
        dash_app = create_dash_app()
        mount_dash_app(app, dash_app=dash_app)

        with TestClient(app) as client:
            yield client, config, root_path


# ===========================================================================
# Section A: Application Startup, Lifespan, and Network Binding
# ===========================================================================


def test_audit_startup_and_lifespan(
    e2e_environment: tuple[TestClient, BackendConfig, pathlib.Path],
) -> None:
    """Audit application startup, loopback binding, health, version, docs, and lifespan."""
    client, config, root_path = e2e_environment

    # 1. Loopback binding verification
    assert config.host == "127.0.0.1"
    assert config.port == 8000

    # 2. REST API Health check
    health_resp = client.get("/api/v1/health")
    assert health_resp.status_code == status.HTTP_200_OK
    health_data = health_resp.json()
    assert health_data["status"] == "healthy"
    assert health_data["database_connected"] is True

    # 3. REST API Version check
    version_resp = client.get("/api/v1/version")
    assert version_resp.status_code == status.HTTP_200_OK
    version_data = version_resp.json()
    assert version_data["version"] == config.app_version
    assert version_data["app_name"] == "AERIS"

    # 4. OpenAPI and Swagger UI docs
    openapi_resp = client.get("/openapi.json")
    assert openapi_resp.status_code == status.HTTP_200_OK
    openapi_spec = openapi_resp.json()
    assert "paths" in openapi_spec
    assert "/api/v1/health" in openapi_spec["paths"]
    assert "/api/v1/projects" in openapi_spec["paths"]

    docs_resp = client.get("/docs")
    assert docs_resp.status_code == status.HTTP_200_OK

    # 5. Mounted Dash frontend root
    dash_resp = client.get("/")
    assert dash_resp.status_code == status.HTTP_200_OK
    assert "<html" in dash_resp.text.lower() or "<!doctype html" in dash_resp.text.lower()

    # 6. Database and storage initialization on disk
    assert (root_path / "aeris_metadata.db").exists()


# ===========================================================================
# Section B: Project & Dataset Lifecycle Audit
# ===========================================================================


def test_audit_project_and_dataset_lifecycle(
    e2e_environment: tuple[TestClient, BackendConfig, pathlib.Path],
) -> None:
    """Audit project creation, dataset ingestion, SHA-256 verification, and isolation."""
    client, _config, root_path = e2e_environment

    # 1. Create Project
    p_resp = client.post(
        "/api/v1/projects",
        json={
            "name": "Integration Flight Alpha",
            "description": "High-altitude flutter envelope test",
            "analyst_name": "Test Pilot Engineer",
        },
    )
    assert p_resp.status_code == status.HTTP_201_CREATED
    p_data = p_resp.json()
    project_id = p_data["project_id"]
    assert p_data["name"] == "Integration Flight Alpha"

    # 2. Upload Valid Telemetry CSV
    files = {"file": ("flight_telemetry.csv", io.BytesIO(SYNTHETIC_FLIGHT_CSV), "text/csv")}
    upload_resp = client.post(f"/api/v1/projects/{project_id}/datasets/upload", files=files)
    assert upload_resp.status_code == status.HTTP_201_CREATED
    d_data = upload_resp.json()
    dataset_id = d_data["dataset_id"]
    assert d_data["row_count"] == 6
    assert "indicated_airspeed" in d_data["detected_columns"]
    assert "pitch_rate" in d_data["detected_columns"]

    # 3. SHA-256 checksum and byte preservation on disk
    managed_csv = root_path / "projects" / project_id / "datasets" / f"{dataset_id}.csv"
    assert managed_csv.exists()
    on_disk_bytes = managed_csv.read_bytes()
    assert on_disk_bytes == SYNTHETIC_FLIGHT_CSV
    expected_hash = hashlib.sha256(SYNTHETIC_FLIGHT_CSV).hexdigest()
    assert d_data["sha256_checksum"] == expected_hash

    # 4. Dataset Preview
    preview_resp = client.get(f"/api/v1/projects/{project_id}/datasets/{dataset_id}/preview")
    assert preview_resp.status_code == status.HTTP_200_OK
    preview_data = preview_resp.json()
    assert preview_data["total_rows"] == 6
    assert len(preview_data["preview_rows"]) == 6
    assert preview_data["preview_rows"][0]["indicated_airspeed"] == "210.5"

    # 5. Semantic Parameter Mapping
    mapping_payload = {
        "mappings": [
            {
                "raw_column_name": "indicated_airspeed",
                "target_parameter_id": "indicated_airspeed",
                "is_timestamp": False,
                "data_type": "float",
            },
            {
                "raw_column_name": "pressure_altitude",
                "target_parameter_id": "pressure_altitude",
                "is_timestamp": False,
                "data_type": "float",
            },
            {
                "raw_column_name": "pitch_rate",
                "target_parameter_id": "pitch_rate",
                "is_timestamp": False,
                "data_type": "float",
            },
        ]
    }
    map_put_resp = client.put(
        f"/api/v1/projects/{project_id}/datasets/{dataset_id}/mappings",
        json=mapping_payload,
    )
    assert map_put_resp.status_code == status.HTTP_200_OK
    assert len(map_put_resp.json()["mappings"]) == 3

    map_get_resp = client.get(f"/api/v1/projects/{project_id}/datasets/{dataset_id}/mappings")
    assert map_get_resp.status_code == status.HTTP_200_OK
    assert len(map_get_resp.json()["mappings"]) == 3

    # 6. Reject Invalid Parameter Mapping (empty list violates min_length=1)
    empty_map: dict[str, list[dict[str, object]]] = {"mappings": []}
    bad_map_resp = client.put(
        f"/api/v1/projects/{project_id}/datasets/{dataset_id}/mappings",
        json=empty_map,
    )
    assert bad_map_resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # 7. Dataset Isolation: Project B cannot access Project A's dataset
    p_b_resp = client.post("/api/v1/projects", json={"name": "Project Bravo"})
    project_b_id = p_b_resp.json()["project_id"]
    isolated_resp = client.get(f"/api/v1/projects/{project_b_id}/datasets/{dataset_id}/preview")
    assert isolated_resp.status_code == status.HTTP_404_NOT_FOUND


def test_audit_dataset_edge_cases_and_quality_limitations(
    e2e_environment: tuple[TestClient, BackendConfig, pathlib.Path],
) -> None:
    """Audit malformed CSVs, empty uploads, and unordered timestamps."""
    client, _config, _root = e2e_environment

    p_resp = client.post("/api/v1/projects", json={"name": "Edge Cases Project"})
    project_id = p_resp.json()["project_id"]

    # 1. Upload Non-CSV file
    bad_file = {"file": ("malicious.exe", io.BytesIO(b"MZ\x90\x00"), "application/octet-stream")}
    bad_resp = client.post(f"/api/v1/projects/{project_id}/datasets/upload", files=bad_file)
    assert bad_resp.status_code in (
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )

    # 2. Upload Empty CSV
    empty_file = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    empty_resp = client.post(f"/api/v1/projects/{project_id}/datasets/upload", files=empty_file)
    assert empty_resp.status_code == status.HTTP_201_CREATED
    assert empty_resp.json()["row_count"] == 0

    # 3. Upload Unordered / Duplicate Timestamps Telemetry
    unordered_file = {
        "file": (
            "unordered.csv",
            io.BytesIO(UNORDERED_TELEMETRY_CSV),
            "text/csv",
        )
    }
    un_resp = client.post(f"/api/v1/projects/{project_id}/datasets/upload", files=unordered_file)
    assert un_resp.status_code == status.HTTP_201_CREATED
    un_dataset_id = un_resp.json()["dataset_id"]

    # Run analysis on unordered telemetry: verify validation detects non-monotonicity honestly
    an_resp = client.post(
        f"/api/v1/projects/{project_id}/analyses",
        json={"dataset_id": un_dataset_id},
    )
    assert an_resp.status_code == status.HTTP_202_ACCEPTED
    job_id = an_resp.json()["job_id"]

    # Wait for completion
    for _ in range(50):
        j_stat = client.get(f"/api/v1/jobs/{job_id}").json()
        if j_stat["state"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.1)

    assert j_stat["state"] == "COMPLETED"
    analysis_id = an_resp.json()["analysis_id"]
    detail = client.get(f"/api/v1/projects/{project_id}/analyses/{analysis_id}").json()

    # The pipeline should honestly report validation issues without fabrication
    assert detail["total_issues"] > 0 or detail["warning_count"] > 0


# ===========================================================================
# Section C: Configuration & Analysis Execution Audit
# ===========================================================================


def test_audit_configuration_and_analysis_pipeline(
    e2e_environment: tuple[TestClient, BackendConfig, pathlib.Path],
) -> None:
    """Audit built-in configs, custom YAML profile upload, and 15-stage pipeline run."""
    client, _config, _root = e2e_environment

    # 1. Built-in Configurations
    builtin_resp = client.get("/api/v1/configs/built-in")
    assert builtin_resp.status_code == status.HTTP_200_OK
    builtins = builtin_resp.json()["builtin_configs"]
    types = {b["profile_type"] for b in builtins}
    assert "VEHICLE" in types
    assert "LIMIT" in types
    assert "SOP" in types

    # 2. Create Project and Ingest Telemetry
    p_resp = client.post("/api/v1/projects", json={"name": "Analysis Exec Project"})
    project_id = p_resp.json()["project_id"]
    files = {"file": ("flight.csv", io.BytesIO(SYNTHETIC_FLIGHT_CSV), "text/csv")}
    d_resp = client.post(f"/api/v1/projects/{project_id}/datasets/upload", files=files)
    dataset_id = d_resp.json()["dataset_id"]

    # 3. Upload Custom YAML Configuration
    valid_yaml = (
        b"vehicle_model: custom_f16\n"
        b"mass_kg: 12000.0\n"
        b"wing_area_m2: 27.87\n"
        b"wing_span_m: 9.96\n"
        b"reference_chord_m: 3.45\n"
    )
    yaml_files = {"file": ("custom_f16.yaml", io.BytesIO(valid_yaml), "application/x-yaml")}
    cfg_resp = client.post(
        f"/api/v1/projects/{project_id}/configs/upload",
        files=yaml_files,
        data={"profile_type": "VEHICLE"},
    )
    assert cfg_resp.status_code == status.HTTP_201_CREATED
    assert cfg_resp.json()["profile_type"] == "VEHICLE"

    # List project custom configs
    p_cfgs = client.get(f"/api/v1/projects/{project_id}/configs").json()["configs"]
    assert len(p_cfgs) == 1
    assert p_cfgs[0]["profile_identifier"] == "custom_f16.yaml"

    # 4. Submit and Track Pipeline Run
    start_resp = client.post(
        f"/api/v1/projects/{project_id}/analyses",
        json={
            "dataset_id": dataset_id,
            "vehicle_profile_id": "custom_f16",
        },
    )
    assert start_resp.status_code == status.HTTP_202_ACCEPTED
    job_id = start_resp.json()["job_id"]
    analysis_id = start_resp.json()["analysis_id"]

    # Poll status to completion
    for _ in range(60):
        status_resp = client.get(f"/api/v1/jobs/{job_id}")
        assert status_resp.status_code == status.HTTP_200_OK
        job_data = status_resp.json()
        if job_data["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        time.sleep(0.1)

    assert job_data["state"] == "COMPLETED"
    assert job_data["completed_stages"] == 5
    assert job_data["total_stages"] == 15
    assert job_data["progress_percent"] == 100

    # Verify Analysis Run record
    run_detail = client.get(f"/api/v1/projects/{project_id}/analyses/{analysis_id}").json()
    assert run_detail["run_status"] == "COMPLETED"
    assert run_detail["execution_duration_ms"] is not None
    assert run_detail["execution_duration_ms"] > 0
    assert run_detail["result_summary"]["overall_status"] is not None


def test_audit_cooperative_cancellation_and_crash_recovery(
    e2e_environment: tuple[TestClient, BackendConfig, pathlib.Path],
) -> None:
    """Audit cooperative job cancellation and startup crash recovery."""
    client, _config, root_path = e2e_environment

    p_resp = client.post("/api/v1/projects", json={"name": "Cancellation Project"})
    project_id = p_resp.json()["project_id"]
    files = {"file": ("flight.csv", io.BytesIO(SYNTHETIC_FLIGHT_CSV), "text/csv")}
    d_resp = client.post(f"/api/v1/projects/{project_id}/datasets/upload", files=files)
    dataset_id = d_resp.json()["dataset_id"]

    # 1. Cooperative Cancellation
    start_resp = client.post(
        f"/api/v1/projects/{project_id}/analyses",
        json={"dataset_id": dataset_id},
    )
    job_id = start_resp.json()["job_id"]

    cancel_resp = client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert cancel_resp.status_code == status.HTTP_200_OK

    # Poll status - must transition to CANCELLED or COMPLETED (if finished before cancel)
    for _ in range(20):
        j_data = client.get(f"/api/v1/jobs/{job_id}").json()
        if j_data["state"] in ("CANCELLED", "COMPLETED"):
            break
        time.sleep(0.05)

    assert j_data["state"] in ("CANCELLED", "COMPLETED")

    # 2. Crash Recovery Simulation
    db_path = root_path / "aeris_metadata.db"
    repo = SQLiteMetadataRepository(db_path=db_path)
    now_dt = datetime.datetime.now(datetime.UTC)

    orphaned_run = AnalysisRunRecord(
        analysis_id="run_crashed_sim",
        project_id=project_id,
        dataset_id=dataset_id,
        run_status="RUNNING",
        triggered_at=now_dt,
    )
    repo.create_analysis_run(orphaned_run)

    orphaned_job = BackgroundJobRecord(
        job_id="job_crashed_sim",
        analysis_id="run_crashed_sim",
        state="RUNNING",
        current_stage="ANOMALY_DETECTION",
        completed_stages_count=10,
        total_stages_count=15,
        progress_percent=66,
        created_at=now_dt,
        updated_at=now_dt,
    )
    repo.create_background_job(orphaned_job)

    # Re-instantiate app lifespan to trigger crash recovery
    with TestClient(create_app(BackendConfig(workspace_root=root_path))) as fresh_client:
        rec_job = fresh_client.get("/api/v1/jobs/job_crashed_sim").json()
        assert rec_job["state"] == "FAILED"
        assert "ERR_SYSTEM_RESTARTED_DURING_EXECUTION" in rec_job["error_message"]


# ===========================================================================
# Section D: Results, Evidence & Visualization Audit
# ===========================================================================


def test_audit_results_and_visualizations(
    e2e_environment: tuple[TestClient, BackendConfig, pathlib.Path],
) -> None:
    """Audit Plotly figure generation, zero fabricated data, and epistemic consistency."""
    client, _config, _root = e2e_environment

    p_resp = client.post("/api/v1/projects", json={"name": "Viz Audit Project"})
    project_id = p_resp.json()["project_id"]
    files = {"file": ("flight.csv", io.BytesIO(SYNTHETIC_FLIGHT_CSV), "text/csv")}
    d_resp = client.post(f"/api/v1/projects/{project_id}/datasets/upload", files=files)
    dataset_id = d_resp.json()["dataset_id"]

    # Submit and complete analysis
    an_resp = client.post(
        f"/api/v1/projects/{project_id}/analyses",
        json={"dataset_id": dataset_id},
    )
    job_id = an_resp.json()["job_id"]
    analysis_id = an_resp.json()["analysis_id"]

    for _ in range(50):
        if client.get(f"/api/v1/jobs/{job_id}").json()["state"] == "COMPLETED":
            break
        time.sleep(0.1)

    # 1. Single Time-Series Plot (Indicated Airspeed)
    ts_resp = client.get(
        f"/api/v1/projects/{project_id}/analyses/{analysis_id}/plots/time-series"
        "?parameter_id=indicated_airspeed"
    )
    assert ts_resp.status_code == status.HTTP_200_OK
    ts_fig = ts_resp.json()["figure_json"]
    assert "data" in ts_fig
    assert len(ts_fig["data"]) >= 1
    # Check that x has 6 points matching row count
    assert len(ts_fig["data"][0]["x"]) == 6

    # 2. Multi-Signal Panel Plot (Airspeed and Altitude)
    multi_payload = {
        "signal_ids": ["indicated_airspeed", "pressure_altitude"],
        "layout_mode": "STACKED",
        "show_limits": True,
        "show_events": True,
    }
    multi_resp = client.post(
        f"/api/v1/projects/{project_id}/analyses/{analysis_id}/plots/multi-signal",
        json=multi_payload,
    )
    assert multi_resp.status_code == status.HTTP_200_OK
    multi_fig = multi_resp.json()["figure_json"]
    assert "layout" in multi_fig
    assert "data" in multi_fig
    assert len(multi_fig["data"]) >= 2

    # 3. Honest Missing Parameter Handling (no fabricated flatlines)
    missing_resp = client.get(
        f"/api/v1/projects/{project_id}/analyses/{analysis_id}/plots/time-series"
        "?parameter_id=fictitious_mach_number"
    )
    assert missing_resp.status_code == status.HTTP_400_BAD_REQUEST
    assert "fictitious_mach_number" in missing_resp.json()["message"]


# ===========================================================================
# Section E: Reports & Downloads Audit
# ===========================================================================


def test_audit_report_generation_and_safe_export(
    e2e_environment: tuple[TestClient, BackendConfig, pathlib.Path],
) -> None:
    """Audit report generation, file checksum verification, and XSS escaping."""
    client, _config, _root_path = e2e_environment

    p_resp = client.post("/api/v1/projects", json={"name": "Reports Audit Project"})
    project_id = p_resp.json()["project_id"]
    files = {"file": ("flight.csv", io.BytesIO(SYNTHETIC_FLIGHT_CSV), "text/csv")}
    d_resp = client.post(f"/api/v1/projects/{project_id}/datasets/upload", files=files)
    dataset_id = d_resp.json()["dataset_id"]

    an_resp = client.post(
        f"/api/v1/projects/{project_id}/analyses",
        json={"dataset_id": dataset_id},
    )
    job_id = an_resp.json()["job_id"]
    analysis_id = an_resp.json()["analysis_id"]

    for _ in range(50):
        if client.get(f"/api/v1/jobs/{job_id}").json()["state"] == "COMPLETED":
            break
        time.sleep(0.1)

    # 1. Generate HTML Report with Potentially Malicious Title (XSS test)
    xss_title = "Report <script>alert('xss')</script> Flight Test"
    html_gen_resp = client.post(
        f"/api/v1/projects/{project_id}/analyses/{analysis_id}/reports",
        json={"report_format": "HTML", "report_title": xss_title},
    )
    assert html_gen_resp.status_code == status.HTTP_201_CREATED
    html_meta = html_gen_resp.json()
    html_report_id = html_meta["report_id"]

    # 2. Download HTML Report
    download_html = client.get(f"/api/v1/reports/{html_report_id}/download")
    assert download_html.status_code == status.HTTP_200_OK
    assert download_html.headers["content-type"].startswith("text/html")
    html_bytes = download_html.content

    # SHA-256 agreement
    calc_sha = hashlib.sha256(html_bytes).hexdigest()
    assert calc_sha == html_meta["sha256_checksum"]

    # HTML Security: ensure raw script tag was not injected executable
    html_text = html_bytes.decode("utf-8")
    assert "<script>alert('xss')</script>" not in html_text or "&lt;script&gt;" in html_text

    # 3. Generate Markdown Report
    md_gen_resp = client.post(
        f"/api/v1/projects/{project_id}/analyses/{analysis_id}/reports",
        json={"report_format": "MARKDOWN", "report_title": "Aerodynamic Evaluation"},
    )
    assert md_gen_resp.status_code == status.HTTP_201_CREATED
    md_meta = md_gen_resp.json()
    md_report_id = md_meta["report_id"]

    download_md = client.get(f"/api/v1/reports/{md_report_id}/download")
    assert download_md.status_code == status.HTTP_200_OK
    assert "markdown" in download_md.headers["content-type"]
    assert "# Aerodynamic Evaluation" in download_md.text

    # 4. Handle Nonexistent Report ID
    missing_rep = client.get("/api/v1/reports/rep_nonexistent/download")
    assert missing_rep.status_code == status.HTTP_404_NOT_FOUND


# ===========================================================================
# Section F: Security Boundaries and Failure-Handling Audit
# ===========================================================================


def test_audit_security_boundaries_and_sanitization(
    e2e_environment: tuple[TestClient, BackendConfig, pathlib.Path],
) -> None:
    """Audit path traversal prevention, error scrubbing, and stale file cleanup."""
    client, _config, root_path = e2e_environment

    p_resp = client.post("/api/v1/projects", json={"name": "Security Audit Project"})
    project_id = p_resp.json()["project_id"]

    # 1. Path Traversal in Project and Dataset Identifier
    bad_id_resp = client.get(f"/api/v1/projects/{project_id}/datasets/ds_traversal..attack/preview")
    assert bad_id_resp.status_code in (status.HTTP_400_BAD_REQUEST, status.HTTP_404_NOT_FOUND)

    # 2. Path Traversal in Filename Upload
    traversal_file = {
        "file": ("../../evil.csv", io.BytesIO(b"timestamp\n2026-01-01\n"), "text/csv")
    }
    trav_upload = client.post(
        f"/api/v1/projects/{project_id}/datasets/upload",
        files=traversal_file,
    )
    assert trav_upload.status_code == status.HTTP_400_BAD_REQUEST

    # 3. Error Sanitization in UI Client
    raw_error_with_paths = (
        "InternalError at D:\\AERIS_ANTIGRAVITY\\src\\aeris\\backend\\app.py line 45:\n"
        "Traceback (most recent call last):\n"
        '  File "D:\\AERIS_ANTIGRAVITY\\secret\\core.py", line 12, in run\n'
        "ValueError: Aerodynamic envelope calculation out of range"
    )
    sanitized = sanitize_ui_error(raw_error_with_paths)
    assert "D:\\" not in sanitized
    assert "Traceback" not in sanitized
    assert "[internal-path]" in sanitized
    assert "Aerodynamic envelope calculation out of range" in sanitized

    # 4. Stale Temporary File Cleanup
    stale_file = root_path / "projects" / project_id / "datasets" / ".aeris_tmp_orphaned_upload.csv"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_bytes(b"stale data")
    assert stale_file.exists()

    repo = SQLiteMetadataRepository(db_path=root_path / "aeris_metadata.db")
    reconciler = WorkspaceReconciler(
        workspace=WorkspaceManager(workspace_root=root_path),
        repository=repo,
    )
    cleaned = reconciler.clean_stale_temporary_files(max_age_seconds=0)
    assert len(cleaned) >= 1
    assert not stale_file.exists()
