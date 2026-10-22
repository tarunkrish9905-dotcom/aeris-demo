"""
Tests for AERIS Backend Background Jobs, Pipeline Execution, Cancellation, and Crash Recovery.
"""

from __future__ import annotations

import datetime
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
from aeris.storage.models import AnalysisRunRecord, BackgroundJobRecord
from aeris.storage.repository import SQLiteMetadataRepository
from aeris.storage.workspace import WorkspaceManager

SAMPLE_TELEMETRY = (
    b"timestamp,altitude_ft,airspeed_kts,pitch_deg\n"
    b"2026-01-01T12:00:00Z,5000.0,150.2,2.5\n"
    b"2026-01-01T12:00:01Z,5010.0,151.0,2.6\n"
    b"2026-01-01T12:00:02Z,5020.0,151.8,2.7\n"
    b"2026-01-01T12:00:03Z,5030.0,152.5,2.8\n"
)


@pytest.fixture
def test_setup() -> Generator[tuple[TestClient, str, str, pathlib.Path], None, None]:
    """Provide TestClient with a pre-created project and dataset."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = pathlib.Path(tmp_dir)
        config = BackendConfig(workspace_root=root)
        app = create_app(config)

        with TestClient(app) as client:
            # Create project
            p_resp = client.post("/api/v1/projects", json={"name": "Job Test Project"})
            project_id = p_resp.json()["project_id"]

            # Upload dataset
            files = {"file": ("flight.csv", io.BytesIO(SAMPLE_TELEMETRY), "text/csv")}
            d_resp = client.post(f"/api/v1/projects/{project_id}/datasets/upload", files=files)
            dataset_id = d_resp.json()["dataset_id"]

            yield client, project_id, dataset_id, root


def test_submit_analysis_and_poll_to_completion(
    test_setup: tuple[TestClient, str, str, pathlib.Path],
) -> None:
    """Verify submitting analysis job, tracking stage progress, and retrieving completed run."""
    client, project_id, dataset_id, _root = test_setup

    # 1. Submit analysis run
    submit_payload = {
        "dataset_id": dataset_id,
        "report_title": "Automated Flight Test Run",
    }
    resp = client.post(f"/api/v1/projects/{project_id}/analyses", json=submit_payload)
    assert resp.status_code == status.HTTP_202_ACCEPTED
    data = resp.json()
    job_id = data["job_id"]
    analysis_id = data["analysis_id"]
    assert data["state"] == "QUEUED"

    # 2. Poll job status until terminal state
    max_wait_seconds = 10.0
    start = time.time()
    terminal_state = None

    while time.time() - start < max_wait_seconds:
        status_resp = client.get(f"/api/v1/jobs/{job_id}")
        assert status_resp.status_code == status.HTTP_200_OK
        job_data = status_resp.json()
        if job_data["state"] in ("COMPLETED", "SUCCEEDED", "FAILED", "CANCELLED"):
            terminal_state = job_data["state"]
            break
        time.sleep(0.1)

    assert terminal_state == "COMPLETED"
    assert job_data["progress_percent"] == 100
    assert job_data["completed_stages"] == 3
    assert job_data["current_stage"] == "COMPLETED"

    # 3. Retrieve analysis run details
    analysis_resp = client.get(f"/api/v1/projects/{project_id}/analyses/{analysis_id}")
    assert analysis_resp.status_code == status.HTTP_200_OK
    run_data = analysis_resp.json()
    assert run_data["run_status"] == "COMPLETED"
    assert run_data["execution_duration_ms"] is not None
    assert run_data["result_summary"] is not None
    assert "overall_status" in run_data["result_summary"]


def test_cooperative_cancellation(
    test_setup: tuple[TestClient, str, str, pathlib.Path],
) -> None:
    """Verify cooperative cancellation sets job and analysis state to CANCELLED."""
    client, project_id, dataset_id, _root = test_setup

    resp = client.post(
        f"/api/v1/projects/{project_id}/analyses",
        json={"dataset_id": dataset_id},
    )
    job_id = resp.json()["job_id"]
    analysis_id = resp.json()["analysis_id"]

    # Request cancellation
    cancel_resp = client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert cancel_resp.status_code == status.HTTP_200_OK
    assert cancel_resp.json()["success"] is True

    # Poll status
    status_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert status_resp.status_code == status.HTTP_200_OK
    assert status_resp.json()["state"] == "CANCELLED"

    # Check analysis run record
    run_resp = client.get(f"/api/v1/projects/{project_id}/analyses/{analysis_id}")
    assert run_resp.status_code == status.HTTP_200_OK
    assert run_resp.json()["run_status"] == "CANCELLED"


def test_startup_crash_recovery_interrupted_jobs() -> None:
    """Verify jobs left in PENDING/RUNNING at application startup are transitioned to FAILED."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = pathlib.Path(tmp_dir)
        workspace = WorkspaceManager(workspace_root=root)
        workspace.init_workspace()
        repo = SQLiteMetadataRepository(db_path=workspace.db_path)

        now = datetime.datetime.now(datetime.UTC)

        from aeris.storage.models import DatasetRecord, ProjectRecord, ProjectStatus

        repo.create_project(
            ProjectRecord(
                project_id="p_crash",
                name="Crash Test",
                description="",
                analyst_name=None,
                created_at=now,
                updated_at=now,
                status=ProjectStatus.ACTIVE,
            )
        )

        repo.create_dataset(
            DatasetRecord(
                dataset_id="ds_crash",
                project_id="p_crash",
                original_filename="flight.csv",
                storage_relative_path="projects/p_crash/datasets/ds_crash.csv",
                sha256_checksum="0" * 64,
                file_size_bytes=100,
                row_count=10,
                column_count=2,
                detected_columns=["timestamp", "altitude_ft"],
                ingested_at=now,
            )
        )

        # Inject simulated interrupted analysis run and background job
        repo.create_analysis_run(
            AnalysisRunRecord(
                analysis_id="run_crashed",
                project_id="p_crash",
                dataset_id="ds_crash",
                run_status="RUNNING",
                triggered_at=now,
            )
        )
        repo.create_background_job(
            BackgroundJobRecord(
                job_id="job_crashed",
                analysis_id="run_crashed",
                state="RUNNING",
                current_stage="ANOMALY_DETECTION",
                completed_stages_count=9,
                total_stages_count=15,
                progress_percent=60,
                error_message=None,
                created_at=now,
                updated_at=now,
            )
        )

        # Start FastAPI application which runs lifespan recovery
        config = BackendConfig(workspace_root=root)
        app = create_app(config)

        with TestClient(app) as client:
            resp = client.get("/api/v1/jobs/job_crashed")
            assert resp.status_code == status.HTTP_200_OK
            data = resp.json()
            assert data["state"] == "FAILED"
            assert "ERR_SYSTEM_RESTARTED_DURING_EXECUTION" in data["error_message"]

            # Analysis run should also be marked FAILED
            run_resp = client.get("/api/v1/projects/p_crash/analyses/run_crashed")
            assert run_resp.status_code == status.HTTP_200_OK
            assert run_resp.json()["run_status"] == "FAILED"
