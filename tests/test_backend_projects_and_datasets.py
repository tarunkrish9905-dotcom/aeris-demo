"""
Tests for AERIS Backend Projects, Datasets, and Parameter Mappings Endpoints.
"""

from __future__ import annotations

import io
import pathlib
import tempfile
from collections.abc import Generator

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from aeris.backend.app import create_app
from aeris.backend.config import BackendConfig

SAMPLE_CSV = (
    b"timestamp,altitude_ft,airspeed_kts,pitch_deg\n"
    b"2026-01-01T12:00:00Z,5000.0,150.2,2.5\n"
    b"2026-01-01T12:00:01Z,5010.0,151.0,2.6\n"
    b"2026-01-01T12:00:02Z,5020.0,151.8,2.7\n"
)


@pytest.fixture
def client_env() -> Generator[tuple[TestClient, pathlib.Path], None, None]:
    """Provide an isolated TestClient backed by a temporary workspace."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = pathlib.Path(tmp_dir)
        config = BackendConfig(
            workspace_root=root,
            max_upload_size_bytes=500 * 1024,  # 500 KB limit for testing
        )
        app = create_app(config)
        with TestClient(app) as client:
            yield client, root


def test_project_lifecycle(client_env: tuple[TestClient, pathlib.Path]) -> None:
    """Verify project creation, retrieval, listing, patching, and soft deletion."""
    client, root = client_env

    # 1. Create project
    create_payload = {
        "name": "Flight Test Alpha",
        "description": "Initial envelope expansion",
        "analyst_name": "Lead Engineer",
    }
    resp = client.post("/api/v1/projects", json=create_payload)
    assert resp.status_code == status.HTTP_201_CREATED
    data = resp.json()
    project_id = data["project_id"]
    assert data["name"] == "Flight Test Alpha"
    assert data["status"] == "ACTIVE"

    # Verify physical directory was scaffolded
    assert (root / "projects" / project_id / "datasets").exists()

    # 2. Get project details
    resp = client.get(f"/api/v1/projects/{project_id}")
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json()["project_id"] == project_id

    # 3. List projects
    resp = client.get("/api/v1/projects")
    assert resp.status_code == status.HTTP_200_OK
    list_data = resp.json()
    assert list_data["total_count"] >= 1
    assert any(p["project_id"] == project_id for p in list_data["projects"])

    # 4. Patch project
    patch_payload = {"name": "Flight Test Alpha (Updated)", "analyst_name": "Senior FTE"}
    resp = client.patch(f"/api/v1/projects/{project_id}", json=patch_payload)
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json()["name"] == "Flight Test Alpha (Updated)"
    assert resp.json()["analyst_name"] == "Senior FTE"

    # 5. Soft delete
    resp = client.delete(f"/api/v1/projects/{project_id}")
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json()["success"] is True

    # Verify status changed to DELETED
    resp = client.get(f"/api/v1/projects/{project_id}")
    assert resp.json()["status"] == "DELETED"


def test_project_purge(client_env: tuple[TestClient, pathlib.Path]) -> None:
    """Verify permanent purge deletes project record and unlinks workspace directories."""
    client, root = client_env

    resp = client.post("/api/v1/projects", json={"name": "Temp Project"})
    project_id = resp.json()["project_id"]
    proj_dir = root / "projects" / project_id
    assert proj_dir.exists()

    resp = client.delete(f"/api/v1/projects/{project_id}?purge=true")
    assert resp.status_code == status.HTTP_200_OK
    assert not proj_dir.exists()

    # Querying purged project should return 404
    resp = client.get(f"/api/v1/projects/{project_id}")
    assert resp.status_code == status.HTTP_404_NOT_FOUND


def test_dataset_upload_preview_and_mappings(client_env: tuple[TestClient, pathlib.Path]) -> None:
    """Verify dataset upload, tabular preview, and parameter mapping workflow."""
    client, _root = client_env

    # Create parent project
    p_resp = client.post("/api/v1/projects", json={"name": "Dataset Project"})
    project_id = p_resp.json()["project_id"]

    # 1. Upload telemetry CSV
    files = {"file": ("flight_telemetry.csv", io.BytesIO(SAMPLE_CSV), "text/csv")}
    upload_resp = client.post(f"/api/v1/projects/{project_id}/datasets/upload", files=files)
    assert upload_resp.status_code == status.HTTP_201_CREATED
    d_data = upload_resp.json()
    dataset_id = d_data["dataset_id"]
    assert d_data["row_count"] == 3
    assert d_data["column_count"] == 4
    assert "altitude_ft" in d_data["detected_columns"]

    # 2. List datasets in project
    list_resp = client.get(f"/api/v1/projects/{project_id}/datasets")
    assert list_resp.status_code == status.HTTP_200_OK
    assert list_resp.json()["total_count"] == 1

    # 3. Preview dataset
    prev_resp = client.get(f"/api/v1/projects/{project_id}/datasets/{dataset_id}/preview?rows=2")
    assert prev_resp.status_code == status.HTTP_200_OK
    prev_data = prev_resp.json()
    assert len(prev_data["preview_rows"]) == 2
    assert "altitude_ft" in prev_data["columns"]

    # 4. Get auto-suggested mappings
    map_resp = client.get(f"/api/v1/projects/{project_id}/datasets/{dataset_id}/mappings")
    assert map_resp.status_code == status.HTTP_200_OK
    assert len(map_resp.json()["mappings"]) == 4

    # 5. Save parameter mappings
    save_payload = {
        "mappings": [
            {
                "raw_column_name": "altitude_ft",
                "target_parameter_id": "pressure_altitude",
                "source_unit": "ft",
                "target_unit": "m",
                "is_timestamp": False,
                "data_type": "float",
            },
            {
                "raw_column_name": "timestamp",
                "target_parameter_id": "time",
                "is_timestamp": True,
                "data_type": "string",
            },
        ]
    }
    put_resp = client.put(
        f"/api/v1/projects/{project_id}/datasets/{dataset_id}/mappings", json=save_payload
    )
    assert put_resp.status_code == status.HTTP_200_OK
    assert len(put_resp.json()["mappings"]) == 2


def test_dataset_upload_size_limit_and_unsafe_filename(
    client_env: tuple[TestClient, pathlib.Path],
) -> None:
    """Verify upload limit rejection (413) and path traversal rejection (400)."""
    client, _root = client_env

    p_resp = client.post("/api/v1/projects", json={"name": "Safety Project"})
    project_id = p_resp.json()["project_id"]

    # Oversized payload (> 500 KB configured in fixture)
    huge_payload = b"col1,col2\n" + (b"1.0,2.0\n" * 80000)
    files = {"file": ("big_flight.csv", io.BytesIO(huge_payload), "text/csv")}
    resp = client.post(f"/api/v1/projects/{project_id}/datasets/upload", files=files)
    assert resp.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    assert resp.json()["error_code"] == "ERR_FILE_TOO_LARGE"

    # Path traversal in filename
    files = {"file": ("../../escape.csv", io.BytesIO(SAMPLE_CSV), "text/csv")}
    resp = client.post(f"/api/v1/projects/{project_id}/datasets/upload", files=files)
    assert resp.status_code == status.HTTP_400_BAD_REQUEST
    assert resp.json()["error_code"] == "ERR_PATH_TRAVERSAL_DETECTED"
