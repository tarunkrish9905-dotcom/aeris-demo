"""
Tests for AERIS Backend Application Factory, Lifespan, Health, and Error Handling.
"""

from __future__ import annotations

import pathlib
import tempfile

from fastapi import status
from fastapi.testclient import TestClient

from aeris.backend.app import create_app
from aeris.backend.config import BackendConfig
from aeris.backend.errors import sanitize_error_message


def test_sanitize_error_message_removes_paths() -> None:
    """Verify filesystem paths are scrubbed from error messages."""
    win_path = "Error reading D:\\AERIS_ANTIGRAVITY\\secret\\telemetry.csv: corrupted file"
    sanitized = sanitize_error_message(win_path)
    assert "D:\\AERIS_ANTIGRAVITY" not in sanitized
    assert "[managed_path]" in sanitized

    unix_path = "File missing at /var/data/telemetry.csv in directory"
    sanitized_unix = sanitize_error_message(unix_path)
    assert "/var/data" not in sanitized_unix
    assert "[managed_path]" in sanitized_unix


def test_app_creation_and_lifespan_repeated() -> None:
    """Verify application factory and clean startup/shutdown lifecycle."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = BackendConfig(
            workspace_root=pathlib.Path(tmp_dir),
            app_version="0.1.0-test",
        )
        app = create_app(config)

        # First lifecycle
        with TestClient(app) as client:
            resp = client.get("/api/v1/health")
            assert resp.status_code == status.HTTP_200_OK
            data = resp.json()
            assert data["status"] == "healthy"
            assert data["database_connected"] is True
            assert data["active_jobs_count"] == 0

        # Second lifecycle on same workspace
        with TestClient(app) as client:
            resp = client.get("/api/v1/version")
            assert resp.status_code == status.HTTP_200_OK
            data = resp.json()
            assert data["app_name"] == "AERIS"
            assert data["version"] == "0.1.0-test"
            assert data["api_version"] == "v1"


def test_openapi_schema_generated() -> None:
    """Verify OpenAPI documentation endpoint is reachable."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = BackendConfig(workspace_root=pathlib.Path(tmp_dir))
        app = create_app(config)
        with TestClient(app) as client:
            resp = client.get("/openapi.json")
            assert resp.status_code == status.HTTP_200_OK
            schema = resp.json()
            assert "paths" in schema
            assert "/api/v1/health" in schema["paths"]
            assert "/api/v1/projects" in schema["paths"]


def test_validation_error_response_rfc7807() -> None:
    """Verify schema validation error returns structured RFC 7807 response."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = BackendConfig(workspace_root=pathlib.Path(tmp_dir))
        app = create_app(config)
        with TestClient(app) as client:
            # Send invalid project payload (name is empty string)
            resp = client.post("/api/v1/projects", json={"name": ""})
            assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
            data = resp.json()
            assert data["error_code"] == "ERR_REQUEST_VALIDATION_FAILED"
            assert "detail" in data
            assert "timestamp" in data
            assert "request_id" in data


def test_not_found_error_response() -> None:
    """Verify querying non-existent project returns 404 with structured error."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = BackendConfig(workspace_root=pathlib.Path(tmp_dir))
        app = create_app(config)
        with TestClient(app) as client:
            resp = client.get("/api/v1/projects/proj_nonexistent_123")
            assert resp.status_code == status.HTTP_404_NOT_FOUND
            data = resp.json()
            assert data["error_code"] == "ERR_PROJECT_NOT_FOUND"
            assert "proj_nonexistent_123" in data["message"]
