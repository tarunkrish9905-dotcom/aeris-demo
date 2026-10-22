"""
Unit tests for AERIS Frontend REST API Client (AerisApiClient).
"""

from __future__ import annotations

import httpx
import pytest

from aeris.frontend.client import AerisApiClient, AerisApiError, sanitize_ui_error


def test_sanitize_ui_error_removes_paths_and_tracebacks() -> None:
    """Verify error sanitizer removes Windows and Unix paths and stack traces."""
    raw = (
        "Traceback (most recent call last):\n"
        "  File 'D:\\AERIS_ANTIGRAVITY\\secret\\runner.py', line 42, in run\n"
        "ValueError: File not found at /var/private/telemetry.csv: corrupt"
    )
    sanitized = sanitize_ui_error(raw)
    assert "D:\\AERIS_ANTIGRAVITY" not in sanitized
    assert "/var/private" not in sanitized
    assert "Traceback" not in sanitized
    assert "[internal-path]" in sanitized


def test_client_get_health_success() -> None:
    """Verify get_health parses 200 JSON response."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/health"
        return httpx.Response(200, json={"status": "healthy", "database_connected": True})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://test/api/v1")
    api = AerisApiClient(client=client)

    health = api.get_health()
    assert health["status"] == "healthy"
    assert health["database_connected"] is True


def test_client_get_version_success() -> None:
    """Verify get_version returns runtime details."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/version"
        return httpx.Response(
            200, json={"app_name": "AERIS", "version": "0.1.0", "api_version": "v1"}
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://test/api/v1")
    api = AerisApiClient(client=client)

    version = api.get_version()
    assert version["app_name"] == "AERIS"
    assert version["version"] == "0.1.0"


def test_client_project_crud() -> None:
    """Verify project list, create, update, and delete methods."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/v1/projects":
            return httpx.Response(
                200,
                json={"projects": [{"project_id": "proj_1", "name": "Flight 1"}], "total_count": 1},
            )
        if request.method == "POST" and request.url.path == "/api/v1/projects":
            return httpx.Response(201, json={"project_id": "proj_new", "name": "Created Flight"})
        if request.method == "PATCH" and request.url.path == "/api/v1/projects/proj_1":
            return httpx.Response(200, json={"project_id": "proj_1", "name": "Updated Flight"})
        if request.method == "DELETE" and request.url.path == "/api/v1/projects/proj_1":
            return httpx.Response(200, json={"success": True, "message": "Deleted"})
        return httpx.Response(404, json={"detail": "Not found"})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://test/api/v1")
    api = AerisApiClient(client=client)

    projects = api.list_projects()
    assert len(projects) == 1
    assert projects[0]["project_id"] == "proj_1"

    created = api.create_project(name="Created Flight")
    assert created["project_id"] == "proj_new"

    updated = api.update_project("proj_1", name="Updated Flight")
    assert updated["name"] == "Updated Flight"

    deleted = api.delete_project("proj_1")
    assert deleted["success"] is True


def test_client_dataset_operations() -> None:
    """Verify dataset list, upload, preview, and mapping methods."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/v1/projects/p1/datasets":
            return httpx.Response(
                200, json={"datasets": [{"dataset_id": "ds_1", "row_count": 500}]}
            )
        if request.method == "POST" and request.url.path == "/api/v1/projects/p1/datasets/upload":
            return httpx.Response(201, json={"dataset_id": "ds_uploaded", "row_count": 100})
        if (
            request.method == "GET"
            and request.url.path == "/api/v1/projects/p1/datasets/ds_1/preview"
        ):
            return httpx.Response(
                200,
                json={
                    "dataset_id": "ds_1",
                    "total_rows": 500,
                    "columns": ["time", "ias"],
                    "preview_rows": [],
                },
            )
        if (
            request.method == "GET"
            and request.url.path == "/api/v1/projects/p1/datasets/ds_1/mappings"
        ):
            return httpx.Response(
                200,
                json={
                    "dataset_id": "ds_1",
                    "mappings": [{"raw_column_name": "time", "target_parameter_id": "timestamp"}],
                },
            )
        if (
            request.method == "PUT"
            and request.url.path == "/api/v1/projects/p1/datasets/ds_1/mappings"
        ):
            return httpx.Response(200, json={"dataset_id": "ds_1", "mappings": []})
        return httpx.Response(404, json={"detail": "Not found"})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://test/api/v1")
    api = AerisApiClient(client=client)

    datasets = api.list_datasets("p1")
    assert len(datasets) == 1

    uploaded = api.upload_dataset("p1", "flight.csv", b"time,ias\n0,100")
    assert uploaded["dataset_id"] == "ds_uploaded"

    preview = api.get_dataset_preview("p1", "ds_1")
    assert preview["total_rows"] == 500

    mappings = api.get_parameter_mappings("p1", "ds_1")
    assert len(mappings) == 1

    saved = api.save_parameter_mappings("p1", "ds_1", [])
    assert saved["dataset_id"] == "ds_1"


def test_client_analysis_and_jobs() -> None:
    """Verify start_analysis, get_job_status, cancel_job, and get_analysis_detail."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/projects/p1/analyses":
            return httpx.Response(
                202, json={"job_id": "job_1", "analysis_id": "an_1", "state": "QUEUED"}
            )
        if request.method == "GET" and request.url.path == "/api/v1/jobs/job_1":
            return httpx.Response(
                200, json={"job_id": "job_1", "state": "RUNNING", "progress_percent": 50}
            )
        if request.method == "POST" and request.url.path == "/api/v1/jobs/job_1/cancel":
            return httpx.Response(200, json={"success": True, "message": "Signaled"})
        if request.method == "GET" and request.url.path == "/api/v1/projects/p1/analyses/an_1":
            return httpx.Response(
                200, json={"analysis_id": "an_1", "run_status": "COMPLETED", "total_issues": 0}
            )
        return httpx.Response(404, json={"detail": "Not found"})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://test/api/v1")
    api = AerisApiClient(client=client)

    started = api.start_analysis("p1", "ds_1")
    assert started["job_id"] == "job_1"

    status = api.get_job_status("job_1")
    assert status["state"] == "RUNNING"
    assert status["progress_percent"] == 50

    cancel = api.cancel_job("job_1")
    assert cancel["success"] is True

    detail = api.get_analysis_detail("p1", "an_1")
    assert detail["run_status"] == "COMPLETED"


def test_client_plots_and_available_signals() -> None:
    """Verify get_time_series_plot, get_multi_signal_plot, and get_available_signals."""

    def handler(request: httpx.Request) -> httpx.Response:
        if (
            request.method == "GET"
            and request.url.path == "/api/v1/projects/p1/analyses/an_1/plots/signals"
        ):
            return httpx.Response(
                200,
                json={
                    "dataset_id": "ds_1",
                    "signals": [
                        {
                            "signal_id": "indicated_airspeed",
                            "label": "Indicated Airspeed",
                            "unit": "knots",
                            "is_mapped": True,
                            "raw_column_name": "airspeed_kts",
                        }
                    ],
                    "default_signals": ["indicated_airspeed"],
                    "raw_columns": ["timestamp", "airspeed_kts"],
                },
            )
        if (
            request.method == "POST"
            and request.url.path == "/api/v1/projects/p1/analyses/an_1/plots/multi-signal"
        ):
            return httpx.Response(
                200,
                json={"figure_json": {"data": [], "layout": {}}, "parameter_id": "multi_signal"},
            )
        if (
            request.method == "GET"
            and request.url.path == "/api/v1/projects/p1/analyses/an_1/plots/time-series"
        ):
            return httpx.Response(
                200,
                json={
                    "figure_json": {"data": [], "layout": {}},
                    "parameter_id": "indicated_airspeed",
                },
            )
        return httpx.Response(404, json={"detail": "Not found"})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://test/api/v1")
    api = AerisApiClient(client=client)

    signals_resp = api.get_available_signals("p1", "an_1")
    assert signals_resp["dataset_id"] == "ds_1"
    assert signals_resp["default_signals"] == ["indicated_airspeed"]

    multi_plot = api.get_multi_signal_plot("p1", "an_1", ["indicated_airspeed"])
    assert "figure_json" in multi_plot

    ts_plot = api.get_time_series_plot("p1", "an_1", "indicated_airspeed")
    assert "figure_json" in ts_plot


def test_client_error_handling_rfc7807() -> None:
    """Verify RFC 7807 error responses raise AerisApiError with sanitized messages."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "error_code": "ERR_PROJECT_NOT_FOUND",
                "message": "Project D:\\AERIS\\proj_999 does not exist.",
                "detail": "Failed in D:\\AERIS\\repo.py",
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://test/api/v1")
    api = AerisApiClient(client=client)

    with pytest.raises(AerisApiError) as exc_info:
        api.get_project("proj_999")

    err = exc_info.value
    assert err.status_code == 404
    assert err.error_code == "ERR_PROJECT_NOT_FOUND"
    assert "D:\\AERIS" not in err.message
    assert "[internal-path]" in err.message
