"""
Integration tests for AERIS FastAPI + Dash mounting and UI state management.
"""

from __future__ import annotations

import pathlib
import tempfile
from typing import Any
from unittest.mock import MagicMock

from dash import no_update
from fastapi import status
from fastapi.testclient import TestClient

from aeris.backend.app import create_app
from aeris.backend.config import BackendConfig
from aeris.frontend.app import create_dash_app, mount_dash_app
from aeris.frontend.pages import (
    analysis_page,
    configs_page,
    dashboard_page,
    datasets_page,
    reports_page,
)


def test_fastapi_and_dash_coexistence_integration() -> None:
    """Verify FastAPI REST API and Dash WSGI frontend coexist on the same application."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = BackendConfig(workspace_root=pathlib.Path(tmp_dir))
        app = create_app(config=config, include_frontend=False)

        # Mount Dash app explicitly
        dash_app = create_dash_app()
        mount_dash_app(app, dash_app=dash_app)

        with TestClient(app) as client:
            # 1. Verify REST API health endpoint
            resp_api = client.get("/api/v1/health")
            assert resp_api.status_code == status.HTTP_200_OK
            assert resp_api.json()["status"] == "healthy"

            # 2. Verify Dash root index HTML
            resp_dash = client.get("/")
            assert resp_dash.status_code == status.HTTP_200_OK
            assert "html" in resp_dash.text.lower()

            # 3. Verify Dash layout endpoint
            resp_layout = client.get("/_dash-layout")
            assert resp_layout.status_code == status.HTTP_200_OK
            layout_data = resp_layout.json()
            assert "props" in layout_data


def test_polling_termination_logic_on_terminal_states() -> None:
    """Verify that job polling terminates (is_terminal = True) when reaching terminal states."""
    # Test COMPLETED state terminates polling
    status_completed = {
        "job_id": "job_123",
        "state": "COMPLETED",
        "current_stage": "COMPLETED",
        "progress_percent": 100,
        "completed_stages": 15,
        "total_stages": 15,
        "error_message": None,
    }
    card_comp, disabled_comp = analysis_page.process_job_poll_status(
        status_completed, "job_123", "an_123"
    )
    assert disabled_comp is True
    assert card_comp is not None

    # Test FAILED state terminates polling
    status_failed = {
        "job_id": "job_123",
        "state": "FAILED",
        "current_stage": "FAILED",
        "progress_percent": 40,
        "completed_stages": 6,
        "total_stages": 15,
        "error_message": "Fault in anomaly detection",
    }
    card_fail, disabled_fail = analysis_page.process_job_poll_status(
        status_failed, "job_123", "an_123"
    )
    assert disabled_fail is True
    assert card_fail is not None

    # Test CANCELLED state terminates polling
    status_cancelled = {
        "job_id": "job_123",
        "state": "CANCELLED",
        "current_stage": "CANCELLED",
        "progress_percent": 25,
        "completed_stages": 4,
        "total_stages": 15,
        "error_message": None,
    }
    card_cancel, disabled_cancel = analysis_page.process_job_poll_status(
        status_cancelled, "job_123", "an_123"
    )
    assert disabled_cancel is True
    assert card_cancel is not None

    # Test RUNNING state keeps interval active (is_terminal = False)
    status_running = {
        "job_id": "job_123",
        "state": "RUNNING",
        "current_stage": "ANOMALY_DETECTION",
        "progress_percent": 65,
        "completed_stages": 10,
        "total_stages": 15,
        "error_message": None,
    }
    card_run, disabled_run = analysis_page.process_job_poll_status(
        status_running, "job_123", "an_123"
    )
    assert disabled_run is False
    assert card_run is not None


def test_cooperative_cancellation_callback() -> None:
    """Verify cooperative cancellation invokes API client cancel_job."""
    cancelled_jobs: list[str] = []

    class MockCancelClient:
        def cancel_job(self, job_id: str) -> dict[str, Any]:
            cancelled_jobs.append(job_id)
            return {"success": True}

    alert = analysis_page.handle_cancel_job("job_to_cancel", MockCancelClient())  # type: ignore[arg-type]
    assert alert is not None
    assert "job_to_cancel" in cancelled_jobs

    # Test empty job_id
    alert_empty = analysis_page.handle_cancel_job(None, MockCancelClient())  # type: ignore[arg-type]
    assert alert_empty is not None


def test_multipage_switching_and_view_rendering_integration() -> None:
    """Verify all 5 pages render their views cleanly both with and without active project."""
    app = create_dash_app()
    mock_client = MagicMock()
    mock_client.list_datasets.return_value = []
    mock_client.get_builtin_configs.return_value = []
    mock_client.list_project_configs.return_value = []
    mock_client.list_reports.return_value = []
    mock_client.get_multi_signal_plot.return_value = {"figure_json": {"data": [], "layout": {}}}

    datasets_page.register_callbacks(app, mock_client)
    configs_page.register_callbacks(app, mock_client)
    dashboard_page.register_callbacks(app, mock_client)
    reports_page.register_callbacks(app, mock_client)

    # 1. Test when session has active project and analysis
    active_session = {
        "active_project_id": "proj_flight_402",
        "active_project_name": "Flutter Envelope 402",
        "active_analysis_id": "an_run_01",
    }
    cb_map = app.callback_map
    v_ds = cb_map["datasets-page-content.children"]["callback"].__wrapped__(active_session)
    assert v_ds is not None

    v_cfg = cb_map["configs-page-content.children"]["callback"].__wrapped__(active_session)
    assert v_cfg is not None

    v_dash = cb_map["dashboard-page-content.children"]["callback"].__wrapped__(active_session)
    assert v_dash is not None

    v_rep = cb_map["reports-page-content.children"]["callback"].__wrapped__(active_session)
    assert v_rep is not None

    # 2. Test when session has no active project (empty state)
    empty_session: dict[str, Any] = {}
    assert (
        cb_map["datasets-page-content.children"]["callback"].__wrapped__(empty_session) is not None
    )
    assert (
        cb_map["configs-page-content.children"]["callback"].__wrapped__(empty_session) is not None
    )
    assert (
        cb_map["dashboard-page-content.children"]["callback"].__wrapped__(empty_session) is not None
    )
    assert (
        cb_map["reports-page-content.children"]["callback"].__wrapped__(empty_session) is not None
    )


def test_action_callbacks_with_empty_inputs_do_not_throw() -> None:
    """Verify action callbacks safely return no_update when components are absent ([])."""
    app = create_dash_app()
    mock_client = MagicMock()
    mock_client.list_datasets.return_value = []
    mock_client.list_project_configs.return_value = []
    mock_client.list_reports.return_value = []

    datasets_page.register_callbacks(app, mock_client)
    configs_page.register_callbacks(app, mock_client)
    dashboard_page.register_callbacks(app, mock_client)
    reports_page.register_callbacks(app, mock_client)

    # 1. Dataset upload/refresh with empty pattern-matching inputs
    ds_action_fn = app.callback_map[
        "..datasets-table-container.children...dataset-upload-alert.children.."
    ]["callback"].__wrapped__
    ds_res = ds_action_fn([], [], [], {"active_project_id": "proj_1"})
    assert ds_res == (no_update, no_update)
    ds_refresh = ds_action_fn([1], [], [], {"active_project_id": "proj_1"})
    assert ds_refresh[0] is not None and ds_refresh != (no_update, no_update)

    # 2. Config upload/refresh with empty pattern-matching inputs
    cfg_action_fn = app.callback_map[
        "..custom-configs-table-container.children...config-upload-alert.children.."
    ]["callback"].__wrapped__
    cfg_res = cfg_action_fn([], [], [], "VEHICLE", {"active_project_id": "proj_1"})
    assert cfg_res == (no_update, no_update)
    cfg_refresh = cfg_action_fn([1], [], [], "VEHICLE", {"active_project_id": "proj_1"})
    assert cfg_refresh[0] is not None and cfg_refresh != (no_update, no_update)

    # 3. Dashboard plot update with empty pattern-matching clicks
    dash_action_fn = app.callback_map["dashboard-plot-container.children"]["callback"].__wrapped__
    dash_res = dash_action_fn(
        [],
        ["indicated_airspeed"],
        "STACKED",
        True,
        True,
        False,
        {"active_project_id": "proj_1", "active_analysis_id": "an_1"},
    )
    assert dash_res == no_update

    # 4. Reports generation/refresh with empty pattern-matching clicks
    rep_action_fn = app.callback_map[
        "..reports-table-container.children...report-generate-alert.children.."
    ]["callback"].__wrapped__
    rep_session = {"active_project_id": "proj_1", "active_analysis_id": "an_1"}
    rep_res = rep_action_fn([], [], "HTML", "Test", True, rep_session)
    assert rep_res == (no_update, no_update)
    rep_refresh = rep_action_fn([1], [], "HTML", "Test", True, rep_session)
    assert rep_refresh[0] is not None and rep_refresh != (no_update, no_update)
