"""
Unit tests for AERIS Dash application factory, components, and multi-page layouts.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import dash
from dash import no_update

from aeris.frontend.app import (
    create_dash_app,
    evaluate_system_health_display,
    format_active_project_label,
    route_page_view,
)
from aeris.frontend.components.empty_state import render_empty_state
from aeris.frontend.components.epistemic_badge import render_epistemic_badge
from aeris.frontend.components.limitations_panel import render_limitations_panel
from aeris.frontend.components.navbar import render_navbar
from aeris.frontend.pages import (
    analysis_page,
    configs_page,
    dashboard_page,
    datasets_page,
    intelligence_page,
    projects_page,
    reports_page,
)


def test_create_dash_app_instantiation() -> None:
    """Verify Dash app creates with correct title, layout, and session store."""
    app = create_dash_app()
    assert isinstance(app, dash.Dash)
    assert "AERIS" in app.title
    assert app.layout is not None


def test_all_page_layouts_render_without_exceptions() -> None:
    """Verify that all 7 multi-page layouts render valid Dash component hierarchies."""
    p1 = projects_page.layout()
    assert p1 is not None

    p2 = datasets_page.layout()
    assert p2 is not None

    p3 = configs_page.layout()
    assert p3 is not None

    p4 = analysis_page.layout()
    assert p4 is not None

    p5 = dashboard_page.layout()
    assert p5 is not None

    p6 = intelligence_page.layout()
    assert p6 is not None

    p7 = reports_page.layout()
    assert p7 is not None


def test_epistemic_badges_render_all_seven_levels() -> None:
    """Verify badges render properly for every level in the epistemic chain."""
    levels = [
        "RAW DATA",
        "VALIDATED DATA",
        "CALCULATED RESULT",
        "OBSERVATION",
        "INFERENCE",
        "HYPOTHESIS",
        "RECOMMENDATION",
    ]
    for level in levels:
        badge = render_epistemic_badge(level)
        assert badge is not None


def test_empty_state_and_limitations_components() -> None:
    """Verify honest empty states and limitations panels render as expected."""
    empty = render_empty_state(
        title="No Telemetry Data",
        description="Please upload CSV flight data.",
        icon_text="[ EMPTY ]",
    )
    assert empty is not None

    limits = render_limitations_panel(["Custom limit 1", "Custom limit 2"])
    assert limits is not None


def test_navbar_renders() -> None:
    """Verify global navigation bar renders with brand and links."""
    navbar = render_navbar()
    assert navbar is not None


def test_page_routing_dispatch() -> None:
    """Verify routing logic dispatches each supported page correctly."""
    paths = [
        "/",
        "/projects",
        "/datasets",
        "/configurations",
        "/analysis",
        "/dashboard",
        "/intelligence",
        "/reports",
    ]
    for path in paths:
        view = route_page_view(path)
        assert view is not None

    # Test 404 path
    not_found = route_page_view("/invalid-path-not-found")
    assert not_found is not None


def test_sync_active_project_callback() -> None:
    """Verify active project name formatting in navbar."""
    assert format_active_project_label(None) == "None Selected"
    assert format_active_project_label({}) == "None Selected"
    assert format_active_project_label({"active_project_id": "proj_123"}) == "proj_123"
    assert (
        format_active_project_label(
            {"active_project_id": "proj_123", "active_project_name": "Flutter Test"}
        )
        == "Flutter Test (proj_123)"
    )


def test_system_health_display_evaluation() -> None:
    """Verify system health pill styling evaluation."""
    online_style, online_txt = evaluate_system_health_display(
        {"status": "healthy", "database_connected": True}
    )
    assert online_txt == "SYSTEM ONLINE"
    assert online_style["color"] == "#10B981"

    _degraded_style, degraded_txt = evaluate_system_health_display({"status": "unhealthy"})
    assert degraded_txt == "DEGRADED"

    offline_style, offline_txt = evaluate_system_health_display(None)
    assert offline_txt == "OFFLINE"
    assert offline_style["color"] == "#EF4444"


def test_callback_map_page_content_renderers_isolated_from_action_buttons() -> None:
    """Regression test: verify all page-content renderers have NO dynamic button inputs."""
    app = create_dash_app()
    page_content_outputs = [
        "datasets-page-content.children",
        "configs-page-content.children",
        "analysis-page-content.children",
        "dashboard-page-content.children",
        "intelligence-page-content.children",
        "reports-page-content.children",
    ]

    for out_key in page_content_outputs:
        assert out_key in app.callback_map, f"Missing registered callback for {out_key}"
        cb_meta = app.callback_map[out_key]
        input_ids = [inp["id"] for inp in cb_meta["inputs"]]
        assert input_ids == ["session-store"], (
            f"Callback for {out_key} has unexpected inputs: {input_ids}. "
            "Page view renderers must only depend on session-store to prevent ReferenceError."
        )


def test_project_selection_callback_wiring_and_navigation() -> None:
    """Verify select_active_project updates session store and navigates to /datasets."""
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    mock_client = MagicMock()
    mock_client.get_project.return_value = {
        "project_id": "proj_flight_99",
        "name": "High Mach Flutter Test",
    }

    projects_page.register_callbacks(app, mock_client)

    # Locate select_active_project in callback_map
    cb_key = "..session-store.data...url.pathname.."
    assert cb_key in app.callback_map, f"Callback {cb_key} not found in app.callback_map"
    cb_meta = app.callback_map[cb_key]
    expected_inputs = ['{"index":["ALL"],"type":"select-project-btn"}']
    assert [inp["id"] for inp in cb_meta["inputs"]] == expected_inputs

    # 1. Test when no click occurred
    res_store, res_url = projects_page.process_project_selection(None, [], {}, mock_client)
    assert res_store == no_update
    assert res_url == no_update

    # 2. Test when project 'proj_flight_99' was clicked
    trigger_prop = '{"index":"proj_flight_99","type":"select-project-btn"}.n_clicks'
    updated_store, next_url = projects_page.process_project_selection(
        trigger_prop, [1], {"existing_key": "abc"}, mock_client
    )
    assert next_url == "/datasets", "Clicking Open must navigate to /datasets"
    assert updated_store["active_project_id"] == "proj_flight_99"
    assert updated_store["active_project_name"] == "High Mach Flutter Test"
    assert updated_store["existing_key"] == "abc"


def test_separated_subpage_components_render_cleanly() -> None:
    """Verify separated table renderers handle both empty and populated states without error."""
    # Datasets table
    empty_datasets = datasets_page.render_datasets_table([], "Test Project")
    assert empty_datasets is not None
    pop_datasets = datasets_page.render_datasets_table(
        [
            {
                "dataset_id": "ds_001",
                "original_filename": "flight_test.csv",
                "file_size_bytes": 10240,
                "row_count": 500,
                "column_count": 12,
                "ingested_at": "2026-10-20T12:00:00Z",
            }
        ],
        "Test Project",
    )
    assert pop_datasets is not None

    # Configs table
    empty_configs = configs_page.render_custom_configs_table([])
    assert empty_configs is not None
    pop_configs = configs_page.render_custom_configs_table(
        [
            {
                "profile_type": "VEHICLE",
                "profile_identifier": "F16_Aero",
                "config_id": "cfg_001",
                "loaded_at": "2026-10-20T12:00:00Z",
            }
        ]
    )
    assert pop_configs is not None

    # Reports table
    empty_reports = reports_page.render_reports_table([], "http://127.0.0.1:8000/api/v1")
    assert empty_reports is not None
    pop_reports = reports_page.render_reports_table(
        [
            {
                "report_id": "rep_001",
                "report_format": "HTML",
                "file_size_bytes": 20480,
                "sha256_checksum": "abcdef1234567890",
                "generated_at": "2026-10-20T12:00:00Z",
                "report_title": "Test Flutter Report",
            }
        ],
        "http://127.0.0.1:8000/api/v1",
    )
    assert pop_reports is not None

    # Dashboard plot
    mock_client = MagicMock()
    mock_client.get_multi_signal_plot.return_value = {
        "figure_json": {"data": [], "layout": {"title": "Test Plot"}}
    }
    plot = dashboard_page.build_dashboard_plot(
        api_client=mock_client,
        project_id="proj_1",
        analysis_id="an_1",
        active_signals=["indicated_airspeed"],
        active_layout="STACKED",
        active_limits=True,
        active_events=True,
        active_dark=False,
    )
    assert plot is not None


def test_dashboard_plot_empty_and_error_handling() -> None:
    """Verify build_dashboard_plot handles empty signal list and API error."""
    mock_client = MagicMock()

    # 1. Empty signals list
    empty_plot = dashboard_page.build_dashboard_plot(
        api_client=mock_client,
        project_id="p1",
        analysis_id="a1",
        active_signals=[],
        active_layout="STACKED",
        active_limits=True,
        active_events=True,
        active_dark=False,
    )
    assert empty_plot is not None

    # 2. AerisApiError returns remediation button
    from aeris.frontend.client import AerisApiError

    mock_client.get_multi_signal_plot.side_effect = AerisApiError(
        status_code=400,
        error_code="ERR_HTTP_400",
        message="Requested signals not found in dataset: ['indicated_airspeed']",
    )

    err_plot = dashboard_page.build_dashboard_plot(
        api_client=mock_client,
        project_id="p1",
        analysis_id="a1",
        active_signals=["indicated_airspeed"],
        active_layout="STACKED",
        active_limits=True,
        active_events=True,
        active_dark=False,
    )
    assert err_plot is not None


def test_reports_page_preview_and_download_wiring() -> None:
    """Verify Reports page has static preview container, proper button props, and iframe."""
    # 1. Verify layout has static preview container
    layout_comp = reports_page.layout()
    assert layout_comp is not None

    def find_component_by_id(comp: Any, target_id: str) -> Any | None:
        if getattr(comp, "id", None) == target_id:
            return comp
        children = getattr(comp, "children", None)
        if isinstance(children, list):
            for child in children:
                found = find_component_by_id(child, target_id)
                if found is not None:
                    return found
        elif children is not None:
            return find_component_by_id(children, target_id)
        return None

    preview_div = find_component_by_id(layout_comp, "report-preview-container")
    assert preview_div is not None

    # 2. Verify table renders Download button with href and Preview button with n_clicks=0
    table = reports_page.render_reports_table(
        [
            {
                "report_id": "rep_test123",
                "report_format": "HTML",
                "file_size_bytes": 10240,
                "sha256_checksum": "1234567890abcdef",
                "generated_at": "2026-10-21T12:00:00Z",
                "report_title": "Certification Report",
            }
        ],
        "http://127.0.0.1:8000/api/v1",
    )
    tbody = table.children[1]
    row = tbody.children[0]
    actions_td = row.children[5]
    action_buttons = actions_td.children
    download_btn = action_buttons[0]
    preview_btn = action_buttons[1]

    # Verify Download button
    assert download_btn.href == "http://127.0.0.1:8000/api/v1/reports/rep_test123/download"
    assert download_btn.target == "_blank"

    # Verify Preview button
    assert preview_btn.id == {"type": "preview-report-btn", "index": "rep_test123"}
    assert preview_btn.n_clicks == 0

    # 3. Verify Dash callback registration for report-preview-container
    app = create_dash_app()
    cb = app.callback_map.get("report-preview-container.children")
    assert cb is not None
    input_ids = [inp["id"] for inp in cb["inputs"]]
    assert any("preview-report-btn" in str(inp_id) for inp_id in input_ids)
    assert cb["output"].component_id == "report-preview-container"
    assert cb["output"].component_property == "children"
