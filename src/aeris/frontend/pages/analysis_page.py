"""
AERIS Analysis Runner & Live Progress Monitor Page (/analysis).

Submits 15-stage analytical pipeline runs as background jobs and provides real-time,
bounded-interval progress monitoring with cooperative cancellation.
"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from dash.development.base_component import Component

from aeris.frontend.client import AerisApiClient, AerisApiError
from aeris.frontend.components.empty_state import render_empty_state
from aeris.frontend.theme import CARD_STYLE, HEADER_STYLE, SUBTITLE_STYLE


def layout() -> Component:
    """Render the Analysis Runner & Live Progress Monitor page layout."""
    return dbc.Container(
        [
            dcc.Interval(id="job-poll-interval", interval=1000, n_intervals=0, disabled=True),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("15-Stage Pipeline Execution Runner", style=HEADER_STYLE),
                            html.P(
                                (
                                    "Configure and execute the complete AERIS analytical "
                                    "pipeline with real-time stage monitoring."
                                ),
                                style=SUBTITLE_STYLE,
                            ),
                        ],
                        md=12,
                    ),
                ]
            ),
            html.Div(id="analysis-page-content"),
        ],
        fluid=True,
        className="py-4 px-4",
    )


def register_callbacks(app: Any, api_client: AerisApiClient) -> None:
    """Register all Dash callbacks for the Analysis Runner page."""

    @app.callback(
        Output("analysis-page-content", "children"),
        Input("session-store", "data"),
    )
    def render_analysis_setup(session_data: dict[str, Any] | None) -> Component:
        project_id = session_data.get("active_project_id") if session_data else None
        project_name = session_data.get("active_project_name", project_id) if session_data else None

        if not project_id:
            return render_empty_state(
                title="No Active Project Selected",
                description=(
                    "Select or create a flight test project on the Projects page "
                    "before configuring an analysis run."
                ),
                action_button=dbc.Button("Go to Projects", href="/projects", color="primary"),
                icon_text="[ WORKSPACE REQUIRED ]",
            )

        # Fetch datasets for project
        try:
            datasets = api_client.list_datasets(project_id)
        except AerisApiError as err:
            return dbc.Alert(f"Failed to fetch datasets: {err.message}", color="danger")

        if not datasets:
            return render_empty_state(
                title="No Datasets Ingested",
                description=(
                    "Ingest at least one telemetry CSV file on the Datasets & Mapping page "
                    "before triggering analysis."
                ),
                action_button=dbc.Button("Upload Telemetry", href="/datasets", color="primary"),
                icon_text="[ ZERO TELEMETRY ]",
            )

        dataset_options = [
            {
                "label": (
                    f"{d.get('original_filename')} "
                    f"({d.get('row_count', 0):,} rows) [{d.get('dataset_id')}]"
                ),
                "value": d.get("dataset_id"),
            }
            for d in datasets
        ]

        # Fetch built-in configs
        try:
            builtin_configs = api_client.get_builtin_configs()
        except AerisApiError:
            builtin_configs = []

        vehicle_options = [{"label": "-- Default Vehicle Model --", "value": ""}] + [
            {"label": f"{c.get('name')} [{c.get('profile_id')}]", "value": c.get("profile_id")}
            for c in builtin_configs
            if c.get("profile_type") == "VEHICLE"
        ]

        limit_options = [{"label": "-- Default Limits Envelope --", "value": ""}] + [
            {"label": f"{c.get('name')} [{c.get('profile_id')}]", "value": c.get("profile_id")}
            for c in builtin_configs
            if c.get("profile_type") == "LIMIT"
        ]

        setup_card = dbc.Card(
            dbc.CardBody(
                [
                    html.H5(f"Analysis Setup ({project_name})", className="fw-semibold mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Select Flight Telemetry Dataset",
                                        className="fw-semibold small",
                                    ),
                                    dbc.Select(
                                        id="analysis-dataset-select",
                                        options=dataset_options,
                                        value=dataset_options[0]["value"]
                                        if dataset_options
                                        else None,
                                        className="mb-3",
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Analysis Run Title", className="fw-semibold small"),
                                    dbc.Input(
                                        id="analysis-title-input",
                                        value="AERIS Flight Test Investigation Run",
                                        className="mb-3",
                                    ),
                                ],
                                md=6,
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Vehicle Aerodynamic Profile", className="fw-semibold small"
                                    ),
                                    dbc.Select(
                                        id="analysis-vehicle-select",
                                        options=vehicle_options,
                                        value="",
                                        className="mb-3",
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Engineering Limits Envelope", className="fw-semibold small"
                                    ),
                                    dbc.Select(
                                        id="analysis-limit-select",
                                        options=limit_options,
                                        value="",
                                        className="mb-3",
                                    ),
                                ],
                                md=6,
                            ),
                        ]
                    ),
                    html.Div(id="analysis-submit-error-alert"),
                    dbc.Button(
                        "Execute 15-Stage Analysis Pipeline",
                        id="start-analysis-btn",
                        color="primary",
                        size="lg",
                        className="mt-2",
                    ),
                ]
            ),
            style=CARD_STYLE,
            className="mb-4",
        )

        monitor_container = html.Div(id="live-job-monitor-container")

        return html.Div([setup_card, monitor_container])

    @app.callback(
        [
            Output("session-store", "data", allow_duplicate=True),
            Output("job-poll-interval", "disabled"),
            Output("analysis-submit-error-alert", "children"),
            Output("live-job-monitor-container", "children"),
        ],
        Input("start-analysis-btn", "n_clicks"),
        [
            State("session-store", "data"),
            State("analysis-dataset-select", "value"),
            State("analysis-title-input", "value"),
            State("analysis-vehicle-select", "value"),
            State("analysis-limit-select", "value"),
        ],
        prevent_initial_call=True,
    )
    def submit_analysis_job(
        _clicks: int | None,
        session_data: dict[str, Any] | None,
        dataset_id: str | None,
        title: str | None,
        vehicle_prof: str | None,
        limit_prof: str | None,
    ) -> tuple[dict[str, Any], bool, Component | None, Component | None]:
        store = dict(session_data or {})
        project_id = store.get("active_project_id")

        if not project_id or not dataset_id:
            return (
                store,
                True,
                dbc.Alert("Missing active project or dataset selection.", color="danger"),
                None,
            )

        try:
            resp = api_client.start_analysis(
                project_id=project_id,
                dataset_id=dataset_id,
                vehicle_profile_id=vehicle_prof if vehicle_prof else None,
                limit_profile_id=limit_prof if limit_prof else None,
                report_title=title or "AERIS Flight Test Investigation Run",
            )
            job_id = resp["job_id"]
            analysis_id = resp["analysis_id"]

            store["active_job_id"] = job_id
            store["active_analysis_id"] = analysis_id

            initial_monitor = _build_monitor_card(
                job_id=job_id,
                analysis_id=analysis_id,
                state="QUEUED",
                stage="QUEUED",
                progress_percent=0,
                completed_stages=0,
                total_stages=15,
                error_message=None,
            )

            # Enable interval polling
            return store, False, None, initial_monitor
        except AerisApiError as err:
            return (
                store,
                True,
                dbc.Alert(f"Failed to submit analysis: {err.message}", color="danger"),
                None,
            )

    @app.callback(
        [
            Output("live-job-monitor-container", "children", allow_duplicate=True),
            Output("job-poll-interval", "disabled", allow_duplicate=True),
        ],
        Input("job-poll-interval", "n_intervals"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def poll_job_progress(
        _interval: int, session_data: dict[str, Any] | None
    ) -> tuple[Component, bool]:
        if not session_data or not session_data.get("active_job_id"):
            return html.Div(), True

        job_id = session_data["active_job_id"]
        analysis_id = session_data.get("active_analysis_id", "")

        try:
            status = api_client.get_job_status(job_id)
            return process_job_poll_status(status, job_id, analysis_id)
        except Exception:
            return html.Div(), True

    @app.callback(
        Output("cancel-job-feedback-alert", "children"),
        Input("cancel-active-job-btn", "n_clicks"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def cancel_active_job(_clicks: int | None, session_data: dict[str, Any] | None) -> Component:
        job_id = session_data.get("active_job_id") if session_data else None
        return handle_cancel_job(job_id, api_client)


def process_job_poll_status(
    status: dict[str, Any],
    job_id: str,
    analysis_id: str,
) -> tuple[dbc.Card, bool]:
    """Process job status response and determine if polling should terminate."""
    state = status.get("state", "RUNNING")
    stage = status.get("current_stage") or "EXECUTING"
    progress = status.get("progress_percent", 0)
    completed_stages = status.get("completed_stages", 0)
    total_stages = status.get("total_stages", 15)
    err_msg = status.get("error_message")

    monitor_card = _build_monitor_card(
        job_id=job_id,
        analysis_id=analysis_id,
        state=state,
        stage=stage,
        progress_percent=progress,
        completed_stages=completed_stages,
        total_stages=total_stages,
        error_message=err_msg,
    )

    is_terminal = state in ("COMPLETED", "SUCCEEDED", "FAILED", "CANCELLED", "PARTIAL")
    return monitor_card, is_terminal


def handle_cancel_job(job_id: str | None, api_client: AerisApiClient) -> Component:
    """Request cooperative cancellation of a background job."""
    if not job_id:
        return html.Div()
    try:
        api_client.cancel_job(job_id)
        return dbc.Alert(
            "Cooperative cancellation requested. "
            "Worker will terminate cleanly at next stage boundary.",
            color="warning",
            dismissable=True,
        )
    except AerisApiError as err:
        return dbc.Alert(f"Cancellation failed: {err.message}", color="danger")


def _build_monitor_card(
    job_id: str,
    analysis_id: str,
    state: str,
    stage: str,
    progress_percent: int,
    completed_stages: int,
    total_stages: int,
    error_message: str | None,
) -> dbc.Card:
    """Build the live progress monitoring card container."""
    badge_color = "primary"
    if state in ("COMPLETED", "SUCCEEDED"):
        badge_color = "success"
    elif state in ("FAILED", "ERROR"):
        badge_color = "danger"
    elif state == "CANCELLED":
        badge_color = "secondary"
    elif state == "RUNNING":
        badge_color = "info"

    terminal_actions = None
    if state in ("COMPLETED", "SUCCEEDED"):
        terminal_actions = html.Div(
            [
                dbc.Alert(
                    "15-stage analytical pipeline completed successfully! "
                    "Interactive visualizations and findings are ready.",
                    color="success",
                    className="mb-3",
                ),
                dbc.Button(
                    "View Dashboard Plots", href="/dashboard", color="primary", className="me-2"
                ),
                dbc.Button(
                    "Inspect Intelligence & Findings",
                    href="/intelligence",
                    color="info",
                    className="me-2",
                ),
                dbc.Button("Export Engineering Report", href="/reports", color="success"),
            ],
            className="mt-3",
        )
    elif state == "FAILED":
        terminal_actions = html.Div(
            [
                dbc.Alert(
                    f"Pipeline Execution Failed: {error_message or 'Unknown pipeline exception'}",
                    color="danger",
                    className="mb-3",
                ),
            ],
            className="mt-3",
        )
    elif state == "CANCELLED":
        terminal_actions = html.Div(
            [
                dbc.Alert(
                    "Pipeline execution was cancelled cooperatively.",
                    color="secondary",
                    className="mb-3",
                ),
            ],
            className="mt-3",
        )

    cancel_btn = None
    if state in ("QUEUED", "RUNNING"):
        cancel_btn = dbc.Button(
            "Request Cancellation",
            id="cancel-active-job-btn",
            color="danger",
            size="sm",
            outline=True,
            className="ms-auto",
        )

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Span("LIVE PIPELINE MONITOR", className="fw-bold text-muted small"),
                        cancel_btn,
                    ],
                    className="d-flex align-items-center mb-3 pb-2 border-bottom",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Small("Job ID", className="text-muted d-block"),
                                html.Code(job_id, className="fw-semibold"),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Small("Analysis Run ID", className="text-muted d-block"),
                                html.Code(analysis_id, className="fw-semibold"),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Small("Execution State", className="text-muted d-block"),
                                dbc.Badge(state, color=badge_color, className="fs-6"),
                            ],
                            md=4,
                        ),
                    ],
                    className="mb-3",
                ),
                html.Div(
                    [
                        html.Span(
                            f"Current Activity: {stage}",
                            className="fw-semibold text-dark small",
                        ),
                        html.Span(
                            f"Stage {completed_stages} of {total_stages} ({progress_percent}%)",
                            className="text-muted small float-end",
                        ),
                    ],
                    className="mb-1",
                ),
                dbc.Progress(
                    value=progress_percent,
                    striped=state == "RUNNING",
                    animated=state == "RUNNING",
                    color=badge_color,
                    className="mb-3",
                    style={"height": "16px"},
                ),
                html.Div(id="cancel-job-feedback-alert"),
                terminal_actions or html.Div(),
            ]
        ),
        style=CARD_STYLE,
    )
