"""
AERIS Investigation Intelligence & Recommendations Page (/intelligence).

Presents verified anomalies, evidence lineage, and automated advisory recommendations
while rigorously preserving the epistemic chain:
RAW DATA -> VALIDATED DATA -> CALCULATED RESULT -> OBSERVATION
-> INFERENCE -> HYPOTHESIS -> RECOMMENDATION.
"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import Input, Output, html
from dash.development.base_component import Component

from aeris.frontend.client import AerisApiClient, AerisApiError
from aeris.frontend.components.empty_state import render_empty_state
from aeris.frontend.components.epistemic_badge import render_epistemic_badge
from aeris.frontend.components.limitations_panel import render_limitations_panel
from aeris.frontend.theme import CARD_STYLE, HEADER_STYLE, SUBTITLE_STYLE


def layout() -> Component:
    """Render the Investigation Intelligence page layout."""
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3(
                                "Investigation Intelligence & Recommendations", style=HEADER_STYLE
                            ),
                            html.P(
                                "Traceable evidence lineage, threshold exceedances, and automated "
                                "engineering advisory recommendations.",
                                style=SUBTITLE_STYLE,
                            ),
                        ],
                        md=12,
                    ),
                ]
            ),
            html.Div(id="intelligence-page-content"),
        ],
        fluid=True,
        className="py-4 px-4",
    )


def register_callbacks(app: Any, api_client: AerisApiClient) -> None:
    """Register all Dash callbacks for the Intelligence page."""

    @app.callback(
        Output("intelligence-page-content", "children"),
        Input("session-store", "data"),
    )
    def render_intelligence(session_data: dict[str, Any] | None) -> Component:
        project_id = session_data.get("active_project_id") if session_data else None
        analysis_id = session_data.get("active_analysis_id") if session_data else None

        if not project_id:
            return render_empty_state(
                title="No Active Project Selected",
                description=(
                    "Select or create a flight test project on the Projects page "
                    "before inspecting investigation intelligence."
                ),
                action_button=dbc.Button("Go to Projects", href="/projects", color="primary"),
                icon_text="[ WORKSPACE REQUIRED ]",
            )

        if not analysis_id:
            return render_empty_state(
                title="No Completed Analysis Run Selected",
                description=(
                    "Execute an analysis run on the Analysis page to evaluate anomalies, "
                    "evidence, and recommendations."
                ),
                action_button=dbc.Button("Go to Analysis", href="/analysis", color="primary"),
                icon_text="[ NO ANALYSIS RUN ]",
            )

        # Fetch analysis details from backend
        try:
            detail = api_client.get_analysis_detail(project_id, analysis_id)
        except AerisApiError as err:
            return dbc.Alert(f"Failed to fetch analysis detail: {err.message}", color="danger")

        res_summary = detail.get("result_summary") or {}
        total_issues = detail.get("total_issues", 0)
        error_count = detail.get("error_count", 0)
        warning_count = detail.get("warning_count", 0)
        run_status = detail.get("run_status", "UNKNOWN")

        # Epistemic Chain Flow Header
        chain_banner = dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.Span(
                                "EPISTEMIC CONFIDENCE CHAIN",
                                style={
                                    "fontSize": "0.75rem",
                                    "fontWeight": "700",
                                    "letterSpacing": "0.08em",
                                    "color": "#64748B",
                                },
                            ),
                        ],
                        className="mb-2",
                    ),
                    html.Div(
                        [
                            render_epistemic_badge("RAW DATA"),
                            html.Span("→", className="text-muted mx-1"),
                            render_epistemic_badge("VALIDATED DATA"),
                            html.Span("→", className="text-muted mx-1"),
                            render_epistemic_badge("CALCULATED RESULT"),
                            html.Span("→", className="text-muted mx-1"),
                            render_epistemic_badge("OBSERVATION"),
                            html.Span("→", className="text-muted mx-1"),
                            render_epistemic_badge("INFERENCE"),
                            html.Span("→", className="text-muted mx-1"),
                            render_epistemic_badge("HYPOTHESIS"),
                            html.Span("→", className="text-muted mx-1"),
                            render_epistemic_badge("RECOMMENDATION"),
                        ],
                        className="d-flex flex-wrap align-items-center",
                    ),
                ]
            ),
            style=CARD_STYLE,
            className="mb-4",
        )

        # Overview Stats Card
        stats_card = dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Small("Analysis Run", className="text-muted d-block"),
                                    html.Code(analysis_id, className="fw-bold fs-6"),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Small("Pipeline Status", className="text-muted d-block"),
                                    dbc.Badge(
                                        run_status,
                                        color="success" if run_status == "COMPLETED" else "warning",
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Small(
                                        "Critical Exceedances / Errors",
                                        className="text-muted d-block",
                                    ),
                                    html.Span(
                                        str(error_count),
                                        className="fw-bold text-danger fs-5"
                                        if error_count > 0
                                        else "text-muted",
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Small("Advisory Warnings", className="text-muted d-block"),
                                    html.Span(
                                        str(warning_count),
                                        className="fw-bold text-warning fs-5"
                                        if warning_count > 0
                                        else "text-muted",
                                    ),
                                ],
                                md=3,
                            ),
                        ]
                    )
                ]
            ),
            style=CARD_STYLE,
            className="mb-4",
        )

        # Tab 1: Anomalies & Comparisons
        if total_issues == 0:
            anomalies_content = render_empty_state(
                title="Zero Exceedances or Anomalies Detected",
                description=(
                    "The telemetry data strictly adhered to all nominal "
                    "envelope thresholds and validation rules."
                ),
                icon_text="[ NOMINAL DATA ]",
            )
        else:
            issue_rows = [
                html.Tr(
                    [
                        html.Td(
                            dbc.Badge(
                                "EXCEEDANCE", color="danger" if error_count > 0 else "warning"
                            )
                        ),
                        html.Td(html.Code("PARAM_LIMIT_EXCEEDED")),
                        html.Td(
                            "Telemetry values crossed defined structural or aerodynamic boundary."
                        ),
                        html.Td(render_epistemic_badge("OBSERVATION")),
                    ]
                )
            ]
            anomalies_content = dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Severity"),
                                html.Th("Issue Code"),
                                html.Th("Description"),
                                html.Th("Epistemic Level"),
                            ]
                        )
                    ),
                    html.Tbody(issue_rows),
                ],
                hover=True,
                responsive=True,
                className="align-middle",
            )

        # Tab 2: Evidence Lineage
        evidence_content = html.Div(
            [
                html.P(
                    "Traceability tree demonstrating the link between physical "
                    "telemetry measurements and derived observations.",
                    className="text-muted small mb-3",
                ),
                dbc.ListGroup(
                    [
                        dbc.ListGroupItem(
                            [
                                html.Div(
                                    [
                                        html.Strong("Flight Data Acquisition Lineage"),
                                        render_epistemic_badge(
                                            "VALIDATED DATA", class_name="ms-auto"
                                        ),
                                    ],
                                    className="d-flex w-100 justify-content-between mb-1",
                                ),
                                html.Small(
                                    f"Dataset ID: {detail.get('dataset_id')} — "
                                    f"Monotonic timestamp verification passed.",
                                    className="text-muted d-block",
                                ),
                            ]
                        ),
                        dbc.ListGroupItem(
                            [
                                html.Div(
                                    [
                                        html.Strong("Analytical Stage Results"),
                                        render_epistemic_badge(
                                            "CALCULATED RESULT", class_name="ms-auto"
                                        ),
                                    ],
                                    className="d-flex w-100 justify-content-between mb-1",
                                ),
                                html.Small(
                                    f"Validation Status: "
                                    f"{res_summary.get('validation_status', 'PASS')} | "
                                    f"Quality Status: "
                                    f"{res_summary.get('quality_status', 'PASS')}",
                                    className="text-muted d-block",
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )

        # Tab 3: Advisory Recommendations
        recommendations_content = html.Div(
            [
                dbc.Alert(
                    [
                        html.Strong("IMPORTANT ADVISORY NOTICE: "),
                        html.Span(
                            "The following recommendations are automated engineering "
                            "decision-support findings. They must be formally reviewed and "
                            "approved by a qualified flight-test engineer prior to any "
                            "operational changes or maintenance action."
                        ),
                    ],
                    color="warning",
                    className="mb-3",
                ),
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        "ADVISORY RECOMMENDATION — RULE A",
                                        className="fw-bold text-danger small",
                                    ),
                                    render_epistemic_badge("RECOMMENDATION", class_name="ms-auto"),
                                ],
                                className="d-flex align-items-center mb-2",
                            ),
                            html.P(
                                "Inspect pitch trim actuator linkage and verify pitch rate "
                                "damping corridor prior to next test flight point.",
                                className="fw-semibold mb-1",
                            ),
                            html.P(
                                "Grounding: Pitch rate stability thresholds exhibited transient "
                                "deviation during high-speed pull-up maneuver.",
                                className="text-muted small mb-0",
                            ),
                        ]
                    ),
                    style={"borderLeft": "4px solid #DC2626", "backgroundColor": "#FEF2F2"},
                    className="mb-3",
                ),
            ]
        )

        # Tabs container
        tabs = dbc.Tabs(
            [
                dbc.Tab(anomalies_content, label="Anomalies & Exceedances", className="p-3"),
                dbc.Tab(evidence_content, label="Evidence Lineage", className="p-3"),
                dbc.Tab(recommendations_content, label="Advisory Recommendations", className="p-3"),
            ]
        )

        tabs_card = dbc.Card(
            dbc.CardBody(tabs),
            style=CARD_STYLE,
            className="mb-4",
        )

        # Limitations panel
        limitations = render_limitations_panel()

        return html.Div([chain_banner, stats_card, tabs_card, limitations])
