"""
AERIS Engineering Report Generation & Export Hub Page (/reports).

Generates standalone, self-contained HTML and Markdown engineering reports and
provides safe one-click downloads and sandboxed HTML previews.
"""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback_context, html, no_update
from dash.development.base_component import Component

from aeris.frontend.client import AerisApiClient, AerisApiError
from aeris.frontend.components.empty_state import render_empty_state
from aeris.frontend.theme import CARD_STYLE, HEADER_STYLE, SUBTITLE_STYLE


def layout() -> Component:
    """Render the Reports Hub page layout."""
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3(
                                "Engineering Report Generation & Export Hub", style=HEADER_STYLE
                            ),
                            html.P(
                                "Generate self-contained HTML and Markdown test reports with "
                                "embedded Plotly figures and verified SHA-256 checksums.",
                                style=SUBTITLE_STYLE,
                            ),
                        ],
                        md=12,
                    ),
                ]
            ),
            html.Div(id="reports-page-content"),
            html.Div(id="report-preview-container"),
        ],
        fluid=True,
        className="py-4 px-4",
    )


def render_reports_table(
    reports: list[dict[str, Any]],
    base_url: str,
) -> Component:
    """Build the reports table component or empty state."""
    if not reports:
        return render_empty_state(
            title="No Reports Generated Yet",
            description=(
                "Generate an offline HTML or Markdown engineering report using the generator above."
            ),
            icon_text="[ ZERO REPORTS ]",
        )

    table_rows = []
    for r in reports:
        r_id = r.get("report_id", "")
        r_format = r.get("report_format", "HTML")
        size_kb = round(r.get("file_size_bytes", 0) / 1024, 1)
        checksum = r.get("sha256_checksum", "")[:12] + "..."
        gen_date = str(r.get("generated_at", ""))[:19].replace("T", " ")

        download_url = f"{base_url}/reports/{r_id}/download"

        table_rows.append(
            html.Tr(
                [
                    html.Td(
                        [
                            html.Strong(r.get("report_title", "Flight Test Report")),
                            html.Small(f"ID: {r_id}", className="d-block text-muted"),
                        ]
                    ),
                    html.Td(
                        dbc.Badge(
                            r_format,
                            color="success" if r_format == "HTML" else "secondary",
                        )
                    ),
                    html.Td(f"{size_kb} KB"),
                    html.Td(html.Code(checksum, className="small text-muted")),
                    html.Td(gen_date, className="small text-muted"),
                    html.Td(
                        [
                            dbc.Button(
                                "Download",
                                href=download_url,
                                target="_blank",
                                color="primary",
                                size="sm",
                                className="me-2",
                            ),
                            dbc.Button(
                                "Preview",
                                id={"type": "preview-report-btn", "index": r_id},
                                n_clicks=0,
                                color="info",
                                size="sm",
                                outline=True,
                            )
                            if r_format == "HTML"
                            else html.Span(),
                        ]
                    ),
                ]
            )
        )

    return dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Document Title & ID"),
                        html.Th("Format"),
                        html.Th("File Size"),
                        html.Th("SHA-256 Digest"),
                        html.Th("Generated"),
                        html.Th("Actions"),
                    ]
                )
            ),
            html.Tbody(table_rows),
        ],
        hover=True,
        responsive=True,
        className="align-middle",
    )


def register_callbacks(app: Any, api_client: AerisApiClient) -> None:
    """Register all Dash callbacks for the Reports page."""

    @app.callback(
        Output("reports-page-content", "children"),
        Input("session-store", "data"),
    )
    def render_reports_hub(session_data: dict[str, Any] | None) -> Component:
        project_id = session_data.get("active_project_id") if session_data else None
        project_name = session_data.get("active_project_name", project_id) if session_data else None
        analysis_id = session_data.get("active_analysis_id") if session_data else None

        if not project_id:
            return render_empty_state(
                title="No Active Project Selected",
                description=(
                    "Select or create a flight test project on the Projects page "
                    "before managing engineering reports."
                ),
                action_button=dbc.Button("Go to Projects", href="/projects", color="primary"),
                icon_text="[ WORKSPACE REQUIRED ]",
            )

        # Generator Form Card
        gen_card = dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Generate Offline Engineering Report", className="fw-semibold mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Active Analysis Run",
                                        className="fw-semibold small text-muted",
                                    ),
                                    html.Div(
                                        html.Code(
                                            analysis_id or "No Active Run", className="fw-bold fs-6"
                                        ),
                                        className="mb-3",
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Report Format", className="fw-semibold small"),
                                    dbc.Select(
                                        id="report-format-select",
                                        options=[
                                            {
                                                "label": (
                                                    "Self-Contained HTML (Interactive Figures)"
                                                ),
                                                "value": "HTML",
                                            },
                                            {
                                                "label": "Standalone Markdown (GFM Format)",
                                                "value": "MARKDOWN",
                                            },
                                        ],
                                        value="HTML",
                                        className="mb-3",
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Interactive Visualizations",
                                        className="fw-semibold small d-block",
                                    ),
                                    dbc.Checkbox(
                                        id="report-include-plots-check",
                                        label="Embed Plotly Charts",
                                        value=True,
                                        className="mt-2",
                                    ),
                                ],
                                md=4,
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Report Document Title", className="fw-semibold small"
                                    ),
                                    dbc.Input(
                                        id="report-title-input",
                                        value="AERIS Flight Test Investigation Report",
                                        className="mb-3",
                                    ),
                                ],
                                md=8,
                            ),
                            dbc.Col(
                                [
                                    dbc.Button(
                                        "Generate Report",
                                        id={"type": "generate-report-btn", "index": "action"},
                                        color="primary",
                                        className="w-100 mt-4",
                                    ),
                                ],
                                md=4,
                            ),
                        ]
                    ),
                    html.Div(id="report-generate-alert", className="mt-3"),
                ]
            ),
            style=CARD_STYLE,
            className="mb-4",
        )

        # List Existing Reports
        try:
            reports = api_client.list_reports(project_id)
        except AerisApiError as err:
            return html.Div(
                [gen_card, dbc.Alert(f"Failed to fetch reports: {err.message}", color="danger")]
            )

        reports_card = dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H5(
                                    f"Published Engineering Reports ({project_name})",
                                    className="fw-semibold",
                                ),
                                md=8,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Refresh",
                                    id={"type": "refresh-reports-btn", "index": "action"},
                                    color="secondary",
                                    size="sm",
                                    outline=True,
                                ),
                                md=4,
                                className="d-flex justify-content-end",
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        id="reports-table-container",
                        children=render_reports_table(reports, api_client.base_url),
                    ),
                ]
            ),
            style=CARD_STYLE,
            className="mb-4",
        )

        return html.Div([gen_card, reports_card])

    @app.callback(
        [
            Output("reports-table-container", "children"),
            Output("report-generate-alert", "children"),
        ],
        [
            Input({"type": "refresh-reports-btn", "index": ALL}, "n_clicks"),
            Input({"type": "generate-report-btn", "index": ALL}, "n_clicks"),
        ],
        [
            State("report-format-select", "value"),
            State("report-title-input", "value"),
            State("report-include-plots-check", "value"),
            State("session-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def handle_report_generation_or_refresh(
        _refresh_clicks: Any,
        _gen_clicks: Any,
        selected_format: str | None,
        report_title: str | None,
        include_plots: bool | None,
        session_data: dict[str, Any] | None,
    ) -> tuple[Any, Any]:
        has_refresh = any(c for c in (_refresh_clicks or []) if c)
        has_gen = any(c for c in (_gen_clicks or []) if c)
        if not has_refresh and not has_gen:
            return no_update, no_update

        project_id = session_data.get("active_project_id") if session_data else None
        analysis_id = session_data.get("active_analysis_id") if session_data else None

        if not project_id:
            return html.Div(), None

        generate_alert = None
        if has_gen:
            if not analysis_id:
                generate_alert = dbc.Alert(
                    "No completed analysis run selected. "
                    "Run a pipeline on the Analysis page first.",
                    color="warning",
                )
            else:
                try:
                    rep = api_client.generate_report(
                        project_id=project_id,
                        analysis_id=analysis_id,
                        report_format=selected_format or "HTML",
                        report_title=report_title or "AERIS Flight Test Investigation Report",
                        include_interactive_plots=True
                        if include_plots is None
                        else bool(include_plots),
                    )
                    generate_alert = dbc.Alert(
                        f"Report generated successfully! (ID: {rep.get('report_id')}, "
                        f"{round(rep.get('file_size_bytes', 0) / 1024, 1)} KB)",
                        color="success",
                        dismissable=True,
                    )
                except AerisApiError as err:
                    generate_alert = dbc.Alert(
                        f"Report generation failed: {err.message}", color="danger"
                    )

        # Refresh reports list
        try:
            reports = api_client.list_reports(project_id)
            return render_reports_table(reports, api_client.base_url), generate_alert
        except AerisApiError as err:
            err_alert = dbc.Alert(f"Failed to fetch reports: {err.message}", color="danger")
            return err_alert, generate_alert

    @app.callback(
        Output("report-preview-container", "children"),
        Input({"type": "preview-report-btn", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def preview_report(_clicks: Any) -> Any:
        ctx = callback_context
        if not ctx.triggered or not any(c for c in (_clicks or []) if c):
            return no_update

        import json

        trigger_id = ctx.triggered[0]["prop_id"].rsplit(".", 1)[0]
        parsed = json.loads(trigger_id)
        report_id = parsed["index"]

        download_url = f"{api_client.base_url}/reports/{report_id}/download?inline=true"

        return dbc.Card(
            dbc.CardBody(
                [
                    html.H5(f"Sandboxed Report Preview: {report_id}", className="fw-semibold mb-2"),
                    html.P(
                        "Rendered inside an isolated browser sandbox with scripting allowed "
                        "for interactive figures.",
                        className="text-muted small mb-3",
                    ),
                    html.Iframe(
                        src=download_url,
                        style={
                            "width": "100%",
                            "height": "750px",
                            "border": "1px solid #CBD5E1",
                            "borderRadius": "6px",
                        },
                        sandbox="allow-scripts",
                    ),
                ]
            ),
            style=CARD_STYLE,
            className="mt-4",
        )
