"""
AERIS Datasets & Mapping Wizard Page (/datasets).

Provides flight telemetry ingestion, tabular data preview, and semantic
parameter mapping configuration.
"""

from __future__ import annotations

import base64
from typing import Any

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback_context, dcc, html, no_update
from dash.development.base_component import Component

from aeris.core.registry.canonical import infer_unit_from_column_name
from aeris.frontend.client import AerisApiClient, AerisApiError
from aeris.frontend.components.empty_state import render_empty_state
from aeris.frontend.theme import CARD_STYLE, HEADER_STYLE, SUBTITLE_STYLE

CANONICAL_TARGET_PARAMETERS = [
    ("timestamp", "Timestamp (Primary Time)", True, "s"),
    ("indicated_airspeed", "Indicated Airspeed (IAS)", False, "knots"),
    ("true_airspeed", "True Airspeed (TAS)", False, "knots"),
    ("pressure_altitude", "Pressure Altitude", False, "ft"),
    ("pitch_attitude", "Pitch Attitude", False, "deg"),
    ("roll_attitude", "Roll Attitude", False, "deg"),
    ("yaw_attitude", "Yaw / Heading Attitude", False, "deg"),
    ("pitch_rate", "Pitch Angular Rate (q)", False, "deg/s"),
    ("roll_rate", "Roll Angular Rate (p)", False, "deg/s"),
    ("yaw_rate", "Yaw Angular Rate (r)", False, "deg/s"),
    ("angle_of_attack", "Angle of Attack (alpha)", False, "deg"),
    ("sideslip_angle", "Sideslip Angle (beta)", False, "deg"),
    ("vertical_acceleration", "Normal Load Factor (Nz)", False, "g"),
    ("dynamic_pressure", "Dynamic Pressure (qbar)", False, "psf"),
    ("mach_number", "Mach Number (M)", False, "mach"),
    ("engine_n1", "Engine Fan Speed (N1)", False, "%"),
]

CANONICAL_TARGET_UNITS: dict[str, str] = {item[0]: item[3] for item in CANONICAL_TARGET_PARAMETERS}


def layout() -> Component:
    """Render the Datasets & Mapping Wizard page layout."""
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Telemetry Ingestion & Parameter Mapping", style=HEADER_STYLE),
                            html.P(
                                "Ingest raw flight CSV telemetry, inspect preview rows, "
                                "and configure semantic parameter bindings.",
                                style=SUBTITLE_STYLE,
                            ),
                        ],
                        md=12,
                    ),
                ]
            ),
            html.Div(id="datasets-page-content"),
        ],
        fluid=True,
        className="py-4 px-4",
    )


def render_datasets_table(
    datasets: list[dict[str, Any]],
    project_name: str | None,
) -> Component:
    """Build the dataset table component or empty state."""
    if not datasets:
        return render_empty_state(
            title="No Datasets Ingested",
            description=(
                "Upload a CSV telemetry log above to inspect rows and configure parameter mappings."
            ),
            icon_text="[ ZERO DATASETS ]",
        )

    rows = []
    for d in datasets:
        d_id = d.get("dataset_id", "")
        size_kb = round(d.get("file_size_bytes", 0) / 1024, 1)
        ingested = str(d.get("ingested_at", ""))[:19].replace("T", " ")
        rows.append(
            html.Tr(
                [
                    html.Td(html.Strong(d.get("original_filename", "telemetry.csv"))),
                    html.Td(html.Code(d_id, className="small text-muted")),
                    html.Td(f"{size_kb} KB"),
                    html.Td(f"{d.get('row_count', 0):,}"),
                    html.Td(f"{d.get('column_count', 0)}"),
                    html.Td(ingested, className="small text-muted"),
                    html.Td(
                        dbc.Button(
                            "Inspect & Map",
                            id={"type": "inspect-dataset-btn", "index": d_id},
                            color="primary",
                            size="sm",
                        )
                    ),
                ]
            )
        )

    return dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Filename"),
                        html.Th("Dataset ID"),
                        html.Th("File Size"),
                        html.Th("Rows"),
                        html.Th("Columns"),
                        html.Th("Ingested"),
                        html.Th("Action"),
                    ]
                )
            ),
            html.Tbody(rows),
        ],
        hover=True,
        responsive=True,
        className="align-middle",
    )


def register_callbacks(app: Any, api_client: AerisApiClient) -> None:
    """Register all Dash callbacks for the Datasets & Mapping page."""

    @app.callback(
        Output("datasets-page-content", "children"),
        Input("session-store", "data"),
    )
    def render_datasets_view(session_data: dict[str, Any] | None) -> Component:
        project_id = session_data.get("active_project_id") if session_data else None
        project_name = session_data.get("active_project_name", project_id) if session_data else None

        if not project_id:
            return render_empty_state(
                title="No Active Project Selected",
                description=(
                    "Please select or create a flight test project on the Projects page "
                    "before uploading telemetry."
                ),
                action_button=dbc.Button("Go to Projects", href="/projects", color="primary"),
                icon_text="[ WORKSPACE REQUIRED ]",
            )

        # Fetch datasets for project
        try:
            datasets = api_client.list_datasets(project_id)
        except AerisApiError as err:
            return dbc.Alert(f"Failed to fetch datasets: {err.message}", color="danger")

        # Ingestion Upload Zone
        upload_zone = dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Upload Telemetry File", className="fw-semibold mb-2"),
                    html.P(
                        "Drag and drop flight test CSV files (max 200MB) "
                        "or browse your filesystem.",
                        className="text-muted small mb-3",
                    ),
                    dcc.Upload(
                        id={"type": "dataset-upload-target", "index": "action"},
                        children=html.Div(
                            [
                                html.Span("Drag and Drop or ", className="text-muted"),
                                html.A("Select CSV File", className="text-primary fw-semibold"),
                            ],
                            className="py-4 text-center border border-2 border-dashed rounded",
                            style={"backgroundColor": "#F8FAFC", "cursor": "pointer"},
                        ),
                        accept=".csv",
                        multiple=False,
                    ),
                    html.Div(id="dataset-upload-alert", className="mt-3"),
                ]
            ),
            style=CARD_STYLE,
            className="mb-4",
        )

        # Datasets Section
        datasets_section = dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H5(
                                    f"Project Datasets ({project_name})",
                                    className="fw-semibold",
                                ),
                                md=8,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Refresh List",
                                    id={"type": "refresh-datasets-btn", "index": "action"},
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
                        id="datasets-table-container",
                        children=render_datasets_table(datasets, project_name),
                    ),
                ]
            ),
            style=CARD_STYLE,
        )

        return html.Div(
            [
                upload_zone,
                datasets_section,
                html.Div(id="dataset-inspector-container"),
            ]
        )

    @app.callback(
        [
            Output("datasets-table-container", "children"),
            Output("dataset-upload-alert", "children"),
        ],
        [
            Input({"type": "refresh-datasets-btn", "index": ALL}, "n_clicks"),
            Input({"type": "dataset-upload-target", "index": ALL}, "contents"),
        ],
        [
            State({"type": "dataset-upload-target", "index": ALL}, "filename"),
            State("session-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def handle_dataset_upload_or_refresh(
        _refresh_clicks: Any,
        upload_contents_list: list[str | None],
        upload_filenames_list: list[str | None],
        session_data: dict[str, Any] | None,
    ) -> tuple[Any, Any]:
        has_refresh = any(c for c in (_refresh_clicks or []) if c)
        has_upload = any(c for c in (upload_contents_list or []) if c)
        if not has_refresh and not has_upload:
            return no_update, no_update

        project_id = session_data.get("active_project_id") if session_data else None
        project_name = session_data.get("active_project_name", project_id) if session_data else None

        if not project_id:
            return html.Div(), None

        upload_alert = None
        if has_upload and upload_contents_list and upload_filenames_list:
            upload_contents = upload_contents_list[0]
            upload_filename = upload_filenames_list[0]
            if upload_contents and upload_filename:
                try:
                    # Decode base64 payload from dcc.Upload
                    _, content_string = upload_contents.split(",", 1)
                    decoded_bytes = base64.b64decode(content_string)

                    # Validate 200MB ceiling
                    if len(decoded_bytes) > 209_715_200:
                        upload_alert = dbc.Alert(
                            "File exceeds 200 MB maximum upload limit.",
                            color="danger",
                            dismissable=True,
                        )
                    else:
                        resp = api_client.upload_dataset(
                            project_id=project_id,
                            filename=upload_filename,
                            content=decoded_bytes,
                        )
                        upload_alert = dbc.Alert(
                            f"Uploaded '{upload_filename}' successfully "
                            f"(ID: {resp.get('dataset_id')}, {resp.get('row_count')} rows).",
                            color="success",
                            dismissable=True,
                        )
                except AerisApiError as err:
                    upload_alert = dbc.Alert(f"Upload failed: {err.message}", color="danger")
                except Exception as exc:
                    upload_alert = dbc.Alert(f"Upload error: {exc!s}", color="danger")

        # Refresh dataset list
        try:
            datasets = api_client.list_datasets(project_id)
            return render_datasets_table(datasets, project_name), upload_alert
        except AerisApiError as err:
            err_alert = dbc.Alert(f"Failed to fetch datasets: {err.message}", color="danger")
            return err_alert, upload_alert

    @app.callback(
        Output("dataset-inspector-container", "children"),
        Input({"type": "inspect-dataset-btn", "index": ALL}, "n_clicks"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def inspect_dataset(_clicks: Any, session_data: dict[str, Any] | None) -> Component:
        ctx = callback_context
        if not ctx.triggered or not session_data:
            return html.Div()

        import json

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        parsed = json.loads(trigger_id)
        dataset_id = parsed["index"]
        project_id = session_data.get("active_project_id", "")

        # Fetch preview and mappings
        try:
            preview = api_client.get_dataset_preview(project_id, dataset_id, rows=15)
            existing_mappings = api_client.get_parameter_mappings(project_id, dataset_id)
        except AerisApiError as err:
            return dbc.Alert(f"Failed to inspect dataset: {err.message}", color="danger")

        # 1. Preview Table
        columns = preview.get("columns", [])
        preview_rows = preview.get("preview_rows", [])

        th_elements = [html.Th(col) for col in columns[:10]]
        if len(columns) > 10:
            th_elements.append(html.Th(f"... (+{len(columns) - 10} more)"))

        tr_elements = []
        for row in preview_rows:
            tds = [html.Td(str(row.get(col, ""))) for col in columns[:10]]
            if len(columns) > 10:
                tds.append(html.Td("...", className="text-muted"))
            tr_elements.append(html.Tr(tds))

        preview_table = dbc.Table(
            [
                html.Thead(html.Tr(th_elements)),
                html.Tbody(tr_elements),
            ],
            bordered=True,
            responsive=True,
            size="sm",
            className="mb-3 small",
        )

        # 2. Mapping Configuration Rows
        mapped_dict = {m.get("raw_column_name"): m for m in existing_mappings}
        mapping_rows = []

        dropdown_options = [{"label": "-- Pass Through / Ignore --", "value": ""}] + [
            {"label": f"{label} [{param_id}]", "value": param_id}
            for param_id, label, _, _ in CANONICAL_TARGET_PARAMETERS
        ]

        for col in columns:
            existing = mapped_dict.get(col, {})
            current_target = existing.get("target_parameter_id", "")
            current_is_time = existing.get(
                "is_timestamp", col.lower() in ("time", "timestamp", "utc")
            )

            mapping_rows.append(
                html.Tr(
                    [
                        html.Td(html.Code(col, className="fw-semibold")),
                        html.Td(
                            dbc.Select(
                                id={"type": "map-target-select", "col": col},
                                options=dropdown_options,
                                value=current_target,
                            )
                        ),
                        html.Td(
                            dbc.Checkbox(
                                id={"type": "map-istime-check", "col": col},
                                value=bool(current_is_time),
                            )
                        ),
                    ]
                )
            )

        mappings_table = dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Raw Telemetry Column"),
                            html.Th("Target Semantic Parameter"),
                            html.Th("Is Primary Time?"),
                        ]
                    )
                ),
                html.Tbody(mapping_rows),
            ],
            hover=True,
            responsive=True,
            size="sm",
            className="align-middle",
        )

        return dbc.Card(
            dbc.CardBody(
                [
                    dcc.Store(id="active-inspected-dataset-id", data=dataset_id),
                    html.H5(
                        f"Dataset Inspection & Telemetry Preview: {dataset_id}",
                        className="fw-semibold mb-2",
                    ),
                    html.P(
                        f"Showing first {len(preview_rows)} rows out of "
                        f"{preview.get('total_rows', 0):,} total rows.",
                        className="text-muted small mb-3",
                    ),
                    preview_table,
                    html.Hr(),
                    html.H5("Guided Parameter Mapping Wizard", className="fw-semibold mb-2"),
                    html.P(
                        "Bind raw telemetry columns to standardized AERIS aerodynamic "
                        "and flight-dynamics parameters.",
                        className="text-muted small mb-3",
                    ),
                    mappings_table,
                    html.Div(id="save-mappings-status-alert", className="mt-3"),
                    dbc.Button(
                        "Save Parameter Mappings",
                        id={"type": "save-parameter-mappings-btn", "index": "action"},
                        color="success",
                        className="mt-2",
                    ),
                ]
            ),
            style=CARD_STYLE,
            className="mt-4",
        )

    @app.callback(
        Output("save-mappings-status-alert", "children"),
        Input({"type": "save-parameter-mappings-btn", "index": ALL}, "n_clicks"),
        [
            State("active-inspected-dataset-id", "data"),
            State("session-store", "data"),
            State({"type": "map-target-select", "col": ALL}, "value"),
            State({"type": "map-target-select", "col": ALL}, "id"),
            State({"type": "map-istime-check", "col": ALL}, "value"),
        ],
        prevent_initial_call=True,
    )
    def save_mappings(
        _clicks: Any,
        dataset_id: str | None,
        session_data: dict[str, Any] | None,
        select_values: list[str | None],
        select_ids: list[dict[str, str]],
        istime_values: list[bool | None],
    ) -> Any:
        if not _clicks or not any(c for c in _clicks if c):
            return no_update
        if not dataset_id or not session_data:
            return dbc.Alert("Dataset or active project not found.", color="danger")

        project_id = session_data.get("active_project_id", "")
        mappings = []

        for val, sel_id, is_time in zip(select_values, select_ids, istime_values, strict=False):
            if val and val.strip():
                target_id = val.strip()
                raw_col = sel_id["col"]
                target_u = CANONICAL_TARGET_UNITS.get(target_id)
                source_u = infer_unit_from_column_name(raw_col) or target_u
                mappings.append(
                    {
                        "raw_column_name": raw_col,
                        "target_parameter_id": target_id,
                        "source_unit": source_u,
                        "target_unit": target_u,
                        "data_type": "float",
                        "is_timestamp": bool(is_time),
                    }
                )

        if not mappings:
            return dbc.Alert(
                "No parameter mappings specified. Select at least one target parameter.",
                color="warning",
            )

        try:
            api_client.save_parameter_mappings(project_id, dataset_id, mappings)
            return dbc.Alert(
                f"Successfully saved {len(mappings)} verified parameter mappings.",
                color="success",
                dismissable=True,
            )
        except AerisApiError as err:
            return dbc.Alert(f"Failed to save parameter mappings: {err.message}", color="danger")
