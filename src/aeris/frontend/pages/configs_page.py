"""
AERIS Configurations & Flight Profiles Page (/configurations).

Displays built-in vehicle profiles, aerodynamic limits envelopes, and SOP procedures,
and enables uploading custom YAML configuration profiles for a project.
"""

from __future__ import annotations

import base64
from typing import Any

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, dcc, html, no_update
from dash.development.base_component import Component

from aeris.frontend.client import AerisApiClient, AerisApiError
from aeris.frontend.components.empty_state import render_empty_state
from aeris.frontend.theme import CARD_STYLE, HEADER_STYLE, SUBTITLE_STYLE


def layout() -> Component:
    """Render the Configurations & Flight Profiles page layout."""
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Flight Vehicle Configurations & Profiles", style=HEADER_STYLE),
                            html.P(
                                "Inspect built-in aerodynamic vehicle models, threshold "
                                "limit envelopes, and upload custom YAML profiles.",
                                style=SUBTITLE_STYLE,
                            ),
                        ],
                        md=12,
                    ),
                ]
            ),
            html.Div(id="configs-page-content"),
        ],
        fluid=True,
        className="py-4 px-4",
    )


def render_custom_configs_table(custom_configs: list[dict[str, Any]]) -> Component:
    """Build the custom configs table component or empty state."""
    if not custom_configs:
        return render_empty_state(
            title="No Custom Configurations Attached",
            description=(
                "Upload custom vehicle mass properties, limit envelopes, "
                "or SOP tolerances in YAML format below."
            ),
            icon_text="[ ZERO PROFILES ]",
        )

    c_rows = []
    for cc in custom_configs:
        loaded = str(cc.get("loaded_at", ""))[:19].replace("T", " ")
        c_rows.append(
            html.Tr(
                [
                    html.Td(dbc.Badge(cc.get("profile_type", "VEHICLE"), color="secondary")),
                    html.Td(html.Strong(cc.get("profile_identifier", ""))),
                    html.Td(html.Code(cc.get("config_id", ""), className="text-muted small")),
                    html.Td(loaded, className="text-muted small"),
                ]
            )
        )

    return dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Profile Type"),
                        html.Th("Profile Identifier"),
                        html.Th("Config ID"),
                        html.Th("Loaded Date"),
                    ]
                )
            ),
            html.Tbody(c_rows),
        ],
        hover=True,
        responsive=True,
        className="align-middle",
    )


def register_callbacks(app: Any, api_client: AerisApiClient) -> None:
    """Register all Dash callbacks for the Configurations page."""

    @app.callback(
        Output("configs-page-content", "children"),
        Input("session-store", "data"),
    )
    def render_configs_view(session_data: dict[str, Any] | None) -> Component:
        project_id = session_data.get("active_project_id") if session_data else None
        project_name = session_data.get("active_project_name", project_id) if session_data else None

        # 1. Fetch built-in configs
        try:
            builtin_configs = api_client.get_builtin_configs()
        except AerisApiError as err:
            return dbc.Alert(
                f"Failed to load built-in configurations: {err.message}", color="danger"
            )

        builtin_rows = []
        for c in builtin_configs:
            p_type = c.get("profile_type", "GENERAL")
            badge_color = (
                "primary" if p_type == "VEHICLE" else ("warning" if p_type == "LIMIT" else "info")
            )
            builtin_rows.append(
                html.Tr(
                    [
                        html.Td(dbc.Badge(p_type, color=badge_color, className="me-2")),
                        html.Td(html.Strong(c.get("name", "Default Profile"))),
                        html.Td(html.Code(c.get("profile_id", ""), className="text-muted small")),
                        html.Td(c.get("description", ""), className="text-muted small"),
                    ]
                )
            )

        builtin_card = dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Built-in System Configurations", className="fw-semibold mb-2"),
                    html.P(
                        "Default aerodynamic profiles and baseline parameter validation "
                        "registries available to all projects.",
                        className="text-muted small mb-3",
                    ),
                    dbc.Table(
                        [
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th("Type"),
                                        html.Th("Profile Name"),
                                        html.Th("Identifier"),
                                        html.Th("Description"),
                                    ]
                                )
                            ),
                            html.Tbody(builtin_rows),
                        ],
                        hover=True,
                        responsive=True,
                        className="align-middle",
                    ),
                ]
            ),
            style=CARD_STYLE,
            className="mb-4",
        )

        # 2. Custom Project Configs & Upload Form
        if not project_id:
            custom_section = dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Custom Project Profiles", className="fw-semibold mb-2"),
                        render_empty_state(
                            title="No Active Project Selected",
                            description=(
                                "Select an active project on the Projects page to upload "
                                "custom aircraft limits and vehicle YAMLs."
                            ),
                            icon_text="[ WORKSPACE CONTEXT ]",
                        ),
                    ]
                ),
                style=CARD_STYLE,
            )
            return html.Div([builtin_card, custom_section])

        # Fetch custom configs for project
        try:
            custom_configs = api_client.list_project_configs(project_id)
        except AerisApiError as err:
            return html.Div(
                [
                    builtin_card,
                    dbc.Alert(f"Failed to fetch custom configs: {err.message}", color="danger"),
                ]
            )

        upload_card = dbc.Card(
            dbc.CardBody(
                [
                    html.H5(
                        f"Upload Custom YAML Profile ({project_name})", className="fw-semibold mb-2"
                    ),
                    html.P(
                        "Attach custom aircraft profiles or envelope limits to "
                        "this flight investigation workspace.",
                        className="text-muted small mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Profile Type", className="small fw-semibold"),
                                    dbc.Select(
                                        id="custom-config-type-select",
                                        options=[
                                            {
                                                "label": (
                                                    "Vehicle Envelope / Aerodynamic Model (VEHICLE)"
                                                ),
                                                "value": "VEHICLE",
                                            },
                                            {
                                                "label": "Engineering Limit Envelopes (LIMITS)",
                                                "value": "LIMITS",
                                            },
                                            {
                                                "label": "Standard Operating Procedure (SOP)",
                                                "value": "SOP",
                                            },
                                        ],
                                        value="VEHICLE",
                                        className="mb-3",
                                    ),
                                ],
                                md=6,
                            ),
                        ]
                    ),
                    dcc.Upload(
                        id={"type": "config-upload-target", "index": "action"},
                        children=html.Div(
                            [
                                html.Span("Drag and Drop or ", className="text-muted"),
                                html.A(
                                    "Select YAML Profile (.yaml, .yml)",
                                    className="text-primary fw-semibold",
                                ),
                            ],
                            className="py-3 text-center border border-2 border-dashed rounded",
                            style={"backgroundColor": "#F8FAFC", "cursor": "pointer"},
                        ),
                        accept=".yaml,.yml",
                        multiple=False,
                    ),
                    html.Div(id="config-upload-alert", className="mt-3"),
                ]
            ),
            style=CARD_STYLE,
            className="mb-4",
        )

        custom_card = dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H5(
                                    f"Active Custom Configurations ({project_name})",
                                    className="fw-semibold",
                                ),
                                md=8,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Refresh",
                                    id={"type": "refresh-configs-btn", "index": "action"},
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
                        id="custom-configs-table-container",
                        children=render_custom_configs_table(custom_configs),
                    ),
                ]
            ),
            style=CARD_STYLE,
        )

        return html.Div([builtin_card, upload_card, custom_card])

    @app.callback(
        [
            Output("custom-configs-table-container", "children"),
            Output("config-upload-alert", "children"),
        ],
        [
            Input({"type": "refresh-configs-btn", "index": ALL}, "n_clicks"),
            Input({"type": "config-upload-target", "index": ALL}, "contents"),
        ],
        [
            State({"type": "config-upload-target", "index": ALL}, "filename"),
            State("custom-config-type-select", "value"),
            State("session-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def handle_config_upload_or_refresh(
        _refresh_clicks: Any,
        upload_contents_list: list[str | None],
        upload_filenames_list: list[str | None],
        selected_type: str | None,
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
                prof_type = selected_type or "VEHICLE"
                try:
                    _, content_string = upload_contents.split(",", 1)
                    decoded_bytes = base64.b64decode(content_string)
                    api_client.upload_custom_config(
                        project_id=project_id,
                        profile_type=prof_type,
                        filename=upload_filename,
                        content=decoded_bytes,
                    )
                    upload_alert = dbc.Alert(
                        f"Custom {prof_type} configuration '{upload_filename}' "
                        f"attached to {project_name}.",
                        color="success",
                        dismissable=True,
                    )
                except AerisApiError as err:
                    upload_alert = dbc.Alert(f"Config upload failed: {err.message}", color="danger")
                except Exception as exc:
                    upload_alert = dbc.Alert(f"Config upload error: {exc!s}", color="danger")

        # Fetch custom configs for project
        try:
            custom_configs = api_client.list_project_configs(project_id)
            return render_custom_configs_table(custom_configs), upload_alert
        except AerisApiError as err:
            err_alert = dbc.Alert(f"Failed to fetch custom configs: {err.message}", color="danger")
            return err_alert, upload_alert
