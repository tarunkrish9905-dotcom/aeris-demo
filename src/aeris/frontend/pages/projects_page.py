"""
AERIS Projects Hub Page (/projects).

Enables flight test engineers to create, list, select, and manage project workspaces.
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
    """Render the Projects Hub page layout."""
    create_project_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Create Flight Test Project")),
            dbc.ModalBody(
                [
                    dbc.Label("Project Name", html_for="new-project-name", className="fw-semibold"),
                    dbc.Input(
                        id="new-project-name",
                        placeholder="e.g. Flight Test 402 - Flutter Envelope",
                        className="mb-3",
                    ),
                    dbc.Label("Description", html_for="new-project-desc", className="fw-semibold"),
                    dbc.Textarea(
                        id="new-project-desc",
                        placeholder="Investigation objectives, flight conditions, test cards...",
                        rows=3,
                        className="mb-3",
                    ),
                    dbc.Label(
                        "Analyst Name", html_for="new-project-analyst", className="fw-semibold"
                    ),
                    dbc.Input(
                        id="new-project-analyst",
                        placeholder="e.g. Lead Flight Test Engineer",
                        className="mb-3",
                    ),
                    html.Div(id="create-project-error-alert"),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Cancel",
                        id="cancel-create-project-btn",
                        color="secondary",
                        outline=True,
                    ),
                    dbc.Button(
                        "Create Project",
                        id="submit-create-project-btn",
                        color="primary",
                        className="ms-2",
                    ),
                ]
            ),
        ],
        id="create-project-modal",
        is_open=False,
    )

    return dbc.Container(
        [
            create_project_modal,
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Flight Test Projects", style=HEADER_STYLE),
                            html.P(
                                (
                                    "Manage flight investigation workspaces, attached datasets, "
                                    "and test programs."
                                ),
                                style=SUBTITLE_STYLE,
                            ),
                        ],
                        md=8,
                    ),
                    dbc.Col(
                        [
                            dbc.Button(
                                "+ New Project",
                                id="open-create-project-modal-btn",
                                color="primary",
                                className="me-2",
                            ),
                            dbc.Button(
                                "Refresh",
                                id="refresh-projects-btn",
                                color="secondary",
                                outline=True,
                            ),
                        ],
                        md=4,
                        className="d-flex justify-content-md-end align-items-center mb-3",
                    ),
                ],
                className="align-items-center",
            ),
            html.Div(id="project-action-status-alert"),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.Span(
                                    "WORKSPACE REPOSITORY", className="fw-bold text-muted small"
                                ),
                            ],
                            className="mb-3 pb-2 border-bottom",
                        ),
                        html.Div(id="projects-table-container"),
                    ]
                ),
                style=CARD_STYLE,
            ),
        ],
        fluid=True,
        className="py-4 px-4",
    )


def process_project_selection(
    trigger_prop: str | None,
    clicks: list[int | None] | None,
    current_store: dict[str, Any] | None,
    api_client: AerisApiClient,
) -> tuple[Any, Any]:
    """Process project selection event, update store, and return next route."""
    if not trigger_prop or not clicks or not any(c for c in clicks if c):
        return no_update, no_update
    import json

    trigger_id = trigger_prop.split(".")[0]
    store = dict(current_store or {})
    try:
        parsed = json.loads(trigger_id)
        selected_id = parsed["index"]
        store["active_project_id"] = selected_id
        try:
            proj = api_client.get_project(selected_id)
            store["active_project_name"] = proj.get("name", selected_id)
        except Exception:
            store["active_project_name"] = selected_id
        return store, "/datasets"
    except Exception:
        return store, no_update


def register_callbacks(app: Any, api_client: AerisApiClient) -> None:
    """Register all Dash callbacks for the Projects Hub."""

    @app.callback(
        Output("create-project-modal", "is_open"),
        [
            Input("open-create-project-modal-btn", "n_clicks"),
            Input("cancel-create-project-btn", "n_clicks"),
            Input("submit-create-project-btn", "n_clicks"),
        ],
        [State("create-project-modal", "is_open"), State("new-project-name", "value")],
        prevent_initial_call=True,
    )
    def toggle_create_modal(
        open_clicks: int | None,
        cancel_clicks: int | None,
        submit_clicks: int | None,
        is_open: bool,
        name_val: str | None,
    ) -> bool:
        ctx = callback_context
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "open-create-project-modal-btn":
            return True
        if button_id == "cancel-create-project-btn":
            return False
        if button_id == "submit-create-project-btn" and name_val and name_val.strip():
            return False
        return is_open

    @app.callback(
        [
            Output("projects-table-container", "children"),
            Output("project-action-status-alert", "children"),
            Output("create-project-error-alert", "children"),
        ],
        [
            Input("refresh-projects-btn", "n_clicks"),
            Input("submit-create-project-btn", "n_clicks"),
            Input({"type": "delete-project-btn", "index": ALL}, "n_clicks"),
        ],
        [
            State("new-project-name", "value"),
            State("new-project-desc", "value"),
            State("new-project-analyst", "value"),
        ],
    )
    def refresh_and_mutate_projects(
        _refresh: int | None,
        _submit: int | None,
        _deletes: Any,
        new_name: str | None,
        new_desc: str | None,
        new_analyst: str | None,
    ) -> tuple[Component, Component | None, Component | None]:
        ctx = callback_context
        status_alert = None
        modal_error = None

        if ctx.triggered:
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            # Handle project creation
            if trigger_id == "submit-create-project-btn":
                if not new_name or not new_name.strip():
                    modal_error = dbc.Alert("Project name cannot be empty.", color="danger")
                else:
                    try:
                        created = api_client.create_project(
                            name=new_name.strip(),
                            description=(new_desc or "").strip(),
                            analyst_name=new_analyst.strip() if new_analyst else None,
                        )
                        status_alert = dbc.Alert(
                            f"Project '{created.get('name')}' created successfully.",
                            color="success",
                            dismissable=True,
                            duration=4000,
                        )
                    except AerisApiError as err:
                        modal_error = dbc.Alert(f"Error: {err.message}", color="danger")

            # Handle project deletion
            elif "delete-project-btn" in trigger_id:
                try:
                    import json

                    parsed = json.loads(trigger_id)
                    p_id = parsed["index"]
                    api_client.delete_project(p_id, purge=False)
                    status_alert = dbc.Alert(
                        f"Project '{p_id}' marked as DELETED.",
                        color="warning",
                        dismissable=True,
                        duration=4000,
                    )
                except Exception as err:
                    status_alert = dbc.Alert(f"Delete failed: {err!s}", color="danger")

        # Fetch projects list
        try:
            projects = api_client.list_projects()
        except AerisApiError as err:
            return (
                dbc.Alert(f"Failed to load projects: {err.message}", color="danger"),
                status_alert,
                modal_error,
            )

        if not projects:
            empty = render_empty_state(
                title="No Flight Projects Found",
                description=(
                    "Create a project to initialize an isolated workspace directory "
                    "and begin telemetry analysis."
                ),
                icon_text="[ EMPTY WORKSPACE ]",
            )
            return empty, status_alert, modal_error

        # Build projects table
        table_rows = []
        for p in projects:
            p_id = p.get("project_id", "")
            created_date = str(p.get("created_at", ""))[:19].replace("T", " ")
            table_rows.append(
                html.Tr(
                    [
                        html.Td(
                            [
                                html.Div(
                                    p.get("name", "Untitled"), className="fw-semibold text-dark"
                                ),
                                html.Small(
                                    p.get("description", ""), className="text-muted d-block"
                                ),
                            ]
                        ),
                        html.Td(html.Code(p_id, className="text-muted small")),
                        html.Td(p.get("analyst_name") or "-"),
                        html.Td(
                            dbc.Badge(
                                f"{p.get('dataset_count', 0)} Datasets",
                                color="info",
                                className="me-1",
                            )
                        ),
                        html.Td(
                            dbc.Badge(
                                f"{p.get('analysis_count', 0)} Runs",
                                color="secondary",
                            )
                        ),
                        html.Td(created_date, className="text-muted small"),
                        html.Td(
                            [
                                dbc.Button(
                                    "Open",
                                    id={"type": "select-project-btn", "index": p_id},
                                    color="primary",
                                    size="sm",
                                    className="me-2",
                                ),
                                dbc.Button(
                                    "Delete",
                                    id={"type": "delete-project-btn", "index": p_id},
                                    color="danger",
                                    size="sm",
                                    outline=True,
                                ),
                            ]
                        ),
                    ]
                )
            )

        table = dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Project Name & Scope"),
                            html.Th("Project ID"),
                            html.Th("Analyst"),
                            html.Th("Datasets"),
                            html.Th("Analyses"),
                            html.Th("Created"),
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

        return table, status_alert, modal_error

    @app.callback(
        [
            Output("session-store", "data"),
            Output("url", "pathname"),
        ],
        Input({"type": "select-project-btn", "index": ALL}, "n_clicks"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def select_active_project(
        _clicks: Any, current_store: dict[str, Any] | None
    ) -> tuple[Any, Any]:
        ctx = callback_context
        try:
            if not ctx.triggered:
                return no_update, no_update
            trigger_prop = ctx.triggered[0]["prop_id"]
        except Exception:
            return no_update, no_update
        return process_project_selection(trigger_prop, _clicks, current_store, api_client)
