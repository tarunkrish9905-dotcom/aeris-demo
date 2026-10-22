"""
AERIS Plotly Dash Application Factory & Multi-Page Router.

Constructs and configures the interactive engineering portal, session store,
global navigation, and multi-page URL routing.
"""

from __future__ import annotations

from typing import Any

import dash
import dash_bootstrap_components as dbc
from a2wsgi import WSGIMiddleware
from dash import Input, Output, dcc, html
from dash.development.base_component import Component
from fastapi import FastAPI

from aeris.frontend.client import AerisApiClient
from aeris.frontend.components.empty_state import render_empty_state
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
from aeris.frontend.theme import BG_LIGHT, FONT_FAMILY


def route_page_view(pathname: str | None) -> Component:
    """Route a URL path to the corresponding AERIS page layout."""
    clean_path = (pathname or "/").rstrip("/") or "/"

    if clean_path in ("/", "/projects"):
        return projects_page.layout()
    if clean_path == "/datasets":
        return datasets_page.layout()
    if clean_path == "/configurations":
        return configs_page.layout()
    if clean_path == "/analysis":
        return analysis_page.layout()
    if clean_path == "/dashboard":
        return dashboard_page.layout()
    if clean_path == "/intelligence":
        return intelligence_page.layout()
    if clean_path == "/reports":
        return reports_page.layout()

    return render_empty_state(
        title="Page Not Found (404)",
        description=f"The requested path '{clean_path}' does not exist in the AERIS portal.",
        action_button=dbc.Button("Return to Projects", href="/projects", color="primary"),
        icon_text="[ 404 NOT FOUND ]",
    )


def format_active_project_label(session_data: dict[str, Any] | None) -> str:
    """Format active project identifier and name for top navigation display."""
    if not session_data or not session_data.get("active_project_id"):
        return "None Selected"
    name = session_data.get("active_project_name")
    p_id = session_data["active_project_id"]
    return f"{name} ({p_id})" if name and name != p_id else str(p_id)


def evaluate_system_health_display(health: dict[str, Any] | None) -> tuple[dict[str, str], str]:
    """Determine navigation bar pill color and label based on backend health status."""
    if health and health.get("status") == "healthy" and health.get("database_connected", True):
        return {"color": "#10B981", "marginRight": "0.35rem"}, "SYSTEM ONLINE"
    if health:
        return {"color": "#F59E0B", "marginRight": "0.35rem"}, "DEGRADED"
    return {"color": "#EF4444", "marginRight": "0.35rem"}, "OFFLINE"


def create_dash_app(
    api_client: AerisApiClient | None = None,
    requests_pathname_prefix: str = "/",
) -> dash.Dash:
    """
    Construct the AERIS Plotly Dash application.

    Parameters
    ----------
    api_client : AerisApiClient | None
        Synchronous client for communicating with the backend REST API.
        If None, default client targeting 127.0.0.1:8000/api/v1 is created.
    requests_pathname_prefix : str
        URL prefix for Dash routing (defaults to '/').

    Returns
    -------
    dash.Dash
        Configured Dash application with all multi-page callbacks registered.
    """
    client = api_client or AerisApiClient()

    dash_app = dash.Dash(
        __name__,
        requests_pathname_prefix=requests_pathname_prefix,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
        ],
        suppress_callback_exceptions=True,
        title="AERIS — Flight Test Investigation Portal",
    )

    # Global multi-page layout wrapper
    dash_app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="session-store", storage_type="session", data={}),
            dcc.Interval(id="health-poll-interval", interval=10000, n_intervals=0),
            render_navbar(),
            html.Div(
                id="page-content",
                style={"backgroundColor": BG_LIGHT, "minHeight": "calc(100vh - 60px)"},
            ),
        ],
        style={"fontFamily": FONT_FAMILY},
    )

    # 1. Multi-Page URL Router
    @dash_app.callback(
        Output("page-content", "children"),
        Input("url", "pathname"),
    )
    def route_page(pathname: str | None) -> Component:
        return route_page_view(pathname)

    # 2. Navbar Active Project Sync
    @dash_app.callback(
        Output("active-project-name-display", "children"),
        Input("session-store", "data"),
    )
    def sync_active_project(session_data: dict[str, Any] | None) -> str:
        return format_active_project_label(session_data)

    # 3. Navbar Backend Health Pill
    @dash_app.callback(
        [
            Output("system-status-indicator", "style"),
            Output("system-status-text", "children"),
        ],
        Input("health-poll-interval", "n_intervals"),
    )
    def check_system_health(_intervals: int) -> tuple[dict[str, str], str]:
        try:
            health = client.get_health()
            return evaluate_system_health_display(health)
        except Exception:
            return evaluate_system_health_display(None)

    # Register individual page callbacks
    projects_page.register_callbacks(dash_app, client)
    datasets_page.register_callbacks(dash_app, client)
    configs_page.register_callbacks(dash_app, client)
    analysis_page.register_callbacks(dash_app, client)
    dashboard_page.register_callbacks(dash_app, client)
    intelligence_page.register_callbacks(dash_app, client)
    reports_page.register_callbacks(dash_app, client)

    return dash_app


def mount_dash_app(
    fastapi_app: FastAPI,
    dash_app: dash.Dash | None = None,
    api_client: AerisApiClient | None = None,
) -> dash.Dash:
    """
    Mount the Plotly Dash WSGI application onto a FastAPI ASGI application
    using a2wsgi.WSGIMiddleware.

    Parameters
    ----------
    fastapi_app : FastAPI
        Target FastAPI application instance.
    dash_app : dash.Dash | None
        Optional pre-configured Dash instance. If None, create_dash_app() is called.
    api_client : AerisApiClient | None
        Optional client instance to use for API communication.

    Returns
    -------
    dash.Dash
        The mounted Dash instance.
    """
    d_app = dash_app or create_dash_app(api_client=api_client)
    fastapi_app.mount("/", WSGIMiddleware(d_app.server))  # type: ignore[arg-type]
    return d_app
