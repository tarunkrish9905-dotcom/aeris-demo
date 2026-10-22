"""
AERIS Top Navigation Bar Component.

Provides consistent global navigation, active project context display, and
real-time backend system health indicator across all multi-page views.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from aeris.frontend.theme import PRIMARY_NAVY


def render_navbar() -> dbc.Navbar:
    """Construct the responsive top navigation bar for the AERIS portal."""
    brand = html.A(
        dbc.Row(
            [
                dbc.Col(
                    html.Span(
                        "AERIS",
                        style={
                            "fontWeight": "800",
                            "fontSize": "1.35rem",
                            "letterSpacing": "0.08em",
                            "color": "#FFFFFF",
                        },
                    )
                ),
                dbc.Col(
                    dbc.Badge(
                        "FLIGHT TELEMETRY",
                        style={
                            "fontSize": "0.65rem",
                            "backgroundColor": "#0284C7",
                            "marginLeft": "0.5rem",
                            "letterSpacing": "0.05em",
                        },
                    )
                ),
            ],
            align="center",
            className="g-0",
        ),
        href="/projects",
        style={"textDecoration": "none"},
    )

    nav_links = dbc.Nav(
        [
            dbc.NavLink("Projects", href="/projects", active="exact"),
            dbc.NavLink("Datasets & Mapping", href="/datasets", active="exact"),
            dbc.NavLink("Configurations", href="/configurations", active="exact"),
            dbc.NavLink("Analysis", href="/analysis", active="exact"),
            dbc.NavLink("Dashboard", href="/dashboard", active="exact"),
            dbc.NavLink("Intelligence", href="/intelligence", active="exact"),
            dbc.NavLink("Reports", href="/reports", active="exact"),
        ],
        className="ms-auto",
        navbar=True,
    )

    active_project_badge = html.Div(
        [
            html.Span("ACTIVE PROJECT: ", style={"fontSize": "0.75rem", "color": "#94A3B8"}),
            html.Span(
                "None Selected",
                id="active-project-name-display",
                style={
                    "fontSize": "0.8rem",
                    "fontWeight": "600",
                    "color": "#38BDF8",
                    "fontFamily": "monospace",
                },
            ),
        ],
        className="d-flex align-items-center me-3 px-2 py-1",
        style={"backgroundColor": "#1E293B", "borderRadius": "4px"},
    )

    system_status_pill = html.Div(
        [
            html.Span(
                "●",
                id="system-status-indicator",
                style={"color": "#10B981", "marginRight": "0.35rem"},
            ),
            html.Span(
                "SYSTEM READY",
                id="system-status-text",
                style={"fontSize": "0.75rem", "fontWeight": "600", "color": "#E2E8F0"},
            ),
        ],
        id="system-health-pill",
        className="d-flex align-items-center px-2 py-1",
        style={
            "backgroundColor": "#1E293B",
            "borderRadius": "4px",
            "border": "1px solid #334155",
        },
    )

    return dbc.Navbar(
        dbc.Container(
            [
                brand,
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                dbc.Collapse(
                    [
                        nav_links,
                        html.Div(
                            [
                                active_project_badge,
                                system_status_pill,
                            ],
                            className="d-flex align-items-center ms-lg-3 mt-2 mt-lg-0",
                        ),
                    ],
                    id="navbar-collapse",
                    navbar=True,
                ),
            ],
            fluid=True,
            className="px-4",
        ),
        color=PRIMARY_NAVY,
        dark=True,
        sticky="top",
        className="border-bottom border-secondary shadow-sm",
    )
