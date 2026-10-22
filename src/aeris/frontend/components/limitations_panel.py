"""
AERIS Analysis Limitations & Technical Caveats Panel.

Presents explicit data quality caveats, sensor limitations, missing reference envelopes,
and flight-test boundaries returned by the analysis pipeline.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from aeris.frontend.theme import CARD_STYLE


def render_limitations_panel(limitations: list[str] | None = None) -> dbc.Card:
    """
    Render an engineering limitations and caveats panel.

    Parameters
    ----------
    limitations : list[str] | None
        List of limitation strings from the backend result summary.
    """
    items = limitations or [
        "Analysis derived strictly from supplied telemetry parameters.",
        (
            "Engineering thresholds reflect nominal envelope limits "
            "unless custom profiles are attached."
        ),
        (
            "Recommendations are automated decision-support advisories "
            "requiring flight-test engineer sign-off."
        ),
    ]

    list_items = [
        html.Li(
            lim,
            style={"fontSize": "0.875rem", "color": "#475569", "marginBottom": "0.35rem"},
        )
        for lim in items
    ]

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Span(
                            "ENGINEERING LIMITATIONS & OPERATIONAL BOUNDARIES",
                            style={
                                "fontSize": "0.75rem",
                                "fontWeight": "700",
                                "letterSpacing": "0.08em",
                                "color": "#B45309",
                            },
                        ),
                    ],
                    className="mb-2",
                ),
                html.Ul(list_items, className="mb-0 ps-3"),
            ],
            style={"backgroundColor": "#FFFBEB", "borderLeft": "4px solid #F59E0B"},
        ),
        style=CARD_STYLE,
    )
