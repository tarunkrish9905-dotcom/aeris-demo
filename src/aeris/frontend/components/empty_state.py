"""
AERIS Honest Empty State Component.

Presents unambiguous, helpful empty state messages when entities (projects, datasets,
analyses, or reports) do not yet exist, preventing user confusion and avoiding fake data.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html
from dash.development.base_component import Component

from aeris.frontend.theme import CARD_STYLE


def render_empty_state(
    title: str,
    description: str,
    action_button: Component | None = None,
    icon_text: str = "[ No Data ]",
) -> dbc.Card:
    """
    Render an honest empty-state card container.

    Parameters
    ----------
    title : str
        Primary message indicating what is absent.
    description : str
        Guidance explaining why data is absent and what action to take next.
    action_button : Component | None
        Optional button or link enabling the user to resolve the empty state.
    icon_text : str
        Text glyph or badge label.
    """
    children: list[Component] = [
        html.Div(
            icon_text,
            style={
                "fontSize": "0.9rem",
                "fontWeight": "600",
                "color": "#94A3B8",
                "letterSpacing": "0.1em",
                "marginBottom": "0.5rem",
                "fontFamily": "monospace",
            },
        ),
        html.H5(title, className="mb-2", style={"fontWeight": "600", "color": "#1E293B"}),
        html.P(
            description,
            className="text-muted mb-3",
            style={"maxWidth": "540px", "margin": "0 auto", "fontSize": "0.925rem"},
        ),
    ]

    if action_button is not None:
        children.append(html.Div(action_button, className="mt-2"))

    return dbc.Card(
        dbc.CardBody(
            children,
            className="text-center py-5",
        ),
        style=CARD_STYLE,
    )
