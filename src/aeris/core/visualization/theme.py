"""
AERIS Plotly Visualization Theme & Styling.

Provides a consistent, publication-grade engineering aesthetic for all Plotly
figures generated within the AERIS core platform.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Standard AERIS Engineering Palette
# ---------------------------------------------------------------------------

AERIS_PALETTE: dict[str, str] = {
    "primary": "#1D4ED8",  # Standard Telemetry Blue
    "secondary": "#0D9488",  # Teal (Calculated signals)
    "accent": "#7C3AED",  # Purple (Target / Model references)
    "caution": "#D97706",  # Amber / Caution limit
    "warning": "#EA580C",  # Orange-Red / Warning limit
    "critical": "#DC2626",  # Crimson / Critical exceedance
    "neutral": "#64748B",  # Slate (Grid, axis notes)
    "grid_light": "#E2E8F0",
    "grid_dark": "#334155",
    "bg_light": "#FFFFFF",
    "paper_light": "#F8FAFC",
    "text_light": "#0F172A",
    "bg_dark": "#0F172A",
    "paper_dark": "#1E293B",
    "text_dark": "#F8FAFC",
}

FONT_FAMILY: str = (
    "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
)


def get_aeris_layout(
    title: str | None = None,
    x_title: str = "Time (UTC)",
    y_title: str | None = None,
    height: int = 450,
    width: int | None = None,
    show_grid: bool = True,
    dark_mode: bool = False,
) -> dict[str, Any]:
    """
    Construct a standard AERIS engineering layout dictionary for Plotly figures.

    Parameters
    ----------
    title : str | None
        Figure title string.
    x_title : str
        X-axis label (default "Time (UTC)").
    y_title : str | None
        Y-axis label with unit where available.
    height : int
        Figure height in pixels.
    width : int | None
        Figure width in pixels (None for responsive).
    show_grid : bool
        Whether to display background grid lines.
    dark_mode : bool
        Whether to render in dark theme.

    Returns
    -------
    dict[str, Any]
        Plotly layout dictionary.
    """
    bg_color = AERIS_PALETTE["bg_dark"] if dark_mode else AERIS_PALETTE["bg_light"]
    paper_color = AERIS_PALETTE["paper_dark"] if dark_mode else AERIS_PALETTE["paper_light"]
    text_color = AERIS_PALETTE["text_dark"] if dark_mode else AERIS_PALETTE["text_light"]
    grid_color = AERIS_PALETTE["grid_dark"] if dark_mode else AERIS_PALETTE["grid_light"]

    layout: dict[str, Any] = {
        "font": {
            "family": FONT_FAMILY,
            "size": 12,
            "color": text_color,
        },
        "plot_bgcolor": bg_color,
        "paper_bgcolor": paper_color,
        "height": height,
        "margin": {"l": 65, "r": 35, "t": 60, "b": 50},
        "hovermode": "closest",
        "xaxis": {
            "title": {"text": x_title, "font": {"size": 12}},
            "showgrid": show_grid,
            "gridcolor": grid_color,
            "gridwidth": 1,
            "zeroline": False,
            "linecolor": grid_color,
            "linewidth": 1,
            "ticks": "outside",
            "tickfont": {"size": 11, "color": text_color},
        },
        "yaxis": {
            "title": {"text": y_title or "", "font": {"size": 12}},
            "showgrid": show_grid,
            "gridcolor": grid_color,
            "gridwidth": 1,
            "zeroline": True,
            "zerolinecolor": grid_color,
            "zerolinewidth": 1,
            "linecolor": grid_color,
            "linewidth": 1,
            "ticks": "outside",
            "tickfont": {"size": 11, "color": text_color},
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
    }

    if width is not None:
        layout["width"] = width

    if title:
        layout["title"] = {
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 14, "color": text_color},
        }

    return layout


def apply_aeris_theme(
    fig: go.Figure,
    dark_mode: bool = False,
    show_grid: bool = True,
) -> go.Figure:
    """
    Apply standard AERIS styling to an existing Plotly Figure in-place and return it.

    Parameters
    ----------
    fig : go.Figure
        Target Plotly figure.
    dark_mode : bool
        Whether to style for dark mode.
    show_grid : bool
        Whether grid lines are visible.

    Returns
    -------
    go.Figure
        The updated Plotly figure.
    """
    bg_color = AERIS_PALETTE["bg_dark"] if dark_mode else AERIS_PALETTE["bg_light"]
    paper_color = AERIS_PALETTE["paper_dark"] if dark_mode else AERIS_PALETTE["paper_light"]
    text_color = AERIS_PALETTE["text_dark"] if dark_mode else AERIS_PALETTE["text_light"]
    grid_color = AERIS_PALETTE["grid_dark"] if dark_mode else AERIS_PALETTE["grid_light"]

    fig.update_layout(
        font={"family": FONT_FAMILY, "size": 12, "color": text_color},
        plot_bgcolor=bg_color,
        paper_bgcolor=paper_color,
        hovermode="closest",
    )
    fig.update_xaxes(
        showgrid=show_grid,
        gridcolor=grid_color,
        linecolor=grid_color,
        ticks="outside",
        tickfont={"size": 11, "color": text_color},
    )
    fig.update_yaxes(
        showgrid=show_grid,
        gridcolor=grid_color,
        zerolinecolor=grid_color,
        linecolor=grid_color,
        ticks="outside",
        tickfont={"size": 11, "color": text_color},
    )
    return fig
