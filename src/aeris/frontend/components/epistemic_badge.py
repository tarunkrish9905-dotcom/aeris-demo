"""
AERIS Epistemic Chain Badge Component.

Renders visually distinct badges representing the confidence and provenance level
of data, observations, hypotheses, and recommendations in the analysis pipeline.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from aeris.frontend.theme import EPISTEMIC_COLORS

EPISTEMIC_TOOLTIPS: dict[str, str] = {
    "RAW DATA": (
        "Direct telemetry values ingested from flight data acquisition with zero transforms."
    ),
    "VALIDATED DATA": ("Telemetry validated against schema, ranges, monotonicity, and dropouts."),
    "CALCULATED RESULT": (
        "Derived flight-dynamics parameter calculated via deterministic physical equations."
    ),
    "OBSERVATION": (
        "Factual test condition, exceedance, or statistical characteristic observed in data."
    ),
    "INFERENCE": ("Logical deduction correlating observed parameters with aerodynamic models."),
    "HYPOTHESIS": (
        "Plausible candidate explanation undergoing evaluation against falsification criteria."
    ),
    "RECOMMENDATION": (
        "Automated engineering advisory proposed for qualified flight-test engineer review."
    ),
}


def render_epistemic_badge(level: str, class_name: str = "me-2") -> html.Span:
    """
    Render a styled badge for an epistemic chain level.

    Parameters
    ----------
    level : str
        One of the 7 epistemic levels (e.g. 'CALCULATED RESULT', 'RECOMMENDATION').
    class_name : str
        Additional CSS classes.
    """
    canonical_level = level.upper().strip()
    bg_color = EPISTEMIC_COLORS.get(canonical_level, "#64748B")
    tooltip_text = EPISTEMIC_TOOLTIPS.get(
        canonical_level, "Engineering artifact with recorded data provenance."
    )

    badge_id = f"epistemic-badge-{abs(hash(canonical_level)) % 100000}"

    return html.Span(
        [
            dbc.Badge(
                canonical_level,
                id=badge_id,
                style={
                    "backgroundColor": bg_color,
                    "fontSize": "0.75rem",
                    "fontWeight": "600",
                    "letterSpacing": "0.05em",
                    "padding": "0.35rem 0.65rem",
                    "borderRadius": "4px",
                    "cursor": "help",
                },
                className=class_name,
            ),
            dbc.Tooltip(
                tooltip_text,
                target=badge_id,
                placement="bottom",
            ),
        ]
    )
