"""
AERIS Frontend Theme & Styling Constants.

Provides color palettes, typography standards, and Bootstrap helper classes
consistent with the AERIS engineering visualization design language.
"""

from __future__ import annotations

# =============================================================================
# Engineering Color Palette
# =============================================================================

PRIMARY_NAVY = "#0F172A"  # Deep slate/navy for navigation and headers
SECONDARY_SLATE = "#334155"  # Neutral secondary
ACCENT_BLUE = "#0284C7"  # Interactive action blue
SUCCESS_EMERALD = "#059669"  # Validated / Pass / Healthy
WARNING_AMBER = "#D97706"  # Quality warnings / Advisory thresholds
DANGER_CRIMSON = "#DC2626"  # Critical failures / Exceedances / Errors
INFO_CYAN = "#0891B2"  # Inferences / Telemetry metadata
BG_LIGHT = "#F8FAFC"  # Content card background
BORDER_MUTED = "#E2E8F0"  # Subtle divider lines

# Epistemic Level Colors
EPISTEMIC_COLORS: dict[str, str] = {
    "RAW DATA": "#64748B",  # Slate
    "VALIDATED DATA": "#0284C7",  # Ocean Blue
    "CALCULATED RESULT": "#2563EB",  # Engineering Blue
    "OBSERVATION": "#0D9488",  # Teal
    "INFERENCE": "#7C3AED",  # Violet
    "HYPOTHESIS": "#D97706",  # Amber
    "RECOMMENDATION": "#DC2626",  # Crimson
}

# Typography & Layout Styles
FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO_FONT = "SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace"

CARD_STYLE = {
    "borderRadius": "8px",
    "boxShadow": "0 1px 3px 0 rgba(0, 0, 0, 0.08), 0 1px 2px 0 rgba(0, 0, 0, 0.04)",
    "border": f"1px solid {BORDER_MUTED}",
    "backgroundColor": "#FFFFFF",
    "marginBottom": "1.25rem",
}

HEADER_STYLE = {
    "fontWeight": "600",
    "color": PRIMARY_NAVY,
}

SUBTITLE_STYLE = {
    "color": "#64748B",
    "fontSize": "0.9rem",
    "marginBottom": "1rem",
}
