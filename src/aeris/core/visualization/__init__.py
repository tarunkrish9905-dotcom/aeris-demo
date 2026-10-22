"""
AERIS Visualization Package.

Provides Plotly visualization foundations, publication-grade styling themes,
reusable parameter time-series builders, contextual engineering overlays
(limits, envelopes, anomalies, events), and multi-signal panel visualizers
for flight telemetry analysis.
"""

from __future__ import annotations

from aeris.core.visualization.models import (
    EngineeringEvent,
    MultiSignalPanelConfig,
    MultiSignalPanelResult,
    OverlayResult,
    PanelLayoutMode,
    RenderMode,
    SignalSpec,
    TimeSeriesPlotConfig,
    TimeSeriesPlotResult,
)
from aeris.core.visualization.overlays import (
    ANOMALY_SEVERITY_STYLES,
    LIMIT_SEVERITY_STYLES,
    add_anomaly_markers,
    add_event_markers,
    add_limit_band,
    add_limit_line,
    add_limit_overlays,
    generate_overlaid_time_series_plot,
)
from aeris.core.visualization.panels import (
    SIGNAL_COLOR_CYCLE,
    build_multi_signal_panel,
    generate_multi_signal_panel,
)
from aeris.core.visualization.theme import (
    AERIS_PALETTE,
    FONT_FAMILY,
    apply_aeris_theme,
    get_aeris_layout,
)
from aeris.core.visualization.time_series import (
    build_time_series_figure,
    generate_time_series_plot,
)

__all__ = [
    "AERIS_PALETTE",
    "ANOMALY_SEVERITY_STYLES",
    "FONT_FAMILY",
    "LIMIT_SEVERITY_STYLES",
    "SIGNAL_COLOR_CYCLE",
    "EngineeringEvent",
    "MultiSignalPanelConfig",
    "MultiSignalPanelResult",
    "OverlayResult",
    "PanelLayoutMode",
    "RenderMode",
    "SignalSpec",
    "TimeSeriesPlotConfig",
    "TimeSeriesPlotResult",
    "add_anomaly_markers",
    "add_event_markers",
    "add_limit_band",
    "add_limit_line",
    "add_limit_overlays",
    "apply_aeris_theme",
    "build_multi_signal_panel",
    "build_time_series_figure",
    "generate_multi_signal_panel",
    "generate_overlaid_time_series_plot",
    "generate_time_series_plot",
    "get_aeris_layout",
]
