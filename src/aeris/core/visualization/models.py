"""
AERIS Visualization Models.

Defines immutable configuration and structured result models for Plotly figure
generation across the AERIS analytical core.
"""

from __future__ import annotations

import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.results.types import Issue, ResultStatus


class RenderMode(StrEnum):
    """Rendering mode for Plotly time-series traces."""

    AUTO = "AUTO"  # Automatically select WebGL (Scattergl) if points >= 10,000, else SVG
    SVG = "SVG"  # Standard SVG rendering via Scatter
    WEBGL = "WEBGL"  # Hardware-accelerated WebGL rendering via Scattergl


class TimeSeriesPlotConfig(BaseModel):
    """
    Immutable configuration options for parameter time-series figure generation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str | None = Field(
        default=None,
        description="Explicit figure title. If None, automatically formatted.",
    )
    display_name: str | None = Field(
        default=None,
        description="Human-readable parameter name for labels and legends.",
    )
    unit: str | None = Field(
        default=None,
        description="Supported engineering unit. Never guessed or fabricated.",
    )
    timestamp_column: str = Field(
        default="timestamp",
        description="Name of the timestamp column in the input DataFrame.",
    )
    epistemic_level: EpistemicLevel = Field(
        default=EpistemicLevel.OBSERVED,
        description="Epistemic standing of the telemetry signal.",
    )
    mode: str = Field(
        default="lines",
        description="Plotly scatter trace mode: 'lines', 'lines+markers', or 'markers'.",
    )
    line_color: str | None = Field(
        default=None,
        description="Hex or CSS color string for trace line. If None, theme default is used.",
    )
    line_width: float = Field(
        default=1.8,
        ge=0.5,
        le=10.0,
        description="Width of trace line in pixels.",
    )
    marker_size: float = Field(
        default=4.0,
        ge=1.0,
        le=20.0,
        description="Size of markers in pixels when mode includes markers.",
    )
    height: int = Field(
        default=450,
        ge=200,
        le=2000,
        description="Figure height in pixels.",
    )
    width: int | None = Field(
        default=None,
        ge=300,
        le=4000,
        description="Figure width in pixels (None for responsive).",
    )
    show_grid: bool = Field(
        default=True,
        description="Whether to display cartesian grid lines.",
    )
    dark_mode: bool = Field(
        default=False,
        description="Whether to render figure in dark theme.",
    )
    render_mode: RenderMode = Field(
        default=RenderMode.AUTO,
        description="Rendering backend selection (AUTO, SVG, or WEBGL).",
    )
    require_monotonic_timestamps: bool = Field(
        default=False,
        description="If True, raises error when timestamps are non-monotonic.",
    )
    connect_gaps: bool = Field(
        default=False,
        description="If False, missing values (NaNs) are rendered as gaps rather than connected.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary user metadata attached to figure layout.",
    )


class TimeSeriesPlotResult(BaseModel):
    """
    Structured outcome of a parameter time-series figure generation request.

    Provides the generated Plotly figure alongside execution issues, status,
    point metrics, and full epistemic provenance.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    status: ResultStatus = Field(
        description="Execution status of the figure builder.",
    )
    parameter_id: str = Field(
        description="Identifier of the plotted parameter.",
    )
    figure: Any = Field(
        description="The generated plotly.graph_objects.Figure instance.",
    )
    issues: list[Issue] = Field(
        default_factory=list,
        description="Issues, warnings, or notes identified during figure construction.",
    )
    provenance: ProvenanceRecord | None = Field(
        default=None,
        description="Epistemic provenance record for this visualization.",
    )
    epistemic_level: EpistemicLevel = Field(
        default=EpistemicLevel.OBSERVED,
        description="Epistemic standing of the plotted signal.",
    )
    point_count: int = Field(
        default=0,
        ge=0,
        description="Total number of rows in the input time-series slice.",
    )
    non_null_count: int = Field(
        default=0,
        ge=0,
        description="Total number of non-null, valid plotted values.",
    )
    unit: str | None = Field(
        default=None,
        description="Unit of measurement displayed on the figure axis.",
    )

    @property
    def null_count(self) -> int:
        """Total number of missing (null or NaN) values in the series."""
        return max(0, self.point_count - self.non_null_count)


class EngineeringEvent(BaseModel):
    """
    Explicit engineering event or test marker for overlay visualization.

    Preserves epistemic level, severity, and temporal boundaries.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(..., description="Stable deterministic event identifier.")
    name: str = Field(..., description="Short descriptive name of the event.")
    timestamp: datetime.datetime | None = Field(
        default=None, description="Point-in-time timestamp (for discrete events)."
    )
    start_timestamp: datetime.datetime | None = Field(
        default=None, description="Start boundary for interval events."
    )
    end_timestamp: datetime.datetime | None = Field(
        default=None, description="End boundary for interval events."
    )
    severity: str = Field(
        default="INFO",
        description="Analytical or operational severity (INFO, CAUTION, WARNING, CRITICAL).",
    )
    description: str | None = Field(default=None, description="Engineering context or explanation.")
    epistemic_level: EpistemicLevel = Field(
        default=EpistemicLevel.OBSERVED,
        description="Epistemic standing of the event record.",
    )


class OverlayResult(BaseModel):
    """
    Structured outcome of applying engineering overlays (limits, bands, events) to a figure.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    status: ResultStatus = Field(
        description="Outcome status of the overlay operation.",
    )
    figure: Any = Field(
        description="The modified plotly.graph_objects.Figure instance.",
    )
    applied_count: int = Field(
        default=0,
        ge=0,
        description="Number of overlays successfully applied.",
    )
    skipped_count: int = Field(
        default=0,
        ge=0,
        description="Number of overlays skipped due to incompatibility or invalidity.",
    )
    issues: list[Issue] = Field(
        default_factory=list,
        description="Diagnostic issues, warnings, or notes encountered.",
    )
    provenance: ProvenanceRecord | None = Field(
        default=None,
        description="Optional provenance record tracking applied overlay sources.",
    )


class PanelLayoutMode(StrEnum):
    """Layout arrangement for multi-signal telemetry panels."""

    STACKED = "STACKED"
    """Each parameter occupies its own dedicated subplot row sharing a synchronized X-axis."""

    UNIT_GROUPED = "UNIT_GROUPED"
    """Parameters with identical units share a subplot row."""

    OVERLAY = "OVERLAY"
    """All parameters plotted on single coordinate axis (requires identical units)."""


class SignalSpec(BaseModel):
    """
    Immutable specification for a single signal to render within a multi-signal panel.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    parameter_id: str = Field(
        ...,
        description="Column name of the parameter in the telemetry dataset.",
    )
    display_name: str | None = Field(
        default=None,
        description="Human-readable parameter name for labels and legends.",
    )
    unit: str | None = Field(
        default=None,
        description="Supported engineering unit. Never guessed or fabricated.",
    )
    epistemic_level: EpistemicLevel = Field(
        default=EpistemicLevel.OBSERVED,
        description="Epistemic standing of the telemetry signal.",
    )
    mode: str = Field(
        default="lines",
        description="Plotly scatter trace mode: 'lines', 'lines+markers', or 'markers'.",
    )
    line_color: str | None = Field(
        default=None,
        description="Hex or CSS color string for trace line. If None, palette default is used.",
    )
    line_width: float = Field(
        default=1.5,
        gt=0.0,
        description="Trace line width in pixels.",
    )
    limits: Any = Field(
        default=None,
        description="Optional parameter-specific engineering limits to overlay.",
    )
    anomalies: Any = Field(
        default=None,
        description="Optional parameter-specific detected anomaly records to overlay.",
    )


class MultiSignalPanelConfig(BaseModel):
    """
    Immutable configuration options for multi-signal panel figure generation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    title: str | None = Field(
        default=None,
        description="Main title of the multi-signal panel. If None, automatically formatted.",
    )
    timestamp_column: str = Field(
        default="timestamp",
        description="Name of the timestamp column in the input DataFrame.",
    )
    layout_mode: PanelLayoutMode = Field(
        default=PanelLayoutMode.STACKED,
        description="Subplot arrangement mode (STACKED, UNIT_GROUPED, or OVERLAY).",
    )
    render_mode: RenderMode = Field(
        default=RenderMode.AUTO,
        description="Rendering backend selection (AUTO, SVG, or WEBGL).",
    )
    row_height: int = Field(
        default=220,
        ge=100,
        description="Height in pixels per subplot row.",
    )
    width: int | None = Field(
        default=None,
        description="Total figure width in pixels. If None, figure is responsive.",
    )
    shared_xaxes: bool = Field(
        default=True,
        description="Whether all subplot rows link and share the X-axis timeline.",
    )
    vertical_spacing: float = Field(
        default=0.06,
        ge=0.01,
        le=0.3,
        description="Normalized vertical spacing between subplot rows.",
    )
    connect_gaps: bool = Field(
        default=False,
        description="If False, missing values (NaNs) appear as physical breaks in the trace.",
    )
    require_monotonic_timestamps: bool = Field(
        default=False,
        description="If True, raises error when timestamps are non-monotonic.",
    )
    dark_mode: bool = Field(
        default=False,
        description="If True, renders using dark aerospace cockpit theme.",
    )
    show_grid: bool = Field(
        default=True,
        description="Whether to display grid lines across subplots.",
    )
    events: Any = Field(
        default=None,
        description="Optional global flight test events or test points to overlay across all rows.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary user metadata attached to figure layout.",
    )


class MultiSignalPanelResult(BaseModel):
    """
    Structured outcome of a multi-signal panel generation request.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    status: ResultStatus = Field(
        description="Execution status of the panel builder.",
    )
    figure: Any = Field(
        description="The generated plotly.graph_objects.Figure instance.",
    )
    signal_ids: list[str] = Field(
        default_factory=list,
        description="Identifiers of requested telemetry parameters.",
    )
    included_signals: list[str] = Field(
        default_factory=list,
        description="Identifiers of signals successfully rendered in the panel.",
    )
    row_count: int = Field(
        default=0,
        ge=0,
        description="Total number of subplot rows generated.",
    )
    point_count: int = Field(
        default=0,
        ge=0,
        description="Total number of rows in the input telemetry dataset.",
    )
    issues: list[Issue] = Field(
        default_factory=list,
        description="Issues, warnings, or notes encountered during panel construction.",
    )
    provenance: ProvenanceRecord | None = Field(
        default=None,
        description="Epistemic provenance record for this multi-signal visualization.",
    )
