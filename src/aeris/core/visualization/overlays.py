"""
AERIS Contextual Engineering Visualizers & Overlay Builders.

Provides reusable overlays for:
- Engineering upper and lower limits (with severity-specific line styling).
- Engineering bands and operational envelopes (with shaded corridors).
- Detected anomalies and temporal deviation intervals (with epistemic badges).
- Flight test events and test point observations (with vertical span regions).

Preserves strict aerospace engineering standards:
- Applies limits only when parameter identity, units, and vehicle/configuration
  applicability are strictly compatible.
- Never guesses or fabricates units; never performs silent unit conversion.
- Distinguishes overlays via both line style/marker glyph and color (accessible).
- Preserves epistemic boundaries: anomalies and events are explicitly labeled
  as analytical detections, never as confirmed faults or root causes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from aeris.core.anomaly.models import AnomalyRecord, AnomalySeverity
from aeris.core.limits.models import (
    EngineeringLimitProfile,
    LimitSeverity,
    LimitType,
    ParameterLimitDefinition,
)
from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus
from aeris.core.test_operations.test_point_observation import TestPointObservation
from aeris.core.vehicle_evaluation.models import LimitEvaluationRecord
from aeris.core.visualization.models import (
    EngineeringEvent,
    OverlayResult,
    TimeSeriesPlotConfig,
    TimeSeriesPlotResult,
)
from aeris.core.visualization.theme import AERIS_PALETTE
from aeris.core.visualization.time_series import generate_time_series_plot

# ---------------------------------------------------------------------------
# Visual Styling Mappings (Color + Pattern distinction)
# ---------------------------------------------------------------------------

LIMIT_SEVERITY_STYLES: dict[LimitSeverity | str, dict[str, Any]] = {
    LimitSeverity.EXCEEDANCE: {
        "color": AERIS_PALETTE["critical"],
        "dash": "solid",
        "width": 2.0,
        "tag": "EXCEEDANCE",
    },
    LimitSeverity.WARNING: {
        "color": AERIS_PALETTE["warning"],
        "dash": "dash",
        "width": 1.5,
        "tag": "WARNING",
    },
    LimitSeverity.CAUTION: {
        "color": AERIS_PALETTE["caution"],
        "dash": "dot",
        "width": 1.5,
        "tag": "CAUTION",
    },
    LimitSeverity.ADVISORY: {
        "color": AERIS_PALETTE["accent"],
        "dash": "dashdot",
        "width": 1.0,
        "tag": "ADVISORY",
    },
}

ANOMALY_SEVERITY_STYLES: dict[AnomalySeverity | str, dict[str, Any]] = {
    AnomalySeverity.HIGH: {
        "color": AERIS_PALETTE["critical"],
        "symbol": "diamond",
        "size": 10,
        "tag": "HIGH",
    },
    AnomalySeverity.MEDIUM: {
        "color": AERIS_PALETTE["warning"],
        "symbol": "triangle-up",
        "size": 9,
        "tag": "MEDIUM",
    },
    AnomalySeverity.LOW: {
        "color": AERIS_PALETTE["caution"],
        "symbol": "square",
        "size": 8,
        "tag": "LOW",
    },
    AnomalySeverity.INFO: {
        "color": AERIS_PALETTE["neutral"],
        "symbol": "circle",
        "size": 7,
        "tag": "INFO",
    },
}


# ---------------------------------------------------------------------------
# Single-Element Overlay Helpers
# ---------------------------------------------------------------------------


def add_limit_line(
    fig: go.Figure,
    value: float,
    label: str,
    severity: LimitSeverity | str = LimitSeverity.WARNING,
    unit: str | None = None,
    dash: str | None = None,
    color: str | None = None,
    width: float | None = None,
    show_legend: bool = True,
    show_annotations: bool = True,
) -> go.Figure:
    """
    Add a single horizontal engineering limit line to an existing Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        Target Plotly figure.
    value : float
        Numeric threshold value.
    label : str
        Descriptive label for the limit line.
    severity : LimitSeverity | str
        Limit severity (EXCEEDANCE, WARNING, CAUTION, ADVISORY).
    unit : str | None
        Engineering unit string. Never fabricated.
    dash : str | None
        Explicit dash pattern override (e.g. 'solid', 'dash', 'dot').
    color : str | None
        Explicit hex color override.
    width : float | None
        Line width in pixels.
    show_legend : bool
        Whether to create a legend entry for interactive toggling.
    show_annotations : bool
        Whether to display an in-graph annotation badge.

    Returns
    -------
    go.Figure
        Modified figure with limit line overlay.
    """
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"Limit value must be a finite numeric float, got {value}.")

    style = LIMIT_SEVERITY_STYLES.get(severity, LIMIT_SEVERITY_STYLES[LimitSeverity.WARNING])
    line_dash = dash or style["dash"]
    line_color = color or style["color"]
    line_width = width or style["width"]

    unit_str = f" {unit}" if unit else ""
    full_label = f"[LIMIT] {label}: {value:g}{unit_str}"
    annot_text = f"<b>[LIMIT]</b> {label}: {value:g}{unit_str}" if show_annotations else None

    fig.add_hline(
        y=value,
        line_dash=line_dash,
        line_color=line_color,
        line_width=line_width,
        annotation_text=annot_text,
        annotation_position="top right",
        annotation_font={"size": 10, "color": line_color},
    )

    if show_legend:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=full_label,
                line={"color": line_color, "dash": line_dash, "width": line_width},
                showlegend=True,
                hoverinfo="none",
            )
        )

    return fig


def add_limit_band(
    fig: go.Figure,
    lower_bound: float,
    upper_bound: float,
    label: str = "Allowable Band",
    severity: LimitSeverity | str = LimitSeverity.CAUTION,
    unit: str | None = None,
    color: str | None = None,
    opacity: float = 0.12,
    show_legend: bool = True,
    show_annotations: bool = True,
) -> go.Figure:
    """
    Add a horizontal engineering limit band / envelope to an existing Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        Target Plotly figure.
    lower_bound : float
        Lower numeric threshold.
    upper_bound : float
        Upper numeric threshold.
    label : str
        Descriptive label for the band.
    severity : LimitSeverity | str
        Limit severity classification.
    unit : str | None
        Engineering unit string.
    color : str | None
        Explicit color string.
    opacity : float
        Fill opacity (0.0 to 1.0).
    show_legend : bool
        Whether to display in the legend.
    show_annotations : bool
        Whether to show an in-graph annotation badge.

    Returns
    -------
    go.Figure
        Modified figure with envelope overlay.

    Raises
    ------
    ValueError
        If lower_bound > upper_bound or bounds are non-finite.
    """
    if math.isnan(lower_bound) or math.isinf(lower_bound):
        raise ValueError(f"lower_bound must be finite, got {lower_bound}.")
    if math.isnan(upper_bound) or math.isinf(upper_bound):
        raise ValueError(f"upper_bound must be finite, got {upper_bound}.")
    if lower_bound > upper_bound:
        raise ValueError(f"lower_bound ({lower_bound}) cannot exceed upper_bound ({upper_bound}).")

    style = LIMIT_SEVERITY_STYLES.get(severity, LIMIT_SEVERITY_STYLES[LimitSeverity.CAUTION])
    band_color = color or style["color"]
    unit_str = f" {unit}" if unit else ""
    full_label = f"[LIMIT ENVELOPE] {label}: [{lower_bound:g}, {upper_bound:g}]{unit_str}"
    annot_text = (
        f"<b>[LIMIT ENVELOPE]</b> {label}: [{lower_bound:g}, {upper_bound:g}]{unit_str}"
        if show_annotations
        else None
    )

    fig.add_hrect(
        y0=lower_bound,
        y1=upper_bound,
        fillcolor=band_color,
        opacity=opacity,
        line_width=1,
        line_dash=style["dash"],
        line_color=band_color,
        annotation_text=annot_text,
        annotation_position="top left",
        annotation_font={"size": 10, "color": band_color},
    )

    if show_legend:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=full_label,
                line={"color": band_color, "dash": style["dash"], "width": 1.5},
                showlegend=True,
                hoverinfo="none",
            )
        )

    return fig


# ---------------------------------------------------------------------------
# Structured Multi-Limit Overlay Helper
# ---------------------------------------------------------------------------


def add_limit_overlays(
    fig: go.Figure,
    limits: (Sequence[ParameterLimitDefinition | LimitEvaluationRecord] | EngineeringLimitProfile),
    parameter_id: str,
    unit: str | None = None,
    configuration_id: str | None = None,
    vehicle_id: str | None = None,
    show_legend: bool = True,
    show_annotations: bool = True,
    source_uri: str | None = None,
    parent_record_ids: list[str] | None = None,
) -> OverlayResult:
    """
    Apply engineering limit overlays (upper, lower, bands) to a figure with unit safety.

    Validates parameter identity, unit compatibility, and vehicle/configuration
    scoping before applying any boundary overlay. Never performs silent unit conversions.

    Parameters
    ----------
    fig : go.Figure
        Target Plotly figure.
    limits : Sequence[ParameterLimitDefinition | LimitEvaluationRecord] | EngineeringLimitProfile
        Collection of defined limits or evaluated records.
    parameter_id : str
        Target telemetry parameter identifier.
    unit : str | None
        Telemetry unit. If None, inspected from figure layout metadata.
    configuration_id : str | None
        Optional vehicle configuration identifier filter.
    vehicle_id : str | None
        Optional vehicle identity filter.
    show_legend : bool
        Whether to generate legend entries for applied limits.
    show_annotations : bool
        Whether to render annotation labels.
    source_uri : str | None
        Provenance source URI.
    parent_record_ids : list[str] | None
        Parent provenance identifiers.

    Returns
    -------
    OverlayResult
        Structured result detailing applied overlays, skipped limits, and issues.
    """
    issues: list[Issue] = []
    applied_count = 0
    skipped_count = 0

    # Resolve target unit
    active_unit: str | None = unit
    if active_unit is None and hasattr(fig.layout, "meta") and isinstance(fig.layout.meta, dict):
        active_unit = fig.layout.meta.get("unit")

    # Extract limit sequence
    limit_list: list[Any]
    if isinstance(limits, EngineeringLimitProfile):
        if vehicle_id is not None and limits.vehicle_id != vehicle_id:
            issues.append(
                Issue(
                    issue_code="WARN_VEHICLE_MISMATCH",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"LimitProfile vehicle_id '{limits.vehicle_id}' does not match "
                        f"requested vehicle_id '{vehicle_id}'. Overlays skipped."
                    ),
                    field_name=parameter_id,
                )
            )
            return OverlayResult(
                status=ResultStatus.WARNING,
                figure=fig,
                applied_count=0,
                skipped_count=len(limits.limits),
                issues=issues,
            )

        limit_list = list(
            limits.get_limits_for_parameter(
                parameter_id,
                configuration_id=configuration_id,
                include_all_configurations=(configuration_id is None),
            )
        )
        if len(limit_list) == 0:
            issues.append(
                Issue(
                    issue_code="INFO_NO_LIMITS_FOUND",
                    severity=IssueSeverity.INFO,
                    message=(
                        f"No engineering limits configured for parameter '{parameter_id}' "
                        f"in configuration '{configuration_id}'."
                    ),
                    field_name=parameter_id,
                )
            )
            # Epistemic safeguard: Explicitly record unconfigured status
            if hasattr(fig.layout, "meta") and isinstance(fig.layout.meta, dict):
                fig.layout.meta["limits_status"] = "UNCONFIGURED"

            return OverlayResult(
                status=ResultStatus.SUCCESS,
                figure=fig,
                applied_count=0,
                skipped_count=0,
                issues=issues,
            )
    else:
        limit_list = list(limits)

    for lim in limit_list:
        lim_id = getattr(lim, "limit_id", "unnamed_limit")
        lim_param = getattr(lim, "parameter_id", None)

        # 1. Parameter Identity Check
        if lim_param != parameter_id:
            issues.append(
                Issue(
                    issue_code="WARN_PARAMETER_MISMATCH",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"Limit '{lim_id}' parameter '{lim_param}' does not match "
                        f"target parameter '{parameter_id}'."
                    ),
                    field_name=parameter_id,
                )
            )
            skipped_count += 1
            continue

        # 2. Configuration Applicability Check
        lim_config = getattr(lim, "configuration_id", None)
        if (
            configuration_id is not None
            and lim_config is not None
            and lim_config != configuration_id
        ):
            issues.append(
                Issue(
                    issue_code="INFO_LIMIT_CONFIGURATION_MISMATCH",
                    severity=IssueSeverity.INFO,
                    message=(
                        f"Limit '{lim_id}' configuration '{lim_config}' does not match "
                        f"active configuration '{configuration_id}'. Overlay skipped."
                    ),
                    field_name=parameter_id,
                )
            )
            skipped_count += 1
            continue

        # 3. Vehicle Applicability Check
        lim_vehicle = getattr(lim, "vehicle_id", None)
        if vehicle_id is not None and lim_vehicle is not None and lim_vehicle != vehicle_id:
            issues.append(
                Issue(
                    issue_code="WARN_VEHICLE_MISMATCH",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"Limit '{lim_id}' vehicle_id '{lim_vehicle}' does not match "
                        f"active vehicle '{vehicle_id}'. Overlay skipped."
                    ),
                    field_name=parameter_id,
                )
            )
            skipped_count += 1
            continue

        # 4. Strict Unit Safety Gate
        lim_unit = getattr(lim, "unit", getattr(lim, "limit_unit", None))
        if active_unit is not None and lim_unit is not None and active_unit != lim_unit:
            issues.append(
                Issue(
                    issue_code="ERR_INCOMPATIBLE_LIMIT_UNIT",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"Limit '{lim_id}' unit '{lim_unit}' is incompatible with signal "
                        f"unit '{active_unit}'. Overlay suppressed."
                    ),
                    field_name=parameter_id,
                )
            )
            # Add subtle visual indicator of unit mismatch on figure
            if show_annotations:
                mismatch_note = (
                    f"Limit '{lim_id}' suppressed: unit mismatch ({lim_unit} vs {active_unit})"
                )
                fig.add_annotation(
                    text=mismatch_note,
                    xref="paper",
                    yref="paper",
                    x=0.02,
                    y=0.02 + 0.04 * skipped_count,
                    showarrow=False,
                    font={"size": 10, "color": AERIS_PALETTE["warning"]},
                    bgcolor="rgba(255, 255, 255, 0.7)",
                )
            skipped_count += 1
            continue

        # 5. Boundary Application
        lim_type = getattr(lim, "limit_type", None)
        lim_sev = getattr(lim, "severity", LimitSeverity.WARNING)
        style = LIMIT_SEVERITY_STYLES.get(lim_sev, LIMIT_SEVERITY_STYLES[LimitSeverity.WARNING])

        try:
            if lim_type == LimitType.UPPER_BOUND:
                val = getattr(lim, "upper_bound", None)
                if val is None:
                    val = getattr(lim, "limit_value", None)
                if val is None or math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Limit '{lim_id}' upper_bound is missing or non-finite.")

                label_text = f"Upper {style['tag']}"
                add_limit_line(
                    fig=fig,
                    value=float(val),
                    label=label_text,
                    severity=lim_sev,
                    unit=lim_unit or active_unit,
                    show_legend=show_legend,
                    show_annotations=show_annotations,
                )
                applied_count += 1

            elif lim_type == LimitType.LOWER_BOUND:
                val = getattr(lim, "lower_bound", None)
                if val is None:
                    val = getattr(lim, "limit_value", None)
                if val is None or math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Limit '{lim_id}' lower_bound is missing or non-finite.")

                label_text = f"Lower {style['tag']}"
                add_limit_line(
                    fig=fig,
                    value=float(val),
                    label=label_text,
                    severity=lim_sev,
                    unit=lim_unit or active_unit,
                    show_legend=show_legend,
                    show_annotations=show_annotations,
                )
                applied_count += 1

            elif lim_type == LimitType.BAND:
                lower = getattr(lim, "lower_bound", None)
                upper = getattr(lim, "upper_bound", None)
                if lower is None or upper is None:
                    raise ValueError(f"Limit '{lim_id}' BAND requires both lower and upper bounds.")
                if lower > upper:
                    raise ValueError(
                        f"Limit '{lim_id}' lower_bound ({lower}) exceeds upper_bound ({upper})."
                    )

                label_text = f"{style['tag']} Band"
                add_limit_band(
                    fig=fig,
                    lower_bound=float(lower),
                    upper_bound=float(upper),
                    label=label_text,
                    severity=lim_sev,
                    unit=lim_unit or active_unit,
                    show_legend=show_legend,
                    show_annotations=show_annotations,
                )
                applied_count += 1

            else:
                issues.append(
                    Issue(
                        issue_code="WARN_UNSUPPORTED_LIMIT_TYPE",
                        severity=IssueSeverity.WARNING,
                        message=f"Unsupported limit_type '{lim_type}' on limit '{lim_id}'.",
                        field_name=parameter_id,
                    )
                )
                skipped_count += 1

        except ValueError as exc:
            issues.append(
                Issue(
                    issue_code="ERR_INVALID_BOUNDS",
                    severity=IssueSeverity.ERROR,
                    message=f"Failed applying limit '{lim_id}': {exc}",
                    field_name=parameter_id,
                )
            )
            skipped_count += 1

    # Record metadata
    if hasattr(fig.layout, "meta") and isinstance(fig.layout.meta, dict):
        fig.layout.meta["applied_limits_count"] = applied_count
        fig.layout.meta["skipped_limits_count"] = skipped_count
        fig.layout.meta["limits_status"] = "APPLIED" if applied_count > 0 else "NONE"

    prov = create_provenance(
        activity_name="limit_overlay_application",
        source_uri=source_uri or "in_memory",
        epistemic_level=EpistemicLevel.CALCULATED,
        parent_record_ids=parent_record_ids or [],
        parameters={
            "parameter_id": parameter_id,
            "applied_count": applied_count,
            "skipped_count": skipped_count,
        },
    )

    status = (
        ResultStatus.ERROR
        if any(i.severity == IssueSeverity.ERROR for i in issues)
        else (ResultStatus.WARNING if issues else ResultStatus.SUCCESS)
    )

    return OverlayResult(
        status=status,
        figure=fig,
        applied_count=applied_count,
        skipped_count=skipped_count,
        issues=issues,
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# Anomaly Markers Overlay Helper
# ---------------------------------------------------------------------------


def add_anomaly_markers(
    fig: go.Figure,
    anomalies: Sequence[AnomalyRecord],
    parameter_id: str | None = None,
    show_legend: bool = True,
    show_annotations: bool = True,
    source_uri: str | None = None,
    parent_record_ids: list[str] | None = None,
) -> OverlayResult:
    """
    Apply detected anomaly markers (point anomalies and temporal intervals) to a figure.

    Distinguishes anomalies using both marker shapes/dash lines and high-contrast
    severity colors. Preserves epistemic boundaries: anomalies are tagged explicitly
    as analytical detections, never as confirmed root causes or operational faults.

    Parameters
    ----------
    fig : go.Figure
        Target Plotly figure.
    anomalies : Sequence[AnomalyRecord]
        List of detected anomaly records.
    parameter_id : str | None
        Target parameter ID filter.
    show_legend : bool
        Whether to generate legend entries.
    show_annotations : bool
        Whether to display in-figure text labels.
    source_uri : str | None
        Provenance source URI.
    parent_record_ids : list[str] | None
        Parent provenance lineage IDs.

    Returns
    -------
    OverlayResult
        Structured overlay result with counts and issues.
    """
    issues: list[Issue] = []
    applied_count = 0
    skipped_count = 0

    for anomaly in anomalies:
        if parameter_id is not None and anomaly.target_column != parameter_id:
            # Not an issue; simply targeted to another parameter
            continue

        style = ANOMALY_SEVERITY_STYLES.get(
            anomaly.severity, ANOMALY_SEVERITY_STYLES[AnomalySeverity.HIGH]
        )

        # 1. Interval Anomaly Overlay
        if anomaly.start_timestamp is not None and anomaly.end_timestamp is not None:
            if anomaly.start_timestamp > anomaly.end_timestamp:
                issues.append(
                    Issue(
                        issue_code="ERR_INVALID_TIMESTAMP",
                        severity=IssueSeverity.WARNING,
                        message=(
                            f"Anomaly '{anomaly.anomaly_id}' start_timestamp "
                            f"({anomaly.start_timestamp}) exceeds end_timestamp "
                            f"({anomaly.end_timestamp}). Overlay skipped."
                        ),
                        field_name=anomaly.target_column,
                    )
                )
                skipped_count += 1
                continue

            annot_text = (
                f"<b>[ANOMALY: {style['tag']}]</b> {anomaly.anomaly_id}"
                if show_annotations
                else None
            )

            fig.add_vrect(
                x0=anomaly.start_timestamp,
                x1=anomaly.end_timestamp,
                fillcolor=style["color"],
                opacity=0.18,
                line_dash="dot",
                line_color=style["color"],
                line_width=1.5,
                annotation_text=annot_text,
                annotation_position="top left",
                annotation_font={"size": 10, "color": style["color"]},
            )

            if show_legend:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        name=f"[ANOMALY: {style['tag']}] {anomaly.anomaly_id}",
                        line={"color": style["color"], "dash": "dot", "width": 1.5},
                        showlegend=True,
                        hoverinfo="none",
                    )
                )
            applied_count += 1

        # 2. Point Anomaly Overlay
        elif anomaly.timestamp is not None:
            ts = anomaly.timestamp
            obs_val = anomaly.observed_value

            # If observed value is numeric, plot marker glyph at exact coordinate
            if obs_val is not None and not isinstance(obs_val, str) and not math.isnan(obs_val):
                hover_text = (
                    f"<b>[ANOMALY DETECTED]</b><br>"
                    f"<b>Anomaly ID</b>: {anomaly.anomaly_id}<br>"
                    f"<b>Severity</b>: {anomaly.severity.value}<br>"
                    f"<b>Observed Value</b>: {obs_val:g}<br>"
                    f"<b>Method</b>: {anomaly.detection_method.value}<br>"
                    f"<i>Epistemic: [{anomaly.epistemic_level.value}] "
                    f"(Analytical finding; not confirmed root cause)</i>"
                    "<extra></extra>"
                )

                fig.add_trace(
                    go.Scatter(
                        x=[ts],
                        y=[float(obs_val)],
                        mode="markers",
                        name=f"[ANOMALY: {style['tag']}] {anomaly.anomaly_id}",
                        marker={
                            "symbol": style["symbol"],
                            "size": style["size"],
                            "color": style["color"],
                            "line": {"color": "#FFFFFF", "width": 1},
                        },
                        hovertemplate=hover_text,
                        showlegend=show_legend,
                    )
                )
                applied_count += 1
            else:
                # Value not numeric or missing: draw vertical event line
                annot_text = (
                    f"<b>[ANOMALY: {style['tag']}]</b> {anomaly.anomaly_id}"
                    if show_annotations
                    else None
                )
                fig.add_vline(
                    x=ts,
                    line_dash="dash",
                    line_color=style["color"],
                    line_width=1.5,
                    annotation_text=annot_text,
                    annotation_position="top right",
                    annotation_font={"size": 10, "color": style["color"]},
                )
                if show_legend:
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            name=f"[ANOMALY: {style['tag']}] {anomaly.anomaly_id}",
                            line={"color": style["color"], "dash": "dash", "width": 1.5},
                            showlegend=True,
                            hoverinfo="none",
                        )
                    )
                applied_count += 1

        else:
            issues.append(
                Issue(
                    issue_code="ERR_INVALID_TIMESTAMP",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"Anomaly '{anomaly.anomaly_id}' specifies neither point timestamp "
                        f"nor interval boundaries. Overlay skipped."
                    ),
                    field_name=anomaly.target_column,
                )
            )
            skipped_count += 1

    prov = create_provenance(
        activity_name="anomaly_marker_overlay",
        source_uri=source_uri or "in_memory",
        epistemic_level=EpistemicLevel.CALCULATED,
        parent_record_ids=parent_record_ids or [],
        parameters={
            "parameter_id": parameter_id,
            "applied_count": applied_count,
            "skipped_count": skipped_count,
        },
    )

    status = (
        ResultStatus.ERROR
        if any(i.severity == IssueSeverity.ERROR for i in issues)
        else (ResultStatus.WARNING if issues else ResultStatus.SUCCESS)
    )

    return OverlayResult(
        status=status,
        figure=fig,
        applied_count=applied_count,
        skipped_count=skipped_count,
        issues=issues,
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# Test Point & Engineering Event Markers Overlay Helper
# ---------------------------------------------------------------------------


def add_event_markers(
    fig: go.Figure,
    events: Sequence[TestPointObservation | EngineeringEvent | Any],
    show_legend: bool = True,
    show_annotations: bool = True,
    source_uri: str | None = None,
    parent_record_ids: list[str] | None = None,
) -> OverlayResult:
    """
    Apply flight test points or explicit engineering event markers to a figure.

    Supports both closed temporal intervals (e.g. TestPointObservation) and
    discrete point events (EngineeringEvent).

    Parameters
    ----------
    fig : go.Figure
        Target Plotly figure.
    events : Sequence[TestPointObservation | EngineeringEvent | Any]
        Sequence of events or test point observation records.
    show_legend : bool
        Whether to generate legend entries.
    show_annotations : bool
        Whether to render in-graph annotation badges.
    source_uri : str | None
        Provenance source URI.
    parent_record_ids : list[str] | None
        Lineage record IDs.

    Returns
    -------
    OverlayResult
        Structured overlay result with counts and issues.
    """
    issues: list[Issue] = []
    applied_count = 0
    skipped_count = 0

    for event in events:
        # 1. TestPointObservation (Temporal Interval)
        if isinstance(event, TestPointObservation):
            start = event.start_timestamp
            end = event.end_timestamp
            if start > end:
                issues.append(
                    Issue(
                        issue_code="ERR_INVALID_TIMESTAMP",
                        severity=IssueSeverity.WARNING,
                        message=(
                            f"TestPointObservation '{event.observation_id}' start ({start}) "
                            f"exceeds end ({end}). Overlay skipped."
                        ),
                    )
                )
                skipped_count += 1
                continue

            color = AERIS_PALETTE["accent"]
            label_text = f"[TEST POINT] {event.test_point_id}"
            annot_text = f"<b>[TEST POINT]</b> {event.test_point_id}" if show_annotations else None

            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=color,
                opacity=0.12,
                line_dash="dot",
                line_color=color,
                line_width=1,
                annotation_text=annot_text,
                annotation_position="top left",
                annotation_font={"size": 10, "color": color},
            )

            if show_legend:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        name=label_text,
                        line={"color": color, "dash": "dot", "width": 1.5},
                        showlegend=True,
                        hoverinfo="none",
                    )
                )
            applied_count += 1

        # 2. EngineeringEvent (Discrete point or interval)
        elif isinstance(event, EngineeringEvent):
            color = AERIS_PALETTE["accent"]
            if event.severity.upper() in ("WARNING", "CRITICAL"):
                color = AERIS_PALETTE["warning"]

            if event.start_timestamp is not None and event.end_timestamp is not None:
                if event.start_timestamp > event.end_timestamp:
                    issues.append(
                        Issue(
                            issue_code="ERR_INVALID_TIMESTAMP",
                            severity=IssueSeverity.WARNING,
                            message=(
                                f"EngineeringEvent '{event.event_id}' start "
                                f"({event.start_timestamp}) exceeds end ({event.end_timestamp})."
                            ),
                        )
                    )
                    skipped_count += 1
                    continue

                label_text = f"[EVENT] {event.name}"
                annot_text = f"<b>[EVENT]</b> {event.name}" if show_annotations else None

                fig.add_vrect(
                    x0=event.start_timestamp,
                    x1=event.end_timestamp,
                    fillcolor=color,
                    opacity=0.12,
                    line_dash="dashdot",
                    line_color=color,
                    line_width=1,
                    annotation_text=annot_text,
                    annotation_position="top left",
                    annotation_font={"size": 10, "color": color},
                )
                if show_legend:
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            name=label_text,
                            line={"color": color, "dash": "dashdot", "width": 1.5},
                            showlegend=True,
                            hoverinfo="none",
                        )
                    )
                applied_count += 1

            elif event.timestamp is not None:
                label_text = f"[EVENT] {event.name}"
                annot_text = f"<b>[EVENT]</b> {event.name}" if show_annotations else None

                fig.add_vline(
                    x=event.timestamp,
                    line_dash="dashdot",
                    line_color=color,
                    line_width=1.5,
                    annotation_text=annot_text,
                    annotation_position="top",
                    annotation_font={"size": 10, "color": color},
                )
                if show_legend:
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            name=label_text,
                            line={"color": color, "dash": "dashdot", "width": 1.5},
                            showlegend=True,
                            hoverinfo="none",
                        )
                    )
                applied_count += 1

            else:
                issues.append(
                    Issue(
                        issue_code="ERR_INVALID_TIMESTAMP",
                        severity=IssueSeverity.WARNING,
                        message=f"EngineeringEvent '{event.event_id}' lacks timestamp information.",
                    )
                )
                skipped_count += 1

        else:
            issues.append(
                Issue(
                    issue_code="WARN_UNRECOGNIZED_EVENT_TYPE",
                    severity=IssueSeverity.WARNING,
                    message=f"Unrecognized event object type: {type(event).__name__}.",
                )
            )
            skipped_count += 1

    prov = create_provenance(
        activity_name="event_marker_overlay",
        source_uri=source_uri or "in_memory",
        epistemic_level=EpistemicLevel.OBSERVED,
        parent_record_ids=parent_record_ids or [],
        parameters={
            "applied_count": applied_count,
            "skipped_count": skipped_count,
        },
    )

    status = (
        ResultStatus.ERROR
        if any(i.severity == IssueSeverity.ERROR for i in issues)
        else (ResultStatus.WARNING if issues else ResultStatus.SUCCESS)
    )

    return OverlayResult(
        status=status,
        figure=fig,
        applied_count=applied_count,
        skipped_count=skipped_count,
        issues=issues,
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# Integrated Overlaid Plot Generator
# ---------------------------------------------------------------------------


def generate_overlaid_time_series_plot(
    data: pd.DataFrame,
    parameter_id: str,
    limits: (
        Sequence[ParameterLimitDefinition | LimitEvaluationRecord] | EngineeringLimitProfile | None
    ) = None,
    anomalies: Sequence[AnomalyRecord] | None = None,
    events: Sequence[TestPointObservation | EngineeringEvent] | None = None,
    timestamp_column: str = "timestamp",
    unit: str | None = None,
    display_name: str | None = None,
    epistemic_level: EpistemicLevel = EpistemicLevel.OBSERVED,
    config: TimeSeriesPlotConfig | None = None,
    configuration_id: str | None = None,
    vehicle_id: str | None = None,
    source_uri: str | None = None,
    parent_record_ids: list[str] | None = None,
) -> TimeSeriesPlotResult:
    """
    Generate a parameter time-series figure and overlay limits, anomalies, and events.

    Synthesizes the base time-series trace with all compatible engineering context,
    enforcing unit safety, epistemic boundaries, and DataFrame immutability.

    Parameters
    ----------
    data : pd.DataFrame
        Input flight data (read-only; never mutated).
    parameter_id : str
        Column name of the parameter.
    limits : Sequence[Any] | EngineeringLimitProfile | None
        Applicable engineering limits (ParameterLimitDefinition or LimitEvaluationRecord).
    anomalies : Sequence[AnomalyRecord] | None
        Detected anomalies.
    events : Sequence[TestPointObservation | EngineeringEvent] | None
        Test points or mission events.
    timestamp_column : str
        Name of timestamp column.
    unit : str | None
        Engineering unit string.
    display_name : str | None
        Display name.
    epistemic_level : EpistemicLevel
        Epistemic classification of primary telemetry trace.
    config : TimeSeriesPlotConfig | None
        Plot configuration options.
    configuration_id : str | None
        Vehicle configuration scope.
    vehicle_id : str | None
        Target vehicle identifier.
    source_uri : str | None
        Source dataset URI.
    parent_record_ids : list[str] | None
        Parent provenance identifiers.

    Returns
    -------
    TimeSeriesPlotResult
        Structured container with composite figure, execution status, and all diagnostic issues.
    """
    # 1. Generate Base Time-Series Plot
    base_result = generate_time_series_plot(
        data=data,
        parameter_id=parameter_id,
        timestamp_column=timestamp_column,
        unit=unit,
        display_name=display_name,
        epistemic_level=epistemic_level,
        config=config,
        source_uri=source_uri,
        parent_record_ids=parent_record_ids,
    )

    if base_result.status == ResultStatus.ERROR:
        # Base figure construction failed (e.g. missing column or unmonotonic timestamps)
        return base_result

    fig = base_result.figure
    composite_issues = list(base_result.issues)

    # 2. Apply Limit Overlays
    if limits is not None:
        limit_overlay_res = add_limit_overlays(
            fig=fig,
            limits=limits,
            parameter_id=parameter_id,
            unit=unit,
            configuration_id=configuration_id,
            vehicle_id=vehicle_id,
            source_uri=source_uri,
            parent_record_ids=parent_record_ids,
        )
        composite_issues.extend(limit_overlay_res.issues)

    # 3. Apply Anomaly Markers
    if anomalies is not None:
        anomaly_overlay_res = add_anomaly_markers(
            fig=fig,
            anomalies=anomalies,
            parameter_id=parameter_id,
            source_uri=source_uri,
            parent_record_ids=parent_record_ids,
        )
        composite_issues.extend(anomaly_overlay_res.issues)

    # 4. Apply Event Markers
    if events is not None:
        event_overlay_res = add_event_markers(
            fig=fig,
            events=events,
            source_uri=source_uri,
            parent_record_ids=parent_record_ids,
        )
        composite_issues.extend(event_overlay_res.issues)

    # 5. Determine Overall Result Status
    has_error = any(i.severity == IssueSeverity.ERROR for i in composite_issues)
    has_warning = any(i.severity == IssueSeverity.WARNING for i in composite_issues)
    if has_error:
        overall_status = ResultStatus.ERROR
    elif has_warning:
        overall_status = (
            ResultStatus.WARNING
            if base_result.status != ResultStatus.PARTIAL
            else ResultStatus.PARTIAL
        )
    else:
        overall_status = base_result.status

    return TimeSeriesPlotResult(
        status=overall_status,
        parameter_id=parameter_id,
        figure=fig,
        issues=composite_issues,
        provenance=base_result.provenance,
        epistemic_level=epistemic_level,
        point_count=base_result.point_count,
        non_null_count=base_result.non_null_count,
        unit=unit or base_result.unit,
    )
