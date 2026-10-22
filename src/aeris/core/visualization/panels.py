"""
AERIS Multi-Signal Telemetry Panel Builder.

Coordinates multiple telemetry parameters into a unified, publication-grade Plotly
visualization with synchronized timelines, physical unit separation, and contextual overlays.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aeris.core.anomaly.models import AnomalyRecord, AnomalySeverity
from aeris.core.limits.models import LimitSeverity, LimitType
from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus
from aeris.core.test_operations.test_point_observation import TestPointObservation
from aeris.core.visualization.models import (
    EngineeringEvent,
    MultiSignalPanelConfig,
    MultiSignalPanelResult,
    PanelLayoutMode,
    RenderMode,
    SignalSpec,
)
from aeris.core.visualization.overlays import (
    ANOMALY_SEVERITY_STYLES,
    LIMIT_SEVERITY_STYLES,
)
from aeris.core.visualization.theme import (
    AERIS_PALETTE,
    FONT_FAMILY,
    apply_aeris_theme,
)

# ---------------------------------------------------------------------------
# Standard Multi-Trace Color Progression
# ---------------------------------------------------------------------------

SIGNAL_COLOR_CYCLE: list[str] = [
    "#1D4ED8",  # Primary Blue
    "#0D9488",  # Teal
    "#7C3AED",  # Purple
    "#EA580C",  # Orange-Red
    "#2563EB",  # Royal Blue
    "#059669",  # Emerald
    "#D97706",  # Amber
    "#4F46E5",  # Indigo
]


# ---------------------------------------------------------------------------
# Multi-Signal Panel Builders
# ---------------------------------------------------------------------------


def build_multi_signal_panel(
    data: pd.DataFrame,
    signals: Sequence[SignalSpec | str],
    timestamp_column: str = "timestamp",
    config: MultiSignalPanelConfig | None = None,
    events: Sequence[TestPointObservation | EngineeringEvent] | None = None,
) -> go.Figure:
    """
    Construct a coordinated multi-signal Plotly figure across subplots.

    Parameters
    ----------
    data : pd.DataFrame
        Input flight data (read-only; never mutated).
    signals : Sequence[SignalSpec | str]
        List of signal specifications or parameter IDs to render.
    timestamp_column : str
        Column name of the timestamp reference.
    config : MultiSignalPanelConfig | None
        Panel layout and styling options.
    events : Sequence[TestPointObservation | EngineeringEvent] | None
        Optional global flight test events to overlay across all subplot rows.

    Returns
    -------
    go.Figure
        A publication-grade multi-signal Plotly figure.

    Raises
    ------
    ValueError
        If signals list is empty or timestamps are non-monotonic when configured.
    KeyError
        If timestamp_column is missing from the DataFrame.
    """
    active_config = config or MultiSignalPanelConfig(timestamp_column=timestamp_column)
    ts_col = active_config.timestamp_column or timestamp_column

    # 1. Input Validation
    if not signals:
        raise ValueError("At least one signal must be specified for multi-signal panel.")

    if ts_col not in data.columns:
        raise KeyError(
            f"Timestamp column '{ts_col}' not found in DataFrame columns: {list(data.columns)}"
        )

    # 2. Monotonicity Enforcement
    if active_config.require_monotonic_timestamps and not data[ts_col].is_monotonic_increasing:
        raise ValueError(
            f"Timestamp column '{ts_col}' contains non-monotonic or unordered rows. "
            f"Silent sorting is forbidden by AERIS integrity rules."
        )

    # 3. Normalize Signals
    normalized_signals: list[SignalSpec] = []
    for s in signals:
        if isinstance(s, str):
            normalized_signals.append(SignalSpec(parameter_id=s))
        else:
            normalized_signals.append(s)

    # Filter signals available in DataFrame
    valid_signals = [s for s in normalized_signals if s.parameter_id in data.columns]
    if not valid_signals:
        raise KeyError(
            f"None of the requested signals {[s.parameter_id for s in normalized_signals]} "
            f"were found in DataFrame columns: {list(data.columns)}"
        )

    # 4. Determine Subplot Layout and Groupings
    layout_mode = active_config.layout_mode

    # Check for unit compatibility in OVERLAY mode
    if layout_mode == PanelLayoutMode.OVERLAY:
        distinct_units = {s.unit for s in valid_signals}
        if len(distinct_units) > 1:
            # Different units cannot be placed on the same numerical axis!
            # Gracefully fall back to STACKED layout
            layout_mode = PanelLayoutMode.STACKED

    # Build row mapping: list of lists of SignalSpec per row
    rows: list[list[SignalSpec]] = []
    if layout_mode == PanelLayoutMode.STACKED:
        for sig in valid_signals:
            rows.append([sig])
    elif layout_mode == PanelLayoutMode.UNIT_GROUPED:
        grouped: dict[str | None, list[SignalSpec]] = {}
        for sig in valid_signals:
            grouped.setdefault(sig.unit, []).append(sig)
        rows = list(grouped.values())
    elif layout_mode == PanelLayoutMode.OVERLAY:
        rows = [valid_signals]

    n_rows = len(rows)
    total_height = active_config.row_height * n_rows + 100
    if active_config.width:
        fig_width: int | None = active_config.width
    else:
        fig_width = None

    # 5. Create Subplots Figure
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=active_config.shared_xaxes,
        vertical_spacing=active_config.vertical_spacing,
    )

    n_points = len(data)

    # 6. Render Traces per Row
    for row_idx, row_signals in enumerate(rows, start=1):
        # Determine shared Y-axis label for this row
        if len(row_signals) == 1:
            sig = row_signals[0]
            label = sig.display_name or sig.parameter_id
            u = sig.unit
            y_title = f"{label} ({u})" if u else label
        else:
            shared_u = row_signals[0].unit
            if all(s.unit == shared_u for s in row_signals) and shared_u:
                y_title = f"Multiple Signals ({shared_u})"
            else:
                y_title = "Multiple Signals"

        # Set Y-axis properties for this row
        fig.update_yaxes(
            title_text=y_title,
            title_font={"size": 11},
            row=row_idx,
            col=1,
            showgrid=active_config.show_grid,
            zeroline=False,
            ticks="outside",
        )

        for sig_idx, sig in enumerate(row_signals):
            pid = sig.parameter_id
            label = sig.display_name or pid
            u = sig.unit
            color_idx = (row_idx - 1 + sig_idx) % len(SIGNAL_COLOR_CYCLE)
            trace_color = sig.line_color or SIGNAL_COLOR_CYCLE[color_idx]

            # Epistemic naming in legend
            legend_name = f"{label} [{sig.epistemic_level.value}]"
            hover_unit = f" {u}" if u else ""
            hover_template = (
                f"<b>{label}</b>: %{{y:.4g}}{hover_unit}<br>"
                "<b>Time</b>: %{x}<br>"
                f"<i>[{sig.epistemic_level.value}]</i>"
                "<extra></extra>"
            )

            # Determine backend
            if active_config.render_mode == RenderMode.WEBGL or (
                active_config.render_mode == RenderMode.AUTO and n_points >= 10000
            ):
                trace_cls = go.Scattergl
            else:
                trace_cls = go.Scatter

            trace = trace_cls(
                x=data[ts_col],
                y=data[pid],
                mode=sig.mode,
                name=legend_name,
                line={"color": trace_color, "width": sig.line_width},
                connectgaps=active_config.connect_gaps,
                hovertemplate=hover_template,
            )

            fig.add_trace(trace, row=row_idx, col=1)

            # 7. Apply Parameter-Specific Limits to This Subplot
            if sig.limits is not None:
                _apply_signal_limits_to_subplot(
                    fig=fig,
                    limits=sig.limits,
                    parameter_id=pid,
                    signal_unit=u,
                    row=row_idx,
                )

            # 8. Apply Parameter-Specific Anomalies to This Subplot
            if sig.anomalies is not None:
                _apply_signal_anomalies_to_subplot(
                    fig=fig,
                    anomalies=sig.anomalies,
                    parameter_id=pid,
                    row=row_idx,
                )

    # 9. Set Shared X-Axis Title on Bottom Subplot
    x_axis_title = "Time (UTC)" if "time" in ts_col.lower() else ts_col
    fig.update_xaxes(
        title_text=x_axis_title,
        title_font={"size": 11},
        row=n_rows,
        col=1,
        showgrid=active_config.show_grid,
        ticks="outside",
    )

    # 10. Apply Global Events Across All Subplot Rows
    active_events = events or active_config.events
    if active_events is not None:
        _apply_global_events_to_panel(fig=fig, events=active_events, n_rows=n_rows)

    # 11. Apply Layout Theme
    title_text = active_config.title or "AERIS Multi-Signal Flight Telemetry Panel"
    fig.update_layout(
        title={
            "text": title_text,
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 14, "family": FONT_FAMILY},
        },
        height=total_height,
        hovermode="x unified" if active_config.shared_xaxes else "closest",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 10},
        },
    )
    if fig_width:
        fig.update_layout(width=fig_width)

    apply_aeris_theme(fig, dark_mode=active_config.dark_mode, show_grid=active_config.show_grid)

    # 12. Record Traceability Metadata
    fig.layout.meta = {
        "panel_title": title_text,
        "timestamp_column": ts_col,
        "layout_mode": str(layout_mode.value),
        "row_count": n_rows,
        "point_count": int(n_points),
        "requested_signals": [s.parameter_id for s in normalized_signals],
        "rendered_signals": [s.parameter_id for s in valid_signals],
        **active_config.metadata,
    }

    return fig


def generate_multi_signal_panel(
    data: pd.DataFrame,
    signals: Sequence[SignalSpec | str],
    timestamp_column: str = "timestamp",
    config: MultiSignalPanelConfig | None = None,
    events: Sequence[TestPointObservation | EngineeringEvent] | None = None,
    source_uri: str | None = None,
    parent_record_ids: list[str] | None = None,
) -> MultiSignalPanelResult:
    """
    Generate a multi-signal panel with comprehensive issue reporting and provenance.

    Parameters
    ----------
    data : pd.DataFrame
        Input flight data (read-only; never mutated).
    signals : Sequence[SignalSpec | str]
        List of signal specifications or parameter IDs.
    timestamp_column : str
        Name of timestamp column.
    config : MultiSignalPanelConfig | None
        Panel configuration options.
    events : Sequence[TestPointObservation | EngineeringEvent] | None
        Global flight test events.
    source_uri : str | None
        Source dataset URI for provenance.
    parent_record_ids : list[str] | None
        Parent provenance identifiers.

    Returns
    -------
    MultiSignalPanelResult
        Structured container with generated figure, status, and all issues.
    """
    issues: list[Issue] = []
    active_config = config or MultiSignalPanelConfig(timestamp_column=timestamp_column)
    ts_col = active_config.timestamp_column or timestamp_column

    # 1. Empty Signals Selection Check
    if not signals:
        issues.append(
            Issue(
                issue_code="ERR_EMPTY_SIGNAL_SELECTION",
                severity=IssueSeverity.ERROR,
                message="No signals provided for multi-signal panel visualization.",
            )
        )
        empty_fig = go.Figure()
        apply_aeris_theme(empty_fig)
        return MultiSignalPanelResult(
            status=ResultStatus.ERROR,
            figure=empty_fig,
            signal_ids=[],
            included_signals=[],
            row_count=0,
            point_count=len(data),
            issues=issues,
        )

    # 2. Timestamp Column Check
    if ts_col not in data.columns:
        issues.append(
            Issue(
                issue_code="ERR_TIMESTAMP_COLUMN_NOT_FOUND",
                severity=IssueSeverity.ERROR,
                message=f"Timestamp column '{ts_col}' not found in DataFrame.",
                field_name=ts_col,
            )
        )
        empty_fig = go.Figure()
        apply_aeris_theme(empty_fig)
        return MultiSignalPanelResult(
            status=ResultStatus.ERROR,
            figure=empty_fig,
            signal_ids=[s if isinstance(s, str) else s.parameter_id for s in signals],
            included_signals=[],
            row_count=0,
            point_count=len(data),
            issues=issues,
        )

    # 3. Check for Empty Dataset
    n_points = len(data)
    if n_points == 0:
        issues.append(
            Issue(
                issue_code="WARN_EMPTY_DATASET",
                severity=IssueSeverity.WARNING,
                message="Input DataFrame contains zero telemetry rows.",
            )
        )

    # 4. Normalize and Validate Parameter Columns
    normalized_signals: list[SignalSpec] = []
    for s in signals:
        if isinstance(s, str):
            normalized_signals.append(SignalSpec(parameter_id=s))
        else:
            normalized_signals.append(s)

    requested_ids = [s.parameter_id for s in normalized_signals]
    valid_signals: list[SignalSpec] = []
    for sig in normalized_signals:
        if sig.parameter_id in data.columns:
            valid_signals.append(sig)
        else:
            issues.append(
                Issue(
                    issue_code="WARN_PARAMETER_NOT_FOUND",
                    severity=IssueSeverity.WARNING,
                    message=f"Parameter '{sig.parameter_id}' not found in DataFrame.",
                    field_name=sig.parameter_id,
                )
            )

    if not valid_signals:
        issues.append(
            Issue(
                issue_code="ERR_ALL_SIGNALS_MISSING",
                severity=IssueSeverity.ERROR,
                message="None of the requested signals exist in the provided DataFrame.",
            )
        )
        empty_fig = go.Figure()
        apply_aeris_theme(empty_fig)
        return MultiSignalPanelResult(
            status=ResultStatus.ERROR,
            figure=empty_fig,
            signal_ids=requested_ids,
            included_signals=[],
            row_count=0,
            point_count=n_points,
            issues=issues,
        )

    # 5. Check Unit Compatibility in OVERLAY mode
    if active_config.layout_mode == PanelLayoutMode.OVERLAY:
        distinct_units = {s.unit for s in valid_signals}
        if len(distinct_units) > 1:
            issues.append(
                Issue(
                    issue_code="ERR_INCOMPATIBLE_OVERLAY_UNITS",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"Requested OVERLAY mode for signals with incompatible units "
                        f"({distinct_units}). Incompatible quantities cannot share a numerical "
                        f"axis. Falling back to STACKED subplots."
                    ),
                )
            )

    # 6. Build Figure
    try:
        fig = build_multi_signal_panel(
            data=data,
            signals=valid_signals,
            timestamp_column=ts_col,
            config=active_config,
            events=events,
        )
    except ValueError as exc:
        issues.append(
            Issue(
                issue_code="ERR_FIGURE_BUILD_FAILED",
                severity=IssueSeverity.ERROR,
                message=str(exc),
            )
        )
        empty_fig = go.Figure()
        apply_aeris_theme(empty_fig)
        return MultiSignalPanelResult(
            status=ResultStatus.ERROR,
            figure=empty_fig,
            signal_ids=requested_ids,
            included_signals=[],
            row_count=0,
            point_count=n_points,
            issues=issues,
        )

    row_count = fig.layout.meta.get("row_count", len(valid_signals))
    included_ids = [s.parameter_id for s in valid_signals]

    # 7. Attach Provenance
    prov = create_provenance(
        activity_name="multi_signal_panel_generation",
        source_uri=source_uri or "in_memory",
        epistemic_level=EpistemicLevel.OBSERVED,
        parent_record_ids=parent_record_ids or [],
        parameters={
            "requested_signals": requested_ids,
            "included_signals": included_ids,
            "row_count": row_count,
            "point_count": n_points,
        },
    )

    has_error = any(i.severity == IssueSeverity.ERROR for i in issues)
    has_warning = any(i.severity == IssueSeverity.WARNING for i in issues)
    if has_error:
        status = ResultStatus.ERROR
    elif has_warning:
        status = ResultStatus.WARNING
    else:
        status = ResultStatus.SUCCESS

    return MultiSignalPanelResult(
        status=status,
        figure=fig,
        signal_ids=requested_ids,
        included_signals=included_ids,
        row_count=row_count,
        point_count=n_points,
        issues=issues,
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# Internal Subplot Overlay Helpers
# ---------------------------------------------------------------------------


def _apply_signal_limits_to_subplot(
    fig: go.Figure,
    limits: Any,
    parameter_id: str,
    signal_unit: str | None,
    row: int,
) -> None:
    """Apply parameter limits to a specific subplot row."""
    limit_list: list[Any]
    if hasattr(limits, "get_limits_for_parameter"):
        limit_list = list(
            limits.get_limits_for_parameter(parameter_id, include_all_configurations=True)
        )
    elif isinstance(limits, Sequence):
        limit_list = list(limits)
    else:
        limit_list = [limits]

    for lim in limit_list:
        lim_unit = getattr(lim, "unit", getattr(lim, "limit_unit", None))
        # Unit Safety Gate
        if signal_unit is not None and lim_unit is not None and signal_unit != lim_unit:
            # Skip mismatched unit
            continue

        lim_type = getattr(lim, "limit_type", None)
        lim_sev = getattr(lim, "severity", LimitSeverity.WARNING)
        style = LIMIT_SEVERITY_STYLES.get(lim_sev, LIMIT_SEVERITY_STYLES[LimitSeverity.WARNING])

        if lim_type == LimitType.UPPER_BOUND:
            val = getattr(lim, "upper_bound", getattr(lim, "limit_value", None))
            if val is not None:
                fig.add_hline(
                    y=float(val),
                    row=row,
                    col=1,
                    line_dash=style["dash"],
                    line_color=style["color"],
                    line_width=style["width"],
                    annotation_text=f"<b>[LIMIT]</b> Upper {style['tag']}",
                    annotation_position="top right",
                    annotation_font={"size": 9, "color": style["color"]},
                )
        elif lim_type == LimitType.LOWER_BOUND:
            val = getattr(lim, "lower_bound", getattr(lim, "limit_value", None))
            if val is not None:
                fig.add_hline(
                    y=float(val),
                    row=row,
                    col=1,
                    line_dash=style["dash"],
                    line_color=style["color"],
                    line_width=style["width"],
                    annotation_text=f"<b>[LIMIT]</b> Lower {style['tag']}",
                    annotation_position="bottom right",
                    annotation_font={"size": 9, "color": style["color"]},
                )
        elif lim_type == LimitType.BAND:
            lower = getattr(lim, "lower_bound", None)
            upper = getattr(lim, "upper_bound", None)
            if lower is not None and upper is not None and lower <= upper:
                fig.add_hrect(
                    y0=float(lower),
                    y1=float(upper),
                    row=row,
                    col=1,
                    fillcolor=style["color"],
                    opacity=0.12,
                    line_width=1,
                    line_dash=style["dash"],
                    line_color=style["color"],
                    annotation_text=f"<b>[LIMIT]</b> {style['tag']} Band",
                    annotation_position="top left",
                    annotation_font={"size": 9, "color": style["color"]},
                )


def _apply_signal_anomalies_to_subplot(
    fig: go.Figure,
    anomalies: Sequence[AnomalyRecord],
    parameter_id: str,
    row: int,
) -> None:
    """Apply parameter-specific anomalies to a specific subplot row."""
    for anomaly in anomalies:
        if anomaly.target_column != parameter_id:
            continue

        style = ANOMALY_SEVERITY_STYLES.get(
            anomaly.severity, ANOMALY_SEVERITY_STYLES[AnomalySeverity.HIGH]
        )

        if anomaly.start_timestamp is not None and anomaly.end_timestamp is not None:
            if anomaly.start_timestamp <= anomaly.end_timestamp:
                fig.add_vrect(
                    x0=anomaly.start_timestamp,
                    x1=anomaly.end_timestamp,
                    row=row,
                    col=1,
                    fillcolor=style["color"],
                    opacity=0.18,
                    line_dash="dot",
                    line_color=style["color"],
                    line_width=1.5,
                )
        elif anomaly.timestamp is not None:
            obs = anomaly.observed_value
            if obs is not None and not isinstance(obs, str):
                fig.add_trace(
                    go.Scatter(
                        x=[anomaly.timestamp],
                        y=[float(obs)],
                        mode="markers",
                        name=f"[ANOMALY: {style['tag']}] {anomaly.anomaly_id}",
                        marker={
                            "symbol": style["symbol"],
                            "size": style["size"],
                            "color": style["color"],
                            "line": {"color": "#FFFFFF", "width": 1},
                        },
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )
            else:
                fig.add_vline(
                    x=anomaly.timestamp,
                    row=row,
                    col=1,
                    line_dash="dash",
                    line_color=style["color"],
                    line_width=1.5,
                )


def _apply_global_events_to_panel(
    fig: go.Figure,
    events: Sequence[Any],
    n_rows: int,
) -> None:
    """Apply global flight test events across all subplot rows."""
    color = AERIS_PALETTE["accent"]

    for evt in events:
        if isinstance(evt, TestPointObservation):
            if evt.start_timestamp <= evt.end_timestamp:
                fig.add_vrect(
                    x0=evt.start_timestamp,
                    x1=evt.end_timestamp,
                    row="all",
                    col=1,
                    fillcolor=color,
                    opacity=0.10,
                    line_dash="dot",
                    line_color=color,
                    line_width=1,
                    annotation_text=f"<b>[TEST POINT]</b> {evt.test_point_id}",
                    annotation_position="top left",
                    annotation_font={"size": 9, "color": color},
                )
        elif isinstance(evt, EngineeringEvent):
            if evt.start_timestamp is not None and evt.end_timestamp is not None:
                if evt.start_timestamp <= evt.end_timestamp:
                    fig.add_vrect(
                        x0=evt.start_timestamp,
                        x1=evt.end_timestamp,
                        row="all",
                        col=1,
                        fillcolor=color,
                        opacity=0.10,
                        line_dash="dashdot",
                        line_color=color,
                        line_width=1,
                        annotation_text=f"<b>[EVENT]</b> {evt.name}",
                        annotation_position="top left",
                        annotation_font={"size": 9, "color": color},
                    )
            elif evt.timestamp is not None:
                fig.add_vline(
                    x=evt.timestamp,
                    row="all",
                    col=1,
                    line_dash="dashdot",
                    line_color=color,
                    line_width=1.5,
                    annotation_text=f"<b>[EVENT]</b> {evt.name}",
                    annotation_position="top",
                    annotation_font={"size": 9, "color": color},
                )
