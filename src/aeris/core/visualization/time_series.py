"""
AERIS Parameter Time-Series Figure Builder.

Builds interactive, publication-grade Plotly time-series figures for telemetry
parameters with strict unit transparency, epistemic tagging, and source immutability.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus
from aeris.core.visualization.models import (
    RenderMode,
    TimeSeriesPlotConfig,
    TimeSeriesPlotResult,
)
from aeris.core.visualization.theme import (
    AERIS_PALETTE,
    apply_aeris_theme,
    get_aeris_layout,
)


def build_time_series_figure(
    data: pd.DataFrame,
    parameter_id: str,
    timestamp_column: str = "timestamp",
    unit: str | None = None,
    display_name: str | None = None,
    epistemic_level: EpistemicLevel = EpistemicLevel.OBSERVED,
    config: TimeSeriesPlotConfig | None = None,
) -> go.Figure:
    """
    Construct a publication-grade Plotly Figure for a parameter plotted against time.

    Parameters
    ----------
    data : pd.DataFrame
        Input flight data or telemetry slice. Treated as strictly read-only.
    parameter_id : str
        Column name of the parameter to plot.
    timestamp_column : str
        Column name of the time reference (default 'timestamp').
    unit : str | None
        Unit of measurement. If provided, overrides config.unit. Never guessed.
    display_name : str | None
        Human-readable display name. If provided, overrides config.display_name.
    epistemic_level : EpistemicLevel
        Epistemic classification of the signal (OBSERVED or CALCULATED).
    config : TimeSeriesPlotConfig | None
        Configuration options. If None, default settings are applied.

    Returns
    -------
    go.Figure
        A themed, interactive Plotly Figure.

    Raises
    ------
    KeyError
        If parameter_id or timestamp_column is missing from data.columns.
    ValueError
        If require_monotonic_timestamps is True and timestamps are unordered.
    """
    # 1. Resolve configuration with explicit overrides
    active_unit = unit if unit is not None else (config.unit if config else None)
    active_display_name = (
        display_name if display_name is not None else (config.display_name if config else None)
    )
    active_epistemic = (
        epistemic_level
        if epistemic_level is not None
        else (config.epistemic_level if config else EpistemicLevel.OBSERVED)
    )

    if config is None:
        active_config = TimeSeriesPlotConfig(
            unit=active_unit,
            display_name=active_display_name,
            timestamp_column=timestamp_column,
            epistemic_level=active_epistemic,
        )
    else:
        # Create updated config if explicit arguments were supplied
        active_config = config.model_copy(
            update={
                "unit": active_unit,
                "display_name": active_display_name,
                "timestamp_column": timestamp_column,
                "epistemic_level": active_epistemic,
            }
        )

    # 2. Strict Input Validation (without mutating input DataFrame)
    if parameter_id not in data.columns:
        raise KeyError(
            f"Parameter column '{parameter_id}' not found in DataFrame columns: "
            f"{list(data.columns)}"
        )

    if active_config.timestamp_column not in data.columns:
        raise KeyError(
            f"Timestamp column '{active_config.timestamp_column}' not found in "
            f"DataFrame columns: {list(data.columns)}"
        )

    label = active_config.display_name or parameter_id
    u = active_config.unit

    # Construct titles and labels
    if active_config.title:
        chart_title = active_config.title
    elif u:
        chart_title = f"{label} vs Time ({u}) — [{active_config.epistemic_level.value}]"
    else:
        chart_title = f"{label} vs Time — [{active_config.epistemic_level.value}]"

    y_axis_title = f"{label} ({u})" if u else label
    x_axis_title = (
        "Time (UTC)"
        if "time" in active_config.timestamp_column.lower()
        else active_config.timestamp_column
    )

    # 3. Handle Empty DataFrame
    n_points = len(data)
    if n_points == 0:
        fig = go.Figure()
        layout_dict = get_aeris_layout(
            title=chart_title,
            x_title=x_axis_title,
            y_title=y_axis_title,
            height=active_config.height,
            width=active_config.width,
            show_grid=active_config.show_grid,
            dark_mode=active_config.dark_mode,
        )
        fig.update_layout(**layout_dict)
        fig.add_annotation(
            text="No telemetry data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": AERIS_PALETTE["neutral"]},
        )
        fig.layout.meta = {
            "parameter_id": parameter_id,
            "display_name": label,
            "unit": u,
            "epistemic_level": active_config.epistemic_level.value,
            "timestamp_column": active_config.timestamp_column,
            "point_count": 0,
            "non_null_count": 0,
            "status": "EMPTY",
        }
        return fig

    # 4. Monotonicity validation (if explicitly requested, never silently sort)
    if active_config.require_monotonic_timestamps:
        ts_series = data[active_config.timestamp_column]
        # Inspect monotonicity safely
        if not ts_series.is_monotonic_increasing:
            raise ValueError(
                f"Timestamp column '{active_config.timestamp_column}' contains non-monotonic "
                f"or unordered rows. Silent sorting is forbidden by AERIS integrity rules."
            )

    # 5. Determine rendering backend (SVG vs WebGL)
    if active_config.render_mode == RenderMode.WEBGL or (
        active_config.render_mode == RenderMode.AUTO and n_points >= 10000
    ):
        trace_cls = go.Scattergl
    else:
        trace_cls = go.Scatter

    # 6. Determine trace color
    if active_config.line_color:
        trace_color = active_config.line_color
    elif active_config.epistemic_level == EpistemicLevel.CALCULATED:
        trace_color = AERIS_PALETTE["secondary"]
    else:
        trace_color = AERIS_PALETTE["primary"]

    legend_name = f"{label} [{active_config.epistemic_level.value}]"
    hover_unit = f" {u}" if u else ""
    hover_template = (
        f"<b>{label}</b>: %{{y:.4g}}{hover_unit}<br>"
        "<b>Time</b>: %{x}<br>"
        f"<i>[{active_config.epistemic_level.value}]</i>"
        "<extra></extra>"
    )

    trace = trace_cls(
        x=data[active_config.timestamp_column],
        y=data[parameter_id],
        mode=active_config.mode,
        name=legend_name,
        line={"color": trace_color, "width": active_config.line_width},
        marker={"size": active_config.marker_size} if "marker" in active_config.mode else None,
        connectgaps=active_config.connect_gaps,
        hovertemplate=hover_template,
    )

    fig = go.Figure(data=[trace])

    # 7. Apply standard layout and theme
    layout_dict = get_aeris_layout(
        title=chart_title,
        x_title=x_axis_title,
        y_title=y_axis_title,
        height=active_config.height,
        width=active_config.width,
        show_grid=active_config.show_grid,
        dark_mode=active_config.dark_mode,
    )
    fig.update_layout(**layout_dict)

    # 8. Record traceability metadata in layout
    non_null = int(data[parameter_id].notna().sum())
    fig.layout.meta = {
        "parameter_id": parameter_id,
        "display_name": label,
        "unit": u,
        "epistemic_level": active_config.epistemic_level.value,
        "timestamp_column": active_config.timestamp_column,
        "point_count": int(n_points),
        "non_null_count": non_null,
        "render_mode": str(active_config.render_mode.value),
        "connect_gaps": active_config.connect_gaps,
        **active_config.metadata,
    }

    return fig


def generate_time_series_plot(
    data: pd.DataFrame,
    parameter_id: str,
    timestamp_column: str = "timestamp",
    unit: str | None = None,
    display_name: str | None = None,
    epistemic_level: EpistemicLevel = EpistemicLevel.OBSERVED,
    config: TimeSeriesPlotConfig | None = None,
    source_uri: str | None = None,
    parent_record_ids: list[str] | None = None,
) -> TimeSeriesPlotResult:
    """
    Generate a parameter time-series figure with structured issue and provenance reporting.

    Parameters
    ----------
    data : pd.DataFrame
        Input flight data or telemetry slice.
    parameter_id : str
        Column name of the parameter to plot.
    timestamp_column : str
        Column name of the timestamp reference.
    unit : str | None
        Unit of measurement.
    display_name : str | None
        Display name of the parameter.
    epistemic_level : EpistemicLevel
        Epistemic classification of the signal.
    config : TimeSeriesPlotConfig | None
        Plot configuration options.
    source_uri : str | None
        URI or path of the source dataset.
    parent_record_ids : list[str] | None
        Parent provenance record identifiers.

    Returns
    -------
    TimeSeriesPlotResult
        Structured container with the generated Plotly figure, execution status,
        identified issues, and provenance tracking.
    """
    issues: list[Issue] = []
    active_unit = unit if unit is not None else (config.unit if config else None)

    # Validation Checks
    if parameter_id not in data.columns:
        issues.append(
            Issue(
                issue_code="ERR_PARAMETER_NOT_FOUND",
                severity=IssueSeverity.ERROR,
                message=f"Parameter '{parameter_id}' not found in DataFrame columns.",
                field_name=parameter_id,
            )
        )
        empty_fig = go.Figure()
        apply_aeris_theme(empty_fig)
        empty_fig.add_annotation(
            text=f"Parameter '{parameter_id}' not found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": AERIS_PALETTE["critical"]},
        )
        return TimeSeriesPlotResult(
            status=ResultStatus.ERROR,
            parameter_id=parameter_id,
            figure=empty_fig,
            issues=issues,
            epistemic_level=epistemic_level,
            point_count=len(data),
            non_null_count=0,
            unit=active_unit,
        )

    if timestamp_column not in data.columns:
        issues.append(
            Issue(
                issue_code="ERR_TIMESTAMP_COLUMN_NOT_FOUND",
                severity=IssueSeverity.ERROR,
                message=f"Timestamp column '{timestamp_column}' not found in DataFrame columns.",
                field_name=timestamp_column,
            )
        )
        empty_fig = go.Figure()
        apply_aeris_theme(empty_fig)
        empty_fig.add_annotation(
            text=f"Timestamp column '{timestamp_column}' not found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": AERIS_PALETTE["critical"]},
        )
        return TimeSeriesPlotResult(
            status=ResultStatus.ERROR,
            parameter_id=parameter_id,
            figure=empty_fig,
            issues=issues,
            epistemic_level=epistemic_level,
            point_count=len(data),
            non_null_count=0,
            unit=active_unit,
        )

    n_points = len(data)
    if n_points == 0:
        issues.append(
            Issue(
                issue_code="WARN_EMPTY_DATASET",
                severity=IssueSeverity.WARNING,
                message="Input DataFrame contains zero rows.",
            )
        )
        fig = build_time_series_figure(
            data=data,
            parameter_id=parameter_id,
            timestamp_column=timestamp_column,
            unit=active_unit,
            display_name=display_name,
            epistemic_level=epistemic_level,
            config=config,
        )
        return TimeSeriesPlotResult(
            status=ResultStatus.WARNING,
            parameter_id=parameter_id,
            figure=fig,
            issues=issues,
            epistemic_level=epistemic_level,
            point_count=0,
            non_null_count=0,
            unit=active_unit,
        )

    # Missing Values Analysis
    non_null = int(data[parameter_id].notna().sum())
    nan_count = n_points - non_null

    if non_null == 0:
        issues.append(
            Issue(
                issue_code="WARN_ALL_VALUES_MISSING",
                severity=IssueSeverity.WARNING,
                message=(
                    f"All {n_points} observations for parameter '{parameter_id}' are NaN or null."
                ),
                field_name=parameter_id,
            )
        )
        status = ResultStatus.PARTIAL
    elif nan_count > 0:
        nan_pct = (nan_count / n_points) * 100.0
        issues.append(
            Issue(
                issue_code="INFO_MISSING_VALUES",
                severity=IssueSeverity.INFO,
                message=(
                    f"Parameter '{parameter_id}' contains {nan_count} missing values "
                    f"({nan_pct:.1f}% of {n_points} points)."
                ),
                field_name=parameter_id,
            )
        )
        status = ResultStatus.SUCCESS
    else:
        status = ResultStatus.SUCCESS

    # Try building figure
    try:
        fig = build_time_series_figure(
            data=data,
            parameter_id=parameter_id,
            timestamp_column=timestamp_column,
            unit=active_unit,
            display_name=display_name,
            epistemic_level=epistemic_level,
            config=config,
        )
    except ValueError as exc:
        issues.append(
            Issue(
                issue_code="ERR_FIGURE_BUILD_FAILED",
                severity=IssueSeverity.ERROR,
                message=str(exc),
                field_name=parameter_id,
            )
        )
        empty_fig = go.Figure()
        apply_aeris_theme(empty_fig)
        empty_fig.add_annotation(
            text=f"Plot generation failed: {exc}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 13, "color": AERIS_PALETTE["critical"]},
        )
        return TimeSeriesPlotResult(
            status=ResultStatus.ERROR,
            parameter_id=parameter_id,
            figure=empty_fig,
            issues=issues,
            epistemic_level=epistemic_level,
            point_count=n_points,
            non_null_count=non_null,
            unit=active_unit,
        )

    # Attach provenance
    prov = create_provenance(
        activity_name="time_series_figure_generation",
        source_uri=source_uri or "in_memory",
        epistemic_level=epistemic_level,
        parent_record_ids=parent_record_ids or [],
        parameters={
            "parameter_id": parameter_id,
            "timestamp_column": timestamp_column,
            "unit": active_unit,
            "point_count": n_points,
            "non_null_count": non_null,
        },
    )

    return TimeSeriesPlotResult(
        status=status,
        parameter_id=parameter_id,
        figure=fig,
        issues=issues,
        provenance=prov,
        epistemic_level=epistemic_level,
        point_count=n_points,
        non_null_count=non_null,
        unit=active_unit,
    )
