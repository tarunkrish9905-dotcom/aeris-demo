"""
Focused unit and integration tests for the AERIS Visualization Foundation and Time-Series Builder.

Milestone 14, Step 14b Verification Suite.

Covers:
1. TimeSeriesPlotConfig immutability, extra-forbid validation, and default parameters.
2. Aeris Theme, engineering palette, and dark/light layout generation.
3. build_time_series_figure standard rendering, axis titles, and trace structure.
4. Epistemic level tagging (OBSERVED vs CALCULATED) in titles, legends, and styling.
5. Unit transparency (never fabricating units, correctly presenting known units).
6. Source DataFrame immutability (input dataframe is never modified).
7. Observation order preservation and non-monotonic timestamp handling.
8. Monotonicity validation enforcement when explicitly requested.
9. Missing column exception handling (KeyError for parameter and timestamp).
10. Empty dataset handling with informative annotations.
11. Missing and NaN value handling (physical trace gaps, missing value issue reporting).
12. Render mode resolution (SVG Scatter vs WebGL Scattergl thresholding).
13. generate_time_series_plot result container, issue logging, and provenance tracking.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pydantic import ValidationError

from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.results.types import IssueSeverity, ResultStatus
from aeris.core.visualization.models import (
    RenderMode,
    TimeSeriesPlotConfig,
    TimeSeriesPlotResult,
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_telemetry_df() -> pd.DataFrame:
    """Standard monotonic synthetic telemetry dataset."""
    times = pd.date_range("2026-10-15T12:00:00Z", periods=100, freq="1s")
    # Sine wave altitude trace
    altitudes = 1000.0 + 50.0 * np.sin(np.linspace(0, 2 * np.pi, 100))
    airspeeds = 150.0 + 10.0 * np.cos(np.linspace(0, 2 * np.pi, 100))
    return pd.DataFrame(
        {
            "timestamp": times,
            "altitude_ft": altitudes,
            "airspeed_kts": airspeeds,
        }
    )


# ---------------------------------------------------------------------------
# 1. Configuration Model Tests
# ---------------------------------------------------------------------------


class TestTimeSeriesPlotConfig:
    """Tests for TimeSeriesPlotConfig immutability and validation."""

    def test_default_config_properties(self) -> None:
        cfg = TimeSeriesPlotConfig()
        assert cfg.mode == "lines"
        assert cfg.timestamp_column == "timestamp"
        assert cfg.epistemic_level == EpistemicLevel.OBSERVED
        assert cfg.render_mode == RenderMode.AUTO
        assert cfg.connect_gaps is False
        assert cfg.require_monotonic_timestamps is False
        assert cfg.show_grid is True
        assert cfg.dark_mode is False
        assert cfg.unit is None
        assert cfg.display_name is None

    def test_immutability_frozen(self) -> None:
        cfg = TimeSeriesPlotConfig(unit="ft")
        with pytest.raises(ValidationError):
            cfg.unit = "m"  # type: ignore[misc]

    def test_forbid_extra_attributes(self) -> None:
        with pytest.raises(ValidationError):
            TimeSeriesPlotConfig(unknown_field="invalid")  # type: ignore[call-arg]

    def test_custom_valid_config(self) -> None:
        cfg = TimeSeriesPlotConfig(
            title="Custom Altitude Profile",
            display_name="Geometric Altitude",
            unit="ft",
            epistemic_level=EpistemicLevel.CALCULATED,
            mode="lines+markers",
            line_color="#00FF00",
            line_width=2.5,
            render_mode=RenderMode.WEBGL,
            connect_gaps=True,
            require_monotonic_timestamps=True,
            dark_mode=True,
            height=600,
            width=800,
        )
        assert cfg.title == "Custom Altitude Profile"
        assert cfg.display_name == "Geometric Altitude"
        assert cfg.unit == "ft"
        assert cfg.epistemic_level == EpistemicLevel.CALCULATED
        assert cfg.mode == "lines+markers"
        assert cfg.line_color == "#00FF00"
        assert cfg.line_width == 2.5
        assert cfg.render_mode == RenderMode.WEBGL
        assert cfg.connect_gaps is True
        assert cfg.require_monotonic_timestamps is True
        assert cfg.dark_mode is True
        assert cfg.height == 600
        assert cfg.width == 800


# ---------------------------------------------------------------------------
# 2. Theme & Styling Tests
# ---------------------------------------------------------------------------


class TestThemeAndLayout:
    """Tests for AERIS visualization styling and themes."""

    def test_palette_keys(self) -> None:
        required_keys = [
            "primary",
            "secondary",
            "accent",
            "caution",
            "warning",
            "critical",
            "neutral",
            "grid_light",
            "grid_dark",
            "bg_light",
            "paper_light",
            "text_light",
            "bg_dark",
            "paper_dark",
            "text_dark",
        ]
        for key in required_keys:
            assert key in AERIS_PALETTE, f"Missing palette key: {key}"
            assert AERIS_PALETTE[key].startswith("#"), f"Palette {key} must be a hex color"

    def test_default_light_layout(self) -> None:
        layout = get_aeris_layout(
            title="Test Figure",
            x_title="Time (UTC)",
            y_title="Pressure (hPa)",
        )
        assert layout["title"]["text"] == "Test Figure"
        assert layout["font"]["family"] == FONT_FAMILY
        assert layout["plot_bgcolor"] == AERIS_PALETTE["bg_light"]
        assert layout["paper_bgcolor"] == AERIS_PALETTE["paper_light"]
        assert layout["xaxis"]["title"]["text"] == "Time (UTC)"
        assert layout["yaxis"]["title"]["text"] == "Pressure (hPa)"
        assert layout["hovermode"] == "closest"

    def test_dark_mode_layout(self) -> None:
        layout = get_aeris_layout(
            title="Dark Mode Plot",
            dark_mode=True,
        )
        assert layout["plot_bgcolor"] == AERIS_PALETTE["bg_dark"]
        assert layout["paper_bgcolor"] == AERIS_PALETTE["paper_dark"]
        assert layout["font"]["color"] == AERIS_PALETTE["text_dark"]
        assert layout["xaxis"]["gridcolor"] == AERIS_PALETTE["grid_dark"]

    def test_apply_aeris_theme_to_figure(self) -> None:
        fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])
        apply_aeris_theme(fig, dark_mode=False)
        assert fig.layout.plot_bgcolor == AERIS_PALETTE["bg_light"]
        assert fig.layout.font.family == FONT_FAMILY


# ---------------------------------------------------------------------------
# 3. Figure Builder (build_time_series_figure) Tests
# ---------------------------------------------------------------------------


class TestBuildTimeSeriesFigure:
    """Tests for core build_time_series_figure function."""

    def test_standard_figure_structure(self, sample_telemetry_df: pd.DataFrame) -> None:
        fig = build_time_series_figure(
            data=sample_telemetry_df,
            parameter_id="altitude_ft",
            unit="ft",
            display_name="Pressure Altitude",
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        trace = fig.data[0]
        assert isinstance(trace, go.Scatter)
        assert trace.mode == "lines"
        assert "Pressure Altitude" in trace.name
        assert "[OBSERVED]" in trace.name
        assert len(trace.x) == 100
        assert len(trace.y) == 100
        # Axis titles
        assert fig.layout.yaxis.title.text == "Pressure Altitude (ft)"
        assert fig.layout.xaxis.title.text == "Time (UTC)"
        # Metadata check
        assert fig.layout.meta["parameter_id"] == "altitude_ft"
        assert fig.layout.meta["unit"] == "ft"
        assert fig.layout.meta["point_count"] == 100

    def test_dataframe_immutability(self, sample_telemetry_df: pd.DataFrame) -> None:
        """Verify that build_time_series_figure never alters the input DataFrame."""
        df_copy = sample_telemetry_df.copy(deep=True)
        build_time_series_figure(
            data=sample_telemetry_df,
            parameter_id="altitude_ft",
            unit="ft",
        )
        pd.testing.assert_frame_equal(sample_telemetry_df, df_copy)

    def test_unit_transparency_without_fabrication(self, sample_telemetry_df: pd.DataFrame) -> None:
        """When unit is None, no unit in parentheses must be invented."""
        fig = build_time_series_figure(
            data=sample_telemetry_df,
            parameter_id="altitude_ft",
            unit=None,
        )
        assert fig.layout.yaxis.title.text == "altitude_ft"
        assert "(" not in fig.layout.yaxis.title.text
        assert ")" not in fig.layout.yaxis.title.text
        assert fig.layout.meta["unit"] is None

    def test_epistemic_level_calculated(self, sample_telemetry_df: pd.DataFrame) -> None:
        """Calculated signals must be tagged [CALCULATED] and use secondary palette by default."""
        fig = build_time_series_figure(
            data=sample_telemetry_df,
            parameter_id="airspeed_kts",
            unit="kts",
            epistemic_level=EpistemicLevel.CALCULATED,
        )
        trace = fig.data[0]
        assert "[CALCULATED]" in trace.name
        assert "[CALCULATED]" in fig.layout.title.text
        assert trace.line.color == AERIS_PALETTE["secondary"]
        assert fig.layout.meta["epistemic_level"] == "CALCULATED"

    def test_observation_order_preserved_without_sorting(self) -> None:
        """Telemetry must be plotted in exact observation order without silent reordering."""
        unordered_times = [
            datetime.datetime(2026, 10, 15, 12, 0, 5),
            datetime.datetime(2026, 10, 15, 12, 0, 1),
            datetime.datetime(2026, 10, 15, 12, 0, 3),
        ]
        values = [500.0, 100.0, 300.0]
        df = pd.DataFrame({"timestamp": unordered_times, "val": values})

        # By default (require_monotonic_timestamps=False), observation order is preserved
        fig = build_time_series_figure(data=df, parameter_id="val")
        trace = fig.data[0]
        assert list(trace.y) == [500.0, 100.0, 300.0]

    def test_monotonicity_enforcement_raises(self) -> None:
        """When require_monotonic_timestamps=True, unordered timestamps must raise ValueError."""
        unordered_times = [
            datetime.datetime(2026, 10, 15, 12, 0, 5),
            datetime.datetime(2026, 10, 15, 12, 0, 1),
        ]
        df = pd.DataFrame({"timestamp": unordered_times, "val": [1.0, 2.0]})
        cfg = TimeSeriesPlotConfig(require_monotonic_timestamps=True)

        with pytest.raises(ValueError, match="non-monotonic or unordered"):
            build_time_series_figure(data=df, parameter_id="val", config=cfg)

    def test_missing_parameter_raises_key_error(self, sample_telemetry_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="Parameter column 'missing_col' not found"):
            build_time_series_figure(data=sample_telemetry_df, parameter_id="missing_col")

    def test_missing_timestamp_raises_key_error(self, sample_telemetry_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="Timestamp column 'missing_ts' not found"):
            build_time_series_figure(
                data=sample_telemetry_df,
                parameter_id="altitude_ft",
                timestamp_column="missing_ts",
            )

    def test_empty_dataframe_handling(self) -> None:
        empty_df = pd.DataFrame(columns=["timestamp", "altitude_ft"])
        fig = build_time_series_figure(data=empty_df, parameter_id="altitude_ft")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        assert len(fig.layout.annotations) >= 1
        assert "No telemetry data available" in fig.layout.annotations[0].text
        assert fig.layout.meta["status"] == "EMPTY"
        assert fig.layout.meta["point_count"] == 0

    def test_connect_gaps_policy(self) -> None:
        """Trace connectgaps must default to False so missing data is not silently interpolated."""
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4],
                "val": [10.0, np.nan, 20.0, 30.0],
            }
        )
        fig_default = build_time_series_figure(data=df, parameter_id="val")
        assert fig_default.data[0].connectgaps is False

        # Explicit override
        cfg = TimeSeriesPlotConfig(connect_gaps=True)
        fig_connected = build_time_series_figure(data=df, parameter_id="val", config=cfg)
        assert fig_connected.data[0].connectgaps is True

    def test_render_mode_selection(self) -> None:
        """Verify Scatter (SVG) vs Scattergl (WebGL) selection."""
        # Small dataset (< 10,000 points) with AUTO -> Scatter
        df_small = pd.DataFrame({"timestamp": [1, 2], "val": [10.0, 20.0]})
        fig_auto = build_time_series_figure(data=df_small, parameter_id="val")
        assert isinstance(fig_auto.data[0], go.Scatter)

        # Explicit WEBGL mode -> Scattergl
        cfg_webgl = TimeSeriesPlotConfig(render_mode=RenderMode.WEBGL)
        fig_webgl = build_time_series_figure(data=df_small, parameter_id="val", config=cfg_webgl)
        assert isinstance(fig_webgl.data[0], go.Scattergl)

        # Explicit SVG mode -> Scatter
        cfg_svg = TimeSeriesPlotConfig(render_mode=RenderMode.SVG)
        fig_svg = build_time_series_figure(data=df_small, parameter_id="val", config=cfg_svg)
        assert isinstance(fig_svg.data[0], go.Scatter)


# ---------------------------------------------------------------------------
# 4. Result Generator (generate_time_series_plot) Tests
# ---------------------------------------------------------------------------


class TestGenerateTimeSeriesPlot:
    """Tests for structured generate_time_series_plot and issue reporting."""

    def test_successful_plot_generation(self, sample_telemetry_df: pd.DataFrame) -> None:
        res = generate_time_series_plot(
            data=sample_telemetry_df,
            parameter_id="altitude_ft",
            unit="ft",
            display_name="Altitude",
            source_uri="telemetry://flight_001.parquet",
        )
        assert isinstance(res, TimeSeriesPlotResult)
        assert res.status == ResultStatus.SUCCESS
        assert res.parameter_id == "altitude_ft"
        assert res.unit == "ft"
        assert res.point_count == 100
        assert res.non_null_count == 100
        assert res.null_count == 0
        assert len(res.issues) == 0
        assert res.provenance is not None
        assert res.provenance.activity_name == "time_series_figure_generation"
        assert res.provenance.source_uri == "telemetry://flight_001.parquet"
        assert res.provenance.parameters["point_count"] == 100

    def test_missing_parameter_column_issue(self, sample_telemetry_df: pd.DataFrame) -> None:
        res = generate_time_series_plot(
            data=sample_telemetry_df,
            parameter_id="non_existent_param",
        )
        assert res.status == ResultStatus.ERROR
        assert len(res.issues) == 1
        issue = res.issues[0]
        assert issue.issue_code == "ERR_PARAMETER_NOT_FOUND"
        assert issue.severity == IssueSeverity.ERROR
        assert "not found" in issue.message

    def test_missing_timestamp_column_issue(self, sample_telemetry_df: pd.DataFrame) -> None:
        res = generate_time_series_plot(
            data=sample_telemetry_df,
            parameter_id="altitude_ft",
            timestamp_column="missing_ts",
        )
        assert res.status == ResultStatus.ERROR
        assert len(res.issues) == 1
        issue = res.issues[0]
        assert issue.issue_code == "ERR_TIMESTAMP_COLUMN_NOT_FOUND"
        assert issue.severity == IssueSeverity.ERROR

    def test_empty_dataset_issue(self) -> None:
        empty_df = pd.DataFrame(columns=["timestamp", "param"])
        res = generate_time_series_plot(data=empty_df, parameter_id="param")
        assert res.status == ResultStatus.WARNING
        assert len(res.issues) == 1
        assert res.issues[0].issue_code == "WARN_EMPTY_DATASET"
        assert res.issues[0].severity == IssueSeverity.WARNING
        assert res.point_count == 0

    def test_missing_values_detection(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4, 5],
                "val": [10.0, np.nan, 30.0, np.nan, 50.0],
            }
        )
        res = generate_time_series_plot(data=df, parameter_id="val")
        assert res.status == ResultStatus.SUCCESS
        assert res.point_count == 5
        assert res.non_null_count == 3
        assert res.null_count == 2
        assert len(res.issues) == 1
        assert res.issues[0].issue_code == "INFO_MISSING_VALUES"
        assert "2 missing values (40.0% of 5 points)" in res.issues[0].message

    def test_all_missing_values_warning(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3],
                "val": [np.nan, np.nan, np.nan],
            }
        )
        res = generate_time_series_plot(data=df, parameter_id="val")
        assert res.status == ResultStatus.PARTIAL
        assert res.point_count == 3
        assert res.non_null_count == 0
        assert res.null_count == 3
        assert len(res.issues) == 1
        assert res.issues[0].issue_code == "WARN_ALL_VALUES_MISSING"
        assert res.issues[0].severity == IssueSeverity.WARNING

    def test_monotonic_check_failure_issue(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": [10, 5, 20],
                "val": [1.0, 2.0, 3.0],
            }
        )
        cfg = TimeSeriesPlotConfig(require_monotonic_timestamps=True)
        res = generate_time_series_plot(data=df, parameter_id="val", config=cfg)
        assert res.status == ResultStatus.ERROR
        assert len(res.issues) == 1
        assert res.issues[0].issue_code == "ERR_FIGURE_BUILD_FAILED"
        assert "non-monotonic" in res.issues[0].message
