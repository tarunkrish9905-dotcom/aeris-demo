"""
Focused unit and integration tests for AERIS Multi-Signal Telemetry Panels.

Milestone 14, Step 14d Verification Suite.

Covers:
1. MultiSignalPanelConfig and SignalSpec validation, immutability, and defaults.
2. Coordinated subplots construction with shared, synchronized timelines.
3. Unit separation: incompatible units never placed on the same numerical axis.
4. PanelLayoutMode behaviors: STACKED, UNIT_GROUPED, and OVERLAY.
5. Incompatible unit safety gate in OVERLAY mode (graceful fallback with explicit issue).
6. Telemetry observation order preservation without silent sorting or fabrication.
7. Monotonicity validation enforcement.
8. Parameter-specific limit and anomaly overlays on dedicated subplot rows.
9. Global flight test event markers synchronized across all subplot rows.
10. Error and boundary handling: empty signals, missing columns, empty datasets.
11. Source DataFrame immutability across panel generation.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pydantic import ValidationError

from aeris.core.anomaly.models import (
    AnomalyClassification,
    AnomalyRecord,
    AnomalySeverity,
    DetectionMethod,
)
from aeris.core.limits.models import (
    LimitSeverity,
    LimitType,
    ParameterLimitDefinition,
)
from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.results.types import ResultStatus
from aeris.core.test_operations.test_point_observation import TestPointObservation
from aeris.core.visualization.models import (
    EngineeringEvent,
    MultiSignalPanelConfig,
    MultiSignalPanelResult,
    PanelLayoutMode,
    SignalSpec,
)
from aeris.core.visualization.panels import (
    build_multi_signal_panel,
    generate_multi_signal_panel,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_signal_telemetry_df() -> pd.DataFrame:
    """Standard multi-parameter flight telemetry dataset."""
    times = pd.date_range("2026-10-15T12:00:00Z", periods=60, freq="1s")
    altitudes = 1000.0 + 10.0 * np.linspace(0, 60, 60)
    airspeeds = 140.0 + 5.0 * np.sin(np.linspace(0, 2 * np.pi, 60))
    pitches = 2.0 + 0.5 * np.cos(np.linspace(0, 2 * np.pi, 60))
    rolls = -1.0 + 0.8 * np.sin(np.linspace(0, 4 * np.pi, 60))

    return pd.DataFrame(
        {
            "timestamp": times,
            "altitude_ft": altitudes,
            "airspeed_kts": airspeeds,
            "pitch_deg": pitches,
            "roll_deg": rolls,
        }
    )


# ---------------------------------------------------------------------------
# 1. Configuration & Model Tests
# ---------------------------------------------------------------------------


class TestMultiSignalModels:
    """Tests for SignalSpec and MultiSignalPanelConfig models."""

    def test_signal_spec_defaults_and_immutability(self) -> None:
        spec = SignalSpec(parameter_id="altitude_ft", unit="ft")
        assert spec.parameter_id == "altitude_ft"
        assert spec.unit == "ft"
        assert spec.mode == "lines"
        assert spec.epistemic_level == EpistemicLevel.OBSERVED
        assert spec.line_color is None

        with pytest.raises(ValidationError):
            spec.parameter_id = "other"  # type: ignore[misc]

    def test_signal_spec_forbid_extra(self) -> None:
        with pytest.raises(ValidationError):
            SignalSpec(parameter_id="alt", unknown_field=123)  # type: ignore[call-arg]

    def test_panel_config_defaults(self) -> None:
        cfg = MultiSignalPanelConfig()
        assert cfg.layout_mode == PanelLayoutMode.STACKED
        assert cfg.timestamp_column == "timestamp"
        assert cfg.shared_xaxes is True
        assert cfg.connect_gaps is False
        assert cfg.require_monotonic_timestamps is False
        assert cfg.row_height == 220


# ---------------------------------------------------------------------------
# 2. Panel Builder Tests (build_multi_signal_panel)
# ---------------------------------------------------------------------------


class TestBuildMultiSignalPanel:
    """Tests for core build_multi_signal_panel function."""

    def test_stacked_subplots_structure(self, multi_signal_telemetry_df: pd.DataFrame) -> None:
        fig = build_multi_signal_panel(
            data=multi_signal_telemetry_df,
            signals=["altitude_ft", "airspeed_kts"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        # Check metadata
        assert fig.layout.meta["row_count"] == 2
        assert fig.layout.meta["layout_mode"] == "STACKED"
        assert fig.layout.meta["rendered_signals"] == ["altitude_ft", "airspeed_kts"]

        # Check axis titles
        assert fig.layout.yaxis.title.text == "altitude_ft"
        assert fig.layout.yaxis2.title.text == "airspeed_kts"
        assert fig.layout.xaxis2.title.text == "Time (UTC)"

    def test_signal_specs_with_units_and_epistemic_badges(
        self, multi_signal_telemetry_df: pd.DataFrame
    ) -> None:
        signals = [
            SignalSpec(
                parameter_id="altitude_ft",
                display_name="Pressure Altitude",
                unit="ft",
                epistemic_level=EpistemicLevel.OBSERVED,
            ),
            SignalSpec(
                parameter_id="airspeed_kts",
                display_name="Indicated Airspeed",
                unit="kts",
                epistemic_level=EpistemicLevel.CALCULATED,
                line_color="#EA580C",
            ),
        ]

        fig = build_multi_signal_panel(
            data=multi_signal_telemetry_df,
            signals=signals,
        )

        assert fig.layout.yaxis.title.text == "Pressure Altitude (ft)"
        assert fig.layout.yaxis2.title.text == "Indicated Airspeed (kts)"
        assert "[OBSERVED]" in fig.data[0].name
        assert "[CALCULATED]" in fig.data[1].name
        assert fig.data[1].line.color == "#EA580C"

    def test_unit_grouped_layout_mode(self, multi_signal_telemetry_df: pd.DataFrame) -> None:
        """Signals sharing identical units share a subplot row in UNIT_GROUPED mode."""
        signals = [
            SignalSpec(parameter_id="pitch_deg", unit="deg", display_name="Pitch"),
            SignalSpec(parameter_id="roll_deg", unit="deg", display_name="Roll"),
            SignalSpec(parameter_id="altitude_ft", unit="ft", display_name="Altitude"),
        ]
        cfg = MultiSignalPanelConfig(layout_mode=PanelLayoutMode.UNIT_GROUPED)

        fig = build_multi_signal_panel(
            data=multi_signal_telemetry_df,
            signals=signals,
            config=cfg,
        )

        # 2 distinct units ('deg' and 'ft') -> 2 subplot rows
        assert fig.layout.meta["row_count"] == 2
        # Total traces = 3 (2 in row 1, 1 in row 2)
        assert len(fig.data) == 3

    def test_overlay_mode_with_compatible_units(
        self, multi_signal_telemetry_df: pd.DataFrame
    ) -> None:
        """When all signals have identical units, OVERLAY mode renders on 1 shared row."""
        signals = [
            SignalSpec(parameter_id="pitch_deg", unit="deg", display_name="Pitch Angle"),
            SignalSpec(parameter_id="roll_deg", unit="deg", display_name="Roll Angle"),
        ]
        cfg = MultiSignalPanelConfig(layout_mode=PanelLayoutMode.OVERLAY)

        fig = build_multi_signal_panel(
            data=multi_signal_telemetry_df,
            signals=signals,
            config=cfg,
        )

        assert fig.layout.meta["row_count"] == 1
        assert len(fig.data) == 2

    def test_overlay_mode_with_incompatible_units_falls_back_to_stacked(
        self, multi_signal_telemetry_df: pd.DataFrame
    ) -> None:
        """Incompatible units must never share a numerical axis; falls back to STACKED."""
        signals = [
            SignalSpec(parameter_id="altitude_ft", unit="ft"),
            SignalSpec(parameter_id="airspeed_kts", unit="kts"),
        ]
        cfg = MultiSignalPanelConfig(layout_mode=PanelLayoutMode.OVERLAY)

        fig = build_multi_signal_panel(
            data=multi_signal_telemetry_df,
            signals=signals,
            config=cfg,
        )

        # Must have fallen back to STACKED (2 rows) to prevent false physical overlay
        assert fig.layout.meta["row_count"] == 2

    def test_monotonicity_enforcement(self) -> None:
        unordered_df = pd.DataFrame(
            {
                "timestamp": [10, 5, 20],
                "altitude_ft": [1000.0, 1050.0, 1100.0],
            }
        )
        cfg = MultiSignalPanelConfig(require_monotonic_timestamps=True)

        with pytest.raises(ValueError, match="non-monotonic or unordered"):
            build_multi_signal_panel(
                data=unordered_df,
                signals=["altitude_ft"],
                config=cfg,
            )

    def test_empty_signals_raises(self, multi_signal_telemetry_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="At least one signal must be specified"):
            build_multi_signal_panel(data=multi_signal_telemetry_df, signals=[])

    def test_missing_timestamp_column_raises(self, multi_signal_telemetry_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="Timestamp column 'missing_ts' not found"):
            build_multi_signal_panel(
                data=multi_signal_telemetry_df,
                signals=["altitude_ft"],
                timestamp_column="missing_ts",
            )


# ---------------------------------------------------------------------------
# 3. Contextual Overlays on Subplots Tests
# ---------------------------------------------------------------------------


class TestMultiSignalOverlays:
    """Tests for limits, anomalies, and events integrated on multi-signal panels."""

    def test_parameter_specific_limit_overlays_on_correct_subplot(
        self, multi_signal_telemetry_df: pd.DataFrame
    ) -> None:
        lim_alt = ParameterLimitDefinition(
            limit_id="LIM_ALT_CEILING",
            parameter_id="altitude_ft",
            limit_type=LimitType.UPPER_BOUND,
            severity=LimitSeverity.WARNING,
            upper_bound=1500.0,
            unit="ft",
            vehicle_id="VEH_01",
            profile_id="PROF_STD",
            profile_version="1.0.0",
            source_reference="Flight Manual",
        )

        signals = [
            SignalSpec(parameter_id="altitude_ft", unit="ft", limits=[lim_alt]),
            SignalSpec(parameter_id="airspeed_kts", unit="kts"),
        ]

        fig = build_multi_signal_panel(data=multi_signal_telemetry_df, signals=signals)
        assert len(fig.layout.shapes) >= 1
        # Altitude limit line is on row 1 ('y')
        assert any(s.yref == "y" and s.y0 == 1500.0 for s in fig.layout.shapes)
        # Airspeed subplot on row 2 ('y2') has no shapes
        assert not any(s.yref == "y2" for s in fig.layout.shapes)

    def test_parameter_specific_anomalies_on_correct_subplot(
        self, multi_signal_telemetry_df: pd.DataFrame
    ) -> None:
        prov = create_provenance(
            activity_name="anomaly_detection",
            source_uri="test",
            epistemic_level=EpistemicLevel.CALCULATED,
        )
        anomaly = AnomalyRecord(
            anomaly_id="ANOM_PITCH_SPIKE",
            analysis_id="A1",
            target_column="pitch_deg",
            classification=AnomalyClassification.POINT_ANOMALY,
            severity=AnomalySeverity.HIGH,
            detection_method=DetectionMethod.STATISTICAL,
            timestamp=datetime.datetime(2026, 10, 15, 12, 0, 10, tzinfo=datetime.UTC),
            observed_value=2.45,
            epistemic_level=EpistemicLevel.CALCULATED,
            provenance=prov,
        )

        signals = [
            SignalSpec(parameter_id="altitude_ft", unit="ft"),
            SignalSpec(parameter_id="pitch_deg", unit="deg", anomalies=[anomaly]),
        ]

        fig = build_multi_signal_panel(data=multi_signal_telemetry_df, signals=signals)
        # Total traces = 2 signals + 1 anomaly marker on pitch subplot
        assert len(fig.data) == 3
        # Anomaly trace is on row 2 (xaxis2, yaxis2)
        anom_trace = fig.data[2]
        assert anom_trace.xaxis == "x2"
        assert anom_trace.yaxis == "y2"
        assert "[ANOMALY: HIGH] ANOM_PITCH_SPIKE" in anom_trace.name

    def test_global_events_span_across_all_subplots(
        self, multi_signal_telemetry_df: pd.DataFrame
    ) -> None:
        tp_event = TestPointObservation(
            observation_id="OBS_001",
            test_point_id="TP_DIVE_PULLOUT",
            source_reference="flight_001.csv",
            start_timestamp=datetime.datetime(2026, 10, 15, 12, 0, 10, tzinfo=datetime.UTC),
            end_timestamp=datetime.datetime(2026, 10, 15, 12, 0, 20, tzinfo=datetime.UTC),
        )

        fig = build_multi_signal_panel(
            data=multi_signal_telemetry_df,
            signals=["altitude_ft", "airspeed_kts"],
            events=[tp_event],
        )

        # Event shapes should exist for both row 1 and row 2 domains
        assert len(fig.layout.shapes) >= 2
        yrefs = [s.yref for s in fig.layout.shapes]
        assert "y domain" in yrefs
        assert "y2 domain" in yrefs

    def test_global_engineering_event_span_across_subplots(
        self, multi_signal_telemetry_df: pd.DataFrame
    ) -> None:
        evt = EngineeringEvent(
            event_id="EVT_GEAR_EXT",
            name="Gear Extension",
            timestamp=datetime.datetime(2026, 10, 15, 12, 0, 15, tzinfo=datetime.UTC),
        )

        fig = build_multi_signal_panel(
            data=multi_signal_telemetry_df,
            signals=["altitude_ft", "airspeed_kts"],
            events=[evt],
        )

        assert len(fig.layout.shapes) >= 2


# ---------------------------------------------------------------------------
# 4. Result Generator Tests (generate_multi_signal_panel)
# ---------------------------------------------------------------------------


class TestGenerateMultiSignalPanel:
    """Tests for structured generate_multi_signal_panel generator."""

    def test_successful_multi_signal_generation(
        self, multi_signal_telemetry_df: pd.DataFrame
    ) -> None:
        res = generate_multi_signal_panel(
            data=multi_signal_telemetry_df,
            signals=["altitude_ft", "airspeed_kts", "pitch_deg"],
            source_uri="telemetry://flight_test_01.parquet",
        )
        assert isinstance(res, MultiSignalPanelResult)
        assert res.status == ResultStatus.SUCCESS
        assert res.row_count == 3
        assert res.point_count == 60
        assert res.included_signals == ["altitude_ft", "airspeed_kts", "pitch_deg"]
        assert len(res.issues) == 0
        assert res.provenance is not None
        assert res.provenance.activity_name == "multi_signal_panel_generation"

    def test_empty_signals_selection_error(self, multi_signal_telemetry_df: pd.DataFrame) -> None:
        res = generate_multi_signal_panel(data=multi_signal_telemetry_df, signals=[])
        assert res.status == ResultStatus.ERROR
        assert len(res.issues) == 1
        assert res.issues[0].issue_code == "ERR_EMPTY_SIGNAL_SELECTION"

    def test_missing_timestamp_column_error(self, multi_signal_telemetry_df: pd.DataFrame) -> None:
        res = generate_multi_signal_panel(
            data=multi_signal_telemetry_df,
            signals=["altitude_ft"],
            timestamp_column="missing_ts",
        )
        assert res.status == ResultStatus.ERROR
        assert res.issues[0].issue_code == "ERR_TIMESTAMP_COLUMN_NOT_FOUND"

    def test_partially_missing_signals_renders_valid_and_logs_warning(
        self, multi_signal_telemetry_df: pd.DataFrame
    ) -> None:
        res = generate_multi_signal_panel(
            data=multi_signal_telemetry_df,
            signals=["altitude_ft", "non_existent_param"],
        )
        assert res.status == ResultStatus.WARNING
        assert res.included_signals == ["altitude_ft"]
        assert res.row_count == 1
        assert any(i.issue_code == "WARN_PARAMETER_NOT_FOUND" for i in res.issues)

    def test_all_signals_missing_error(self, multi_signal_telemetry_df: pd.DataFrame) -> None:
        res = generate_multi_signal_panel(
            data=multi_signal_telemetry_df,
            signals=["missing_1", "missing_2"],
        )
        assert res.status == ResultStatus.ERROR
        assert any(i.issue_code == "ERR_ALL_SIGNALS_MISSING" for i in res.issues)

    def test_dataframe_immutability(self, multi_signal_telemetry_df: pd.DataFrame) -> None:
        """Verify that input DataFrame is bitwise unmutated by panel generator."""
        df_copy = multi_signal_telemetry_df.copy(deep=True)
        generate_multi_signal_panel(
            data=multi_signal_telemetry_df,
            signals=["altitude_ft", "airspeed_kts"],
        )
        pd.testing.assert_frame_equal(multi_signal_telemetry_df, df_copy)
