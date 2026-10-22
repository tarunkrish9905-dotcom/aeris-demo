"""
Focused unit and integration tests for AERIS Visualization Overlays.

Milestone 14, Step 14c Verification Suite.

Covers:
1. Upper and lower limit line overlays with severity-specific dash styles and colors.
2. Engineering limit band / envelope overlays with bounds validation.
3. Unit safety gate: strictly suppresses overlays with incompatible units (no silent conversion).
4. Vehicle identity and configuration-scoped applicability filtering.
5. Parameter mismatch detection and unconfigured limits handling (without false safety implication).
6. Point anomaly markers with custom severity glyphs and epistemic badges.
7. Interval anomaly regions and invalid timestamp handling.
8. Flight test events and TestPointObservation interval overlays.
9. End-to-end generate_overlaid_time_series_plot orchestration.
10. Source DataFrame immutability across overlaid plot generation.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from aeris.core.anomaly.models import (
    AnomalyClassification,
    AnomalyRecord,
    AnomalySeverity,
    DetectionMethod,
)
from aeris.core.limits.models import (
    EngineeringLimitProfile,
    LimitSeverity,
    LimitType,
    ParameterLimitDefinition,
)
from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.results.types import ResultStatus
from aeris.core.test_operations.test_point_observation import TestPointObservation
from aeris.core.vehicle_evaluation.models import EvaluationOutcome, LimitEvaluationRecord
from aeris.core.visualization.models import EngineeringEvent, OverlayResult, TimeSeriesPlotResult
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
from aeris.core.visualization.theme import AERIS_PALETTE
from aeris.core.visualization.time_series import build_time_series_figure

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def telemetry_df() -> pd.DataFrame:
    """Monotonic synthetic telemetry dataframe."""
    times = pd.date_range("2026-10-15T12:00:00Z", periods=50, freq="1s")
    altitudes = 1000.0 + np.linspace(0, 100, 50)
    airspeeds = 150.0 + 5.0 * np.sin(np.linspace(0, np.pi, 50))
    return pd.DataFrame(
        {
            "timestamp": times,
            "altitude_ft": altitudes,
            "airspeed_kts": airspeeds,
        }
    )


@pytest.fixture
def base_altitude_fig(telemetry_df: pd.DataFrame) -> go.Figure:
    """Base Plotly figure with altitude telemetry."""
    return build_time_series_figure(
        data=telemetry_df,
        parameter_id="altitude_ft",
        unit="ft",
        display_name="Altitude",
    )


@pytest.fixture
def sample_limit_profile() -> EngineeringLimitProfile:
    """Authoritative limit profile for testing."""
    lim_upper = ParameterLimitDefinition(
        limit_id="LIM_ALT_MAX",
        parameter_id="altitude_ft",
        limit_type=LimitType.UPPER_BOUND,
        severity=LimitSeverity.WARNING,
        upper_bound=1080.0,
        unit="ft",
        vehicle_id="VEH_AERIS_01",
        profile_id="PROF_AERIS_STD",
        profile_version="1.0.0",
        source_reference="DO-160 Section 4",
    )
    lim_lower = ParameterLimitDefinition(
        limit_id="LIM_ALT_MIN",
        parameter_id="altitude_ft",
        limit_type=LimitType.LOWER_BOUND,
        severity=LimitSeverity.CAUTION,
        lower_bound=1010.0,
        unit="ft",
        vehicle_id="VEH_AERIS_01",
        profile_id="PROF_AERIS_STD",
        profile_version="1.0.0",
        source_reference="DO-160 Section 4",
    )
    return EngineeringLimitProfile(
        limit_profile_id="LIM_PROF_01",
        vehicle_id="VEH_AERIS_01",
        profile_id="PROF_AERIS_STD",
        profile_version="1.0.0",
        name="Baseline Envelope Profile",
        limits=(lim_upper, lim_lower),
    )


# ---------------------------------------------------------------------------
# 1. Single Limit Line and Band Tests
# ---------------------------------------------------------------------------


class TestSingleLimitElements:
    """Tests for direct add_limit_line and add_limit_band helpers."""

    def test_add_limit_line_shapes_and_legend(self, base_altitude_fig: go.Figure) -> None:
        initial_traces = len(base_altitude_fig.data)
        initial_shapes = len(base_altitude_fig.layout.shapes)

        add_limit_line(
            fig=base_altitude_fig,
            value=1050.0,
            label="Service Ceiling",
            severity=LimitSeverity.WARNING,
            unit="ft",
        )

        assert len(base_altitude_fig.layout.shapes) == initial_shapes + 1
        assert len(base_altitude_fig.data) == initial_traces + 1
        legend_trace = base_altitude_fig.data[-1]
        assert "[LIMIT] Service Ceiling: 1050 ft" in legend_trace.name
        assert legend_trace.line.dash == LIMIT_SEVERITY_STYLES[LimitSeverity.WARNING]["dash"]

    def test_add_limit_line_non_finite_raises(self, base_altitude_fig: go.Figure) -> None:
        with pytest.raises(ValueError, match="must be a finite numeric float"):
            add_limit_line(base_altitude_fig, float("nan"), "Invalid NaN")

        with pytest.raises(ValueError, match="must be a finite numeric float"):
            add_limit_line(base_altitude_fig, float("inf"), "Invalid Inf")

    def test_add_limit_band_shapes_and_legend(self, base_altitude_fig: go.Figure) -> None:
        initial_traces = len(base_altitude_fig.data)
        initial_shapes = len(base_altitude_fig.layout.shapes)

        add_limit_band(
            fig=base_altitude_fig,
            lower_bound=1020.0,
            upper_bound=1080.0,
            label="Normal Corridor",
            severity=LimitSeverity.CAUTION,
            unit="ft",
        )

        assert len(base_altitude_fig.layout.shapes) == initial_shapes + 1
        assert len(base_altitude_fig.data) == initial_traces + 1
        legend_trace = base_altitude_fig.data[-1]
        assert "[LIMIT ENVELOPE] Normal Corridor: [1020, 1080] ft" in legend_trace.name

    def test_add_limit_band_invalid_bounds_raises(self, base_altitude_fig: go.Figure) -> None:
        with pytest.raises(ValueError, match="cannot exceed upper_bound"):
            add_limit_band(base_altitude_fig, lower_bound=500.0, upper_bound=200.0)


# ---------------------------------------------------------------------------
# 2. Multi-Limit Overlay and Compatibility Safety Tests
# ---------------------------------------------------------------------------


class TestLimitOverlays:
    """Tests for add_limit_overlays, unit safety, and applicability filtering."""

    def test_apply_profile_limits(
        self, base_altitude_fig: go.Figure, sample_limit_profile: EngineeringLimitProfile
    ) -> None:
        res = add_limit_overlays(
            fig=base_altitude_fig,
            limits=sample_limit_profile,
            parameter_id="altitude_ft",
            vehicle_id="VEH_AERIS_01",
        )
        assert isinstance(res, OverlayResult)
        assert res.status == ResultStatus.SUCCESS
        assert res.applied_count == 2
        assert res.skipped_count == 0
        assert len(res.issues) == 0
        assert base_altitude_fig.layout.meta["limits_status"] == "APPLIED"

    def test_strict_unit_safety_gate_suppresses_overlay(self, base_altitude_fig: go.Figure) -> None:
        """Limits declaring incompatible units must be suppressed; no silent conversion."""
        mismatched_limit = ParameterLimitDefinition(
            limit_id="LIM_ALT_METRIC",
            parameter_id="altitude_ft",
            limit_type=LimitType.UPPER_BOUND,
            severity=LimitSeverity.EXCEEDANCE,
            upper_bound=300.0,
            unit="m",  # Incompatible with signal unit 'ft'
            vehicle_id="VEH_AERIS_01",
            profile_id="PROF_AERIS_STD",
            profile_version="1.0.0",
            source_reference="ICAO Annex 6",
        )

        res = add_limit_overlays(
            fig=base_altitude_fig,
            limits=[mismatched_limit],
            parameter_id="altitude_ft",
            unit="ft",
        )
        assert res.applied_count == 0
        assert res.skipped_count == 1
        assert len(res.issues) == 1
        assert res.issues[0].issue_code == "ERR_INCOMPATIBLE_LIMIT_UNIT"
        assert "incompatible with signal unit 'ft'" in res.issues[0].message
        assert "Overlay suppressed" in res.issues[0].message

    def test_vehicle_identity_mismatch_skips(
        self, base_altitude_fig: go.Figure, sample_limit_profile: EngineeringLimitProfile
    ) -> None:
        res = add_limit_overlays(
            fig=base_altitude_fig,
            limits=sample_limit_profile,
            parameter_id="altitude_ft",
            vehicle_id="OTHER_VEHICLE_99",
        )
        assert res.status == ResultStatus.WARNING
        assert res.applied_count == 0
        assert len(res.issues) == 1
        assert res.issues[0].issue_code == "WARN_VEHICLE_MISMATCH"

    def test_configuration_scoped_applicability(self, base_altitude_fig: go.Figure) -> None:
        lim_clean = ParameterLimitDefinition(
            limit_id="LIM_FLAPS_UP",
            parameter_id="altitude_ft",
            limit_type=LimitType.UPPER_BOUND,
            configuration_id="CONF_CLEAN",
            severity=LimitSeverity.WARNING,
            upper_bound=1050.0,
            unit="ft",
            vehicle_id="VEH_AERIS_01",
            profile_id="PROF_AERIS_STD",
            profile_version="1.0.0",
            source_reference="Manual Sec 2",
        )
        lim_landing = ParameterLimitDefinition(
            limit_id="LIM_FLAPS_DN",
            parameter_id="altitude_ft",
            limit_type=LimitType.UPPER_BOUND,
            configuration_id="CONF_LANDING",
            severity=LimitSeverity.WARNING,
            upper_bound=1030.0,
            unit="ft",
            vehicle_id="VEH_AERIS_01",
            profile_id="PROF_AERIS_STD",
            profile_version="1.0.0",
            source_reference="Manual Sec 2",
        )

        res = add_limit_overlays(
            fig=base_altitude_fig,
            limits=[lim_clean, lim_landing],
            parameter_id="altitude_ft",
            configuration_id="CONF_CLEAN",
        )
        assert res.applied_count == 1
        assert res.skipped_count == 1
        assert res.issues[0].issue_code == "INFO_LIMIT_CONFIGURATION_MISMATCH"

    def test_unconfigured_limits_explicitly_tagged_without_safety_implication(
        self, base_altitude_fig: go.Figure
    ) -> None:
        empty_profile = EngineeringLimitProfile(
            limit_profile_id="EMPTY_PROF",
            vehicle_id="VEH_AERIS_01",
            profile_id="PROF_AERIS_STD",
            profile_version="1.0.0",
            name="Empty Limits",
            limits=(),
        )
        res = add_limit_overlays(
            fig=base_altitude_fig,
            limits=empty_profile,
            parameter_id="altitude_ft",
        )
        assert res.applied_count == 0
        assert res.issues[0].issue_code == "INFO_NO_LIMITS_FOUND"
        # Must explicitly note UNCONFIGURED; never implied to be safe or within limits
        assert base_altitude_fig.layout.meta["limits_status"] == "UNCONFIGURED"

    def test_limit_evaluation_record_overlay(self, base_altitude_fig: go.Figure) -> None:
        eval_record = LimitEvaluationRecord(
            evaluation_id="EVAL_001",
            limit_id="LIM_ALT_EVAL",
            parameter_id="altitude_ft",
            limit_type=LimitType.UPPER_BOUND,
            severity=LimitSeverity.EXCEEDANCE,
            outcome=EvaluationOutcome.OUTSIDE_BOUNDS,
            reason="Exceeded ceiling",
            observed_value=1090.0,
            observed_unit="ft",
            upper_bound=1085.0,
            limit_unit="ft",
            is_exceeded=True,
            margin=-5.0,
            vehicle_id="VEH_AERIS_01",
            profile_id="PROF_AERIS_STD",
            profile_version="1.0.0",
            source_reference="DO-160 Section 4",
        )

        res = add_limit_overlays(
            fig=base_altitude_fig,
            limits=[eval_record],
            parameter_id="altitude_ft",
        )
        assert res.applied_count == 1
        assert res.skipped_count == 0
        # Exceedance lines must use critical color and solid dash
        last_trace = base_altitude_fig.data[-1]
        assert last_trace.line.color == AERIS_PALETTE["critical"]
        assert last_trace.line.dash == "solid"


# ---------------------------------------------------------------------------
# 3. Anomaly and Event Overlays Tests
# ---------------------------------------------------------------------------


class TestAnomalyAndEventOverlays:
    """Tests for add_anomaly_markers and add_event_markers."""

    def test_point_anomaly_with_numeric_observed_value(self, base_altitude_fig: go.Figure) -> None:
        prov = create_provenance(
            activity_name="anomaly_detection",
            source_uri="telemetry://stream.parquet",
            epistemic_level=EpistemicLevel.CALCULATED,
        )
        anomaly = AnomalyRecord(
            anomaly_id="ANOM_001",
            analysis_id="ANALYSIS_01",
            target_column="altitude_ft",
            classification=AnomalyClassification.POINT_ANOMALY,
            severity=AnomalySeverity.HIGH,
            detection_method=DetectionMethod.RULE_BASED,
            timestamp=datetime.datetime(2026, 10, 15, 12, 0, 10, tzinfo=datetime.UTC),
            observed_value=1055.0,
            epistemic_level=EpistemicLevel.CALCULATED,
            provenance=prov,
        )

        res = add_anomaly_markers(
            fig=base_altitude_fig,
            anomalies=[anomaly],
            parameter_id="altitude_ft",
        )
        assert res.status == ResultStatus.SUCCESS
        assert res.applied_count == 1
        last_trace = base_altitude_fig.data[-1]
        assert isinstance(last_trace, go.Scatter)
        assert last_trace.mode == "markers"
        assert last_trace.marker.symbol == ANOMALY_SEVERITY_STYLES[AnomalySeverity.HIGH]["symbol"]
        assert "[ANOMALY: HIGH] ANOM_001" in last_trace.name
        # Check epistemic disclaimer in hover template
        assert "not confirmed root cause" in last_trace.hovertemplate

    def test_interval_anomaly_and_invalid_timestamps(self, base_altitude_fig: go.Figure) -> None:
        prov = create_provenance(
            activity_name="anomaly_detection",
            source_uri="test",
            epistemic_level=EpistemicLevel.CALCULATED,
        )
        valid_interval = AnomalyRecord(
            anomaly_id="ANOM_VALID_INT",
            analysis_id="ANALYSIS_01",
            target_column="altitude_ft",
            classification=AnomalyClassification.COLLECTIVE_ANOMALY,
            severity=AnomalySeverity.MEDIUM,
            detection_method=DetectionMethod.STATISTICAL,
            start_timestamp=datetime.datetime(2026, 10, 15, 12, 0, 15, tzinfo=datetime.UTC),
            end_timestamp=datetime.datetime(2026, 10, 15, 12, 0, 25, tzinfo=datetime.UTC),
            epistemic_level=EpistemicLevel.CALCULATED,
            provenance=prov,
        )
        # Inverted interval (start > end)
        invalid_interval = AnomalyRecord(
            anomaly_id="ANOM_INVALID_INT",
            analysis_id="ANALYSIS_01",
            target_column="altitude_ft",
            classification=AnomalyClassification.COLLECTIVE_ANOMALY,
            severity=AnomalySeverity.MEDIUM,
            detection_method=DetectionMethod.STATISTICAL,
            start_timestamp=datetime.datetime(2026, 10, 15, 12, 0, 30, tzinfo=datetime.UTC),
            end_timestamp=datetime.datetime(2026, 10, 15, 12, 0, 20, tzinfo=datetime.UTC),
            epistemic_level=EpistemicLevel.CALCULATED,
            provenance=prov,
        )

        res = add_anomaly_markers(
            fig=base_altitude_fig,
            anomalies=[valid_interval, invalid_interval],
            parameter_id="altitude_ft",
        )
        assert res.applied_count == 1
        assert res.skipped_count == 1
        assert res.issues[0].issue_code == "ERR_INVALID_TIMESTAMP"

    def test_test_point_observation_event_overlay(self, base_altitude_fig: go.Figure) -> None:
        tp_obs = TestPointObservation(
            observation_id="OBS_TP_01",
            test_point_id="TP_LEVEL_ACCEL",
            source_reference="flight_001.csv",
            start_timestamp=datetime.datetime(2026, 10, 15, 12, 0, 5, tzinfo=datetime.UTC),
            end_timestamp=datetime.datetime(2026, 10, 15, 12, 0, 15, tzinfo=datetime.UTC),
        )

        res = add_event_markers(fig=base_altitude_fig, events=[tp_obs])
        assert res.status == ResultStatus.SUCCESS
        assert res.applied_count == 1
        legend_trace = base_altitude_fig.data[-1]
        assert "[TEST POINT] TP_LEVEL_ACCEL" in legend_trace.name

    def test_generic_engineering_event_discrete_and_interval(
        self, base_altitude_fig: go.Figure
    ) -> None:
        evt_point = EngineeringEvent(
            event_id="EVT_001",
            name="Landing Gear Retracted",
            timestamp=datetime.datetime(2026, 10, 15, 12, 0, 8, tzinfo=datetime.UTC),
            severity="INFO",
        )
        evt_interval = EngineeringEvent(
            event_id="EVT_002",
            name="Sideslip Maneuver",
            start_timestamp=datetime.datetime(2026, 10, 15, 12, 0, 20, tzinfo=datetime.UTC),
            end_timestamp=datetime.datetime(2026, 10, 15, 12, 0, 35, tzinfo=datetime.UTC),
            severity="WARNING",
        )

        res = add_event_markers(fig=base_altitude_fig, events=[evt_point, evt_interval])
        assert res.status == ResultStatus.SUCCESS
        assert res.applied_count == 2
        assert res.skipped_count == 0


# ---------------------------------------------------------------------------
# 4. Integrated Overlaid Plot Generator Tests
# ---------------------------------------------------------------------------


class TestGenerateOverlaidTimeSeriesPlot:
    """Tests for full composite generate_overlaid_time_series_plot."""

    def test_end_to_end_overlaid_plot(
        self, telemetry_df: pd.DataFrame, sample_limit_profile: EngineeringLimitProfile
    ) -> None:
        prov = create_provenance(
            activity_name="anomaly_detection",
            source_uri="test",
            epistemic_level=EpistemicLevel.CALCULATED,
        )
        anomaly = AnomalyRecord(
            anomaly_id="ANOM_001",
            analysis_id="A1",
            target_column="altitude_ft",
            classification=AnomalyClassification.POINT_ANOMALY,
            severity=AnomalySeverity.MEDIUM,
            detection_method=DetectionMethod.RULE_BASED,
            timestamp=datetime.datetime(2026, 10, 15, 12, 0, 10, tzinfo=datetime.UTC),
            observed_value=1020.0,
            epistemic_level=EpistemicLevel.CALCULATED,
            provenance=prov,
        )
        evt = EngineeringEvent(
            event_id="EVT_001",
            name="Climb Initiation",
            timestamp=datetime.datetime(2026, 10, 15, 12, 0, 2, tzinfo=datetime.UTC),
        )

        res = generate_overlaid_time_series_plot(
            data=telemetry_df,
            parameter_id="altitude_ft",
            limits=sample_limit_profile,
            anomalies=[anomaly],
            events=[evt],
            unit="ft",
            display_name="Altitude Profile",
        )

        assert isinstance(res, TimeSeriesPlotResult)
        assert res.status == ResultStatus.SUCCESS
        assert res.point_count == 50
        assert res.non_null_count == 50
        assert res.unit == "ft"
        assert res.figure is not None
        assert res.figure.layout.meta["applied_limits_count"] == 2

    def test_source_dataframe_immutability(
        self, telemetry_df: pd.DataFrame, sample_limit_profile: EngineeringLimitProfile
    ) -> None:
        """Verify input dataframe is bitwise preserved."""
        df_copy = telemetry_df.copy(deep=True)
        generate_overlaid_time_series_plot(
            data=telemetry_df,
            parameter_id="altitude_ft",
            limits=sample_limit_profile,
            unit="ft",
        )
        pd.testing.assert_frame_equal(telemetry_df, df_copy)
