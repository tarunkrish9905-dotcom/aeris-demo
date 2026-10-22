from datetime import UTC

import pandas as pd

from aeris.core.expected_observed.models import (
    ComparisonMethod,
    ExpectationSource,
    ExpectedValue,
    PipelineExpectedObservedRequest,
    ScalarComparisonRequest,
    Tolerance,
    ToleranceType,
)
from aeris.core.normalization.engine import NormalizationRequest
from aeris.core.pipeline.orchestrator import TestPointAssociationRequest, run_csv_pipeline
from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.results.types import ResultStatus
from aeris.core.test_operations.test_point_observation import TestPointObservation


def create_sample_csv(tmp_path):
    csv_file = tmp_path / "test_data.csv"
    data = {
        "timestamp": ["2023-01-01T12:00:00Z", "2023-01-01T12:00:10Z", "2023-01-01T12:00:20Z"],
        "airspeed_kts": [100.0, 105.0, 110.0],
        "altitude_ft": [5000.0, 5010.0, 5020.0],
        "pitch_deg": [0.0, 0.0, 0.0],
        "roll_deg": [0.0, 0.0, 0.0],
        "engine_rpm": [0.0, 0.0, 0.0],
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)
    return str(csv_file)


def create_dummy_expected(param: str, val: float, unit: str | None = "kt"):
    return ExpectedValue(
        expected_id="e1",
        parameter_id=param,
        value=val,
        unit=unit,
        source=ExpectationSource.TEST_CARD,
        source_reference="tc_001",
        epistemic_level=EpistemicLevel.PROCEDURAL,
    )


def test_pipeline_eo_skipped(tmp_path):
    csv_file = create_sample_csv(tmp_path)
    res = run_csv_pipeline(csv_file, expected_observed_request=None)
    assert res.expected_observed is None
    assert res.status == ResultStatus.SUCCESS


def test_pipeline_eo_success_absolute(tmp_path):
    csv_file = create_sample_csv(tmp_path)

    req = PipelineExpectedObservedRequest(
        scalar_requests=(
            ScalarComparisonRequest(
                comparison_id="comp1",
                expected=create_dummy_expected("airspeed_kts", 100.0),
                observed_parameter_id="airspeed_kts",
                observed_row_index=1,  # 105.0
                observed_unit="kt",
                method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
                tolerance=Tolerance(
                    tolerance_type=ToleranceType.ABSOLUTE,
                    lower_bound=-10.0,
                    upper_bound=10.0,
                    unit="kt",
                ),
            ),
        )
    )

    res = run_csv_pipeline(csv_file, expected_observed_request=req)
    assert res.expected_observed is not None
    assert res.expected_observed.status == ResultStatus.SUCCESS
    assert len(res.expected_observed.comparisons) == 1

    comp = res.expected_observed.comparisons[0]
    assert comp.within_tolerance is True
    assert comp.deviation.deviation_value == 5.0
    assert comp.provenance.epistemic_level == EpistemicLevel.CALCULATED
    assert res.ingestion.provenance.record_id in comp.provenance.parent_record_ids


def test_pipeline_eo_failure_isolation(tmp_path):
    csv_file = create_sample_csv(tmp_path)

    req = PipelineExpectedObservedRequest(
        scalar_requests=(
            ScalarComparisonRequest(
                comparison_id="comp_good",
                expected=create_dummy_expected("airspeed_kts", 100.0),
                observed_parameter_id="airspeed_kts",
                observed_row_index=1,
                observed_unit="kt",
                method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
            ),
            ScalarComparisonRequest(
                comparison_id="comp_bad_param",
                expected=create_dummy_expected("airspeed_kts", 100.0),
                observed_parameter_id="altitude_ft",  # Mismatch
                observed_row_index=1,
                observed_unit="kt",
                method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
            ),
        )
    )

    res = run_csv_pipeline(csv_file, expected_observed_request=req)
    assert res.expected_observed is not None
    assert res.expected_observed.status == ResultStatus.PARTIAL
    assert len(res.expected_observed.comparisons) == 2

    comp1 = next(c for c in res.expected_observed.comparisons if c.comparison_id == "comp_good")
    assert comp1.status == ResultStatus.SUCCESS

    comp2 = next(
        c for c in res.expected_observed.comparisons if c.comparison_id == "comp_bad_param"
    )
    assert comp2.status == ResultStatus.ERROR


def test_pipeline_eo_zero_denominator(tmp_path):
    csv_file = create_sample_csv(tmp_path)

    req = PipelineExpectedObservedRequest(
        scalar_requests=(
            ScalarComparisonRequest(
                comparison_id="comp_zero",
                expected=create_dummy_expected("airspeed_kts", 0.0),
                observed_parameter_id="airspeed_kts",
                observed_row_index=1,
                observed_unit="kt",
                method=ComparisonMethod.RELATIVE_DIFFERENCE,
            ),
        )
    )

    res = run_csv_pipeline(csv_file, expected_observed_request=req)
    assert res.expected_observed.status == ResultStatus.ERROR
    assert res.expected_observed.comparisons[0].deviation is None


def test_pipeline_eo_incompatible_units(tmp_path):
    csv_file = create_sample_csv(tmp_path)

    req = PipelineExpectedObservedRequest(
        scalar_requests=(
            ScalarComparisonRequest(
                comparison_id="comp_units",
                expected=create_dummy_expected("airspeed_kts", 100.0, "kt"),
                observed_parameter_id="airspeed_kts",
                observed_row_index=1,
                observed_unit="m/s",
                method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
            ),
        )
    )
    res = run_csv_pipeline(csv_file, expected_observed_request=req)
    assert res.expected_observed.status == ResultStatus.ERROR
    assert "ERR_INCOMPATIBLE_UNITS" in [
        i.issue_code for i in res.expected_observed.comparisons[0].issues
    ]


def test_pipeline_eo_missing_column(tmp_path):
    csv_file = create_sample_csv(tmp_path)
    req = PipelineExpectedObservedRequest(
        scalar_requests=(
            ScalarComparisonRequest(
                comparison_id="comp_miss",
                expected=create_dummy_expected("fake_col", 100.0),
                observed_parameter_id="fake_col",
                observed_row_index=1,
                method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
            ),
        )
    )
    res = run_csv_pipeline(csv_file, expected_observed_request=req)
    assert res.expected_observed.status == ResultStatus.ERROR
    assert res.expected_observed.comparisons[0].status == ResultStatus.ERROR
    assert "ERR_MISSING_OBSERVED_COLUMN" in [
        i.issue_code for i in res.expected_observed.comparisons[0].issues
    ]


def test_pipeline_eo_normalization_hierarchy(tmp_path):
    csv_file = create_sample_csv(tmp_path)

    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")

    req = PipelineExpectedObservedRequest(
        scalar_requests=(
            ScalarComparisonRequest(
                comparison_id="comp_norm",
                expected=create_dummy_expected("airspeed_kts", 100.0),
                observed_parameter_id="airspeed_kts",
                observed_row_index=1,
                observed_unit="kt",
                method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
            ),
        )
    )

    res = run_csv_pipeline(csv_file, normalization_request=norm_req, expected_observed_request=req)
    assert res.expected_observed is not None
    assert res.expected_observed.status == ResultStatus.SUCCESS

    comp = res.expected_observed.comparisons[0]
    # Check provenance traces back to normalization, not ingestion directly
    assert res.normalization.provenance.record_id in comp.provenance.parent_record_ids


def test_pipeline_eo_association_failure_no_fallback(tmp_path):
    csv_file = create_sample_csv(tmp_path)

    # Intentionally fail association by selecting wrong time range
    from datetime import datetime

    assoc_req = TestPointAssociationRequest(
        observation=TestPointObservation(
            observation_id="obs1",
            test_point_id="tp1",
            source_reference="test_src",
            start_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            end_timestamp=datetime(2025, 1, 2, tzinfo=UTC),
        ),
        timestamp_column="timestamp",
    )

    req = PipelineExpectedObservedRequest(
        scalar_requests=(
            ScalarComparisonRequest(
                comparison_id="comp_assoc",
                expected=create_dummy_expected("airspeed_kts", 100.0),
                observed_parameter_id="airspeed_kts",
                observed_row_index=0,
                observed_unit="kt",
                method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
            ),
        )
    )

    res = run_csv_pipeline(csv_file, association_request=assoc_req, expected_observed_request=req)
    assert res.association.status == ResultStatus.ERROR  # Association fails because no rows match

    # EO should fail because association failed
    assert res.expected_observed.status == ResultStatus.ERROR
    assert len(res.expected_observed.comparisons) == 0
    assert "ERR_EXPECTED_OBSERVED_DATA_UNAVAILABLE" in [
        i.issue_code for i in res.expected_observed.issues
    ]
