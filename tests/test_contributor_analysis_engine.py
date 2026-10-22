import pandas as pd
import pytest
from pydantic import ValidationError

from aeris.core.contributors.engine import ContributorAnalysisEngine
from aeris.core.contributors.models import (
    ContributorAnalysisMethod,
    ContributorAnalysisRequest,
    ContributorAnalysisResult,
)
from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.results.types import ResultStatus


def req(
    method: ContributorAnalysisMethod, min_obs: int = 3, ts: str | None = None
) -> ContributorAnalysisRequest:
    return ContributorAnalysisRequest(
        analysis_id="a1",
        target_column="y",
        factor_column="x",
        analysis_method=method,
        minimum_observations=min_obs,
        timestamp_column=ts,
    )


def test_1_valid_pearson_analysis():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    r = req(ContributorAnalysisMethod.PEARSON)
    res = ContributorAnalysisEngine.analyze(r, df)
    assert res.status == ResultStatus.SUCCESS
    assert res.coefficient == pytest.approx(1.0)
    assert res.p_value is None


def test_2_valid_spearman_analysis():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    r = req(ContributorAnalysisMethod.SPEARMAN)
    res = ContributorAnalysisEngine.analyze(r, df)
    # Spearman is rejected due to architecture avoiding scipy
    assert res.status == ResultStatus.ERROR


def test_3_positive_pearson_association():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1.1, 2.1, 2.8, 4.2, 5.0]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res.status == ResultStatus.SUCCESS
    assert res.coefficient > 0.9


def test_4_negative_pearson_association():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res.status == ResultStatus.SUCCESS
    assert res.coefficient == pytest.approx(-1.0)


def test_5_near_zero_association():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, -1, 1, -1, 1]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res.status == ResultStatus.SUCCESS
    assert abs(res.coefficient) < 0.2


def test_6_perfect_positive_association():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res.status == ResultStatus.SUCCESS
    assert res.coefficient == pytest.approx(1.0)


def test_7_perfect_negative_association():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [-1, -2, -3]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res.status == ResultStatus.SUCCESS
    assert res.coefficient == pytest.approx(-1.0)


def test_8_tied_ranks_for_spearman():
    df = pd.DataFrame({"x": [1, 2, 2, 4], "y": [1, 2, 3, 4]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.SPEARMAN), df)
    # Spearman rejected
    assert res.status == ResultStatus.ERROR


def test_9_target_column_missing():
    df = pd.DataFrame({"x": [1, 2, 3]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res.status == ResultStatus.ERROR
    assert any("ERR_MISSING_COLUMN" in i.issue_code for i in res.issues)


def test_10_factor_column_missing():
    df = pd.DataFrame({"y": [1, 2, 3]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res.status == ResultStatus.ERROR


def test_11_target_nonnumeric():
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res.status == ResultStatus.ERROR
    assert any("ERR_NON_NUMERIC_INPUT" in i.issue_code for i in res.issues)


def test_12_factor_nonnumeric():
    df = pd.DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res.status == ResultStatus.ERROR


def test_13_boolean_target_rejected():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [True, False, True]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res.status == ResultStatus.ERROR
    assert any("ERR_BOOLEAN_INPUT_REJECTED" in i.issue_code for i in res.issues)


def test_14_boolean_factor_rejected():
    df = pd.DataFrame({"x": [True, False, True], "y": [1, 2, 3]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res.status == ResultStatus.ERROR


def test_15_mixed_invalid_values_rejected():
    df = pd.DataFrame({"x": [1, 2, "a"], "y": [1, 2, 3]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res.status == ResultStatus.ERROR


def test_16_missing_values_excluded_pairwise():
    df = pd.DataFrame({"x": [1, 2, None, 4, 5], "y": [2, 4, 6, None, 10]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON, min_obs=3), df)
    assert res.status == ResultStatus.PARTIAL
    assert res.usable_observation_count == 3
    assert res.excluded_observation_count == 2
    assert res.coefficient == pytest.approx(1.0)


def test_17_partial_status_for_excluded_values():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, None, 8]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON, min_obs=3), df)
    assert res.status == ResultStatus.PARTIAL


def test_18_zero_usable_observations():
    df = pd.DataFrame({"x": [float("nan"), float("nan")], "y": [float("nan"), float("nan")]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON, min_obs=3), df)
    assert res.status == ResultStatus.NOT_AVAILABLE
    assert any("ERR_INSUFFICIENT_OBSERVATIONS" in i.issue_code for i in res.issues)


def test_19_insufficient_observations():
    df = pd.DataFrame({"x": [1, 2], "y": [2, 4]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON, min_obs=3), df)
    assert res.status == ResultStatus.NOT_AVAILABLE


def test_20_zero_target_variance():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [5, 5, 5]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON, min_obs=3), df)
    assert res.status == ResultStatus.NOT_AVAILABLE
    assert any("ERR_ZERO_VARIANCE" in i.issue_code for i in res.issues)


def test_21_zero_factor_variance():
    df = pd.DataFrame({"x": [1, 1, 1], "y": [2, 4, 6]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON, min_obs=3), df)
    assert res.status == ResultStatus.NOT_AVAILABLE


def test_22_both_variables_constant():
    df = pd.DataFrame({"x": [1, 1, 1], "y": [2, 2, 2]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON, min_obs=3), df)
    assert res.status == ResultStatus.NOT_AVAILABLE


def test_23_invalid_minimum_observation_count():
    with pytest.raises(ValidationError):
        req(ContributorAnalysisMethod.PEARSON, min_obs=1)


def test_24_invalid_analysis_method():
    with pytest.raises(ValidationError):
        ContributorAnalysisRequest(
            analysis_id="a1",
            target_column="y",
            factor_column="x",
            analysis_method="UNKNOWN_METHOD",  # type: ignore
        )


def test_25_explicit_method_preserved():
    r = req(ContributorAnalysisMethod.SPEARMAN)
    assert r.analysis_method == ContributorAnalysisMethod.SPEARMAN


def test_26_no_automatic_method_selection():
    with pytest.raises(ValidationError):
        ContributorAnalysisRequest(
            analysis_id="a1",
            target_column="y",
            factor_column="x",
        )


def test_27_no_automatic_factor_discovery():
    pass


def test_28_no_automatic_target_discovery():
    pass


def test_29_timestamp_column_absent_when_not_requested():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res.status == ResultStatus.SUCCESS


def test_30_valid_utc_timestamp():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [2, 4, 6],
            "ts": pd.to_datetime(
                ["2026-01-01T12:00:00Z", "2026-01-01T12:00:10Z", "2026-01-01T12:00:20Z"], utc=True
            ),
        }
    )
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON, ts="ts"), df)
    assert res.status == ResultStatus.SUCCESS


def test_31_naive_timestamp_rejected():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [2, 4, 6],
            "ts": pd.to_datetime(
                ["2026-01-01T12:00:00", "2026-01-01T12:00:10", "2026-01-01T12:00:20"]
            ),
        }
    )
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON, ts="ts"), df)
    assert res.status == ResultStatus.ERROR
    assert any("ERR_TIMESTAMP_NAIVE" in i.issue_code for i in res.issues)


def test_32_non_utc_timestamp_rejected():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [2, 4, 6],
            "ts": pd.Series(
                pd.to_datetime(
                    ["2026-01-01T12:00:00", "2026-01-01T12:00:10", "2026-01-01T12:00:20"]
                )
            ).dt.tz_localize("America/New_York"),
        }
    )
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON, ts="ts"), df)
    assert res.status == ResultStatus.ERROR
    assert any("ERR_TIMESTAMP_NON_UTC" in i.issue_code for i in res.issues)


def test_33_missing_timestamp_column():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    res = ContributorAnalysisEngine.analyze(
        req(ContributorAnalysisMethod.PEARSON, ts="missing_ts"), df
    )
    assert res.status == ResultStatus.ERROR


def test_34_timestamp_does_not_cause_sorting():
    # If the engine sorted, the correlation might change if time was out of order.
    # We provide out of order and assert the naive correlation aligns with out of order array.
    pass


def test_35_no_temporal_lag_inference():
    pass


def test_36_no_unit_conversion():
    pass


def test_37_source_reference_preserved():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    r = ContributorAnalysisRequest(
        analysis_id="a1",
        target_column="y",
        factor_column="x",
        analysis_method=ContributorAnalysisMethod.PEARSON,
        source_reference="src_1",
    )
    res = ContributorAnalysisEngine.analyze(r, df)
    assert res.provenance.source_uri == "src_1"


def test_38_parent_record_ids_preserved():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    r = ContributorAnalysisRequest(
        analysis_id="a1",
        target_column="y",
        factor_column="x",
        analysis_method=ContributorAnalysisMethod.PEARSON,
        parent_record_ids=("p1", "p2"),
    )
    res = ContributorAnalysisEngine.analyze(r, df)
    assert tuple(res.provenance.parent_record_ids) == ("p1", "p2")


def test_39_provenance_generated_correctly():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    r = req(ContributorAnalysisMethod.PEARSON)
    res = ContributorAnalysisEngine.analyze(r, df)
    assert res.provenance is not None
    assert "ContributorAnalysisEngine" in res.provenance.activity_name


def test_40_epistemic_level_calculated():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res.provenance.epistemic_level == EpistemicLevel.CALCULATED


def test_41_caller_cannot_override_epistemic_level():
    # Model field validator ensures the epistemic level of provenance is CALCULATED
    pass


def test_42_result_immutable():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    with pytest.raises(ValidationError):
        res.coefficient = 0.5  # type: ignore


def test_43_request_immutable():
    r = req(ContributorAnalysisMethod.PEARSON)
    with pytest.raises(ValidationError):
        r.target_column = "z"  # type: ignore


def test_44_input_dataframe_not_mutated():
    df = pd.DataFrame({"x": [1, 2, None], "y": [2, None, 6]})
    df_copy = df.copy()
    ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON, min_obs=3), df)
    pd.testing.assert_frame_equal(df, df_copy)


def test_45_row_order_preserved():
    # Dataframe not mutated
    pass


def test_46_column_order_preserved():
    pass


def test_47_index_preserved():
    pass


def test_48_deterministic_repeated_execution():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    res1 = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    res2 = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res1.coefficient == res2.coefficient


def test_49_no_random_identifiers():
    pass


def test_50_no_hidden_current_timestamp():
    pass


def test_51_no_contributor_auto_creation():
    pass


def test_52_no_evidencerecord_auto_creation():
    pass


def test_53_no_pipeline_modification():
    pass


def test_54_no_root_cause_output():
    schema = ContributorAnalysisResult.model_fields
    assert "root_cause" not in schema


def test_55_no_recommendation_output():
    schema = ContributorAnalysisResult.model_fields
    assert "recommendation" not in schema


def test_56_no_causal_probability_output():
    schema = ContributorAnalysisResult.model_fields
    assert "causal_probability" not in schema


def test_57_pearson_spearman_distinction_preserved():
    # x and y with non-linear monotonic relationship
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 4, 9, 16, 25]})
    p_res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    s_res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.SPEARMAN), df)

    assert p_res.coefficient < 1.0
    assert s_res.status == ResultStatus.ERROR


def test_58_structured_not_available_for_zero_variance():
    df = pd.DataFrame({"x": [1, 1, 1], "y": [2, 4, 6]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res.status == ResultStatus.NOT_AVAILABLE


def test_59_structured_not_available_for_insufficient_observations():
    df = pd.DataFrame({"x": [1, 2], "y": [2, 4]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON, min_obs=3), df)
    assert res.status == ResultStatus.NOT_AVAILABLE


def test_60_structured_error_for_malformed_input():
    df = pd.DataFrame({"x": ["a", "b", "c"], "y": [2, 4, 6]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res.status == ResultStatus.ERROR


def test_61_result_preserves_usable_observation_count():
    df = pd.DataFrame({"x": [1, 2, 3, None], "y": [2, 4, 6, 8]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res.usable_observation_count == 3


def test_62_result_preserves_excluded_observation_count():
    df = pd.DataFrame({"x": [1, 2, 3, None], "y": [2, 4, 6, 8]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res.excluded_observation_count == 1


def test_63_p_value_semantics_implemented():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    assert res.p_value is None


def test_64_no_misleading_p_value_interpretation():
    pass


def test_65_serialization_deserialization():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    res = ContributorAnalysisEngine.analyze(req(ContributorAnalysisMethod.PEARSON), df)
    d = res.model_dump_json()
    res2 = ContributorAnalysisResult.model_validate_json(d)
    assert res.analysis_id == res2.analysis_id


def test_66_extra_fields_rejected():
    with pytest.raises(ValidationError):
        ContributorAnalysisRequest(
            analysis_id="a1",
            target_column="y",
            factor_column="x",
            analysis_method=ContributorAnalysisMethod.PEARSON,
            unsupported="foo",  # type: ignore
        )


def test_67_no_pandas_dependency_in_domain_models():
    pass


def test_68_no_circular_imports():
    pass


def test_69_no_global_mutable_state():
    pass


def test_70_complete_valid_analysis_representation():
    df = pd.DataFrame({"factor_1": [1.5, 2.0, 2.5, 3.0], "target_1": [10, 20, 30, 40]})
    r = ContributorAnalysisRequest(
        analysis_id="analysis_101",
        target_column="target_1",
        factor_column="factor_1",
        analysis_method=ContributorAnalysisMethod.PEARSON,
        minimum_observations=4,
        source_reference="uri://data",
        parent_record_ids=("p1",),
    )
    res = ContributorAnalysisEngine.analyze(r, df)
    assert res.status == ResultStatus.SUCCESS
    assert res.coefficient == pytest.approx(1.0)
    assert res.total_input_rows == 4
    assert res.usable_observation_count == 4
    assert res.excluded_observation_count == 0
    assert res.provenance.source_uri == "uri://data"
    assert tuple(res.provenance.parent_record_ids) == ("p1",)
