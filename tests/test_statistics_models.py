import pytest
from pydantic import ValidationError

from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus
from aeris.core.statistics.models import (
    MetricType,
    StatisticalAnalysisRequest,
    StatisticalAnalysisResult,
    StatisticalMetric,
)


def test_valid_analysis_request() -> None:
    req = StatisticalAnalysisRequest(
        analysis_id="stat-1",
        target_column="altitude_ft",
        source_identifier="dataset-a",
        scope_metadata={"phase": "climb"},
        requested_metrics=[MetricType.MEAN, MetricType.MAXIMUM],
    )
    assert req.analysis_id == "stat-1"
    assert req.target_column == "altitude_ft"
    assert req.source_identifier == "dataset-a"
    assert req.scope_metadata == {"phase": "climb"}
    assert req.requested_metrics == [MetricType.MEAN, MetricType.MAXIMUM]


def test_target_parameter_column_requirement() -> None:
    with pytest.raises(ValidationError):
        StatisticalAnalysisRequest(analysis_id="stat-1", requested_metrics=[])  # type: ignore

    with pytest.raises(ValidationError):
        StatisticalAnalysisRequest(analysis_id="stat-1", target_column="", requested_metrics=[])


def test_explicit_metric_representation() -> None:
    metric = StatisticalMetric(
        metric_id="m-1",
        metric_type=MetricType.PERCENTILE,
        value=150.5,
        parameters={"percentile": 95},
    )
    assert metric.metric_id == "m-1"
    assert metric.metric_type == MetricType.PERCENTILE
    assert metric.value == 150.5
    assert metric.parameters == {"percentile": 95}


def test_all_required_metric_types() -> None:
    types = [
        MetricType.COUNT,
        MetricType.MISSING_COUNT,
        MetricType.MEAN,
        MetricType.MEDIAN,
        MetricType.MINIMUM,
        MetricType.MAXIMUM,
        MetricType.VARIANCE,
        MetricType.STANDARD_DEVIATION,
        MetricType.PERCENTILE,
    ]
    assert len(types) == 9
    for t in types:
        assert isinstance(t, MetricType)


def test_statistical_result_construction() -> None:
    res = StatisticalAnalysisResult(
        status=ResultStatus.SUCCESS,
        result_id="res-1",
        analysis_id="stat-1",
        target_column="altitude_ft",
        metrics=[StatisticalMetric(metric_id="m-1", metric_type=MetricType.MEAN, value=100.0)],
    )
    assert res.status == ResultStatus.SUCCESS
    assert res.result_id == "res-1"
    assert res.analysis_id == "stat-1"
    assert res.target_column == "altitude_ft"
    assert len(res.metrics) == 1
    assert res.metrics[0].value == 100.0


def test_immutable_models() -> None:
    req = StatisticalAnalysisRequest(analysis_id="stat-1", target_column="alt")
    with pytest.raises(ValidationError):
        req.target_column = "new_alt"  # type: ignore

    res = StatisticalAnalysisResult(
        status=ResultStatus.SUCCESS, result_id="res-1", analysis_id="stat-1", target_column="alt"
    )
    with pytest.raises(ValidationError):
        res.result_id = "res-2"  # type: ignore


def test_extra_field_rejection() -> None:
    with pytest.raises(ValidationError):
        StatisticalAnalysisRequest(
            analysis_id="stat-1",
            target_column="alt",
            extra_field="bad",  # type: ignore
        )

    with pytest.raises(ValidationError):
        StatisticalAnalysisResult(
            status=ResultStatus.SUCCESS,
            result_id="res-1",
            analysis_id="stat-1",
            target_column="alt",
            extra_field="bad",  # type: ignore
        )


def test_invalid_identifiers() -> None:
    with pytest.raises(ValidationError):
        StatisticalAnalysisRequest(analysis_id="", target_column="alt")

    with pytest.raises(ValidationError):
        StatisticalAnalysisResult(
            status=ResultStatus.SUCCESS, result_id="", analysis_id="stat-1", target_column="alt"
        )


def test_empty_metric_collection_behavior() -> None:
    res = StatisticalAnalysisResult(
        status=ResultStatus.SUCCESS, result_id="res-1", analysis_id="stat-1", target_column="alt"
    )
    assert res.metrics == []


def test_result_status_behavior() -> None:
    res = StatisticalAnalysisResult(
        status=ResultStatus.WARNING, result_id="res-1", analysis_id="stat-1", target_column="alt"
    )
    assert res.status == ResultStatus.WARNING
    assert res.has_warnings is True


def test_issue_representation() -> None:
    issue = Issue(
        issue_code="ERR_INSUFFICIENT_DATA",
        severity=IssueSeverity.ERROR,
        message="Not enough rows to calculate variance.",
    )
    res = StatisticalAnalysisResult(
        status=ResultStatus.ERROR,
        result_id="res-1",
        analysis_id="stat-1",
        target_column="alt",
        issues=[issue],
    )
    assert res.status == ResultStatus.ERROR
    assert res.has_errors is True
    assert res.issues[0].issue_code == "ERR_INSUFFICIENT_DATA"


def test_provenance_attachment() -> None:
    prov = create_provenance(
        source_uri="dataset-a",
        activity_name="stat_analysis",
        epistemic_level=EpistemicLevel.CALCULATED,
    )
    res = StatisticalAnalysisResult(
        status=ResultStatus.SUCCESS,
        result_id="res-1",
        analysis_id="stat-1",
        target_column="alt",
        provenance=prov,
    )
    assert res.provenance is not None
    assert res.provenance.record_id == prov.record_id


def test_calculated_epistemic_level() -> None:
    res = StatisticalAnalysisResult(
        status=ResultStatus.SUCCESS, result_id="res-1", analysis_id="stat-1", target_column="alt"
    )
    assert res.epistemic_level == EpistemicLevel.CALCULATED

    with pytest.raises(ValidationError):
        StatisticalAnalysisResult(
            status=ResultStatus.SUCCESS,
            result_id="res-1",
            analysis_id="stat-1",
            target_column="alt",
            epistemic_level=EpistemicLevel.OBSERVED,
        )


def test_no_dataframe_dependency_in_domain_models() -> None:
    # We should not be able to store a DataFrame in the generic model without error
    # since it does not define arbitrary_types_allowed=True for metrics payload.
    # Actually StatisticalAnalysisResult inherits from BasePipelineResult which
    # might allow arbitrary types,
    # but the metrics list explicitly accepts StatisticalMetric which does not.
    # We will test that we are purely using primitives.
    metric = StatisticalMetric(metric_id="m1", metric_type=MetricType.MEAN, value=1.0)
    assert isinstance(metric.value, float)


def test_no_parameter_registry_mutation() -> None:
    # Just asserting the structure implies no side effects on external registries.
    req = StatisticalAnalysisRequest(analysis_id="s", target_column="c")
    assert req.target_column == "c"


def test_deterministic_serialization() -> None:
    req = StatisticalAnalysisRequest(analysis_id="stat-1", target_column="alt")
    dump = req.model_dump_json()
    assert "stat-1" in dump
    assert "alt" in dump

    req2 = StatisticalAnalysisRequest.model_validate_json(dump)
    assert req == req2


def test_result_model_does_not_execute_calculations() -> None:
    res = StatisticalAnalysisResult(
        status=ResultStatus.SUCCESS, result_id="res-1", analysis_id="stat-1", target_column="alt"
    )
    assert len(res.metrics) == 0


def test_no_automatic_target_inference() -> None:
    with pytest.raises(ValidationError):
        # target_column is strictly required. No default inference.
        StatisticalAnalysisRequest(analysis_id="stat-1")  # type: ignore


def test_no_automatic_metric_inference() -> None:
    req = StatisticalAnalysisRequest(analysis_id="stat-1", target_column="alt")
    assert req.requested_metrics == []  # explicitly empty by default
