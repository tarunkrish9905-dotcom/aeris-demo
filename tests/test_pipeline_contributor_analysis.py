import pathlib
from datetime import UTC, datetime

import pandas as pd
import pytest

from aeris.core.contributors.models import ContributorAnalysisMethod, ContributorAnalysisRequest
from aeris.core.contributors.pipeline import PipelineContributorAnalysisRequest
from aeris.core.normalization.engine import NormalizationRequest
from aeris.core.pipeline.orchestrator import (
    DataPipelineOrchestrator,
    TestPointAssociationRequest,
    run_csv_pipeline,
)
from aeris.core.results.types import ResultStatus
from aeris.core.test_operations.test_point_observation import TestPointObservation


@pytest.fixture
def test_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "test_contrib.csv"
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T12:00:00Z",
                "2026-01-01T12:00:10Z",
                "2026-01-01T12:00:20Z",
                "2026-01-01T12:00:30Z",
                "2026-01-01T12:00:40Z",
            ],
            "altitude_ft": [1000, 1100, 1200, 1300, 1400],
            "airspeed_kts": [100, 110, 120, 130, 140],
            "pitch_deg": [15, 14, 13, 12, 11],
            "roll_deg": [1.0, None, 3.0, 4.0, 5.0],
            "engine_rpm": [2000, 2000, 2000, 2000, 2000],
            "category": ["A", "B", "A", "B", "A"],
        }
    )
    df.to_csv(p, index=False)
    return p


def test_01_backward_compatibility(test_csv: pathlib.Path):
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(test_csv)
    print([i.message for i in res.issues])
    assert res.contributor_analysis is None


def test_02_single_valid_pearson(test_csv: pathlib.Path):
    req = PipelineContributorAnalysisRequest(
        requests=(
            ContributorAnalysisRequest(
                analysis_id="c1",
                target_column="altitude_ft",
                factor_column="airspeed_kts",
                analysis_method=ContributorAnalysisMethod.PEARSON,
            ),
        )
    )
    res = run_csv_pipeline(test_csv, contributor_analysis_request=req)
    assert res.contributor_analysis is not None
    assert res.contributor_analysis.status == ResultStatus.SUCCESS
    assert len(res.contributor_analysis.results) == 1
    assert res.contributor_analysis.results[0].coefficient == pytest.approx(1.0)
    assert res.contributor_analysis.results[0].target_column == "altitude_ft"
    assert res.contributor_analysis.results[0].factor_column == "airspeed_kts"


def test_03_multiple_requests_isolation(test_csv: pathlib.Path):
    req = PipelineContributorAnalysisRequest(
        requests=(
            ContributorAnalysisRequest(
                analysis_id="c1",
                target_column="altitude_ft",
                factor_column="airspeed_kts",
                analysis_method=ContributorAnalysisMethod.PEARSON,
            ),
            ContributorAnalysisRequest(
                analysis_id="c2",
                target_column="altitude_ft",
                factor_column="category",  # non-numeric, fails
                analysis_method=ContributorAnalysisMethod.PEARSON,
            ),
            ContributorAnalysisRequest(
                analysis_id="c3",
                target_column="altitude_ft",
                factor_column="pitch_deg",
                analysis_method=ContributorAnalysisMethod.PEARSON,
            ),
        )
    )
    res = run_csv_pipeline(test_csv, contributor_analysis_request=req)
    assert res.contributor_analysis is not None
    assert res.contributor_analysis.status == ResultStatus.PARTIAL
    assert len(res.contributor_analysis.results) == 3
    assert res.contributor_analysis.results[0].status == ResultStatus.SUCCESS
    assert res.contributor_analysis.results[1].status == ResultStatus.ERROR
    assert res.contributor_analysis.results[2].status == ResultStatus.SUCCESS


def test_04_upstream_normalization_source(test_csv: pathlib.Path):
    norm = NormalizationRequest()
    req = PipelineContributorAnalysisRequest(
        requests=(
            ContributorAnalysisRequest(
                analysis_id="c1",
                target_column="altitude_ft",
                factor_column="airspeed_kts",
                analysis_method=ContributorAnalysisMethod.PEARSON,
            ),
        )
    )
    res = run_csv_pipeline(test_csv, normalization_request=norm, contributor_analysis_request=req)
    assert res.contributor_analysis is not None
    # Provenance should point to normalization
    assert res.normalization is not None
    assert (
        res.normalization.provenance.record_id
        in res.contributor_analysis.provenance.parent_record_ids
    )


def test_05_upstream_association_source(test_csv: pathlib.Path):
    norm = NormalizationRequest()
    norm.standardize_timestamp("timestamp")
    assoc = TestPointAssociationRequest(
        observation=TestPointObservation(
            observation_id="obs1",
            test_point_id="tp1",
            source_reference="ref1",
            start_timestamp=datetime(2026, 1, 1, 12, 0, 5, tzinfo=UTC),
            end_timestamp=datetime(2026, 1, 1, 12, 0, 35, tzinfo=UTC),
        ),
        timestamp_column="timestamp",
    )
    req = PipelineContributorAnalysisRequest(
        requests=(
            ContributorAnalysisRequest(
                analysis_id="c1",
                target_column="altitude_ft",
                factor_column="airspeed_kts",
                analysis_method=ContributorAnalysisMethod.PEARSON,
                minimum_observations=3,
            ),
        )
    )
    res = run_csv_pipeline(
        test_csv,
        normalization_request=norm,
        association_request=assoc,
        contributor_analysis_request=req,
    )
    assert res.association is not None
    assert res.contributor_analysis is not None
    assert (
        res.association.provenance.record_id
        in res.contributor_analysis.provenance.parent_record_ids
    )
    assert res.contributor_analysis.results[0].usable_observation_count == 3


def test_06_no_silent_fallback_on_association_failure(tmp_path: pathlib.Path):
    # Create empty csv
    p = tmp_path / "empty.csv"
    pd.DataFrame({"timestamp": [], "altitude_ft": [], "airspeed_kts": []}).to_csv(p, index=False)

    norm = NormalizationRequest()
    norm.standardize_timestamp("timestamp")
    assoc = TestPointAssociationRequest(
        observation=TestPointObservation(
            observation_id="obs1",
            test_point_id="tp1",
            source_reference="ref1",
            start_timestamp=datetime(2026, 1, 1, 12, 0, 15, tzinfo=UTC),
            end_timestamp=datetime(2026, 1, 1, 12, 0, 25, tzinfo=UTC),
        ),
        timestamp_column="timestamp",
    )
    req = PipelineContributorAnalysisRequest(
        requests=(
            ContributorAnalysisRequest(
                analysis_id="c1",
                target_column="altitude_ft",
                factor_column="airspeed_kts",
                analysis_method=ContributorAnalysisMethod.PEARSON,
            ),
        )
    )
    res = run_csv_pipeline(
        p, normalization_request=norm, association_request=assoc, contributor_analysis_request=req
    )
    # association fails because df is empty
    assert res.association is not None
    assert res.association.status == ResultStatus.ERROR
    assert res.contributor_analysis is not None
    assert res.contributor_analysis.status == ResultStatus.ERROR
    assert any(
        "ERR_UPSTREAM_ASSOCIATION_FAILED" in i.issue_code for i in res.contributor_analysis.issues
    )
    # The individual request should not be executed
    assert len(res.contributor_analysis.results) == 0


def test_07_spearman_rejected(test_csv: pathlib.Path):
    req = PipelineContributorAnalysisRequest(
        requests=(
            ContributorAnalysisRequest(
                analysis_id="c1",
                target_column="altitude_ft",
                factor_column="airspeed_kts",
                analysis_method=ContributorAnalysisMethod.SPEARMAN,
            ),
        )
    )
    res = run_csv_pipeline(test_csv, contributor_analysis_request=req)
    assert res.contributor_analysis is not None
    assert res.contributor_analysis.status == ResultStatus.ERROR
    assert res.contributor_analysis.results[0].status == ResultStatus.ERROR


def test_08_missing_values_partial_status(test_csv: pathlib.Path):
    req = PipelineContributorAnalysisRequest(
        requests=(
            ContributorAnalysisRequest(
                analysis_id="c1",
                target_column="altitude_ft",
                factor_column="roll_deg",
                analysis_method=ContributorAnalysisMethod.PEARSON,
            ),
        )
    )
    res = run_csv_pipeline(test_csv, contributor_analysis_request=req)
    assert res.contributor_analysis is not None
    assert res.contributor_analysis.status == ResultStatus.PARTIAL
    assert res.contributor_analysis.results[0].status == ResultStatus.PARTIAL
    assert res.contributor_analysis.results[0].usable_observation_count == 4
    assert res.contributor_analysis.results[0].excluded_observation_count == 1


def test_09_no_automatic_chaining_downstream(test_csv: pathlib.Path):
    req = PipelineContributorAnalysisRequest(
        requests=(
            ContributorAnalysisRequest(
                analysis_id="c1",
                target_column="altitude_ft",
                factor_column="airspeed_kts",
                analysis_method=ContributorAnalysisMethod.PEARSON,
            ),
        )
    )
    res = run_csv_pipeline(test_csv, contributor_analysis_request=req)
    # Should not trigger anomaly, statistics, evidence
    assert res.anomaly is None
    assert res.statistics is None
    assert res.evidence is None


def test_10_immutable_request_and_result(test_csv: pathlib.Path):
    req = PipelineContributorAnalysisRequest(
        requests=(
            ContributorAnalysisRequest(
                analysis_id="c1",
                target_column="altitude_ft",
                factor_column="airspeed_kts",
                analysis_method=ContributorAnalysisMethod.PEARSON,
            ),
        )
    )
    res = run_csv_pipeline(test_csv, contributor_analysis_request=req)
    with pytest.raises(Exception):
        res.contributor_analysis.status = ResultStatus.SUCCESS
