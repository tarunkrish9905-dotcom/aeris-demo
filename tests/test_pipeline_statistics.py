import pathlib
import uuid

import numpy as np
import pandas as pd
import pytest

from aeris.core.normalization.engine import NormalizationRequest
from aeris.core.pipeline.orchestrator import (
    DataPipelineOrchestrator,
    TestPointAssociationRequest,
)
from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.results.types import ResultStatus
from aeris.core.statistics.models import MetricType, StatisticalAnalysisRequest
from aeris.core.test_operations.test_point_observation import TestPointObservation


def create_mock_csv(tmp_path: pathlib.Path, df: pd.DataFrame) -> pathlib.Path:
    p = tmp_path / f"test_{uuid.uuid4().hex}.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-01T12:00:00Z", "2026-01-01T12:00:10Z", "2026-01-01T12:00:20Z"],
            "altitude_ft": [1000.0, 1100.0, 1200.0],
            "airspeed_kts": [100.0, 110.0, np.nan],
            "pitch_deg": [1.0, 1.0, 1.0],
            "roll_deg": [0.0, 0.0, 0.0],
            "engine_rpm": [2500.0, 2500.0, 2500.0],
        }
    )


def test_statistics_disabled_by_default(tmp_path: pathlib.Path, base_df: pd.DataFrame):
    csv = create_mock_csv(tmp_path, base_df)
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(csv)

    assert res.statistics is None


def test_statistics_explicitly_requested(tmp_path: pathlib.Path, base_df: pd.DataFrame):
    csv = create_mock_csv(tmp_path, base_df)
    req = StatisticalAnalysisRequest(
        analysis_id="stat-test",
        target_column="altitude_ft",
        requested_metrics=[MetricType.MEAN],
    )
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(csv, statistics_request=req)

    assert res.statistics is not None
    assert res.statistics.analysis_id == "stat-test"


def test_successful_statistical_analysis(tmp_path: pathlib.Path, base_df: pd.DataFrame):
    csv = create_mock_csv(tmp_path, base_df)
    req = StatisticalAnalysisRequest(
        analysis_id="stat-test",
        target_column="altitude_ft",
        requested_metrics=[MetricType.MEAN, MetricType.MAXIMUM],
    )
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(csv, statistics_request=req)

    assert res.status in (ResultStatus.SUCCESS, ResultStatus.WARNING)
    assert res.statistics.status == ResultStatus.SUCCESS
    mean_val = next(m.value for m in res.statistics.metrics if m.metric_type == MetricType.MEAN)
    assert mean_val == pytest.approx(1100.0)


def test_percentile_request(tmp_path: pathlib.Path, base_df: pd.DataFrame):
    csv = create_mock_csv(tmp_path, base_df)
    req = StatisticalAnalysisRequest(
        analysis_id="stat-test",
        target_column="altitude_ft",
        requested_metrics=[MetricType.PERCENTILE],
        requested_percentiles=[50.0],
    )
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(csv, statistics_request=req)

    assert res.statistics.status == ResultStatus.SUCCESS
    p50_val = next(
        m.value for m in res.statistics.metrics if m.metric_type == MetricType.PERCENTILE
    )
    assert p50_val == pytest.approx(1100.0)


def test_statistics_using_normalized_data(tmp_path: pathlib.Path, base_df: pd.DataFrame):
    csv = create_mock_csv(tmp_path, base_df)
    norm = NormalizationRequest()
    norm.rename_column("altitude_ft", "alt")
    req = StatisticalAnalysisRequest(
        analysis_id="stat-test",
        target_column="alt",
        requested_metrics=[MetricType.COUNT],
    )
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(csv, normalization_request=norm, statistics_request=req)

    assert res.normalization is not None
    assert res.statistics is not None
    assert res.statistics.status == ResultStatus.SUCCESS
    assert res.normalization.provenance.record_id in res.statistics.provenance.parent_record_ids


def test_statistics_using_associated_data(tmp_path: pathlib.Path, base_df: pd.DataFrame):
    csv = create_mock_csv(tmp_path, base_df)
    t_start = pd.Timestamp("2026-01-01T12:00:10Z").to_pydatetime()
    t_end = pd.Timestamp("2026-01-01T12:00:20Z").to_pydatetime()
    obs = TestPointObservation(
        test_point_id="tp-1",
        observation_id="obs-1",
        start_timestamp=t_start,
        end_timestamp=t_end,
        source_reference="f",
    )
    assoc_req = TestPointAssociationRequest(observation=obs, timestamp_column="timestamp")

    req = StatisticalAnalysisRequest(
        analysis_id="stat-test",
        target_column="altitude_ft",
        requested_metrics=[MetricType.COUNT],
    )
    norm = NormalizationRequest()
    norm.standardize_timestamp("timestamp")
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(
        csv, normalization_request=norm, association_request=assoc_req, statistics_request=req
    )

    assert res.association is not None
    assert res.statistics.status == ResultStatus.SUCCESS
    c_val = next(m.value for m in res.statistics.metrics if m.metric_type == MetricType.COUNT)
    assert c_val == 2.0  # Only rows 2 and 3 match


def test_normalization_failure_prevents_statistics_fallback(
    tmp_path: pathlib.Path, base_df: pd.DataFrame
):
    base_df.loc[0, "timestamp"] = "bad"
    csv = create_mock_csv(tmp_path, base_df)
    norm = NormalizationRequest()
    norm.standardize_timestamp("timestamp")

    req = StatisticalAnalysisRequest(
        analysis_id="stat-test",
        target_column="altitude_ft",
        requested_metrics=[MetricType.COUNT],
    )
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(csv, normalization_request=norm, statistics_request=req)

    assert res.normalization.status == ResultStatus.ERROR
    assert res.statistics.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_STATISTICS_DATA_UNAVAILABLE" for i in res.statistics.issues)


def test_association_failure_prevents_statistics_fallback(
    tmp_path: pathlib.Path, base_df: pd.DataFrame
):
    csv = create_mock_csv(tmp_path, base_df)
    t_start = pd.Timestamp("2027-01-01T12:00:00Z").to_pydatetime()
    t_end = pd.Timestamp("2027-01-01T12:00:10Z").to_pydatetime()
    obs = TestPointObservation(
        test_point_id="tp-1",
        observation_id="obs-1",
        start_timestamp=t_start,
        end_timestamp=t_end,
        source_reference="f",
    )
    assoc_req = TestPointAssociationRequest(observation=obs, timestamp_column="timestamp")

    req = StatisticalAnalysisRequest(
        analysis_id="stat-test",
        target_column="altitude_ft",
        requested_metrics=[MetricType.COUNT],
    )
    norm = NormalizationRequest()
    norm.standardize_timestamp("timestamp")
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(
        csv, normalization_request=norm, association_request=assoc_req, statistics_request=req
    )

    assert res.association.status == ResultStatus.NOT_AVAILABLE
    assert res.statistics.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_STATISTICS_DATA_UNAVAILABLE" for i in res.statistics.issues)


def test_missing_target_column(tmp_path: pathlib.Path, base_df: pd.DataFrame):
    csv = create_mock_csv(tmp_path, base_df)
    req = StatisticalAnalysisRequest(
        analysis_id="stat-test",
        target_column="does_not_exist",
        requested_metrics=[MetricType.COUNT],
    )
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(csv, statistics_request=req)

    assert res.statistics.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_TARGET_COLUMN_MISSING" for i in res.statistics.issues)


def test_upstream_data_immutability(tmp_path: pathlib.Path, base_df: pd.DataFrame):
    csv = create_mock_csv(tmp_path, base_df)
    req = StatisticalAnalysisRequest(
        analysis_id="stat-test",
        target_column="altitude_ft",
        requested_metrics=[MetricType.MEAN],
    )
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(csv, statistics_request=req)

    assert res.ingestion.data is not None
    # Ensure statistics doesn't add any columns or mutate
    assert "altitude_ft" in res.ingestion.data.columns
    assert len(res.ingestion.data.columns) == 6


def test_statistics_provenance(tmp_path: pathlib.Path, base_df: pd.DataFrame):
    csv = create_mock_csv(tmp_path, base_df)
    req = StatisticalAnalysisRequest(
        analysis_id="stat-test",
        target_column="altitude_ft",
        requested_metrics=[MetricType.COUNT],
    )
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(csv, statistics_request=req)

    assert res.statistics.epistemic_level == EpistemicLevel.CALCULATED
    assert res.statistics.provenance is not None
    assert res.ingestion.provenance.record_id in res.statistics.provenance.parent_record_ids


def test_independent_repeated_pipeline_executions(tmp_path: pathlib.Path, base_df: pd.DataFrame):
    csv = create_mock_csv(tmp_path, base_df)
    req1 = StatisticalAnalysisRequest(
        analysis_id="stat-1", target_column="altitude_ft", requested_metrics=[MetricType.MEAN]
    )
    req2 = StatisticalAnalysisRequest(
        analysis_id="stat-2",
        target_column="airspeed_kts",
        requested_metrics=[MetricType.MAXIMUM],
    )

    orch = DataPipelineOrchestrator()
    res1 = orch.run_csv_pipeline(csv, statistics_request=req1)
    res2 = orch.run_csv_pipeline(csv, statistics_request=req2)

    assert res1.statistics.analysis_id == "stat-1"
    assert res2.statistics.analysis_id == "stat-2"
    assert res1.statistics.target_column == "altitude_ft"
    assert res2.statistics.target_column == "airspeed_kts"
