import pathlib

import pandas as pd
import pytest

from aeris.core.anomaly.models import PipelineAnomalyDetectionResult
from aeris.core.anomaly.statistical import StatisticalAnomalyDetectionRequest
from aeris.core.anomaly.threshold import ThresholdDetectionRequest, ThresholdOperator, ThresholdRule
from aeris.core.anomaly.timeseries import TimeSeriesAnomalyDetectionRequest
from aeris.core.normalization.engine import NormalizationRequest
from aeris.core.pipeline.orchestrator import (
    PipelineAnomalyDetectionRequest,
    TestPointAssociationRequest,
    run_csv_pipeline,
)
from aeris.core.results.types import ResultStatus
from aeris.core.test_operations.test_point_observation import TestPointObservation


@pytest.fixture
def base_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T12:00:00Z",
                "2026-01-01T12:00:01Z",
                "2026-01-01T12:00:02Z",
                "2026-01-01T12:00:03Z",
                "2026-01-01T12:00:04Z",
                "2026-01-01T12:00:05Z",
            ],
            "target": [100.0, 101.0, 99.0, 100.0, 102.0, 150.0],
            "altitude_ft": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "airspeed_kts": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "pitch_deg": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "roll_deg": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "engine_rpm": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    path = tmp_path / "hardening.csv"
    df.to_csv(path, index=False)
    return path


def test_timeseries_execution_and_zero_anomalies(base_csv: pathlib.Path) -> None:
    req = PipelineAnomalyDetectionRequest(
        time_series_requests=tuple(
            [
                TimeSeriesAnomalyDetectionRequest(
                    analysis_id="ts1",
                    target_column="target",
                    timestamp_column="timestamp",
                    window_size=2,
                    z_threshold=100.0,  # High threshold to guarantee 0 anomalies
                )
            ]
        )
    )

    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")

    res = run_csv_pipeline(base_csv, normalization_request=norm_req, anomaly_request=req)

    assert res.anomaly is not None
    assert isinstance(res.anomaly, PipelineAnomalyDetectionResult)
    # The first 2 rows might be NOT_AVAILABLE for timeseries, meaning PARTIAL result.
    # Wait, TimeSeriesAnomalyEngine returns PARTIAL if any rows were unevaluable.
    # We just want to check there are 0 anomalies.
    assert len(res.anomaly.detector_results) == 1
    det_res = res.anomaly.detector_results[0]
    assert len(det_res.anomalies) == 0


def test_association_failure_no_fallback(base_csv: pathlib.Path) -> None:
    import dateutil.parser

    # Invalid association: start > end
    start_ts = dateutil.parser.isoparse("2026-01-01T12:00:10Z")
    end_ts = dateutil.parser.isoparse("2026-01-01T12:00:04Z")
    obs = TestPointObservation(
        test_point_id="tp1",
        source_reference="f1",
        observation_id="obs1",
        start_timestamp=end_ts,
        end_timestamp=start_ts,
    )

    assoc_req = TestPointAssociationRequest(observation=obs, timestamp_column="timestamp")

    req = PipelineAnomalyDetectionRequest(
        threshold_requests=tuple(
            [
                ThresholdDetectionRequest(
                    analysis_id="th1",
                    target_column="target",
                    rule=ThresholdRule(
                        rule_id="r1",
                        target_column="target",
                        operator=ThresholdOperator.GREATER_THAN,
                        threshold_value=120.0,
                    ),
                )
            ]
        )
    )

    res = run_csv_pipeline(base_csv, association_request=assoc_req, anomaly_request=req)

    assert res.association.status == ResultStatus.ERROR
    assert res.anomaly is not None
    assert res.anomaly.status == ResultStatus.ERROR
    assert "ERR_ASSOCIATION_FAILED" in res.anomaly.issues[0].issue_code
    assert len(res.anomaly.detector_results) == 0


def test_normalization_failure_no_fallback(base_csv: pathlib.Path) -> None:
    # Invalid normalization
    norm_req = NormalizationRequest()
    norm_req.convert_type("missing_column", float)

    req = PipelineAnomalyDetectionRequest(
        threshold_requests=tuple(
            [
                ThresholdDetectionRequest(
                    analysis_id="th1",
                    target_column="target",
                    rule=ThresholdRule(
                        rule_id="r1",
                        target_column="target",
                        operator=ThresholdOperator.GREATER_THAN,
                        threshold_value=120.0,
                    ),
                )
            ]
        )
    )

    res = run_csv_pipeline(base_csv, normalization_request=norm_req, anomaly_request=req)

    assert res.normalization.status == ResultStatus.ERROR
    assert res.anomaly is not None
    assert res.anomaly.status == ResultStatus.ERROR
    assert "ERR_NORMALIZATION_FAILED" in res.anomaly.issues[0].issue_code


def test_immutability_and_isolation(base_csv: pathlib.Path) -> None:
    req = PipelineAnomalyDetectionRequest(
        threshold_requests=tuple(
            [
                ThresholdDetectionRequest(
                    analysis_id="th1",
                    target_column="target",
                    rule=ThresholdRule(
                        rule_id="r1",
                        target_column="target",
                        operator=ThresholdOperator.GREATER_THAN,
                        threshold_value=120.0,
                    ),
                )
            ]
        ),
        statistical_requests=tuple(
            [
                StatisticalAnomalyDetectionRequest(
                    analysis_id="st1", target_column="target", z_threshold=2.0
                )
            ]
        ),
    )

    res1 = run_csv_pipeline(base_csv, anomaly_request=req)
    res2 = run_csv_pipeline(base_csv, anomaly_request=req)

    assert len(res1.anomaly.detector_results[0].anomalies) == len(
        res2.anomaly.detector_results[0].anomalies
    )
    assert len(res1.anomaly.detector_results[1].anomalies) == len(
        res2.anomaly.detector_results[1].anomalies
    )
