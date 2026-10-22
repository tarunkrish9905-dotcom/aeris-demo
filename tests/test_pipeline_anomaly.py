import pathlib

import dateutil.parser
import pandas as pd
import pytest

from aeris.core.anomaly.models import DetectionMethod, PipelineAnomalyDetectionResult
from aeris.core.anomaly.statistical import StatisticalAnomalyDetectionRequest
from aeris.core.anomaly.threshold import ThresholdDetectionRequest, ThresholdOperator, ThresholdRule
from aeris.core.normalization.engine import NormalizationRequest
from aeris.core.pipeline.orchestrator import (
    PipelineAnomalyDetectionRequest,
    TestPointAssociationRequest,
    run_csv_pipeline,
)
from aeris.core.results.types import ResultStatus
from aeris.core.test_operations.test_point_observation import TestPointObservation


@pytest.fixture
def dummy_csv(tmp_path: pathlib.Path) -> pathlib.Path:
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
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return path


def test_pipeline_without_anomaly_request(dummy_csv: pathlib.Path) -> None:
    res = run_csv_pipeline(dummy_csv)
    assert res.status != ResultStatus.ERROR
    assert res.anomaly is None


def test_pipeline_with_threshold_anomaly(dummy_csv: pathlib.Path) -> None:
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
    res = run_csv_pipeline(dummy_csv, anomaly_request=req)
    assert res.anomaly is not None
    assert isinstance(res.anomaly, PipelineAnomalyDetectionResult)
    assert res.anomaly.status == ResultStatus.SUCCESS

    det_res = res.anomaly.detector_results[0]
    assert len(det_res.anomalies) == 1
    assert det_res.anomalies[0].observed_value == 150.0


def test_pipeline_with_multiple_detectors(dummy_csv: pathlib.Path) -> None:
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
    res = run_csv_pipeline(dummy_csv, anomaly_request=req)
    assert len(res.anomaly.detector_results) == 2

    # Threshold res
    th_res = res.anomaly.detector_results[0]
    assert th_res.detection_method == DetectionMethod.RULE_BASED
    assert len(th_res.anomalies) == 1

    # Statistical res
    st_res = res.anomaly.detector_results[1]
    assert st_res.detection_method == DetectionMethod.STATISTICAL
    assert len(st_res.anomalies) == 1


def test_pipeline_detector_failure_isolation(dummy_csv: pathlib.Path) -> None:
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
                    analysis_id="st1", target_column="MISSING_COL", z_threshold=2.0
                )
            ]
        ),
    )
    res = run_csv_pipeline(dummy_csv, anomaly_request=req)
    assert res.status == ResultStatus.PARTIAL
    assert res.anomaly.status == ResultStatus.PARTIAL
    assert len(res.anomaly.detector_results) == 2
    assert res.anomaly.detector_results[0].status == ResultStatus.SUCCESS
    assert res.anomaly.detector_results[1].status == ResultStatus.ERROR


def test_pipeline_association_priority(dummy_csv: pathlib.Path) -> None:
    start_ts = dateutil.parser.isoparse("2026-01-01T12:00:02Z")
    end_ts = dateutil.parser.isoparse("2026-01-01T12:00:04Z")
    obs = TestPointObservation(
        test_point_id="tp1",
        source_reference="f1",
        observation_id="obs1",
        start_timestamp=start_ts,
        end_timestamp=end_ts,
    )

    assoc_req = TestPointAssociationRequest(observation=obs, timestamp_column="timestamp")

    req = PipelineAnomalyDetectionRequest(
        statistical_requests=tuple(
            [
                StatisticalAnomalyDetectionRequest(
                    analysis_id="st1", target_column="target", z_threshold=0.1
                )
            ]
        )
    )

    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")

    res = run_csv_pipeline(
        dummy_csv,
        normalization_request=norm_req,
        association_request=assoc_req,
        anomaly_request=req,
    )

    det_res = res.anomaly.detector_results[0]
    assert det_res.metadata["total_observations"] == 3
    assert res.association.provenance.record_id in det_res.provenance.parent_record_ids
