"""Tests for Explicit Test Point Association Pipeline Integration."""

import datetime
from collections.abc import Generator
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from aeris.core.normalization.engine import NormalizationRequest
from aeris.core.pipeline.orchestrator import TestPointAssociationRequest, run_csv_pipeline
from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.results.types import ResultStatus
from aeris.core.test_operations.test_point_observation import TestPointObservation


@pytest.fixture
def sample_csv() -> Generator[str, None, None]:
    with TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "flight.csv"
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2026-01-01T10:00:00Z",
                    "2026-01-01T10:00:01Z",
                    "2026-01-01T10:00:02Z",
                    "2026-01-01T10:00:03Z",
                    "2026-01-01T10:00:04Z",
                ],
                "altitude_ft": [1000, 1010, 1020, 1030, 1040],
                "airspeed_kts": [100, 110, 120, 130, 140],
                "pitch_deg": [1, 2, 3, 4, 5],
                "roll_deg": [1, 2, 3, 4, 5],
                "engine_rpm": [2000, 2000, 2000, 2000, 2000],
            }
        )
        df.to_csv(csv_path, index=False)
        yield str(csv_path)


@pytest.fixture
def empty_csv() -> Generator[str, None, None]:
    with TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "empty.csv"
        pd.DataFrame().to_csv(csv_path, index=False)
        yield str(csv_path)


@pytest.fixture
def naive_csv() -> Generator[str, None, None]:
    with TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "naive.csv"
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2026-01-01T10:00:00",
                    "2026-01-01T10:00:01",
                    "2026-01-01T10:00:02",
                ],
                "altitude_ft": [1000, 1010, 1020],
                "airspeed_kts": [100, 110, 120],
                "pitch_deg": [1, 2, 3],
                "roll_deg": [1, 2, 3],
                "engine_rpm": [2000, 2000, 2000],
            }
        )
        df.to_csv(csv_path, index=False)
        yield str(csv_path)


@pytest.fixture
def base_obs() -> TestPointObservation:
    t_start = datetime.datetime(2026, 1, 1, 10, 0, 1, tzinfo=datetime.UTC)
    t_end = datetime.datetime(2026, 1, 1, 10, 0, 3, tzinfo=datetime.UTC)
    return TestPointObservation(
        observation_id="OBS-1",
        test_point_id="TP-1",
        source_reference="clean_flight.csv",
        start_timestamp=t_start,
        end_timestamp=t_end,
    )


def test_association_not_requested_is_none(sample_csv: str) -> None:
    res = run_csv_pipeline(sample_csv)
    assert res.association is None
    assert res.status == ResultStatus.SUCCESS


def test_association_requested_executes(sample_csv: str, base_obs: TestPointObservation) -> None:
    req = TestPointAssociationRequest(observation=base_obs, timestamp_column="timestamp")
    res = run_csv_pipeline(sample_csv, association_request=req)
    assert res.association is not None
    assert res.association.status == ResultStatus.ERROR
    assert res.association.issues[0].issue_code == "ERR_TIMESTAMP_COLUMN_INVALID"


def test_successful_association_through_pipeline(
    sample_csv: str, base_obs: TestPointObservation
) -> None:
    req = TestPointAssociationRequest(observation=base_obs, timestamp_column="timestamp")
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, association_request=req)

    assert res.association is not None
    assert res.association.status == ResultStatus.SUCCESS
    assert res.status == ResultStatus.SUCCESS
    assert res.association.row_count == 3


def test_exact_start_boundary(sample_csv: str) -> None:
    t_start = datetime.datetime(2026, 1, 1, 10, 0, 0, tzinfo=datetime.UTC)
    t_end = datetime.datetime(2026, 1, 1, 10, 0, 0, tzinfo=datetime.UTC)
    obs = TestPointObservation(
        observation_id="O1",
        test_point_id="T1",
        source_reference="s",
        start_timestamp=t_start,
        end_timestamp=t_end,
    )
    req = TestPointAssociationRequest(observation=obs, timestamp_column="timestamp")
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, association_request=req)
    assert res.association is not None
    assert res.association.row_count == 1
    assert res.association.data["airspeed_kts"].iloc[0] == 100


def test_exact_end_boundary(sample_csv: str) -> None:
    t_start = datetime.datetime(2026, 1, 1, 10, 0, 4, tzinfo=datetime.UTC)
    t_end = datetime.datetime(2026, 1, 1, 10, 0, 4, tzinfo=datetime.UTC)
    obs = TestPointObservation(
        observation_id="O1",
        test_point_id="T1",
        source_reference="s",
        start_timestamp=t_start,
        end_timestamp=t_end,
    )
    req = TestPointAssociationRequest(observation=obs, timestamp_column="timestamp")
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, association_request=req)
    assert res.association is not None
    assert res.association.row_count == 1
    assert res.association.data["airspeed_kts"].iloc[0] == 140


def test_both_boundaries(sample_csv: str, base_obs: TestPointObservation) -> None:
    req = TestPointAssociationRequest(observation=base_obs, timestamp_column="timestamp")
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, association_request=req)
    assert res.association is not None
    assert res.association.row_count == 3
    assert res.association.data["timestamp"].iloc[0] == pd.Timestamp("2026-01-01T10:00:01Z")
    assert res.association.data["timestamp"].iloc[-1] == pd.Timestamp("2026-01-01T10:00:03Z")


def test_no_matching_rows(sample_csv: str) -> None:
    t_start = datetime.datetime(2026, 1, 1, 11, 0, 0, tzinfo=datetime.UTC)
    t_end = datetime.datetime(2026, 1, 1, 11, 0, 4, tzinfo=datetime.UTC)
    obs = TestPointObservation(
        observation_id="O1",
        test_point_id="T1",
        source_reference="s",
        start_timestamp=t_start,
        end_timestamp=t_end,
    )
    req = TestPointAssociationRequest(observation=obs, timestamp_column="timestamp")
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, association_request=req)
    assert res.association is not None
    assert res.association.status == ResultStatus.NOT_AVAILABLE
    assert res.status == ResultStatus.NOT_AVAILABLE  # Since association returned NOT_AVAILABLE


def test_normalized_timestamp_used(sample_csv: str, base_obs: TestPointObservation) -> None:
    req = TestPointAssociationRequest(observation=base_obs, timestamp_column="std_timestamp")
    norm_req = NormalizationRequest()
    norm_req.rename_column("timestamp", "std_timestamp")
    norm_req.standardize_timestamp("std_timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, association_request=req)
    assert res.association is not None
    assert res.association.status == ResultStatus.SUCCESS


def test_raw_string_timestamp_rejected_when_normalization_not_requested(
    sample_csv: str, base_obs: TestPointObservation
) -> None:
    req = TestPointAssociationRequest(observation=base_obs, timestamp_column="timestamp")
    res = run_csv_pipeline(sample_csv, association_request=req)
    assert res.association is not None
    assert res.association.status == ResultStatus.ERROR
    assert res.association.issues[0].issue_code == "ERR_TIMESTAMP_COLUMN_INVALID"


def test_invalid_timestamp_representation(naive_csv: str, base_obs: TestPointObservation) -> None:
    req = TestPointAssociationRequest(observation=base_obs, timestamp_column="timestamp")
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(naive_csv, normalization_request=norm_req, association_request=req)
    # The normalization itself converts to UTC, so this wouldn't trigger ERR_TIMESTAMP_NAIVE
    # Wait, pandas `pd.to_datetime` automatically converts naives if `utc=True` is used.
    # The test in `test_test_point_association` was for purely naive timestamps.
    # Since `normalize_flight_data` explicitly converts them to UTC properly, association succeeds!
    assert res.association is not None
    assert res.association.status == ResultStatus.SUCCESS


def test_missing_timestamp_column(sample_csv: str, base_obs: TestPointObservation) -> None:
    req = TestPointAssociationRequest(observation=base_obs, timestamp_column="missing_col")
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, association_request=req)
    assert res.association is not None
    assert res.association.status == ResultStatus.ERROR
    assert res.association.issues[0].issue_code == "ERR_TIMESTAMP_COLUMN_MISSING"


def test_empty_input(empty_csv: str, base_obs: TestPointObservation) -> None:
    req = TestPointAssociationRequest(observation=base_obs, timestamp_column="timestamp")
    res = run_csv_pipeline(empty_csv, association_request=req)
    assert res.status == ResultStatus.ERROR
    assert res.association is not None
    assert res.association.status == ResultStatus.ERROR
    assert res.association.issues[0].issue_code == "ERR_ASSOCIATION_DATA_UNAVAILABLE"


def test_association_failure_propagation(sample_csv: str, base_obs: TestPointObservation) -> None:
    req = TestPointAssociationRequest(observation=base_obs, timestamp_column="missing_col")
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, association_request=req)
    assert res.association is not None
    assert res.association.status == ResultStatus.ERROR
    assert res.status == ResultStatus.ERROR


def test_previous_pipeline_results_preserved_when_association_fails(
    sample_csv: str, base_obs: TestPointObservation
) -> None:
    req = TestPointAssociationRequest(observation=base_obs, timestamp_column="missing_col")
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, association_request=req)
    assert res.normalization is not None
    assert res.normalization.status == ResultStatus.SUCCESS


def test_association_does_not_require_parameter_registry(
    sample_csv: str, base_obs: TestPointObservation
) -> None:
    req = TestPointAssociationRequest(observation=base_obs, timestamp_column="timestamp")
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, association_request=req)
    assert res.association is not None
    assert res.association.status == ResultStatus.SUCCESS
    assert res.parameter_validation is None


def test_parameter_validation_and_association_coexist(
    sample_csv: str, base_obs: TestPointObservation
) -> None:
    from aeris.core.parameter_validation.validator import ParameterValidationRequest
    from aeris.core.registry import ParameterRegistry
    from aeris.core.registry.models import ParameterDataType, ParameterDefinition

    registry = ParameterRegistry()
    registry.register(
        ParameterDefinition(
            parameter_id="airspeed_kts",
            physical_quantity="speed",
            data_type=ParameterDataType.INTEGER,
            display_name="Speed",
            description="Aircraft Speed",
        )
    )

    req_assoc = TestPointAssociationRequest(observation=base_obs, timestamp_column="timestamp")
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    pval_req = ParameterValidationRequest(requested_parameters=["airspeed_kts"])

    res = run_csv_pipeline(
        sample_csv,
        normalization_request=norm_req,
        parameter_registry=registry,
        parameter_validation_request=pval_req,
        association_request=req_assoc,
    )

    assert res.association is not None
    assert res.association.status == ResultStatus.SUCCESS
    assert res.parameter_validation is not None
    assert res.parameter_validation.status == ResultStatus.SUCCESS


def test_source_dataframe_remains_unchanged(
    sample_csv: str, base_obs: TestPointObservation
) -> None:
    req = TestPointAssociationRequest(observation=base_obs, timestamp_column="timestamp")
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, association_request=req)

    assert res.normalization is not None
    original_len = len(res.normalization.data)

    # Mutating association data
    assert res.association is not None
    assert res.association.data is not None
    res.association.data.loc[:, "airspeed_kts"] = 999

    assert len(res.normalization.data) == original_len
    assert (res.normalization.data["airspeed_kts"] != 999).all()


def test_extracted_dataframe_independence(sample_csv: str, base_obs: TestPointObservation) -> None:
    req = TestPointAssociationRequest(observation=base_obs, timestamp_column="timestamp")
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, association_request=req)

    assert res.normalization is not None
    assert res.association is not None

    res.normalization.data.loc[:, "airspeed_kts"] = 888
    assert (res.association.data["airspeed_kts"] != 888).all()


def test_explicit_timestamp_column_required(base_obs: TestPointObservation) -> None:
    with pytest.raises(ValueError):
        TestPointAssociationRequest(observation=base_obs)  # type: ignore[call-arg]


def test_provenance_linkage_exists(sample_csv: str, base_obs: TestPointObservation) -> None:
    req = TestPointAssociationRequest(observation=base_obs, timestamp_column="timestamp")
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, association_request=req)

    assert res.association is not None
    assert res.association.provenance is not None
    assert res.association.provenance.epistemic_level == EpistemicLevel.CALCULATED


def test_normalized_upstream_provenance_links_correctly(
    sample_csv: str, base_obs: TestPointObservation
) -> None:
    req = TestPointAssociationRequest(observation=base_obs, timestamp_column="timestamp")
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, association_request=req)

    assert res.normalization is not None
    assert res.normalization.provenance is not None
    assert res.association is not None
    assert res.association.provenance is not None
    assert res.normalization.provenance.record_id in res.association.provenance.parent_record_ids


def test_association_result_metadata_preserved(
    sample_csv: str, base_obs: TestPointObservation
) -> None:
    req = TestPointAssociationRequest(observation=base_obs, timestamp_column="timestamp")
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, association_request=req)

    assert res.association is not None
    assert res.association.test_point_id == "TP-1"
    assert res.association.observation_id == "OBS-1"


def test_no_automatic_timestamp_inference(sample_csv: str, base_obs: TestPointObservation) -> None:
    req = TestPointAssociationRequest(observation=base_obs, timestamp_column="not_timestamp")
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req, association_request=req)
    assert res.association is not None
    assert res.association.status == ResultStatus.ERROR


def test_no_automatic_sorting(base_obs: TestPointObservation) -> None:
    with TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "unordered.csv"
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2026-01-01T10:00:02Z",
                    "2026-01-01T10:00:01Z",
                    "2026-01-01T10:00:03Z",
                ],
                "airspeed_kts": [10, 20, 30],
                "altitude_ft": [10, 20, 30],
                "pitch_deg": [1, 2, 3],
                "roll_deg": [1, 2, 3],
                "engine_rpm": [2000, 2000, 2000],
            }
        )
        df.to_csv(csv_path, index=False)
        req = TestPointAssociationRequest(observation=base_obs, timestamp_column="timestamp")
        norm_req = NormalizationRequest()
        norm_req.standardize_timestamp("timestamp")
        res = run_csv_pipeline(
            str(csv_path), normalization_request=norm_req, association_request=req
        )

        assert res.association is not None
        assert res.association.status == ResultStatus.SUCCESS
        assert res.association.data["airspeed_kts"].tolist() == [10, 20, 30]


def test_no_automatic_deduplication(base_obs: TestPointObservation) -> None:
    with TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "dupes.csv"
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2026-01-01T10:00:02Z",
                    "2026-01-01T10:00:02Z",
                ],
                "airspeed_kts": [10, 20],
                "altitude_ft": [10, 20],
                "pitch_deg": [1, 2],
                "roll_deg": [1, 2],
                "engine_rpm": [2000, 2000],
            }
        )
        df.to_csv(csv_path, index=False)
        req = TestPointAssociationRequest(observation=base_obs, timestamp_column="timestamp")
        norm_req = NormalizationRequest()
        norm_req.standardize_timestamp("timestamp")
        res = run_csv_pipeline(
            str(csv_path), normalization_request=norm_req, association_request=req
        )

        assert res.association is not None
        assert res.association.status == ResultStatus.SUCCESS
        assert res.association.row_count == 2
        assert res.association.data["airspeed_kts"].tolist() == [10, 20]


def test_existing_pipeline_behavior_unchanged_without_association(sample_csv: str) -> None:
    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    res = run_csv_pipeline(sample_csv, normalization_request=norm_req)
    assert res.status == ResultStatus.SUCCESS
    assert res.association is None
    assert res.normalization is not None
    assert res.normalization.status == ResultStatus.SUCCESS
