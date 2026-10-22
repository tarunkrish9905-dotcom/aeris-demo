"""Tests for Pipeline Parameter Validation Integration."""

import pathlib

import pandas as pd
import pytest

from aeris.core.normalization.engine import NormalizationRequest
from aeris.core.parameter_validation.validator import ParameterValidationRequest
from aeris.core.pipeline.orchestrator import run_csv_pipeline
from aeris.core.registry import ParameterDataType, ParameterDefinition, ParameterRegistry
from aeris.core.results.types import ResultStatus


@pytest.fixture
def test_registry() -> ParameterRegistry:
    registry = ParameterRegistry()
    registry.register(
        ParameterDefinition(
            parameter_id="airspeed",
            display_name="Airspeed",
            description="True airspeed",
            physical_quantity="velocity",
            data_type=ParameterDataType.FLOAT,
            canonical_unit="knots",
            aliases=["airspeed_kts"],
        )
    )
    return registry


@pytest.fixture
def sample_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    csv_file = tmp_path / "flight_data.csv"
    csv_file.write_text(
        "timestamp,altitude_ft,airspeed_kts,pitch_deg,roll_deg,engine_rpm\n"
        "2026-10-06T10:00:00Z,10000,100.5,0.0,0.0,2400\n"
    )
    return csv_file


def test_validation_skipped_when_no_request(sample_csv: pathlib.Path) -> None:
    res = run_csv_pipeline(sample_csv)
    assert res.parameter_validation is None
    assert res.status == ResultStatus.SUCCESS


def test_validation_skipped_when_registry_exists_but_no_request(
    sample_csv: pathlib.Path, test_registry: ParameterRegistry
) -> None:
    res = run_csv_pipeline(sample_csv, parameter_registry=test_registry)
    assert res.parameter_validation is None
    assert res.status == ResultStatus.SUCCESS


def test_registry_required_error(sample_csv: pathlib.Path) -> None:
    req = ParameterValidationRequest(["airspeed"])
    res = run_csv_pipeline(sample_csv, parameter_validation_request=req)

    assert res.parameter_validation is not None
    assert res.parameter_validation.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_PARAMETER_VALIDATION_REGISTRY_REQUIRED" for i in res.issues)
    assert res.status == ResultStatus.ERROR


def test_successful_integrated_validation(
    sample_csv: pathlib.Path, test_registry: ParameterRegistry
) -> None:
    # "airspeed_kts" resolves to "airspeed"
    req = ParameterValidationRequest(["airspeed_kts"])
    res = run_csv_pipeline(
        sample_csv,
        parameter_registry=test_registry,
        parameter_validation_request=req,
    )

    assert res.parameter_validation is not None
    assert res.parameter_validation.status == ResultStatus.SUCCESS
    assert "airspeed_kts" in res.parameter_validation.validated_parameters
    assert res.status == ResultStatus.SUCCESS


def test_validation_after_normalization(
    sample_csv: pathlib.Path, test_registry: ParameterRegistry
) -> None:
    # 1. Normalize: rename "airspeed_kts" -> "airspeed" and convert to float
    norm_req = NormalizationRequest()
    norm_req.resolve_and_rename("airspeed_kts")
    norm_req.resolve_and_convert_type("airspeed_kts", float)

    # 2. Validate: check for "airspeed"
    val_req = ParameterValidationRequest(["airspeed"])

    res = run_csv_pipeline(
        sample_csv,
        normalization_request=norm_req,
        parameter_registry=test_registry,
        parameter_validation_request=val_req,
    )

    assert res.normalization is not None
    assert "airspeed" in res.normalization.data.columns

    assert res.parameter_validation is not None
    assert res.parameter_validation.status == ResultStatus.SUCCESS
    assert "airspeed" in res.parameter_validation.validated_parameters

    assert res.status == ResultStatus.SUCCESS

    # Provenance lineage check
    assert res.parameter_validation.provenance is not None
    assert res.normalization.provenance is not None
    assert (
        res.normalization.provenance.record_id
        in res.parameter_validation.provenance.parent_record_ids
    )


def test_validation_error_propagation(
    tmp_path: pathlib.Path, test_registry: ParameterRegistry
) -> None:
    # Create CSV with bad type (string that can't be parsed)
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text(
        "timestamp,altitude_ft,airspeed_kts,pitch_deg,roll_deg,engine_rpm\n"
        "2026-10-06T10:00:00Z,10000,not-a-number,0.0,0.0,2400\n"
    )

    req = ParameterValidationRequest(["airspeed_kts"])
    res = run_csv_pipeline(
        csv_file,
        parameter_registry=test_registry,
        parameter_validation_request=req,
    )

    assert res.parameter_validation is not None
    assert res.parameter_validation.status == ResultStatus.ERROR
    # Overall pipeline status should become ERROR
    assert res.status == ResultStatus.ERROR


def test_validation_warning_propagation(
    tmp_path: pathlib.Path, test_registry: ParameterRegistry
) -> None:
    # Create CSV with missing value -> generates warning
    csv_file = tmp_path / "warn.csv"
    csv_file.write_text(
        "timestamp,altitude_ft,airspeed_kts,pitch_deg,roll_deg,engine_rpm\n"
        "2026-10-06T10:00:00Z,10000,,0.0,0.0,2400\n"
    )

    req = ParameterValidationRequest(["airspeed_kts"])
    res = run_csv_pipeline(
        csv_file,
        parameter_registry=test_registry,
        parameter_validation_request=req,
    )

    assert res.parameter_validation is not None
    assert res.parameter_validation.status == ResultStatus.WARNING
    # Overall pipeline status should become WARNING
    assert res.status == ResultStatus.WARNING


def test_earlier_stage_results_preserved(
    sample_csv: pathlib.Path, test_registry: ParameterRegistry
) -> None:
    req = ParameterValidationRequest(["airspeed_kts"])
    res = run_csv_pipeline(
        sample_csv,
        parameter_registry=test_registry,
        parameter_validation_request=req,
    )

    # Check all earlier results are present
    assert res.ingestion is not None
    assert res.validation is not None
    assert res.quality is not None


def test_raw_dataframe_immutability(
    sample_csv: pathlib.Path, test_registry: ParameterRegistry
) -> None:
    req = ParameterValidationRequest(["airspeed_kts"])
    res = run_csv_pipeline(
        sample_csv,
        parameter_registry=test_registry,
        parameter_validation_request=req,
    )

    df_raw = pd.read_csv(sample_csv)
    pd.testing.assert_frame_equal(res.ingestion.data, df_raw)


def test_no_automatic_normalization_when_validation_requested(
    sample_csv: pathlib.Path, test_registry: ParameterRegistry
) -> None:
    """IMPORTANT NEGATIVE TEST: Requesting validation does not cause normalization."""
    req = ParameterValidationRequest(["airspeed_kts"])
    res = run_csv_pipeline(
        sample_csv,
        parameter_registry=test_registry,
        parameter_validation_request=req,
    )

    assert res.normalization is None
    # The dataframe is unchanged
    assert "airspeed_kts" in res.ingestion.data.columns
    assert "airspeed" not in res.ingestion.data.columns
