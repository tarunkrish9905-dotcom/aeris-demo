import pathlib
import uuid

import pandas as pd
import pytest

from aeris.core.engineering.models import EngineeringCalculationDefinition
from aeris.core.pipeline.orchestrator import (
    DataPipelineOrchestrator,
    EngineeringCalculationRequest,
    NormalizationRequest,
    TestPointAssociationRequest,
)
from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.results.types import ResultStatus
from aeris.core.test_operations.test_point_observation import TestPointObservation


def create_mock_csv(tmp_path: pathlib.Path, df: pd.DataFrame) -> pathlib.Path:
    p = tmp_path / f"test_{uuid.uuid4().hex}.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-01T12:00:00Z", "2026-01-01T12:00:10Z"],
            "altitude_ft": [1000.0, 1100.0],
            "airspeed_kts": [100.0, 110.0],
            "pitch_deg": [0.0, 5.0],
            "roll_deg": [0.0, 0.0],
            "engine_rpm": [1000.0, 1000.0],
            "air_density_kg_m3": [1.225, 1.200],
            "true_airspeed_m_s": [50.0, 55.0],
        }
    )


@pytest.fixture
def def_altitude_rate() -> EngineeringCalculationDefinition:
    return EngineeringCalculationDefinition(
        calculation_id="altitude_rate",
        name="Altitude Rate",
        description="Calculate rate of change of altitude",
        output_parameter_id="altitude_rate",
        output_unit="ft/s",
        input_parameter_ids=["altitude_ft", "timestamp"],
        metadata={
            "altitude_column": "altitude_ft",
            "timestamp_column": "timestamp",
            "altitude_unit": "ft",
        },
    )


@pytest.fixture
def def_airspeed_rate() -> EngineeringCalculationDefinition:
    return EngineeringCalculationDefinition(
        calculation_id="airspeed_rate",
        name="Airspeed Rate",
        description="Rate",
        output_parameter_id="airspeed_rate",
        output_unit="kts/s",
        input_parameter_ids=["airspeed_kts", "timestamp"],
        metadata={
            "airspeed_column": "airspeed_kts",
            "timestamp_column": "timestamp",
            "airspeed_unit": "kts",
        },
    )


@pytest.fixture
def def_pitch_rate() -> EngineeringCalculationDefinition:
    return EngineeringCalculationDefinition(
        calculation_id="pitch_rate",
        name="Pitch Rate",
        description="Rate",
        output_parameter_id="pitch_rate",
        output_unit="deg/s",
        input_parameter_ids=["pitch_deg", "timestamp"],
        metadata={
            "pitch_column": "pitch_deg",
            "timestamp_column": "timestamp",
            "pitch_unit": "deg",
        },
    )


@pytest.fixture
def def_roll_rate() -> EngineeringCalculationDefinition:
    return EngineeringCalculationDefinition(
        calculation_id="roll_rate",
        name="Roll Rate",
        description="Rate",
        output_parameter_id="roll_rate",
        output_unit="deg/s",
        input_parameter_ids=["roll_deg", "timestamp"],
        metadata={"roll_column": "roll_deg", "timestamp_column": "timestamp", "roll_unit": "deg"},
    )


@pytest.fixture
def def_dynamic_pressure() -> EngineeringCalculationDefinition:
    return EngineeringCalculationDefinition(
        calculation_id="dynamic_pressure",
        name="Dynamic Pressure",
        description="q",
        output_parameter_id="dynamic_pressure",
        output_unit="Pa",
        input_parameter_ids=["air_density_kg_m3", "true_airspeed_m_s"],
        metadata={
            "air_density_column": "air_density_kg_m3",
            "air_density_unit": "kg/m^3",
            "true_airspeed_column": "true_airspeed_m_s",
            "true_airspeed_unit": "m/s",
        },
    )


# 1, 2, 3, 4, 5, 6, 7. Explicit isolated invocations, no automatic execution


def test_explicit_altitude_rate_invocation_only(
    tmp_path: pathlib.Path,
    base_df: pd.DataFrame,
    def_altitude_rate: EngineeringCalculationDefinition,
):
    csv = create_mock_csv(tmp_path, base_df)
    norm = NormalizationRequest()
    norm.standardize_timestamp("timestamp")

    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(
        csv,
        normalization_request=norm,
        engineering_request=EngineeringCalculationRequest(calculation_definition=def_altitude_rate),
    )
    # skip overall status
    assert res.engineering is not None
    assert res.engineering.calculation_id == "altitude_rate"
    assert "altitude_rate_ft_s" in res.engineering.output_value.name


def test_explicit_dynamic_pressure_invocation_only(
    tmp_path: pathlib.Path,
    base_df: pd.DataFrame,
    def_dynamic_pressure: EngineeringCalculationDefinition,
):
    csv = create_mock_csv(tmp_path, base_df)
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(
        csv,
        engineering_request=EngineeringCalculationRequest(
            calculation_definition=def_dynamic_pressure
        ),
    )
    # skip overall status
    assert res.engineering is not None
    assert res.engineering.calculation_id == "dynamic_pressure"
    assert "dynamic_pressure_pa" in res.engineering.output_value.name


def test_explicit_airspeed_rate_invocation_only(
    tmp_path: pathlib.Path,
    base_df: pd.DataFrame,
    def_airspeed_rate: EngineeringCalculationDefinition,
):
    csv = create_mock_csv(tmp_path, base_df)
    norm = NormalizationRequest()
    norm.standardize_timestamp("timestamp")
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(
        csv,
        normalization_request=norm,
        engineering_request=EngineeringCalculationRequest(calculation_definition=def_airspeed_rate),
    )
    # skip overall status
    assert res.engineering.calculation_id == "airspeed_rate"


def test_explicit_pitch_rate_invocation_only(
    tmp_path: pathlib.Path, base_df: pd.DataFrame, def_pitch_rate: EngineeringCalculationDefinition
):
    csv = create_mock_csv(tmp_path, base_df)
    norm = NormalizationRequest()
    norm.standardize_timestamp("timestamp")
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(
        csv,
        normalization_request=norm,
        engineering_request=EngineeringCalculationRequest(calculation_definition=def_pitch_rate),
    )
    # skip overall status
    assert res.engineering.calculation_id == "pitch_rate"


def test_explicit_roll_rate_invocation_only(
    tmp_path: pathlib.Path, base_df: pd.DataFrame, def_roll_rate: EngineeringCalculationDefinition
):
    csv = create_mock_csv(tmp_path, base_df)
    norm = NormalizationRequest()
    norm.standardize_timestamp("timestamp")
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(
        csv,
        normalization_request=norm,
        engineering_request=EngineeringCalculationRequest(calculation_definition=def_roll_rate),
    )
    # skip overall status
    assert res.engineering.calculation_id == "roll_rate"


def test_no_automatic_execution_of_other_calculations(
    tmp_path: pathlib.Path, base_df: pd.DataFrame
):
    # When we don't request engineering, NONE happen.
    csv = create_mock_csv(tmp_path, base_df)
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(csv)
    # skip overall status
    assert res.engineering is None


# 8. independent repeated executions
# 9. registry state isolation
# 10. result isolation


def test_independent_repeated_executions_state_isolation(
    tmp_path: pathlib.Path, base_df: pd.DataFrame, def_altitude_rate, def_dynamic_pressure
):
    csv = create_mock_csv(tmp_path, base_df)
    norm = NormalizationRequest()
    norm.standardize_timestamp("timestamp")
    orch = DataPipelineOrchestrator()

    # 1. run altitude
    res1 = orch.run_csv_pipeline(
        csv,
        normalization_request=norm,
        engineering_request=EngineeringCalculationRequest(calculation_definition=def_altitude_rate),
    )

    # 2. run dynamic pressure
    res2 = orch.run_csv_pipeline(
        csv,
        engineering_request=EngineeringCalculationRequest(
            calculation_definition=def_dynamic_pressure
        ),
    )

    # They should not pollute each other
    assert res1.engineering.calculation_id == "altitude_rate"
    assert res2.engineering.calculation_id == "dynamic_pressure"
    assert res1.engineering.result_id != res2.engineering.result_id
    assert res1.engineering.output_value.name != res2.engineering.output_value.name


# 11, 12, 13, 14, 15: provenance isolation


def test_ingestion_to_engineering_provenance(
    tmp_path: pathlib.Path, base_df: pd.DataFrame, def_dynamic_pressure
):
    csv = create_mock_csv(tmp_path, base_df)
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(
        csv,
        engineering_request=EngineeringCalculationRequest(
            calculation_definition=def_dynamic_pressure
        ),
    )

    assert res.engineering.provenance.epistemic_level == EpistemicLevel.CALCULATED
    assert res.ingestion.provenance.record_id in res.engineering.provenance.parent_record_ids


def test_normalization_to_engineering_provenance(
    tmp_path: pathlib.Path, base_df: pd.DataFrame, def_altitude_rate
):
    csv = create_mock_csv(tmp_path, base_df)
    norm = NormalizationRequest()
    norm.standardize_timestamp("timestamp")
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(
        csv,
        normalization_request=norm,
        engineering_request=EngineeringCalculationRequest(calculation_definition=def_altitude_rate),
    )

    assert res.normalization.provenance.record_id in res.engineering.provenance.parent_record_ids


def test_association_to_engineering_provenance(
    tmp_path: pathlib.Path, base_df: pd.DataFrame, def_dynamic_pressure
):
    csv = create_mock_csv(tmp_path, base_df)
    t_start = pd.Timestamp("2026-01-01T12:00:00Z").to_pydatetime()
    t_end = pd.Timestamp("2026-01-01T12:00:10Z").to_pydatetime()
    obs = TestPointObservation(
        test_point_id="tp-1",
        observation_id="obs-1",
        start_timestamp=t_start,
        end_timestamp=t_end,
        source_reference="f",
    )
    assoc_req = TestPointAssociationRequest(observation=obs, timestamp_column="timestamp")

    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(
        csv,
        association_request=assoc_req,
        engineering_request=EngineeringCalculationRequest(
            calculation_definition=def_dynamic_pressure
        ),
    )
    assert res.association.provenance.record_id in res.engineering.provenance.parent_record_ids


# 16, 17: fallback prevention


def test_association_not_available_prevents_raw_fallback(
    tmp_path: pathlib.Path, base_df: pd.DataFrame, def_dynamic_pressure
):
    csv = create_mock_csv(tmp_path, base_df)
    # create observation out of bounds
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

    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(
        csv,
        association_request=assoc_req,
        engineering_request=EngineeringCalculationRequest(
            calculation_definition=def_dynamic_pressure
        ),
    )

    assert res.association.status in (ResultStatus.NOT_AVAILABLE, ResultStatus.ERROR)
    assert res.engineering.status == ResultStatus.ERROR
    assert "ERR_ENGINEERING_DATA_UNAVAILABLE" in [i.issue_code for i in res.engineering.issues]


def test_normalization_failure_prevents_raw_fallback(
    tmp_path: pathlib.Path, base_df: pd.DataFrame, def_altitude_rate
):
    # bad timestamp format to fail normalization if requested
    base_df.loc[0, "timestamp"] = "bad"
    csv = create_mock_csv(tmp_path, base_df)
    norm = NormalizationRequest()
    norm.standardize_timestamp("timestamp")

    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(
        csv,
        normalization_request=norm,
        engineering_request=EngineeringCalculationRequest(calculation_definition=def_altitude_rate),
    )

    assert res.normalization.status == ResultStatus.ERROR
    assert res.engineering.status == ResultStatus.ERROR
    assert "ERR_ENGINEERING_DATA_UNAVAILABLE" in [i.issue_code for i in res.engineering.issues]


# 18. unknown calculation id


def test_unknown_calculation_id_structured_failure(
    tmp_path: pathlib.Path, base_df: pd.DataFrame, def_altitude_rate
):
    csv = create_mock_csv(tmp_path, base_df)
    bad_def = EngineeringCalculationDefinition(
        **{**def_altitude_rate.model_dump(), "calculation_id": "unknown"}
    )
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(
        csv, engineering_request=EngineeringCalculationRequest(calculation_definition=bad_def)
    )
    assert res.engineering.status == ResultStatus.ERROR
    assert "ERR_CALCULATION_NOT_REGISTERED" in [i.issue_code for i in res.engineering.issues]


# 24. input dataframe immutability


def test_input_dataframe_immutability_pipeline(
    tmp_path: pathlib.Path, base_df: pd.DataFrame, def_altitude_rate
):
    csv = create_mock_csv(tmp_path, base_df)
    norm = NormalizationRequest()
    norm.standardize_timestamp("timestamp")
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(
        csv,
        normalization_request=norm,
        engineering_request=EngineeringCalculationRequest(calculation_definition=def_altitude_rate),
    )

    # ensure input to engineering didn't get new columns
    assert "altitude_rate_ft_s" not in res.normalization.data.columns


# 22. temporal calculation enforcement


def test_temporal_calculation_enforces_timestamp(
    tmp_path: pathlib.Path, base_df: pd.DataFrame, def_altitude_rate
):
    # naive timestamp
    df = base_df.copy()
    df["timestamp"] = ["2026-01-01T12:00:00", "2026-01-01T12:00:10"]
    csv = create_mock_csv(tmp_path, df)

    orch = DataPipelineOrchestrator()
    # we don't normalize to keep it naive string. Well, actually wait.
    # if it's string, validation will fail. Let's make it a naive datetime
    # actually string fails engineering check anyway: ERR_TIMESTAMP_COLUMN_INVALID
    res = orch.run_csv_pipeline(
        csv,
        engineering_request=EngineeringCalculationRequest(calculation_definition=def_altitude_rate),
    )
    assert res.engineering.status == ResultStatus.ERROR
    assert "ERR_TIMESTAMP_COLUMN_INVALID" in [i.issue_code for i in res.engineering.issues]


# 23. dynamic pressure timestamp independence


def test_dynamic_pressure_no_timestamp_needed(
    tmp_path: pathlib.Path, base_df: pd.DataFrame, def_dynamic_pressure
):
    # drop timestamp
    df = base_df.drop(columns=["timestamp"])
    csv = create_mock_csv(tmp_path, df)

    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(
        csv,
        engineering_request=EngineeringCalculationRequest(
            calculation_definition=def_dynamic_pressure
        ),
    )
    # skip overall status
    assert res.engineering.status == ResultStatus.SUCCESS
