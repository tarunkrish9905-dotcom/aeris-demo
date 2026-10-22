import pathlib

import pandas as pd
import pytest

from aeris.core.engineering.models import EngineeringCalculationDefinition
from aeris.core.normalization.engine import NormalizationRequest
from aeris.core.pipeline.orchestrator import (
    DataPipelineOrchestrator,
    EngineeringCalculationRequest,
    TestPointAssociationRequest,
)
from aeris.core.results.types import ResultStatus
from aeris.core.test_operations.test_point_observation import TestPointObservation


def create_mock_csv(tmp_path: pathlib.Path) -> str:
    # 4 rows, monotonic UTC timestamp
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T12:00:00Z",
                "2026-01-01T12:00:10Z",
                "2026-01-01T12:00:20Z",
                "2026-01-01T12:00:30Z",
            ],
            "altitude_ft": [1000.0, 1100.0, 1300.0, 1200.0],
            "airspeed_kts": [100.0, 100.0, 100.0, 100.0],
            "pitch_deg": [0.0, 0.0, 0.0, 0.0],
            "roll_deg": [0.0, 0.0, 0.0, 0.0],
            "engine_rpm": [1000.0, 1000.0, 1000.0, 1000.0],
        }
    )
    path = tmp_path / "test_flight.csv"
    df.to_csv(path, index=False)
    return str(path)


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


def test_engineering_not_requested(tmp_path: pathlib.Path) -> None:
    csv = create_mock_csv(tmp_path)
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(csv)

    assert res.status == ResultStatus.SUCCESS
    assert res.engineering is None


def test_engineering_requested_executes(
    tmp_path: pathlib.Path, def_altitude_rate: EngineeringCalculationDefinition
) -> None:
    csv = create_mock_csv(tmp_path)
    orch = DataPipelineOrchestrator()

    req = EngineeringCalculationRequest(calculation_definition=def_altitude_rate)
    norm = NormalizationRequest()
    norm.standardize_timestamp("timestamp")
    res = orch.run_csv_pipeline(csv, engineering_request=req, normalization_request=norm)

    assert res.status == ResultStatus.SUCCESS
    assert res.engineering is not None
    assert res.engineering.status == ResultStatus.SUCCESS

    out = res.engineering.output_value
    assert isinstance(out, pd.Series)
    assert len(out) == 4
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == 10.0  # (1100-1000)/10


def test_unknown_calculation_id(tmp_path: pathlib.Path) -> None:
    csv = create_mock_csv(tmp_path)
    orch = DataPipelineOrchestrator()

    req = EngineeringCalculationRequest(
        calculation_definition=EngineeringCalculationDefinition(
            calculation_id="unknown_magic_calc",
            name="U",
            description="U",
            output_parameter_id="u",
            output_unit="u",
            input_parameter_ids=["altitude_ft"],
        )
    )
    res = orch.run_csv_pipeline(csv, engineering_request=req)

    assert res.status == ResultStatus.ERROR
    assert res.engineering is not None
    assert res.engineering.status == ResultStatus.ERROR
    assert "ERR_CALCULATION_NOT_REGISTERED" in [i.issue_code for i in res.engineering.issues]


def test_missing_altitude_column(tmp_path: pathlib.Path) -> None:
    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T12:00:00Z", "2026-01-01T12:00:10Z"],
            "airspeed_kts": [100.0, 100.0],
            "pitch_deg": [0.0, 0.0],
            "roll_deg": [0.0, 0.0],
            "engine_rpm": [1000.0, 1000.0],
        }
    )
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    orch = DataPipelineOrchestrator()
    req = EngineeringCalculationRequest(
        calculation_definition=EngineeringCalculationDefinition(
            calculation_id="altitude_rate",
            name="A",
            description="A",
            output_parameter_id="A",
            output_unit="ft/s",
            input_parameter_ids=["altitude_ft"],
            metadata={
                "altitude_column": "some_missing_column",
                "timestamp_column": "timestamp",
                "altitude_unit": "ft",
            },
        )
    )
    res = orch.run_csv_pipeline(str(path), engineering_request=req)

    assert res.status == ResultStatus.ERROR
    assert res.engineering is not None
    assert "ERR_ALTITUDE_COLUMN_MISSING" in [i.issue_code for i in res.engineering.issues]


@pytest.mark.parametrize("i", range(32))
def test_association_data_flow(
    i: int, tmp_path: pathlib.Path, def_altitude_rate: EngineeringCalculationDefinition
) -> None:
    csv = create_mock_csv(tmp_path)
    orch = DataPipelineOrchestrator()

    # Associate to the middle two points
    t_start = pd.Timestamp("2026-01-01T12:00:10Z").to_pydatetime()
    t_end = pd.Timestamp("2026-01-01T12:00:20Z").to_pydatetime()
    obs = TestPointObservation(
        test_point_id="tp-1",
        observation_id="obs-1",
        start_timestamp=t_start,
        end_timestamp=t_end,
        source_reference="test_flight.csv",
    )
    assoc_req = TestPointAssociationRequest(observation=obs, timestamp_column="timestamp")

    eng_req = EngineeringCalculationRequest(calculation_definition=def_altitude_rate)

    norm = NormalizationRequest()
    norm.standardize_timestamp("timestamp")
    res = orch.run_csv_pipeline(
        csv, association_request=assoc_req, engineering_request=eng_req, normalization_request=norm
    )

    assert res.status == ResultStatus.SUCCESS
    assert res.association.row_count == 2
    assert len(res.engineering.output_value) == 2
    # The first element is NaN, the second is (1300-1100)/10 = 20.0
    assert res.engineering.output_value.iloc[1] == 20.0


def test_normalization_data_flow(
    tmp_path: pathlib.Path, def_altitude_rate: EngineeringCalculationDefinition
) -> None:
    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T12:00:00Z", "2026-01-01T12:00:10Z"],
            "altitude_ft": [1000.0, 1100.0],
            "airspeed_kts": [100.0, 100.0],
            "pitch_deg": [0.0, 0.0],
            "roll_deg": [0.0, 0.0],
            "engine_rpm": [1000.0, 1000.0],
        }
    )
    path = tmp_path / "test2.csv"
    df.to_csv(path, index=False)

    norm_req = NormalizationRequest()
    norm_req.standardize_timestamp("timestamp")
    norm_req.renames = {"airspeed_kts": "airspeed"}

    eng_req = EngineeringCalculationRequest(calculation_definition=def_altitude_rate)

    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(
        str(path), normalization_request=norm_req, engineering_request=eng_req
    )

    assert res.status == ResultStatus.SUCCESS
    assert res.engineering.status == ResultStatus.SUCCESS


def test_failed_association_prevents_fabricated_engineering(
    tmp_path: pathlib.Path, def_altitude_rate: EngineeringCalculationDefinition
) -> None:
    csv = create_mock_csv(tmp_path)
    orch = DataPipelineOrchestrator()

    # Associate to an interval outside the dataset
    t_start = pd.Timestamp("2026-01-02T12:00:00Z").to_pydatetime()
    t_end = pd.Timestamp("2026-01-02T12:00:10Z").to_pydatetime()
    obs = TestPointObservation(
        test_point_id="tp-1",
        observation_id="obs-1",
        start_timestamp=t_start,
        end_timestamp=t_end,
        source_reference="test_flight.csv",
    )
    assoc_req = TestPointAssociationRequest(observation=obs, timestamp_column="timestamp")

    eng_req = EngineeringCalculationRequest(calculation_definition=def_altitude_rate)

    norm = NormalizationRequest()
    norm.standardize_timestamp("timestamp")
    res = orch.run_csv_pipeline(
        csv, association_request=assoc_req, engineering_request=eng_req, normalization_request=norm
    )

    assert res.status == ResultStatus.ERROR
    assert res.association.status == ResultStatus.NOT_AVAILABLE

    assert res.engineering.status == ResultStatus.ERROR
    assert "ERR_ENGINEERING_DATA_UNAVAILABLE" in [i.issue_code for i in res.engineering.issues]


def test_failed_normalization_prevents_engineering(
    tmp_path: pathlib.Path, def_altitude_rate: EngineeringCalculationDefinition
) -> None:
    csv = create_mock_csv(tmp_path)
    orch = DataPipelineOrchestrator()

    # Request missing column
    norm_req = NormalizationRequest()
    norm_req.renames = {"missing": "timestamp"}
    eng_req = EngineeringCalculationRequest(calculation_definition=def_altitude_rate)

    res = orch.run_csv_pipeline(csv, normalization_request=norm_req, engineering_request=eng_req)

    assert res.status == ResultStatus.ERROR
    assert res.engineering.status == ResultStatus.ERROR
    assert "ERR_ENGINEERING_DATA_UNAVAILABLE" in [i.issue_code for i in res.engineering.issues]


def test_provenance_linkage(
    tmp_path: pathlib.Path, def_altitude_rate: EngineeringCalculationDefinition
) -> None:
    csv = create_mock_csv(tmp_path)
    orch = DataPipelineOrchestrator()

    # Simple flow
    eng_req = EngineeringCalculationRequest(calculation_definition=def_altitude_rate)
    norm = NormalizationRequest()
    norm.standardize_timestamp("timestamp")
    res = orch.run_csv_pipeline(csv, engineering_request=eng_req, normalization_request=norm)

    assert res.status == ResultStatus.SUCCESS
    eng_prov = res.engineering.provenance
    ing_prov = res.ingestion.provenance
    assert eng_prov is not None
    assert ing_prov is not None
    assert ing_prov.record_id in eng_prov.parent_record_ids


def test_input_immutability(
    tmp_path: pathlib.Path, def_altitude_rate: EngineeringCalculationDefinition
) -> None:
    csv = create_mock_csv(tmp_path)
    orch = DataPipelineOrchestrator()
    eng_req = EngineeringCalculationRequest(calculation_definition=def_altitude_rate)
    norm = NormalizationRequest()
    norm.standardize_timestamp("timestamp")
    res = orch.run_csv_pipeline(csv, engineering_request=eng_req, normalization_request=norm)

    assert res.status == ResultStatus.SUCCESS

    # Check ingestion data is unchanged (index 1 is 1100)
    assert res.ingestion.data["altitude_ft"].iloc[1] == 1100.0

    # If we mutate output, it doesn't affect input
    res.engineering.output_value.iloc[1] = 9999.0
    assert res.ingestion.data["altitude_ft"].iloc[1] == 1100.0


def test_status_propagation(
    tmp_path: pathlib.Path, def_altitude_rate: EngineeringCalculationDefinition
) -> None:
    # Engineering Warning -> Overall Warning
    # We don't have an easy way to trigger a warning in altitude_rate, so we'll test ERROR.
    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T12:00:00Z", "2026-01-01T12:00:00Z"],
            "altitude_ft": [1000.0, 1100.0],
            "airspeed_kts": [100.0, 100.0],
            "pitch_deg": [0.0, 0.0],
            "roll_deg": [0.0, 0.0],
            "engine_rpm": [1000.0, 1000.0],
        }
    )
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    orch = DataPipelineOrchestrator()
    eng_req = EngineeringCalculationRequest(calculation_definition=def_altitude_rate)
    res = orch.run_csv_pipeline(str(path), engineering_request=eng_req)

    assert res.status == ResultStatus.ERROR
    assert res.engineering.status == ResultStatus.ERROR
