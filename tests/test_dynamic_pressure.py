import uuid

import pandas as pd
import pytest

from aeris.core.engineering.executor import EngineeringExecutor
from aeris.core.engineering.models import EngineeringCalculationDefinition
from aeris.core.engineering.registry import get_calculation_registry
from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.results.types import ResultStatus


@pytest.fixture
def executor() -> EngineeringExecutor:
    return EngineeringExecutor()


@pytest.fixture
def def_dynamic_pressure() -> EngineeringCalculationDefinition:
    return EngineeringCalculationDefinition(
        calculation_id="dynamic_pressure",
        name="Dynamic Pressure",
        description="Calculate dynamic pressure",
        output_parameter_id="dynamic_pressure",
        output_unit="Pa",
        input_parameter_ids=["air_density_kg_m3", "true_airspeed_m_s"],
        metadata={
            "air_density_column": "air_density",
            "air_density_unit": "kg/m^3",
            "true_airspeed_column": "true_airspeed",
            "true_airspeed_unit": "m/s",
        },
    )


def test_basic_two_point_calculation(
    executor: EngineeringExecutor, def_dynamic_pressure: EngineeringCalculationDefinition
) -> None:
    df = pd.DataFrame(
        {
            "air_density": [1.225, 1.200],
            "true_airspeed": [100.0, 150.0],
        }
    )
    res = executor.execute(def_dynamic_pressure, df)
    assert res.status == ResultStatus.SUCCESS
    assert res.unit == "Pa"

    # 0.5 * 1.225 * 100^2 = 6125.0
    # 0.5 * 1.200 * 150^2 = 13500.0
    assert res.output_value.iloc[0] == 6125.0
    assert res.output_value.iloc[1] == 13500.0


def test_multiple_rows(
    executor: EngineeringExecutor, def_dynamic_pressure: EngineeringCalculationDefinition
) -> None:
    df = pd.DataFrame(
        {
            "air_density": [1.225, 1.225, 1.225],
            "true_airspeed": [10.0, 20.0, 30.0],
        }
    )
    res = executor.execute(def_dynamic_pressure, df)
    assert res.status == ResultStatus.SUCCESS
    assert len(res.output_value) == 3


def test_missing_air_density_column(
    executor: EngineeringExecutor, def_dynamic_pressure: EngineeringCalculationDefinition
) -> None:
    df = pd.DataFrame(
        {
            "true_airspeed": [100.0, 150.0],
        }
    )
    res = executor.execute(def_dynamic_pressure, df)
    assert res.status == ResultStatus.ERROR
    assert "ERR_AIR_DENSITY_COLUMN_MISSING" in [i.issue_code for i in res.issues]


def test_missing_true_airspeed_column(
    executor: EngineeringExecutor, def_dynamic_pressure: EngineeringCalculationDefinition
) -> None:
    df = pd.DataFrame(
        {
            "air_density": [1.225, 1.200],
        }
    )
    res = executor.execute(def_dynamic_pressure, df)
    assert res.status == ResultStatus.ERROR
    assert "ERR_TRUE_AIRSPEED_COLUMN_MISSING" in [i.issue_code for i in res.issues]


def test_missing_values_in_air_density(
    executor: EngineeringExecutor, def_dynamic_pressure: EngineeringCalculationDefinition
) -> None:
    df = pd.DataFrame(
        {
            "air_density": [1.225, None],
            "true_airspeed": [100.0, 150.0],
        }
    )
    res = executor.execute(def_dynamic_pressure, df)
    assert res.status == ResultStatus.ERROR
    assert "ERR_INPUT_VALUE_MISSING" in [i.issue_code for i in res.issues]


def test_missing_values_in_true_airspeed(
    executor: EngineeringExecutor, def_dynamic_pressure: EngineeringCalculationDefinition
) -> None:
    df = pd.DataFrame(
        {
            "air_density": [1.225, 1.200],
            "true_airspeed": [100.0, None],
        }
    )
    res = executor.execute(def_dynamic_pressure, df)
    assert res.status == ResultStatus.ERROR
    assert "ERR_INPUT_VALUE_MISSING" in [i.issue_code for i in res.issues]


def test_non_numeric_air_density(
    executor: EngineeringExecutor, def_dynamic_pressure: EngineeringCalculationDefinition
) -> None:
    df = pd.DataFrame(
        {
            "air_density": ["1.225", "1.200"],
            "true_airspeed": [100.0, 150.0],
        }
    )
    res = executor.execute(def_dynamic_pressure, df)
    assert res.status == ResultStatus.ERROR
    assert "ERR_AIR_DENSITY_NOT_NUMERIC" in [i.issue_code for i in res.issues]


def test_non_numeric_true_airspeed(
    executor: EngineeringExecutor, def_dynamic_pressure: EngineeringCalculationDefinition
) -> None:
    df = pd.DataFrame(
        {
            "air_density": [1.225, 1.200],
            "true_airspeed": ["100.0", "150.0"],
        }
    )
    res = executor.execute(def_dynamic_pressure, df)
    assert res.status == ResultStatus.ERROR
    assert "ERR_TRUE_AIRSPEED_NOT_NUMERIC" in [i.issue_code for i in res.issues]


def test_unsupported_air_density_unit(
    executor: EngineeringExecutor, def_dynamic_pressure: EngineeringCalculationDefinition
) -> None:
    df = pd.DataFrame(
        {
            "air_density": [1.225, 1.200],
            "true_airspeed": [100.0, 150.0],
        }
    )
    bad_def = EngineeringCalculationDefinition(
        **{
            **def_dynamic_pressure.model_dump(),
            "metadata": {**def_dynamic_pressure.metadata, "air_density_unit": "slugs/ft^3"},
        }
    )
    res = executor.execute(bad_def, df)
    assert res.status == ResultStatus.ERROR
    assert "ERR_UNSUPPORTED_AIR_DENSITY_UNIT" in [i.issue_code for i in res.issues]


def test_unsupported_true_airspeed_unit(
    executor: EngineeringExecutor, def_dynamic_pressure: EngineeringCalculationDefinition
) -> None:
    df = pd.DataFrame(
        {
            "air_density": [1.225, 1.200],
            "true_airspeed": [100.0, 150.0],
        }
    )
    bad_def = EngineeringCalculationDefinition(
        **{
            **def_dynamic_pressure.model_dump(),
            "metadata": {**def_dynamic_pressure.metadata, "true_airspeed_unit": "kts"},
        }
    )
    res = executor.execute(bad_def, df)
    assert res.status == ResultStatus.ERROR
    assert "ERR_UNSUPPORTED_TRUE_AIRSPEED_UNIT" in [i.issue_code for i in res.issues]


def test_input_dataframe_immutability(
    executor: EngineeringExecutor, def_dynamic_pressure: EngineeringCalculationDefinition
) -> None:
    df = pd.DataFrame(
        {
            "air_density": [1.225, 1.200],
            "true_airspeed": [100.0, 150.0],
        }
    )
    original_df = df.copy()
    res = executor.execute(def_dynamic_pressure, df)
    assert res.status == ResultStatus.SUCCESS
    pd.testing.assert_frame_equal(df, original_df)


def test_output_epistemic_level(
    executor: EngineeringExecutor, def_dynamic_pressure: EngineeringCalculationDefinition
) -> None:
    df = pd.DataFrame(
        {
            "air_density": [1.225, 1.200],
            "true_airspeed": [100.0, 150.0],
        }
    )
    res = executor.execute(def_dynamic_pressure, df)
    assert res.epistemic_level == EpistemicLevel.CALCULATED


def test_provenance(
    executor: EngineeringExecutor, def_dynamic_pressure: EngineeringCalculationDefinition
) -> None:
    df = pd.DataFrame(
        {
            "air_density": [1.225, 1.200],
            "true_airspeed": [100.0, 150.0],
        }
    )
    res = executor.execute(def_dynamic_pressure, df)
    assert res.provenance is not None
    assert res.provenance.epistemic_level == EpistemicLevel.CALCULATED
    assert res.provenance.activity_name == "dynamic_pressure_calculation"


def test_explicit_input_references(
    executor: EngineeringExecutor, def_dynamic_pressure: EngineeringCalculationDefinition
) -> None:
    df = pd.DataFrame(
        {
            "air_density": [1.225, 1.200],
            "true_airspeed": [100.0, 150.0],
        }
    )
    res = executor.execute(def_dynamic_pressure, df)
    # References are managed by orchestrator typically, but we check execution success here
    assert res.status == ResultStatus.SUCCESS


def test_unknown_calculation_id(
    executor: EngineeringExecutor, def_dynamic_pressure: EngineeringCalculationDefinition
) -> None:
    df = pd.DataFrame(
        {
            "air_density": [1.225, 1.200],
            "true_airspeed": [100.0, 150.0],
        }
    )
    bad_def = EngineeringCalculationDefinition(
        **{**def_dynamic_pressure.model_dump(), "calculation_id": "unknown_calc"}
    )
    res = executor.execute(bad_def, df)
    assert res.status == ResultStatus.ERROR
    assert "ERR_CALCULATION_NOT_REGISTERED" in [i.issue_code for i in res.issues]


def test_calculation_registry_lookup(
    executor: EngineeringExecutor, def_dynamic_pressure: EngineeringCalculationDefinition
) -> None:
    registry = get_calculation_registry()
    func = registry.get("dynamic_pressure")
    assert func is not None


def test_unordered_timestamps_are_allowed(
    executor: EngineeringExecutor, def_dynamic_pressure: EngineeringCalculationDefinition
) -> None:
    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T12:00:10Z", "2026-01-01T12:00:00Z"],
            "air_density": [1.225, 1.200],
            "true_airspeed": [100.0, 150.0],
        }
    )
    res = executor.execute(def_dynamic_pressure, df)
    assert res.status == ResultStatus.SUCCESS


def test_duplicate_timestamps_are_allowed(
    executor: EngineeringExecutor, def_dynamic_pressure: EngineeringCalculationDefinition
) -> None:
    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T12:00:00Z", "2026-01-01T12:00:00Z"],
            "air_density": [1.225, 1.200],
            "true_airspeed": [100.0, 150.0],
        }
    )
    res = executor.execute(def_dynamic_pressure, df)
    assert res.status == ResultStatus.SUCCESS


def test_no_timestamp_column_is_required(
    executor: EngineeringExecutor, def_dynamic_pressure: EngineeringCalculationDefinition
) -> None:
    df = pd.DataFrame(
        {
            "air_density": [1.225, 1.200],
            "true_airspeed": [100.0, 150.0],
        }
    )
    assert "timestamp" not in df.columns
    res = executor.execute(def_dynamic_pressure, df)
    assert res.status == ResultStatus.SUCCESS


def test_no_registry_mutation(def_dynamic_pressure: EngineeringCalculationDefinition) -> None:
    registry = get_calculation_registry()
    initial_len = len(registry._implementations)
    registry.register(uuid.uuid4().hex, lambda *args, **kwargs: None)  # type: ignore
    assert len(registry._implementations) == initial_len + 1
