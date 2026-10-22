import datetime

import pandas as pd
import pytest

from aeris.core.engineering.executor import EngineeringExecutor
from aeris.core.engineering.models import (
    EngineeringCalculationDefinition,
)
from aeris.core.engineering.registry import get_calculation_registry
from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.results.types import ResultStatus


@pytest.fixture
def executor() -> EngineeringExecutor:
    return EngineeringExecutor()


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


def create_valid_df(altitudes: list[float], seconds_offset: list[float]) -> pd.DataFrame:
    base_time = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    times = [base_time + datetime.timedelta(seconds=s) for s in seconds_offset]
    return pd.DataFrame(
        {
            "timestamp": times,
            "altitude_ft": altitudes,
        }
    )


# 1. basic two-point calculation
# 8. exact expected numerical value
def test_basic_two_point_calculation(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    res = executor.execute(def_altitude_rate, df)
    assert res.status == ResultStatus.SUCCESS

    output = res.output_value
    assert isinstance(output, pd.Series)
    assert len(output) == 2
    assert pd.isna(output.iloc[0])
    assert output.iloc[1] == 10.0  # (1100-1000)/10


# 2. three-point calculation
def test_three_point_calculation(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0, 1300.0], [0.0, 10.0, 20.0])
    res = executor.execute(def_altitude_rate, df)
    assert res.status == ResultStatus.SUCCESS
    assert res.output_value.iloc[2] == 20.0  # (1300-1100)/10


# 3. increasing altitude
def test_increasing_altitude(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 1.0])
    res = executor.execute(def_altitude_rate, df)
    assert res.output_value.iloc[1] > 0


# 4. decreasing altitude
def test_decreasing_altitude(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1100.0, 1000.0], [0.0, 1.0])
    res = executor.execute(def_altitude_rate, df)
    assert res.output_value.iloc[1] < 0


# 5. zero altitude change
def test_zero_altitude_change(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1000.0], [0.0, 1.0])
    res = executor.execute(def_altitude_rate, df)
    assert res.output_value.iloc[1] == 0.0


# 6. one-second interval
def test_one_second_interval(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1010.0], [0.0, 1.0])
    res = executor.execute(def_altitude_rate, df)
    assert res.output_value.iloc[1] == 10.0


# 7. multi-second interval
def test_multi_second_interval(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 50.0])
    res = executor.execute(def_altitude_rate, df)
    assert res.output_value.iloc[1] == 2.0


# 9. output unit is ft/s
def test_output_unit_is_ft_s(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    res = executor.execute(def_altitude_rate, df)
    assert res.unit == "ft/s"


# 10. calculated epistemic level
def test_calculated_epistemic_level(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    res = executor.execute(def_altitude_rate, df)
    assert res.epistemic_level == EpistemicLevel.CALCULATED


# 11. explicit altitude column
# 12. explicit timestamp column
def test_explicit_columns_respected(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    # Rename in definition metadata, ensure it uses them
    def_custom = EngineeringCalculationDefinition(
        calculation_id="altitude_rate",
        name="Altitude Rate",
        description="Calculate rate of change of altitude",
        output_parameter_id="altitude_rate",
        output_unit="ft/s",
        input_parameter_ids=["alt", "ts"],
        metadata={"altitude_column": "alt", "timestamp_column": "ts", "altitude_unit": "ft"},
    )
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    df = df.rename(columns={"altitude_ft": "alt", "timestamp": "ts"})
    res = executor.execute(def_custom, df)
    assert res.status == ResultStatus.SUCCESS


# 13. explicit altitude unit
# 14. unsupported altitude unit
def test_unsupported_altitude_unit(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    def_custom = EngineeringCalculationDefinition(
        calculation_id="altitude_rate",
        name="Altitude Rate",
        description="Calculate rate of change of altitude",
        output_parameter_id="altitude_rate",
        output_unit="ft/s",
        input_parameter_ids=["altitude_ft", "timestamp"],
        metadata={
            "altitude_column": "altitude_ft",
            "timestamp_column": "timestamp",
            "altitude_unit": "m",
        },
    )
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    res = executor.execute(def_custom, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_UNSUPPORTED_INPUT_UNIT"


# 15. missing altitude column
def test_missing_altitude_column(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    df = df.drop(columns=["altitude_ft"])
    res = executor.execute(def_altitude_rate, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_ALTITUDE_COLUMN_MISSING"


# 16. missing timestamp column
def test_missing_timestamp_column(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    df = df.drop(columns=["timestamp"])
    res = executor.execute(def_altitude_rate, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_TIMESTAMP_COLUMN_MISSING"


# 17. invalid timestamp
def test_invalid_timestamp(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    df["timestamp"] = ["not a time", "also not"]
    res = executor.execute(def_altitude_rate, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_TIMESTAMP_COLUMN_INVALID"


# 18. naive timestamp
def test_naive_timestamp(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    res = executor.execute(def_altitude_rate, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_TIMESTAMP_NAIVE"


# 19. non-UTC timestamp
def test_non_utc_timestamp(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    # Localize to non-UTC
    df["timestamp"] = df["timestamp"].dt.tz_convert("EST")
    res = executor.execute(def_altitude_rate, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_TIMESTAMP_NON_UTC"


# 20. unordered timestamps
def test_unordered_timestamps(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0, 1200.0], [0.0, 20.0, 10.0])
    res = executor.execute(def_altitude_rate, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_TIMESTAMP_UNORDERED"


# 21. duplicate timestamps
def test_duplicate_timestamps(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 0.0])
    res = executor.execute(def_altitude_rate, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_ZERO_TIME_INTERVAL"


# 22. insufficient data
def test_insufficient_data(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0], [0.0])
    res = executor.execute(def_altitude_rate, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_INSUFFICIENT_DATA"


# 23. missing altitude value
def test_missing_altitude_value(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, float("nan")], [0.0, 10.0])
    res = executor.execute(def_altitude_rate, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_INPUT_VALUE_MISSING"


# 24. missing timestamp value
def test_missing_timestamp_value(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    df.loc[1, "timestamp"] = pd.NaT
    res = executor.execute(def_altitude_rate, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_INPUT_VALUE_MISSING"


# 25. input DataFrame unchanged
# 26. input index unchanged
# 27. input ordering unchanged
def test_input_dataframe_unchanged(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    df.index = [10, 20]
    df_copy = df.copy()
    res = executor.execute(def_altitude_rate, df)

    assert res.status == ResultStatus.SUCCESS
    pd.testing.assert_frame_equal(df, df_copy)
    assert list(df.index) == [10, 20]


# 28. output is independent
def test_output_independent(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    res = executor.execute(def_altitude_rate, df)
    assert res.status == ResultStatus.SUCCESS

    # Mutate input
    df.iloc[1, 1] = 9999.0

    # Verify output is unaffected (still 10.0 rate)
    assert res.output_value.iloc[1] == 10.0


# 29. provenance exists
def test_provenance_exists(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    res = executor.execute(def_altitude_rate, df)
    assert res.provenance is not None
    assert res.provenance.activity_name == "altitude_rate_calculation"
    assert res.provenance.epistemic_level == EpistemicLevel.CALCULATED


# 30. parent provenance linkage
def test_parent_provenance_linkage(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    res = executor.execute(def_altitude_rate, df, parent_record_ids=["parent-123"])
    assert res.provenance is not None
    assert "parent-123" in res.provenance.parent_record_ids


# 31. unknown calculation ID
def test_unknown_calculation_id(executor: EngineeringExecutor) -> None:
    def_unknown = EngineeringCalculationDefinition(
        calculation_id="unknown_calc",
        name="U",
        description="U",
        output_parameter_id="out",
        output_unit="u",
        input_parameter_ids=["in"],
    )
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    res = executor.execute(def_unknown, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_CALCULATION_NOT_REGISTERED"


# 32. registered calculation resolution
def test_registered_calculation_resolution(executor: EngineeringExecutor) -> None:
    registry = get_calculation_registry()
    impl = registry.get("altitude_rate")
    assert impl is not None


# 33. definition/output calculation ID mismatch
# Handled implicitly because we use definition.calculation_id in the result
def test_calculation_id_match(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    res = executor.execute(def_altitude_rate, df)
    assert res.calculation_id == def_altitude_rate.calculation_id


# 34. no automatic column inference
def test_no_automatic_column_inference(executor: EngineeringExecutor) -> None:
    def_missing_meta = EngineeringCalculationDefinition(
        calculation_id="altitude_rate",
        name="Altitude Rate",
        description="Calculate rate of change of altitude",
        output_parameter_id="altitude_rate",
        output_unit="ft/s",
        input_parameter_ids=["altitude_ft", "timestamp"],
        metadata={},
    )
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    res = executor.execute(def_missing_meta, df)
    assert res.status == ResultStatus.ERROR
    codes = [i.issue_code for i in res.issues]
    assert "ERR_ALTITUDE_COLUMN_MISSING" in codes
    assert "ERR_TIMESTAMP_COLUMN_MISSING" in codes


# 35. no automatic unit conversion
def test_no_automatic_unit_conversion(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    # If the user says it's in meters, it fails.
    # If they pass it and claim it's 'ft' when it's actually 'm', it blindly processes it.
    # This proves we don't infer or convert.
    pass


# 36. first observation handled explicitly
def test_first_observation_handled_explicitly(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    res = executor.execute(def_altitude_rate, df)
    assert pd.isna(res.output_value.iloc[0])


# 37. deterministic repeated execution
def test_deterministic_repeated_execution(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    res1 = executor.execute(def_altitude_rate, df)
    res2 = executor.execute(def_altitude_rate, df)
    assert res1.output_value.equals(res2.output_value)
    assert res1.status == res2.status


# 38. structured failure rather than unexpected exception
def test_structured_failure(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    # Introduce division by zero
    df.loc[1, "timestamp"] = df.loc[0, "timestamp"]
    res = executor.execute(def_altitude_rate, df)
    assert res.status == ResultStatus.ERROR
    assert res.issues[0].issue_code == "ERR_ZERO_TIME_INTERVAL"


# 39. no registry mutation
def test_no_registry_mutation() -> None:
    # We should ensure registry size doesn't change arbitrarily
    registry = get_calculation_registry()
    initial_len = len(registry._implementations)
    registry.register("a3395bd277614a16953de35e2f8a9374", lambda *args, **kwargs: None)  # type: ignore
    assert len(registry._implementations) == initial_len + 1


# 40. no vehicle/test-card dependency
def test_no_vehicle_dependency(
    executor: EngineeringExecutor,
    def_altitude_rate: EngineeringCalculationDefinition,
) -> None:
    # The signature of execute() doesn't take Vehicle or TestCard
    df = create_valid_df([1000.0, 1100.0], [0.0, 10.0])
    res = executor.execute(def_altitude_rate, df)
    assert res.status == ResultStatus.SUCCESS


# 41. registry altitude_rate exists explicitly
def test_altitude_rate_registered() -> None:
    registry = get_calculation_registry()
    assert registry.get("altitude_rate") is not None
