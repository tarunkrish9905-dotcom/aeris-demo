"""Tests for Parameter Semantic Validation."""

import pandas as pd
import pytest

from aeris.core.parameter_validation import ParameterValidationRequest, ParameterValidator
from aeris.core.registry import ParameterDataType, ParameterDefinition, ParameterRegistry
from aeris.core.results.types import ResultStatus


@pytest.fixture
def test_registry() -> ParameterRegistry:
    """Fixture providing a configured registry."""
    registry = ParameterRegistry()
    registry.register(
        ParameterDefinition(
            parameter_id="airspeed",
            display_name="Airspeed",
            description="True airspeed",
            physical_quantity="velocity",
            data_type=ParameterDataType.FLOAT,
            canonical_unit="knots",
            aliases=["speed"],
        )
    )
    registry.register(
        ParameterDefinition(
            parameter_id="altitude",
            display_name="Altitude",
            description="Barometric altitude",
            physical_quantity="length",
            data_type=ParameterDataType.INTEGER,
            canonical_unit="feet",
            aliases=["alt"],
        )
    )
    registry.register(
        ParameterDefinition(
            parameter_id="status_str",
            display_name="Status String",
            description="System status",
            physical_quantity="dimensionless",
            data_type=ParameterDataType.STRING,
            canonical_unit="none",
            aliases=[],
        )
    )
    registry.register(
        ParameterDefinition(
            parameter_id="is_active",
            display_name="Is Active",
            description="Active flag",
            physical_quantity="dimensionless",
            data_type=ParameterDataType.BOOLEAN,
            canonical_unit="none",
            aliases=[],
        )
    )
    registry.register(
        ParameterDefinition(
            parameter_id="timestamp",
            display_name="Timestamp",
            description="UTC time",
            physical_quantity="time",
            data_type=ParameterDataType.TIMESTAMP,
            canonical_unit="iso8601",
            aliases=["time"],
        )
    )
    return registry


def test_valid_float(test_registry: ParameterRegistry) -> None:
    df = pd.DataFrame({"airspeed": [100.5, 120.0]})
    req = ParameterValidationRequest(["airspeed"])
    validator = ParameterValidator(test_registry)
    res = validator.validate(df, req)
    assert res.status == ResultStatus.SUCCESS
    assert "airspeed" in res.validated_parameters


def test_valid_integer(test_registry: ParameterRegistry) -> None:
    df = pd.DataFrame({"altitude": [1000, 2000]})
    req = ParameterValidationRequest(["altitude"])
    validator = ParameterValidator(test_registry)
    res = validator.validate(df, req)
    assert res.status == ResultStatus.SUCCESS
    assert "altitude" in res.validated_parameters


def test_valid_string(test_registry: ParameterRegistry) -> None:
    df = pd.DataFrame({"status_str": ["OK", "WARN"]})
    req = ParameterValidationRequest(["status_str"])
    validator = ParameterValidator(test_registry)
    res = validator.validate(df, req)
    assert res.status == ResultStatus.SUCCESS
    assert "status_str" in res.validated_parameters


def test_valid_boolean(test_registry: ParameterRegistry) -> None:
    df = pd.DataFrame({"is_active": [True, False]})
    req = ParameterValidationRequest(["is_active"])
    validator = ParameterValidator(test_registry)
    res = validator.validate(df, req)
    assert res.status == ResultStatus.SUCCESS
    assert "is_active" in res.validated_parameters


def test_valid_timestamp(test_registry: ParameterRegistry) -> None:
    df = pd.DataFrame({"time": pd.to_datetime(["2026-10-06T10:00:00Z"])})
    req = ParameterValidationRequest(["time"])  # tests alias resolution
    validator = ParameterValidator(test_registry)
    res = validator.validate(df, req)
    assert res.status == ResultStatus.SUCCESS
    assert "time" in res.validated_parameters


def test_unknown_parameter(test_registry: ParameterRegistry) -> None:
    df = pd.DataFrame({"unknown": [1]})
    req = ParameterValidationRequest(["unknown"])
    validator = ParameterValidator(test_registry)
    res = validator.validate(df, req)
    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_PARAMETER_NOT_FOUND" for i in res.issues)
    assert "unknown" in res.failed_parameters


def test_missing_source_column(test_registry: ParameterRegistry) -> None:
    df = pd.DataFrame({"other": [1]})
    req = ParameterValidationRequest(["airspeed"])
    validator = ParameterValidator(test_registry)
    res = validator.validate(df, req)
    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_PARAMETER_COLUMN_MISSING" for i in res.issues)
    assert "airspeed" in res.failed_parameters


def test_negative_float_type_mismatch_and_no_casting(test_registry: ParameterRegistry) -> None:
    """IMPORTANT NEGATIVE TEST: String '100' should not be silently cast to float."""
    df = pd.DataFrame({"airspeed": ["100", "110", "120"]})
    df_copy = df.copy(deep=True)

    req = ParameterValidationRequest(["airspeed"])
    validator = ParameterValidator(test_registry)
    res = validator.validate(df, req)

    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_PARAMETER_TYPE_MISMATCH" for i in res.issues)
    # df should be unchanged (no casting)
    pd.testing.assert_frame_equal(df, df_copy)


def test_invalid_value_for_numeric(test_registry: ParameterRegistry) -> None:
    df = pd.DataFrame({"airspeed": ["100", "abc", "120"]})
    req = ParameterValidationRequest(["airspeed"])
    validator = ParameterValidator(test_registry)
    res = validator.validate(df, req)

    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_PARAMETER_TYPE_MISMATCH" for i in res.issues)
    assert any(i.issue_code == "ERR_PARAMETER_INVALID_VALUE" for i in res.issues)


def test_invalid_timestamp(test_registry: ParameterRegistry) -> None:
    df = pd.DataFrame({"time": ["2026-10-06", "not-a-date"]})
    req = ParameterValidationRequest(["time"])
    validator = ParameterValidator(test_registry)
    res = validator.validate(df, req)

    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_PARAMETER_TYPE_MISMATCH" for i in res.issues)
    assert any(i.issue_code == "ERR_PARAMETER_TIMESTAMP_INVALID" for i in res.issues)


def test_missing_values_reported(test_registry: ParameterRegistry) -> None:
    df = pd.DataFrame({"airspeed": [100.0, None, 120.0]})
    req = ParameterValidationRequest(["airspeed"])
    validator = ParameterValidator(test_registry)
    res = validator.validate(df, req)

    assert res.status == ResultStatus.WARNING  # Warning for missing values
    assert any(i.issue_code == "WARN_PARAMETER_MISSING_VALUES" for i in res.issues)


def test_empty_dataframe(test_registry: ParameterRegistry) -> None:
    df = pd.DataFrame()
    req = ParameterValidationRequest(["airspeed"])
    validator = ParameterValidator(test_registry)
    res = validator.validate(df, req)

    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_NO_DATA_TO_VALIDATE" for i in res.issues)


def test_multiple_parameter_validation(test_registry: ParameterRegistry) -> None:
    df = pd.DataFrame({"airspeed": [100.5, 120.0], "alt": [1000, 2000]})
    req = ParameterValidationRequest(["airspeed", "alt"])
    validator = ParameterValidator(test_registry)
    res = validator.validate(df, req)

    assert res.status == ResultStatus.SUCCESS
    assert len(res.validated_parameters) == 2


def test_negative_no_unit_conversion(test_registry: ParameterRegistry) -> None:
    """IMPORTANT NEGATIVE TEST: No unit conversion is performed."""
    df = pd.DataFrame({"airspeed": [100.0]})
    df_copy = df.copy(deep=True)
    req = ParameterValidationRequest(["airspeed"])
    validator = ParameterValidator(test_registry)
    res = validator.validate(df, req)

    assert res.status == ResultStatus.SUCCESS
    pd.testing.assert_frame_equal(df, df_copy)


def test_deterministic_repeated_validation(test_registry: ParameterRegistry) -> None:
    df = pd.DataFrame({"airspeed": [100.0]})
    req = ParameterValidationRequest(["airspeed"])
    validator = ParameterValidator(test_registry)

    res1 = validator.validate(df, req)
    res2 = validator.validate(df, req)

    assert res1.status == res2.status
    assert res1.validated_parameters == res2.validated_parameters


# --- AUDIT CASES ---


def test_case_1_float_strings(test_registry: ParameterRegistry) -> None:
    """CASE 1 — FLOAT: Verify that strings are NOT silently cast to floats, even if parseable."""
    df = pd.DataFrame({"airspeed": ["100", "110", "120"]})
    req = ParameterValidationRequest(["airspeed"])
    validator = ParameterValidator(test_registry)
    res = validator.validate(df, req)

    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_PARAMETER_TYPE_MISMATCH" for i in res.issues)


def test_case_2_float_with_invalid_string(test_registry: ParameterRegistry) -> None:
    """CASE 2 — FLOAT WITH ONE INVALID STRING: Ensure coercion doesn't mask primary mismatch."""
    df = pd.DataFrame({"airspeed": ["100", "110", "bad"]})
    req = ParameterValidationRequest(["airspeed"])
    validator = ParameterValidator(test_registry)
    res = validator.validate(df, req)

    assert res.status == ResultStatus.ERROR
    # Must report type mismatch since it's a string column
    assert any(i.issue_code == "ERR_PARAMETER_TYPE_MISMATCH" for i in res.issues)
    # Secondary diagnostic reports invalid value
    assert any(i.issue_code == "ERR_PARAMETER_INVALID_VALUE" for i in res.issues)


def test_case_3_integer_float_dtype(test_registry: ParameterRegistry) -> None:
    """CASE 3 — INTEGER: Float dtype is NOT accepted as INTEGER. It must be explicit."""
    df = pd.DataFrame({"altitude": [1.0, 2.0, 3.0]})
    req = ParameterValidationRequest(["altitude"])
    validator = ParameterValidator(test_registry)
    res = validator.validate(df, req)

    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_PARAMETER_TYPE_MISMATCH" for i in res.issues)


def test_case_4_timestamp_types(test_registry: ParameterRegistry) -> None:
    """CASE 4 — TIMESTAMP: Test datetime64, parseable strings, and unparseable strings."""
    # A: datetime64 (valid)
    df_a = pd.DataFrame({"time": pd.to_datetime(["2026-10-06T10:00:00Z"])})
    res_a = ParameterValidator(test_registry).validate(df_a, ParameterValidationRequest(["time"]))
    assert res_a.status == ResultStatus.SUCCESS

    # B: String timestamps that are parseable (TYPE MISMATCH, requires normalization)
    df_b = pd.DataFrame({"time": ["2026-10-06T10:00:00Z"]})
    res_b = ParameterValidator(test_registry).validate(df_b, ParameterValidationRequest(["time"]))
    assert res_b.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_PARAMETER_TYPE_MISMATCH" for i in res_b.issues)

    # C: String values containing invalid timestamp (TYPE MISMATCH + INVALID VALUE)
    df_c = pd.DataFrame({"time": ["2026-10-06", "bad-date"]})
    res_c = ParameterValidator(test_registry).validate(df_c, ParameterValidationRequest(["time"]))
    assert res_c.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_PARAMETER_TYPE_MISMATCH" for i in res_c.issues)
    assert any(i.issue_code == "ERR_PARAMETER_TIMESTAMP_INVALID" for i in res_c.issues)


def test_case_5_boolean_strings(test_registry: ParameterRegistry) -> None:
    """CASE 5 — BOOLEAN: Test boolean vs string boolean."""
    # Valid
    df_valid = pd.DataFrame({"is_active": [True, False, True]})
    res_valid = ParameterValidator(test_registry).validate(
        df_valid, ParameterValidationRequest(["is_active"])
    )
    assert res_valid.status == ResultStatus.SUCCESS

    # String representations must fail
    df_str = pd.DataFrame({"is_active": ["True", "False", "True"]})
    res_str = ParameterValidator(test_registry).validate(
        df_str, ParameterValidationRequest(["is_active"])
    )
    assert res_str.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_PARAMETER_TYPE_MISMATCH" for i in res_str.issues)


def test_case_6_string_numeric(test_registry: ParameterRegistry) -> None:
    """CASE 6 — STRING: Test string vs numeric."""
    # Valid
    df_valid = pd.DataFrame({"status_str": ["A", "B", "C"]})
    res_valid = ParameterValidator(test_registry).validate(
        df_valid, ParameterValidationRequest(["status_str"])
    )
    assert res_valid.status == ResultStatus.SUCCESS

    # Numeric values in a STRING parameter must fail
    df_num = pd.DataFrame({"status_str": [1, 2, 3]})
    res_num = ParameterValidator(test_registry).validate(
        df_num, ParameterValidationRequest(["status_str"])
    )
    assert res_num.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_PARAMETER_TYPE_MISMATCH" for i in res_num.issues)


def test_case_7_missing_values(test_registry: ParameterRegistry) -> None:
    """CASE 7 — MISSING VALUES: Missing data is not confused with invalid type."""
    df = pd.DataFrame({"airspeed": [100.0, None, 120.0]})
    req = ParameterValidationRequest(["airspeed"])
    validator = ParameterValidator(test_registry)
    res = validator.validate(df, req)

    # Should be WARNING because float dtype accepts NaNs, so no type mismatch
    assert res.status == ResultStatus.WARNING
    assert any(i.issue_code == "WARN_PARAMETER_MISSING_VALUES" for i in res.issues)


def test_case_8_no_mutation(test_registry: ParameterRegistry) -> None:
    """CASE 8 — NO MUTATION: Deep copy comparison."""
    df = pd.DataFrame({"airspeed": ["100", "bad", "120"]})
    df_copy = df.copy(deep=True)

    req = ParameterValidationRequest(["airspeed"])
    validator = ParameterValidator(test_registry)
    res = validator.validate(df, req)

    assert res.status == ResultStatus.ERROR
    pd.testing.assert_frame_equal(df, df_copy)
