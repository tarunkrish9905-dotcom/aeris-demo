import pytest

from aeris.core.expected_observed.engine import ExpectedObservedEngine
from aeris.core.expected_observed.models import (
    ComparisonMethod,
    ExpectationSource,
    ExpectedValue,
    ObservedValue,
    Tolerance,
    ToleranceType,
)
from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.results.types import ResultStatus


def create_expected(value: float, unit: str | None = "kt", param: str = "p1") -> ExpectedValue:
    return ExpectedValue(
        expected_id="e1",
        parameter_id=param,
        value=value,
        unit=unit,
        source=ExpectationSource.EXPLICIT_INPUT,
        source_reference="req1",
        epistemic_level=EpistemicLevel.PROCEDURAL,
    )


def create_observed(value: float, unit: str | None = "kt", param: str = "p1") -> ObservedValue:
    return ObservedValue(
        observed_id="o1",
        parameter_id=param,
        value=value,
        unit=unit,
        source_reference="telem1",
    )


def test_absolute_difference():
    res = ExpectedObservedEngine.compare(
        create_expected(100.0), create_observed(105.0), ComparisonMethod.ABSOLUTE_DIFFERENCE
    )
    assert res.deviation is not None
    assert res.deviation.deviation_value == 5.0
    assert res.deviation.deviation_unit == "kt"
    assert res.status == ResultStatus.SUCCESS


def test_relative_difference():
    res = ExpectedObservedEngine.compare(
        create_expected(100.0), create_observed(105.0), ComparisonMethod.RELATIVE_DIFFERENCE
    )
    assert res.deviation is not None
    assert res.deviation.deviation_value == 0.05
    assert res.status == ResultStatus.SUCCESS


def test_percentage_difference():
    res = ExpectedObservedEngine.compare(
        create_expected(100.0), create_observed(105.0), ComparisonMethod.PERCENT_DIFFERENCE
    )
    assert res.deviation is not None
    assert res.deviation.deviation_value == 5.0
    assert res.deviation.deviation_unit == "%"


def test_ratio():
    res = ExpectedObservedEngine.compare(
        create_expected(100.0), create_observed(105.0), ComparisonMethod.RATIO
    )
    assert res.deviation is not None
    assert res.deviation.deviation_value == 1.05


def test_zero_expected_absolute_valid():
    res = ExpectedObservedEngine.compare(
        create_expected(0.0), create_observed(5.0), ComparisonMethod.ABSOLUTE_DIFFERENCE
    )
    assert res.deviation is not None
    assert res.deviation.deviation_value == 5.0


def test_zero_expected_relative_invalid():
    res = ExpectedObservedEngine.compare(
        create_expected(0.0), create_observed(5.0), ComparisonMethod.RELATIVE_DIFFERENCE
    )
    assert res.status == ResultStatus.ERROR
    assert res.deviation is None
    assert "ERR_ZERO_DENOMINATOR" in [i.issue_code for i in res.issues]


def test_zero_expected_percentage_invalid():
    res = ExpectedObservedEngine.compare(
        create_expected(0.0), create_observed(5.0), ComparisonMethod.PERCENT_DIFFERENCE
    )
    assert res.status == ResultStatus.ERROR
    assert res.deviation is None
    assert "ERR_ZERO_DENOMINATOR" in [i.issue_code for i in res.issues]


def test_zero_expected_ratio_invalid():
    res = ExpectedObservedEngine.compare(
        create_expected(0.0), create_observed(5.0), ComparisonMethod.RATIO
    )
    assert res.status == ResultStatus.ERROR
    assert res.deviation is None
    assert "ERR_ZERO_DENOMINATOR" in [i.issue_code for i in res.issues]


def test_negative_values():
    res = ExpectedObservedEngine.compare(
        create_expected(-100.0), create_observed(-90.0), ComparisonMethod.RELATIVE_DIFFERENCE
    )
    # (-90 - -100) / -100 = 10 / -100 = -0.1
    assert res.deviation.deviation_value == pytest.approx(-0.1)


def test_identity_mismatch():
    res = ExpectedObservedEngine.compare(
        create_expected(100.0, param="airspeed"),
        create_observed(105.0, param="altitude"),
        ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    assert res.status == ResultStatus.ERROR
    assert "ERR_PARAMETER_MISMATCH" in [i.issue_code for i in res.issues]


def test_incompatible_units():
    res = ExpectedObservedEngine.compare(
        create_expected(100.0, unit="kt"),
        create_observed(105.0, unit="m/s"),
        ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    assert res.status == ResultStatus.ERROR
    assert "ERR_INCOMPATIBLE_UNITS" in [i.issue_code for i in res.issues]


def test_unitless():
    res = ExpectedObservedEngine.compare(
        create_expected(100.0, unit=None),
        create_observed(105.0, unit=None),
        ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    assert res.status == ResultStatus.SUCCESS
    assert res.deviation.deviation_value == 5.0


def test_tolerance_absolute_within():
    tol = Tolerance(
        tolerance_type=ToleranceType.ABSOLUTE, lower_bound=-10.0, upper_bound=10.0, unit="kt"
    )
    res = ExpectedObservedEngine.compare(
        create_expected(100.0),
        create_observed(105.0),
        ComparisonMethod.ABSOLUTE_DIFFERENCE,
        tolerance=tol,
    )
    assert res.status == ResultStatus.SUCCESS
    assert res.tolerance_evaluated is True
    assert res.within_tolerance is True
    assert res.deviation.deviation_value == 5.0


def test_tolerance_absolute_outside():
    tol = Tolerance(
        tolerance_type=ToleranceType.ABSOLUTE, lower_bound=-10.0, upper_bound=10.0, unit="kt"
    )
    res = ExpectedObservedEngine.compare(
        create_expected(100.0),
        create_observed(115.0),
        ComparisonMethod.ABSOLUTE_DIFFERENCE,
        tolerance=tol,
    )
    assert res.status == ResultStatus.SUCCESS
    assert res.tolerance_evaluated is True
    assert res.within_tolerance is False
    assert res.deviation.deviation_value == 15.0


def test_tolerance_relative():
    tol = Tolerance(tolerance_type=ToleranceType.RELATIVE, lower_bound=-0.1, upper_bound=0.1)
    res = ExpectedObservedEngine.compare(
        create_expected(100.0),
        create_observed(105.0),
        ComparisonMethod.ABSOLUTE_DIFFERENCE,
        tolerance=tol,
    )
    assert res.within_tolerance is True


def test_tolerance_percentage():
    tol = Tolerance(tolerance_type=ToleranceType.PERCENTAGE, lower_bound=-10.0, upper_bound=10.0)
    res = ExpectedObservedEngine.compare(
        create_expected(100.0),
        create_observed(105.0),
        ComparisonMethod.ABSOLUTE_DIFFERENCE,
        tolerance=tol,
    )
    assert res.within_tolerance is True


def test_no_tolerance():
    res = ExpectedObservedEngine.compare(
        create_expected(100.0), create_observed(105.0), ComparisonMethod.ABSOLUTE_DIFFERENCE
    )
    assert res.tolerance_evaluated is False
    assert res.within_tolerance is None


def test_no_anomaly_generation():
    res = ExpectedObservedEngine.compare(
        create_expected(100.0), create_observed(105.0), ComparisonMethod.ABSOLUTE_DIFFERENCE
    )
    assert not hasattr(res, "is_anomaly")
    assert not hasattr(res, "anomaly_record")
    assert not hasattr(res, "root_cause")


def test_determinism():
    res1 = ExpectedObservedEngine.compare(
        create_expected(100.0), create_observed(105.0), ComparisonMethod.ABSOLUTE_DIFFERENCE
    )
    res2 = ExpectedObservedEngine.compare(
        create_expected(100.0), create_observed(105.0), ComparisonMethod.ABSOLUTE_DIFFERENCE
    )
    assert res1.deviation.deviation_value == res2.deviation.deviation_value
    assert res1.status == res2.status
