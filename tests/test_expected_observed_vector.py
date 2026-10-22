import math

import pytest

from aeris.core.expected_observed.engine import VectorExpectedObservedEngine
from aeris.core.expected_observed.models import (
    ComparisonMethod,
    ExpectationSource,
    ExpectedValue,
    ObservedValue,
    Tolerance,
    ToleranceType,
    VectorComparisonRequest,
)
from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.results.types import ResultStatus


def create_expected(
    val: float, param: str = "airspeed", unit: str | None = "kt", id: str = "e1"
) -> ExpectedValue:
    return ExpectedValue(
        expected_id=id,
        parameter_id=param,
        value=val,
        unit=unit,
        source=ExpectationSource.EXPLICIT_INPUT,
        source_reference="test",
        epistemic_level=EpistemicLevel.PROCEDURAL,
    )


def create_observed(
    val: float, param: str = "airspeed", unit: str | None = "kt", id: str = "o1"
) -> ObservedValue:
    return ObservedValue(
        observed_id=id, parameter_id=param, value=val, unit=unit, source_reference="test_obs"
    )


def test_vector_valid_equal_length():
    e1 = create_expected(100.0)
    e2 = create_expected(200.0, id="e2")
    o1 = create_observed(105.0)
    o2 = create_observed(190.0, id="o2")

    req = VectorComparisonRequest(
        vector_id="v1",
        parameter_id="airspeed",
        expected_values=(e1, e2),
        observed_values=(o1, o2),
        method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    res = VectorExpectedObservedEngine.compare_vectors(req)
    assert res.status == ResultStatus.SUCCESS
    assert len(res.elements) == 2
    assert res.elements[0].expected.expected_id == "e1"
    assert res.elements[0].observed.observed_id == "o1"
    assert res.elements[1].expected.expected_id == "e2"
    assert res.elements[1].observed.observed_id == "o2"


def test_vector_length_mismatch():
    e1 = create_expected(100.0)
    e2 = create_expected(200.0, id="e2")
    o1 = create_observed(105.0)

    with pytest.raises(ValueError, match="same length"):
        VectorComparisonRequest(
            vector_id="v1",
            parameter_id="airspeed",
            expected_values=(e1, e2),
            observed_values=(o1,),
            method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
        )


def test_vector_absolute_difference():
    req = VectorComparisonRequest(
        vector_id="v1",
        parameter_id="airspeed",
        expected_values=(create_expected(100.0), create_expected(200.0), create_expected(300.0)),
        observed_values=(create_observed(105.0), create_observed(190.0), create_observed(320.0)),
        method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    res = VectorExpectedObservedEngine.compare_vectors(req)
    assert res.elements[0].deviation.deviation_value == 5.0
    assert res.elements[1].deviation.deviation_value == -10.0
    assert res.elements[2].deviation.deviation_value == 20.0


def test_vector_percentage_difference():
    req = VectorComparisonRequest(
        vector_id="v1",
        parameter_id="airspeed",
        expected_values=(create_expected(100.0), create_expected(200.0), create_expected(400.0)),
        observed_values=(create_observed(105.0), create_observed(190.0), create_observed(440.0)),
        method=ComparisonMethod.PERCENT_DIFFERENCE,
    )
    res = VectorExpectedObservedEngine.compare_vectors(req)
    assert res.elements[0].deviation.deviation_value == 5.0
    assert res.elements[1].deviation.deviation_value == -5.0
    assert res.elements[2].deviation.deviation_value == 10.0


def test_vector_zero_denominator_isolated():
    req = VectorComparisonRequest(
        vector_id="v1",
        parameter_id="airspeed",
        expected_values=(create_expected(100.0), create_expected(0.0), create_expected(200.0)),
        observed_values=(create_observed(105.0), create_observed(5.0), create_observed(210.0)),
        method=ComparisonMethod.PERCENT_DIFFERENCE,
    )
    res = VectorExpectedObservedEngine.compare_vectors(req)
    assert res.status == ResultStatus.PARTIAL
    assert res.elements[0].status == ResultStatus.SUCCESS
    assert res.elements[0].deviation.deviation_value == 5.0

    assert res.elements[1].status == ResultStatus.ERROR
    assert res.elements[1].deviation is None
    assert "ERR_ZERO_DENOMINATOR" in [i.issue_code for i in res.elements[1].issues]

    assert res.elements[2].status == ResultStatus.SUCCESS
    assert res.elements[2].deviation.deviation_value == 5.0


def test_vector_tolerance():
    req = VectorComparisonRequest(
        vector_id="v1",
        parameter_id="airspeed",
        expected_values=(create_expected(100.0), create_expected(200.0), create_expected(300.0)),
        observed_values=(create_observed(105.0), create_observed(190.0), create_observed(320.0)),
        method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
        tolerance=Tolerance(
            tolerance_type=ToleranceType.ABSOLUTE, lower_bound=-10.0, upper_bound=10.0
        ),
    )
    res = VectorExpectedObservedEngine.compare_vectors(req)
    assert res.elements[0].within_tolerance is True
    assert res.elements[1].within_tolerance is True
    assert res.elements[2].within_tolerance is False


def test_vector_parameter_identity_match():
    with pytest.raises(ValueError, match="must match the request parameter_id"):
        VectorComparisonRequest(
            vector_id="v1",
            parameter_id="airspeed",
            expected_values=(create_expected(100.0, param="altitude"),),
            observed_values=(create_observed(105.0, param="airspeed"),),
            method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
        )


def test_vector_incompatible_units():
    req = VectorComparisonRequest(
        vector_id="v1",
        parameter_id="airspeed",
        expected_values=(create_expected(100.0, unit="kt"),),
        observed_values=(create_observed(105.0, unit="m/s"),),
        method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    res = VectorExpectedObservedEngine.compare_vectors(req)
    assert res.status == ResultStatus.ERROR
    assert res.elements[0].status == ResultStatus.ERROR
    assert "ERR_INCOMPATIBLE_UNITS" in [i.issue_code for i in res.elements[0].issues]


def test_vector_ordering_preservation():
    e_vals = [
        create_expected(100.0, id="e0"),
        create_expected(200.0, id="e1"),
        create_expected(300.0, id="e2"),
    ]
    o_vals = [
        create_observed(105.0, id="o0"),
        create_observed(190.0, id="o1"),
        create_observed(320.0, id="o2"),
    ]
    req = VectorComparisonRequest(
        vector_id="v1",
        parameter_id="airspeed",
        expected_values=tuple(e_vals),
        observed_values=tuple(o_vals),
        method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    res = VectorExpectedObservedEngine.compare_vectors(req)
    assert [c.expected.expected_id for c in res.elements] == ["e0", "e1", "e2"]
    assert [c.observed.observed_id for c in res.elements] == ["o0", "o1", "o2"]


def test_vector_empty():
    req = VectorComparisonRequest(
        vector_id="v1",
        parameter_id="airspeed",
        expected_values=(),
        observed_values=(),
        method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    res = VectorExpectedObservedEngine.compare_vectors(req)
    assert res.status == ResultStatus.SUCCESS
    assert len(res.elements) == 0


def test_vector_finite_values_nan():
    with pytest.raises(ValueError, match="finite"):
        create_observed(float("nan"))


def test_vector_finite_values_inf():
    with pytest.raises(ValueError, match="finite"):
        create_expected(float("inf"))


def test_vector_relative_difference():
    req = VectorComparisonRequest(
        vector_id="v1",
        parameter_id="airspeed",
        expected_values=(create_expected(100.0),),
        observed_values=(create_observed(105.0),),
        method=ComparisonMethod.RELATIVE_DIFFERENCE,
    )
    res = VectorExpectedObservedEngine.compare_vectors(req)
    assert math.isclose(res.elements[0].deviation.deviation_value, 0.05)


def test_vector_ratio():
    req = VectorComparisonRequest(
        vector_id="v1",
        parameter_id="airspeed",
        expected_values=(create_expected(100.0),),
        observed_values=(create_observed(105.0),),
        method=ComparisonMethod.RATIO,
    )
    res = VectorExpectedObservedEngine.compare_vectors(req)
    assert math.isclose(res.elements[0].deviation.deviation_value, 1.05)


def test_vector_serialization_round_trip():
    req = VectorComparisonRequest(
        vector_id="v1",
        parameter_id="airspeed",
        expected_values=(create_expected(100.0),),
        observed_values=(create_observed(105.0),),
        method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    res = VectorExpectedObservedEngine.compare_vectors(req)
    data = res.model_dump(mode="json")
    res2 = type(res).model_validate(data)
    assert res.vector_id == res2.vector_id
    assert len(res.elements) == len(res2.elements)
    assert res.elements[0].deviation.deviation_value == res2.elements[0].deviation.deviation_value


def test_vector_epistemic_level():
    req = VectorComparisonRequest(
        vector_id="v1",
        parameter_id="airspeed",
        expected_values=(create_expected(100.0),),
        observed_values=(create_observed(105.0),),
        method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    res = VectorExpectedObservedEngine.compare_vectors(req)
    assert res.provenance.epistemic_level == EpistemicLevel.CALCULATED
    assert res.elements[0].provenance.epistemic_level == EpistemicLevel.CALCULATED
