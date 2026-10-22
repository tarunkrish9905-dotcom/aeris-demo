import datetime
import math

import pytest
from pydantic import ValidationError

from aeris.core.expected_observed.models import (
    ComparisonMethod,
    Deviation,
    ExpectationSource,
    ExpectedObservedComparison,
    ExpectedValue,
    ObservedValue,
    Tolerance,
    ToleranceType,
)
from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.results.types import ResultStatus


def create_provenance() -> ProvenanceRecord:
    return ProvenanceRecord(
        record_id="prov1",
        activity_name="test_activity",
        epistemic_level=EpistemicLevel.CALCULATED,
        created_at=datetime.datetime.now(datetime.UTC),
        source_uri="f1",
    )


def test_expected_value_valid_construction() -> None:
    ev = ExpectedValue(
        expected_id="ev1",
        parameter_id="param1",
        value=100.0,
        unit="kt",
        source=ExpectationSource.EXPLICIT_INPUT,
        source_reference="req_1",
        epistemic_level=EpistemicLevel.PROCEDURAL,
        metadata={"key": "val"},
    )
    assert ev.expected_id == "ev1"
    assert ev.value == 100.0
    assert ev.epistemic_level == EpistemicLevel.PROCEDURAL


def test_expected_value_immutability() -> None:
    ev = ExpectedValue(
        expected_id="ev1",
        parameter_id="param1",
        value=100.0,
        source=ExpectationSource.EXPLICIT_INPUT,
        source_reference="req_1",
        epistemic_level=EpistemicLevel.PROCEDURAL,
    )
    with pytest.raises(ValidationError):
        ev.value = 101.0


def test_expected_value_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ExpectedValue(
            expected_id="ev1",
            parameter_id="param1",
            value=100.0,
            source=ExpectationSource.EXPLICIT_INPUT,
            source_reference="req_1",
            epistemic_level=EpistemicLevel.PROCEDURAL,
            unknown_field=42,
        )


def test_expected_value_invalid_identifier() -> None:
    with pytest.raises(ValidationError):
        ExpectedValue(
            expected_id="",
            parameter_id="param1",
            value=100.0,
            source=ExpectationSource.EXPLICIT_INPUT,
            source_reference="req_1",
            epistemic_level=EpistemicLevel.PROCEDURAL,
        )


def test_expected_value_nan_infinity() -> None:
    for invalid_val in [math.nan, math.inf, -math.inf]:
        with pytest.raises(ValidationError):
            ExpectedValue(
                expected_id="ev1",
                parameter_id="param1",
                value=invalid_val,
                source=ExpectationSource.EXPLICIT_INPUT,
                source_reference="req_1",
                epistemic_level=EpistemicLevel.PROCEDURAL,
            )


def test_observed_value_valid_construction() -> None:
    ov = ObservedValue(
        observed_id="ov1",
        parameter_id="param1",
        value=105.0,
        unit="kt",
        source_reference="telem_1",
    )
    assert ov.observed_id == "ov1"
    assert ov.value == 105.0
    assert ov.epistemic_level == EpistemicLevel.OBSERVED


def test_observed_value_nan_infinity() -> None:
    for invalid_val in [math.nan, math.inf, -math.inf]:
        with pytest.raises(ValidationError):
            ObservedValue(
                observed_id="ov1",
                parameter_id="param1",
                value=invalid_val,
                source_reference="telem_1",
            )


def test_deviation_valid_construction() -> None:
    dev = Deviation(
        expected_value=100.0,
        observed_value=105.0,
        deviation_value=5.0,
        deviation_unit="kt",
        comparison_method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    assert dev.deviation_value == 5.0
    assert dev.comparison_method == ComparisonMethod.ABSOLUTE_DIFFERENCE


def test_deviation_not_an_anomaly() -> None:
    dev = Deviation(
        expected_value=100.0,
        observed_value=105.0,
        deviation_value=5.0,
        comparison_method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    # Ensure no anomaly/root cause inference properties exist
    assert not hasattr(dev, "anomaly")
    assert not hasattr(dev, "severity")
    assert not hasattr(dev, "root_cause")
    assert not hasattr(dev, "recommendation")


def test_tolerance_immutability() -> None:
    tol = Tolerance(
        tolerance_type=ToleranceType.ABSOLUTE, lower_bound=-10.0, upper_bound=10.0, unit="kt"
    )
    with pytest.raises(ValidationError):
        tol.upper_bound = 20.0


def test_expected_observed_comparison_valid_construction() -> None:
    ev = ExpectedValue(
        expected_id="ev1",
        parameter_id="param1",
        value=100.0,
        source=ExpectationSource.EXPLICIT_INPUT,
        source_reference="r1",
        epistemic_level=EpistemicLevel.PROCEDURAL,
    )
    ov = ObservedValue(observed_id="ov1", parameter_id="param1", value=105.0, source_reference="r2")
    dev = Deviation(
        expected_value=100.0,
        observed_value=105.0,
        deviation_value=5.0,
        comparison_method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )

    comp = ExpectedObservedComparison(
        comparison_id="comp1",
        expected=ev,
        observed=ov,
        deviation=dev,
        status=ResultStatus.SUCCESS,
        provenance=create_provenance(),
    )

    assert comp.comparison_id == "comp1"
    assert comp.status == ResultStatus.SUCCESS
    assert comp.expected.value == 100.0
    assert comp.observed.value == 105.0
    assert comp.deviation.deviation_value == 5.0


def test_parameter_identity_consistency() -> None:
    ev = ExpectedValue(
        expected_id="ev1",
        parameter_id="param1",
        value=100.0,
        source=ExpectationSource.EXPLICIT_INPUT,
        source_reference="r1",
        epistemic_level=EpistemicLevel.PROCEDURAL,
    )
    ov = ObservedValue(observed_id="ov1", parameter_id="param2", value=105.0, source_reference="r2")

    with pytest.raises(ValidationError) as exc:
        ExpectedObservedComparison(
            comparison_id="comp1",
            expected=ev,
            observed=ov,
            status=ResultStatus.SUCCESS,
            provenance=create_provenance(),
        )
    assert "Expected and observed parameter_ids must match" in str(exc.value)


def test_deterministic_serialization() -> None:
    ev = ExpectedValue(
        expected_id="ev1",
        parameter_id="param1",
        value=100.0,
        source=ExpectationSource.EXPLICIT_INPUT,
        source_reference="r1",
        epistemic_level=EpistemicLevel.PROCEDURAL,
    )
    ov = ObservedValue(observed_id="ov1", parameter_id="param1", value=105.0, source_reference="r2")
    comp = ExpectedObservedComparison(
        comparison_id="comp1",
        expected=ev,
        observed=ov,
        status=ResultStatus.SUCCESS,
        provenance=create_provenance(),
    )

    json_str = comp.model_dump_json()
    assert "comp1" in json_str

    restored = ExpectedObservedComparison.model_validate_json(json_str)
    assert restored.comparison_id == "comp1"
    assert restored.expected.value == 100.0
    assert restored.observed.value == 105.0


def test_negative_architectural_constraints() -> None:
    ev = ExpectedValue(
        expected_id="ev1",
        parameter_id="param1",
        value=100.0,
        source=ExpectationSource.EXPLICIT_INPUT,
        source_reference="r1",
        epistemic_level=EpistemicLevel.PROCEDURAL,
    )

    assert not hasattr(ev, "anomaly_record")
    assert not hasattr(ev, "hypothesis")
    assert not hasattr(ev, "root_cause")
    assert not hasattr(ev, "recommendation")
    assert not hasattr(ev, "dataframe")
    assert not hasattr(ev, "pandas")
    assert not hasattr(ev, "registry")


def test_conceptual_deviation_is_not_anomaly() -> None:
    # Expected = 100 kt, Observed = 105 kt, Tolerance = ±10 kt
    ev = ExpectedValue(
        expected_id="ev1",
        parameter_id="param1",
        value=100.0,
        source=ExpectationSource.EXPLICIT_INPUT,
        source_reference="r1",
        epistemic_level=EpistemicLevel.PROCEDURAL,
        unit="kt",
    )
    ov = ObservedValue(
        observed_id="ov1", parameter_id="param1", value=105.0, source_reference="r2", unit="kt"
    )
    dev = Deviation(
        expected_value=100.0,
        observed_value=105.0,
        deviation_value=5.0,
        deviation_unit="kt",
        comparison_method=ComparisonMethod.ABSOLUTE_DIFFERENCE,
    )
    tol = Tolerance(
        tolerance_type=ToleranceType.ABSOLUTE, lower_bound=-10.0, upper_bound=10.0, unit="kt"
    )

    comp = ExpectedObservedComparison(
        comparison_id="comp1",
        expected=ev,
        observed=ov,
        deviation=dev,
        tolerance=tol,
        status=ResultStatus.SUCCESS,
        provenance=create_provenance(),
    )

    # +5kt deviation is not automatically a limit violation or anomaly
    assert comp.status == ResultStatus.SUCCESS
    assert not hasattr(comp, "is_anomaly")
    assert not hasattr(comp, "violation")
    assert not hasattr(comp, "failure")
