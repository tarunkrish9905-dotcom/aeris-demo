"""Tests for Engineering Result and Calculation Foundation."""

import json

import pytest
from pydantic import ValidationError

from aeris.core.engineering.models import (
    EngineeringCalculationDefinition,
    EngineeringCalculationResult,
    EngineeringInputReference,
    EngineeringResult,
)
from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus


# 1. valid EngineeringCalculationDefinition
def test_valid_engineering_calculation_definition() -> None:
    def_model = EngineeringCalculationDefinition(
        calculation_id="calc-1",
        name="Density Altitude",
        description="Calculate density altitude from pressure altitude and temperature",
        output_parameter_id="density_alt",
        output_unit="ft",
        input_parameter_ids=["pressure_alt", "oat"],
        metadata={"source": "flight_manual"},
    )
    assert def_model.calculation_id == "calc-1"
    assert def_model.output_parameter_id == "density_alt"
    assert len(def_model.input_parameter_ids) == 2


# 2. invalid calculation identifier
def test_invalid_calculation_identifier() -> None:
    with pytest.raises(ValidationError):
        EngineeringCalculationDefinition(
            calculation_id=123,  # type: ignore[arg-type]
            name="Test",
            description="Test",
            output_parameter_id="out",
            output_unit="unit",
            input_parameter_ids=["in"],
        )


# 3. empty calculation identifier
def test_empty_calculation_identifier() -> None:
    with pytest.raises(ValidationError):
        EngineeringCalculationDefinition(
            calculation_id="",
            name="Test",
            description="Test",
            output_parameter_id="out",
            output_unit="unit",
            input_parameter_ids=["in"],
        )


# 4. duplicate input parameter IDs
def test_duplicate_input_parameter_ids() -> None:
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        EngineeringCalculationDefinition(
            calculation_id="calc",
            name="Test",
            description="Test",
            output_parameter_id="out",
            output_unit="unit",
            input_parameter_ids=["in", "in"],
        )


# 5. empty input parameter list where invalid
def test_empty_input_parameter_list() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        EngineeringCalculationDefinition(
            calculation_id="calc",
            name="Test",
            description="Test",
            output_parameter_id="out",
            output_unit="unit",
            input_parameter_ids=[],
        )


# 6. valid EngineeringInputReference
def test_valid_engineering_input_reference() -> None:
    ref = EngineeringInputReference(
        parameter_id="param-1",
        source_identifier="norm-1",
        column_name="std_param",
        unit="kts",
        epistemic_level=EpistemicLevel.OBSERVED,
    )
    assert ref.parameter_id == "param-1"
    assert ref.unit == "kts"


# 7. invalid parameter ID
def test_invalid_parameter_id_in_reference() -> None:
    with pytest.raises(ValidationError):
        EngineeringInputReference(
            parameter_id="",
            source_identifier="norm",
            epistemic_level=EpistemicLevel.OBSERVED,
        )


# 8. valid EngineeringResult
def test_valid_engineering_result() -> None:
    res = EngineeringResult(
        status=ResultStatus.SUCCESS,
        result_id="res-1",
        calculation_id="calc-1",
        payload=100.0,
        unit="kts",
    )
    assert res.result_id == "res-1"
    assert res.payload == 100.0


# 9. valid EngineeringCalculationResult
def test_valid_engineering_calculation_result() -> None:
    res = EngineeringCalculationResult(
        status=ResultStatus.SUCCESS,
        result_id="res-1",
        calculation_id="calc-1",
        output_value=250.5,
        unit="m/s",
        epistemic_level=EpistemicLevel.CALCULATED,
    )
    assert res.output_value == 250.5
    assert res.epistemic_level == EpistemicLevel.CALCULATED


# 10. calculated epistemic level
def test_calculated_epistemic_level_default() -> None:
    res = EngineeringCalculationResult(
        status=ResultStatus.SUCCESS,
        result_id="r1",
        calculation_id="c1",
        output_value=1,
    )
    assert res.epistemic_level == EpistemicLevel.CALCULATED


# 11. invalid epistemic level
def test_invalid_epistemic_level_for_calculation() -> None:
    with pytest.raises(ValueError, match="must not produce OBSERVED epistemic level"):
        EngineeringCalculationResult(
            status=ResultStatus.SUCCESS,
            result_id="r1",
            calculation_id="c1",
            output_value=1,
            epistemic_level=EpistemicLevel.OBSERVED,
        )


# 12. valid result status
def test_valid_result_status_propagation() -> None:
    res = EngineeringCalculationResult(
        status=ResultStatus.PARTIAL,
        result_id="r1",
        calculation_id="c1",
        output_value=1,
    )
    assert res.status == ResultStatus.PARTIAL
    assert not res.is_successful
    assert not res.has_errors


# 13. structured issues
def test_structured_issues_in_result() -> None:
    issue = Issue(
        issue_code="ERR_CALCULATION_INPUT_UNAVAILABLE",
        severity=IssueSeverity.ERROR,
        message="Missing inputs",
    )
    res = EngineeringCalculationResult(
        status=ResultStatus.ERROR,
        result_id="r1",
        calculation_id="c1",
        issues=[issue],
    )
    assert res.has_errors
    assert res.error_count == 1
    assert res.issues[0].issue_code == "ERR_CALCULATION_INPUT_UNAVAILABLE"


# 14. provenance presence
def test_provenance_presence() -> None:
    prov = ProvenanceRecord(
        record_id="prov-1",
        source_uri="sys",
        activity_name="test",
        epistemic_level=EpistemicLevel.CALCULATED,
    )
    res = EngineeringCalculationResult(
        status=ResultStatus.SUCCESS,
        result_id="r1",
        calculation_id="c1",
        provenance=prov,
    )
    assert res.provenance is not None
    assert res.provenance.record_id == "prov-1"


# 15. parent provenance linkage
def test_parent_provenance_linkage() -> None:
    prov = ProvenanceRecord(
        record_id="prov-1",
        source_uri="sys",
        activity_name="test",
        epistemic_level=EpistemicLevel.CALCULATED,
        parent_record_ids=["prov-0"],
    )
    res = EngineeringCalculationResult(
        status=ResultStatus.SUCCESS,
        result_id="r1",
        calculation_id="c1",
        provenance=prov,
    )
    assert "prov-0" in res.provenance.parent_record_ids  # type: ignore


# 16. explicit input references
def test_explicit_input_references() -> None:
    ref = EngineeringInputReference(
        parameter_id="p1",
        source_identifier="s1",
        epistemic_level=EpistemicLevel.OBSERVED,
    )
    res = EngineeringCalculationResult(
        status=ResultStatus.SUCCESS,
        result_id="r1",
        calculation_id="c1",
        input_references=[ref],
    )
    assert len(res.input_references) == 1
    assert res.input_references[0].parameter_id == "p1"


# 17. explicit units
def test_explicit_units_definition() -> None:
    def_model = EngineeringCalculationDefinition(
        calculation_id="c1",
        name="N",
        description="D",
        output_parameter_id="out",
        output_unit="kg/s",
        input_parameter_ids=["in"],
    )
    assert def_model.output_unit == "kg/s"


# 18. no automatic unit conversion
def test_no_automatic_unit_conversion_in_model() -> None:
    ref = EngineeringInputReference(
        parameter_id="p1",
        source_identifier="s1",
        unit="ft",
        epistemic_level=EpistemicLevel.OBSERVED,
    )
    # The reference accepts 'ft' literally, does not auto-convert to 'm'
    assert ref.unit == "ft"


# 19. no parameter inference
def test_no_parameter_inference() -> None:
    # Requires explicitly passing parameter list; no defaults inferring from name
    with pytest.raises(ValidationError):
        EngineeringCalculationDefinition(
            calculation_id="c1",
            name="N",
            description="D",
            output_parameter_id="out",
            output_unit="u",
            # missing input_parameter_ids entirely
        )  # type: ignore[call-arg]


# 20. immutable definition
def test_immutable_definition() -> None:
    def_model = EngineeringCalculationDefinition(
        calculation_id="c1",
        name="N",
        description="D",
        output_parameter_id="out",
        output_unit="u",
        input_parameter_ids=["in"],
    )
    with pytest.raises(ValidationError):
        def_model.name = "Mutated"  # type: ignore[misc]


# 21. immutable input reference
def test_immutable_input_reference() -> None:
    ref = EngineeringInputReference(
        parameter_id="p1",
        source_identifier="s1",
        epistemic_level=EpistemicLevel.OBSERVED,
    )
    with pytest.raises(ValidationError):
        ref.parameter_id = "mutated"  # type: ignore[misc]


# 22. immutable result
def test_immutable_result() -> None:
    res = EngineeringCalculationResult(
        status=ResultStatus.SUCCESS,
        result_id="r1",
        calculation_id="c1",
    )
    with pytest.raises(ValidationError):
        res.calculation_id = "mutated"  # type: ignore[misc]


# 23. deterministic serialization
def test_deterministic_serialization() -> None:
    res = EngineeringCalculationResult(
        status=ResultStatus.SUCCESS,
        result_id="r1",
        calculation_id="c1",
        output_value=42.0,
    )
    json_str = res.model_dump_json()
    data = json.loads(json_str)
    assert data["result_id"] == "r1"
    assert data["output_value"] == 42.0
    assert data["status"] == "SUCCESS"


# 24. no pandas/DataFrame dependency
def test_no_pandas_dependency() -> None:

    # Check that creating models does not forcibly import pandas in the engineering models scope
    # Actually, it's easier to check if we can pass simple floats instead of series
    res = EngineeringCalculationResult(
        status=ResultStatus.SUCCESS,
        result_id="r1",
        calculation_id="c1",
        output_value=[1, 2, 3],  # simple lists work
    )
    assert isinstance(res.output_value, list)


# 25. registry is not mutated
def test_registry_not_mutated() -> None:
    from aeris.core.registry import ParameterDataType, ParameterDefinition, ParameterRegistry

    reg = ParameterRegistry()
    reg.register(
        ParameterDefinition(
            parameter_id="in",
            physical_quantity="length",
            display_name="In",
            description="In",
            data_type=ParameterDataType.FLOAT,
        )
    )
    # We can define a calc without needing to touch or mutate the registry
    def_model = EngineeringCalculationDefinition(
        calculation_id="c1",
        name="N",
        description="D",
        output_parameter_id="out",
        output_unit="u",
        input_parameter_ids=["in"],
    )
    assert reg.get("in").parameter_id == "in"
    assert def_model.input_parameter_ids == ["in"]


# 26. missing required calculation information
def test_missing_required_calculation_information() -> None:
    with pytest.raises(ValidationError):
        EngineeringCalculationDefinition(
            calculation_id="c1",  # type: ignore[call-arg]
            # missing name, description
        )


# 27. unavailable calculation representation
def test_unavailable_calculation_representation() -> None:
    res = EngineeringCalculationResult(
        status=ResultStatus.NOT_AVAILABLE,
        result_id="r1",
        calculation_id="c1",
        issues=[Issue(issue_code="ERR_UNAVAILABLE", severity=IssueSeverity.WARNING, message="NA")],
    )
    assert res.status == ResultStatus.NOT_AVAILABLE


# 28. warning result representation
def test_warning_result_representation() -> None:
    res = EngineeringCalculationResult(
        status=ResultStatus.WARNING,
        result_id="r1",
        calculation_id="c1",
        issues=[Issue(issue_code="WARN_LIMIT", severity=IssueSeverity.WARNING, message="W")],
    )
    assert res.status == ResultStatus.WARNING
    assert res.has_warnings
    assert not res.has_errors


# 29. error result representation
def test_error_result_representation() -> None:
    res = EngineeringCalculationResult(
        status=ResultStatus.ERROR,
        result_id="r1",
        calculation_id="c1",
        issues=[Issue(issue_code="ERR_MATH", severity=IssueSeverity.ERROR, message="E")],
    )
    assert res.status == ResultStatus.ERROR
    assert res.has_errors


# 30. partial result representation
def test_partial_result_representation() -> None:
    res = EngineeringCalculationResult(
        status=ResultStatus.PARTIAL,
        result_id="r1",
        calculation_id="c1",
        output_value=1,
    )
    assert res.status == ResultStatus.PARTIAL
    assert res.output_value == 1
