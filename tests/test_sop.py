"""
Tests for AERIS SOP and Test Procedure Requirements (Milestone 13, Step 13d).

Verifies:
1. Valid numeric and non-numeric requirement definitions.
2. Missing or malformed identifiers and document revisions.
3. Missing source references and invalid applicability.
4. Duplicate and conflicting requirements.
5. Explicit and ambiguous applicability.
6. Vehicle/profile/configuration mismatches.
7. Unknown test-card, test-point, or parameter references when validation is supported.
8. Numeric requirements with missing or incompatible units.
9. Malformed YAML, missing fields, and forbidden extra fields.
10. Atomic failure without partial collection exposure.
11. Deterministic repeated loading and lookup.
12. Separation between requirement definition and verification status.
13. Regression tests for existing vehicle, limit, registry, and test-operation behavior.
"""

from __future__ import annotations

import pathlib

import pytest

from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.registry import ParameterDataType, ParameterDefinition, ParameterRegistry
from aeris.core.sop import (
    ConflictingRequirementError,
    DuplicateRequirementError,
    IncompatibleProcedureUnitError,
    InvalidProcedureProfileError,
    InvalidProcedureRequirementError,
    ProcedureProfileMismatchError,
    ProcedureRequirement,
    ProcedureRequirementProfile,
    ProcedureRequirementSet,
    ProcedureRequirementVerification,
    RequirementCategory,
    RequirementVerificationStatus,
    UnknownParameterProcedureError,
    UnknownTestOperationReferenceError,
    load_procedure_requirement_profile_from_yaml,
    load_procedure_requirements_from_yaml,
    load_sop_requirements_from_yaml,
)
from aeris.core.test_operations.test_card import TestCard
from aeris.core.test_operations.test_point import TestPoint
from aeris.core.vehicle.envelope_models import ConfigurationEnvelope
from aeris.core.vehicle.profile_models import VehicleProfile

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures" / "sop"


@pytest.fixture
def sample_registry() -> ParameterRegistry:
    """Create a minimal parameter registry for procedure cross-validation."""
    registry = ParameterRegistry()
    registry.register(
        ParameterDefinition(
            parameter_id="altitude",
            display_name="Pressure Altitude",
            description="Pressure altitude above sea level",
            physical_quantity="distance",
            data_type=ParameterDataType.FLOAT,
            canonical_unit="ft",
        )
    )
    registry.register(
        ParameterDefinition(
            parameter_id="airspeed",
            display_name="Calibrated Airspeed",
            description="Calibrated airspeed",
            physical_quantity="velocity",
            data_type=ParameterDataType.FLOAT,
            canonical_unit="kts",
        )
    )
    registry.register(
        ParameterDefinition(
            parameter_id="load_factor_nz",
            display_name="Normal Load Factor",
            description="Normal acceleration nz",
            physical_quantity="acceleration",
            data_type=ParameterDataType.FLOAT,
            canonical_unit="g",
        )
    )
    registry.register(
        ParameterDefinition(
            parameter_id="unitless_ratio",
            display_name="Dimensionless Ratio",
            description="Dimensionless test factor",
            physical_quantity="ratio",
            data_type=ParameterDataType.FLOAT,
            canonical_unit=None,
        )
    )
    return registry


@pytest.fixture
def sample_vehicle_profile() -> VehicleProfile:
    """Create a synthetic non-operational VehicleProfile for cross-validation."""
    return VehicleProfile(
        profile_id="SYNTH-PROFILE-01",
        vehicle_id="SYNTH-AIRCRAFT-01",
        profile_version="1.0.0",
        name="Synthetic Aircraft Profile",
        supported_parameter_ids=("altitude", "airspeed", "load_factor_nz", "unitless_ratio"),
    )


@pytest.fixture
def sample_configuration_envelope() -> ConfigurationEnvelope:
    """Create a synthetic non-operational ConfigurationEnvelope for cross-validation."""
    return ConfigurationEnvelope(
        configuration_id="CLEAN-CRUISE",
        vehicle_id="SYNTH-AIRCRAFT-01",
        profile_id="SYNTH-PROFILE-01",
        profile_version="1.0.0",
        name="Clean Cruise Configuration",
    )


@pytest.fixture
def sample_test_operations() -> tuple[list[TestCard], list[TestPoint]]:
    """Create synthetic non-operational test card and test point instances."""
    cards = [
        TestCard(
            test_card_id="CARD-PERF-01",
            context_id="CTX-01",
            name="Performance Test Card",
            test_point_ids=("TP-STAB-01", "TP-CLIMB-02"),
        )
    ]
    points = [
        TestPoint(
            test_point_id="TP-STAB-01",
            context_id="CTX-01",
            name="Steady Level Flight Point",
        ),
        TestPoint(
            test_point_id="TP-CLIMB-02",
            context_id="CTX-01",
            name="Climb Performance Point",
        ),
    ]
    return cards, points


# -----------------------------------------------------------------------------
# 1. Valid numeric and non-numeric requirement definitions
# -----------------------------------------------------------------------------


def test_valid_numeric_target_and_symmetric_tolerance() -> None:
    req = ProcedureRequirement(
        requirement_id="REQ-ALT-01",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Section 3.1",
        description="Maintain entry altitude at 10,000 ft +/- 100 ft",
        category=RequirementCategory.ENTRY_CONDITION,
        parameter_id="altitude",
        target_value=10000.0,
        tolerance=100.0,
        unit="ft",
    )
    assert req.is_numeric is True
    assert req.effective_lower_bound == 9900.0
    assert req.effective_upper_bound == 10100.0
    assert req.is_satisfied_by_value(10050.0) is True
    assert req.is_satisfied_by_value(9850.0) is False
    assert req.calculate_margin(10050.0) == 50.0
    assert req.calculate_margin(10150.0) == -50.0


def test_valid_numeric_asymmetric_tolerance() -> None:
    req = ProcedureRequirement(
        requirement_id="REQ-SPD-ASYM",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Section 3.2",
        description="Airspeed target 150 kts (-5 / +10 kts)",
        category=RequirementCategory.TOLERANCE_BAND,
        parameter_id="airspeed",
        target_value=150.0,
        tolerance_lower=5.0,
        tolerance_upper=10.0,
        unit="kts",
    )
    assert req.effective_lower_bound == 145.0
    assert req.effective_upper_bound == 160.0
    assert req.is_satisfied_by_value(146.0) is True
    assert req.is_satisfied_by_value(144.0) is False
    assert req.is_satisfied_by_value(160.0) is True
    assert req.is_satisfied_by_value(161.0) is False


def test_valid_numeric_band_and_single_sided() -> None:
    # Band requirement without explicit nominal target
    band_req = ProcedureRequirement(
        requirement_id="REQ-SPD-BAND",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Section 3.3",
        description="Operate within 140 to 180 kts",
        category=RequirementCategory.EXECUTION,
        parameter_id="airspeed",
        min_value=140.0,
        max_value=180.0,
        unit="kts",
    )
    assert band_req.effective_lower_bound == 140.0
    assert band_req.effective_upper_bound == 180.0
    assert band_req.is_satisfied_by_value(150.0) is True
    assert band_req.is_satisfied_by_value(139.0) is False

    # Single-sided lower bound
    min_req = ProcedureRequirement(
        requirement_id="REQ-MIN-ALT",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Section 3.4",
        description="Minimum abort altitude 5000 ft",
        category=RequirementCategory.ABORT_CRITERIA,
        parameter_id="altitude",
        min_value=5000.0,
        unit="ft",
    )
    assert min_req.effective_lower_bound == 5000.0
    assert min_req.effective_upper_bound is None
    assert min_req.is_satisfied_by_value(5001.0) is True
    assert min_req.is_satisfied_by_value(4999.0) is False


def test_valid_non_numeric_procedural_requirement() -> None:
    req = ProcedureRequirement(
        requirement_id="REQ-CHECK-GEAR",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Section 4.1",
        description="Verify landing gear is down and locked",
        category=RequirementCategory.OPERATIONAL_CHECK,
    )
    assert req.is_numeric is False
    assert req.unit is None
    assert req.effective_lower_bound is None
    assert req.effective_upper_bound is None
    assert req.calculate_margin(10.0) is None
    with pytest.raises(ValueError, match="is non-numeric"):
        req.is_satisfied_by_value(10.0)


# -----------------------------------------------------------------------------
# 2. Missing or malformed identifiers and document revisions
# -----------------------------------------------------------------------------


def test_malformed_identifiers() -> None:
    with pytest.raises(ValueError, match="requirement_id cannot be empty"):
        ProcedureRequirement(
            requirement_id="   ",
            document_id="SOP-001",
            document_version="1.0.0",
            source_reference="Sec 1",
            description="Test req",
            category=RequirementCategory.OPERATIONAL_CHECK,
        )

    with pytest.raises(ValueError, match="must contain only alphanumeric"):
        ProcedureRequirement(
            requirement_id="REQ@INVALID!",
            document_id="SOP-001",
            document_version="1.0.0",
            source_reference="Sec 1",
            description="Test req",
            category=RequirementCategory.OPERATIONAL_CHECK,
        )

    with pytest.raises(ValueError, match="document_id cannot be empty"):
        ProcedureRequirement(
            requirement_id="REQ-01",
            document_id="   ",
            document_version="1.0.0",
            source_reference="Sec 1",
            description="Test req",
            category=RequirementCategory.OPERATIONAL_CHECK,
        )


def test_malformed_document_version() -> None:
    with pytest.raises(ValueError, match="document_version cannot be empty"):
        ProcedureRequirement(
            requirement_id="REQ-01",
            document_id="SOP-001",
            document_version="  ",
            source_reference="Sec 1",
            description="Test req",
            category=RequirementCategory.OPERATIONAL_CHECK,
        )

    with pytest.raises(ValueError, match="must contain only alphanumeric characters, dots"):
        ProcedureRequirement(
            requirement_id="REQ-01",
            document_id="SOP-001",
            document_version="1.0.0 (draft)",
            source_reference="Sec 1",
            description="Test req",
            category=RequirementCategory.OPERATIONAL_CHECK,
        )


# -----------------------------------------------------------------------------
# 3. Missing source references and invalid applicability
# -----------------------------------------------------------------------------


def test_missing_source_reference_and_description() -> None:
    with pytest.raises(ValueError, match="source_reference cannot be empty"):
        ProcedureRequirement(
            requirement_id="REQ-01",
            document_id="SOP-001",
            document_version="1.0.0",
            source_reference="  ",
            description="Valid description",
            category=RequirementCategory.OPERATIONAL_CHECK,
        )

    with pytest.raises(ValueError, match="description cannot be empty"):
        ProcedureRequirement(
            requirement_id="REQ-01",
            document_id="SOP-001",
            document_version="1.0.0",
            source_reference="Sec 1",
            description="   ",
            category=RequirementCategory.OPERATIONAL_CHECK,
        )


def test_forbidden_metadata_keys() -> None:
    with pytest.raises(ValueError, match="metadata contains forbidden operational key"):
        ProcedureRequirement(
            requirement_id="REQ-01",
            document_id="SOP-001",
            document_version="1.0.0",
            source_reference="Sec 1",
            description="Check item",
            category=RequirementCategory.OPERATIONAL_CHECK,
            metadata={"observed_data": [1, 2, 3]},
        )


# -----------------------------------------------------------------------------
# 4. Duplicate and conflicting requirements
# -----------------------------------------------------------------------------


def test_duplicate_requirement_id_in_profile() -> None:
    req1 = ProcedureRequirement(
        requirement_id="REQ-01",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Sec 1",
        description="First item",
        category=RequirementCategory.OPERATIONAL_CHECK,
    )
    req2 = ProcedureRequirement(
        requirement_id="REQ-01",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Sec 2",
        description="Second item with same ID",
        category=RequirementCategory.OPERATIONAL_CHECK,
    )
    with pytest.raises(DuplicateRequirementError, match="Duplicate requirement_id"):
        ProcedureRequirementProfile(
            profile_id="PROF-01",
            document_id="SOP-001",
            document_version="1.0.0",
            requirements=(req1, req2),
        )


def test_contradictory_bounds_in_single_requirement() -> None:
    # min_value > max_value
    with pytest.raises(ConflictingRequirementError, match="cannot exceed max_value"):
        ProcedureRequirement(
            requirement_id="REQ-01",
            document_id="SOP-001",
            document_version="1.0.0",
            source_reference="Sec 1",
            description="Bad bounds",
            category=RequirementCategory.EXECUTION,
            parameter_id="altitude",
            min_value=12000.0,
            max_value=10000.0,
            unit="ft",
        )

    # target_value < min_value
    with pytest.raises(ConflictingRequirementError, match="less than min_value"):
        ProcedureRequirement(
            requirement_id="REQ-02",
            document_id="SOP-001",
            document_version="1.0.0",
            source_reference="Sec 1",
            description="Bad target",
            category=RequirementCategory.EXECUTION,
            parameter_id="altitude",
            target_value=9000.0,
            min_value=10000.0,
            unit="ft",
        )


def test_conflicting_numeric_requirements_in_profile() -> None:
    req1 = ProcedureRequirement(
        requirement_id="REQ-ALT-A",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Sec 1",
        description="Altitude requirement 1",
        category=RequirementCategory.ENTRY_CONDITION,
        test_point_id="TP-01",
        parameter_id="altitude",
        target_value=10000.0,
        tolerance=50.0,
        unit="ft",
    )
    req2 = ProcedureRequirement(
        requirement_id="REQ-ALT-B",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Sec 2",
        description="Altitude requirement 2 for exact same scope",
        category=RequirementCategory.ENTRY_CONDITION,
        test_point_id="TP-01",
        parameter_id="altitude",
        target_value=12000.0,
        tolerance=50.0,
        unit="ft",
    )
    with pytest.raises(
        ConflictingRequirementError, match="Conflicting or duplicate numeric requirement"
    ):
        ProcedureRequirementProfile(
            profile_id="PROF-01",
            document_id="SOP-001",
            document_version="1.0.0",
            requirements=(req1, req2),
        )


# -----------------------------------------------------------------------------
# 5. Explicit and ambiguous applicability
# -----------------------------------------------------------------------------


def test_explicit_applicability_filtering() -> None:
    req1 = ProcedureRequirement(
        requirement_id="REQ-TP1",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Sec 1",
        description="Specific to TP-01",
        category=RequirementCategory.ENTRY_CONDITION,
        test_card_id="CARD-01",
        test_point_id="TP-01",
        parameter_id="altitude",
        target_value=10000.0,
        unit="ft",
    )
    req2 = ProcedureRequirement(
        requirement_id="REQ-TP2",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Sec 2",
        description="Specific to TP-02",
        category=RequirementCategory.ENTRY_CONDITION,
        test_card_id="CARD-01",
        test_point_id="TP-02",
        parameter_id="altitude",
        target_value=15000.0,
        unit="ft",
    )
    profile = ProcedureRequirementProfile(
        profile_id="PROF-01",
        document_id="SOP-001",
        document_version="1.0.0",
        requirements=(req1, req2),
    )
    tp1_reqs = profile.get_requirements_for_test_point("TP-01")
    assert len(tp1_reqs) == 1
    assert tp1_reqs[0].requirement_id == "REQ-TP1"

    tp2_reqs = profile.get_requirements_for_test_point("TP-02")
    assert len(tp2_reqs) == 1
    assert tp2_reqs[0].requirement_id == "REQ-TP2"

    missing_tp_reqs = profile.get_requirements_for_test_point("TP-99")
    assert missing_tp_reqs == ()


def test_ambiguous_test_point_not_in_test_card(
    sample_test_operations: tuple[list[TestCard], list[TestPoint]],
) -> None:
    cards, points = sample_test_operations
    # CARD-PERF-01 only contains TP-STAB-01 and TP-CLIMB-02
    # Add a third point not in CARD-PERF-01
    extra_point = TestPoint(test_point_id="TP-EXTRA-03", context_id="CTX-01", name="Unlinked Point")
    points.append(extra_point)

    req = ProcedureRequirement(
        requirement_id="REQ-MISMATCH-POINT",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Sec 1",
        description="Mismatched point and card",
        category=RequirementCategory.EXECUTION,
        test_card_id="CARD-PERF-01",
        test_point_id="TP-EXTRA-03",
    )
    profile = ProcedureRequirementProfile(
        profile_id="PROF-01",
        document_id="SOP-001",
        document_version="1.0.0",
        requirements=(req,),
    )
    with pytest.raises(ConflictingRequirementError, match="not listed in test_card"):
        profile.validate_test_operations_compatibility(test_cards=cards, test_points=points)


# -----------------------------------------------------------------------------
# 6. Vehicle/profile/configuration mismatches
# -----------------------------------------------------------------------------


def test_requirement_document_and_vehicle_mismatch_with_profile() -> None:
    # Document mismatch
    req_bad_doc = ProcedureRequirement(
        requirement_id="REQ-01",
        document_id="SOP-002",
        document_version="1.0.0",
        source_reference="Sec 1",
        description="Doc mismatch",
        category=RequirementCategory.OPERATIONAL_CHECK,
    )
    with pytest.raises(ProcedureProfileMismatchError, match="does not match profile document_id"):
        ProcedureRequirementProfile(
            profile_id="PROF-01",
            document_id="SOP-001",
            document_version="1.0.0",
            requirements=(req_bad_doc,),
        )

    # Vehicle mismatch inside profile
    req_bad_veh = ProcedureRequirement(
        requirement_id="REQ-02",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Sec 1",
        description="Vehicle mismatch",
        category=RequirementCategory.OPERATIONAL_CHECK,
        vehicle_id="VEHICLE-B",
    )
    with pytest.raises(ProcedureProfileMismatchError, match="does not match profile vehicle_id"):
        ProcedureRequirementProfile(
            profile_id="PROF-01",
            document_id="SOP-001",
            document_version="1.0.0",
            vehicle_id="VEHICLE-A",
            requirements=(req_bad_veh,),
        )


def test_profile_cross_validation_mismatches(
    sample_vehicle_profile: VehicleProfile,
    sample_configuration_envelope: ConfigurationEnvelope,
) -> None:
    profile = ProcedureRequirementProfile(
        profile_id="PROF-01",
        document_id="SOP-001",
        document_version="1.0.0",
        vehicle_id="WRONG-VEHICLE",
        requirements=(),
    )
    with pytest.raises(
        ProcedureProfileMismatchError, match="does not match expected VehicleProfile"
    ):
        profile.validate_vehicle_profile_compatibility(sample_vehicle_profile)

    with pytest.raises(ProcedureProfileMismatchError, match="does not match ConfigurationEnvelope"):
        profile.validate_configuration_compatibility(sample_configuration_envelope)


# -----------------------------------------------------------------------------
# 7. Unknown test-card, test-point, or parameter references
# -----------------------------------------------------------------------------


def test_unknown_parameter_and_test_ops_references(
    sample_registry: ParameterRegistry,
    sample_vehicle_profile: VehicleProfile,
    sample_test_operations: tuple[list[TestCard], list[TestPoint]],
) -> None:
    cards, points = sample_test_operations

    # Unknown parameter in registry
    req_unknown_param = ProcedureRequirement(
        requirement_id="REQ-UNKNOWN-PARAM",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Sec 1",
        description="Unknown param test",
        category=RequirementCategory.ENTRY_CONDITION,
        parameter_id="unregistered_parameter",
        target_value=10.0,
        unit="ft",
    )
    prof1 = ProcedureRequirementProfile(
        profile_id="PROF-01",
        document_id="SOP-001",
        document_version="1.0.0",
        requirements=(req_unknown_param,),
    )
    with pytest.raises(UnknownParameterProcedureError, match="references unknown parameter"):
        prof1.validate_registry_compatibility(sample_registry)

    # Parameter not supported in VehicleProfile
    prof1_veh = ProcedureRequirementProfile(
        profile_id="PROF-01",
        document_id="SOP-001",
        document_version="1.0.0",
        vehicle_id="SYNTH-AIRCRAFT-01",
        requirements=(req_unknown_param,),
    )
    with pytest.raises(UnknownParameterProcedureError, match="is not supported by vehicle profile"):
        prof1_veh.validate_vehicle_profile_compatibility(sample_vehicle_profile)

    # Unknown test card
    req_unknown_card = ProcedureRequirement(
        requirement_id="REQ-UNKNOWN-CARD",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Sec 1",
        description="Unknown card test",
        category=RequirementCategory.OPERATIONAL_CHECK,
        test_card_id="NONEXISTENT-CARD",
    )
    prof2 = ProcedureRequirementProfile(
        profile_id="PROF-01",
        document_id="SOP-001",
        document_version="1.0.0",
        requirements=(req_unknown_card,),
    )
    with pytest.raises(UnknownTestOperationReferenceError, match="references unknown test_card_id"):
        prof2.validate_test_operations_compatibility(test_cards=cards)

    # Unknown test point
    req_unknown_point = ProcedureRequirement(
        requirement_id="REQ-UNKNOWN-POINT",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Sec 1",
        description="Unknown point test",
        category=RequirementCategory.OPERATIONAL_CHECK,
        test_point_id="NONEXISTENT-POINT",
    )
    prof3 = ProcedureRequirementProfile(
        profile_id="PROF-01",
        document_id="SOP-001",
        document_version="1.0.0",
        requirements=(req_unknown_point,),
    )
    with pytest.raises(
        UnknownTestOperationReferenceError, match="references unknown test_point_id"
    ):
        prof3.validate_test_operations_compatibility(test_points=points)


# -----------------------------------------------------------------------------
# 8. Numeric requirements with missing or incompatible units
# -----------------------------------------------------------------------------


def test_missing_or_forbidden_units() -> None:
    # Numeric requirement without unit
    with pytest.raises(
        InvalidProcedureRequirementError, match="requires an explicit non-empty unit"
    ):
        ProcedureRequirement(
            requirement_id="REQ-NO-UNIT",
            document_id="SOP-001",
            document_version="1.0.0",
            source_reference="Sec 1",
            description="Altitude without unit",
            category=RequirementCategory.ENTRY_CONDITION,
            parameter_id="altitude",
            target_value=10000.0,
            unit=None,
        )

    # Non-numeric requirement with unit
    with pytest.raises(InvalidProcedureRequirementError, match="must not specify a unit"):
        ProcedureRequirement(
            requirement_id="REQ-EXTRA-UNIT",
            document_id="SOP-001",
            document_version="1.0.0",
            source_reference="Sec 1",
            description="Check item with unit",
            category=RequirementCategory.OPERATIONAL_CHECK,
            unit="kts",
        )


def test_incompatible_unit_against_registry(sample_registry: ParameterRegistry) -> None:
    # Altitude registered with canonical_unit="ft", requirement specifies "m"
    req_incompat = ProcedureRequirement(
        requirement_id="REQ-ALT-METERS",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Sec 1",
        description="Altitude in meters",
        category=RequirementCategory.ENTRY_CONDITION,
        parameter_id="altitude",
        target_value=3000.0,
        unit="m",
    )
    profile = ProcedureRequirementProfile(
        profile_id="PROF-01",
        document_id="SOP-001",
        document_version="1.0.0",
        requirements=(req_incompat,),
    )
    with pytest.raises(
        IncompatibleProcedureUnitError, match="conflicts with registry canonical unit 'ft'"
    ):
        profile.validate_registry_compatibility(sample_registry)

    # Parameter without canonical unit permits explicit unit
    req_unitless = ProcedureRequirement(
        requirement_id="REQ-RATIO",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Sec 1",
        description="Ratio requirement",
        category=RequirementCategory.ENTRY_CONDITION,
        parameter_id="unitless_ratio",
        target_value=1.5,
        unit="ratio",
    )
    profile_ratio = ProcedureRequirementProfile(
        profile_id="PROF-01",
        document_id="SOP-001",
        document_version="1.0.0",
        requirements=(req_unitless,),
    )
    # Must not raise
    profile_ratio.validate_registry_compatibility(sample_registry)


# -----------------------------------------------------------------------------
# 9. Malformed YAML, missing fields, and forbidden extra fields
# -----------------------------------------------------------------------------


def test_load_malformed_yaml() -> None:
    path = FIXTURES_DIR / "malformed_procedure_profile.yaml"
    with pytest.raises(InvalidProcedureProfileError, match="Malformed YAML syntax"):
        load_procedure_requirement_profile_from_yaml(path)


def test_load_nonexistent_yaml() -> None:
    with pytest.raises(InvalidProcedureProfileError, match="Cannot read procedure profile file"):
        load_procedure_requirement_profile_from_yaml(FIXTURES_DIR / "nonexistent.yaml")


def test_load_missing_required_fields() -> None:
    path = FIXTURES_DIR / "missing_required_procedure_profile.yaml"
    with pytest.raises(InvalidProcedureProfileError, match="Validation failed"):
        load_procedure_requirement_profile_from_yaml(path)


def test_load_extra_forbidden_fields() -> None:
    path = FIXTURES_DIR / "extra_fields_procedure_profile.yaml"
    with pytest.raises(InvalidProcedureProfileError, match="Validation failed"):
        load_procedure_requirement_profile_from_yaml(path)


def test_load_duplicate_and_conflicting_fixtures() -> None:
    with pytest.raises(DuplicateRequirementError):
        load_procedure_requirement_profile_from_yaml(
            FIXTURES_DIR / "duplicate_requirement_id_profile.yaml"
        )

    with pytest.raises(ConflictingRequirementError):
        load_procedure_requirement_profile_from_yaml(
            FIXTURES_DIR / "conflicting_requirements_profile.yaml"
        )


# -----------------------------------------------------------------------------
# 10. Atomic failure without partial collection exposure
# -----------------------------------------------------------------------------


def test_atomic_failure_returns_no_partial_profile() -> None:
    bad_fixture = FIXTURES_DIR / "duplicate_requirement_id_profile.yaml"
    loaded_result = None
    with pytest.raises(DuplicateRequirementError):
        loaded_result = load_procedure_requirement_profile_from_yaml(bad_fixture)
    assert loaded_result is None


# -----------------------------------------------------------------------------
# 11. Deterministic repeated loading and lookup
# -----------------------------------------------------------------------------


def test_valid_yaml_loading_and_deterministic_lookups(
    sample_vehicle_profile: VehicleProfile,
    sample_configuration_envelope: ConfigurationEnvelope,
    sample_test_operations: tuple[list[TestCard], list[TestPoint]],
    sample_registry: ParameterRegistry,
) -> None:
    cards, points = sample_test_operations
    path = FIXTURES_DIR / "valid_procedure_profile.yaml"

    prof1 = load_procedure_requirement_profile_from_yaml(
        path,
        expected_profile=sample_vehicle_profile,
        expected_configuration=sample_configuration_envelope,
        test_cards=cards,
        test_points=points,
        parameter_registry=sample_registry,
    )
    prof2 = load_procedure_requirement_profile_from_yaml(
        path,
        expected_profile=sample_vehicle_profile,
        expected_configuration=sample_configuration_envelope,
        test_cards=cards,
        test_points=points,
        parameter_registry=sample_registry,
    )

    assert prof1 == prof2
    assert len(prof1.requirements) == 3

    # Test lookup methods
    req_alt = prof1.get_requirement_by_id("SYNTH-REQ-ENTRY-ALT")
    assert req_alt is not None
    assert req_alt.target_value == 10000.0

    tp_reqs = prof1.get_requirements_for_test_point("TP-STAB-01")
    assert len(tp_reqs) == 3

    card_reqs = prof1.get_requirements_for_test_card("CARD-PERF-01")
    assert len(card_reqs) == 3

    param_reqs = prof1.get_requirements_for_parameter("airspeed")
    assert len(param_reqs) == 1
    assert param_reqs[0].requirement_id == "SYNTH-REQ-HOLD-SPD"

    # Multi-criteria filtered lookup
    filtered = prof1.get_requirements(category=RequirementCategory.OPERATIONAL_CHECK)
    assert len(filtered) == 1
    assert filtered[0].requirement_id == "SYNTH-REQ-CHECK-FLAPS"


def test_loader_aliases() -> None:
    path = FIXTURES_DIR / "valid_procedure_profile.yaml"
    prof_a = load_procedure_requirements_from_yaml(path)
    prof_b = load_sop_requirements_from_yaml(path)
    assert prof_a == prof_b
    assert isinstance(prof_a, ProcedureRequirementSet)


# -----------------------------------------------------------------------------
# 12. Separation between requirement definition and verification status
# -----------------------------------------------------------------------------


def test_separation_of_definition_and_verification() -> None:
    req = ProcedureRequirement(
        requirement_id="REQ-ALT-01",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Section 3.1",
        description="Maintain entry altitude",
        category=RequirementCategory.ENTRY_CONDITION,
        parameter_id="altitude",
        target_value=10000.0,
        tolerance=50.0,
        unit="ft",
    )
    # The requirement definition itself does not possess a mutable satisfaction status
    assert not hasattr(req, "is_satisfied")

    # Verification is modeled as a distinct, independent evaluation object
    verif = ProcedureRequirementVerification(
        verification_id="VERIF-001",
        requirement_id=req.requirement_id,
        document_id=req.document_id,
        status=RequirementVerificationStatus.SATISFIED,
        observed_value=10010.0,
        observed_unit="ft",
        is_satisfied=True,
        epistemic_level=EpistemicLevel.PROCEDURAL,
    )
    assert verif.is_satisfied is True
    assert verif.status == RequirementVerificationStatus.SATISFIED
    assert verif.epistemic_level == EpistemicLevel.PROCEDURAL

    # Coherence validation on verification status
    with pytest.raises(ValueError, match="is_satisfied cannot be False when status is SATISFIED"):
        ProcedureRequirementVerification(
            verification_id="VERIF-BAD",
            requirement_id=req.requirement_id,
            document_id=req.document_id,
            status=RequirementVerificationStatus.SATISFIED,
            is_satisfied=False,
        )


# -----------------------------------------------------------------------------
# 13. Regression tests for existing vehicle, limit, registry, and test operations
# -----------------------------------------------------------------------------


def test_regression_vehicle_limits_and_sop_coexistence(
    sample_vehicle_profile: VehicleProfile,
    sample_configuration_envelope: ConfigurationEnvelope,
    sample_test_operations: tuple[list[TestCard], list[TestPoint]],
    sample_registry: ParameterRegistry,
) -> None:
    cards, points = sample_test_operations

    # Verify existing models instantiate cleanly
    assert sample_vehicle_profile.profile_id == "SYNTH-PROFILE-01"
    assert sample_configuration_envelope.configuration_id == "CLEAN-CRUISE"
    assert len(cards) == 1
    assert len(points) == 2
    assert sample_registry.contains("altitude") is True

    # Build procedure profile binding to vehicle profile and test operations
    req = ProcedureRequirement(
        requirement_id="REQ-STAB-01",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Section 1",
        description="Verify airspeed on stabilization point",
        category=RequirementCategory.TOLERANCE_BAND,
        vehicle_id=sample_vehicle_profile.vehicle_id,
        profile_version=sample_vehicle_profile.profile_version,
        configuration_id=sample_configuration_envelope.configuration_id,
        test_card_id=cards[0].test_card_id,
        test_point_id=points[0].test_point_id,
        parameter_id="airspeed",
        target_value=150.0,
        tolerance=5.0,
        unit="kts",
    )
    profile = ProcedureRequirementProfile(
        profile_id="PROF-STAB-01",
        document_id="SOP-001",
        document_version="1.0.0",
        vehicle_id=sample_vehicle_profile.vehicle_id,
        profile_version=sample_vehicle_profile.profile_version,
        configuration_id=sample_configuration_envelope.configuration_id,
        requirements=(req,),
    )

    # Perform full cross-validations
    profile.validate_vehicle_profile_compatibility(sample_vehicle_profile)
    profile.validate_configuration_compatibility(sample_configuration_envelope)
    profile.validate_test_operations_compatibility(test_cards=cards, test_points=points)
    profile.validate_registry_compatibility(sample_registry)

    # Confirm correct evaluation and lookup
    assert profile.get_requirement_by_id("REQ-STAB-01") == req
    assert len(profile.get_requirements_for_test_point(points[0].test_point_id)) == 1
    assert len(profile.get_requirements_for_test_point("NONEXISTENT")) == 0


def test_non_finite_numeric_values_rejected() -> None:
    with pytest.raises(InvalidProcedureRequirementError, match="must be a finite number"):
        ProcedureRequirement(
            requirement_id="REQ-NAN",
            document_id="SOP-001",
            document_version="1.0.0",
            source_reference="Sec 1",
            description="NaN target",
            category=RequirementCategory.ENTRY_CONDITION,
            parameter_id="altitude",
            target_value=float("nan"),
            unit="ft",
        )

    with pytest.raises(InvalidProcedureRequirementError, match="must be a finite number"):
        ProcedureRequirement(
            requirement_id="REQ-INF",
            document_id="SOP-001",
            document_version="1.0.0",
            source_reference="Sec 1",
            description="Inf target",
            category=RequirementCategory.ENTRY_CONDITION,
            parameter_id="altitude",
            target_value=float("inf"),
            unit="ft",
        )


def test_negative_tolerance_and_tolerance_without_target_rejected() -> None:
    with pytest.raises(InvalidProcedureRequirementError, match="must be non-negative"):
        ProcedureRequirement(
            requirement_id="REQ-NEG-TOL",
            document_id="SOP-001",
            document_version="1.0.0",
            source_reference="Sec 1",
            description="Negative tolerance",
            category=RequirementCategory.ENTRY_CONDITION,
            parameter_id="altitude",
            target_value=10000.0,
            tolerance=-5.0,
            unit="ft",
        )

    with pytest.raises(
        InvalidProcedureRequirementError, match="cannot be specified without target_value"
    ):
        ProcedureRequirement(
            requirement_id="REQ-NO-TARGET-TOL",
            document_id="SOP-001",
            document_version="1.0.0",
            source_reference="Sec 1",
            description="Tolerance without target",
            category=RequirementCategory.ENTRY_CONDITION,
            parameter_id="altitude",
            tolerance=5.0,
            unit="ft",
        )


def test_duplicate_qualitative_checklist_items_rejected() -> None:
    req1 = ProcedureRequirement(
        requirement_id="REQ-CHK-01",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Sec 1",
        description="Verify pitot heat ON",
        category=RequirementCategory.OPERATIONAL_CHECK,
        test_point_id="TP-01",
    )
    req2 = ProcedureRequirement(
        requirement_id="REQ-CHK-02",
        document_id="SOP-001",
        document_version="1.0.0",
        source_reference="Sec 2",
        description="Verify pitot heat ON",
        category=RequirementCategory.OPERATIONAL_CHECK,
        test_point_id="TP-01",
    )
    with pytest.raises(ConflictingRequirementError, match="Duplicate non-numeric checklist item"):
        ProcedureRequirementProfile(
            profile_id="PROF-CHK",
            document_id="SOP-001",
            document_version="1.0.0",
            requirements=(req1, req2),
        )
