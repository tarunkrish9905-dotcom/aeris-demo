"""
Focused tests for the Controlled Vehicle Engineering Evaluation Engine.

Covers:
1. Upper-bound evaluation (within, on, outside).
2. Lower-bound evaluation (within, on, outside).
3. Band evaluation (within, on lower/upper, outside).
4. Boundary equality inclusivity policy (value == bound -> WITHIN_BOUNDS, margin=0.0).
5. Mathematical margin calculation and sign conventions (+ inside, 0 on bound, - outside).
6. Severity vs outcome separation (severity preserved regardless of outcome).
7. Unit compatibility and incompatible unit enforcement (INCONCLUSIVE, no silent conversion).
8. Vehicle profile, configuration envelope, and profile version mismatch validation (ERROR).
9. Configuration-scoped applicability and observation configuration discrepancy detection.
10. Unconfigured parameter limits (NOT_EVALUATED, never auto-pass).
11. Quantitative procedure requirements (symmetric, asymmetric, min-only, max-only).
12. Qualitative checklist requirements (CONFIRMED -> SATISFIED, FAILED -> VIOLATED, UNVERIFIED).
13. Non-finite values (NaN, Inf -> INCONCLUSIVE) and missing measurements.
14. Test card and test point context filtering.
15. Immutability, side-effect freedom, and result summary properties.
16. Traceability and provenance tracking.

DISCLAIMER: All test vehicles, parameters, limits, and procedure requirements in this module
are synthetic and non-operational. They do NOT establish flight safety or manufacturer specs.
"""

from __future__ import annotations

import datetime
import math

import pytest

from aeris.core.limits.models import (
    EngineeringLimitProfile,
    LimitSeverity,
    LimitType,
    ParameterLimitDefinition,
)
from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.results.types import ResultStatus
from aeris.core.sop.models import (
    ProcedureRequirement,
    ProcedureRequirementProfile,
    RequirementCategory,
    RequirementVerificationStatus,
)
from aeris.core.vehicle.envelope_models import ConfigurationEnvelope
from aeris.core.vehicle.profile_models import VehicleProfile
from aeris.core.vehicle_evaluation.engine import VehicleEngineeringEvaluationEngine
from aeris.core.vehicle_evaluation.models import (
    EvaluationObservation,
    EvaluationOutcome,
    VehicleEvaluationRequest,
)


@pytest.fixture
def sample_vehicle_profile() -> VehicleProfile:
    """Return a standard synthetic VehicleProfile for testing."""
    return VehicleProfile(
        profile_id="PROF-TEST-01",
        vehicle_id="AC-TEST-X1",
        profile_version="1.0.0",
        name="Synthetic Evaluation Test Profile",
        supported_parameter_ids=("airspeed", "altitude", "nz", "gear_handle"),
    )


@pytest.fixture
def sample_configuration_envelope() -> ConfigurationEnvelope:
    """Return an active ConfigurationEnvelope for testing."""
    return ConfigurationEnvelope(
        configuration_id="CRUISE",
        vehicle_id="AC-TEST-X1",
        profile_id="PROF-TEST-01",
        profile_version="1.0.0",
        name="Cruise Configuration",
    )


@pytest.fixture
def sample_limit_profile() -> EngineeringLimitProfile:
    """Return an EngineeringLimitProfile with upper, lower, and band limits."""
    return EngineeringLimitProfile(
        limit_profile_id="LIM-PROF-01",
        vehicle_id="AC-TEST-X1",
        profile_id="PROF-TEST-01",
        profile_version="1.0.0",
        limits=(
            ParameterLimitDefinition(
                limit_id="LIM-AIRSPEED-VNE",
                parameter_id="airspeed",
                limit_type=LimitType.UPPER_BOUND,
                severity=LimitSeverity.EXCEEDANCE,
                limit_value=250.0,
                unit="kts",
                vehicle_id="AC-TEST-X1",
                profile_id="PROF-TEST-01",
                profile_version="1.0.0",
                source_reference="SYN-POH-SEC-2",
            ),
            ParameterLimitDefinition(
                limit_id="LIM-ALTITUDE-MIN",
                parameter_id="altitude",
                limit_type=LimitType.LOWER_BOUND,
                severity=LimitSeverity.WARNING,
                limit_value=1000.0,
                unit="ft",
                vehicle_id="AC-TEST-X1",
                profile_id="PROF-TEST-01",
                profile_version="1.0.0",
                source_reference="SYN-SOP-SEC-4",
            ),
            ParameterLimitDefinition(
                limit_id="LIM-NZ-BAND",
                parameter_id="nz",
                limit_type=LimitType.BAND,
                severity=LimitSeverity.CAUTION,
                lower_bound=-1.0,
                upper_bound=2.5,
                unit="g",
                vehicle_id="AC-TEST-X1",
                profile_id="PROF-TEST-01",
                profile_version="1.0.0",
                source_reference="SYN-STRUCT-01",
            ),
        ),
    )


@pytest.fixture
def sample_procedure_profile() -> ProcedureRequirementProfile:
    """Return a ProcedureRequirementProfile with quantitative and qualitative requirements."""
    return ProcedureRequirementProfile(
        profile_id="PROC-PROF-01",
        document_id="SOP-FLT-001",
        document_version="2.0",
        vehicle_id="AC-TEST-X1",
        profile_version="1.0.0",
        requirements=(
            ProcedureRequirement(
                requirement_id="REQ-TARGET-ALT",
                document_id="SOP-FLT-001",
                document_version="2.0",
                category=RequirementCategory.TOLERANCE_BAND,
                parameter_id="altitude",
                target_value=10000.0,
                tolerance=100.0,
                unit="ft",
                source_reference="SOP-FLT-001 Section 3.1",
                description="Maintain cruise altitude within 100 ft tolerance",
            ),
            ProcedureRequirement(
                requirement_id="REQ-ASYM-SPEED",
                document_id="SOP-FLT-001",
                document_version="2.0",
                category=RequirementCategory.TOLERANCE_BAND,
                parameter_id="airspeed",
                target_value=150.0,
                tolerance_lower=5.0,
                tolerance_upper=10.0,
                unit="kts",
                source_reference="SOP-FLT-001 Section 3.2",
                description="Maintain cruise speed -5/+10 kts",
            ),
            ProcedureRequirement(
                requirement_id="REQ-GEAR-UP",
                document_id="SOP-FLT-001",
                document_version="2.0",
                category=RequirementCategory.OPERATIONAL_CHECK,
                source_reference="SOP-FLT-001 Section 2.4",
                description="Confirm landing gear handle UP",
            ),
        ),
    )


@pytest.fixture
def evaluation_engine() -> VehicleEngineeringEvaluationEngine:
    """Return a VehicleEngineeringEvaluationEngine instance."""
    return VehicleEngineeringEvaluationEngine()


# -----------------------------------------------------------------------------
# Test 1: Upper-Bound Evaluation
# -----------------------------------------------------------------------------
def test_upper_bound_evaluation(
    evaluation_engine: VehicleEngineeringEvaluationEngine,
    sample_vehicle_profile: VehicleProfile,
    sample_limit_profile: EngineeringLimitProfile,
) -> None:
    """Test upper-bound limit: within (< limit), on boundary (== limit), exceeded (> limit)."""
    # Airspeed VNE limit is 250.0 kts
    obs_below = EvaluationObservation(
        observation_id="OBS-V-240",
        parameter_id="airspeed",
        value=240.0,
        unit="kts",
        source_reference="test_telem",
    )
    obs_on = EvaluationObservation(
        observation_id="OBS-V-250",
        parameter_id="airspeed",
        value=250.0,
        unit="kts",
        source_reference="test_telem",
    )
    obs_above = EvaluationObservation(
        observation_id="OBS-V-260",
        parameter_id="airspeed",
        value=260.0,
        unit="kts",
        source_reference="test_telem",
    )

    req = VehicleEvaluationRequest(
        request_id="REQ-UPPER-TEST",
        vehicle_profile=sample_vehicle_profile,
        limit_profile=sample_limit_profile,
        observations=(obs_below, obs_on, obs_above),
        evaluate_procedures=False,
    )

    result = evaluation_engine.evaluate(req)

    assert result.status == ResultStatus.WARNING  # Warning due to exceedance
    assert len(result.limit_evaluations) == 3

    # Within bound
    rec_below = next(r for r in result.limit_evaluations if r.observation_id == "OBS-V-240")
    assert rec_below.outcome == EvaluationOutcome.WITHIN_BOUNDS
    assert rec_below.is_exceeded is False
    assert rec_below.margin == 10.0

    # On boundary (inclusive policy: value == limit is WITHIN_BOUNDS, margin=0.0)
    rec_on = next(r for r in result.limit_evaluations if r.observation_id == "OBS-V-250")
    assert rec_on.outcome == EvaluationOutcome.WITHIN_BOUNDS
    assert rec_on.is_exceeded is False
    assert rec_on.margin == 0.0

    # Exceeded
    rec_above = next(r for r in result.limit_evaluations if r.observation_id == "OBS-V-260")
    assert rec_above.outcome == EvaluationOutcome.OUTSIDE_BOUNDS
    assert rec_above.is_exceeded is True
    assert rec_above.margin == -10.0


# -----------------------------------------------------------------------------
# Test 2: Lower-Bound Evaluation
# -----------------------------------------------------------------------------
def test_lower_bound_evaluation(
    evaluation_engine: VehicleEngineeringEvaluationEngine,
    sample_vehicle_profile: VehicleProfile,
    sample_limit_profile: EngineeringLimitProfile,
) -> None:
    """Test lower-bound limit: within (> limit), on boundary (== limit), exceeded (< limit)."""
    # Altitude min is 1000.0 ft
    obs_above = EvaluationObservation(
        observation_id="OBS-ALT-1500",
        parameter_id="altitude",
        value=1500.0,
        unit="ft",
        source_reference="test_telem",
    )
    obs_on = EvaluationObservation(
        observation_id="OBS-ALT-1000",
        parameter_id="altitude",
        value=1000.0,
        unit="ft",
        source_reference="test_telem",
    )
    obs_below = EvaluationObservation(
        observation_id="OBS-ALT-800",
        parameter_id="altitude",
        value=800.0,
        unit="ft",
        source_reference="test_telem",
    )

    req = VehicleEvaluationRequest(
        request_id="REQ-LOWER-TEST",
        vehicle_profile=sample_vehicle_profile,
        limit_profile=sample_limit_profile,
        observations=(obs_above, obs_on, obs_below),
        evaluate_procedures=False,
    )

    result = evaluation_engine.evaluate(req)

    # Within bound
    rec_above = next(r for r in result.limit_evaluations if r.observation_id == "OBS-ALT-1500")
    assert rec_above.outcome == EvaluationOutcome.WITHIN_BOUNDS
    assert rec_above.is_exceeded is False
    assert rec_above.margin == 500.0

    # On boundary
    rec_on = next(r for r in result.limit_evaluations if r.observation_id == "OBS-ALT-1000")
    assert rec_on.outcome == EvaluationOutcome.WITHIN_BOUNDS
    assert rec_on.is_exceeded is False
    assert rec_on.margin == 0.0

    # Exceeded
    rec_below = next(r for r in result.limit_evaluations if r.observation_id == "OBS-ALT-800")
    assert rec_below.outcome == EvaluationOutcome.OUTSIDE_BOUNDS
    assert rec_below.is_exceeded is True
    assert rec_below.margin == -200.0


# -----------------------------------------------------------------------------
# Test 3: Band Evaluation
# -----------------------------------------------------------------------------
def test_band_evaluation(
    evaluation_engine: VehicleEngineeringEvaluationEngine,
    sample_vehicle_profile: VehicleProfile,
    sample_limit_profile: EngineeringLimitProfile,
) -> None:
    """Test band limit [-1.0, 2.5]: inside, on lower/upper bound, below, above."""
    obs_inside = EvaluationObservation(
        observation_id="OBS-NZ-10",
        parameter_id="nz",
        value=1.0,
        unit="g",
        source_reference="test_telem",
    )
    obs_lower_bound = EvaluationObservation(
        observation_id="OBS-NZ-MINUS-1",
        parameter_id="nz",
        value=-1.0,
        unit="g",
        source_reference="test_telem",
    )
    obs_upper_bound = EvaluationObservation(
        observation_id="OBS-NZ-PLUS-25",
        parameter_id="nz",
        value=2.5,
        unit="g",
        source_reference="test_telem",
    )
    obs_below = EvaluationObservation(
        observation_id="OBS-NZ-BELOW",
        parameter_id="nz",
        value=-1.5,
        unit="g",
        source_reference="test_telem",
    )
    obs_above = EvaluationObservation(
        observation_id="OBS-NZ-ABOVE",
        parameter_id="nz",
        value=3.0,
        unit="g",
        source_reference="test_telem",
    )

    req = VehicleEvaluationRequest(
        request_id="REQ-BAND-TEST",
        vehicle_profile=sample_vehicle_profile,
        limit_profile=sample_limit_profile,
        observations=(obs_inside, obs_lower_bound, obs_upper_bound, obs_below, obs_above),
        evaluate_procedures=False,
    )

    result = evaluation_engine.evaluate(req)

    # Inside
    rec_inside = next(r for r in result.limit_evaluations if r.observation_id == "OBS-NZ-10")
    assert rec_inside.outcome == EvaluationOutcome.WITHIN_BOUNDS
    assert rec_inside.is_exceeded is False
    assert rec_inside.margin == 1.5  # min(1.0 - (-1.0), 2.5 - 1.0) = min(2.0, 1.5) = 1.5

    # On lower bound
    rec_low = next(r for r in result.limit_evaluations if r.observation_id == "OBS-NZ-MINUS-1")
    assert rec_low.outcome == EvaluationOutcome.WITHIN_BOUNDS
    assert rec_low.is_exceeded is False
    assert rec_low.margin == 0.0

    # On upper bound
    rec_up = next(r for r in result.limit_evaluations if r.observation_id == "OBS-NZ-PLUS-25")
    assert rec_up.outcome == EvaluationOutcome.WITHIN_BOUNDS
    assert rec_up.is_exceeded is False
    assert rec_up.margin == 0.0

    # Below
    rec_below = next(r for r in result.limit_evaluations if r.observation_id == "OBS-NZ-BELOW")
    assert rec_below.outcome == EvaluationOutcome.OUTSIDE_BOUNDS
    assert rec_below.is_exceeded is True
    assert rec_below.margin == -0.5

    # Above
    rec_above = next(r for r in result.limit_evaluations if r.observation_id == "OBS-NZ-ABOVE")
    assert rec_above.outcome == EvaluationOutcome.OUTSIDE_BOUNDS
    assert rec_above.is_exceeded is True
    assert rec_above.margin == -0.5


# -----------------------------------------------------------------------------
# Test 4: Boundary Equality Inclusivity Policy
# -----------------------------------------------------------------------------
def test_boundary_inclusivity_policy(
    evaluation_engine: VehicleEngineeringEvaluationEngine,
    sample_vehicle_profile: VehicleProfile,
    sample_limit_profile: EngineeringLimitProfile,
) -> None:
    """Verify strictly that exact boundary equality is WITHIN_BOUNDS with margin=0.0."""
    obs_vne = EvaluationObservation(
        observation_id="OBS-EXACT-VNE",
        parameter_id="airspeed",
        value=250.0,
        unit="kts",
        source_reference="test_telem",
    )
    req = VehicleEvaluationRequest(
        request_id="REQ-INCLUSIVE-TEST",
        vehicle_profile=sample_vehicle_profile,
        limit_profile=sample_limit_profile,
        observations=(obs_vne,),
        evaluate_procedures=False,
    )
    result = evaluation_engine.evaluate(req)
    assert result.status == ResultStatus.SUCCESS
    rec = result.limit_evaluations[0]
    assert rec.outcome == EvaluationOutcome.WITHIN_BOUNDS
    assert rec.is_exceeded is False
    assert rec.margin == 0.0


# -----------------------------------------------------------------------------
# Test 5: Mathematical Margin Calculation & Sign Conventions
# -----------------------------------------------------------------------------
def test_margin_calculation_and_signs(
    evaluation_engine: VehicleEngineeringEvaluationEngine,
    sample_vehicle_profile: VehicleProfile,
    sample_limit_profile: EngineeringLimitProfile,
) -> None:
    """Confirm sign convention: positive inside bounds, 0.0 on bound, negative outside bounds."""
    obs_within = EvaluationObservation(
        observation_id="OBS-POS",
        parameter_id="airspeed",
        value=200.0,
        unit="kts",
        source_reference="test",
    )
    obs_zero = EvaluationObservation(
        observation_id="OBS-ZERO",
        parameter_id="airspeed",
        value=250.0,
        unit="kts",
        source_reference="test",
    )
    obs_neg = EvaluationObservation(
        observation_id="OBS-NEG",
        parameter_id="airspeed",
        value=275.0,
        unit="kts",
        source_reference="test",
    )
    req = VehicleEvaluationRequest(
        request_id="REQ-MARGINS",
        vehicle_profile=sample_vehicle_profile,
        limit_profile=sample_limit_profile,
        observations=(obs_within, obs_zero, obs_neg),
        evaluate_procedures=False,
    )
    result = evaluation_engine.evaluate(req)

    rec_pos = next(r for r in result.limit_evaluations if r.observation_id == "OBS-POS")
    rec_zero = next(r for r in result.limit_evaluations if r.observation_id == "OBS-ZERO")
    rec_neg = next(r for r in result.limit_evaluations if r.observation_id == "OBS-NEG")

    assert rec_pos.margin is not None and rec_pos.margin > 0.0
    assert rec_zero.margin == 0.0
    assert rec_neg.margin is not None and rec_neg.margin < 0.0


# -----------------------------------------------------------------------------
# Test 6: Severity vs Outcome Separation
# -----------------------------------------------------------------------------
def test_severity_vs_outcome_separation(
    evaluation_engine: VehicleEngineeringEvaluationEngine,
    sample_vehicle_profile: VehicleProfile,
    sample_limit_profile: EngineeringLimitProfile,
) -> None:
    """Ensure LimitEvaluationRecord.severity retains configured severity regardless of outcome."""
    # LIM-AIRSPEED-VNE has severity=EXCEEDANCE
    obs_pass = EvaluationObservation(
        observation_id="OBS-PASS",
        parameter_id="airspeed",
        value=200.0,
        unit="kts",
        source_reference="test",
    )
    obs_fail = EvaluationObservation(
        observation_id="OBS-FAIL",
        parameter_id="airspeed",
        value=280.0,
        unit="kts",
        source_reference="test",
    )
    req = VehicleEvaluationRequest(
        request_id="REQ-SEV-TEST",
        vehicle_profile=sample_vehicle_profile,
        limit_profile=sample_limit_profile,
        observations=(obs_pass, obs_fail),
        evaluate_procedures=False,
    )
    result = evaluation_engine.evaluate(req)

    rec_pass = next(r for r in result.limit_evaluations if r.observation_id == "OBS-PASS")
    rec_fail = next(r for r in result.limit_evaluations if r.observation_id == "OBS-FAIL")

    # Both records preserve the limit definition severity EXCEEDANCE
    assert rec_pass.severity == LimitSeverity.EXCEEDANCE
    assert rec_pass.outcome == EvaluationOutcome.WITHIN_BOUNDS

    assert rec_fail.severity == LimitSeverity.EXCEEDANCE
    assert rec_fail.outcome == EvaluationOutcome.OUTSIDE_BOUNDS


# -----------------------------------------------------------------------------
# Test 7: Unit Compatibility & Incompatible Unit Enforcement
# -----------------------------------------------------------------------------
def test_unit_compatibility_and_incompatible_unit(
    evaluation_engine: VehicleEngineeringEvaluationEngine,
    sample_vehicle_profile: VehicleProfile,
    sample_limit_profile: EngineeringLimitProfile,
) -> None:
    """Unit mismatch yields INCONCLUSIVE outcome with ERR_INCOMPATIBLE_UNIT issue; no conversion."""
    obs_wrong_unit = EvaluationObservation(
        observation_id="OBS-WRONG-UNIT",
        parameter_id="airspeed",
        value=120.0,
        unit="m/s",  # Limit is configured in kts
        source_reference="test",
    )
    obs_missing_unit = EvaluationObservation(
        observation_id="OBS-NO-UNIT",
        parameter_id="airspeed",
        value=200.0,
        unit=None,
        source_reference="test",
    )
    req = VehicleEvaluationRequest(
        request_id="REQ-UNIT-TEST",
        vehicle_profile=sample_vehicle_profile,
        limit_profile=sample_limit_profile,
        observations=(obs_wrong_unit, obs_missing_unit),
        evaluate_procedures=False,
    )
    result = evaluation_engine.evaluate(req)

    # Inconclusive outcome due to unit incompatibilities
    assert result.status == ResultStatus.WARNING
    assert any(i.issue_code == "ERR_INCOMPATIBLE_UNIT" for i in result.issues)

    rec_wrong = next(r for r in result.limit_evaluations if r.observation_id == "OBS-WRONG-UNIT")
    assert rec_wrong.outcome == EvaluationOutcome.INCONCLUSIVE
    assert rec_wrong.is_exceeded is False
    assert rec_wrong.margin is None

    rec_no = next(r for r in result.limit_evaluations if r.observation_id == "OBS-NO-UNIT")
    assert rec_no.outcome == EvaluationOutcome.INCONCLUSIVE
    assert rec_no.is_exceeded is False


# -----------------------------------------------------------------------------
# Test 8: Vehicle Profile & Version Mismatch Validation
# -----------------------------------------------------------------------------
def test_vehicle_profile_and_version_mismatch(
    evaluation_engine: VehicleEngineeringEvaluationEngine,
    sample_vehicle_profile: VehicleProfile,
    sample_configuration_envelope: ConfigurationEnvelope,
    sample_limit_profile: EngineeringLimitProfile,
) -> None:
    """Mismatches between envelope/limit and vehicle profile must produce ResultStatus.ERROR."""
    # Envelope vehicle mismatch
    mismatched_env = ConfigurationEnvelope(
        configuration_id="CRUISE",
        vehicle_id="AC-DIFFERENT-VEHICLE",
        profile_id="PROF-TEST-01",
        profile_version="1.0.0",
        name="Cruise",
    )
    req_bad_env = VehicleEvaluationRequest(
        request_id="REQ-BAD-ENV",
        vehicle_profile=sample_vehicle_profile,
        configuration_envelope=mismatched_env,
    )
    result_env = evaluation_engine.evaluate(req_bad_env)
    assert result_env.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_VEHICLE_PROFILE_MISMATCH" for i in result_env.issues)

    # Limit profile version mismatch
    mismatched_limit = EngineeringLimitProfile(
        limit_profile_id="LIM-PROF-01",
        vehicle_id="AC-TEST-X1",
        profile_id="PROF-TEST-01",
        profile_version="2.0.0",  # Does not match vehicle_profile "1.0.0"
        limits=(
            ParameterLimitDefinition(
                limit_id="LIM-AIRSPEED-VNE",
                parameter_id="airspeed",
                limit_type=LimitType.UPPER_BOUND,
                severity=LimitSeverity.EXCEEDANCE,
                limit_value=250.0,
                unit="kts",
                vehicle_id="AC-TEST-X1",
                profile_id="PROF-TEST-01",
                profile_version="2.0.0",
                source_reference="SYN-POH-SEC-2",
            ),
        ),
    )
    req_bad_limit = VehicleEvaluationRequest(
        request_id="REQ-BAD-LIM",
        vehicle_profile=sample_vehicle_profile,
        limit_profile=mismatched_limit,
    )
    result_lim = evaluation_engine.evaluate(req_bad_limit)
    assert result_lim.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_LIMIT_PROFILE_MISMATCH" for i in result_lim.issues)


# -----------------------------------------------------------------------------
# Test 9: Configuration-Scoped Applicability & Discrepancies
# -----------------------------------------------------------------------------
def test_configuration_scoped_applicability_and_discrepancy(
    evaluation_engine: VehicleEngineeringEvaluationEngine,
    sample_vehicle_profile: VehicleProfile,
    sample_configuration_envelope: ConfigurationEnvelope,
) -> None:
    """Observations flagged with differing configuration generate discrepancies and warnings."""
    # Active configuration is CRUISE
    obs_matching = EvaluationObservation(
        observation_id="OBS-CRUISE",
        parameter_id="airspeed",
        value=200.0,
        unit="kts",
        configuration_id="CRUISE",
        source_reference="test",
    )
    obs_discrepant = EvaluationObservation(
        observation_id="OBS-LANDING",
        parameter_id="airspeed",
        value=110.0,
        unit="kts",
        configuration_id="LANDING",  # Conflicts with active CRUISE envelope
        source_reference="test",
    )

    req = VehicleEvaluationRequest(
        request_id="REQ-CONFIG-TEST",
        vehicle_profile=sample_vehicle_profile,
        configuration_envelope=sample_configuration_envelope,
        observations=(obs_matching, obs_discrepant),
        evaluate_procedures=False,
    )
    result = evaluation_engine.evaluate(req)

    assert len(result.configuration_discrepancies) == 1
    assert "LANDING" in result.configuration_discrepancies[0]
    assert any(i.issue_code == "VEH_CONFIG_DISCREPANCY" for i in result.issues)


# -----------------------------------------------------------------------------
# Test 10: Unconfigured Parameter Limits
# -----------------------------------------------------------------------------
def test_unconfigured_parameter_evaluation(
    evaluation_engine: VehicleEngineeringEvaluationEngine,
    sample_vehicle_profile: VehicleProfile,
    sample_limit_profile: EngineeringLimitProfile,
) -> None:
    """Observations with no configured limits yield NOT_EVALUATED, never auto-passing."""
    obs_unconfigured = EvaluationObservation(
        observation_id="OBS-UNKNOWN-PARAM",
        parameter_id="unconfigured_pressure",
        value=14.7,
        unit="psi",
        source_reference="test",
    )
    req = VehicleEvaluationRequest(
        request_id="REQ-UNCONFIG",
        vehicle_profile=sample_vehicle_profile,
        limit_profile=sample_limit_profile,
        observations=(obs_unconfigured,),
        evaluate_procedures=False,
    )
    result = evaluation_engine.evaluate(req)

    assert len(result.limit_evaluations) == 1
    rec = result.limit_evaluations[0]
    assert rec.outcome == EvaluationOutcome.NOT_EVALUATED
    assert rec.is_exceeded is False
    assert rec.margin is None


# -----------------------------------------------------------------------------
# Test 11: Quantitative SOP Procedure Requirements
# -----------------------------------------------------------------------------
def test_quantitative_procedure_requirements(
    evaluation_engine: VehicleEngineeringEvaluationEngine,
    sample_vehicle_profile: VehicleProfile,
    sample_procedure_profile: ProcedureRequirementProfile,
) -> None:
    """Quantitative requirements evaluate SATISFIED or VIOLATED deterministically."""
    # REQ-TARGET-ALT: 10000 +/- 100 ft -> [9900, 10100]
    # REQ-ASYM-SPEED: 150 -5/+10 kts -> [145, 160]
    obs_alt_good = EvaluationObservation(
        observation_id="OBS-ALT-GOOD",
        parameter_id="altitude",
        value=10050.0,
        unit="ft",
        source_reference="test",
    )
    obs_speed_bad = EvaluationObservation(
        observation_id="OBS-SPD-BAD",
        parameter_id="airspeed",
        value=140.0,  # Below lower bound of 145 kts
        unit="kts",
        source_reference="test",
    )

    req = VehicleEvaluationRequest(
        request_id="REQ-PROC-TEST",
        vehicle_profile=sample_vehicle_profile,
        procedure_profile=sample_procedure_profile,
        observations=(obs_alt_good, obs_speed_bad),
        evaluate_limits=False,
    )
    result = evaluation_engine.evaluate(req)

    # Overall warning because of speed violation
    assert result.status == ResultStatus.WARNING
    assert result.procedure_violation_count == 1
    assert result.has_violations is True

    rec_alt = next(r for r in result.procedure_evaluations if r.requirement_id == "REQ-TARGET-ALT")
    assert rec_alt.outcome == EvaluationOutcome.SATISFIED
    assert rec_alt.verification_status == RequirementVerificationStatus.SATISFIED
    assert rec_alt.is_satisfied is True
    assert rec_alt.margin == 50.0  # min(10050 - 9900, 10100 - 10050) = 50.0

    rec_spd = next(r for r in result.procedure_evaluations if r.requirement_id == "REQ-ASYM-SPEED")
    assert rec_spd.outcome == EvaluationOutcome.VIOLATED
    assert rec_spd.verification_status == RequirementVerificationStatus.VIOLATED
    assert rec_spd.is_satisfied is False
    assert rec_spd.margin == -5.0  # 140 - 145 = -5.0


# -----------------------------------------------------------------------------
# Test 12: Qualitative Checklist Requirements
# -----------------------------------------------------------------------------
def test_qualitative_checklist_requirements(
    evaluation_engine: VehicleEngineeringEvaluationEngine,
    sample_vehicle_profile: VehicleProfile,
    sample_procedure_profile: ProcedureRequirementProfile,
) -> None:
    """Qualitative requirements require explicit qualitative observations; no auto-satisfaction."""
    # 1. No qualitative observation supplied -> UNVERIFIED / NOT_EVALUATED
    req_empty = VehicleEvaluationRequest(
        request_id="REQ-QUAL-EMPTY",
        vehicle_profile=sample_vehicle_profile,
        procedure_profile=sample_procedure_profile,
        observations=(),
        evaluate_limits=False,
    )
    res_empty = evaluation_engine.evaluate(req_empty)
    rec_gear_empty = next(
        r for r in res_empty.procedure_evaluations if r.requirement_id == "REQ-GEAR-UP"
    )
    assert rec_gear_empty.verification_status == RequirementVerificationStatus.UNVERIFIED
    assert rec_gear_empty.outcome == EvaluationOutcome.NOT_EVALUATED
    assert rec_gear_empty.is_satisfied is None

    # 2. Confirmed qualitative observation -> SATISFIED
    obs_gear_confirmed = EvaluationObservation(
        observation_id="OBS-GEAR-CONF",
        parameter_id="REQ-GEAR-UP",
        is_qualitative=True,
        qualitative_state="CONFIRMED",
        source_reference="crew_checklist",
    )
    req_confirmed = VehicleEvaluationRequest(
        request_id="REQ-QUAL-CONF",
        vehicle_profile=sample_vehicle_profile,
        procedure_profile=sample_procedure_profile,
        observations=(obs_gear_confirmed,),
        evaluate_limits=False,
    )
    res_confirmed = evaluation_engine.evaluate(req_confirmed)
    rec_gear_conf = next(
        r for r in res_confirmed.procedure_evaluations if r.requirement_id == "REQ-GEAR-UP"
    )
    assert rec_gear_conf.verification_status == RequirementVerificationStatus.SATISFIED
    assert rec_gear_conf.outcome == EvaluationOutcome.SATISFIED
    assert rec_gear_conf.is_satisfied is True

    # 3. Failed qualitative observation -> VIOLATED
    obs_gear_fail = EvaluationObservation(
        observation_id="OBS-GEAR-FAIL",
        parameter_id="REQ-GEAR-UP",
        is_qualitative=True,
        qualitative_state="FAILED",
        source_reference="crew_checklist",
    )
    req_failed = VehicleEvaluationRequest(
        request_id="REQ-QUAL-FAIL",
        vehicle_profile=sample_vehicle_profile,
        procedure_profile=sample_procedure_profile,
        observations=(obs_gear_fail,),
        evaluate_limits=False,
    )
    res_failed = evaluation_engine.evaluate(req_failed)
    rec_gear_f = next(
        r for r in res_failed.procedure_evaluations if r.requirement_id == "REQ-GEAR-UP"
    )
    assert rec_gear_f.verification_status == RequirementVerificationStatus.VIOLATED
    assert rec_gear_f.outcome == EvaluationOutcome.VIOLATED
    assert rec_gear_f.is_satisfied is False


# -----------------------------------------------------------------------------
# Test 13: Non-Finite Values & Missing Observations
# -----------------------------------------------------------------------------
def test_non_finite_values_and_missing_observations(
    evaluation_engine: VehicleEngineeringEvaluationEngine,
    sample_vehicle_profile: VehicleProfile,
    sample_limit_profile: EngineeringLimitProfile,
) -> None:
    """Non-finite observations (NaN, Inf) must be reported as INCONCLUSIVE with issues."""
    obs_nan = EvaluationObservation(
        observation_id="OBS-NAN",
        parameter_id="airspeed",
        value=math.nan,
        unit="kts",
        source_reference="test",
    )
    obs_inf = EvaluationObservation(
        observation_id="OBS-INF",
        parameter_id="altitude",
        value=math.inf,
        unit="ft",
        source_reference="test",
    )
    req = VehicleEvaluationRequest(
        request_id="REQ-NON-FINITE",
        vehicle_profile=sample_vehicle_profile,
        limit_profile=sample_limit_profile,
        observations=(obs_nan, obs_inf),
        evaluate_procedures=False,
    )
    result = evaluation_engine.evaluate(req)

    assert result.status == ResultStatus.WARNING
    assert result.inconclusive_count == 2
    assert all(r.outcome == EvaluationOutcome.INCONCLUSIVE for r in result.limit_evaluations)
    assert any(i.issue_code == "ERR_NON_FINITE_OBSERVATION" for i in result.issues)


# -----------------------------------------------------------------------------
# Test 14: Context Filtering (Card & Point)
# -----------------------------------------------------------------------------
def test_context_filtering_card_and_point(
    evaluation_engine: VehicleEngineeringEvaluationEngine,
    sample_vehicle_profile: VehicleProfile,
) -> None:
    """Quantitative procedure requirements match observations adhering to card/point context."""
    proc_profile = ProcedureRequirementProfile(
        profile_id="PROC-CARD-01",
        document_id="CARD-SOP-01",
        document_version="1.0",
        vehicle_id="AC-TEST-X1",
        profile_version="1.0.0",
        requirements=(
            ProcedureRequirement(
                requirement_id="REQ-POINT-SPD",
                document_id="CARD-SOP-01",
                document_version="1.0",
                category=RequirementCategory.TOLERANCE_BAND,
                test_card_id="TC-01",
                test_point_id="TP-05",
                parameter_id="airspeed",
                target_value=180.0,
                tolerance=5.0,
                unit="kts",
                source_reference="Card TC-01",
                description="Airspeed criteria for point TP-05",
            ),
        ),
    )
    # Observation matching point TP-05
    obs_point_5 = EvaluationObservation(
        observation_id="OBS-TP05",
        parameter_id="airspeed",
        value=182.0,
        unit="kts",
        test_card_id="TC-01",
        test_point_id="TP-05",
        source_reference="flight_log",
    )
    req = VehicleEvaluationRequest(
        request_id="REQ-POINT-TEST",
        vehicle_profile=sample_vehicle_profile,
        procedure_profile=proc_profile,
        observations=(obs_point_5,),
        evaluate_limits=False,
    )
    result = evaluation_engine.evaluate(req)
    assert result.status == ResultStatus.SUCCESS
    rec = result.procedure_evaluations[0]
    assert rec.outcome == EvaluationOutcome.SATISFIED
    assert rec.observation_id == "OBS-TP05"


# -----------------------------------------------------------------------------
# Test 15: Immutability & Side-Effect Freedom
# -----------------------------------------------------------------------------
def test_immutability_and_result_properties(
    evaluation_engine: VehicleEngineeringEvaluationEngine,
    sample_vehicle_profile: VehicleProfile,
    sample_limit_profile: EngineeringLimitProfile,
    sample_procedure_profile: ProcedureRequirementProfile,
) -> None:
    """Caller observations and request objects must remain frozen and unchanged after evaluation."""
    obs_orig = EvaluationObservation(
        observation_id="OBS-IMMUTABLE",
        parameter_id="airspeed",
        value=200.0,
        unit="kts",
        source_reference="test",
    )
    obs_tuple = (obs_orig,)
    req = VehicleEvaluationRequest(
        request_id="REQ-IMMUTABLE",
        vehicle_profile=sample_vehicle_profile,
        limit_profile=sample_limit_profile,
        procedure_profile=sample_procedure_profile,
        observations=obs_tuple,
    )

    # Evaluate
    result = evaluation_engine.evaluate(req)

    # Observations tuple unchanged
    assert req.observations == obs_tuple
    assert obs_orig.value == 200.0

    # Summary properties
    assert result.total_evaluations == (
        len(result.limit_evaluations) + len(result.procedure_evaluations)
    )
    assert isinstance(result.limit_exceedance_count, int)
    assert isinstance(result.procedure_violation_count, int)
    assert isinstance(result.inconclusive_count, int)
    assert isinstance(result.has_exceedances, bool)
    assert isinstance(result.has_violations, bool)


# -----------------------------------------------------------------------------
# Test 16: Traceability, Lineage & Epistemic Boundaries
# -----------------------------------------------------------------------------
def test_traceability_provenance_and_metadata(
    evaluation_engine: VehicleEngineeringEvaluationEngine,
    sample_vehicle_profile: VehicleProfile,
    sample_limit_profile: EngineeringLimitProfile,
) -> None:
    """Evaluation result must record provenance with LIMIT epistemic level and full lineage."""
    obs = EvaluationObservation(
        observation_id="OBS-TRACE",
        parameter_id="airspeed",
        value=210.0,
        unit="kts",
        row_index=42,
        timestamp=datetime.datetime(2026, 10, 13, 12, 0, 0, tzinfo=datetime.UTC),
        source_reference="flight_test_042.csv",
    )
    req = VehicleEvaluationRequest(
        request_id="REQ-TRACE",
        vehicle_profile=sample_vehicle_profile,
        limit_profile=sample_limit_profile,
        observations=(obs,),
        evaluate_procedures=False,
    )
    result = evaluation_engine.evaluate(req)

    assert result.provenance is not None
    assert result.provenance.activity_name == "vehicle_engineering_evaluation"
    assert result.provenance.epistemic_level == EpistemicLevel.LIMIT
    assert result.provenance.parameters["vehicle_id"] == sample_vehicle_profile.vehicle_id

    rec = result.limit_evaluations[0]
    assert rec.row_index == 42
    assert rec.observation_id == "OBS-TRACE"
    assert rec.source_reference == "SYN-POH-SEC-2"
    assert rec.epistemic_level == EpistemicLevel.LIMIT
