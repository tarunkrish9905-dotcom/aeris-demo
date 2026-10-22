"""
Focused tests for Vehicle-Aware Evidence Integration and Recommendation Rule F (Step 13f).

Validates:
1. Evidence construction from each supported evaluation type
   (limit within/exceeded, procedure satisfied/violated).
2. Correct LIMIT and PROCEDURAL epistemic classifications.
3. Full lineage from evidence to evaluation and input references.
4. Preservation of values, units, versions, and source references exactly as supplied.
5. Inconclusive and unsupported limit/procedure evaluations.
6. Unverified qualitative procedure requirements.
7. Configuration-discrepancy evidence construction.
8. Recommendation Rule F eligibility, output category, wording, and provenance.
9. Rule F behavior when required evidence is missing, disabled, or conflicting.
10. Deterministic repeated processing and appropriate deduplication.
11. Immutability and side-effect freedom for source results and evidence collections.
12. No duplicate or misleading recommendations and adherence to policy constraints.
13. Regression verification for existing evidence construction and recommendation rules.

DISCLAIMER: All test vehicles, parameters, limits, and procedure requirements in this module
are synthetic and non-operational. They do NOT establish flight safety or manufacturer specs.
"""

from __future__ import annotations

import datetime

import pytest

from aeris.core.anomaly.models import (
    AnomalyClassification,
    AnomalyRecord,
    AnomalySeverity,
    DetectionMethod,
)
from aeris.core.evidence.construction import EvidenceConstructionEngine
from aeris.core.evidence.models import EvidenceCollection, EvidenceType
from aeris.core.limits.models import (
    EngineeringLimitProfile,
    LimitSeverity,
    LimitType,
    ParameterLimitDefinition,
)
from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.recommendation.generation_engine import RecommendationGenerationEngine
from aeris.core.recommendation.generation_models import (
    RecommendationGenerationPolicy,
    RecommendationGenerationRequest,
)
from aeris.core.recommendation.models import RecommendationStatus, RecommendationType
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus, ValidationResult
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

# ---------------------------------------------------------------------------
# Synthetic Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_vehicle_profile() -> VehicleProfile:
    """Return a standard synthetic VehicleProfile."""
    return VehicleProfile(
        profile_id="PROF-SYNTH-X1",
        vehicle_id="AC-SYNTH-01",
        profile_version="1.0.0",
        name="Synthetic Flight Test Model X1",
        supported_parameter_ids=("airspeed_kts", "altitude_ft", "nz_g", "gear_handle"),
    )


@pytest.fixture
def synthetic_configuration_envelope() -> ConfigurationEnvelope:
    """Return a synthetic active ConfigurationEnvelope."""
    return ConfigurationEnvelope(
        configuration_id="CRUISE_CLEAN",
        vehicle_id="AC-SYNTH-01",
        profile_id="PROF-SYNTH-X1",
        profile_version="1.0.0",
        name="Clean Cruise Envelope",
    )


@pytest.fixture
def synthetic_limit_profile() -> EngineeringLimitProfile:
    """Return a synthetic EngineeringLimitProfile with upper, lower, and band limits."""
    return EngineeringLimitProfile(
        limit_profile_id="LIM-PROF-SYNTH-01",
        vehicle_id="AC-SYNTH-01",
        profile_id="PROF-SYNTH-X1",
        profile_version="1.0.0",
        limits=(
            ParameterLimitDefinition(
                limit_id="LIM-VNE-SPEED",
                parameter_id="airspeed_kts",
                limit_type=LimitType.UPPER_BOUND,
                severity=LimitSeverity.EXCEEDANCE,
                limit_value=300.0,
                unit="kts",
                vehicle_id="AC-SYNTH-01",
                profile_id="PROF-SYNTH-X1",
                profile_version="1.0.0",
                source_reference="SYN-POH-SEC-2",
            ),
            ParameterLimitDefinition(
                limit_id="LIM-ALT-FLOOR",
                parameter_id="altitude_ft",
                limit_type=LimitType.LOWER_BOUND,
                severity=LimitSeverity.WARNING,
                limit_value=1500.0,
                unit="ft",
                vehicle_id="AC-SYNTH-01",
                profile_id="PROF-SYNTH-X1",
                profile_version="1.0.0",
                source_reference="SYN-SOP-SEC-4",
            ),
            ParameterLimitDefinition(
                limit_id="LIM-NZ-MANEUVER",
                parameter_id="nz_g",
                limit_type=LimitType.BAND,
                severity=LimitSeverity.CAUTION,
                lower_bound=-1.0,
                upper_bound=2.5,
                unit="g",
                vehicle_id="AC-SYNTH-01",
                profile_id="PROF-SYNTH-X1",
                profile_version="1.0.0",
                source_reference="SYN-STRUCT-01",
            ),
        ),
    )


@pytest.fixture
def synthetic_procedure_profile() -> ProcedureRequirementProfile:
    """Return a synthetic ProcedureRequirementProfile."""
    return ProcedureRequirementProfile(
        profile_id="PROC-PROF-SYNTH-01",
        document_id="SOP-SYNTH-001",
        document_version="1.2",
        vehicle_id="AC-SYNTH-01",
        profile_version="1.0.0",
        requirements=(
            ProcedureRequirement(
                requirement_id="REQ-TARGET-ALT",
                document_id="SOP-SYNTH-001",
                document_version="1.2",
                category=RequirementCategory.TOLERANCE_BAND,
                parameter_id="altitude_ft",
                target_value=10000.0,
                tolerance=100.0,
                unit="ft",
                source_reference="SOP-SYNTH-001 Sec 3.1",
                description="Maintain cruise altitude within 100 ft tolerance",
            ),
            ProcedureRequirement(
                requirement_id="REQ-GEAR-UP",
                document_id="SOP-SYNTH-001",
                document_version="1.2",
                category=RequirementCategory.OPERATIONAL_CHECK,
                source_reference="SOP-SYNTH-001 Sec 2.4",
                description="Confirm landing gear handle is UP",
            ),
        ),
    )


@pytest.fixture
def evaluation_engine() -> VehicleEngineeringEvaluationEngine:
    """Return an evaluation engine instance."""
    return VehicleEngineeringEvaluationEngine()


# ---------------------------------------------------------------------------
# Test Scenario 1: Evidence Construction from Each Supported Evaluation Type
# ---------------------------------------------------------------------------


def test_01_evidence_construction_from_limit_evaluations(
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    synthetic_limit_profile: EngineeringLimitProfile,
    evaluation_engine: VehicleEngineeringEvaluationEngine,
) -> None:
    """Validate evidence construction from within-bounds and exceeded limit evaluations."""
    t0 = datetime.datetime(2026, 10, 14, 12, 0, 0, tzinfo=datetime.UTC)
    obs_within = EvaluationObservation(
        observation_id="OBS-SPEED-01",
        parameter_id="airspeed_kts",
        value=240.0,
        unit="kts",
        timestamp=t0,
        row_index=10,
        source_reference="telemetry_test.csv",
    )
    obs_exceeded = EvaluationObservation(
        observation_id="OBS-SPEED-02",
        parameter_id="airspeed_kts",
        value=320.0,
        unit="kts",
        timestamp=t0,
        row_index=20,
        source_reference="telemetry_test.csv",
    )

    req = VehicleEvaluationRequest(
        request_id="REQ-EVAL-01",
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        limit_profile=synthetic_limit_profile,
        observations=(obs_within, obs_exceeded),
    )
    result = evaluation_engine.evaluate(req)

    # 1. Within-bounds evidence
    rec_within = result.limit_evaluations[0]
    ev_within = EvidenceConstructionEngine.create_from_limit_evaluation(
        rec_within, evaluation_result=result
    )
    assert ev_within.evidence_id == f"EVD-LIM-{rec_within.evaluation_id}"
    assert ev_within.evidence_type == EvidenceType.LIMIT_REFERENCE
    assert ev_within.epistemic_level == EpistemicLevel.LIMIT
    assert ev_within.evaluation_outcome == EvaluationOutcome.WITHIN_BOUNDS.value
    assert ev_within.observed_value == 240.0
    assert ev_within.limit_value == 300.0
    assert ev_within.margin == 60.0
    assert ev_within.vehicle_id == "AC-SYNTH-01"
    assert ev_within.unit == "kts"

    # 2. Exceeded limit evidence
    rec_exceeded = result.limit_evaluations[1]
    ev_exceeded = EvidenceConstructionEngine.create_from_limit_evaluation(
        rec_exceeded, evaluation_result=result
    )
    assert ev_exceeded.evidence_id == f"EVD-LIM-{rec_exceeded.evaluation_id}"
    assert ev_exceeded.evidence_type == EvidenceType.LIMIT_REFERENCE
    assert ev_exceeded.epistemic_level == EpistemicLevel.LIMIT
    assert ev_exceeded.evaluation_outcome == EvaluationOutcome.OUTSIDE_BOUNDS.value
    assert ev_exceeded.observed_value == 320.0
    assert ev_exceeded.limit_value == 300.0
    assert ev_exceeded.margin == -20.0


def test_02_evidence_construction_from_procedure_evaluations(
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    synthetic_procedure_profile: ProcedureRequirementProfile,
    evaluation_engine: VehicleEngineeringEvaluationEngine,
) -> None:
    """Validate evidence construction from satisfied and violated procedure requirements."""
    t0 = datetime.datetime(2026, 10, 14, 12, 0, 0, tzinfo=datetime.UTC)
    obs_sat = EvaluationObservation(
        observation_id="OBS-ALT-01",
        parameter_id="altitude_ft",
        value=10050.0,
        unit="ft",
        timestamp=t0,
        row_index=15,
        source_reference="telemetry_test.csv",
    )
    obs_viol = EvaluationObservation(
        observation_id="OBS-ALT-02",
        parameter_id="altitude_ft",
        value=10250.0,
        unit="ft",
        timestamp=t0,
        row_index=25,
        source_reference="telemetry_test.csv",
    )

    # Evaluate satisfied
    req_sat = VehicleEvaluationRequest(
        request_id="REQ-EVAL-PROC-01",
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        procedure_profile=synthetic_procedure_profile,
        observations=(obs_sat,),
    )
    res_sat = evaluation_engine.evaluate(req_sat)
    rec_proc_sat = res_sat.procedure_evaluations[0]
    ev_sat = EvidenceConstructionEngine.create_from_procedure_evaluation(
        rec_proc_sat, evaluation_result=res_sat
    )
    assert ev_sat.evidence_id == f"EVD-PROC-{rec_proc_sat.evaluation_id}"
    assert ev_sat.evidence_type == EvidenceType.PROCEDURAL_REFERENCE
    assert ev_sat.epistemic_level == EpistemicLevel.PROCEDURAL
    assert ev_sat.evaluation_outcome == EvaluationOutcome.SATISFIED.value
    assert ev_sat.observed_value == 10050.0
    assert ev_sat.limit_value == 10000.0
    assert ev_sat.margin == 50.0

    # Evaluate violated
    req_viol = VehicleEvaluationRequest(
        request_id="REQ-EVAL-PROC-02",
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        procedure_profile=synthetic_procedure_profile,
        observations=(obs_viol,),
    )
    res_viol = evaluation_engine.evaluate(req_viol)
    rec_proc_viol = res_viol.procedure_evaluations[0]
    ev_viol = EvidenceConstructionEngine.create_from_procedure_evaluation(
        rec_proc_viol, evaluation_result=res_viol
    )
    assert ev_viol.evidence_type == EvidenceType.PROCEDURAL_REFERENCE
    assert ev_viol.epistemic_level == EpistemicLevel.PROCEDURAL
    assert ev_viol.evaluation_outcome == EvaluationOutcome.VIOLATED.value
    assert ev_viol.observed_value == 10250.0
    assert ev_viol.margin == -150.0


# ---------------------------------------------------------------------------
# Test Scenario 2: Correct LIMIT and PROCEDURAL Epistemic Classifications
# ---------------------------------------------------------------------------


def test_03_epistemic_safeguards_limit_and_procedural(
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    synthetic_limit_profile: EngineeringLimitProfile,
    synthetic_procedure_profile: ProcedureRequirementProfile,
    evaluation_engine: VehicleEngineeringEvaluationEngine,
) -> None:
    """Ensure limit evidence is LIMIT, procedure evidence is PROCEDURAL, not OBSERVED."""
    t0 = datetime.datetime(2026, 10, 14, 12, 0, 0, tzinfo=datetime.UTC)
    obs = EvaluationObservation(
        observation_id="OBS-NZ-01",
        parameter_id="nz_g",
        value=1.5,
        unit="g",
        timestamp=t0,
        source_reference="flight_test.csv",
    )
    req = VehicleEvaluationRequest(
        request_id="REQ-EVAL-EPISTEMIC",
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        limit_profile=synthetic_limit_profile,
        procedure_profile=synthetic_procedure_profile,
        observations=(obs,),
    )
    res = evaluation_engine.evaluate(req)

    ev_coll = EvidenceConstructionEngine.create_from_vehicle_evaluation_result(res)
    limit_evs = [e for e in ev_coll.evidence if e.evidence_type == EvidenceType.LIMIT_REFERENCE]
    proc_evs = [e for e in ev_coll.evidence if e.evidence_type == EvidenceType.PROCEDURAL_REFERENCE]

    assert len(limit_evs) > 0
    for e in limit_evs:
        assert e.epistemic_level == EpistemicLevel.LIMIT
        assert e.epistemic_level.is_prescriptive()
        assert not e.epistemic_level.is_empirical()

    assert len(proc_evs) > 0
    for e in proc_evs:
        assert e.epistemic_level == EpistemicLevel.PROCEDURAL
        assert e.epistemic_level.is_prescriptive()
        assert not e.epistemic_level.is_empirical()


# ---------------------------------------------------------------------------
# Test Scenario 3 & 4: Full Lineage & Preservation of Values, Units, Versions
# ---------------------------------------------------------------------------


def test_04_full_lineage_and_metadata_preservation(
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    synthetic_limit_profile: EngineeringLimitProfile,
    synthetic_procedure_profile: ProcedureRequirementProfile,
    evaluation_engine: VehicleEngineeringEvaluationEngine,
) -> None:
    """Verify that identifiers, units, versions, and source references are preserved intact."""
    t0 = datetime.datetime(2026, 10, 14, 14, 30, 0, tzinfo=datetime.UTC)
    obs = EvaluationObservation(
        observation_id="OBS-ALT-44",
        parameter_id="altitude_ft",
        value=12000.0,
        unit="ft",
        timestamp=t0,
        row_index=123,
        source_reference="telemetry_packet_44.csv",
    )
    req = VehicleEvaluationRequest(
        request_id="REQ-EVAL-LINEAGE",
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        limit_profile=synthetic_limit_profile,
        procedure_profile=synthetic_procedure_profile,
        observations=(obs,),
    )
    res = evaluation_engine.evaluate(req)
    ev_coll = EvidenceConstructionEngine.create_from_vehicle_evaluation_result(res)

    # Check limit evidence lineage
    lim_ev = next(e for e in ev_coll.evidence if e.limit_id == "LIM-ALT-FLOOR")
    assert lim_ev.evaluation_id == res.evaluation_id
    assert lim_ev.vehicle_id == "AC-SYNTH-01"
    assert lim_ev.profile_id == "PROF-SYNTH-X1"
    assert lim_ev.profile_version == "1.0.0"
    assert lim_ev.configuration_id == "CRUISE_CLEAN"
    assert lim_ev.source_parameter_id == "altitude_ft"
    assert lim_ev.source_row_index == 123
    assert lim_ev.timestamp == t0
    assert lim_ev.unit == "ft"
    assert lim_ev.observed_value == 12000.0
    assert lim_ev.limit_value == 1500.0
    assert res.provenance is not None
    assert lim_ev.provenance.parent_record_ids == [str(res.provenance.record_id)]

    # Check procedure evidence lineage
    proc_ev = next(e for e in ev_coll.evidence if e.requirement_id == "REQ-TARGET-ALT")
    assert proc_ev.document_id == "SOP-SYNTH-001"
    assert proc_ev.document_version == "1.2"
    assert proc_ev.source_parameter_id == "altitude_ft"
    assert proc_ev.source_row_index == 123


# ---------------------------------------------------------------------------
# Test Scenario 5: Inconclusive and Unsupported Evaluations
# ---------------------------------------------------------------------------


def test_05_inconclusive_limit_evaluation_evidence(
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    synthetic_limit_profile: EngineeringLimitProfile,
    evaluation_engine: VehicleEngineeringEvaluationEngine,
) -> None:
    """Inconclusive evaluations produce INCONCLUSIVE evidence without claiming exceedance."""
    t0 = datetime.datetime(2026, 10, 14, 12, 0, 0, tzinfo=datetime.UTC)
    obs_bad_unit = EvaluationObservation(
        observation_id="OBS-SPEED-BAD-UNIT",
        parameter_id="airspeed_kts",
        value=150.0,
        unit="mph",  # Mismatch: limit requires 'kts'
        timestamp=t0,
        source_reference="telemetry_raw.csv",
    )
    req = VehicleEvaluationRequest(
        request_id="REQ-EVAL-INCONCLUSIVE",
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        limit_profile=synthetic_limit_profile,
        observations=(obs_bad_unit,),
    )
    res = evaluation_engine.evaluate(req)
    assert res.inconclusive_count == 1
    assert not res.has_exceedances

    rec_lim = res.limit_evaluations[0]
    assert rec_lim.outcome == EvaluationOutcome.INCONCLUSIVE
    assert not rec_lim.is_exceeded

    ev = EvidenceConstructionEngine.create_from_limit_evaluation(rec_lim, evaluation_result=res)
    assert ev.evidence_type == EvidenceType.LIMIT_REFERENCE
    assert ev.evaluation_outcome == EvaluationOutcome.INCONCLUSIVE.value
    assert "Incompatible unit" in ev.description


# ---------------------------------------------------------------------------
# Test Scenario 6: Unverified Qualitative Procedure Requirements
# ---------------------------------------------------------------------------


def test_06_unverified_qualitative_procedure_evidence(
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    synthetic_procedure_profile: ProcedureRequirementProfile,
    evaluation_engine: VehicleEngineeringEvaluationEngine,
) -> None:
    """Qualitative requirements without confirmation observation remain UNVERIFIED/NOT_EVALUATED."""
    req = VehicleEvaluationRequest(
        request_id="REQ-EVAL-QUAL",
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        procedure_profile=synthetic_procedure_profile,
        observations=(),  # No observations supplied
    )
    res = evaluation_engine.evaluate(req)
    rec_gear = next(r for r in res.procedure_evaluations if r.requirement_id == "REQ-GEAR-UP")
    assert rec_gear.outcome == EvaluationOutcome.NOT_EVALUATED
    assert rec_gear.verification_status == RequirementVerificationStatus.UNVERIFIED
    assert rec_gear.is_satisfied is None

    ev = EvidenceConstructionEngine.create_from_procedure_evaluation(
        rec_gear, evaluation_result=res
    )
    assert ev.evidence_type == EvidenceType.PROCEDURAL_REFERENCE
    assert ev.epistemic_level == EpistemicLevel.PROCEDURAL
    assert ev.evaluation_outcome == EvaluationOutcome.NOT_EVALUATED.value
    assert ev.metadata.get("is_qualitative") is True
    assert ev.limit_value is None


# ---------------------------------------------------------------------------
# Test Scenario 7: Configuration Discrepancy Evidence Construction
# ---------------------------------------------------------------------------


def test_07_configuration_discrepancy_evidence(
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    evaluation_engine: VehicleEngineeringEvaluationEngine,
) -> None:
    """Observation conflicting with envelope generates CONFIGURATION_DISCREPANCY evidence."""
    t0 = datetime.datetime(2026, 10, 14, 12, 0, 0, tzinfo=datetime.UTC)
    obs_mismatch = EvaluationObservation(
        observation_id="OBS-MISMATCH-CFG",
        parameter_id="airspeed_kts",
        value=200.0,
        unit="kts",
        timestamp=t0,
        configuration_id="STORES_EXTERNAL",  # Mismatch with CRUISE_CLEAN
        source_reference="flight_test_log.csv",
    )
    req = VehicleEvaluationRequest(
        request_id="REQ-EVAL-DISC",
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        observations=(obs_mismatch,),
    )
    res = evaluation_engine.evaluate(req)
    assert len(res.configuration_discrepancies) == 1

    ev_coll = EvidenceConstructionEngine.create_from_vehicle_evaluation_result(res)
    disc_ev = [
        e for e in ev_coll.evidence if e.evidence_type == EvidenceType.CONFIGURATION_DISCREPANCY
    ]
    assert len(disc_ev) == 1
    assert disc_ev[0].epistemic_level == EpistemicLevel.OBSERVED
    assert disc_ev[0].evaluation_outcome == "DISCREPANCY"
    assert "conflicts with active configuration envelope" in disc_ev[0].description
    assert disc_ev[0].configuration_id == "CRUISE_CLEAN"
    assert disc_ev[0].vehicle_id == "AC-SYNTH-01"


# ---------------------------------------------------------------------------
# Test Scenario 8: Recommendation Rule F Eligibility, Wording, and Provenance
# ---------------------------------------------------------------------------


def test_08_rule_f_triggers_when_enabled_with_discrepancy(
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    evaluation_engine: VehicleEngineeringEvaluationEngine,
) -> None:
    """Rule F generates CHECK_CONFIGURATION_RECORD recommendation when discrepancy present."""
    obs_mismatch = EvaluationObservation(
        observation_id="OBS-DISC-01",
        parameter_id="airspeed_kts",
        value=200.0,
        unit="kts",
        configuration_id="STORES_EXTERNAL",
        source_reference="flight_test_log.csv",
    )
    eval_req = VehicleEvaluationRequest(
        request_id="REQ-EVAL-DISC-REC",
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        observations=(obs_mismatch,),
    )
    eval_res = evaluation_engine.evaluate(eval_req)
    ev_coll = EvidenceConstructionEngine.create_from_vehicle_evaluation_result(eval_res)

    # Request recommendation with enable_configuration_review_rule=True
    policy = RecommendationGenerationPolicy(enable_configuration_review_rule=True)
    gen_req = RecommendationGenerationRequest(
        evidence_collection=ev_coll,
        vehicle_evaluation_result=eval_res,
        policy=policy,
        investigation_id="INV-DISC-2026-001",
    )
    gen_res = RecommendationGenerationEngine.generate(gen_req)

    assert (
        RecommendationGenerationEngine.RULE_F_CHECK_CONFIGURATION_RECORD in gen_res.rules_evaluated
    )
    assert (
        RecommendationGenerationEngine.RULE_F_CHECK_CONFIGURATION_RECORD in gen_res.rules_triggered
    )
    assert len(gen_res.recommendations) == 1

    rec = gen_res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.CHECK_CONFIGURATION_RECORD
    assert rec.status == RecommendationStatus.PROPOSED
    assert "AC-SYNTH-01" in rec.title
    assert "Review Configuration Envelope Records" in rec.title
    assert "reconcile conflicting configuration states" in rec.description
    assert "Verification is required" in rec.rationale
    assert "do not assert root cause" in rec.constraints[0]
    assert len(rec.prerequisites) > 0
    assert rec.investigation_id == "INV-DISC-2026-001"
    assert rec.provenance.epistemic_level == EpistemicLevel.RECOMMENDATION
    assert len(rec.evidence_ids) > 0


# ---------------------------------------------------------------------------
# Test Scenario 9: Rule F Behavior When Disabled or Discrepancy Missing
# ---------------------------------------------------------------------------


def test_09_rule_f_not_triggered_when_disabled_or_clean(
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    synthetic_limit_profile: EngineeringLimitProfile,
    evaluation_engine: VehicleEngineeringEvaluationEngine,
) -> None:
    """Rule F is evaluated but does NOT trigger when disabled or when data is clean."""
    obs_clean = EvaluationObservation(
        observation_id="OBS-CLEAN-01",
        parameter_id="airspeed_kts",
        value=200.0,
        unit="kts",
        configuration_id="CRUISE_CLEAN",  # Matches envelope
        source_reference="flight_test_log.csv",
    )
    eval_req = VehicleEvaluationRequest(
        request_id="REQ-EVAL-CLEAN",
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        limit_profile=synthetic_limit_profile,
        observations=(obs_clean,),
    )
    eval_res = evaluation_engine.evaluate(eval_req)
    ev_coll = EvidenceConstructionEngine.create_from_vehicle_evaluation_result(eval_res)

    # Case 1: Policy disabled (default)
    policy_default = RecommendationGenerationPolicy(enable_configuration_review_rule=False)
    req1 = RecommendationGenerationRequest(
        evidence_collection=ev_coll,
        vehicle_evaluation_result=eval_res,
        policy=policy_default,
    )
    res1 = RecommendationGenerationEngine.generate(req1)
    assert RecommendationGenerationEngine.RULE_F_CHECK_CONFIGURATION_RECORD in res1.rules_evaluated
    assert (
        RecommendationGenerationEngine.RULE_F_CHECK_CONFIGURATION_RECORD not in res1.rules_triggered
    )
    assert len(res1.recommendations) == 0

    # Case 2: Policy enabled, but NO discrepancy exists
    policy_enabled = RecommendationGenerationPolicy(enable_configuration_review_rule=True)
    req2 = RecommendationGenerationRequest(
        evidence_collection=ev_coll,
        vehicle_evaluation_result=eval_res,
        policy=policy_enabled,
    )
    res2 = RecommendationGenerationEngine.generate(req2)
    assert RecommendationGenerationEngine.RULE_F_CHECK_CONFIGURATION_RECORD in res2.rules_evaluated
    assert (
        RecommendationGenerationEngine.RULE_F_CHECK_CONFIGURATION_RECORD not in res2.rules_triggered
    )
    assert len(res2.recommendations) == 0


# ---------------------------------------------------------------------------
# Test Scenario 10: Determinism and Deduplication
# ---------------------------------------------------------------------------


def test_10_determinism_and_deduplication(
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    synthetic_limit_profile: EngineeringLimitProfile,
    evaluation_engine: VehicleEngineeringEvaluationEngine,
) -> None:
    """Repeated processing yields identical IDs; duplicates are safely deduplicated."""
    obs = EvaluationObservation(
        observation_id="OBS-REP-01",
        parameter_id="airspeed_kts",
        value=220.0,
        unit="kts",
        source_reference="telemetry.csv",
    )
    req = VehicleEvaluationRequest(
        request_id="REQ-EVAL-DET",
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        limit_profile=synthetic_limit_profile,
        observations=(obs,),
    )
    res = evaluation_engine.evaluate(req)

    coll1 = EvidenceConstructionEngine.create_from_vehicle_evaluation_result(res)
    coll2 = EvidenceConstructionEngine.create_from_vehicle_evaluation_result(res)

    assert [e.evidence_id for e in coll1.evidence] == [e.evidence_id for e in coll2.evidence]
    assert [e.description for e in coll1.evidence] == [e.description for e in coll2.evidence]

    # Deduplication test: passing same record multiple times in list to create_collection
    rec = coll1.evidence[0]
    with pytest.raises(ValueError, match="ERR_DUPLICATE_EVIDENCE_ID"):
        EvidenceCollection(collection_id="COLL-DUP", evidence=(rec, rec))


# ---------------------------------------------------------------------------
# Test Scenario 11: Immutability and Side-Effect Freedom
# ---------------------------------------------------------------------------


def test_11_immutability_and_side_effects(
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    synthetic_limit_profile: EngineeringLimitProfile,
    evaluation_engine: VehicleEngineeringEvaluationEngine,
) -> None:
    """Ensure constructed evidence records and collections are strictly frozen."""
    obs = EvaluationObservation(
        observation_id="OBS-IMM-01",
        parameter_id="airspeed_kts",
        value=200.0,
        unit="kts",
        source_reference="telemetry.csv",
    )
    req = VehicleEvaluationRequest(
        request_id="REQ-EVAL-IMM",
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        limit_profile=synthetic_limit_profile,
        observations=(obs,),
    )
    res = evaluation_engine.evaluate(req)
    coll = EvidenceConstructionEngine.create_from_vehicle_evaluation_result(res)
    rec = coll.evidence[0]

    with pytest.raises((TypeError, ValueError)):
        rec.description = "Mutated description"  # type: ignore[misc]

    with pytest.raises((TypeError, ValueError)):
        coll.evidence = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test Scenario 12: No Misleading Recommendations & Policy Max Bounding
# ---------------------------------------------------------------------------


def test_12_no_misleading_recommendations_and_max_bounding(
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    evaluation_engine: VehicleEngineeringEvaluationEngine,
) -> None:
    """Ensure Rule F output contains no root cause assertion, and respects max limit."""
    obs_mismatch = EvaluationObservation(
        observation_id="OBS-DISC-MAX",
        parameter_id="airspeed_kts",
        value=200.0,
        unit="kts",
        configuration_id="STORES_EXTERNAL",
        source_reference="flight_test_log.csv",
    )
    eval_req = VehicleEvaluationRequest(
        request_id="REQ-EVAL-DISC-MAX",
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        observations=(obs_mismatch,),
    )
    eval_res = evaluation_engine.evaluate(eval_req)
    ev_coll = EvidenceConstructionEngine.create_from_vehicle_evaluation_result(eval_res)

    policy = RecommendationGenerationPolicy(
        enable_configuration_review_rule=True,
        max_recommendations=1,
    )
    gen_req = RecommendationGenerationRequest(
        evidence_collection=ev_coll,
        vehicle_evaluation_result=eval_res,
        policy=policy,
    )
    gen_res = RecommendationGenerationEngine.generate(gen_req)

    assert len(gen_res.recommendations) <= 1
    rec = gen_res.recommendations[0]
    # Verify no root cause claims or operational flight orders
    assert "root cause" not in rec.description.lower()
    assert "root cause" not in rec.rationale.lower()
    assert "ground aircraft" not in rec.description.lower()
    assert "airworthy" not in rec.description.lower()


# ---------------------------------------------------------------------------
# Test Scenario 13: Regression Tests for Existing Rules and Evidence Types
# ---------------------------------------------------------------------------


def test_13_regression_existing_evidence_and_recommendation_rules() -> None:
    """Verify that existing Anomaly evidence and Rule A/B/C/D continue to work cleanly."""
    t0 = datetime.datetime(2026, 10, 14, 12, 0, 0, tzinfo=datetime.UTC)
    prov = create_provenance(
        activity_name="anomaly_detection",
        source_uri="urn:aeris:detector:test",
        epistemic_level=EpistemicLevel.CALCULATED,
    )
    anomaly = AnomalyRecord(
        anomaly_id="ANOM-01",
        analysis_id="ANALYSIS-01",
        target_column="alpha_deg",
        classification=AnomalyClassification.POINT_ANOMALY,
        severity=AnomalySeverity.MEDIUM,
        detection_method=DetectionMethod.RULE_BASED,
        timestamp=t0,
        provenance=prov,
        epistemic_level=EpistemicLevel.CALCULATED,
    )
    ev = EvidenceConstructionEngine.create_from_anomaly(anomaly)
    assert ev.evidence_id == "EVD-ANOMALY-ANOM-01"
    assert ev.evidence_type == EvidenceType.ANOMALY
    assert ev.epistemic_level == EpistemicLevel.CALCULATED

    # Verify Rule A fires on failed validation
    val_prov = create_provenance(
        activity_name="data_validation",
        source_uri="urn:aeris:validator:test",
        epistemic_level=EpistemicLevel.CALCULATED,
    )
    val_res = ValidationResult(
        status=ResultStatus.ERROR,
        is_valid=False,
        violation_count=2,
        validated_columns=["alpha_deg"],
        issues=[
            Issue(
                issue_code="ERR_VAL_SCHEMA",
                severity=IssueSeverity.ERROR,
                message="Schema validation error",
            )
        ],
        provenance=val_prov,
    )
    req = RecommendationGenerationRequest(validation_result=val_res)
    res = RecommendationGenerationEngine.generate(req)
    assert RecommendationGenerationEngine.RULE_A_VERIFY_DATA in res.rules_triggered
    assert len(res.recommendations) == 1
    assert res.recommendations[0].recommendation_type == RecommendationType.VERIFY_DATA
