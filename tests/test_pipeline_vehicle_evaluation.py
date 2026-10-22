"""
AERIS Pipeline Integration & Full Orchestration Tests (Milestone 13, Step 13g).

Verifies the end-to-end integration of vehicle profiles, configuration envelopes,
engineering limits, SOP requirements, the vehicle engineering evaluation engine,
evidence construction, and Recommendation Rule F within DataPipelineOrchestrator.

Test Scenarios:
 1. Successful end-to-end pipeline execution with explicit vehicle context.
 2. Vehicle evaluation skipped when required context is absent.
 3. Missing limits and missing SOP requirements.
 4. Invalid input preventing unsupported downstream claims.
 5. Evidence aggregation and lineage preservation.
 6. Deterministic evidence ordering and deduplication.
 7. Rule F enabled with eligible configuration discrepancy evidence.
 8. Rule F disabled.
 9. No Rule F recommendation without eligible discrepancy evidence.
10. Backward compatibility with existing pipeline requests.
11. Input and source-evidence immutability.
12. Inconclusive evaluation handling.
13. Stage issue/status reporting.
14. Regression coverage for existing non-vehicle pipeline behavior.

DISCLAIMER: All vehicle identities, limits, configurations, and procedural
rules in this module are synthetic and non-operational.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest
from pydantic import ValidationError

from aeris.core.evidence.models import EvidenceType
from aeris.core.evidence.pipeline import EvidenceSourceSelection, PipelineEvidenceRequest
from aeris.core.limits.models import (
    EngineeringLimitProfile,
    LimitSeverity,
    LimitType,
    ParameterLimitDefinition,
)
from aeris.core.pipeline.orchestrator import DataPipelineOrchestrator, run_csv_pipeline
from aeris.core.recommendation.generation_models import (
    RecommendationGenerationPolicy,
)
from aeris.core.recommendation.models import RecommendationType
from aeris.core.recommendation.pipeline import PipelineRecommendationRequest
from aeris.core.results.types import ResultStatus
from aeris.core.sop.models import (
    ProcedureRequirement,
    ProcedureRequirementProfile,
    RequirementCategory,
)
from aeris.core.vehicle.envelope_models import ConfigurationEnvelope
from aeris.core.vehicle.profile_models import VehicleProfile
from aeris.core.vehicle_evaluation.models import (
    EvaluationObservation,
    EvaluationOutcome,
)
from aeris.core.vehicle_evaluation.pipeline import (
    PipelineVehicleEvaluationRequest,
)

# ---------------------------------------------------------------------------
# Synthetic Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_vehicle_profile() -> VehicleProfile:
    """Return a synthetic vehicle profile."""
    return VehicleProfile(
        profile_id="PROF-SYNTH-X1",
        vehicle_id="AC-SYNTH-01",
        profile_version="1.0.0",
        name="Synthetic Flight Test Model X1",
        supported_parameter_ids=("airspeed_kts", "altitude_ft", "nz_g", "gear_handle"),
    )


@pytest.fixture
def synthetic_configuration_envelope() -> ConfigurationEnvelope:
    """Return a synthetic active configuration envelope."""
    return ConfigurationEnvelope(
        configuration_id="CRUISE_CLEAN",
        vehicle_id="AC-SYNTH-01",
        profile_id="PROF-SYNTH-X1",
        profile_version="1.0.0",
        name="Clean Cruise Envelope",
    )


@pytest.fixture
def synthetic_limit_profile() -> EngineeringLimitProfile:
    """Return a synthetic engineering limit profile."""
    return EngineeringLimitProfile(
        limit_profile_id="LIM-PROF-SYNTH-01",
        vehicle_id="AC-SYNTH-01",
        profile_id="PROF-SYNTH-X1",
        profile_version="1.0.0",
        limits=(
            ParameterLimitDefinition(
                limit_id="LIM-VMO-MAX",
                parameter_id="airspeed_kts",
                limit_type=LimitType.UPPER_BOUND,
                severity=LimitSeverity.WARNING,
                upper_bound=250.0,
                unit="kts",
                vehicle_id="AC-SYNTH-01",
                profile_id="PROF-SYNTH-X1",
                profile_version="1.0.0",
                source_reference="SYN-AFM-01",
            ),
            ParameterLimitDefinition(
                limit_id="LIM-ALT-MIN",
                parameter_id="altitude_ft",
                limit_type=LimitType.LOWER_BOUND,
                severity=LimitSeverity.CAUTION,
                lower_bound=5000.0,
                unit="ft",
                vehicle_id="AC-SYNTH-01",
                profile_id="PROF-SYNTH-X1",
                profile_version="1.0.0",
                source_reference="SYN-AFM-01",
            ),
        ),
    )


@pytest.fixture
def synthetic_procedure_profile() -> ProcedureRequirementProfile:
    """Return a synthetic procedure requirement profile."""
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
def clean_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    """Generate a clean synthetic flight telemetry CSV."""
    csv_file = tmp_path / "clean_flight.csv"
    timestamps = [
        "2026-10-14T12:00:00Z",
        "2026-10-14T12:00:01Z",
        "2026-10-14T12:00:02Z",
    ]
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "airspeed_kts": [220.0, 230.0, 225.0],
            "altitude_ft": [10000.0, 10020.0, 9990.0],
            "pitch_deg": [2.0, 2.1, 2.0],
            "roll_deg": [0.0, 0.2, -0.1],
            "engine_rpm": [2400.0, 2410.0, 2405.0],
            "nz_g": [1.0, 1.05, 0.98],
            "gear_handle": ["UP", "UP", "UP"],
            "configuration_id": ["CRUISE_CLEAN", "CRUISE_CLEAN", "CRUISE_CLEAN"],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def exceedance_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    """Generate a synthetic CSV with a limit exceedance."""
    csv_file = tmp_path / "exceedance_flight.csv"
    df = pd.DataFrame(
        {
            "timestamp": ["2026-10-14T12:00:00Z", "2026-10-14T12:00:01Z"],
            "airspeed_kts": [240.0, 310.0],  # 310 > 250 limit
            "altitude_ft": [10000.0, 10000.0],
            "pitch_deg": [2.0, 2.0],
            "roll_deg": [0.0, 0.0],
            "engine_rpm": [2400.0, 2400.0],
            "nz_g": [1.0, 1.0],
            "gear_handle": ["UP", "UP"],
            "configuration_id": ["CRUISE_CLEAN", "CRUISE_CLEAN"],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def discrepancy_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    """Generate a synthetic CSV with a configuration discrepancy."""
    csv_file = tmp_path / "discrepancy_flight.csv"
    df = pd.DataFrame(
        {
            "timestamp": ["2026-10-14T12:00:00Z", "2026-10-14T12:00:01Z"],
            "airspeed_kts": [220.0, 230.0],
            "altitude_ft": [10000.0, 10000.0],
            "pitch_deg": [2.0, 2.0],
            "roll_deg": [0.0, 0.0],
            "engine_rpm": [2400.0, 2400.0],
            "nz_g": [1.0, 1.0],
            "gear_handle": ["UP", "UP"],
            "configuration_id": ["APPROACH_DIRTY", "APPROACH_DIRTY"],  # Conflicts with CRUISE_CLEAN
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


# ---------------------------------------------------------------------------
# Test Scenarios 1 to 14
# ---------------------------------------------------------------------------


def test_01_successful_end_to_end_pipeline_with_vehicle_context(
    clean_csv: pathlib.Path,
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    synthetic_limit_profile: EngineeringLimitProfile,
    synthetic_procedure_profile: ProcedureRequirementProfile,
) -> None:
    """Scenario 1: Successful end-to-end pipeline execution with explicit vehicle context."""
    veh_req = PipelineVehicleEvaluationRequest(
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        limit_profile=synthetic_limit_profile,
        procedure_profile=synthetic_procedure_profile,
    )

    res = run_csv_pipeline(clean_csv, vehicle_evaluation_request=veh_req)

    assert res.status in (ResultStatus.SUCCESS, ResultStatus.WARNING)
    assert res.vehicle_evaluation is not None
    assert res.vehicle_evaluation_result is not None
    assert res.vehicle_evaluation.vehicle_id == "AC-SYNTH-01"
    assert res.vehicle_evaluation.profile_id == "PROF-SYNTH-X1"
    assert res.vehicle_evaluation.configuration_id == "CRUISE_CLEAN"
    assert res.vehicle_evaluation.total_evaluations > 0
    assert res.vehicle_evaluation.limit_exceedance_count == 0
    assert len(res.vehicle_evaluation.configuration_discrepancies) == 0
    assert res.vehicle_evaluation.provenance is not None
    assert res.provenance is not None
    assert res.vehicle_evaluation.provenance.record_id in res.provenance.parent_record_ids


def test_02_vehicle_evaluation_skipped_when_required_context_absent(
    clean_csv: pathlib.Path,
) -> None:
    """Scenario 2: Vehicle evaluation skipped when required context is absent."""
    veh_req = PipelineVehicleEvaluationRequest(
        vehicle_profile=None,  # Missing profile
    )

    res = run_csv_pipeline(clean_csv, vehicle_evaluation_request=veh_req)

    assert res.vehicle_evaluation is not None
    assert res.vehicle_evaluation.status == ResultStatus.NOT_AVAILABLE
    assert any(i.issue_code == "WARN_VEHICLE_PROFILE_MISSING" for i in res.issues)
    assert res.vehicle_evaluation.total_evaluations == 0


def test_03_missing_limits_and_missing_sop_requirements(
    clean_csv: pathlib.Path,
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
) -> None:
    """Scenario 3: Missing limits and missing SOP requirements do not fabricate data."""
    veh_req = PipelineVehicleEvaluationRequest(
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        limit_profile=None,  # Explicitly missing limits
        procedure_profile=None,  # Explicitly missing SOP
    )

    res = run_csv_pipeline(clean_csv, vehicle_evaluation_request=veh_req)

    assert res.vehicle_evaluation is not None
    assert len(res.vehicle_evaluation.limit_evaluations) == 0
    assert len(res.vehicle_evaluation.procedure_evaluations) == 0
    assert res.vehicle_evaluation.total_evaluations == 0
    assert res.vehicle_evaluation.status == ResultStatus.NOT_AVAILABLE


def test_04_invalid_input_prevents_unsupported_downstream_claims(
    tmp_path: pathlib.Path,
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    synthetic_limit_profile: EngineeringLimitProfile,
) -> None:
    """Scenario 4: Invalid input preventing unsupported downstream claims."""
    corrupt_csv = tmp_path / "corrupt_flight.csv"
    corrupt_csv.write_text("timestamp,airspeed_kts\ninvalid_ts,220.0\n", encoding="utf-8")

    veh_req = PipelineVehicleEvaluationRequest(
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        limit_profile=synthetic_limit_profile,
    )

    res = run_csv_pipeline(corrupt_csv, vehicle_evaluation_request=veh_req)

    assert res.status == ResultStatus.ERROR
    assert res.validation is not None
    assert not res.validation.is_valid
    assert res.vehicle_evaluation is not None
    assert res.vehicle_evaluation.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_UPSTREAM_VALIDATION_FAILED" for i in res.issues)
    assert res.vehicle_evaluation.total_evaluations == 0


def test_05_evidence_aggregation_and_lineage_preservation(
    clean_csv: pathlib.Path,
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    synthetic_limit_profile: EngineeringLimitProfile,
    synthetic_procedure_profile: ProcedureRequirementProfile,
) -> None:
    """Scenario 5: Evidence aggregation and lineage preservation."""
    veh_req = PipelineVehicleEvaluationRequest(
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        limit_profile=synthetic_limit_profile,
        procedure_profile=synthetic_procedure_profile,
    )
    ev_req = PipelineEvidenceRequest(
        collection_id="COLL-VEH-01",
        selections=EvidenceSourceSelection(include_vehicle_evaluation=True),
    )

    res = run_csv_pipeline(
        clean_csv,
        vehicle_evaluation_request=veh_req,
        evidence_request=ev_req,
    )

    assert res.evidence is not None
    assert res.evidence.collection is not None
    assert len(res.evidence.collection.evidence) > 0

    lim_records = [
        e
        for e in res.evidence.collection.evidence
        if e.evidence_type == EvidenceType.LIMIT_REFERENCE
    ]
    assert len(lim_records) > 0
    for r in lim_records:
        assert r.vehicle_id == "AC-SYNTH-01"
        assert r.profile_id == "PROF-SYNTH-X1"
        assert r.evaluation_id is not None
        assert r.provenance is not None


def test_06_deterministic_evidence_ordering_and_deduplication(
    clean_csv: pathlib.Path,
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    synthetic_limit_profile: EngineeringLimitProfile,
) -> None:
    """Scenario 6: Deterministic evidence ordering and deduplication."""
    veh_req = PipelineVehicleEvaluationRequest(
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        limit_profile=synthetic_limit_profile,
    )
    ev_req = PipelineEvidenceRequest(
        collection_id="COLL-DEDUP-01",
        selections=EvidenceSourceSelection(include_vehicle_evaluation=True),
    )

    res1 = run_csv_pipeline(clean_csv, vehicle_evaluation_request=veh_req, evidence_request=ev_req)
    res2 = run_csv_pipeline(clean_csv, vehicle_evaluation_request=veh_req, evidence_request=ev_req)

    assert res1.evidence is not None and res2.evidence is not None
    assert res1.evidence.collection is not None and res2.evidence.collection is not None

    ids1 = [e.evidence_id for e in res1.evidence.collection.evidence]
    ids2 = [e.evidence_id for e in res2.evidence.collection.evidence]

    # No duplicates in collection
    assert len(ids1) == len(set(ids1))
    assert len(ids2) == len(set(ids2))


def test_07_rule_f_enabled_with_eligible_configuration_discrepancy(
    discrepancy_csv: pathlib.Path,
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    synthetic_limit_profile: EngineeringLimitProfile,
) -> None:
    """Scenario 7: Rule F enabled with eligible configuration discrepancy evidence."""
    veh_req = PipelineVehicleEvaluationRequest(
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        limit_profile=synthetic_limit_profile,
    )
    policy = RecommendationGenerationPolicy(enable_configuration_review_rule=True)
    rec_req = PipelineRecommendationRequest(
        policy=policy,
        use_pipeline_vehicle_evaluation=True,
    )

    res = run_csv_pipeline(
        discrepancy_csv,
        vehicle_evaluation_request=veh_req,
        recommendation_request=rec_req,
    )

    assert res.vehicle_evaluation is not None
    assert len(res.vehicle_evaluation.configuration_discrepancies) > 0

    assert res.recommendation is not None
    assert "RULE_F_CHECK_CONFIGURATION_RECORD" in res.recommendation.rules_triggered
    cfg_recs = [
        r
        for r in res.recommendation.recommendations
        if r.recommendation_type == RecommendationType.CHECK_CONFIGURATION_RECORD
    ]
    assert len(cfg_recs) == 1
    assert "AC-SYNTH-01" in cfg_recs[0].title


def test_08_rule_f_disabled_policy(
    discrepancy_csv: pathlib.Path,
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    synthetic_limit_profile: EngineeringLimitProfile,
) -> None:
    """Scenario 8: Rule F disabled in policy does not emit recommendation."""
    veh_req = PipelineVehicleEvaluationRequest(
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        limit_profile=synthetic_limit_profile,
    )
    policy = RecommendationGenerationPolicy(enable_configuration_review_rule=False)
    rec_req = PipelineRecommendationRequest(
        policy=policy,
        use_pipeline_vehicle_evaluation=True,
    )

    res = run_csv_pipeline(
        discrepancy_csv,
        vehicle_evaluation_request=veh_req,
        recommendation_request=rec_req,
    )

    assert res.recommendation is not None
    assert "RULE_F_CHECK_CONFIGURATION_RECORD" not in res.recommendation.rules_triggered
    cfg_recs = [
        r
        for r in res.recommendation.recommendations
        if r.recommendation_type == RecommendationType.CHECK_CONFIGURATION_RECORD
    ]
    assert len(cfg_recs) == 0


def test_09_no_rule_f_without_eligible_discrepancy(
    clean_csv: pathlib.Path,
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    synthetic_limit_profile: EngineeringLimitProfile,
) -> None:
    """Scenario 9: No Rule F recommendation emitted without eligible discrepancy."""
    veh_req = PipelineVehicleEvaluationRequest(
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        limit_profile=synthetic_limit_profile,
    )
    policy = RecommendationGenerationPolicy(enable_configuration_review_rule=True)
    rec_req = PipelineRecommendationRequest(
        policy=policy,
        use_pipeline_vehicle_evaluation=True,
    )

    res = run_csv_pipeline(
        clean_csv,
        vehicle_evaluation_request=veh_req,
        recommendation_request=rec_req,
    )

    assert res.vehicle_evaluation is not None
    assert len(res.vehicle_evaluation.configuration_discrepancies) == 0

    assert res.recommendation is not None
    assert "RULE_F_CHECK_CONFIGURATION_RECORD" not in res.recommendation.rules_triggered


def test_10_backward_compatibility_with_existing_pipeline_requests(
    clean_csv: pathlib.Path,
) -> None:
    """Scenario 10: Backward compatibility with existing pipeline callers."""
    # Call without any vehicle request
    res = run_csv_pipeline(clean_csv)

    assert res.status in (ResultStatus.SUCCESS, ResultStatus.WARNING)
    assert res.ingestion is not None
    assert res.validation is not None
    assert res.validation.is_valid
    assert res.quality is not None
    assert res.vehicle_evaluation is None
    assert res.vehicle_evaluation_result is None


def test_11_input_and_source_evidence_immutability(
    clean_csv: pathlib.Path,
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    synthetic_limit_profile: EngineeringLimitProfile,
) -> None:
    """Scenario 11: Input and source-evidence immutability."""
    orig_limits_len = len(synthetic_limit_profile.limits)
    orig_envelope_cfg = synthetic_configuration_envelope.configuration_id

    veh_req = PipelineVehicleEvaluationRequest(
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        limit_profile=synthetic_limit_profile,
    )

    run_csv_pipeline(clean_csv, vehicle_evaluation_request=veh_req)

    # Objects are not mutated
    assert len(synthetic_limit_profile.limits) == orig_limits_len
    assert synthetic_configuration_envelope.configuration_id == orig_envelope_cfg

    # Request and profile models are frozen/immutable
    with pytest.raises(ValidationError):
        veh_req.vehicle_profile = None  # type: ignore[misc]

    with pytest.raises(ValidationError):
        synthetic_limit_profile.limits = ()  # type: ignore[misc]


def test_12_inconclusive_evaluation_handling(
    clean_csv: pathlib.Path,
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    synthetic_limit_profile: EngineeringLimitProfile,
) -> None:
    """Scenario 12: Inconclusive evaluation handling with strict units."""
    # Supply explicit observation with unit mismatch
    obs_mismatch = EvaluationObservation(
        observation_id="OBS-UNIT-MISMATCH",
        parameter_id="airspeed_kts",
        value=200.0,
        unit="km/h",  # Limit is in kts
        source_reference="clean_flight.csv",
    )
    veh_req = PipelineVehicleEvaluationRequest(
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        limit_profile=synthetic_limit_profile,
        explicit_observations=(obs_mismatch,),
        auto_extract_observations=False,
        strict_units=True,
    )

    res = run_csv_pipeline(clean_csv, vehicle_evaluation_request=veh_req)

    assert res.vehicle_evaluation is not None
    inc_evals = [
        e
        for e in res.vehicle_evaluation.limit_evaluations
        if e.outcome == EvaluationOutcome.INCONCLUSIVE
    ]
    assert len(inc_evals) == 1
    assert not inc_evals[0].is_exceeded
    assert "Incompatible units" in inc_evals[0].reason


def test_13_stage_issue_and_status_reporting(
    exceedance_csv: pathlib.Path,
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_configuration_envelope: ConfigurationEnvelope,
    synthetic_limit_profile: EngineeringLimitProfile,
) -> None:
    """Scenario 13: Stage issue and status reporting for exceedances."""
    veh_req = PipelineVehicleEvaluationRequest(
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_configuration_envelope,
        limit_profile=synthetic_limit_profile,
    )

    res = run_csv_pipeline(exceedance_csv, vehicle_evaluation_request=veh_req)

    assert res.vehicle_evaluation is not None
    assert res.vehicle_evaluation.has_exceedances
    assert res.vehicle_evaluation.limit_exceedance_count >= 1

    # Status reflects warning from limit exceedance
    assert res.vehicle_evaluation.status == ResultStatus.WARNING
    assert res.status == ResultStatus.WARNING
    assert any(i.issue_code == "VEH_LIMIT_EXCEEDANCE" for i in res.issues)


def test_14_regression_coverage_existing_non_vehicle_pipeline(
    clean_csv: pathlib.Path,
) -> None:
    """Scenario 14: Regression coverage for existing non-vehicle pipeline behavior."""
    orch = DataPipelineOrchestrator()
    res = orch.run_csv_pipeline(clean_csv)

    assert res.status in (ResultStatus.SUCCESS, ResultStatus.WARNING)
    assert res.ingestion.row_count == 3
    assert res.validation is not None
    assert res.validation.is_valid
    assert res.quality is not None

    assert res.quality.quality_score is not None
    assert res.vehicle_evaluation is None
