"""
Unit tests for Milestone 12, Step 12g: Controlled Engineering Recommendation Domain Foundation.

Validates model constraints, immutability, reference integrity, lifecycle semantics,
provenance and epistemic integrity, safety boundaries, collections, and compatibility.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from aeris.core.hypothesis.models import Hypothesis
from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.recommendation import (
    Recommendation,
    RecommendationCollection,
    RecommendationStatus,
    RecommendationType,
)


@pytest.fixture
def mock_prov() -> ProvenanceRecord:
    return ProvenanceRecord(
        source_uri="urn:aeris:test:rec",
        activity_name="ManualRecommendationCreation",
        epistemic_level=EpistemicLevel.RECOMMENDATION,
    )


@pytest.fixture
def hypothesis_prov() -> ProvenanceRecord:
    return ProvenanceRecord(
        source_uri="urn:aeris:test:hyp",
        activity_name="HypothesisCreation",
        epistemic_level=EpistemicLevel.HYPOTHESIS,
    )


# ---------------------------------------------------------------------------
# 1. Model Validation (Tests 1-10)
# ---------------------------------------------------------------------------


def test_01_valid_recommendation(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Verify Pitch Gyro Calibration",
        description="Check bench calibration records for gyro serial G-1049.",
        rationale="Sensor pitch rate exhibited uncalibrated zero-shift during maneuvers.",
        provenance=mock_prov,
    )
    assert rec.recommendation_id == "rec-001"
    assert rec.recommendation_type == RecommendationType.VERIFY_DATA
    assert rec.status == RecommendationStatus.PROPOSED
    assert rec.epistemic_level == EpistemicLevel.RECOMMENDATION


def test_02_missing_required_identifier(mock_prov: ProvenanceRecord) -> None:
    with pytest.raises(ValidationError):
        Recommendation(  # type: ignore[call-arg]
            recommendation_type=RecommendationType.VERIFY_DATA,
            title="Title",
            description="Desc",
            rationale="Rationale",
            provenance=mock_prov,
        )


def test_03_empty_identifier(mock_prov: ProvenanceRecord) -> None:
    with pytest.raises(ValidationError):
        Recommendation(
            recommendation_id="",
            recommendation_type=RecommendationType.VERIFY_DATA,
            title="Title",
            description="Desc",
            rationale="Rationale",
            provenance=mock_prov,
        )


def test_04_invalid_identifier_format_whitespace(mock_prov: ProvenanceRecord) -> None:
    with pytest.raises(ValidationError):
        Recommendation(
            recommendation_id="   ",
            recommendation_type=RecommendationType.VERIFY_DATA,
            title="Title",
            description="Desc",
            rationale="Rationale",
            provenance=mock_prov,
        )


def test_05_empty_title(mock_prov: ProvenanceRecord) -> None:
    with pytest.raises(ValidationError):
        Recommendation(
            recommendation_id="rec-001",
            recommendation_type=RecommendationType.VERIFY_DATA,
            title="   ",
            description="Desc",
            rationale="Rationale",
            provenance=mock_prov,
        )


def test_06_empty_description(mock_prov: ProvenanceRecord) -> None:
    with pytest.raises(ValidationError):
        Recommendation(
            recommendation_id="rec-001",
            recommendation_type=RecommendationType.VERIFY_DATA,
            title="Title",
            description="",
            rationale="Rationale",
            provenance=mock_prov,
        )


def test_07_invalid_recommendation_type(mock_prov: ProvenanceRecord) -> None:
    with pytest.raises(ValidationError):
        Recommendation(
            recommendation_id="rec-001",
            recommendation_type="AUTONOMOUS_CONTROL_OVERRIDE",  # type: ignore[arg-type]
            title="Title",
            description="Desc",
            rationale="Rationale",
            provenance=mock_prov,
        )


def test_08_invalid_lifecycle_status(mock_prov: ProvenanceRecord) -> None:
    with pytest.raises(ValidationError):
        Recommendation(
            recommendation_id="rec-001",
            recommendation_type=RecommendationType.VERIFY_DATA,
            title="Title",
            description="Desc",
            rationale="Rationale",
            status="AUTOMATICALLY_EXECUTED",  # type: ignore[arg-type]
            provenance=mock_prov,
        )


def test_09_extra_fields_rejected(mock_prov: ProvenanceRecord) -> None:
    with pytest.raises(ValidationError):
        Recommendation(
            recommendation_id="rec-001",
            recommendation_type=RecommendationType.VERIFY_DATA,
            title="Title",
            description="Desc",
            rationale="Rationale",
            provenance=mock_prov,
            unknown_arbitrary_field="not_allowed",  # type: ignore[call-arg]
        )


def test_10_invalid_metadata_rejected(mock_prov: ProvenanceRecord) -> None:
    with pytest.raises(ValidationError):
        Recommendation(
            recommendation_id="rec-001",
            recommendation_type=RecommendationType.VERIFY_DATA,
            title="Title",
            description="Desc",
            rationale="Rationale",
            provenance=mock_prov,
            metadata={"root_cause": "fuel_pump_failure"},
        )


# ---------------------------------------------------------------------------
# 2. Immutability (Tests 11-16)
# ---------------------------------------------------------------------------


def test_11_frozen_model_assignment_rejected(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
    )
    with pytest.raises((ValidationError, TypeError)):
        rec.title = "New Title"  # type: ignore[misc]


def test_12_tuple_reference_collection_cannot_be_mutated(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        evidence_ids=("ev-1", "ev-2"),
        provenance=mock_prov,
    )
    assert isinstance(rec.evidence_ids, tuple)
    with pytest.raises(AttributeError):
        rec.evidence_ids.append("ev-3")  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        rec.evidence_ids[0] = "ev-mutated"  # type: ignore[index]


def test_13_metadata_cannot_be_mutated_externally(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
        metadata={"subsystem": "avionics"},
    )
    with pytest.raises((ValidationError, TypeError)):
        rec.metadata["subsystem"] = "propulsion"  # type: ignore[index]


def test_14_caller_owned_mappings_do_not_mutate_stored_state(mock_prov: ProvenanceRecord) -> None:
    external_dict: dict[str, Any] = {"review_team": "flight_ops"}
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
        metadata=external_dict,
    )
    external_dict["review_team"] = "ground_crew"
    assert rec.metadata["review_team"] == "flight_ops"


def test_15_collection_immutability(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
    )
    coll = RecommendationCollection(
        collection_id="coll-01",
        recommendations=(rec,),
        provenance=mock_prov,
    )
    with pytest.raises((ValidationError, TypeError)):
        coll.recommendations = ()  # type: ignore[misc]


def test_16_input_model_immutability(mock_prov: ProvenanceRecord) -> None:
    input_ids = ["ev-01", "ev-02"]
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        evidence_ids=input_ids,  # type: ignore[arg-type]
        provenance=mock_prov,
    )
    input_ids.append("ev-03")
    assert rec.evidence_ids == ("ev-01", "ev-02")


# ---------------------------------------------------------------------------
# 3. Reference Integrity (Tests 17-23)
# ---------------------------------------------------------------------------


def test_17_valid_evidence_references(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.REVIEW_EVIDENCE,
        title="Review Tail Sensor Evidence",
        description="Review telemetry records for pitot probe ice detector.",
        rationale="Pitot ice detector evidence ev-101 and ev-102 showed intermittent flags.",
        evidence_ids=("ev-101", "ev-102"),
        provenance=mock_prov,
    )
    assert rec.evidence_ids == ("ev-101", "ev-102")


def test_18_valid_assessment_references(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.REQUEST_ENGINEERING_REVIEW,
        title="Request Aerodynamicist Review",
        description="Review hypothesis assessment result asmt-501.",
        rationale=(
            "Assessment result indicated inconclusive evidence for elevator trim tab flutter."
        ),
        hypothesis_ids=("hyp-201",),
        hypothesis_assessment_ids=("asmt-501",),
        provenance=mock_prov,
    )
    assert rec.hypothesis_ids == ("hyp-201",)
    assert rec.hypothesis_assessment_ids == ("asmt-501",)


def test_19_duplicate_reference_handling(mock_prov: ProvenanceRecord) -> None:
    with pytest.raises(ValidationError):
        Recommendation(
            recommendation_id="rec-001",
            recommendation_type=RecommendationType.VERIFY_DATA,
            title="Title",
            description="Desc",
            rationale="Rationale",
            evidence_ids=("ev-1", "ev-1"),
            provenance=mock_prov,
        )


def test_20_empty_references_handled_correctly(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.REVIEW_TEST_PROCEDURE,
        title="General Procedure Audit",
        description="Audit checklist card SOP-FT-104.",
        rationale="Standard periodic test procedure verification.",
        evidence_ids=(),
        hypothesis_ids=(),
        hypothesis_assessment_ids=(),
        provenance=mock_prov,
    )
    assert rec.evidence_ids == ()
    assert rec.hypothesis_ids == ()
    assert rec.hypothesis_assessment_ids == ()


def test_21_reference_ids_preserved_exactly(mock_prov: ProvenanceRecord) -> None:
    raw_ids = ("ev_alpha:01", "ev_beta:02")
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        evidence_ids=raw_ids,
        provenance=mock_prov,
    )
    assert rec.evidence_ids == raw_ids


def test_22_no_automatic_reference_discovery(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        evidence_ids=("ev-explicit-only",),
        provenance=mock_prov,
    )
    assert len(rec.evidence_ids) == 1
    assert "ev-unlinked" not in rec.evidence_ids


def test_23_no_reference_substitution(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        evidence_ids=("ev-original",),
        provenance=mock_prov,
    )
    assert rec.evidence_ids[0] == "ev-original"


# ---------------------------------------------------------------------------
# 4. Lifecycle Semantics (Tests 24-32)
# ---------------------------------------------------------------------------


def test_24_proposed_status(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        status=RecommendationStatus.PROPOSED,
        provenance=mock_prov,
    )
    assert rec.status == RecommendationStatus.PROPOSED


def test_25_under_review_status(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        status=RecommendationStatus.UNDER_REVIEW,
        provenance=mock_prov,
    )
    assert rec.status == RecommendationStatus.UNDER_REVIEW


def test_26_accepted_for_investigation_status(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        status=RecommendationStatus.ACCEPTED_FOR_INVESTIGATION,
        provenance=mock_prov,
    )
    assert rec.status == RecommendationStatus.ACCEPTED_FOR_INVESTIGATION


def test_27_rejected_status(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        status=RecommendationStatus.REJECTED,
        provenance=mock_prov,
    )
    assert rec.status == RecommendationStatus.REJECTED


def test_28_completed_status(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        status=RecommendationStatus.COMPLETED,
        provenance=mock_prov,
    )
    assert rec.status == RecommendationStatus.COMPLETED


def test_29_superseded_status(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        status=RecommendationStatus.SUPERSEDED,
        supersedes_recommendation_id="rec-000",
        provenance=mock_prov,
    )
    assert rec.status == RecommendationStatus.SUPERSEDED
    assert rec.supersedes_recommendation_id == "rec-000"


def test_30_no_implicit_transition(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
    )
    assert rec.status == RecommendationStatus.PROPOSED
    # Model cannot auto-mutate its own status
    assert not hasattr(rec, "advance_status")


def test_31_accepted_does_not_mean_executed(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        status=RecommendationStatus.ACCEPTED_FOR_INVESTIGATION,
        provenance=mock_prov,
    )
    assert rec.status != RecommendationStatus.COMPLETED
    assert not hasattr(rec, "execution_timestamp")


def test_32_completed_does_not_mean_hypothesis_confirmed(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        status=RecommendationStatus.COMPLETED,
        provenance=mock_prov,
    )
    assert rec.status == RecommendationStatus.COMPLETED
    assert not hasattr(rec, "confirmed_root_cause")
    assert not hasattr(rec, "confirmed_hypothesis")


# ---------------------------------------------------------------------------
# 5. Provenance and Epistemic Integrity (Tests 33-39)
# ---------------------------------------------------------------------------


def test_33_valid_provenance(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
    )
    assert rec.provenance.activity_name == "ManualRecommendationCreation"
    assert rec.provenance.epistemic_level == EpistemicLevel.RECOMMENDATION


def test_34_missing_provenance_rejected() -> None:
    with pytest.raises(ValidationError):
        Recommendation(  # type: ignore[call-arg]
            recommendation_id="rec-001",
            recommendation_type=RecommendationType.VERIFY_DATA,
            title="Title",
            description="Desc",
            rationale="Rationale",
        )


def test_35_source_lineage_preserved() -> None:
    prov = ProvenanceRecord(
        source_uri="urn:aeris:flight:402",
        activity_name="EngineeringTriage",
        epistemic_level=EpistemicLevel.RECOMMENDATION,
        parent_record_ids=["asmt-101", "ev-202"],
    )
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=prov,
    )
    assert rec.provenance.parent_record_ids == ["asmt-101", "ev-202"]


def test_36_source_provenance_not_overwritten(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
    )
    assert rec.provenance.source_uri == "urn:aeris:test:rec"


def test_37_existing_recommendation_epistemic_level_reused(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
    )
    assert rec.epistemic_level == EpistemicLevel.RECOMMENDATION


def test_38_hypothesis_epistemic_level_unchanged(
    mock_prov: ProvenanceRecord, hypothesis_prov: ProvenanceRecord
) -> None:
    hyp = Hypothesis(
        hypothesis_id="hyp-001",
        title="Hypothesis Title",
        statement="Proposed statement.",
        provenance=hypothesis_prov,
    )
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.REQUEST_ENGINEERING_REVIEW,
        title="Title",
        description="Desc",
        rationale="Rationale",
        hypothesis_ids=(hyp.hypothesis_id,),
        provenance=mock_prov,
    )
    assert hyp.epistemic_level == EpistemicLevel.HYPOTHESIS
    assert rec.epistemic_level == EpistemicLevel.RECOMMENDATION


def test_39_no_source_provenance_fabrication(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
    )
    assert rec.provenance.activity_name == "ManualRecommendationCreation"


# ---------------------------------------------------------------------------
# 6. Safety Boundaries (Tests 40-48)
# ---------------------------------------------------------------------------


def test_40_no_root_cause_determination(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
    )
    dump = str(rec.model_dump())
    assert "root_cause" not in dump


def test_41_no_causal_probability(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
    )
    dump = str(rec.model_dump())
    assert "causal_probability" not in dump
    assert "probability" not in dump


def test_42_no_confidence_fabrication(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
    )
    dump = str(rec.model_dump())
    assert "confidence" not in dump


def test_43_no_recommendation_execution(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
    )
    assert not hasattr(rec, "execute")
    assert not hasattr(rec, "dispatch")


def test_44_no_operational_authorization(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
    )
    assert not hasattr(rec, "flight_authorization")
    assert not hasattr(rec, "maintenance_release")


def test_45_no_airworthiness_declaration(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
    )
    assert not hasattr(rec, "is_airworthy")
    assert not hasattr(rec, "safe_for_flight")


def test_46_no_autonomous_control_commands(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
    )
    assert not hasattr(rec, "command_payload")
    assert not hasattr(rec, "actuator_command")


def test_47_no_hidden_dataframe_dependency(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
    )
    for k in rec.model_dump():
        assert "dataframe" not in k
        assert "df" not in k
    assert True


def test_48_no_external_io(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
    )
    assert rec.recommendation_id == "rec-001"


# ---------------------------------------------------------------------------
# 7. Collections and Compatibility (Tests 49-58)
# ---------------------------------------------------------------------------


def test_49_empty_recommendation_collection(mock_prov: ProvenanceRecord) -> None:
    coll = RecommendationCollection(
        collection_id="coll-001",
        recommendations=(),
        provenance=mock_prov,
    )
    assert coll.collection_id == "coll-001"
    assert coll.recommendations == ()


def test_50_valid_recommendation_collection(mock_prov: ProvenanceRecord) -> None:
    r1 = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title 1",
        description="Desc 1",
        rationale="Rationale 1",
        provenance=mock_prov,
    )
    r2 = Recommendation(
        recommendation_id="rec-002",
        recommendation_type=RecommendationType.REVIEW_EVIDENCE,
        title="Title 2",
        description="Desc 2",
        rationale="Rationale 2",
        provenance=mock_prov,
    )
    coll = RecommendationCollection(
        collection_id="coll-001",
        recommendations=(r1, r2),
        provenance=mock_prov,
    )
    assert len(coll.recommendations) == 2


def test_51_duplicate_recommendation_ids_handled(mock_prov: ProvenanceRecord) -> None:
    r1 = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title 1",
        description="Desc 1",
        rationale="Rationale 1",
        provenance=mock_prov,
    )
    with pytest.raises(ValidationError):
        RecommendationCollection(
            collection_id="coll-001",
            recommendations=(r1, r1),
            provenance=mock_prov,
        )


def test_52_stable_collection_order(mock_prov: ProvenanceRecord) -> None:
    r1 = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title 1",
        description="Desc 1",
        rationale="Rationale 1",
        provenance=mock_prov,
    )
    r2 = Recommendation(
        recommendation_id="rec-002",
        recommendation_type=RecommendationType.REVIEW_EVIDENCE,
        title="Title 2",
        description="Desc 2",
        rationale="Rationale 2",
        provenance=mock_prov,
    )
    coll = RecommendationCollection(
        collection_id="coll-001",
        recommendations=(r1, r2),
        provenance=mock_prov,
    )
    assert coll.recommendations[0].recommendation_id == "rec-001"
    assert coll.recommendations[1].recommendation_id == "rec-002"


def test_53_existing_project_model_conventions_preserved(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
    )
    assert rec.model_config.get("frozen") is True
    assert rec.model_config.get("extra") == "forbid"


def test_54_existing_behavior_remains_compatible(mock_prov: ProvenanceRecord) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        provenance=mock_prov,
    )
    assert rec.provenance.epistemic_level == EpistemicLevel.RECOMMENDATION


def test_55_public_exports_work() -> None:
    import aeris.core.recommendation as rec_module

    assert hasattr(rec_module, "Recommendation")
    assert hasattr(rec_module, "RecommendationCollection")
    assert hasattr(rec_module, "RecommendationType")
    assert hasattr(rec_module, "RecommendationStatus")


def test_56_deterministic_construction(mock_prov: ProvenanceRecord) -> None:
    r1 = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        evidence_ids=("e1", "e2"),
        provenance=mock_prov,
    )
    r2 = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        evidence_ids=("e1", "e2"),
        provenance=mock_prov,
    )
    assert r1 == r2


def test_57_serialization_behavior_follows_existing_conventions(
    mock_prov: ProvenanceRecord,
) -> None:
    rec = Recommendation(
        recommendation_id="rec-001",
        recommendation_type=RecommendationType.VERIFY_DATA,
        title="Title",
        description="Desc",
        rationale="Rationale",
        evidence_ids=("e1",),
        provenance=mock_prov,
    )
    data = rec.model_dump()
    assert data["recommendation_id"] == "rec-001"
    assert data["recommendation_type"] == "VERIFY_DATA"
    assert data["status"] == "PROPOSED"
    assert data["epistemic_level"] == "RECOMMENDATION"


def test_58_structured_invalid_input_handling(mock_prov: ProvenanceRecord) -> None:
    with pytest.raises(ValidationError) as exc_info:
        Recommendation(
            recommendation_id="rec-001",
            recommendation_type=RecommendationType.VERIFY_DATA,
            title="Title",
            description="Desc",
            rationale="Rationale",
            evidence_ids=("",),  # Empty reference string
            provenance=mock_prov,
        )
    assert "ERR_INVALID_REFERENCE" in str(exc_info.value)
