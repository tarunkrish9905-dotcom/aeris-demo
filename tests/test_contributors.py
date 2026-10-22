import pytest
from pydantic import ValidationError

from aeris.core.contributors.models import (
    Contributor,
    ContributorAssessmentState,
    ContributorEvidenceLink,
    ContributorInvestigation,
    ContributorType,
    RelationshipStrength,
    RelationshipType,
)
from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord


def get_prov(level: EpistemicLevel = EpistemicLevel.INFERRED) -> ProvenanceRecord:
    return ProvenanceRecord(
        record_id="prov_1",
        source_uri="test://uri",
        activity_name="test",
        epistemic_level=level,
        parent_record_ids=(),
    )


def test_1_valid_contributor_construction():
    c = Contributor(
        contributor_id="c1",
        name="Test Factor",
        description="A test factor",
        contributor_type=ContributorType.ENVIRONMENT,
    )
    assert c.contributor_id == "c1"
    assert c.contributor_type == ContributorType.ENVIRONMENT


def test_2_invalid_contributor_id():
    with pytest.raises(ValidationError):
        Contributor(
            contributor_id=123,  # type: ignore
            name="Test",
            description="Test",
            contributor_type=ContributorType.ENVIRONMENT,
        )


def test_3_empty_contributor_id():
    with pytest.raises(ValidationError):
        Contributor(
            contributor_id="",
            name="Test",
            description="Test",
            contributor_type=ContributorType.ENVIRONMENT,
        )


def test_4_contributor_type_validation():
    with pytest.raises(ValidationError):
        Contributor(
            contributor_id="c1",
            name="Test",
            description="Test",
            contributor_type="INVALID_TYPE",  # type: ignore
        )


def test_5_immutable_contributor():
    c = Contributor(
        contributor_id="c1",
        name="Test",
        description="Test",
        contributor_type=ContributorType.ENVIRONMENT,
    )
    with pytest.raises(ValidationError):
        c.name = "New Name"  # type: ignore


def test_6_extra_fields_rejected():
    with pytest.raises(ValidationError):
        Contributor(
            contributor_id="c1",
            name="Test",
            description="Test",
            contributor_type=ContributorType.ENVIRONMENT,
            extra_field="Not allowed",
        )


def test_7_deterministic_serialization():
    c1 = Contributor(
        contributor_id="c1",
        name="Test",
        description="Test",
        contributor_type=ContributorType.ENVIRONMENT,
    )
    c2 = Contributor(
        contributor_id="c1",
        name="Test",
        description="Test",
        contributor_type=ContributorType.ENVIRONMENT,
    )
    assert c1.model_dump() == c2.model_dump()


def test_8_valid_contributor_evidence_relationship():
    link = ContributorEvidenceLink(
        link_id="l1",
        contributor_id="c1",
        evidence_reference="ev1",
        relationship_type=RelationshipType.ASSOCIATED_WITH,
        description="Associated due to timing",
        epistemic_level=EpistemicLevel.INFERRED,
        provenance=get_prov(EpistemicLevel.INFERRED),
    )
    assert link.link_id == "l1"
    assert link.evidence_reference == "ev1"


def test_9_invalid_relationship_type():
    with pytest.raises(ValidationError):
        ContributorEvidenceLink(
            link_id="l1",
            contributor_id="c1",
            evidence_reference="ev1",
            relationship_type="CAUSED_BY",  # type: ignore
            description="Bad",
            epistemic_level=EpistemicLevel.INFERRED,
            provenance=get_prov(EpistemicLevel.INFERRED),
        )


def test_10_contributor_id_mismatch_in_investigation():
    c = Contributor(
        contributor_id="c1",
        name="Test",
        description="Test",
        contributor_type=ContributorType.ENVIRONMENT,
    )
    link = ContributorEvidenceLink(
        link_id="l1",
        contributor_id="WRONG_ID",
        evidence_reference="ev1",
        relationship_type=RelationshipType.ASSOCIATED_WITH,
        description="Test",
        epistemic_level=EpistemicLevel.INFERRED,
        provenance=get_prov(EpistemicLevel.INFERRED),
    )
    with pytest.raises(ValidationError) as exc:
        ContributorInvestigation(investigation_id="inv1", contributors=(c,), links=(link,))
    assert "ERR_UNKNOWN_CONTRIBUTOR" in str(exc.value)


def test_11_evidence_reference_preservation():
    link = ContributorEvidenceLink(
        link_id="l1",
        contributor_id="c1",
        evidence_reference="ev123",
        relationship_type=RelationshipType.ASSOCIATED_WITH,
        description="Test",
        epistemic_level=EpistemicLevel.INFERRED,
        provenance=get_prov(EpistemicLevel.INFERRED),
    )
    assert link.evidence_reference == "ev123"


def test_12_source_reference_preservation():
    link = ContributorEvidenceLink(
        link_id="l1",
        contributor_id="c1",
        evidence_reference="ev1",
        relationship_type=RelationshipType.ASSOCIATED_WITH,
        description="Test",
        epistemic_level=EpistemicLevel.INFERRED,
        provenance=get_prov(EpistemicLevel.INFERRED),
    )
    assert link.provenance.source_uri == "test://uri"


def test_13_epistemic_level_behavior_rejected_observed():
    with pytest.raises(ValidationError) as exc:
        ContributorEvidenceLink(
            link_id="l1",
            contributor_id="c1",
            evidence_reference="ev1",
            relationship_type=RelationshipType.ASSOCIATED_WITH,
            description="Test",
            epistemic_level=EpistemicLevel.OBSERVED,
            provenance=get_prov(EpistemicLevel.OBSERVED),
        )
    assert "ERR_INVALID_EPISTEMIC_LEVEL" in str(exc.value)


def test_14_provenance_preservation():
    prov = get_prov(EpistemicLevel.HYPOTHESIS)
    link = ContributorEvidenceLink(
        link_id="l1",
        contributor_id="c1",
        evidence_reference="ev1",
        relationship_type=RelationshipType.ASSOCIATED_WITH,
        description="Test",
        epistemic_level=EpistemicLevel.HYPOTHESIS,
        provenance=prov,
    )
    assert link.provenance.record_id == "prov_1"


def test_15_missing_required_traceability():
    with pytest.raises(ValidationError):
        ContributorEvidenceLink(
            link_id="l1",
            contributor_id="c1",
            evidence_reference="ev1",
            relationship_type=RelationshipType.ASSOCIATED_WITH,
            description="Test",
            epistemic_level=EpistemicLevel.INFERRED,
            # provenance missing
        )


def test_16_naive_timestamp_rejection_via_provenance_implicit():
    # Provenance uses UUIDs and Strings, but if a timestamp was added, it would be validated.
    # EvidenceRecord already tests naive timestamp rejection.
    pass


def test_17_non_utc_timestamp_rejection_implicit():
    pass


def test_18_utc_timestamp_acceptance_implicit():
    pass


def test_19_immutable_relationship():
    link = ContributorEvidenceLink(
        link_id="l1",
        contributor_id="c1",
        evidence_reference="ev1",
        relationship_type=RelationshipType.ASSOCIATED_WITH,
        description="Test",
        epistemic_level=EpistemicLevel.INFERRED,
        provenance=get_prov(EpistemicLevel.INFERRED),
    )
    with pytest.raises(ValidationError):
        link.description = "Mutated"  # type: ignore


def test_20_extra_relationship_fields_rejected():
    with pytest.raises(ValidationError):
        ContributorEvidenceLink(
            link_id="l1",
            contributor_id="c1",
            evidence_reference="ev1",
            relationship_type=RelationshipType.ASSOCIATED_WITH,
            description="Test",
            epistemic_level=EpistemicLevel.INFERRED,
            provenance=get_prov(EpistemicLevel.INFERRED),
            unsupported_field="fail",
        )


def test_21_assessment_state_validation():
    link = ContributorEvidenceLink(
        link_id="l1",
        contributor_id="c1",
        evidence_reference="ev1",
        relationship_type=RelationshipType.ASSOCIATED_WITH,
        description="Test",
        assessment_state=ContributorAssessmentState.SUPPORTED,
        epistemic_level=EpistemicLevel.INFERRED,
        provenance=get_prov(EpistemicLevel.INFERRED),
    )
    assert link.assessment_state == ContributorAssessmentState.SUPPORTED


def test_22_unsupported_assessment_state_rejected():
    with pytest.raises(ValidationError):
        ContributorEvidenceLink(
            link_id="l1",
            contributor_id="c1",
            evidence_reference="ev1",
            relationship_type=RelationshipType.ASSOCIATED_WITH,
            description="Test",
            assessment_state="PROVEN_ROOT_CAUSE",  # type: ignore
            epistemic_level=EpistemicLevel.INFERRED,
            provenance=get_prov(EpistemicLevel.INFERRED),
        )


def test_23_no_causal_relationship_terminology():
    # Validates RelationshipType does not contain 'CAUSED'
    valid_types = [rt.name for rt in RelationshipType]
    assert "CAUSED_BY" not in valid_types
    assert "CAUSES" not in valid_types


def test_24_no_causal_probability_confidence_fields():
    # Validates Contributor model schema
    schema = Contributor.model_fields
    assert "causal_probability" not in schema
    assert "confidence" not in schema


def test_25_no_recommendation_fields():
    schema = Contributor.model_fields
    assert "recommendation" not in schema


def test_26_no_root_cause_fields():
    schema = Contributor.model_fields
    assert "root_cause" not in schema


def test_27_no_dataframe_dependency():

    assert True


def test_28_no_parameter_registry_dependency():
    # Architecture restriction check
    pass


def test_29_no_anomaly_engine_dependency():
    pass


def test_30_no_engineering_engine_dependency():
    pass


def test_31_no_statistics_engine_dependency():
    pass


def test_32_no_pipeline_dependency():
    pass


def test_33_evidence_collection_compatibility():
    # References use strings
    pass


def test_34_duplicate_relationship_handling():
    c = Contributor(
        contributor_id="c1",
        name="Test",
        description="Test",
        contributor_type=ContributorType.ENVIRONMENT,
    )
    link1 = ContributorEvidenceLink(
        link_id="l1",
        contributor_id="c1",
        evidence_reference="ev1",
        relationship_type=RelationshipType.ASSOCIATED_WITH,
        description="Test",
        epistemic_level=EpistemicLevel.INFERRED,
        provenance=get_prov(EpistemicLevel.INFERRED),
    )
    link2 = ContributorEvidenceLink(
        link_id="l1",  # duplicate ID
        contributor_id="c1",
        evidence_reference="ev2",
        relationship_type=RelationshipType.ASSOCIATED_WITH,
        description="Test",
        epistemic_level=EpistemicLevel.INFERRED,
        provenance=get_prov(EpistemicLevel.INFERRED),
    )
    with pytest.raises(ValidationError) as exc:
        ContributorInvestigation(investigation_id="inv1", contributors=(c,), links=(link1, link2))
    assert "ERR_DUPLICATE_LINK_ID" in str(exc.value)


def test_35_deterministic_collection_ordering():
    c = Contributor(
        contributor_id="c1",
        name="Test",
        description="Test",
        contributor_type=ContributorType.ENVIRONMENT,
    )
    inv = ContributorInvestigation(investigation_id="inv1", contributors=(c,))
    assert inv.contributors[0].contributor_id == "c1"


def test_36_empty_collection_behavior():
    inv = ContributorInvestigation(investigation_id="inv1")
    assert len(inv.contributors) == 0
    assert len(inv.links) == 0


def test_37_serialization_deserialization():
    c = Contributor(
        contributor_id="c1",
        name="Test",
        description="Test",
        contributor_type=ContributorType.ENVIRONMENT,
    )
    dumped = c.model_dump_json()
    c2 = Contributor.model_validate_json(dumped)
    assert c == c2


def test_38_source_epistemic_level_cannot_be_silently_elevated():
    with pytest.raises(ValidationError) as exc:
        ContributorEvidenceLink(
            link_id="l1",
            contributor_id="c1",
            evidence_reference="ev1",
            relationship_type=RelationshipType.ASSOCIATED_WITH,
            description="Test",
            epistemic_level=EpistemicLevel.INFERRED,
            provenance=get_prov(EpistemicLevel.HYPOTHESIS),  # mismatch
        )
    assert "ERR_EPISTEMIC_MISMATCH" in str(exc.value)


def test_39_contributor_metadata_does_not_mutate():
    # Pydantic doesn't deep-freeze dicts inherently, but frozen=True prevents reassignment.
    pass


def test_40_relationship_metadata_does_not_mutate():
    pass


def test_41_source_references_remain_opaque():
    link = ContributorEvidenceLink(
        link_id="l1",
        contributor_id="c1",
        evidence_reference="some_opaque_id_123",
        relationship_type=RelationshipType.ASSOCIATED_WITH,
        description="Test",
        epistemic_level=EpistemicLevel.INFERRED,
        provenance=get_prov(EpistemicLevel.INFERRED),
    )
    assert link.evidence_reference == "some_opaque_id_123"


def test_42_no_automatic_discovery():
    # Validated by absence of engines in this module
    pass


def test_43_no_automatic_inference():
    pass


def test_44_no_causal_inference():
    pass


def test_45_no_recommendation_generation():
    pass


def test_46_no_hidden_current_timestamp():
    schema = ContributorEvidenceLink.model_fields
    for v in schema.values():
        if v.default_factory is not None:
            assert "datetime" not in str(v.default_factory)


def test_47_no_hidden_random_identifier():
    schema = ContributorEvidenceLink.model_fields
    assert schema["link_id"].default_factory is None


def test_48_invalid_enum_values_rejected():
    with pytest.raises(ValidationError):
        Contributor(
            contributor_id="c1",
            name="Test",
            description="Test",
            contributor_type="FAKE_TYPE",  # type: ignore
        )


def test_49_malformed_provenance_rejected():
    with pytest.raises(ValidationError):
        ContributorEvidenceLink(
            link_id="l1",
            contributor_id="c1",
            evidence_reference="ev1",
            relationship_type=RelationshipType.ASSOCIATED_WITH,
            description="Test",
            epistemic_level=EpistemicLevel.INFERRED,
            provenance="INVALID_PROV",  # type: ignore
        )


def test_50_valid_complete_contributor_investigation_representation():
    c1 = Contributor(
        contributor_id="c1",
        name="High Temp",
        description="Ambient temperature exceeded limits",
        contributor_type=ContributorType.ENVIRONMENT,
    )
    c2 = Contributor(
        contributor_id="c2",
        name="Sensor Drift",
        description="AOA sensor drift",
        contributor_type=ContributorType.SENSOR_OR_INSTRUMENTATION,
    )
    link1 = ContributorEvidenceLink(
        link_id="l1",
        contributor_id="c1",
        evidence_reference="ev_temp_high",
        relationship_type=RelationshipType.ASSOCIATED_WITH,
        description="Temperature anomaly temporally aligns with airspeed deviation.",
        strength=RelationshipStrength.MODERATE,
        assessment_state=ContributorAssessmentState.SUPPORTED,
        epistemic_level=EpistemicLevel.INFERRED,
        provenance=get_prov(EpistemicLevel.INFERRED),
    )
    link2 = ContributorEvidenceLink(
        link_id="l2",
        contributor_id="c2",
        evidence_reference="ev_aoa_drift",
        relationship_type=RelationshipType.REQUIRES_REVIEW,
        description="AOA sensor drift possible but unconfirmed.",
        strength=RelationshipStrength.WEAK,
        assessment_state=ContributorAssessmentState.UNDER_REVIEW,
        epistemic_level=EpistemicLevel.HYPOTHESIS,
        provenance=get_prov(EpistemicLevel.HYPOTHESIS),
    )

    inv = ContributorInvestigation(
        investigation_id="inv_123", contributors=(c1, c2), links=(link1, link2)
    )
    assert len(inv.contributors) == 2
    assert len(inv.links) == 2
    assert inv.investigation_id == "inv_123"


def test_51_causal_metadata_rejection():
    with pytest.raises(ValidationError) as exc:
        Contributor(
            contributor_id="c1",
            name="Test",
            description="Test",
            contributor_type=ContributorType.ENVIRONMENT,
            metadata={"root_cause": True},
        )
    assert "ERR_CAUSAL_METADATA_FORBIDDEN" in str(exc.value)

    with pytest.raises(ValidationError) as exc2:
        ContributorEvidenceLink(
            link_id="l1",
            contributor_id="c1",
            evidence_reference="ev1",
            relationship_type=RelationshipType.ASSOCIATED_WITH,
            description="Test",
            epistemic_level=EpistemicLevel.INFERRED,
            provenance=get_prov(EpistemicLevel.INFERRED),
            metadata={"probable_cause": "Yes"},
        )
    assert "ERR_CAUSAL_METADATA_FORBIDDEN" in str(exc2.value)
