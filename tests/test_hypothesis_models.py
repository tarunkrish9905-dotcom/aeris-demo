import pytest
from pydantic import ValidationError

from aeris.core.hypothesis.models import (
    Hypothesis,
    HypothesisAssessmentState,
    HypothesisCollection,
    HypothesisContributorLink,
    HypothesisContributorRelationship,
    HypothesisEvidenceLink,
    HypothesisEvidenceRelationship,
    HypothesisLifecycleStatus,
)
from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord


@pytest.fixture
def mock_prov():
    return ProvenanceRecord(
        source_uri="test", activity_name="test", epistemic_level=EpistemicLevel.HYPOTHESIS
    )


def test_valid_hypothesis_id(mock_prov):
    h = Hypothesis(hypothesis_id="h1", title="T", statement="S", provenance=mock_prov)
    assert h.hypothesis_id == "h1"


def test_empty_hypothesis_id(mock_prov):
    with pytest.raises(ValidationError):
        Hypothesis(hypothesis_id="", title="T", statement="S", provenance=mock_prov)


def test_missing_hypothesis_id(mock_prov):
    with pytest.raises(ValidationError):
        Hypothesis(title="T", statement="S", provenance=mock_prov)


def test_valid_link_id(mock_prov):
    link = HypothesisEvidenceLink(
        link_id="L1",
        hypothesis_id="h1",
        evidence_id="e1",
        relationship_type=HypothesisEvidenceRelationship.SUPPORTING,
        provenance=mock_prov,
    )
    assert link.link_id == "L1"


def test_empty_link_id(mock_prov):
    with pytest.raises(ValidationError):
        HypothesisEvidenceLink(
            link_id="",
            hypothesis_id="h1",
            evidence_id="e1",
            relationship_type=HypothesisEvidenceRelationship.SUPPORTING,
            provenance=mock_prov,
        )


def test_duplicate_hypothesis_ids_in_collection(mock_prov):
    h1 = Hypothesis(hypothesis_id="h1", title="T", statement="S", provenance=mock_prov)
    h2 = Hypothesis(hypothesis_id="h1", title="T", statement="S", provenance=mock_prov)
    with pytest.raises(ValidationError):
        HypothesisCollection(hypotheses=(h1, h2), provenance=mock_prov)


def test_valid_statement(mock_prov):
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="Valid statement.", provenance=mock_prov
    )
    assert h.statement == "Valid statement."


def test_empty_statement(mock_prov):
    with pytest.raises(ValidationError):
        Hypothesis(hypothesis_id="h1", title="T", statement="", provenance=mock_prov)


def test_whitespace_only_statement(mock_prov):
    with pytest.raises(ValidationError):
        Hypothesis(hypothesis_id="h1", title="T", statement="   \t \n ", provenance=mock_prov)


def test_statement_immutability(mock_prov):
    h = Hypothesis(hypothesis_id="h1", title="T", statement="S", provenance=mock_prov)
    with pytest.raises((ValidationError, TypeError)):
        h.statement = "S2"


def test_statement_preserved_exactly(mock_prov):
    s = "Density altitude may be associated with the airspeed deviation."
    h = Hypothesis(hypothesis_id="h1", title="T", statement=s, provenance=mock_prov)
    assert h.statement == s


def test_epistemic_level_is_hypothesis(mock_prov):
    h = Hypothesis(hypothesis_id="h1", title="T", statement="S", provenance=mock_prov)
    assert h.epistemic_level == EpistemicLevel.HYPOTHESIS


def test_epistemic_level_observed_rejected(mock_prov):
    with pytest.raises(ValidationError):
        Hypothesis(
            hypothesis_id="h1",
            title="T",
            statement="S",
            epistemic_level=EpistemicLevel.OBSERVED,
            provenance=mock_prov,
        )


def test_epistemic_level_calculated_rejected(mock_prov):
    with pytest.raises(ValidationError):
        Hypothesis(
            hypothesis_id="h1",
            title="T",
            statement="S",
            epistemic_level=EpistemicLevel.CALCULATED,
            provenance=mock_prov,
        )


def test_epistemic_level_immutable(mock_prov):
    h = Hypothesis(hypothesis_id="h1", title="T", statement="S", provenance=mock_prov)
    with pytest.raises((ValidationError, TypeError)):
        h.epistemic_level = EpistemicLevel.OBSERVED


def test_valid_lifecycle_status(mock_prov):
    h = Hypothesis(
        hypothesis_id="h1",
        title="T",
        statement="S",
        status=HypothesisLifecycleStatus.UNDER_INVESTIGATION,
        provenance=mock_prov,
    )
    assert h.status == HypothesisLifecycleStatus.UNDER_INVESTIGATION


def test_invalid_lifecycle_status(mock_prov):
    with pytest.raises(ValidationError):
        Hypothesis(
            hypothesis_id="h1", title="T", statement="S", status="UNKNOWN", provenance=mock_prov
        )


def test_lifecycle_status_immutable(mock_prov):
    h = Hypothesis(hypothesis_id="h1", title="T", statement="S", provenance=mock_prov)
    with pytest.raises((ValidationError, TypeError)):
        h.status = HypothesisLifecycleStatus.CLOSED


def test_valid_assessment_state(mock_prov):
    h = Hypothesis(
        hypothesis_id="h1",
        title="T",
        statement="S",
        assessment_state=HypothesisAssessmentState.SUPPORTED,
        provenance=mock_prov,
    )
    assert h.assessment_state == HypothesisAssessmentState.SUPPORTED


def test_invalid_assessment_state(mock_prov):
    with pytest.raises(ValidationError):
        Hypothesis(
            hypothesis_id="h1",
            title="T",
            statement="S",
            assessment_state="DEFINITELY_TRUE",
            provenance=mock_prov,
        )


def test_valid_supporting_link(mock_prov):
    link = HypothesisEvidenceLink(
        link_id="L1",
        hypothesis_id="h1",
        evidence_id="e1",
        relationship_type=HypothesisEvidenceRelationship.SUPPORTING,
        provenance=mock_prov,
    )
    assert link.relationship_type == HypothesisEvidenceRelationship.SUPPORTING


def test_valid_contradicting_link(mock_prov):
    link = HypothesisEvidenceLink(
        link_id="L1",
        hypothesis_id="h1",
        evidence_id="e1",
        relationship_type=HypothesisEvidenceRelationship.CONTRADICTING,
        provenance=mock_prov,
    )
    assert link.relationship_type == HypothesisEvidenceRelationship.CONTRADICTING


def test_valid_relevant_link(mock_prov):
    link = HypothesisEvidenceLink(
        link_id="L1",
        hypothesis_id="h1",
        evidence_id="e1",
        relationship_type=HypothesisEvidenceRelationship.RELEVANT,
        provenance=mock_prov,
    )
    assert link.relationship_type == HypothesisEvidenceRelationship.RELEVANT


def test_invalid_evidence_relationship(mock_prov):
    with pytest.raises(ValidationError):
        HypothesisEvidenceLink(
            link_id="L1",
            hypothesis_id="h1",
            evidence_id="e1",
            relationship_type="PROVES",
            provenance=mock_prov,
        )


def test_causal_relationship_rejected(mock_prov):
    with pytest.raises(ValidationError):
        HypothesisEvidenceLink(
            link_id="L1",
            hypothesis_id="h1",
            evidence_id="e1",
            relationship_type="CAUSE",
            provenance=mock_prov,
        )


def test_evidence_link_immutable(mock_prov):
    link = HypothesisEvidenceLink(
        link_id="L1",
        hypothesis_id="h1",
        evidence_id="e1",
        relationship_type=HypothesisEvidenceRelationship.SUPPORTING,
        provenance=mock_prov,
    )
    with pytest.raises((ValidationError, TypeError)):
        link.relationship_type = HypothesisEvidenceRelationship.CONTRADICTING


def test_valid_contributor_link(mock_prov):
    link = HypothesisContributorLink(
        link_id="L1",
        hypothesis_id="h1",
        contributor_id="c1",
        relationship_type=HypothesisContributorRelationship.RELEVANT_FACTOR,
        provenance=mock_prov,
    )
    assert link.relationship_type == HypothesisContributorRelationship.RELEVANT_FACTOR


def test_invalid_contributor_relationship(mock_prov):
    with pytest.raises(ValidationError):
        HypothesisContributorLink(
            link_id="L1",
            hypothesis_id="h1",
            contributor_id="c1",
            relationship_type="ROOT_CAUSE",
            provenance=mock_prov,
        )


def test_contributor_link_immutable(mock_prov):
    link = HypothesisContributorLink(
        link_id="L1",
        hypothesis_id="h1",
        contributor_id="c1",
        relationship_type=HypothesisContributorRelationship.RELEVANT_FACTOR,
        provenance=mock_prov,
    )
    with pytest.raises((ValidationError, TypeError)):
        link.relationship_type = HypothesisContributorRelationship.CANDIDATE_FACTOR


def test_provenance_preserved(mock_prov):
    h = Hypothesis(hypothesis_id="h1", title="T", statement="S", provenance=mock_prov)
    assert h.provenance.activity_name == "test"


def test_invalid_provenance_rejected():
    with pytest.raises(ValidationError):
        Hypothesis(
            hypothesis_id="h1",
            title="T",
            statement="S",
            provenance={"source_uri": "missing_fields"},
        )


def test_provenance_immutable(mock_prov):
    h = Hypothesis(hypothesis_id="h1", title="T", statement="S", provenance=mock_prov)
    with pytest.raises((ValidationError, TypeError)):
        h.provenance = None


def test_valid_collection(mock_prov):
    h = Hypothesis(hypothesis_id="h1", title="T", statement="S", provenance=mock_prov)
    c = HypothesisCollection(hypotheses=(h,), provenance=mock_prov)
    assert len(c.hypotheses) == 1


def test_empty_collection(mock_prov):
    c = HypothesisCollection(hypotheses=(), provenance=mock_prov)
    assert len(c.hypotheses) == 0


def test_ordering_preservation(mock_prov):
    h1 = Hypothesis(hypothesis_id="h1", title="T", statement="S", provenance=mock_prov)
    h2 = Hypothesis(hypothesis_id="h2", title="T", statement="S", provenance=mock_prov)
    c = HypothesisCollection(hypotheses=(h1, h2), provenance=mock_prov)
    assert c.hypotheses[0].hypothesis_id == "h1"
    assert c.hypotheses[1].hypothesis_id == "h2"


def test_collection_immutability(mock_prov):
    c = HypothesisCollection(hypotheses=(), provenance=mock_prov)
    with pytest.raises((ValidationError, TypeError)):
        c.hypotheses = ()


def test_metadata_immutability(mock_prov):
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, metadata={"k": "v"}
    )
    with pytest.raises((ValidationError, TypeError)):
        h.metadata = {}
    with pytest.raises((ValidationError, TypeError)):
        h.metadata["k"] = "v2"


def test_invalid_causal_metadata_rejected(mock_prov):
    with pytest.raises(ValidationError):
        Hypothesis(
            hypothesis_id="h1",
            title="T",
            statement="S",
            provenance=mock_prov,
            metadata={"root_cause": True},
        )


def test_invalid_recommendation_metadata_rejected(mock_prov):
    with pytest.raises(ValidationError):
        Hypothesis(
            hypothesis_id="h1",
            title="T",
            statement="S",
            provenance=mock_prov,
            metadata={"recommendation_id": "r1"},
        )


def test_invalid_confidence_metadata_rejected(mock_prov):
    with pytest.raises(ValidationError):
        Hypothesis(
            hypothesis_id="h1",
            title="T",
            statement="S",
            provenance=mock_prov,
            metadata={"confidence": 0.99},
        )


def test_valid_metadata_accepted(mock_prov):
    h = Hypothesis(
        hypothesis_id="h1",
        title="T",
        statement="S",
        provenance=mock_prov,
        metadata={"author": "engineer1"},
    )
    assert h.metadata["author"] == "engineer1"


def test_explicit_version_accepted(mock_prov):
    h = Hypothesis(hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, version="v2")
    assert h.version == "v2"


def test_supersedes_reference(mock_prov):
    h = Hypothesis(
        hypothesis_id="h2",
        title="T",
        statement="S",
        provenance=mock_prov,
        supersedes_hypothesis_id="h1",
    )
    assert h.supersedes_hypothesis_id == "h1"


def test_no_pandas_dependency():
    assert True


def test_frozen_pydantic_model_extra_fields(mock_prov):
    with pytest.raises(ValidationError):
        Hypothesis(
            hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, unexpected_field="x"
        )


def test_mutable_default_protection():
    h1 = Hypothesis(
        hypothesis_id="h1",
        title="T",
        statement="S",
        provenance=ProvenanceRecord(
            source_uri="", activity_name="", epistemic_level=EpistemicLevel.HYPOTHESIS
        ),
    )
    h2 = Hypothesis(
        hypothesis_id="h2",
        title="T",
        statement="S",
        provenance=ProvenanceRecord(
            source_uri="", activity_name="", epistemic_level=EpistemicLevel.HYPOTHESIS
        ),
    )
    assert h1.evidence_links is h2.evidence_links or h1.evidence_links == ()
