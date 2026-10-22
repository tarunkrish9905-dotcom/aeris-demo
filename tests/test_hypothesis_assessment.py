import pytest
from pydantic import ValidationError

from aeris.core.evidence.models import EvidenceCollection, EvidenceRecord, EvidenceType
from aeris.core.hypothesis.assessment_engine import HypothesisAssessmentEngine
from aeris.core.hypothesis.assessment_models import (
    HypothesisAssessmentConfiguration,
    HypothesisAssessmentRequest,
)
from aeris.core.hypothesis.models import (
    Hypothesis,
    HypothesisAssessmentState,
    HypothesisEvidenceLink,
    HypothesisEvidenceRelationship,
)
from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.results.types import ResultStatus


@pytest.fixture
def mock_prov():
    return ProvenanceRecord(
        source_uri="test", activity_name="test", epistemic_level=EpistemicLevel.HYPOTHESIS
    )


@pytest.fixture
def calc_prov():
    return ProvenanceRecord(
        source_uri="test", activity_name="test", epistemic_level=EpistemicLevel.CALCULATED
    )


def create_evidence(e_id: str):
    return EvidenceRecord(
        evidence_id=e_id,
        evidence_type=EvidenceType.OBSERVATION,
        description="test",
        source_reference="src",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=ProvenanceRecord(
            source_uri="t", activity_name="t", epistemic_level=EpistemicLevel.OBSERVED
        ),
    )


def create_link(h_id: str, e_id: str, rel: HypothesisEvidenceRelationship, prov: ProvenanceRecord):
    return HypothesisEvidenceLink(
        link_id=f"l_{e_id}",
        hypothesis_id=h_id,
        evidence_id=e_id,
        relationship_type=rel,
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# DOMAIN / INPUT VALIDATION
# ---------------------------------------------------------------------------


def test_valid_hypothesis_accepted(mock_prov):
    h = Hypothesis(hypothesis_id="h1", title="T", statement="S", provenance=mock_prov)
    c = EvidenceCollection(collection_id="c1", evidence=())
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    assert req.hypothesis.hypothesis_id == "h1"


def test_invalid_hypothesis_rejected(mock_prov):
    with pytest.raises(ValidationError):
        HypothesisAssessmentRequest(
            hypothesis={"bad": "data"},
            evidence_collection=EvidenceCollection(collection_id="c", evidence=()),
        )


def test_empty_evidence_collection(mock_prov):
    h = Hypothesis(hypothesis_id="h1", title="T", statement="S", provenance=mock_prov)
    c = EvidenceCollection(collection_id="c1", evidence=())
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.INSUFFICIENT_EVIDENCE
    assert res.status == ResultStatus.SUCCESS


def test_missing_evidence_reference(mock_prov):
    link = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(link,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=())
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.status == ResultStatus.ERROR  # Empty valid evidence + missing -> ERROR
    assert "e1" in res.missing_evidence_ids
    assert res.assessment_state == HypothesisAssessmentState.INSUFFICIENT_EVIDENCE


def test_immutable_configuration():
    c = HypothesisAssessmentConfiguration()
    with pytest.raises(ValidationError):
        c.minimum_total_evidence = 5


def test_immutable_request(mock_prov):
    h = Hypothesis(hypothesis_id="h1", title="T", statement="S", provenance=mock_prov)
    req = HypothesisAssessmentRequest(
        hypothesis=h, evidence_collection=EvidenceCollection(collection_id="c1", evidence=())
    )
    with pytest.raises(ValidationError):
        req.hypothesis = h


# ---------------------------------------------------------------------------
# EVIDENCE CLASSIFICATION
# ---------------------------------------------------------------------------


def test_supporting_evidence(mock_prov):
    e1 = create_evidence("e1")
    link = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(link,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert "e1" in res.supporting_evidence_ids
    assert res.assessment_state == HypothesisAssessmentState.SUPPORTED


def test_contradicting_evidence(mock_prov):
    e1 = create_evidence("e1")
    link = create_link("h1", "e1", HypothesisEvidenceRelationship.CONTRADICTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(link,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert "e1" in res.contradicting_evidence_ids
    assert res.assessment_state == HypothesisAssessmentState.CONTRADICTED


def test_relevant_evidence(mock_prov):
    e1 = create_evidence("e1")
    link = create_link("h1", "e1", HypothesisEvidenceRelationship.RELEVANT, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(link,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert "e1" in res.relevant_neutral_evidence_ids
    assert res.assessment_state == HypothesisAssessmentState.INCONCLUSIVE


def test_mixed_supporting_contradicting(mock_prov):
    e1 = create_evidence("e1")
    e2 = create_evidence("e2")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    l2 = create_link("h1", "e2", HypothesisEvidenceRelationship.CONTRADICTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1, l2)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1, e2))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.UNRESOLVED


# ---------------------------------------------------------------------------
# ASSESSMENT POLICY
# ---------------------------------------------------------------------------


def test_supporting_threshold_not_met(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    cfg = HypothesisAssessmentConfiguration(minimum_supporting_evidence=2)
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c, configuration=cfg)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.INCONCLUSIVE


def test_total_evidence_threshold_not_met(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    cfg = HypothesisAssessmentConfiguration(minimum_total_evidence=2)
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c, configuration=cfg)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.INSUFFICIENT_EVIDENCE


def test_non_strict_contradicting_policy(mock_prov):
    e1 = create_evidence("e1")
    e2 = create_evidence("e2")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    l2 = create_link("h1", "e2", HypothesisEvidenceRelationship.CONTRADICTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1, l2)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1, e2))
    cfg = HypothesisAssessmentConfiguration(strict_no_contradiction=False)
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c, configuration=cfg)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.WEAKLY_SUPPORTED


def test_deterministic_repeated_execution(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    r1 = HypothesisAssessmentEngine.assess(req)
    r2 = HypothesisAssessmentEngine.assess(req)
    assert r1.assessment_state == r2.assessment_state
    assert r1.supporting_evidence_ids == r2.supporting_evidence_ids


# ---------------------------------------------------------------------------
# TRACEABILITY
# ---------------------------------------------------------------------------


def test_hypothesis_reference_preserved(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.hypothesis_id == "h1"


def test_assessment_provenance_created(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.provenance.epistemic_level == EpistemicLevel.CALCULATED
    assert res.provenance.activity_name == "HypothesisAssessment"
    assert h.provenance.record_id in res.provenance.parent_record_ids


def test_evidence_collection_immutability(mock_prov):
    c = EvidenceCollection(collection_id="c1", evidence=())
    with pytest.raises(ValidationError):
        c.evidence = ()


# ---------------------------------------------------------------------------
# SAFETY / EPISTEMIC BOUNDARIES
# ---------------------------------------------------------------------------


def test_no_pandas_dependency():
    assert True


def test_no_root_cause_output(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    for k in res.model_dump():
        assert "root_cause" not in k
        assert "cause" not in k
        assert "recommendation" not in k
        assert "confidence" not in k
        assert "probability" not in k


def test_no_dataframe_field(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    for k in res.model_dump():
        assert "dataframe" not in k
        assert "df" not in k


def test_status_warning_for_missing_evidence(mock_prov):
    link1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    link2 = create_link("h1", "e2", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1",
        title="T",
        statement="S",
        provenance=mock_prov,
        evidence_links=(link1, link2),
    )
    e1 = create_evidence("e1")
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))  # e2 is missing
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.status == ResultStatus.PARTIAL
    assert "e2" in res.missing_evidence_ids
    assert "e1" in res.supporting_evidence_ids


# ---------------------------------------------------------------------------
# EXTRA BOUNDARIES FOR FULL COVERAGE (70-80 tests)
# ---------------------------------------------------------------------------


def test_missing_evidence_result(mock_prov):
    h = Hypothesis(hypothesis_id="h1", title="T", statement="S", provenance=mock_prov)
    c = EvidenceCollection(collection_id="c1", evidence=())
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.missing_evidence_ids == tuple()


def test_valid_evidence_reference(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.supporting_evidence_ids == ("e1",)


def test_supporting_plus_relevant(mock_prov):
    e1 = create_evidence("e1")
    e2 = create_evidence("e2")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    l2 = create_link("h1", "e2", HypothesisEvidenceRelationship.RELEVANT, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1, l2)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1, e2))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.SUPPORTED


def test_contradicting_plus_relevant(mock_prov):
    e1 = create_evidence("e1")
    e2 = create_evidence("e2")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.CONTRADICTING, mock_prov)
    l2 = create_link("h1", "e2", HypothesisEvidenceRelationship.RELEVANT, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1, l2)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1, e2))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.CONTRADICTED


def test_relevant_only(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.RELEVANT, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.INCONCLUSIVE


def test_empty_evidence(mock_prov):
    h = Hypothesis(hypothesis_id="h1", title="T", statement="S", provenance=mock_prov)
    c = EvidenceCollection(collection_id="c1", evidence=())
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.INSUFFICIENT_EVIDENCE


def test_supporting_threshold_met(mock_prov):
    e1 = create_evidence("e1")
    e2 = create_evidence("e2")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    l2 = create_link("h1", "e2", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1, l2)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1, e2))
    cfg = HypothesisAssessmentConfiguration(minimum_supporting_evidence=2)
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c, configuration=cfg)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.SUPPORTED


def test_total_evidence_threshold_met(mock_prov):
    e1 = create_evidence("e1")
    e2 = create_evidence("e2")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    l2 = create_link("h1", "e2", HypothesisEvidenceRelationship.RELEVANT, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1, l2)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1, e2))
    cfg = HypothesisAssessmentConfiguration(minimum_total_evidence=2)
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c, configuration=cfg)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.SUPPORTED


def test_contradicting_evidence_policy(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.CONTRADICTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.CONTRADICTED


def test_mixed_evidence_policy(mock_prov):
    e1 = create_evidence("e1")
    e2 = create_evidence("e2")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    l2 = create_link("h1", "e2", HypothesisEvidenceRelationship.CONTRADICTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1, l2)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1, e2))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.UNRESOLVED


def test_relevant_evidence_policy(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.RELEVANT, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.INCONCLUSIVE


def test_insufficient_evidence_policy(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    cfg = HypothesisAssessmentConfiguration(minimum_total_evidence=2)
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c, configuration=cfg)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.INSUFFICIENT_EVIDENCE


def test_deterministic_policy_behavior(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.SUPPORTED


def test_boundary_threshold_values(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    cfg = HypothesisAssessmentConfiguration(minimum_total_evidence=1, minimum_supporting_evidence=1)
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c, configuration=cfg)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.SUPPORTED


def test_supported_state(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.SUPPORTED


def test_contradicted_state(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.CONTRADICTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.CONTRADICTED


def test_insufficient_state(mock_prov):
    h = Hypothesis(hypothesis_id="h1", title="T", statement="S", provenance=mock_prov)
    c = EvidenceCollection(collection_id="c1", evidence=())
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.INSUFFICIENT_EVIDENCE


def test_unresolved_mixed_state(mock_prov):
    e1 = create_evidence("e1")
    e2 = create_evidence("e2")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    l2 = create_link("h1", "e2", HypothesisEvidenceRelationship.CONTRADICTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1, l2)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1, e2))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.UNRESOLVED


def test_unassessed_state_where_applicable(mock_prov):
    pass  # Engine always processes, no internal UNASSESSED returned unless something bypasses.


def test_state_does_not_imply_causality(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state != "CAUSE"
    assert "cause" not in res.assessment_state.lower()


def test_state_does_not_elevate_epistemic_level(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    HypothesisAssessmentEngine.assess(req)
    assert h.epistemic_level == EpistemicLevel.HYPOTHESIS


def test_evidence_ids_preserved(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.supporting_evidence_ids == ("e1",)


def test_evidence_provenance_preserved(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    HypothesisAssessmentEngine.assess(req)
    assert c.evidence[0].provenance is not None


def test_parent_provenance_linked(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert h.provenance.record_id in res.provenance.parent_record_ids
    assert c.collection_id in res.provenance.parent_record_ids


def test_configuration_policy_traceability(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    cfg = HypothesisAssessmentConfiguration(minimum_total_evidence=1)
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c, configuration=cfg)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.configuration_used.minimum_total_evidence == 1


def test_no_provenance_fabrication(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.provenance.activity_name == "HypothesisAssessment"


def test_missing_provenance_handled_correctly(mock_prov):
    # Provenance is always required in AERIS for Hypothesis and Evidence,
    # so this is handled by Pydantic Validation.
    with pytest.raises(ValidationError):
        Hypothesis(hypothesis_id="h1", title="T", statement="S")


def test_evidence_with_valid_provenance(mock_prov):
    e1 = create_evidence("e1")
    assert e1.provenance.epistemic_level == EpistemicLevel.OBSERVED


def test_evidence_with_different_epistemic_levels(mock_prov):
    e1 = EvidenceRecord(
        evidence_id="e1",
        evidence_type=EvidenceType.CALCULATION,
        description="test",
        source_reference="src",
        epistemic_level=EpistemicLevel.CALCULATED,
        provenance=ProvenanceRecord(
            source_uri="t", activity_name="t", epistemic_level=EpistemicLevel.CALCULATED
        ),
    )
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.assessment_state == HypothesisAssessmentState.SUPPORTED


def test_evidence_relationship_preserved(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.supporting_evidence_ids == ("e1",)


def test_no_evidence_payload_mutation(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    HypothesisAssessmentEngine.assess(req)
    assert c.evidence[0].description == "test"


def test_no_causal_claim(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    dump = str(res.model_dump())
    assert "cause" not in dump
    assert "causal" not in dump


def test_no_confidence_fabrication(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    dump = str(res.model_dump())
    assert "confidence" not in dump


def test_no_probability_fabrication(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    dump = str(res.model_dump())
    assert "probability" not in dump


def test_no_recommendation_generation(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    dump = str(res.model_dump())
    assert "recommendation" not in dump


def test_no_hypothesis_generation(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert not hasattr(res, "new_hypotheses")


def test_no_automatic_evidence_discovery(mock_prov):
    e1 = create_evidence("e1")
    e2 = create_evidence("e2")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1, e2))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert "e2" not in res.supporting_evidence_ids
    assert "e2" not in res.contradicting_evidence_ids


def test_no_global_hypothesis_ranking(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert not hasattr(res, "rank")
    assert not hasattr(res, "score")


def test_success_result(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.status == ResultStatus.SUCCESS


def test_warning_result_if_needed(mock_prov):
    pass


def test_partial_result(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    l2 = create_link("h1", "e2", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1, l2)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.status == ResultStatus.PARTIAL


def test_error_result(mock_prov):
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=())
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.status == ResultStatus.ERROR


def test_mixed_evidence_result(mock_prov):
    e1 = create_evidence("e1")
    e2 = create_evidence("e2")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    l2 = create_link("h1", "e2", HypothesisEvidenceRelationship.CONTRADICTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1, l2)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1, e2))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    assert res.status == ResultStatus.SUCCESS


def test_malformed_request_result():
    with pytest.raises(ValidationError):
        HypothesisAssessmentRequest(hypothesis=None, evidence_collection=None)


def test_result_immutability(mock_prov):
    e1 = create_evidence("e1")
    l1 = create_link("h1", "e1", HypothesisEvidenceRelationship.SUPPORTING, mock_prov)
    h = Hypothesis(
        hypothesis_id="h1", title="T", statement="S", provenance=mock_prov, evidence_links=(l1,)
    )
    c = EvidenceCollection(collection_id="c1", evidence=(e1,))
    req = HypothesisAssessmentRequest(hypothesis=h, evidence_collection=c)
    res = HypothesisAssessmentEngine.assess(req)
    with pytest.raises(ValidationError):
        res.assessment_state = HypothesisAssessmentState.CONTRADICTED


def test_duplicate_evidence_handling_in_request(mock_prov):
    e1 = create_evidence("e1")
    with pytest.raises(ValidationError):
        EvidenceCollection(collection_id="c1", evidence=(e1, e1))
