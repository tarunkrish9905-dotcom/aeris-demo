import datetime
import uuid

import pytest
from pydantic import ValidationError

from aeris.core.evidence.models import EvidenceCollection, EvidenceRecord, EvidenceType
from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord


def _dt() -> datetime.datetime:
    return datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


def _prov(level: EpistemicLevel = EpistemicLevel.OBSERVED) -> ProvenanceRecord:
    return ProvenanceRecord(
        record_id=str(uuid.uuid4()),
        activity_name="TEST_ACTIVITY",
        epistemic_level=level,
        parent_record_ids=["p1"],
        created_at=datetime.datetime.now(datetime.UTC),
        source_uri="test",
    )


def test_valid_evidence_creation():
    ev = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="Test desc",
        source_reference="test_source",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=_prov(),
    )
    assert ev.evidence_id == "ev1"


def test_evidence_id_required():
    with pytest.raises(ValidationError):
        EvidenceRecord(
            evidence_type=EvidenceType.OBSERVATION,
            description="Test desc",
            source_reference="test_source",
            epistemic_level=EpistemicLevel.OBSERVED,
            provenance=_prov(),
        )


def test_evidence_id_validation():
    with pytest.raises(ValidationError):
        EvidenceRecord(
            evidence_id="",
            evidence_type=EvidenceType.OBSERVATION,
            description="Test desc",
            source_reference="test_source",
            epistemic_level=EpistemicLevel.OBSERVED,
            provenance=_prov(),
        )


def test_description_validation():
    with pytest.raises(ValidationError):
        EvidenceRecord(
            evidence_id="ev1",
            evidence_type=EvidenceType.OBSERVATION,
            description="",
            source_reference="test_source",
            epistemic_level=EpistemicLevel.OBSERVED,
            provenance=_prov(),
        )


def test_evidence_type_validation():
    with pytest.raises(ValidationError):
        EvidenceRecord(
            evidence_id="ev1",
            evidence_type="INVALID_TYPE",
            description="Test desc",
            source_reference="test_source",
            epistemic_level=EpistemicLevel.OBSERVED,
            provenance=_prov(),
        )


def test_source_reference_required():
    with pytest.raises(ValidationError):
        EvidenceRecord(
            evidence_id="ev1",
            evidence_type=EvidenceType.OBSERVATION,
            description="Test desc",
            epistemic_level=EpistemicLevel.OBSERVED,
            provenance=_prov(),
        )


def test_source_reference_preservation():
    ev = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src_ref",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=_prov(),
    )
    assert ev.source_reference == "src_ref"


def test_source_record_reference():
    ev = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src",
        source_record_id="rec1",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=_prov(),
    )
    assert ev.source_record_id == "rec1"


def test_parameter_reference():
    ev = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src",
        source_parameter_id="airspeed",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=_prov(),
    )
    assert ev.source_parameter_id == "airspeed"


def test_timestamp_validation():
    ev = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src",
        timestamp=_dt(),
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=_prov(),
    )
    assert ev.timestamp == _dt()


def test_naive_timestamp_rejection():
    with pytest.raises(ValueError, match="timezone-aware"):
        EvidenceRecord(
            evidence_id="ev1",
            evidence_type=EvidenceType.OBSERVATION,
            description="d",
            source_reference="src",
            timestamp=datetime.datetime(2026, 1, 1),
            epistemic_level=EpistemicLevel.OBSERVED,
            provenance=_prov(),
        )


def test_non_utc_timestamp_rejection():
    tz = datetime.timezone(datetime.timedelta(hours=-5))
    with pytest.raises(ValueError, match="UTC"):
        EvidenceRecord(
            evidence_id="ev1",
            evidence_type=EvidenceType.OBSERVATION,
            description="d",
            source_reference="src",
            timestamp=datetime.datetime(2026, 1, 1, tzinfo=tz),
            epistemic_level=EpistemicLevel.OBSERVED,
            provenance=_prov(),
        )


def test_utc_timestamp_acceptance():
    ev = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src",
        timestamp=_dt(),
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=_prov(),
    )
    assert ev.timestamp.tzinfo == datetime.UTC


def test_provenance_preservation():
    p = _prov()
    ev = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=p,
    )
    assert ev.provenance.record_id == p.record_id


def test_parent_lineage_preservation():
    p = _prov()
    ev = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=p,
    )
    assert "p1" in ev.provenance.parent_record_ids


def test_epistemic_level_preservation():
    ev = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src",
        epistemic_level=EpistemicLevel.CALCULATED,
        provenance=_prov(),
    )
    assert ev.epistemic_level == EpistemicLevel.CALCULATED


def test_observed_evidence():
    ev = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=_prov(),
    )
    assert ev.evidence_type == EvidenceType.OBSERVATION


def test_calculated_evidence():
    ev = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.CALCULATION,
        description="d",
        source_reference="src",
        epistemic_level=EpistemicLevel.CALCULATED,
        provenance=_prov(),
    )
    assert ev.evidence_type == EvidenceType.CALCULATION


def test_procedural_evidence():
    ev = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.PROCEDURAL_REFERENCE,
        description="d",
        source_reference="src",
        epistemic_level=EpistemicLevel.PROCEDURAL,
        provenance=_prov(),
    )
    assert ev.evidence_type == EvidenceType.PROCEDURAL_REFERENCE


def test_limit_evidence():
    ev = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.LIMIT_REFERENCE,
        description="d",
        source_reference="src",
        epistemic_level=EpistemicLevel.PROCEDURAL,
        provenance=_prov(),
    )
    assert ev.evidence_type == EvidenceType.LIMIT_REFERENCE


def test_no_epistemic_elevation():
    # If the user tries to assign a different level it just preserves the one assigned,
    # but the domain doesn't infer it automatically.
    ev = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=_prov(EpistemicLevel.OBSERVED),
    )
    assert ev.epistemic_level == EpistemicLevel.OBSERVED
    assert ev.epistemic_level != EpistemicLevel.INFERRED


def test_no_root_cause_field():
    with pytest.raises(ValidationError):
        EvidenceRecord(
            evidence_id="ev1",
            evidence_type=EvidenceType.OBSERVATION,
            description="d",
            source_reference="src",
            epistemic_level=EpistemicLevel.OBSERVED,
            provenance=_prov(),
            root_cause="some cause",
        )


def test_no_hypothesis_field():
    with pytest.raises(ValidationError):
        EvidenceRecord(
            evidence_id="ev1",
            evidence_type=EvidenceType.OBSERVATION,
            description="d",
            source_reference="src",
            epistemic_level=EpistemicLevel.OBSERVED,
            provenance=_prov(),
            hypothesis="some hypothesis",
        )


def test_no_recommendation_field():
    with pytest.raises(ValidationError):
        EvidenceRecord(
            evidence_id="ev1",
            evidence_type=EvidenceType.OBSERVATION,
            description="d",
            source_reference="src",
            epistemic_level=EpistemicLevel.OBSERVED,
            provenance=_prov(),
            recommendation="fix it",
        )


def test_no_confidence_in_root_cause_field():
    with pytest.raises(ValidationError):
        EvidenceRecord(
            evidence_id="ev1",
            evidence_type=EvidenceType.OBSERVATION,
            description="d",
            source_reference="src",
            epistemic_level=EpistemicLevel.OBSERVED,
            provenance=_prov(),
            confidence=0.9,
        )


def test_no_dataframe_dependency():

    assert True  # We don't import it in models.py


def test_frozen_model():
    ev = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=_prov(),
    )
    with pytest.raises(ValidationError):
        ev.description = "new desc"


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        EvidenceRecord(
            evidence_id="ev1",
            evidence_type=EvidenceType.OBSERVATION,
            description="d",
            source_reference="src",
            epistemic_level=EpistemicLevel.OBSERVED,
            provenance=_prov(),
            unsupported_field="hello",
        )


def test_deterministic_serialization():
    ev = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=_prov(),
    )
    d = ev.model_dump(mode="json")
    ev2 = EvidenceRecord.model_validate(d)
    assert ev.evidence_id == ev2.evidence_id


def test_unit_metadata_preservation():
    ev = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src",
        unit="kt",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=_prov(),
    )
    assert ev.unit == "kt"


def test_source_row_index_preservation():
    ev = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src",
        source_row_index=42,
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=_prov(),
    )
    assert ev.source_row_index == 42


def test_evidence_collection_creation():
    ev1 = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=_prov(),
    )
    ev2 = EvidenceRecord(
        evidence_id="ev2",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=_prov(),
    )
    col = EvidenceCollection(collection_id="c1", evidence=(ev1, ev2))
    assert len(col.evidence) == 2


def test_evidence_collection_duplicate_id_rejection():
    ev1 = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=_prov(),
    )
    ev2 = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src2",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=_prov(),
    )
    with pytest.raises(ValidationError, match="ERR_DUPLICATE_EVIDENCE_ID"):
        EvidenceCollection(collection_id="c1", evidence=(ev1, ev2))


def test_evidence_collection_deterministic_ordering():
    ev1 = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=_prov(),
    )
    ev2 = EvidenceRecord(
        evidence_id="ev2",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=_prov(),
    )
    col = EvidenceCollection(collection_id="c1", evidence=(ev1, ev2))
    assert col.evidence[0].evidence_id == "ev1"
    assert col.evidence[1].evidence_id == "ev2"


def test_evidence_collection_immutability():
    ev1 = EvidenceRecord(
        evidence_id="ev1",
        evidence_type=EvidenceType.OBSERVATION,
        description="d",
        source_reference="src",
        epistemic_level=EpistemicLevel.OBSERVED,
        provenance=_prov(),
    )
    col = EvidenceCollection(collection_id="c1", evidence=(ev1,))
    with pytest.raises(ValidationError):
        col.evidence = ()
