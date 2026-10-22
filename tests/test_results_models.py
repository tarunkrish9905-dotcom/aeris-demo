"""
Unit tests for AERIS core results, issues, epistemic levels, and provenance models.

Verifies schema validation, immutability, epistemic integrity enforcement,
and result container behavior.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest
from pydantic import ValidationError

from aeris.core.provenance.tracker import (
    EpistemicIntegrityError,
    EpistemicLevel,
    ProvenanceRecord,
    compute_file_checksum,
    create_provenance,
)
from aeris.core.results.types import (
    BasePipelineResult,
    IngestionResult,
    Issue,
    IssueSeverity,
    QualityResult,
    ResultStatus,
    ValidationResult,
)

# ---------------------------------------------------------------------------
# Enumeration Tests
# ---------------------------------------------------------------------------


def test_result_status_values() -> None:
    """Verify all required result statuses are present and accessible."""
    expected_statuses = {"SUCCESS", "WARNING", "ERROR", "NOT_AVAILABLE", "PARTIAL"}
    actual_statuses = {s.value for s in ResultStatus}
    assert actual_statuses == expected_statuses


def test_issue_severity_values() -> None:
    """Verify all required issue severities are present."""
    expected_severities = {"INFO", "WARNING", "ERROR", "CRITICAL"}
    actual_severities = {s.value for s in IssueSeverity}
    assert actual_severities == expected_severities


def test_epistemic_level_hierarchy() -> None:
    """Verify all 8 evidence hierarchy levels are defined."""
    expected_levels = {
        "OBSERVED",
        "CALCULATED",
        "PROCEDURAL",
        "LIMIT",
        "HISTORICAL",
        "INFERRED",
        "HYPOTHESIS",
        "RECOMMENDATION",
    }
    actual_levels = {lvl.value for lvl in EpistemicLevel}
    assert actual_levels == expected_levels


def test_epistemic_integrity_classifications() -> None:
    """Verify empirical, factual, and interpretive epistemic classifications."""
    # Empirical measurements
    assert EpistemicLevel.OBSERVED.is_empirical()
    assert EpistemicLevel.CALCULATED.is_empirical()
    assert not EpistemicLevel.INFERRED.is_empirical()
    assert not EpistemicLevel.HYPOTHESIS.is_empirical()

    # Established facts / baselines
    assert EpistemicLevel.OBSERVED.is_factual()
    assert EpistemicLevel.CALCULATED.is_factual()
    assert EpistemicLevel.HISTORICAL.is_factual()
    assert not EpistemicLevel.INFERRED.is_factual()
    assert not EpistemicLevel.HYPOTHESIS.is_factual()
    assert not EpistemicLevel.RECOMMENDATION.is_factual()

    # Deductions / suggestions
    assert EpistemicLevel.INFERRED.is_interpretive()
    assert EpistemicLevel.HYPOTHESIS.is_interpretive()
    assert EpistemicLevel.RECOMMENDATION.is_interpretive()
    assert not EpistemicLevel.OBSERVED.is_interpretive()

    # Prescriptive standards
    assert EpistemicLevel.PROCEDURAL.is_prescriptive()
    assert EpistemicLevel.LIMIT.is_prescriptive()


def test_epistemic_integrity_assertion_enforcement() -> None:
    """Verify assert_factual prevents treating inferences/hypotheses as facts."""
    # Factual levels must pass without exception
    EpistemicLevel.OBSERVED.assert_factual("sensor_check")
    EpistemicLevel.CALCULATED.assert_factual("density_altitude_calc")
    EpistemicLevel.HISTORICAL.assert_factual("baseline_comparison")

    # Interpretive levels must raise EpistemicIntegrityError
    with pytest.raises(EpistemicIntegrityError) as exc_info:
        EpistemicLevel.INFERRED.assert_factual("flutter_check")
    assert "cannot treat interpretive level 'INFERRED' as an established fact" in str(
        exc_info.value
    )

    with pytest.raises(EpistemicIntegrityError):
        EpistemicLevel.HYPOTHESIS.assert_factual()

    with pytest.raises(EpistemicIntegrityError):
        EpistemicLevel.RECOMMENDATION.assert_factual()


# ---------------------------------------------------------------------------
# Issue Model Tests
# ---------------------------------------------------------------------------


def test_issue_model_valid() -> None:
    """Verify Issue model instantiates with required and optional fields."""
    issue = Issue(
        issue_code="ERR_TIMESTAMP_INVALID",
        severity=IssueSeverity.ERROR,
        message="Timestamp could not be parsed as ISO 8601 UTC string",
        field_name="timestamp",
        location=7,
        context={"raw_value": "INVALID_TIMESTAMP_STRING"},
    )
    assert issue.issue_code == "ERR_TIMESTAMP_INVALID"
    assert issue.severity == IssueSeverity.ERROR
    assert issue.location == 7
    assert issue.context["raw_value"] == "INVALID_TIMESTAMP_STRING"


def test_issue_model_immutability_and_validation() -> None:
    """Verify Issue is frozen and rejects empty codes or invalid severities."""
    issue = Issue(
        issue_code="WARN_MISSING_VALUE",
        severity=IssueSeverity.WARNING,
        message="Missing value encountered in non-critical parameter",
    )
    with pytest.raises(ValidationError):
        # Frozen model should reject mutations
        issue.message = "Mutated"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        # Empty code rejected
        Issue(issue_code="", severity=IssueSeverity.INFO, message="msg")

    with pytest.raises(ValidationError):
        # Invalid severity string rejected
        Issue(
            issue_code="CODE",
            severity="NOT_A_SEVERITY",  # type: ignore[arg-type]
            message="msg",
        )


# ---------------------------------------------------------------------------
# Provenance Model & Tracker Tests
# ---------------------------------------------------------------------------


def test_provenance_record_defaults_and_validation() -> None:
    """Verify ProvenanceRecord creation, auto-generated UUID, and immutability."""
    record = ProvenanceRecord(
        source_uri="tests/fixtures/csv/clean_flight.csv",
        activity_name="csv_ingestion",
        epistemic_level=EpistemicLevel.OBSERVED,
    )
    assert len(record.record_id) == 36  # Valid UUID string format
    assert record.created_at.tzinfo is not None
    assert record.parent_record_ids == []
    assert record.parameters == {}

    with pytest.raises(ValidationError):
        # Extra fields forbidden
        ProvenanceRecord(
            source_uri="file.csv",
            activity_name="act",
            epistemic_level=EpistemicLevel.OBSERVED,
            unknown_arg="extra",  # type: ignore[call-arg]
        )


def test_create_provenance_helper() -> None:
    """Verify create_provenance helper function."""
    prov = create_provenance(
        source_uri="dataset.csv",
        activity_name="validation",
        epistemic_level=EpistemicLevel.CALCULATED,
        parameters={"strict": True},
        parent_record_ids=["upstream-123"],
    )
    assert prov.source_uri == "dataset.csv"
    assert prov.activity_name == "validation"
    assert prov.epistemic_level == EpistemicLevel.CALCULATED
    assert prov.parameters == {"strict": True}
    assert prov.parent_record_ids == ["upstream-123"]


def test_compute_file_checksum(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify real SHA-256 calculation on existing fixture and error on missing file."""
    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    checksum = compute_file_checksum(clean_csv)
    assert len(checksum) == 64  # SHA-256 is 64 hex characters
    assert int(checksum, 16) > 0

    # Repeating computation on same file yields identical hash
    assert compute_file_checksum(clean_csv) == checksum

    # Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        compute_file_checksum(csv_fixtures_dir / "non_existent_file.csv")


# ---------------------------------------------------------------------------
# Pipeline Result Container Tests
# ---------------------------------------------------------------------------


def test_base_pipeline_result_status_and_properties() -> None:
    """Verify BasePipelineResult error/warning detection properties."""
    clean_result = BasePipelineResult(status=ResultStatus.SUCCESS)
    assert clean_result.is_successful
    assert not clean_result.has_errors
    assert not clean_result.has_warnings

    err_issue = Issue(
        issue_code="ERR_PARSE",
        severity=IssueSeverity.ERROR,
        message="Parsing failed",
    )
    failed_result = BasePipelineResult(
        status=ResultStatus.ERROR,
        issues=[err_issue],
    )
    assert not failed_result.is_successful
    assert failed_result.has_errors
    assert failed_result.error_count == 1
    assert failed_result.warning_count == 0


def test_ingestion_result_container() -> None:
    """Verify IngestionResult holds data, column metadata, and provenance."""
    sample_df = pd.DataFrame({"time": [1, 2], "val": [10.0, 20.0]})
    prov = create_provenance("sample.csv", "ingest", EpistemicLevel.OBSERVED)

    res = IngestionResult(
        status=ResultStatus.SUCCESS,
        data=sample_df,
        row_count=2,
        column_names=["time", "val"],
        source_path="sample.csv",
        provenance=prov,
    )
    assert res.status == ResultStatus.SUCCESS
    assert res.row_count == 2
    assert res.column_names == ["time", "val"]
    assert res.provenance == prov
    assert isinstance(res.data, pd.DataFrame)

    with pytest.raises(ValidationError):
        # Negative row_count must be rejected (ge=0)
        IngestionResult(status=ResultStatus.SUCCESS, row_count=-1)


def test_validation_result_container() -> None:
    """Verify ValidationResult captures validation metrics and issue logs."""
    issue = Issue(
        issue_code="WARN_RANGE",
        severity=IssueSeverity.WARNING,
        message="Value slightly above nominal",
        field_name="altitude_ft",
    )
    res = ValidationResult(
        status=ResultStatus.WARNING,
        is_valid=True,
        validated_row_count=60,
        violation_count=1,
        validated_columns=["altitude_ft"],
        issues=[issue],
    )
    assert res.is_valid
    assert res.has_warnings
    assert not res.has_errors
    assert res.violation_count == 1
    assert len(res.issues) == 1


def test_quality_result_container_and_score_bounds() -> None:
    """Verify QualityResult validates quality_score within [0.0, 1.0]."""
    valid_res = QualityResult(
        status=ResultStatus.SUCCESS,
        quality_score=0.95,
        metrics={"completeness": 0.98, "duplicate_rate": 0.0},
        summary="High data quality; suitable for analysis",
    )
    assert valid_res.quality_score == 0.95
    assert valid_res.metrics["completeness"] == 0.98

    # Quality score can be None when NOT_AVAILABLE
    na_res = QualityResult(
        status=ResultStatus.NOT_AVAILABLE,
        quality_score=None,
        summary="Quality assessment could not be performed on empty file",
    )
    assert na_res.quality_score is None

    # Scores outside [0.0, 1.0] must be rejected
    with pytest.raises(ValidationError):
        QualityResult(status=ResultStatus.SUCCESS, quality_score=1.05)

    with pytest.raises(ValidationError):
        QualityResult(status=ResultStatus.SUCCESS, quality_score=-0.01)
