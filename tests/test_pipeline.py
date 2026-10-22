"""Tests for the AERIS core data pipeline orchestration."""

import pathlib

import pandas as pd

from aeris.core.normalization.engine import NormalizationRequest
from aeris.core.pipeline.orchestrator import run_csv_pipeline
from aeris.core.results.types import ResultStatus


def test_pipeline_clean_flight(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify end-to-end pipeline on clean data yields SUCCESS."""
    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    res = run_csv_pipeline(clean_csv)

    assert res.status == ResultStatus.SUCCESS
    assert res.ingestion.status == ResultStatus.SUCCESS
    assert res.normalization is None

    assert res.validation is not None
    assert res.validation.status == ResultStatus.SUCCESS

    assert res.quality is not None
    assert res.quality.status == ResultStatus.SUCCESS

    # Provenance chain
    assert res.provenance is not None
    assert res.ingestion.provenance is not None
    assert res.validation.provenance is not None
    assert res.quality.provenance is not None

    assert res.ingestion.provenance.record_id in res.provenance.parent_record_ids
    assert res.validation.provenance.record_id in res.provenance.parent_record_ids
    assert res.quality.provenance.record_id in res.provenance.parent_record_ids


def test_pipeline_missing_values(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify pipeline degrades gracefully to WARNING status with missing values."""
    csv_path = csv_fixtures_dir / "missing_values.csv"
    res = run_csv_pipeline(csv_path)

    # Ingestion works
    assert res.ingestion.status == ResultStatus.SUCCESS
    # Validation flags warning (unexpected column) or success (if none).
    # Missing values might just pass validation if they don't break required schema.
    assert res.validation is not None
    # Quality flags missing cells -> WARNING
    assert res.quality is not None
    assert res.quality.status in (ResultStatus.WARNING, ResultStatus.PARTIAL)
    assert res.status in (ResultStatus.WARNING, ResultStatus.PARTIAL)


def test_pipeline_invalid_timestamps(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify pipeline propagates ERROR status for invalid timestamps."""
    csv_path = csv_fixtures_dir / "invalid_timestamps.csv"
    res = run_csv_pipeline(csv_path)

    assert res.ingestion.status == ResultStatus.SUCCESS

    assert res.validation is not None
    assert res.validation.status == ResultStatus.ERROR

    assert res.quality is not None
    assert res.quality.status in (ResultStatus.PARTIAL, ResultStatus.WARNING)

    # Pipeline inherits ERROR
    assert res.status == ResultStatus.ERROR


def test_pipeline_duplicate_timestamps(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify pipeline handles duplicate timestamps properly (WARNING)."""
    csv_path = csv_fixtures_dir / "duplicate_timestamps.csv"
    res = run_csv_pipeline(csv_path)

    assert res.validation is not None
    assert res.validation.status == ResultStatus.WARNING
    assert res.quality is not None
    assert res.quality.status in (ResultStatus.WARNING, ResultStatus.PARTIAL, ResultStatus.SUCCESS)
    assert res.status in (ResultStatus.WARNING, ResultStatus.PARTIAL)


def test_pipeline_unordered_timestamps(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify pipeline handles unordered timestamps properly (WARNING)."""
    csv_path = csv_fixtures_dir / "unordered_timestamps.csv"
    res = run_csv_pipeline(csv_path)

    assert res.validation is not None
    assert res.validation.status == ResultStatus.WARNING
    assert res.status in (ResultStatus.WARNING, ResultStatus.PARTIAL)


def test_pipeline_nonexistent_file() -> None:
    """Verify pipeline handles missing file gracefully without crashing downstream."""
    res = run_csv_pipeline("does_not_exist.csv")

    assert res.status == ResultStatus.ERROR
    assert res.ingestion.status == ResultStatus.ERROR
    assert res.validation is None
    assert res.quality is None
    assert any(i.issue_code == "ERR_FILE_NOT_FOUND" for i in res.issues)


def test_pipeline_malformed_empty_file(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify pipeline handles 0-byte file gracefully."""
    csv_path = csv_fixtures_dir / "malformed_empty.csv"
    res = run_csv_pipeline(csv_path)

    assert res.status == ResultStatus.ERROR
    assert res.ingestion.status == ResultStatus.ERROR
    assert res.validation is None
    assert res.quality is None
    assert any(i.issue_code == "ERR_EMPTY_FILE" for i in res.issues)


def test_pipeline_immutability(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify raw DataFrame is not modified by the pipeline."""
    csv_path = csv_fixtures_dir / "unordered_timestamps.csv"
    res = run_csv_pipeline(csv_path)

    assert res.ingestion.data is not None
    df = res.ingestion.data
    # If the data was sorted by timestamp, index 14 would be 2026-10-06T10:00:15...
    # The fixture generator swapped rows 14 and 15, so 14 has ...:15 and 15 has ...:14.
    # Check that it's still swapped (not sorted)
    dt14 = str(df.iloc[14]["timestamp"])
    dt15 = str(df.iloc[15]["timestamp"])
    assert "10:00:15" in dt14
    assert "10:00:14" in dt15


def test_pipeline_with_normalization(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify pipeline integrates explicit normalization successfully."""
    csv_path = csv_fixtures_dir / "clean_flight.csv"
    req = NormalizationRequest()
    req.rename_column("airspeed_kts", "airspeed")
    req.convert_type("altitude_ft", float)
    req.standardize_timestamp("timestamp")

    res = run_csv_pipeline(csv_path, normalization_request=req)

    assert res.status == ResultStatus.SUCCESS
    assert res.normalization is not None
    assert res.normalization.status == ResultStatus.SUCCESS

    df_norm = res.normalization.data
    assert df_norm is not None
    assert "airspeed" in df_norm.columns
    assert "airspeed_kts" not in df_norm.columns
    assert pd.api.types.is_datetime64_any_dtype(df_norm["timestamp"])

    # Provenance linkage
    assert res.normalization.provenance is not None
    assert res.provenance is not None
    assert res.normalization.provenance.record_id in res.provenance.parent_record_ids
    assert res.ingestion.provenance is not None
    assert res.ingestion.provenance.record_id in res.normalization.provenance.parent_record_ids


def test_pipeline_normalization_error_propagation(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify pipeline propagates normalization errors to overall status."""
    csv_path = csv_fixtures_dir / "clean_flight.csv"
    req = NormalizationRequest()
    req.rename_column("nonexistent_column", "new_col")

    res = run_csv_pipeline(csv_path, normalization_request=req)

    # Ingestion/Val/Qual are SUCCESS, but Normalization is ERROR
    assert res.status == ResultStatus.ERROR
    assert res.ingestion.status == ResultStatus.SUCCESS
    assert res.normalization is not None
    assert res.normalization.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_MISSING_SOURCE_COLUMN" for i in res.issues)


def test_pipeline_validation_error_with_normalization(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify normalization still attempts to run even if validation yields ERROR."""
    csv_path = csv_fixtures_dir / "invalid_timestamps.csv"
    req = NormalizationRequest()
    req.rename_column("airspeed_kts", "airspeed")

    res = run_csv_pipeline(csv_path, normalization_request=req)

    # Validation is ERROR because of bad timestamps
    assert res.validation is not None
    assert res.validation.status == ResultStatus.ERROR

    # Quality still executes
    assert res.quality is not None
    assert res.quality.status in (ResultStatus.PARTIAL, ResultStatus.WARNING)

    # Normalization executes because the column `airspeed_kts` exists
    assert res.normalization is not None
    assert res.normalization.status == ResultStatus.SUCCESS

    # Overall status is ERROR because of validation
    assert res.status == ResultStatus.ERROR


def test_pipeline_normalization_immutability(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify raw ingestion DataFrame remains completely immutable during normalization."""
    csv_path = csv_fixtures_dir / "clean_flight.csv"
    req = NormalizationRequest()
    req.rename_column("airspeed_kts", "airspeed")

    res = run_csv_pipeline(csv_path, normalization_request=req)

    assert res.ingestion.data is not None
    assert res.normalization is not None
    assert res.normalization.data is not None

    df_raw = res.ingestion.data
    df_norm = res.normalization.data

    # Check isolation
    assert "airspeed_kts" in df_raw.columns
    assert "airspeed" not in df_raw.columns
    assert "airspeed_kts" not in df_norm.columns
    assert "airspeed" in df_norm.columns
    assert id(df_raw) != id(df_norm)


def test_pipeline_no_automatic_inference(csv_fixtures_dir: pathlib.Path) -> None:
    """NEGATIVE TEST: Verify pipeline never applies unrequested normalizations."""
    csv_path = csv_fixtures_dir / "clean_flight.csv"

    # Run pipeline with empty normalization request
    req = NormalizationRequest()
    res = run_csv_pipeline(csv_path, normalization_request=req)

    assert res.normalization is not None
    df_norm = res.normalization.data
    assert df_norm is not None

    # Proves 'airspeed_kts' did not magically become 'airspeed'
    assert "airspeed_kts" in df_norm.columns
    assert "airspeed" not in df_norm.columns
