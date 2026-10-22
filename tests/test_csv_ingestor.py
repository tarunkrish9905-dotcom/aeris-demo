"""
Unit tests for the AERIS CSV Ingestor (CSVIngestor and ingest_csv).

Verifies the Raw Data Principle: ingestion reads and represents data safely
without performing schema validation, timestamp checks, or value transformations.
"""

from __future__ import annotations

import pathlib

import pandas as pd

from aeris.core.ingestion.csv_ingestor import CSVIngestor, ingest_csv
from aeris.core.provenance.tracker import EpistemicLevel, compute_file_checksum
from aeris.core.results.types import ResultStatus


def test_ingest_clean_flight_success(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify clean_flight.csv ingests with SUCCESS, 60 rows, and 6 columns."""
    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    res = ingest_csv(clean_csv)

    assert res.status == ResultStatus.SUCCESS
    assert res.is_successful
    assert not res.has_errors
    assert res.row_count == 60
    assert len(res.column_names) == 6
    assert isinstance(res.data, pd.DataFrame)
    assert len(res.data) == 60
    assert res.issues == []


def test_ingest_clean_flight_preserves_raw_values(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify raw values are preserved exactly as parsed from source."""
    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    res = ingest_csv(clean_csv)

    assert res.data is not None
    # Spot-check initial synthetic values
    assert res.data.loc[0, "timestamp"] == "2026-10-06T10:00:00.000Z"
    assert res.data.loc[0, "altitude_ft"] == 5000.0
    assert res.data.loc[0, "airspeed_kts"] == 138.0


def test_ingest_missing_values_success(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify missing_values.csv ingests successfully and preserves NaNs without modification."""
    missing_csv = csv_fixtures_dir / "missing_values.csv"
    res = ingest_csv(missing_csv)

    # Missing values are a quality/validation concern, not an ingestion failure
    assert res.status == ResultStatus.SUCCESS
    assert res.row_count == 60
    assert res.data is not None

    # Verify raw missing values remain as NaN
    assert pd.isna(res.data.loc[5, "altitude_ft"])
    assert pd.isna(res.data.loc[12, "roll_deg"])
    assert pd.isna(res.data.loc[42, "engine_rpm"])


def test_ingest_invalid_timestamps_success(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify invalid_timestamps.csv ingests successfully (timestamp checks are out of scope)."""
    invalid_csv = csv_fixtures_dir / "invalid_timestamps.csv"
    res = ingest_csv(invalid_csv)

    assert res.status == ResultStatus.SUCCESS
    assert res.data is not None
    # Raw unparseable strings must be preserved intact
    assert res.data.loc[7, "timestamp"] == "INVALID_TIMESTAMP_STRING"
    assert res.data.loc[21, "timestamp"] == "2026-99-99T99:99:99.000Z"


def test_ingest_duplicate_timestamps_success(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify duplicate_timestamps.csv ingests all 60 rows without deduplicating."""
    dup_csv = csv_fixtures_dir / "duplicate_timestamps.csv"
    res = ingest_csv(dup_csv)

    assert res.status == ResultStatus.SUCCESS
    assert res.row_count == 60
    assert res.data is not None
    # Duplicate rows must remain in place
    assert res.data.loc[10, "timestamp"] == res.data.loc[11, "timestamp"]


def test_ingest_unordered_timestamps_success(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify unordered_timestamps.csv ingests without reordering chronological sequence."""
    unordered_csv = csv_fixtures_dir / "unordered_timestamps.csv"
    res = ingest_csv(unordered_csv)

    assert res.status == ResultStatus.SUCCESS
    assert res.row_count == 60
    assert res.data is not None
    # Row sequence preserved (swap at 14 & 15 is not rectified by ingestor)
    ts_14 = res.data.loc[14, "timestamp"]
    ts_15 = res.data.loc[15, "timestamp"]
    assert ts_14 > ts_15  # Non-chronological delta remains


def test_ingest_malformed_empty_file_failure(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify empty/0-byte CSV produces structured ERROR result with ERR_EMPTY_FILE."""
    empty_csv = csv_fixtures_dir / "malformed_empty.csv"
    res = ingest_csv(empty_csv)

    assert res.status == ResultStatus.ERROR
    assert res.has_errors
    assert res.data is None
    assert any(i.issue_code == "ERR_EMPTY_FILE" for i in res.issues)


def test_ingest_nonexistent_file_failure(tmp_path: pathlib.Path) -> None:
    """Verify nonexistent path produces structured ERROR result with ERR_FILE_NOT_FOUND."""
    fake_path = tmp_path / "does_not_exist.csv"
    res = ingest_csv(fake_path)

    assert res.status == ResultStatus.ERROR
    assert res.has_errors
    assert res.data is None
    assert any(i.issue_code == "ERR_FILE_NOT_FOUND" for i in res.issues)


def test_ingest_directory_path_failure(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify passing a directory path produces structured ERROR with ERR_PATH_IS_DIRECTORY."""
    res = ingest_csv(csv_fixtures_dir)

    assert res.status == ResultStatus.ERROR
    assert res.has_errors
    assert res.data is None
    assert any(i.issue_code == "ERR_PATH_IS_DIRECTORY" for i in res.issues)


def test_ingest_provenance_attachment(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify successful ingestion attaches provenance with OBSERVED epistemic level."""
    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    res = ingest_csv(clean_csv)

    assert res.provenance is not None
    assert res.provenance.activity_name == "csv_ingestion"
    assert res.provenance.epistemic_level == EpistemicLevel.OBSERVED
    assert res.provenance.source_uri == str(clean_csv.resolve())
    assert res.provenance.source_checksum is not None
    assert len(res.provenance.source_checksum) == 64


def test_ingest_checksum_determinism(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify repeated ingestion produces identical source checksums matching raw file hash."""
    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    res1 = ingest_csv(clean_csv)
    res2 = ingest_csv(clean_csv)

    expected_hash = compute_file_checksum(clean_csv)
    assert res1.provenance is not None
    assert res2.provenance is not None
    assert res1.provenance.source_checksum == expected_hash
    assert res2.provenance.source_checksum == expected_hash


def test_csv_ingestor_class_instance_and_kwargs(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify CSVIngestor class can be instantiated and accepts kwargs passed to read_csv."""
    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    ingestor = CSVIngestor()
    res = ingestor.ingest(clean_csv, nrows=10)

    assert res.status == ResultStatus.SUCCESS
    assert res.row_count == 10
    assert len(res.data) == 10
