"""Tests for AERIS Normalization Engine."""

import pathlib

import pandas as pd

from aeris.core.ingestion.csv_ingestor import ingest_csv
from aeris.core.normalization.engine import (
    NormalizationRequest,
    normalize_flight_data,
)
from aeris.core.results.types import ResultStatus


def test_normalization_clean_rename_and_types(csv_fixtures_dir: pathlib.Path) -> None:
    csv_path = csv_fixtures_dir / "clean_flight.csv"
    ingest_res = ingest_csv(csv_path)

    req = NormalizationRequest()
    req.rename_column("airspeed_kts", "airspeed")
    req.convert_type("altitude_ft", float)
    req.standardize_timestamp("timestamp")

    res = normalize_flight_data(ingest_res, req)

    assert res.status == ResultStatus.SUCCESS
    assert res.data is not None

    # Input DataFrame immutability check
    df_raw = ingest_res.data
    assert "airspeed_kts" in df_raw.columns
    assert "airspeed" not in df_raw.columns

    df_norm = res.data
    assert "airspeed" in df_norm.columns
    assert "airspeed_kts" not in df_norm.columns

    # Assert type conversion
    assert pd.api.types.is_float_dtype(df_norm["altitude_ft"])

    # Assert timestamp standardization
    assert pd.api.types.is_datetime64_any_dtype(df_norm["timestamp"])

    # Metadata
    assert len(res.transformations) == 3
    assert res.input_row_count == 60
    assert res.output_row_count == 60
    assert "timestamp" in res.input_columns

    # Provenance
    assert res.provenance is not None
    assert ingest_res.provenance is not None
    assert ingest_res.provenance.record_id in res.provenance.parent_record_ids


def test_normalization_missing_source_column(csv_fixtures_dir: pathlib.Path) -> None:
    csv_path = csv_fixtures_dir / "clean_flight.csv"
    ingest_res = ingest_csv(csv_path)

    req = NormalizationRequest()
    req.rename_column("nonexistent_col", "new_col")

    res = normalize_flight_data(ingest_res, req)

    assert res.status == ResultStatus.ERROR
    assert res.data is None
    assert any(i.issue_code == "ERR_MISSING_SOURCE_COLUMN" for i in res.issues)


def test_normalization_invalid_numeric_conversion(csv_fixtures_dir: pathlib.Path) -> None:
    # We use a string value that cannot be coerced to float
    df = pd.DataFrame({"bad_col": ["a", "b", "c"]})

    req = NormalizationRequest()
    req.convert_type("bad_col", float)

    res = normalize_flight_data(df, req)

    assert res.status == ResultStatus.ERROR
    assert res.data is None
    assert any(i.issue_code == "ERR_TYPE_CONVERSION_FAILED" for i in res.issues)


def test_normalization_invalid_timestamp_conversion(csv_fixtures_dir: pathlib.Path) -> None:
    df = pd.DataFrame({"ts": ["NOT_A_TIMESTAMP"]})

    req = NormalizationRequest()
    req.standardize_timestamp("ts")

    res = normalize_flight_data(df, req)

    assert res.status == ResultStatus.ERROR
    assert res.data is None
    assert any(i.issue_code == "ERR_TIMESTAMP_CONVERSION_FAILED" for i in res.issues)


def test_normalization_empty_input(csv_fixtures_dir: pathlib.Path) -> None:
    df = pd.DataFrame()
    req = NormalizationRequest()

    res = normalize_flight_data(df, req)

    assert res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_EMPTY_DATA" for i in res.issues)


def test_normalization_immutability(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify input DataFrame is truly isolated."""
    csv_path = csv_fixtures_dir / "clean_flight.csv"
    df_raw = pd.read_csv(csv_path)

    # Ensure it's not sorted or reordered
    original_order = list(df_raw.index)
    original_timestamp = df_raw.iloc[0]["timestamp"]

    req = NormalizationRequest()
    req.rename_column("altitude_ft", "alt")
    req.standardize_timestamp("timestamp")

    res = normalize_flight_data(df_raw, req)
    assert res.status == ResultStatus.SUCCESS

    df_norm = res.data
    assert df_norm is not None

    # Immutability
    assert "altitude_ft" in df_raw.columns
    assert list(df_raw.index) == original_order
    assert df_raw.iloc[0]["timestamp"] == original_timestamp

    assert "alt" in df_norm.columns
    # Check rows didn't change
    assert len(df_norm) == len(df_raw)
