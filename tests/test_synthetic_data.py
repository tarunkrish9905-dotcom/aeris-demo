"""
Tests for the synthetic flight-test dataset generator and generated CSV fixtures.

Verifies that the generated datasets exist, match the expected schema, and
genuinely exhibit their intended data characteristics and defect conditions.
"""

from __future__ import annotations

import pathlib
from datetime import datetime

import pandas as pd
import pytest
from scripts.generate_synthetic_data import (
    STANDARD_COLUMNS,
    generate_all_datasets,
    generate_clean_dataset,
)

# ---------------------------------------------------------------------------
# Fixture file verification tests
# ---------------------------------------------------------------------------


def test_clean_flight_dataset_characteristics(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify clean_flight.csv exists, has valid schema, ordered timestamps, and no defects."""
    csv_path = csv_fixtures_dir / "clean_flight.csv"
    assert csv_path.is_file(), f"Expected fixture missing: {csv_path}"
    assert csv_path.stat().st_size > 0

    df = pd.read_csv(csv_path)

    # 1. Schema check
    assert list(df.columns) == STANDARD_COLUMNS
    assert len(df) == 60

    # 2. No missing values
    assert df.isna().sum().sum() == 0

    # 3. Valid timestamps and strictly ordered (1 Hz)
    parsed_dts: list[datetime] = []
    for ts in df["timestamp"]:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        parsed_dts.append(dt)

    for i in range(len(parsed_dts) - 1):
        delta = (parsed_dts[i + 1] - parsed_dts[i]).total_seconds()
        assert delta == 1.0, f"Expected 1-second step at index {i}, got {delta}"

    # 4. No duplicate timestamps
    assert df["timestamp"].duplicated().sum() == 0

    # 5. Realistic synthetic numeric ranges
    assert (df["altitude_ft"] >= 4500.0).all() and (df["altitude_ft"] <= 6500.0).all()
    assert (df["airspeed_kts"] >= 120.0).all() and (df["airspeed_kts"] <= 150.0).all()
    assert (df["engine_rpm"] >= 2300.0).all() and (df["engine_rpm"] <= 2600.0).all()


def test_missing_values_dataset_characteristics(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify missing_values.csv contains genuine missing values while preserving schema."""
    csv_path = csv_fixtures_dir / "missing_values.csv"
    assert csv_path.is_file()

    df = pd.read_csv(csv_path)

    # Schema preserved
    assert list(df.columns) == STANDARD_COLUMNS
    assert len(df) == 60

    # Timestamp column has zero missing values
    assert df["timestamp"].isna().sum() == 0

    # Must contain missing values
    total_missing = df.isna().sum().sum()
    assert total_missing > 0, "missing_values.csv must contain missing entries"

    # Specific planned missing values exist
    assert pd.isna(df.loc[5, "altitude_ft"])
    assert pd.isna(df.loc[12, "roll_deg"])
    assert pd.isna(df.loc[13, "roll_deg"])
    assert pd.isna(df.loc[25, "airspeed_kts"])
    assert pd.isna(df.loc[25, "pitch_deg"])
    assert pd.isna(df.loc[42, "engine_rpm"])

    # Vast majority of values remain intact for graceful degradation testing
    assert df.notna().sum().sum() > 300


def test_invalid_timestamps_dataset_characteristics(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify invalid_timestamps.csv contains genuine unparseable timestamps at traceable rows."""
    csv_path = csv_fixtures_dir / "invalid_timestamps.csv"
    assert csv_path.is_file()

    # Read as raw string values to inspect unparseable strings
    df = pd.read_csv(csv_path, dtype=str)

    assert list(df.columns) == STANDARD_COLUMNS
    assert len(df) == 60

    # Check for traceable invalid entries
    assert df.loc[7, "timestamp"] == "INVALID_TIMESTAMP_STRING"
    assert df.loc[21, "timestamp"] == "2026-99-99T99:99:99.000Z"
    assert df.loc[38, "timestamp"] == "10/06/2026 25:70:00"

    # Count unparseable timestamps
    unparseable_count = 0
    for ts in df["timestamp"]:
        if not ts or pd.isna(ts):
            unparseable_count += 1
            continue
        try:
            datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            unparseable_count += 1

    assert unparseable_count >= 4, (
        f"Expected at least 4 unparseable timestamps, got {unparseable_count}"
    )


def test_duplicate_timestamps_dataset_characteristics(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify duplicate_timestamps.csv contains duplicate timestamp entries while retaining data."""
    csv_path = csv_fixtures_dir / "duplicate_timestamps.csv"
    assert csv_path.is_file()

    df = pd.read_csv(csv_path)

    assert list(df.columns) == STANDARD_COLUMNS
    assert len(df) == 60

    duplicate_mask = df["timestamp"].duplicated(keep=False)
    assert duplicate_mask.sum() > 0, "Expected duplicate timestamps in dataset"
    assert df["timestamp"].nunique() < len(df)

    # Traceable duplicates
    assert df.loc[10, "timestamp"] == df.loc[11, "timestamp"]
    assert df.loc[32, "timestamp"] == df.loc[33, "timestamp"]
    assert df.loc[32, "timestamp"] == df.loc[34, "timestamp"]


def test_unordered_timestamps_dataset_characteristics(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify unordered_timestamps.csv has valid timestamps that violate chronological ordering."""
    csv_path = csv_fixtures_dir / "unordered_timestamps.csv"
    assert csv_path.is_file()

    df = pd.read_csv(csv_path)

    assert list(df.columns) == STANDARD_COLUMNS
    assert len(df) == 60

    # All timestamps parse cleanly
    parsed_dts = [datetime.fromisoformat(str(ts).replace("Z", "+00:00")) for ts in df["timestamp"]]

    # Check that chronological monotonicity is broken
    negative_deltas: list[int] = []
    for i in range(len(parsed_dts) - 1):
        if (parsed_dts[i + 1] - parsed_dts[i]).total_seconds() <= 0:
            negative_deltas.append(i)

    assert len(negative_deltas) > 0, "Expected at least one non-chronological step"
    # Swap at row 14 & 15 creates a negative delta at row 14
    assert 14 in negative_deltas


def test_malformed_empty_dataset_characteristics(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify malformed_empty.csv is a 0-byte file that raises EmptyDataError when read."""
    csv_path = csv_fixtures_dir / "malformed_empty.csv"
    assert csv_path.is_file()
    assert csv_path.stat().st_size == 0

    with pytest.raises(pd.errors.EmptyDataError):
        pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Generator module unit tests
# ---------------------------------------------------------------------------


def test_clean_dataset_generator_determinism() -> None:
    """Verify that generating clean datasets produces identical, deterministic results."""
    df1 = generate_clean_dataset()
    df2 = generate_clean_dataset()
    pd.testing.assert_frame_equal(df1, df2)


def test_generator_idempotence_and_all_datasets(tmp_path: pathlib.Path) -> None:
    """Verify generate_all_datasets generates all 6 expected files into any target directory."""
    generated = generate_all_datasets(tmp_path)

    expected_keys = {
        "clean_flight",
        "missing_values",
        "invalid_timestamps",
        "duplicate_timestamps",
        "unordered_timestamps",
        "malformed_empty",
    }
    assert set(generated.keys()) == expected_keys

    for key, path in generated.items():
        assert path.is_file()
        if key == "malformed_empty":
            assert path.stat().st_size == 0
        else:
            assert path.stat().st_size > 0
