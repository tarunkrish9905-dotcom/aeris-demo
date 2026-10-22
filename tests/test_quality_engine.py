"""
Unit tests for the AERIS Data Quality Assessment Engine (DataQualityEngine).

Verifies metric calculation accuracy, component score explainability,
mathematical bounds [0.0, 1.0], raw data immutability, and provenance linkage.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from aeris.core.ingestion.csv_ingestor import ingest_csv
from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.quality.quality_engine import (
    DEFAULT_SCORING_WEIGHTS,
    DataQualityEngine,
    assess_data_quality,
)
from aeris.core.results.types import IngestionResult, ResultStatus
from aeris.core.validation.validators import validate_flight_data


def test_quality_clean_flight_perfect_score(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify clean_flight.csv achieves a 1.0 quality score with 100% completeness."""
    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    ingest_res = ingest_csv(clean_csv)
    val_res = validate_flight_data(ingest_res)

    quality_res = assess_data_quality(ingest_res, val_res)

    assert quality_res.status == ResultStatus.SUCCESS
    assert quality_res.quality_score == 1.0
    assert quality_res.metrics["completeness_ratio"] == 1.0
    assert quality_res.metrics["missing_cells"] == 0
    assert quality_res.metrics["invalid_timestamp_count"] == 0
    assert quality_res.metrics["duplicate_timestamp_count"] == 0
    assert quality_res.metrics["unordered_timestamp_count"] == 0
    assert quality_res.metrics["usable_row_count"] == 60
    assert quality_res.metrics["usable_row_ratio"] == 1.0
    assert quality_res.issues == []


def test_quality_missing_values_exact_calculation(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify missing_values.csv exact cell counts and completeness ratio calculations."""
    missing_csv = csv_fixtures_dir / "missing_values.csv"
    ingest_res = ingest_csv(missing_csv)
    quality_res = assess_data_quality(ingest_res)

    # In missing_values.csv:
    # 60 rows * 6 columns = 360 total cells
    # Missing: altitude(1), roll(2), airspeed(1), pitch(1), rpm(1) = 6 cells
    total_cells = 60 * 6
    expected_missing = 6
    expected_completeness = round((total_cells - expected_missing) / total_cells, 4)

    assert quality_res.metrics["total_cells"] == 360
    assert quality_res.metrics["missing_cells"] == expected_missing
    assert quality_res.metrics["completeness_ratio"] == expected_completeness

    # Per-column completeness spot check
    col_comp = quality_res.metrics["column_completeness"]
    assert col_comp["timestamp"] == 1.0
    assert col_comp["altitude_ft"] == round(59 / 60, 4)
    assert col_comp["roll_deg"] == round(58 / 60, 4)

    # Explainable score calculation:
    # s_comp = 354/360, all other dimensions = 1.0
    w = DEFAULT_SCORING_WEIGHTS
    expected_score = round(
        w["completeness"] * (354 / 360)
        + w["timestamp_validity"] * 1.0
        + w["timestamp_uniqueness"] * 1.0
        + w["timestamp_ordering"] * 1.0,
        4,
    )
    assert quality_res.quality_score == expected_score
    assert any(i.issue_code == "WARN_DATA_GAPS" for i in quality_res.issues)


def test_quality_invalid_timestamps_exact_calculation(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify invalid_timestamps.csv counts exactly 4 invalid timestamps and penalizes score."""
    invalid_csv = csv_fixtures_dir / "invalid_timestamps.csv"
    ingest_res = ingest_csv(invalid_csv)
    quality_res = assess_data_quality(ingest_res)

    assert quality_res.metrics["invalid_timestamp_count"] == 4
    assert quality_res.metrics["invalid_timestamp_rate"] == round(4 / 60, 4)

    comp_scores = quality_res.metrics["component_scores"]
    assert comp_scores["timestamp_validity"] == round(1.0 - (4 / 60), 4)
    assert comp_scores["completeness"] == round(
        1.0 - (1 / 360), 4
    )  # Row 50 timestamp is empty (NaN)

    assert quality_res.quality_score is not None
    assert quality_res.quality_score < 1.0
    assert any(i.issue_code == "WARN_TIMESTAMP_DROPOUTS" for i in quality_res.issues)


def test_quality_duplicate_timestamps_exact_calculation(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify duplicate_timestamps.csv counts exactly 3 duplicates and computes rate."""
    dup_csv = csv_fixtures_dir / "duplicate_timestamps.csv"
    ingest_res = ingest_csv(dup_csv)
    quality_res = assess_data_quality(ingest_res)

    assert quality_res.metrics["duplicate_timestamp_count"] == 3
    assert quality_res.metrics["duplicate_timestamp_rate"] == round(3 / 60, 4)

    comp_scores = quality_res.metrics["component_scores"]
    assert comp_scores["timestamp_uniqueness"] == round(1.0 - (3 / 60), 4)
    assert quality_res.quality_score is not None
    assert quality_res.quality_score < 1.0
    assert any(i.issue_code == "WARN_TIMESTAMP_DUPLICATES" for i in quality_res.issues)


def test_quality_unordered_timestamps_calculation(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify unordered_timestamps.csv detects non-monotonic transitions."""
    unordered_csv = csv_fixtures_dir / "unordered_timestamps.csv"
    ingest_res = ingest_csv(unordered_csv)
    quality_res = assess_data_quality(ingest_res)

    assert quality_res.metrics["unordered_timestamp_count"] > 0
    assert quality_res.metrics["component_scores"]["timestamp_ordering"] < 1.0
    assert any(i.issue_code == "WARN_TIMESTAMP_ORDERING" for i in quality_res.issues)


def test_quality_multiple_simultaneous_defects() -> None:
    """Verify multi-defect dataset compounding penalty across multiple dimensions."""
    data = {
        "timestamp": [
            "2026-10-06T10:00:00Z",
            "2026-10-06T10:00:00Z",  # Duplicate
            "INVALID_TS",  # Unparseable
            "2026-10-06T10:00:01Z",
        ],
        "val_a": [1.0, None, 3.0, 4.0],  # 1 missing
        "val_b": [10.0, 20.0, None, 40.0],  # 1 missing
    }
    df = pd.DataFrame(data)
    quality_res = assess_data_quality(df)

    assert quality_res.quality_score is not None
    assert 0.0 <= quality_res.quality_score <= 1.0
    # Compounded penalties should reduce score below 0.85
    assert quality_res.quality_score < 0.85

    issue_codes = {i.issue_code for i in quality_res.issues}
    assert "WARN_DATA_GAPS" in issue_codes
    assert "WARN_TIMESTAMP_DROPOUTS" in issue_codes
    assert "WARN_TIMESTAMP_DUPLICATES" in issue_codes


def test_quality_empty_and_no_data_graceful_handling(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify 0-byte file and empty DataFrames produce NOT_AVAILABLE without crashing."""
    # 1. Failed ingestion from malformed empty file
    empty_csv = csv_fixtures_dir / "malformed_empty.csv"
    ingest_res = ingest_csv(empty_csv)
    assert not ingest_res.is_successful

    res1 = assess_data_quality(ingest_res)
    assert res1.status == ResultStatus.NOT_AVAILABLE
    assert res1.quality_score is None

    # 2. IngestionResult with None data
    none_res = IngestionResult(status=ResultStatus.ERROR, data=None)
    res2 = assess_data_quality(none_res)
    assert res2.status == ResultStatus.NOT_AVAILABLE
    assert res2.quality_score is None

    # 3. 0-row empty DataFrame
    empty_df = pd.DataFrame()
    res3 = assess_data_quality(empty_df)
    assert res3.status == ResultStatus.NOT_AVAILABLE
    assert res3.quality_score is None


def test_quality_score_bounds_and_determinism(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify quality score is strictly bounded [0.0, 1.0] and bitwise deterministic."""
    for fixture_name in [
        "clean_flight.csv",
        "missing_values.csv",
        "invalid_timestamps.csv",
        "duplicate_timestamps.csv",
        "unordered_timestamps.csv",
    ]:
        path = csv_fixtures_dir / fixture_name
        ingest_res = ingest_csv(path)

        res_a = assess_data_quality(ingest_res)
        res_b = assess_data_quality(ingest_res)

        assert res_a.quality_score is not None
        assert 0.0 <= res_a.quality_score <= 1.0
        # Deterministic repeated scoring
        assert res_a.quality_score == res_b.quality_score
        assert res_a.metrics["completeness_ratio"] == res_b.metrics["completeness_ratio"]


def test_quality_raw_data_immutability(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify assessing quality never modifies, fills, or drops data in the raw DataFrame."""
    missing_csv = csv_fixtures_dir / "missing_values.csv"
    ingest_res = ingest_csv(missing_csv)
    assert ingest_res.data is not None

    df_before = ingest_res.data.copy(deep=True)
    assess_data_quality(ingest_res)

    pd.testing.assert_frame_equal(ingest_res.data, df_before)


def test_quality_provenance_linkage(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify provenance links upstream to both validation and ingestion parent records."""
    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    ingest_res = ingest_csv(clean_csv)
    assert ingest_res.provenance is not None

    val_res = validate_flight_data(ingest_res)
    assert val_res.provenance is not None

    quality_res = assess_data_quality(ingest_res, val_res)

    assert quality_res.provenance is not None
    assert quality_res.provenance.activity_name == "data_quality_assessment"
    assert quality_res.provenance.epistemic_level == EpistemicLevel.CALCULATED
    assert val_res.provenance.record_id in quality_res.provenance.parent_record_ids
    assert ingest_res.provenance.record_id in quality_res.provenance.parent_record_ids


def test_quality_weights_validation() -> None:
    """Verify engine raises ValueError if custom weights do not sum to 1.0."""
    with pytest.raises(ValueError, match=r"must sum to 1\.0"):
        DataQualityEngine(weights={"completeness": 0.5, "timestamp_validity": 0.2})
