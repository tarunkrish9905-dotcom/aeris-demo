"""
Unit tests for the AERIS Schema & Timestamp Validator (FlightDataValidator).

Verifies structural schema verification, timestamp parseability, duplicate
detection, chronological monotonicity, and strict raw data immutability.
"""

from __future__ import annotations

import pathlib

import pandas as pd

from aeris.core.ingestion.csv_ingestor import ingest_csv
from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.results.types import IngestionResult, ResultStatus
from aeris.core.validation.validators import FlightDataValidator, validate_flight_data


def test_validate_clean_flight_success(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify clean_flight.csv passes validation with zero issues and SUCCESS status."""
    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    ingest_res = ingest_csv(clean_csv)
    assert ingest_res.is_successful

    val_res = validate_flight_data(ingest_res)

    assert val_res.is_valid
    assert val_res.status == ResultStatus.SUCCESS
    assert val_res.violation_count == 0
    assert val_res.issues == []
    assert val_res.validated_row_count == 60


def test_validate_alternate_headers_success() -> None:
    """Verify alternate canonical headers pass validation without legacy signal requirements."""
    alternate_data = {
        "timestamp": [
            "2026-10-06T10:00:00Z",
            "2026-10-06T10:00:01Z",
            "2026-10-06T10:00:02Z",
        ],
        "pressure_altitude": [5000.0, 5010.0, 5020.0],
        "indicated_airspeed": [120.0, 121.5, 122.0],
        "pitch_attitude": [1.5, 1.8, 1.6],
        "bank_angle": [0.0, 0.5, 0.2],
    }
    df = pd.DataFrame(alternate_data)
    val_res = validate_flight_data(df)

    assert val_res.is_valid
    assert val_res.status == ResultStatus.SUCCESS
    assert val_res.violation_count == 0
    assert val_res.issues == []
    assert val_res.validated_row_count == 3


def test_validate_electric_dataset_without_engine_rpm_success(
    csv_fixtures_dir: pathlib.Path,
) -> None:
    """Verify dataset without engine_rpm passes raw validation under default settings."""
    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    ingest_res = ingest_csv(clean_csv)
    assert ingest_res.data is not None

    # Electric / glider aircraft without engine_rpm
    df_no_engine = ingest_res.data.drop(columns=["engine_rpm"])
    val_res = validate_flight_data(df_no_engine)

    assert val_res.is_valid
    assert val_res.status == ResultStatus.SUCCESS
    assert val_res.violation_count == 0
    assert val_res.issues == []


def test_validate_explicit_required_columns_enforcement(
    csv_fixtures_dir: pathlib.Path,
) -> None:
    """Verify explicitly requested columns produce ERR_MISSING_REQUIRED_COLUMN when missing."""
    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    ingest_res = ingest_csv(clean_csv)
    assert ingest_res.data is not None

    df_missing_col = ingest_res.data.drop(columns=["engine_rpm"])
    val_res = validate_flight_data(
        df_missing_col, required_columns=["timestamp", "engine_rpm"]
    )

    assert not val_res.is_valid
    assert val_res.status == ResultStatus.ERROR
    assert any(
        i.issue_code == "ERR_MISSING_REQUIRED_COLUMN" and i.field_name == "engine_rpm"
        for i in val_res.issues
    )


def test_validate_unexpected_column_policies(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify handling of unexpected columns under permissive vs strict schema policies."""
    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    ingest_res = ingest_csv(clean_csv)
    assert ingest_res.data is not None

    # Add extra telemetry parameter
    df_extra = ingest_res.data.copy()
    df_extra["fuel_flow_gph"] = 12.5
    known_cols = list(ingest_res.data.columns)

    # Policy 1: Permissive (allow_extra_columns=True, reports WARNING)
    validator_permissive = FlightDataValidator(
        required_columns=known_cols,
        allow_extra_columns=True,
        report_extra_columns=True,
    )
    res_permissive = validator_permissive.validate(df_extra)

    assert res_permissive.is_valid  # Still valid despite advisory warning
    assert res_permissive.status == ResultStatus.WARNING
    assert any(
        i.issue_code == "WARN_UNEXPECTED_COLUMN" and i.field_name == "fuel_flow_gph"
        for i in res_permissive.issues
    )

    # Policy 2: Strict (allow_extra_columns=False, reports ERROR)
    validator_strict = FlightDataValidator(
        required_columns=known_cols,
        allow_extra_columns=False,
    )
    res_strict = validator_strict.validate(df_extra)

    assert not res_strict.is_valid
    assert res_strict.status == ResultStatus.ERROR
    assert any(
        i.issue_code == "ERR_UNEXPECTED_COLUMN" and i.field_name == "fuel_flow_gph"
        for i in res_strict.issues
    )


def test_validate_missing_timestamp_column(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify missing timestamp column produces ERR_TIMESTAMP_COLUMN_MISSING."""
    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    ingest_res = ingest_csv(clean_csv)
    assert ingest_res.data is not None

    df_no_ts = ingest_res.data.drop(columns=["timestamp"])
    val_res = validate_flight_data(df_no_ts)

    assert not val_res.is_valid
    assert val_res.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_TIMESTAMP_COLUMN_MISSING" for i in val_res.issues)


def test_validate_invalid_timestamps(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify invalid_timestamps.csv is flagged with ERR_TIMESTAMP_UNPARSEABLE."""
    invalid_csv = csv_fixtures_dir / "invalid_timestamps.csv"
    ingest_res = ingest_csv(invalid_csv)
    assert ingest_res.is_successful

    val_res = validate_flight_data(ingest_res)

    assert not val_res.is_valid
    assert val_res.status == ResultStatus.ERROR

    unparseable_issues = [i for i in val_res.issues if i.issue_code == "ERR_TIMESTAMP_UNPARSEABLE"]
    assert len(unparseable_issues) >= 4

    # Verify specific defect row locations are identified
    flagged_locations = {i.location for i in unparseable_issues}
    assert 7 in flagged_locations
    assert 21 in flagged_locations
    assert 38 in flagged_locations


def test_validate_duplicate_timestamps(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify duplicate_timestamps.csv is flagged with WARN_DUPLICATE_TIMESTAMPS."""
    dup_csv = csv_fixtures_dir / "duplicate_timestamps.csv"
    ingest_res = ingest_csv(dup_csv)
    assert ingest_res.is_successful

    val_res = validate_flight_data(ingest_res)

    assert val_res.is_valid  # Warnings do not invalidate dataset
    assert val_res.status == ResultStatus.WARNING

    dup_issues = [i for i in val_res.issues if i.issue_code == "WARN_DUPLICATE_TIMESTAMPS"]
    assert len(dup_issues) == 1
    assert dup_issues[0].context["duplicate_count"] >= 3


def test_validate_unordered_timestamps(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify unordered_timestamps.csv flags non-monotonic timestamp ordering."""
    unordered_csv = csv_fixtures_dir / "unordered_timestamps.csv"
    ingest_res = ingest_csv(unordered_csv)
    assert ingest_res.is_successful

    val_res = validate_flight_data(ingest_res)

    assert val_res.is_valid
    assert val_res.status == ResultStatus.WARNING

    unordered_issues = [i for i in val_res.issues if i.issue_code == "WARN_UNORDERED_TIMESTAMPS"]
    assert len(unordered_issues) == 1
    assert unordered_issues[0].field_name == "timestamp"


def test_validate_multiple_defects_concurrently() -> None:
    """Verify validator reports multiple distinct defect categories concurrently."""
    # Synthesize DataFrame with: missing column (from required),
    # unexpected column, and unparseable timestamp
    data = {
        "timestamp": ["2026-10-06T10:00:00Z", "CORRUPTED_TS"],
        "altitude_ft": [5000.0, 5010.0],
        # airspeed_kts missing!
        "pitch_deg": [1.0, 1.2],
        "roll_deg": [0.0, 0.1],
        "engine_rpm": [2400.0, 2410.0],
        "unknown_extra": [99, 100],  # unexpected
    }
    df = pd.DataFrame(data)

    val_res = validate_flight_data(
        df,
        required_columns=[
            "timestamp",
            "altitude_ft",
            "airspeed_kts",
            "pitch_deg",
            "roll_deg",
            "engine_rpm",
        ],
        report_extra_columns=True,
    )

    assert not val_res.is_valid
    assert val_res.status == ResultStatus.ERROR

    codes = {i.issue_code for i in val_res.issues}
    assert "ERR_MISSING_REQUIRED_COLUMN" in codes
    assert "ERR_TIMESTAMP_UNPARSEABLE" in codes
    assert "WARN_UNEXPECTED_COLUMN" in codes


def test_validate_blank_or_whitespace_timestamps() -> None:
    """Verify empty or whitespace timestamp values produce ERR_TIMESTAMP_UNPARSEABLE."""
    data = {
        "timestamp": ["2026-10-06T10:00:00Z", "   ", None, ""],
        "altitude_ft": [1000.0, 1010.0, 1020.0, 1030.0],
    }
    df = pd.DataFrame(data)
    val_res = validate_flight_data(df)

    assert not val_res.is_valid
    assert val_res.status == ResultStatus.ERROR

    unparseable_issues = [i for i in val_res.issues if i.issue_code == "ERR_TIMESTAMP_UNPARSEABLE"]
    assert len(unparseable_issues) == 3


def test_raw_data_immutability(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify that validation does not alter, sort, deduplicate, or mutate raw DataFrame."""
    unordered_csv = csv_fixtures_dir / "unordered_timestamps.csv"
    ingest_res = ingest_csv(unordered_csv)
    assert ingest_res.data is not None

    # Snapshot before validation
    df_before = ingest_res.data.copy(deep=True)

    # Perform validation
    validate_flight_data(ingest_res)

    # Bitwise frame equality verification
    pd.testing.assert_frame_equal(ingest_res.data, df_before)


def test_raw_data_immutability_alternate_headers() -> None:
    """Verify raw DataFrame with alternate canonical headers is strictly unmutated."""
    data = {
        "timestamp": ["2026-10-06T10:00:01Z", "2026-10-06T10:00:00Z"],
        "pressure_altitude": [5000.0, 5010.0],
        "indicated_airspeed": [100.0, 102.0],
    }
    df = pd.DataFrame(data)
    df_before = df.copy(deep=True)

    validate_flight_data(df)
    pd.testing.assert_frame_equal(df, df_before)


def test_provenance_linkage(csv_fixtures_dir: pathlib.Path) -> None:
    """Verify validation preserves upstream provenance parent lineage."""
    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    ingest_res = ingest_csv(clean_csv)
    assert ingest_res.provenance is not None

    val_res = validate_flight_data(ingest_res)

    assert val_res.provenance is not None
    assert val_res.provenance.activity_name == "schema_and_timestamp_validation"
    assert val_res.provenance.epistemic_level == EpistemicLevel.CALCULATED
    # Provenance parent linkage must point to ingestion record ID
    assert val_res.provenance.parent_record_ids == [ingest_res.provenance.record_id]


def test_validate_empty_dataset_handling() -> None:
    """Verify validator handles empty IngestionResult and 0-row DataFrames gracefully."""
    # Case 1: IngestionResult with None data
    empty_res = IngestionResult(status=ResultStatus.ERROR, data=None, source_path="empty.csv")
    val_res_none = validate_flight_data(empty_res)
    assert not val_res_none.is_valid
    assert val_res_none.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_NO_DATA_TO_VALIDATE" for i in val_res_none.issues)

    # Case 2: IngestionResult with 0-row DataFrame
    empty_df_res = IngestionResult(
        status=ResultStatus.SUCCESS,
        data=pd.DataFrame(),
        source_path="empty.csv",
    )
    val_res_df = validate_flight_data(empty_df_res)
    assert not val_res_df.is_valid
    assert val_res_df.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_NO_DATA_TO_VALIDATE" for i in val_res_df.issues)

    # Case 3: Direct 0-row empty DataFrame
    val_res_direct = validate_flight_data(pd.DataFrame())
    assert not val_res_direct.is_valid
    assert val_res_direct.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_NO_DATA_TO_VALIDATE" for i in val_res_direct.issues)

    # Case 4: Direct 0-row DataFrame with column definitions
    empty_with_cols = pd.DataFrame(columns=["timestamp", "pressure_altitude", "indicated_airspeed"])
    val_res_cols = validate_flight_data(empty_with_cols)
    assert not val_res_cols.is_valid
    assert val_res_cols.status == ResultStatus.ERROR
    assert any(i.issue_code == "ERR_NO_DATA_TO_VALIDATE" for i in val_res_cols.issues)
