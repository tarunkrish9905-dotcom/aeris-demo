"""
AERIS Schema and Timestamp Validator.

Evaluates tabular flight-test data for structural schema conformance,
timestamp parseability, duplicate timestamps, and chronological monotonicity.
Operates downstream of the CSV Ingestor and adheres strictly to the Raw Data
Immutability principle: validates without modifying, sorting, deduplicating,
or imputing values in the ingested DataFrame.

Epistemic status of validation output: CALCULATED.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from aeris.core.provenance.tracker import (
    EpistemicLevel,
    create_provenance,
)
from aeris.core.results.types import (
    IngestionResult,
    Issue,
    IssueSeverity,
    ResultStatus,
    ValidationResult,
)

DEFAULT_REQUIRED_COLUMNS: list[str] = [
    "timestamp",
]


class FlightDataValidator:
    """Validates schema conformance and timestamp integrity of flight-test data.

    Evaluates:
    1. Schema presence (detecting missing required columns)
    2. Unexpected column presence (according to schema policy)
    3. Timestamp column existence and parseability
    4. Duplicate timestamp detection
    5. Chronological monotonicity (ordering)
    """

    def __init__(
        self,
        required_columns: list[str] | None = None,
        timestamp_column: str = "timestamp",
        allow_extra_columns: bool = True,
        report_extra_columns: bool = False,
    ) -> None:
        """Initialize the validator with configured schema rules.

        Parameters
        ----------
        required_columns : list[str] | None
            List of columns that must be present in the dataset. Defaults to
            [timestamp_column] (or DEFAULT_REQUIRED_COLUMNS) if None.
        timestamp_column : str
            Name of the column containing chronological timestamps (default: 'timestamp').
        allow_extra_columns : bool
            If False, unexpected columns produce an ERROR rather than WARNING.
        report_extra_columns : bool
            If True, columns not in required_columns are flagged as issues.
        """
        self.timestamp_column = timestamp_column
        self.required_columns = (
            list(required_columns)
            if required_columns is not None
            else [timestamp_column]
        )
        self.allow_extra_columns = allow_extra_columns
        self.report_extra_columns = report_extra_columns

    def validate(
        self,
        data: pd.DataFrame | IngestionResult,
        *,
        source_uri: str | None = None,
    ) -> ValidationResult:
        """Execute validation against an IngestionResult or raw DataFrame.

        Parameters
        ----------
        data : pd.DataFrame | IngestionResult
            The data to validate. If an IngestionResult is provided, its data
            and provenance are extracted automatically.
        source_uri : str | None
            Optional source identifier for provenance if data is a raw DataFrame.

        Returns
        -------
        ValidationResult
            Detailed validation report containing pass/fail status, violation
            counts, structured issues, and provenance lineage.
        """
        # 1. Extract DataFrame and upstream provenance
        df: pd.DataFrame
        parent_record_ids: list[str] = []
        resolved_source_uri = source_uri or "in-memory-dataframe"

        if isinstance(data, IngestionResult):
            if data.provenance is not None:
                parent_record_ids.append(data.provenance.record_id)
                resolved_source_uri = data.provenance.source_uri
            elif data.source_path:
                resolved_source_uri = data.source_path

            if data.data is None or not isinstance(data.data, pd.DataFrame):
                # Ingestion result had no parseable data
                issue = Issue(
                    issue_code="ERR_NO_DATA_TO_VALIDATE",
                    severity=IssueSeverity.ERROR,
                    message="IngestionResult contains no tabular data to validate",
                    location=resolved_source_uri,
                )
                prov = create_provenance(
                    source_uri=resolved_source_uri,
                    activity_name="schema_and_timestamp_validation",
                    epistemic_level=EpistemicLevel.CALCULATED,
                    parent_record_ids=parent_record_ids,
                )
                return ValidationResult(
                    status=ResultStatus.ERROR,
                    is_valid=False,
                    validated_row_count=0,
                    violation_count=1,
                    validated_columns=[],
                    issues=[issue],
                    provenance=prov,
                )
            df = data.data
        else:
            df = data

        if df.empty or len(df) == 0:
            issue = Issue(
                issue_code="ERR_NO_DATA_TO_VALIDATE",
                severity=IssueSeverity.ERROR,
                message="Dataset contains no tabular rows to validate",
                location=resolved_source_uri,
            )
            prov = create_provenance(
                source_uri=resolved_source_uri,
                activity_name="schema_and_timestamp_validation",
                epistemic_level=EpistemicLevel.CALCULATED,
                parent_record_ids=parent_record_ids,
                parameters={
                    "required_columns": self.required_columns,
                    "timestamp_column": self.timestamp_column,
                    "allow_extra_columns": self.allow_extra_columns,
                },
            )
            return ValidationResult(
                status=ResultStatus.ERROR,
                is_valid=False,
                validated_row_count=0,
                violation_count=1,
                validated_columns=list(df.columns),
                issues=[issue],
                provenance=prov,
                metadata={
                    "has_errors": True,
                    "has_warnings": False,
                    "error_count": 1,
                    "warning_count": 0,
                },
            )

        issues: list[Issue] = []
        actual_columns = list(df.columns)

        # -------------------------------------------------------------------
        # 2. Schema Validation
        # -------------------------------------------------------------------
        # Check for missing required columns
        for col in self.required_columns:
            if col not in actual_columns:
                issues.append(
                    Issue(
                        issue_code="ERR_MISSING_REQUIRED_COLUMN",
                        severity=IssueSeverity.ERROR,
                        message=f"Required column '{col}' is missing from dataset",
                        field_name=col,
                    )
                )

        # Check for unexpected columns
        if self.report_extra_columns or not self.allow_extra_columns:
            for col in actual_columns:
                if col not in self.required_columns:
                    severity = (
                        IssueSeverity.WARNING if self.allow_extra_columns else IssueSeverity.ERROR
                    )
                    code = (
                        "WARN_UNEXPECTED_COLUMN"
                        if self.allow_extra_columns
                        else "ERR_UNEXPECTED_COLUMN"
                    )
                    issues.append(
                        Issue(
                            issue_code=code,
                            severity=severity,
                            message=f"Unexpected column '{col}' found in dataset",
                            field_name=col,
                        )
                    )

        # -------------------------------------------------------------------
        # 3. Timestamp Validation (Existence, Parseability, Monotonicity, Duplicates)
        # -------------------------------------------------------------------
        if self.timestamp_column not in actual_columns:
            issues.append(
                Issue(
                    issue_code="ERR_TIMESTAMP_COLUMN_MISSING",
                    severity=IssueSeverity.ERROR,
                    message=(
                        f"Configured timestamp column '{self.timestamp_column}' "
                        f"is not present in the dataset"
                    ),
                    field_name=self.timestamp_column,
                )
            )
        else:
            ts_series = df[self.timestamp_column]

            # 3a. Parseability Check (on temporary representation without modifying df)
            parsed_dts: list[datetime | None] = []
            for idx, val in ts_series.items():
                loc: str | int = idx if isinstance(idx, (int, str)) else str(idx)
                if pd.isna(val) or val == "" or (isinstance(val, str) and not val.strip()):
                    issues.append(
                        Issue(
                            issue_code="ERR_TIMESTAMP_UNPARSEABLE",
                            severity=IssueSeverity.ERROR,
                            message="Empty or null timestamp encountered",
                            field_name=self.timestamp_column,
                            location=loc,
                            context={"raw_value": str(val)},
                        )
                    )
                    parsed_dts.append(None)
                    continue

                if isinstance(val, (datetime, pd.Timestamp)):
                    dt = val.to_pydatetime() if hasattr(val, "to_pydatetime") else val
                    parsed_dts.append(dt)
                    continue

                try:
                    dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                    parsed_dts.append(dt)
                except (ValueError, TypeError) as exc:
                    issues.append(
                        Issue(
                            issue_code="ERR_TIMESTAMP_UNPARSEABLE",
                            severity=IssueSeverity.ERROR,
                            message=f"Unparseable timestamp value '{val}': {exc}",
                            field_name=self.timestamp_column,
                            location=loc,
                            context={"raw_value": str(val)},
                        )
                    )
                    parsed_dts.append(None)

            # 3b. Duplicate Timestamps Detection (reported without removing duplicates)
            dup_mask = ts_series.duplicated(keep=False)
            if dup_mask.any():
                dup_count = int(ts_series.duplicated().sum())
                dup_indices = list(df.index[dup_mask])
                issues.append(
                    Issue(
                        issue_code="WARN_DUPLICATE_TIMESTAMPS",
                        severity=IssueSeverity.WARNING,
                        message=(
                            f"Detected {dup_count} duplicate timestamp rows "
                            f"across {len(dup_indices)} row instances"
                        ),
                        field_name=self.timestamp_column,
                        location=dup_indices[0] if dup_indices else None,
                        context={
                            "duplicate_count": dup_count,
                            "affected_rows": [int(i) for i in dup_indices[:50]],
                        },
                    )
                )

            # 3c. Chronological Monotonicity Check (detects backwards time steps)
            valid_pairs: list[tuple[Any, datetime]] = [
                (idx, dt) for idx, dt in zip(df.index, parsed_dts, strict=False) if dt is not None
            ]
            non_monotonic_rows: list[int] = []
            for k in range(len(valid_pairs) - 1):
                _idx_prev, dt_prev = valid_pairs[k]
                idx_curr, dt_curr = valid_pairs[k + 1]
                delta_sec = (dt_curr - dt_prev).total_seconds()
                # Negative delta indicates non-chronological backwards step
                if delta_sec < 0:
                    non_monotonic_rows.append(int(idx_curr))

            if non_monotonic_rows:
                issues.append(
                    Issue(
                        issue_code="WARN_UNORDERED_TIMESTAMPS",
                        severity=IssueSeverity.WARNING,
                        message=(
                            f"Detected {len(non_monotonic_rows)} non-chronological "
                            f"timestamp transitions (time moves backwards)"
                        ),
                        field_name=self.timestamp_column,
                        location=non_monotonic_rows[0],
                        context={
                            "unordered_count": len(non_monotonic_rows),
                            "non_monotonic_rows": non_monotonic_rows[:50],
                        },
                    )
                )

        # -------------------------------------------------------------------
        # 4. Result Formulation & Provenance
        # -------------------------------------------------------------------
        has_errors = any(
            i.severity in (IssueSeverity.ERROR, IssueSeverity.CRITICAL) for i in issues
        )
        has_warnings = any(i.severity == IssueSeverity.WARNING for i in issues)

        is_valid = not has_errors

        if has_errors:
            status = ResultStatus.ERROR
        elif has_warnings:
            status = ResultStatus.WARNING
        else:
            status = ResultStatus.SUCCESS

        prov = create_provenance(
            source_uri=resolved_source_uri,
            activity_name="schema_and_timestamp_validation",
            epistemic_level=EpistemicLevel.CALCULATED,
            parent_record_ids=parent_record_ids,
            parameters={
                "required_columns": self.required_columns,
                "timestamp_column": self.timestamp_column,
                "allow_extra_columns": self.allow_extra_columns,
            },
        )

        return ValidationResult(
            status=status,
            is_valid=is_valid,
            validated_row_count=len(df),
            violation_count=len(issues),
            validated_columns=actual_columns,
            issues=issues,
            provenance=prov,
            metadata={
                "has_errors": has_errors,
                "has_warnings": has_warnings,
                "error_count": sum(1 for i in issues if i.severity == IssueSeverity.ERROR),
                "warning_count": sum(1 for i in issues if i.severity == IssueSeverity.WARNING),
            },
        )


def validate_flight_data(
    data: pd.DataFrame | IngestionResult,
    *,
    required_columns: list[str] | None = None,
    timestamp_column: str = "timestamp",
    allow_extra_columns: bool = True,
    report_extra_columns: bool = False,
    source_uri: str | None = None,
) -> ValidationResult:
    """Functional convenience entrypoint for flight data validation."""
    validator = FlightDataValidator(
        required_columns=required_columns,
        timestamp_column=timestamp_column,
        allow_extra_columns=allow_extra_columns,
        report_extra_columns=report_extra_columns,
    )
    return validator.validate(data, source_uri=source_uri)
