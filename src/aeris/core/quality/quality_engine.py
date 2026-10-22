"""
AERIS Data Quality Assessment Engine.

Assesses data completeness, gap frequencies, duplicate timestamp rates,
and chronological monotonicity without mutating or imputing the underlying data.
Produces an explainable, bounded quality score within [0.0, 1.0].

Epistemic status of quality output: CALCULATED.

IMPORTANT:
A quality score is an assessment of DATA USABILITY AND INTEGRITY.
It must NOT be presented as flight safety, aircraft safety, engineering
correctness, flight performance, or root-cause probability.
"""

from __future__ import annotations

from typing import Any, Final

import pandas as pd

from aeris.core.provenance.tracker import (
    EpistemicLevel,
    create_provenance,
)
from aeris.core.results.types import (
    IngestionResult,
    Issue,
    IssueSeverity,
    QualityResult,
    ResultStatus,
    ValidationResult,
)
from aeris.core.validation.validators import validate_flight_data

# ---------------------------------------------------------------------------
# Default Scoring Weights (Explainable Linear Formulation)
# ---------------------------------------------------------------------------

DEFAULT_SCORING_WEIGHTS: Final[dict[str, float]] = {
    "completeness": 0.40,  # Proportion of non-missing cells across all columns
    "timestamp_validity": 0.30,  # Proportion of rows with parseable timestamps
    "timestamp_uniqueness": 0.15,  # Proportion of rows without duplicate timestamps
    "timestamp_ordering": 0.15,  # Proportion of transitions that are chronologically monotonic
}


class DataQualityEngine:
    """Evaluates data quality metrics and computes explainable quality scores.

    Operates downstream of Ingestion and Validation. Analyzes:
    - Overall and per-column cell completeness
    - Timestamp validity, duplicates, and out-of-order transitions
    - Usable row proportions
    - Bounded [0.0, 1.0] explainable quality score
    """

    def __init__(
        self,
        timestamp_column: str = "timestamp",
        weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize the quality engine with configured weights.

        Parameters
        ----------
        timestamp_column : str
            Name of the timestamp column (default: 'timestamp').
        weights : dict[str, float] | None
            Component score weights summing to 1.0. Defaults to DEFAULT_SCORING_WEIGHTS.
        """
        self.timestamp_column = timestamp_column
        self.weights = dict(weights or DEFAULT_SCORING_WEIGHTS)

        # Validate weights sum to ~1.0
        total_w = sum(self.weights.values())
        if not (0.99 <= total_w <= 1.01):
            raise ValueError(f"Quality scoring weights must sum to 1.0, got {total_w}")

    def assess(
        self,
        data: pd.DataFrame | IngestionResult,
        validation_result: ValidationResult | None = None,
        *,
        source_uri: str | None = None,
    ) -> QualityResult:
        """Assess quality of the provided dataset and return a QualityResult.

        Parameters
        ----------
        data : pd.DataFrame | IngestionResult
            The data to assess.
        validation_result : ValidationResult | None
            Pre-computed validation result. If None, validation is computed
            automatically for consistency.
        source_uri : str | None
            Optional source identifier for provenance.

        Returns
        -------
        QualityResult
            Structured quality assessment with metrics, score, issues, and provenance.
        """
        # 1. Resolve DataFrame and Upstream Provenance Parent IDs
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
                # Empty / unreadable input case
                prov = create_provenance(
                    source_uri=resolved_source_uri,
                    activity_name="data_quality_assessment",
                    epistemic_level=EpistemicLevel.CALCULATED,
                    parent_record_ids=parent_record_ids,
                )
                return QualityResult(
                    status=ResultStatus.NOT_AVAILABLE,
                    quality_score=None,
                    metrics={
                        "row_count": 0,
                        "column_count": 0,
                        "total_cells": 0,
                        "missing_cells": 0,
                        "completeness_ratio": 0.0,
                    },
                    summary="Quality assessment not available: no tabular data provided.",
                    issues=[
                        Issue(
                            issue_code="WARN_NO_DATA_FOR_QUALITY",
                            severity=IssueSeverity.WARNING,
                            message="Quality assessment could not be performed on empty data",
                            location=resolved_source_uri,
                        )
                    ],
                    provenance=prov,
                )
            df = data.data
        else:
            df = data

        # Link validation provenance if provided
        val_res: ValidationResult
        if validation_result is not None:
            val_res = validation_result
            if val_res.provenance is not None:
                parent_record_ids.append(val_res.provenance.record_id)
        else:
            val_res = validate_flight_data(df, timestamp_column=self.timestamp_column)
            if val_res.provenance is not None:
                parent_record_ids.append(val_res.provenance.record_id)

        row_count, col_count = df.shape

        # Handle 0-row empty DataFrame
        if row_count == 0 or col_count == 0:
            prov = create_provenance(
                source_uri=resolved_source_uri,
                activity_name="data_quality_assessment",
                epistemic_level=EpistemicLevel.CALCULATED,
                parent_record_ids=parent_record_ids,
            )
            return QualityResult(
                status=ResultStatus.NOT_AVAILABLE,
                quality_score=None,
                metrics={
                    "row_count": row_count,
                    "column_count": col_count,
                    "total_cells": 0,
                    "missing_cells": 0,
                    "completeness_ratio": 0.0,
                },
                summary="Quality assessment not available: DataFrame has 0 rows or columns.",
                provenance=prov,
            )

        # -------------------------------------------------------------------
        # 2. Completeness Metrics (Exact cell and column level counts)
        # -------------------------------------------------------------------
        total_cells = row_count * col_count
        missing_cells_per_col = df.isna().sum()
        total_missing_cells = int(missing_cells_per_col.sum())

        completeness_ratio = float((total_cells - total_missing_cells) / total_cells)

        column_completeness: dict[str, float] = {}
        for col_name in df.columns:
            missing_in_col = int(missing_cells_per_col[col_name])
            col_comp = float((row_count - missing_in_col) / row_count)
            column_completeness[str(col_name)] = round(col_comp, 4)

        # -------------------------------------------------------------------
        # 3. Temporal Metrics (Derived from validation issues and series)
        # -------------------------------------------------------------------
        # Count invalid timestamps
        invalid_ts_count = sum(
            1 for i in val_res.issues if i.issue_code == "ERR_TIMESTAMP_UNPARSEABLE"
        )
        invalid_ts_rate = float(invalid_ts_count / row_count)

        # Count duplicate timestamps
        duplicate_ts_count = 0
        if self.timestamp_column in df.columns:
            duplicate_ts_count = int(df[self.timestamp_column].duplicated().sum())
        duplicate_ts_rate = float(duplicate_ts_count / row_count)

        # Count unordered timestamps
        unordered_ts_count = 0
        unordered_issues = [
            i for i in val_res.issues if i.issue_code == "WARN_UNORDERED_TIMESTAMPS"
        ]
        if unordered_issues:
            unordered_ts_count = unordered_issues[0].context.get(
                "unordered_count",
                len(unordered_issues[0].context.get("non_monotonic_rows", [])),
            )
        unordered_ts_rate = float(unordered_ts_count / (row_count - 1)) if row_count > 1 else 0.0

        # -------------------------------------------------------------------
        # 4. Usable Row Count (Rows with valid timestamp and zero missing cells)
        # -------------------------------------------------------------------
        # Identify rows with any null value across columns
        rows_with_nulls = set(df.index[df.isna().any(axis=1)])

        # Identify rows with invalid timestamps
        rows_with_invalid_ts: set[int] = set()
        for i in val_res.issues:
            if i.issue_code == "ERR_TIMESTAMP_UNPARSEABLE" and isinstance(i.location, int):
                rows_with_invalid_ts.add(i.location)

        # Identify rows that are duplicates
        rows_with_duplicates: set[int] = set()
        if self.timestamp_column in df.columns:
            rows_with_duplicates = set(df.index[df[self.timestamp_column].duplicated()])

        unusable_rows = rows_with_nulls | rows_with_invalid_ts | rows_with_duplicates
        usable_row_count = row_count - len(unusable_rows)
        usable_row_ratio = float(usable_row_count / row_count)

        # -------------------------------------------------------------------
        # 5. Component Scores Formulation
        # -------------------------------------------------------------------
        s_comp = completeness_ratio
        s_valid_ts = max(0.0, 1.0 - invalid_ts_rate)
        s_unique_ts = max(0.0, 1.0 - duplicate_ts_rate)
        s_ordered_ts = max(0.0, 1.0 - min(1.0, unordered_ts_rate))

        w = self.weights
        raw_score = (
            w["completeness"] * s_comp
            + w["timestamp_validity"] * s_valid_ts
            + w["timestamp_uniqueness"] * s_unique_ts
            + w["timestamp_ordering"] * s_ordered_ts
        )

        # Strictly clamp within [0.0, 1.0]
        quality_score = round(max(0.0, min(1.0, float(raw_score))), 4)

        # -------------------------------------------------------------------
        # 6. Quality Status, Summary & Issues Formulation
        # -------------------------------------------------------------------
        quality_issues: list[Issue] = []

        if total_missing_cells > 0:
            quality_issues.append(
                Issue(
                    issue_code="WARN_DATA_GAPS",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"Dataset has {total_missing_cells} missing/null cell(s) "
                        f"({(1.0 - completeness_ratio) * 100:.2f}% missing)"
                    ),
                    context={"missing_cells": total_missing_cells},
                )
            )

        if invalid_ts_count > 0:
            quality_issues.append(
                Issue(
                    issue_code="WARN_TIMESTAMP_DROPOUTS",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"{invalid_ts_count} timestamp row(s) could not be parsed "
                        f"({invalid_ts_rate * 100:.2f}% dropout rate)"
                    ),
                    context={"invalid_timestamp_count": invalid_ts_count},
                )
            )

        if duplicate_ts_count > 0:
            quality_issues.append(
                Issue(
                    issue_code="WARN_TIMESTAMP_DUPLICATES",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"{duplicate_ts_count} duplicate timestamp row(s) detected "
                        f"({duplicate_ts_rate * 100:.2f}% duplication rate)"
                    ),
                    context={"duplicate_timestamp_count": duplicate_ts_count},
                )
            )

        if unordered_ts_count > 0:
            quality_issues.append(
                Issue(
                    issue_code="WARN_TIMESTAMP_ORDERING",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"{unordered_ts_count} non-chronological timestamp transition(s) detected"
                    ),
                    context={"unordered_timestamp_count": unordered_ts_count},
                )
            )

        # Status categorization
        if quality_score >= 0.95 and not quality_issues:
            status = ResultStatus.SUCCESS
            summary = (
                f"High quality flight data (score: {quality_score:.2f}). "
                f"Full completeness ({completeness_ratio * 100:.1f}%), monotonic timestamps."
            )
        elif quality_score >= 0.70:
            status = ResultStatus.WARNING
            summary = (
                f"Acceptable data quality with minor defects (score: {quality_score:.2f}). "
                f"{usable_row_count}/{row_count} usable rows ({usable_row_ratio * 100:.1f}%)."
            )
        else:
            status = ResultStatus.PARTIAL
            summary = (
                f"Degraded data quality (score: {quality_score:.2f}). "
                f"Multiple structural or temporal defects present; usable with caution."
            )

        metrics: dict[str, Any] = {
            "row_count": row_count,
            "column_count": col_count,
            "total_cells": total_cells,
            "missing_cells": total_missing_cells,
            "completeness_ratio": round(completeness_ratio, 4),
            "column_completeness": column_completeness,
            "invalid_timestamp_count": invalid_ts_count,
            "invalid_timestamp_rate": round(invalid_ts_rate, 4),
            "duplicate_timestamp_count": duplicate_ts_count,
            "duplicate_timestamp_rate": round(duplicate_ts_rate, 4),
            "unordered_timestamp_count": unordered_ts_count,
            "unordered_timestamp_rate": round(unordered_ts_rate, 4),
            "usable_row_count": usable_row_count,
            "usable_row_ratio": round(usable_row_ratio, 4),
            "component_scores": {
                "completeness": round(s_comp, 4),
                "timestamp_validity": round(s_valid_ts, 4),
                "timestamp_uniqueness": round(s_unique_ts, 4),
                "timestamp_ordering": round(s_ordered_ts, 4),
            },
            "scoring_weights": dict(self.weights),
        }

        prov = create_provenance(
            source_uri=resolved_source_uri,
            activity_name="data_quality_assessment",
            epistemic_level=EpistemicLevel.CALCULATED,
            parent_record_ids=parent_record_ids,
            parameters={"weights": self.weights, "timestamp_column": self.timestamp_column},
        )

        return QualityResult(
            status=status,
            quality_score=quality_score,
            metrics=metrics,
            summary=summary,
            issues=quality_issues,
            provenance=prov,
        )


def assess_data_quality(
    data: pd.DataFrame | IngestionResult,
    validation_result: ValidationResult | None = None,
    *,
    timestamp_column: str = "timestamp",
    weights: dict[str, float] | None = None,
    source_uri: str | None = None,
) -> QualityResult:
    """Convenience functional interface for DataQualityEngine.assess()."""
    engine = DataQualityEngine(timestamp_column=timestamp_column, weights=weights)
    return engine.assess(data, validation_result=validation_result, source_uri=source_uri)
