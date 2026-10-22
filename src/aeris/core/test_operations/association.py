"""
Test Point Association Engine.

Explicitly maps a TestPointObservation to the observed data by slicing
a DataFrame chronologically. Responsibilities are strictly limited to
extracting rows bound by the [start, end] temporal interval without
modifying the source or applying automatic inferences.
"""

from __future__ import annotations

import uuid
from typing import Any

import pandas as pd
from pydantic import Field

from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord
from aeris.core.results.types import BasePipelineResult, Issue, IssueSeverity, ResultStatus
from aeris.core.test_operations.test_point_observation import TestPointObservation


class TestPointAssociationResult(BasePipelineResult):
    """Structured outcome of a Test Point Association extraction."""

    test_point_id: str = Field(description="ID of the associated Test Point definition.")
    observation_id: str = Field(description="ID of the Test Point Observation.")
    timestamp_column: str = Field(description="Name of the column used for temporal slicing.")
    row_count: int = Field(default=0, description="Number of observed rows extracted.")
    data: Any = Field(default=None, description="The extracted DataFrame slice.")


class TestPointAssociationEngine:
    """
    Explicit Temporal Slicer connecting Definitions to Operations.

    Extracts rows falling inclusively within the `[start, end]` boundaries
    of a TestPointObservation. Strictly preserves data source immutability
    and sequence ordering.
    """

    def extract(
        self,
        data: pd.DataFrame,
        observation: TestPointObservation,
        timestamp_column: str,
        parent_provenance: ProvenanceRecord | None = None,
    ) -> TestPointAssociationResult:
        """
        Extract the observation slice from the data.

        Returns a TestPointAssociationResult detailing success/failure, issues,
        the explicit provenance, and the extracted independent DataFrame.
        """

        # Ensure we always generate provenance mapping the event
        def _make_result(
            status: ResultStatus,
            issues: list[Issue],
            result_data: pd.DataFrame | None,
            row_count: int,
        ) -> TestPointAssociationResult:
            prov = ProvenanceRecord(
                record_id=str(uuid.uuid4()),
                parent_record_ids=[parent_provenance.record_id] if parent_provenance else [],
                epistemic_level=EpistemicLevel.CALCULATED,
                activity_name="test_point_association",
                source_uri=observation.source_reference,
                parameters={
                    "observation_id": observation.observation_id,
                    "test_point_id": observation.test_point_id,
                    "timestamp_column": timestamp_column,
                    "start_timestamp": observation.start_timestamp.isoformat(),
                    "end_timestamp": observation.end_timestamp.isoformat(),
                    "extracted_row_count": row_count,
                },
            )
            return TestPointAssociationResult(
                status=status,
                issues=issues,
                provenance=prov,
                test_point_id=observation.test_point_id,
                observation_id=observation.observation_id,
                timestamp_column=timestamp_column,
                row_count=row_count,
                data=result_data,
            )

        if data.empty:
            return _make_result(
                status=ResultStatus.ERROR,
                issues=[
                    Issue(
                        issue_code="ERR_NO_DATA",
                        severity=IssueSeverity.ERROR,
                        message="Input DataFrame is empty.",
                    )
                ],
                result_data=None,
                row_count=0,
            )

        if timestamp_column not in data.columns:
            return _make_result(
                status=ResultStatus.ERROR,
                issues=[
                    Issue(
                        issue_code="ERR_TIMESTAMP_COLUMN_MISSING",
                        severity=IssueSeverity.ERROR,
                        message=f"Timestamp column '{timestamp_column}' missing.",
                        field_name=timestamp_column,
                    )
                ],
                result_data=None,
                row_count=0,
            )

        ts_series = data[timestamp_column]
        if not pd.api.types.is_datetime64_any_dtype(ts_series):
            return _make_result(
                status=ResultStatus.ERROR,
                issues=[
                    Issue(
                        issue_code="ERR_TIMESTAMP_COLUMN_INVALID",
                        severity=IssueSeverity.ERROR,
                        message="Timestamp column is not datetime-like.",
                        field_name=timestamp_column,
                    )
                ],
                result_data=None,
                row_count=0,
            )

        tz = ts_series.dt.tz
        if tz is None:
            return _make_result(
                status=ResultStatus.ERROR,
                issues=[
                    Issue(
                        issue_code="ERR_TIMESTAMP_NAIVE",
                        severity=IssueSeverity.ERROR,
                        message="Timestamp column contains naive timestamps.",
                        field_name=timestamp_column,
                    )
                ],
                result_data=None,
                row_count=0,
            )

        # Check explicitly for UTC offset == 0
        try:
            # We take the first valid timestamp to verify the offset.
            first_valid = ts_series.dropna().iloc[0]
            if first_valid.utcoffset().total_seconds() != 0:
                return _make_result(
                    status=ResultStatus.ERROR,
                    issues=[
                        Issue(
                            issue_code="ERR_TIMESTAMP_NON_UTC",
                            severity=IssueSeverity.ERROR,
                            message="Timestamp column must be explicitly UTC (offset 0).",
                            field_name=timestamp_column,
                        )
                    ],
                    result_data=None,
                    row_count=0,
                )
        except IndexError:
            # Series consists entirely of NaT; wait for slicing logic to handle.
            pass

        # Perform the temporal extraction [start, end] inclusively.
        mask = (ts_series >= observation.start_timestamp) & (ts_series <= observation.end_timestamp)
        extracted_df = data[mask].copy()

        if extracted_df.empty:
            return _make_result(
                status=ResultStatus.NOT_AVAILABLE,
                issues=[
                    Issue(
                        issue_code="ERR_NO_OBSERVED_ROWS_IN_INTERVAL",
                        severity=IssueSeverity.WARNING,
                        message="Valid interval but no rows found in boundaries.",
                        field_name=timestamp_column,
                    )
                ],
                result_data=None,
                row_count=0,
            )

        return _make_result(
            status=ResultStatus.SUCCESS,
            issues=[],
            result_data=extracted_df,
            row_count=len(extracted_df),
        )
