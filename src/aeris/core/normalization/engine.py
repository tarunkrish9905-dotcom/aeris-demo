"""
AERIS Data Normalization Engine.

Applies explicit, traceable, non-destructive transformations to raw datasets.
"""

from __future__ import annotations

import uuid

import pandas as pd

from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.registry import ParameterDataType, ParameterNotFoundError, ParameterRegistry
from aeris.core.results.types import (
    IngestionResult,
    Issue,
    IssueSeverity,
    NormalizationResult,
    ResultStatus,
    TransformationRecord,
    TransformationType,
)


class NormalizationRequest:
    """Explicit request for data normalizations to apply."""

    def __init__(self) -> None:
        self.renames: dict[str, str] = {}
        self.type_conversions: dict[str, type] = {}
        self.timestamp_columns: list[str] = []

        self.registry_renames: list[str] = []
        self.registry_type_conversions: list[tuple[str, type]] = []
        self.registry_timestamp_columns: list[str] = []

    def rename_column(self, source: str, destination: str) -> None:
        self.renames[source] = destination

    def convert_type(self, column: str, target_type: type) -> None:
        self.type_conversions[column] = target_type

    def standardize_timestamp(self, column: str) -> None:
        if column not in self.timestamp_columns:
            self.timestamp_columns.append(column)

    def resolve_and_rename(self, identifier: str) -> None:
        """Explicitly request renaming to the registry-resolved parameter ID."""
        if identifier not in self.registry_renames:
            self.registry_renames.append(identifier)

    def resolve_and_convert_type(self, identifier: str, target_type: type) -> None:
        """Explicitly request type conversion for a registry-resolved parameter."""
        self.registry_type_conversions.append((identifier, target_type))

    def resolve_and_standardize_timestamp(self, identifier: str) -> None:
        """Explicitly request timestamp standardization for a registry-resolved parameter."""
        if identifier not in self.registry_timestamp_columns:
            self.registry_timestamp_columns.append(identifier)


class DataNormalizationEngine:
    """Applies exact normalization requests producing a NormalizationResult."""

    def __init__(self, registry: ParameterRegistry | None = None) -> None:
        self.registry = registry

    def normalize(
        self,
        data: pd.DataFrame | IngestionResult,
        request: NormalizationRequest,
        *,
        source_uri: str | None = None,
    ) -> NormalizationResult:
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
                prov = create_provenance(
                    source_uri=resolved_source_uri,
                    activity_name="data_normalization",
                    epistemic_level=EpistemicLevel.CALCULATED,
                    parent_record_ids=parent_record_ids,
                )
                return NormalizationResult(
                    status=ResultStatus.ERROR,
                    issues=[
                        Issue(
                            issue_code="ERR_NO_DATA",
                            severity=IssueSeverity.ERROR,
                            message="No tabular data available for normalization.",
                            location=resolved_source_uri,
                        )
                    ],
                    provenance=prov,
                )
            df = data.data
        else:
            df = data

        if df.empty:
            prov = create_provenance(
                source_uri=resolved_source_uri,
                activity_name="data_normalization",
                epistemic_level=EpistemicLevel.CALCULATED,
                parent_record_ids=parent_record_ids,
            )
            return NormalizationResult(
                status=ResultStatus.ERROR,
                issues=[
                    Issue(
                        issue_code="ERR_EMPTY_DATA",
                        severity=IssueSeverity.ERROR,
                        message="DataFrame is empty.",
                        location=resolved_source_uri,
                    )
                ],
                provenance=prov,
            )

        # Ensure immutability by operating on a copy
        working_df = df.copy(deep=True)
        transformations: list[TransformationRecord] = []
        issues: list[Issue] = []

        input_columns = list(df.columns)
        input_row_count = len(df)

        resolved_renames = dict(request.renames)
        resolved_type_conversions = dict(request.type_conversions)
        resolved_timestamp_columns = list(request.timestamp_columns)

        for identifier in request.registry_renames:
            if not self.registry:
                issues.append(
                    Issue(
                        issue_code="ERR_REGISTRY_NOT_AVAILABLE",
                        severity=IssueSeverity.ERROR,
                        message="Registry required for resolve_and_rename",
                        field_name=identifier,
                    )
                )
                continue
            try:
                param_id = self.registry.resolve_alias(identifier)
                resolved_renames[identifier] = param_id
            except ParameterNotFoundError as exc:
                issues.append(
                    Issue(
                        issue_code="ERR_PARAMETER_NOT_FOUND",
                        severity=IssueSeverity.ERROR,
                        message=str(exc),
                        field_name=identifier,
                    )
                )

        for identifier, target_type in request.registry_type_conversions:
            if not self.registry:
                issues.append(
                    Issue(
                        issue_code="ERR_REGISTRY_NOT_AVAILABLE",
                        severity=IssueSeverity.ERROR,
                        message="Registry required for resolve_and_convert_type",
                        field_name=identifier,
                    )
                )
                continue
            try:
                param_id = self.registry.resolve_alias(identifier)
                pdef = self.registry.get(param_id)

                # Check semantic compatibility for numeric conversion
                if target_type in (float, int) and pdef.data_type not in (
                    ParameterDataType.NUMERIC,
                    ParameterDataType.FLOAT,
                    ParameterDataType.INTEGER,
                ):
                    issues.append(
                        Issue(
                            issue_code="ERR_INCOMPATIBLE_TRANSFORMATION",
                            severity=IssueSeverity.ERROR,
                            message=f"Target type {target_type} incompatible "
                            f"with parameter {param_id} type {pdef.data_type}",
                            field_name=identifier,
                        )
                    )
                    continue

                resolved_type_conversions[identifier] = target_type
            except ParameterNotFoundError as exc:
                issues.append(
                    Issue(
                        issue_code="ERR_PARAMETER_NOT_FOUND",
                        severity=IssueSeverity.ERROR,
                        message=str(exc),
                        field_name=identifier,
                    )
                )

        for identifier in request.registry_timestamp_columns:
            if not self.registry:
                issues.append(
                    Issue(
                        issue_code="ERR_REGISTRY_NOT_AVAILABLE",
                        severity=IssueSeverity.ERROR,
                        message="Registry required for resolve_and_standardize_timestamp",
                        field_name=identifier,
                    )
                )
                continue
            try:
                param_id = self.registry.resolve_alias(identifier)
                pdef = self.registry.get(param_id)

                if pdef.data_type != ParameterDataType.TIMESTAMP:
                    issues.append(
                        Issue(
                            issue_code="ERR_INCOMPATIBLE_TRANSFORMATION",
                            severity=IssueSeverity.ERROR,
                            message=f"Timestamp standardization requested for parameter "
                            f"{param_id} which has type {pdef.data_type}",
                            field_name=identifier,
                        )
                    )
                    continue

                resolved_timestamp_columns.append(identifier)
            except ParameterNotFoundError as exc:
                issues.append(
                    Issue(
                        issue_code="ERR_PARAMETER_NOT_FOUND",
                        severity=IssueSeverity.ERROR,
                        message=str(exc),
                        field_name=identifier,
                    )
                )

        # 1. Column Renames
        for src, dst in resolved_renames.items():
            if src not in working_df.columns:
                issues.append(
                    Issue(
                        issue_code="ERR_MISSING_SOURCE_COLUMN",
                        severity=IssueSeverity.ERROR,
                        message=f"Cannot rename missing column '{src}'",
                        field_name=src,
                    )
                )
                continue

            working_df.rename(columns={src: dst}, inplace=True)
            transformations.append(
                TransformationRecord(
                    transformation_id=str(uuid.uuid4()),
                    transformation_type=TransformationType.COLUMN_RENAME,
                    source_column=src,
                    destination_column=dst,
                    is_lossless=True,
                )
            )

        # 2. Type Conversions
        for col, target_type in resolved_type_conversions.items():
            actual_col = resolved_renames.get(col, col)
            if actual_col not in working_df.columns:
                issues.append(
                    Issue(
                        issue_code="ERR_MISSING_SOURCE_COLUMN",
                        severity=IssueSeverity.ERROR,
                        message=f"Cannot convert type of missing column '{actual_col}'",
                        field_name=actual_col,
                    )
                )
                continue

            try:
                working_df[actual_col] = working_df[actual_col].astype(target_type)
                transformations.append(
                    TransformationRecord(
                        transformation_id=str(uuid.uuid4()),
                        transformation_type=TransformationType.TYPE_CONVERSION,
                        source_column=actual_col,
                        destination_column=actual_col,
                        parameters={"target_type": str(target_type)},
                        is_lossless=None,
                    )
                )
            except (ValueError, TypeError) as exc:
                issues.append(
                    Issue(
                        issue_code="ERR_TYPE_CONVERSION_FAILED",
                        severity=IssueSeverity.ERROR,
                        message=f"Failed to convert column '{actual_col}' to {target_type}: {exc}",
                        field_name=actual_col,
                    )
                )

        # 3. Timestamp Standardization
        for col in resolved_timestamp_columns:
            actual_col = resolved_renames.get(col, col)
            if actual_col not in working_df.columns:
                issues.append(
                    Issue(
                        issue_code="ERR_MISSING_SOURCE_COLUMN",
                        severity=IssueSeverity.ERROR,
                        message=f"Cannot standardize missing timestamp column '{actual_col}'",
                        field_name=actual_col,
                    )
                )
                continue

            try:
                # Use strict ISO8601 UTC parsing
                converted = pd.to_datetime(working_df[actual_col], utc=True, errors="raise")
                working_df[actual_col] = converted
                transformations.append(
                    TransformationRecord(
                        transformation_id=str(uuid.uuid4()),
                        transformation_type=TransformationType.TIMESTAMP_STANDARDIZATION,
                        source_column=actual_col,
                        destination_column=actual_col,
                        parameters={"target_timezone": "UTC"},
                        is_lossless=False,  # Parsing to datetime can discard string exactness
                    )
                )
            except Exception as exc:
                issues.append(
                    Issue(
                        issue_code="ERR_TIMESTAMP_CONVERSION_FAILED",
                        severity=IssueSeverity.ERROR,
                        message=f"Failed to standardize timestamp column '{actual_col}': {exc}",
                        field_name=actual_col,
                    )
                )

        # Determine status
        has_errors = any(
            i.severity in (IssueSeverity.ERROR, IssueSeverity.CRITICAL) for i in issues
        )
        has_warnings = any(i.severity == IssueSeverity.WARNING for i in issues)

        if has_errors:
            status = ResultStatus.ERROR
        elif has_warnings:
            status = ResultStatus.WARNING
        else:
            status = ResultStatus.SUCCESS

        prov = create_provenance(
            source_uri=resolved_source_uri,
            activity_name="data_normalization",
            epistemic_level=EpistemicLevel.CALCULATED,
            parent_record_ids=parent_record_ids,
        )

        return NormalizationResult(
            status=status,
            data=working_df if not has_errors else None,
            transformations=transformations,
            input_row_count=input_row_count,
            output_row_count=len(working_df) if not has_errors else 0,
            input_columns=input_columns,
            output_columns=list(working_df.columns) if not has_errors else [],
            issues=issues,
            provenance=prov,
        )


def normalize_flight_data(
    data: pd.DataFrame | IngestionResult,
    request: NormalizationRequest,
    *,
    registry: ParameterRegistry | None = None,
    source_uri: str | None = None,
) -> NormalizationResult:
    """Convenience entrypoint for executing normalization."""
    engine = DataNormalizationEngine(registry=registry)
    return engine.normalize(data, request, source_uri=source_uri)
