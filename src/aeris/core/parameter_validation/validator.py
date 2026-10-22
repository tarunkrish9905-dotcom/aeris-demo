"""
Parameter Semantic Validator.
"""

from __future__ import annotations

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.registry import ParameterDataType, ParameterNotFoundError, ParameterRegistry
from aeris.core.results.types import (
    Issue,
    IssueSeverity,
    ParameterValidationResult,
    ResultStatus,
)


class ParameterValidationRequest:
    """Explicitly requests which parameters should undergo semantic validation."""

    def __init__(self, requested_parameters: list[str]) -> None:
        """
        Initialize the validation request.

        Parameters
        ----------
        requested_parameters : list[str]
            Identifiers (aliases or IDs) for the parameters to validate.
        """
        self.requested_parameters = requested_parameters


class ParameterValidator:
    """Evaluates parameter semantic alignment strictly, with zero data mutation."""

    def __init__(self, registry: ParameterRegistry) -> None:
        """
        Initialize with an authoritative ParameterRegistry.

        Parameters
        ----------
        registry : ParameterRegistry
            The registry defining parameter semantic types and aliases.
        """
        self.registry = registry

    def validate(
        self,
        df: pd.DataFrame,
        request: ParameterValidationRequest,
        *,
        source_uri: str | None = None,
    ) -> ParameterValidationResult:
        """
        Evaluate the DataFrame against the explicitly requested registry parameters.

        Parameters
        ----------
        df : pd.DataFrame
            The raw data to validate (will not be mutated).
        request : ParameterValidationRequest
            The specific identifiers to validate.
        source_uri : str | None
            Optional provenance trace identifier.

        Returns
        -------
        ParameterValidationResult
            The structured validation evaluation.
        """
        issues: list[Issue] = []
        validated = []
        failed = []

        resolved_source_uri = source_uri or "in-memory-dataframe"

        if df.empty:
            issues.append(
                Issue(
                    issue_code="ERR_NO_DATA_TO_VALIDATE",
                    severity=IssueSeverity.ERROR,
                    message="DataFrame is empty. Cannot perform semantic validation.",
                    location=resolved_source_uri,
                )
            )
            prov = create_provenance(
                source_uri=resolved_source_uri,
                activity_name="parameter_semantic_validation",
                epistemic_level=EpistemicLevel.CALCULATED,
            )
            return ParameterValidationResult(
                status=ResultStatus.ERROR,
                issues=issues,
                provenance=prov,
                validated_parameters=[],
                failed_parameters=[],
            )

        # Work on a strict view; this prevents unintended mutation
        df_view = df

        for identifier in request.requested_parameters:
            # 1. Parameter Resolution
            try:
                param_id = self.registry.resolve_alias(identifier)
                pdef = self.registry.get(param_id)
            except ParameterNotFoundError:
                issues.append(
                    Issue(
                        issue_code="ERR_PARAMETER_NOT_FOUND",
                        severity=IssueSeverity.ERROR,
                        message=f"Requested parameter '{identifier}' not found in registry.",
                        field_name=identifier,
                    )
                )
                failed.append(identifier)
                continue

            # Determine column name: if the exact identifier requested is a column, use it.
            # Else if the canonical param_id is a column, use it.
            col_name = None
            if identifier in df_view.columns:
                col_name = identifier
            elif param_id in df_view.columns:
                col_name = param_id

            # 2. Column Presence
            if col_name is None:
                issues.append(
                    Issue(
                        issue_code="ERR_PARAMETER_COLUMN_MISSING",
                        severity=IssueSeverity.ERROR,
                        message=f"Expected column for parameter '{identifier}' is missing.",
                        field_name=identifier,
                    )
                )
                failed.append(identifier)
                continue

            series = df_view[col_name]
            is_failed = False

            # 4. Missing Values
            null_count = series.isna().sum()
            if null_count > 0:
                issues.append(
                    Issue(
                        issue_code="WARN_PARAMETER_MISSING_VALUES",
                        severity=IssueSeverity.WARNING,
                        message=f"Parameter '{identifier}' contains {null_count} missing values.",
                        field_name=identifier,
                    )
                )

            # 3, 5, 6. Data Type Compatibility and Semantic Validity
            data_type = pdef.data_type

            if data_type == ParameterDataType.FLOAT:
                if not (is_numeric_dtype(series) or is_float_dtype(series)):
                    issues.append(self._type_mismatch_issue(identifier, "FLOAT", str(series.dtype)))
                    is_failed = True
                    # Check if there are values that cannot reasonably be represented
                    coerced = pd.to_numeric(series, errors="coerce")
                    invalid_mask = coerced.isna() & series.notna()
                    if invalid_mask.any():
                        issues.append(
                            Issue(
                                issue_code="ERR_PARAMETER_INVALID_VALUE",
                                severity=IssueSeverity.ERROR,
                                message=f"Parameter '{identifier}' contains values that "
                                "cannot reasonably be represented as FLOAT.",
                                field_name=identifier,
                            )
                        )

            elif data_type == ParameterDataType.INTEGER:
                if not is_integer_dtype(series):
                    issues.append(
                        self._type_mismatch_issue(identifier, "INTEGER", str(series.dtype))
                    )
                    is_failed = True
                    coerced = pd.to_numeric(series, errors="coerce")
                    invalid_mask = coerced.isna() & series.notna()
                    if invalid_mask.any():
                        issues.append(
                            Issue(
                                issue_code="ERR_PARAMETER_INVALID_VALUE",
                                severity=IssueSeverity.ERROR,
                                message=f"Parameter '{identifier}' contains values that "
                                "cannot reasonably be represented as INTEGER.",
                                field_name=identifier,
                            )
                        )

            elif data_type == ParameterDataType.STRING:
                if not (is_object_dtype(series) or is_string_dtype(series)):
                    issues.append(
                        self._type_mismatch_issue(identifier, "STRING", str(series.dtype))
                    )
                    is_failed = True

            elif data_type == ParameterDataType.BOOLEAN:
                if not is_bool_dtype(series):
                    issues.append(
                        self._type_mismatch_issue(identifier, "BOOLEAN", str(series.dtype))
                    )
                    is_failed = True

            elif data_type == ParameterDataType.TIMESTAMP:
                if not is_datetime64_any_dtype(series):
                    issues.append(
                        self._type_mismatch_issue(identifier, "TIMESTAMP", str(series.dtype))
                    )
                    is_failed = True
                    coerced = pd.to_datetime(series, errors="coerce", format="mixed")
                    invalid_mask = coerced.isna() & series.notna()
                    if invalid_mask.any():
                        issues.append(
                            Issue(
                                issue_code="ERR_PARAMETER_TIMESTAMP_INVALID",
                                severity=IssueSeverity.ERROR,
                                message=f"Parameter '{identifier}' contains unparseable "
                                "timestamp values.",
                                field_name=identifier,
                            )
                        )

            else:
                is_failed = True

            if is_failed:
                failed.append(identifier)
            else:
                validated.append(identifier)

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
            activity_name="parameter_semantic_validation",
            epistemic_level=EpistemicLevel.CALCULATED,
        )

        return ParameterValidationResult(
            status=status,
            issues=issues,
            provenance=prov,
            validated_parameters=validated,
            failed_parameters=failed,
        )

    def _type_mismatch_issue(self, identifier: str, expected: str, actual: str) -> Issue:
        return Issue(
            issue_code="ERR_PARAMETER_TYPE_MISMATCH",
            severity=IssueSeverity.ERROR,
            message=f"Semantic mismatch for '{identifier}': expected {expected}, found {actual}.",
            field_name=identifier,
        )
