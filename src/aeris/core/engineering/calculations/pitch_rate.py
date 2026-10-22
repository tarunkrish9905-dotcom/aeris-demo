import pandas as pd

from aeris.core.engineering.models import (
    EngineeringCalculationDefinition,
    EngineeringCalculationResult,
)
from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus


def calculate_pitch_rate(
    definition: EngineeringCalculationDefinition,
    data: pd.DataFrame,
    parent_record_ids: list[str] | None = None,
    source_uri: str | None = None,
) -> EngineeringCalculationResult:
    """
    Calculates the rate of change of pitch (deg/s).
    Expects metadata: pitch_column, timestamp_column, pitch_unit
    """
    issues = []

    pitch_col = definition.metadata.get("pitch_column")
    timestamp_col = definition.metadata.get("timestamp_column")
    pit_unit = definition.metadata.get("pitch_unit")

    if not pitch_col or pitch_col not in data.columns:
        issues.append(
            Issue(
                issue_code="ERR_PITCH_COLUMN_MISSING",
                severity=IssueSeverity.ERROR,
                message="Missing explicit pitch column.",
            )
        )
    if not timestamp_col or timestamp_col not in data.columns:
        issues.append(
            Issue(
                issue_code="ERR_TIMESTAMP_COLUMN_MISSING",
                severity=IssueSeverity.ERROR,
                message="Missing explicit timestamp column.",
            )
        )

    if issues:
        return EngineeringCalculationResult(
            status=ResultStatus.ERROR,
            result_id=f"res_{definition.calculation_id}",
            calculation_id=definition.calculation_id,
            issues=issues,
        )

    if pit_unit != "deg":
        issues.append(
            Issue(
                issue_code="ERR_UNSUPPORTED_INPUT_UNIT",
                severity=IssueSeverity.ERROR,
                message=f"Unsupported unit {pit_unit}, expected deg.",
            )
        )
        return EngineeringCalculationResult(
            status=ResultStatus.ERROR,
            result_id=f"res_{definition.calculation_id}",
            calculation_id=definition.calculation_id,
            issues=issues,
        )

    if len(data) < 2:
        issues.append(
            Issue(
                issue_code="ERR_INSUFFICIENT_DATA",
                severity=IssueSeverity.ERROR,
                message="At least two points required.",
            )
        )
        return EngineeringCalculationResult(
            status=ResultStatus.ERROR,
            result_id=f"res_{definition.calculation_id}",
            calculation_id=definition.calculation_id,
            issues=issues,
        )

    timestamps = data[timestamp_col]
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        issues.append(
            Issue(
                issue_code="ERR_TIMESTAMP_COLUMN_INVALID",
                severity=IssueSeverity.ERROR,
                message="Timestamp must be datetime type.",
            )
        )
    elif timestamps.dt.tz is None:
        issues.append(
            Issue(
                issue_code="ERR_TIMESTAMP_NAIVE",
                severity=IssueSeverity.ERROR,
                message="Timestamps must be timezone aware.",
            )
        )
    elif (
        str(timestamps.dt.tz) not in ("UTC", "datetime.timezone.utc", "datetime.UTC")
        and getattr(timestamps.dt.tz, "zone", None) != "UTC"
    ):
        issues.append(
            Issue(
                issue_code="ERR_TIMESTAMP_NON_UTC",
                severity=IssueSeverity.ERROR,
                message="Timestamps must be UTC.",
            )
        )

    if issues:
        return EngineeringCalculationResult(
            status=ResultStatus.ERROR,
            result_id=f"res_{definition.calculation_id}",
            calculation_id=definition.calculation_id,
            issues=issues,
        )

    if data[pitch_col].isna().any() or timestamps.isna().any():
        issues.append(
            Issue(
                issue_code="ERR_INPUT_VALUE_MISSING",
                severity=IssueSeverity.ERROR,
                message="Missing values in inputs.",
            )
        )
        return EngineeringCalculationResult(
            status=ResultStatus.ERROR,
            result_id=f"res_{definition.calculation_id}",
            calculation_id=definition.calculation_id,
            issues=issues,
        )

    if not timestamps.is_monotonic_increasing:
        issues.append(
            Issue(
                issue_code="ERR_TIMESTAMP_UNORDERED",
                severity=IssueSeverity.ERROR,
                message="Timestamps must be strictly ordered.",
            )
        )
        return EngineeringCalculationResult(
            status=ResultStatus.ERROR,
            result_id=f"res_{definition.calculation_id}",
            calculation_id=definition.calculation_id,
            issues=issues,
        )

    dt = timestamps.diff().dt.total_seconds()

    if (dt == 0).any():
        issues.append(
            Issue(
                issue_code="ERR_ZERO_TIME_INTERVAL",
                severity=IssueSeverity.ERROR,
                message="Duplicate timestamps found.",
            )
        )
        return EngineeringCalculationResult(
            status=ResultStatus.ERROR,
            result_id=f"res_{definition.calculation_id}",
            calculation_id=definition.calculation_id,
            issues=issues,
        )

    # Make independent copy to ensure immutability
    da = data[pitch_col].diff().copy()
    rate = da / dt

    output_series = pd.Series(rate.values, index=data.index.copy(), name="pitch_rate_deg_s")

    prov = create_provenance(
        source_uri=source_uri or "in-memory",
        activity_name="pitch_rate_calculation",
        epistemic_level=EpistemicLevel.CALCULATED,
        parent_record_ids=parent_record_ids or [],
    )

    return EngineeringCalculationResult(
        status=ResultStatus.SUCCESS,
        result_id=f"res_{definition.calculation_id}_{prov.record_id[:8]}",
        calculation_id=definition.calculation_id,
        output_value=output_series,
        unit="deg/s",
        epistemic_level=EpistemicLevel.CALCULATED,
        provenance=prov,
    )
