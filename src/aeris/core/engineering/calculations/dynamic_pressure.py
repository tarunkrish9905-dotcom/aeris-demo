import pandas as pd

from aeris.core.engineering.models import (
    EngineeringCalculationDefinition,
    EngineeringCalculationResult,
)
from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus


def calculate_dynamic_pressure(
    definition: EngineeringCalculationDefinition,
    data: pd.DataFrame,
    parent_record_ids: list[str] | None = None,
    source_uri: str | None = None,
) -> EngineeringCalculationResult:
    """
    Calculates dynamic pressure (Pa).
    Formula: q = 0.5 * rho * V^2
    Expects metadata: air_density_column, air_density_unit, true_airspeed_column, true_airspeed_unit
    """
    issues = []

    rho_col = definition.metadata.get("air_density_column")
    v_col = definition.metadata.get("true_airspeed_column")
    rho_unit = definition.metadata.get("air_density_unit")
    v_unit = definition.metadata.get("true_airspeed_unit")

    if not rho_col or rho_col not in data.columns:
        issues.append(
            Issue(
                issue_code="ERR_AIR_DENSITY_COLUMN_MISSING",
                severity=IssueSeverity.ERROR,
                message="Missing explicit air density column.",
            )
        )
    if not v_col or v_col not in data.columns:
        issues.append(
            Issue(
                issue_code="ERR_TRUE_AIRSPEED_COLUMN_MISSING",
                severity=IssueSeverity.ERROR,
                message="Missing explicit true airspeed column.",
            )
        )

    if issues:
        return EngineeringCalculationResult(
            status=ResultStatus.ERROR,
            result_id=f"res_{definition.calculation_id}",
            calculation_id=definition.calculation_id,
            issues=issues,
        )

    if rho_unit != "kg/m^3":
        issues.append(
            Issue(
                issue_code="ERR_UNSUPPORTED_AIR_DENSITY_UNIT",
                severity=IssueSeverity.ERROR,
                message=f"Unsupported unit {rho_unit}, expected kg/m^3.",
            )
        )
    if v_unit != "m/s":
        issues.append(
            Issue(
                issue_code="ERR_UNSUPPORTED_TRUE_AIRSPEED_UNIT",
                severity=IssueSeverity.ERROR,
                message=f"Unsupported unit {v_unit}, expected m/s.",
            )
        )

    if issues:
        return EngineeringCalculationResult(
            status=ResultStatus.ERROR,
            result_id=f"res_{definition.calculation_id}",
            calculation_id=definition.calculation_id,
            issues=issues,
        )

    rho_series = data[rho_col]
    v_series = data[v_col]

    if not pd.api.types.is_numeric_dtype(rho_series):
        issues.append(
            Issue(
                issue_code="ERR_AIR_DENSITY_NOT_NUMERIC",
                severity=IssueSeverity.ERROR,
                message="Air density must be numeric.",
            )
        )
    if not pd.api.types.is_numeric_dtype(v_series):
        issues.append(
            Issue(
                issue_code="ERR_TRUE_AIRSPEED_NOT_NUMERIC",
                severity=IssueSeverity.ERROR,
                message="True airspeed must be numeric.",
            )
        )

    if issues:
        return EngineeringCalculationResult(
            status=ResultStatus.ERROR,
            result_id=f"res_{definition.calculation_id}",
            calculation_id=definition.calculation_id,
            issues=issues,
        )

    if rho_series.isna().any() or v_series.isna().any():
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

    # Calculate q = 0.5 * rho * V^2
    # Ensure independent copy so we don't modify the dataframe
    rho = rho_series.copy()
    v = v_series.copy()

    q = 0.5 * rho * (v**2)

    output_series = pd.Series(q.values, index=data.index.copy(), name="dynamic_pressure_pa")

    prov = create_provenance(
        source_uri=source_uri or "in-memory",
        activity_name="dynamic_pressure_calculation",
        epistemic_level=EpistemicLevel.CALCULATED,
        parent_record_ids=parent_record_ids or [],
    )

    return EngineeringCalculationResult(
        status=ResultStatus.SUCCESS,
        result_id=f"res_{definition.calculation_id}_{prov.record_id[:8]}",
        calculation_id=definition.calculation_id,
        output_value=output_series,
        unit="Pa",
        epistemic_level=EpistemicLevel.CALCULATED,
        provenance=prov,
    )
