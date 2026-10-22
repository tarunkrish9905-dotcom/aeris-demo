import pandas as pd

from aeris.core.engineering.models import (
    EngineeringCalculationDefinition,
    EngineeringCalculationResult,
)
from aeris.core.engineering.registry import get_calculation_registry
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus


class EngineeringExecutor:
    def __init__(self) -> None:
        self.registry = get_calculation_registry()

    def execute(
        self,
        definition: EngineeringCalculationDefinition,
        data: pd.DataFrame,
        parent_record_ids: list[str] | None = None,
        source_uri: str | None = None,
    ) -> EngineeringCalculationResult:

        impl = self.registry.get(definition.calculation_id)
        if impl is None:
            return EngineeringCalculationResult(
                status=ResultStatus.ERROR,
                result_id=f"res_{definition.calculation_id}",
                calculation_id=definition.calculation_id,
                issues=[
                    Issue(
                        issue_code="ERR_CALCULATION_NOT_REGISTERED",
                        severity=IssueSeverity.ERROR,
                        message=f"Calculation {definition.calculation_id} is not registered.",
                    )
                ],
            )

        return impl(
            definition=definition,
            data=data,
            parent_record_ids=parent_record_ids,
            source_uri=source_uri,
        )
