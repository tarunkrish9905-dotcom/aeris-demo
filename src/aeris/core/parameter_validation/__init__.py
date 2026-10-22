"""
AERIS Parameter Semantic Validation.

Performs explicit, strict verification that ingested columns
semantically match their declared registry definitions.
Does not mutate or coerce data.
"""

from aeris.core.parameter_validation.validator import (
    ParameterValidationRequest,
    ParameterValidator,
)

__all__ = [
    "ParameterValidationRequest",
    "ParameterValidator",
]
