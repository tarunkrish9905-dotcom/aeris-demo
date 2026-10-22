"""
AERIS Engineering Limits domain module.

Defines parameter-specific limits, limit severity classifications, limit profile
collections, and atomic YAML loaders.
"""

from aeris.core.limits.exceptions import (
    ConflictingLimitError,
    DuplicateLimitError,
    IncompatibleLimitUnitError,
    InvalidLimitDefinitionError,
    InvalidLimitProfileError,
    LimitDomainError,
    LimitProfileError,
    LimitProfileMismatchError,
    UnknownParameterLimitError,
)
from aeris.core.limits.loader import (
    load_engineering_limit_profile_from_yaml,
    load_limit_profile_from_yaml,
)
from aeris.core.limits.models import (
    EngineeringLimitProfile,
    LimitSeverity,
    LimitType,
    ParameterLimitDefinition,
)

__all__ = [
    "ConflictingLimitError",
    "DuplicateLimitError",
    "EngineeringLimitProfile",
    "IncompatibleLimitUnitError",
    "InvalidLimitDefinitionError",
    "InvalidLimitProfileError",
    "LimitDomainError",
    "LimitProfileError",
    "LimitProfileMismatchError",
    "LimitSeverity",
    "LimitType",
    "ParameterLimitDefinition",
    "UnknownParameterLimitError",
    "load_engineering_limit_profile_from_yaml",
    "load_limit_profile_from_yaml",
]
