"""
Exceptions for the engineering limits domain.
"""

from __future__ import annotations

from aeris.core.vehicle.exceptions import VehicleDomainError


class LimitDomainError(VehicleDomainError):
    """Base exception for all engineering limit domain errors."""


class LimitProfileError(LimitDomainError):
    """Base exception for limit profile loading, construction, and validation errors."""


class InvalidLimitDefinitionError(LimitProfileError):
    """Raised when an individual parameter limit definition is structurally invalid."""


class InvalidLimitProfileError(LimitProfileError):
    """Raised when an engineering limit profile is structurally invalid."""


class LimitProfileMismatchError(LimitProfileError):
    """Raised when a limit profile or definition does not match the associated vehicle profile."""


class IncompatibleLimitUnitError(LimitProfileError):
    """Raised when a limit's unit does not match the parameter registry's canonical unit."""


class UnknownParameterLimitError(LimitProfileError):
    """Raised when a limit references a parameter ID not registered in the parameter registry."""


class DuplicateLimitError(LimitProfileError):
    """Raised when duplicate limit identifiers or duplicate limit definitions exist."""


class ConflictingLimitError(LimitProfileError):
    """Raised when ambiguous or conflicting limits exist for the same parameter and scope."""
