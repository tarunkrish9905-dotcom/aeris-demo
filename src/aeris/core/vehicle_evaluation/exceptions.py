"""
Exceptions for the Vehicle Engineering Evaluation domain.
"""

from __future__ import annotations

from aeris.core.vehicle.exceptions import VehicleDomainError


class EvaluationDomainError(VehicleDomainError):
    """Base exception for all vehicle engineering evaluation errors."""


class EvaluationConfigurationError(EvaluationDomainError):
    """Raised when request or configuration is contradictory or missing profile bindings."""


class IncompatibleEvaluationUnitError(EvaluationDomainError):
    """Raised when observed values cannot be evaluated against limits due to unit conflicts."""


class UnknownEvaluationParameterError(EvaluationDomainError):
    """Raised when an unidentifiable parameter is requested for evaluation."""


class AmbiguousApplicabilityError(EvaluationDomainError):
    """Raised when multiple overlapping limits or requirements create ambiguous scope."""
