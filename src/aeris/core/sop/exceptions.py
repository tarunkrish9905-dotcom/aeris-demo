"""
Exceptions for the SOP and Test Procedure Requirements domain.
"""

from __future__ import annotations

from aeris.core.vehicle.exceptions import VehicleDomainError


class SopDomainError(VehicleDomainError):
    """Base exception for all SOP and procedure requirement domain errors."""


class ProcedureRequirementError(SopDomainError):
    """Base exception for procedure requirement loading, construction, and validation errors."""


class InvalidProcedureRequirementError(ProcedureRequirementError):
    """Raised when an individual procedure requirement definition is structurally invalid."""


class InvalidProcedureProfileError(ProcedureRequirementError):
    """Raised when a procedure requirement profile or collection is structurally invalid."""


class ProcedureProfileMismatchError(ProcedureRequirementError):
    """Raised when applicability or metadata conflicts with parent profile, vehicle, or context."""


class IncompatibleProcedureUnitError(ProcedureRequirementError):
    """Raised when numeric requirement unit does not match registry canonical unit."""


class UnknownParameterProcedureError(ProcedureRequirementError):
    """Raised when requirement references a parameter ID not registered in registry."""


class UnknownTestOperationReferenceError(ProcedureRequirementError):
    """Raised when a requirement references an unknown TestCard or TestPoint ID."""


class DuplicateRequirementError(ProcedureRequirementError):
    """Raised when duplicate requirement identifiers exist."""


class ConflictingRequirementError(ProcedureRequirementError):
    """Raised when ambiguous or conflicting requirements exist for the same operational scope."""
