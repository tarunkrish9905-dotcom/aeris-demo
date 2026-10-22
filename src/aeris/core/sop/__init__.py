"""
AERIS SOP and Test Procedure Requirements Package.

Provides immutable domain models, atomic loaders, and validation facilities
for authoritative SOP and flight-test procedure requirements.
"""

from aeris.core.sop.exceptions import (
    ConflictingRequirementError,
    DuplicateRequirementError,
    IncompatibleProcedureUnitError,
    InvalidProcedureProfileError,
    InvalidProcedureRequirementError,
    ProcedureProfileMismatchError,
    ProcedureRequirementError,
    SopDomainError,
    UnknownParameterProcedureError,
    UnknownTestOperationReferenceError,
)
from aeris.core.sop.loader import (
    load_procedure_requirement_profile_from_yaml,
    load_procedure_requirements_from_yaml,
    load_sop_requirements_from_yaml,
)
from aeris.core.sop.models import (
    ProcedureRequirement,
    ProcedureRequirementProfile,
    ProcedureRequirementSet,
    ProcedureRequirementVerification,
    RequirementCategory,
    RequirementVerificationStatus,
    validate_document_version,
)

__all__ = [
    "ConflictingRequirementError",
    "DuplicateRequirementError",
    "IncompatibleProcedureUnitError",
    "InvalidProcedureProfileError",
    "InvalidProcedureRequirementError",
    "ProcedureProfileMismatchError",
    "ProcedureRequirement",
    "ProcedureRequirementError",
    "ProcedureRequirementProfile",
    "ProcedureRequirementSet",
    "ProcedureRequirementVerification",
    "RequirementCategory",
    "RequirementVerificationStatus",
    "SopDomainError",
    "UnknownParameterProcedureError",
    "UnknownTestOperationReferenceError",
    "load_procedure_requirement_profile_from_yaml",
    "load_procedure_requirements_from_yaml",
    "load_sop_requirements_from_yaml",
    "validate_document_version",
]
