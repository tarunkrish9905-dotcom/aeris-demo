"""
AERIS Parameter Registry package.
"""

from aeris.core.registry.canonical import (
    CANONICAL_LABELS,
    CANONICAL_PARAMETERS,
    CANONICAL_UNITS,
    CanonicalParameter,
    infer_unit_from_column_name,
    suggest_canonical_parameter,
)
from aeris.core.registry.exceptions import (
    DuplicateAliasError,
    DuplicateParameterError,
    InvalidParameterDefinitionError,
    ParameterNotFoundError,
    RegistryError,
)
from aeris.core.registry.loader import load_registry_from_yaml
from aeris.core.registry.models import ParameterDataType, ParameterDefinition
from aeris.core.registry.registry import ParameterRegistry

__all__ = [
    "CANONICAL_LABELS",
    "CANONICAL_PARAMETERS",
    "CANONICAL_UNITS",
    "CanonicalParameter",
    "DuplicateAliasError",
    "DuplicateParameterError",
    "InvalidParameterDefinitionError",
    "ParameterDataType",
    "ParameterDefinition",
    "ParameterNotFoundError",
    "ParameterRegistry",
    "RegistryError",
    "infer_unit_from_column_name",
    "load_registry_from_yaml",
    "suggest_canonical_parameter",
]
