"""
AERIS Parameter Registry Exceptions.
"""


class RegistryError(Exception):
    """Base exception for all parameter registry errors."""


class DuplicateParameterError(RegistryError):
    """Raised when attempting to register a parameter with an ID that already exists."""


class DuplicateAliasError(RegistryError):
    """Raised when attempting to register an alias that is already mapped to another parameter."""


class ParameterNotFoundError(RegistryError):
    """Raised when a parameter ID or alias cannot be resolved."""


class InvalidParameterDefinitionError(RegistryError):
    """Raised when a parameter definition is structurally invalid or violates constraints."""
