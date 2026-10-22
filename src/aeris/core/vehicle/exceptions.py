"""
Exceptions for the vehicle domain.
"""

from __future__ import annotations


class VehicleDomainError(Exception):
    """Base exception for all vehicle domain errors."""


class VehicleProfileError(VehicleDomainError):
    """Base exception for vehicle profile loading and validation errors."""


class InvalidVehicleProfileError(VehicleProfileError):
    """Raised when a vehicle profile definition is structurally invalid."""


class ConfigurationEnvelopeError(VehicleDomainError):
    """Base exception for configuration envelope errors."""


class InvalidConfigurationEnvelopeError(ConfigurationEnvelopeError):
    """Raised when a configuration envelope is structurally invalid."""


class ProfileMismatchError(VehicleDomainError):
    """Raised when a configuration envelope is associated with an incompatible profile."""
