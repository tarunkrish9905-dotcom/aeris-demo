"""
Vehicle and aircraft domain modeling foundation.

This module defines the explicit identity and configuration context
for vehicles under test. It strictly maintains domain separation
from observed telemetry and parameter definitions.
"""

from aeris.core.vehicle.configuration import VehicleConfiguration
from aeris.core.vehicle.envelope_models import ConfigurationEnvelope
from aeris.core.vehicle.exceptions import (
    ConfigurationEnvelopeError,
    InvalidConfigurationEnvelopeError,
    InvalidVehicleProfileError,
    ProfileMismatchError,
    VehicleDomainError,
    VehicleProfileError,
)
from aeris.core.vehicle.models import Vehicle
from aeris.core.vehicle.profile_loader import (
    load_configuration_envelope_from_yaml,
    load_vehicle_profile_from_yaml,
)
from aeris.core.vehicle.profile_models import VehicleProfile
from aeris.core.vehicle.test_context import TestContext

__all__ = [
    "ConfigurationEnvelope",
    "ConfigurationEnvelopeError",
    "InvalidConfigurationEnvelopeError",
    "InvalidVehicleProfileError",
    "ProfileMismatchError",
    "TestContext",
    "Vehicle",
    "VehicleConfiguration",
    "VehicleDomainError",
    "VehicleProfile",
    "VehicleProfileError",
    "load_configuration_envelope_from_yaml",
    "load_vehicle_profile_from_yaml",
]
