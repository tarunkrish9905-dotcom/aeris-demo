"""
AERIS Vehicle Profile and Configuration Envelope Atomic YAML Loader.

Loads vehicle profiles and associated configuration envelopes from YAML configuration
files with strict atomic validation. Guarantees that any malformed syntax, schema
violation, parameter conflict, or profile mismatch prevents object exposure.
"""

from __future__ import annotations

import pathlib
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from aeris.core.registry import ParameterRegistry
from aeris.core.vehicle.envelope_models import ConfigurationEnvelope
from aeris.core.vehicle.exceptions import (
    InvalidConfigurationEnvelopeError,
    InvalidVehicleProfileError,
)
from aeris.core.vehicle.profile_models import VehicleProfile


def load_vehicle_profile_from_yaml(
    file_path: str | pathlib.Path,
    parameter_registry: ParameterRegistry | None = None,
) -> VehicleProfile:
    """
    Atomically load and validate a VehicleProfile from a YAML file.

    Parameters
    ----------
    file_path : str | pathlib.Path
        Path to the vehicle profile YAML file.
    parameter_registry : ParameterRegistry | None
        Optional parameter registry to cross-reference supported parameter IDs.
        If provided, any supported_parameter_id not present in the registry raises
        an InvalidVehicleProfileError.

    Returns
    -------
    VehicleProfile
        The validated, immutable VehicleProfile instance.

    Raises
    ------
    InvalidVehicleProfileError
        If YAML is malformed, top-level structure is wrong, required fields are missing,
        extra fields are present, or supported parameters fail registry verification.
    """
    path = pathlib.Path(file_path)

    try:
        with path.open("r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise InvalidVehicleProfileError(f"Malformed YAML syntax in '{path}': {exc}") from exc
    except OSError as exc:
        raise InvalidVehicleProfileError(
            f"Cannot read vehicle profile file '{path}': {exc}"
        ) from exc

    if not isinstance(raw_data, dict):
        raise InvalidVehicleProfileError(
            "Vehicle profile configuration must be a dictionary at the top level."
        )

    # Check for top-level key: either raw dictionary has 'profile' key or the dict itself is profile
    profile_data: dict[str, Any]
    if "vehicle_profile" in raw_data:
        profile_data = raw_data["vehicle_profile"]
    elif "profile" in raw_data:
        profile_data = raw_data["profile"]
    else:
        profile_data = raw_data

    if not isinstance(profile_data, dict):
        raise InvalidVehicleProfileError("Vehicle profile content must be a dictionary.")

    # Instantiate through Pydantic model for strict atomic validation
    try:
        profile = VehicleProfile(**profile_data)
    except ValidationError as exc:
        raise InvalidVehicleProfileError(f"Validation failed for vehicle profile: {exc}") from exc

    # If parameter_registry was supplied, cross-validate supported_parameter_ids
    if parameter_registry is not None:
        unknown_params = [
            param_id
            for param_id in profile.supported_parameter_ids
            if not parameter_registry.contains(param_id)
        ]
        if unknown_params:
            raise InvalidVehicleProfileError(
                f"Vehicle profile '{profile.profile_id}' references unknown parameter IDs "
                f"not found in ParameterRegistry: {unknown_params}."
            )

    return profile


def load_configuration_envelope_from_yaml(
    file_path: str | pathlib.Path,
    expected_profile: VehicleProfile | None = None,
) -> ConfigurationEnvelope:
    """
    Atomically load and validate a ConfigurationEnvelope from a YAML file.

    Parameters
    ----------
    file_path : str | pathlib.Path
        Path to the configuration envelope YAML file.
    expected_profile : VehicleProfile | None
        Optional VehicleProfile to validate compatibility against.
        If provided and mismatched, raises ProfileMismatchError.

    Returns
    -------
    ConfigurationEnvelope
        The validated, immutable ConfigurationEnvelope instance.

    Raises
    ------
    InvalidConfigurationEnvelopeError
        If YAML is malformed, top-level structure is wrong, required fields are missing,
        or extra fields are present.
    ProfileMismatchError
        If the envelope does not match the expected_profile vehicle_id, profile_id,
        or profile_version.
    """
    path = pathlib.Path(file_path)

    try:
        with path.open("r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise InvalidConfigurationEnvelopeError(
            f"Malformed YAML syntax in '{path}': {exc}"
        ) from exc
    except OSError as exc:
        raise InvalidConfigurationEnvelopeError(
            f"Cannot read configuration envelope file '{path}': {exc}"
        ) from exc

    if not isinstance(raw_data, dict):
        raise InvalidConfigurationEnvelopeError(
            "Configuration envelope must be a dictionary at the top level."
        )

    envelope_data: dict[str, Any]
    if "configuration_envelope" in raw_data:
        envelope_data = raw_data["configuration_envelope"]
    elif "envelope" in raw_data:
        envelope_data = raw_data["envelope"]
    else:
        envelope_data = raw_data

    if not isinstance(envelope_data, dict):
        raise InvalidConfigurationEnvelopeError(
            "Configuration envelope content must be a dictionary."
        )

    try:
        envelope = ConfigurationEnvelope(**envelope_data)
    except ValidationError as exc:
        raise InvalidConfigurationEnvelopeError(
            f"Validation failed for configuration envelope: {exc}"
        ) from exc

    if expected_profile is not None:
        envelope.validate_profile_compatibility(expected_profile)

    return envelope
