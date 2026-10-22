"""
AERIS Engineering Limit Profile Atomic YAML Loader.

Loads limit profile definitions from YAML configuration files into an authoritative
EngineeringLimitProfile object. Guarantees fail-fast atomic validation: any syntax error,
structural defect, boundary contradiction, or registry unit mismatch prevents object exposure.
"""

from __future__ import annotations

import pathlib
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from aeris.core.limits.exceptions import (
    InvalidLimitProfileError,
    LimitProfileError,
)
from aeris.core.limits.models import EngineeringLimitProfile
from aeris.core.registry import ParameterRegistry
from aeris.core.vehicle.profile_models import VehicleProfile


def load_limit_profile_from_yaml(
    file_path: str | pathlib.Path,
    expected_profile: VehicleProfile | None = None,
    parameter_registry: ParameterRegistry | None = None,
) -> EngineeringLimitProfile:
    """
    Atomically load and validate an EngineeringLimitProfile from a YAML file.

    Parameters
    ----------
    file_path : str | pathlib.Path
        Path to the limit profile YAML configuration file.
    expected_profile : VehicleProfile | None
        Optional VehicleProfile to validate compatibility against. If provided,
        vehicle_id, profile_id, and profile_version must exactly match.
    parameter_registry : ParameterRegistry | None
        Optional ParameterRegistry to cross-reference parameter IDs and canonical units.

    Returns
    -------
    EngineeringLimitProfile
        The validated, immutable EngineeringLimitProfile instance.

    Raises
    ------
    InvalidLimitProfileError
        If YAML is malformed, top-level structure is invalid, required fields are missing,
        or extra fields are present.
    LimitProfileMismatchError
        If the limit profile does not match the expected_profile.
    UnknownParameterLimitError
        If any limit references an unregistered parameter ID when parameter_registry is provided.
    IncompatibleLimitUnitError
        If any limit specifies a unit incompatible with the registry's canonical_unit.
    DuplicateLimitError
        If duplicate limit IDs or duplicate definitions exist in the profile.
    ConflictingLimitError
        If contradictory or ambiguous overlapping limits are defined.
    """
    path = pathlib.Path(file_path)

    try:
        with path.open("r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise InvalidLimitProfileError(f"Malformed YAML syntax in '{path}': {exc}") from exc
    except OSError as exc:
        raise InvalidLimitProfileError(f"Cannot read limit profile file '{path}': {exc}") from exc

    if not isinstance(raw_data, dict):
        raise InvalidLimitProfileError(
            "Engineering limit profile configuration must be a dictionary at the top level."
        )

    # Check top-level envelope key
    profile_data: dict[str, Any]
    if "engineering_limit_profile" in raw_data:
        profile_data = raw_data["engineering_limit_profile"]
    elif "limit_profile" in raw_data:
        profile_data = raw_data["limit_profile"]
    elif "limits_profile" in raw_data:
        profile_data = raw_data["limits_profile"]
    else:
        profile_data = raw_data

    if not isinstance(profile_data, dict):
        raise InvalidLimitProfileError("Engineering limit profile content must be a dictionary.")

    try:
        limit_profile = EngineeringLimitProfile(**profile_data)
    except ValidationError as exc:
        raise InvalidLimitProfileError(
            f"Validation failed for engineering limit profile: {exc}"
        ) from exc
    except LimitProfileError:
        # Re-raise domain-specific errors (DuplicateLimitError, ConflictingLimitError, etc.)
        raise

    # Cross-validate with expected VehicleProfile if supplied
    if expected_profile is not None:
        limit_profile.validate_vehicle_profile_compatibility(expected_profile)

    # Cross-validate with ParameterRegistry if supplied
    if parameter_registry is not None:
        limit_profile.validate_registry_compatibility(parameter_registry)

    return limit_profile


# Semantic alias
load_engineering_limit_profile_from_yaml = load_limit_profile_from_yaml
