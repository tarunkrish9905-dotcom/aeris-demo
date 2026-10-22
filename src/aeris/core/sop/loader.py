"""
AERIS SOP and Test Procedure Requirements Atomic YAML Loader.

Loads procedure and SOP requirement profiles from YAML configuration files into an authoritative
ProcedureRequirementProfile object. Guarantees fail-fast atomic validation: any syntax error,
structural defect, boundary contradiction, or registry mismatch prevents object exposure.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from aeris.core.registry import ParameterRegistry
from aeris.core.sop.exceptions import (
    InvalidProcedureProfileError,
    ProcedureRequirementError,
)
from aeris.core.sop.models import ProcedureRequirementProfile
from aeris.core.test_operations.test_card import TestCard
from aeris.core.test_operations.test_point import TestPoint
from aeris.core.vehicle.envelope_models import ConfigurationEnvelope
from aeris.core.vehicle.profile_models import VehicleProfile


def load_procedure_requirement_profile_from_yaml(
    file_path: str | pathlib.Path,
    expected_profile: VehicleProfile | None = None,
    expected_configuration: ConfigurationEnvelope | None = None,
    test_cards: Sequence[TestCard] | None = None,
    test_points: Sequence[TestPoint] | None = None,
    parameter_registry: ParameterRegistry | None = None,
) -> ProcedureRequirementProfile:
    """
    Atomically load and validate a ProcedureRequirementProfile from a YAML file.

    Parameters
    ----------
    file_path : str | pathlib.Path
        Path to the procedure requirement profile YAML configuration file.
    expected_profile : VehicleProfile | None
        Optional VehicleProfile to validate vehicle identity and supported parameters against.
    expected_configuration : ConfigurationEnvelope | None
        Optional ConfigurationEnvelope to validate configuration applicability against.
    test_cards : Sequence[TestCard] | None
        Optional sequence of valid TestCards to cross-reference card IDs and test points.
    test_points : Sequence[TestPoint] | None
        Optional sequence of valid TestPoints to cross-reference test point IDs.
    parameter_registry : ParameterRegistry | None
        Optional ParameterRegistry to cross-reference parameter IDs and canonical units.

    Returns
    -------
    ProcedureRequirementProfile
        The validated, immutable ProcedureRequirementProfile instance.

    Raises
    ------
    InvalidProcedureProfileError
        If YAML is malformed, top-level structure is invalid, required fields are missing,
        or forbidden extra fields are present.
    ProcedureProfileMismatchError
        If requirements conflict with expected_profile, expected_configuration,
        or document identity.
    UnknownParameterProcedureError
        If a requirement references an unregistered parameter ID or parameter unsupported
        by the profile.
    IncompatibleProcedureUnitError
        If a numeric requirement specifies a unit incompatible with the registry canonical unit.
    UnknownTestOperationReferenceError
        If a requirement references an unknown test_card_id or test_point_id.
    DuplicateRequirementError
        If duplicate requirement IDs exist in the profile.
    ConflictingRequirementError
        If contradictory or ambiguous overlapping requirements are defined.
    """
    path = pathlib.Path(file_path)

    try:
        with path.open("r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise InvalidProcedureProfileError(f"Malformed YAML syntax in '{path}': {exc}") from exc
    except OSError as exc:
        raise InvalidProcedureProfileError(
            f"Cannot read procedure profile file '{path}': {exc}"
        ) from exc

    if not isinstance(raw_data, dict):
        raise InvalidProcedureProfileError(
            "Procedure requirement profile configuration must be a dictionary at the top level."
        )

    # Check top-level envelope keys
    profile_data: dict[str, Any]
    for key in (
        "procedure_requirement_profile",
        "sop_requirement_profile",
        "procedure_profile",
        "sop_profile",
        "procedure_requirements",
    ):
        if key in raw_data:
            candidate = raw_data[key]
            if isinstance(candidate, dict):
                profile_data = candidate
                break
    else:
        profile_data = raw_data

    if not isinstance(profile_data, dict):
        raise InvalidProcedureProfileError(
            "Procedure requirement profile content must be a dictionary."
        )

    try:
        profile = ProcedureRequirementProfile(**profile_data)
    except ValidationError as exc:
        raise InvalidProcedureProfileError(
            f"Validation failed for procedure requirement profile: {exc}"
        ) from exc
    except ProcedureRequirementError:
        raise

    # Cross-validation against explicitly provided domain models
    if expected_profile is not None:
        profile.validate_vehicle_profile_compatibility(expected_profile)

    if expected_configuration is not None:
        profile.validate_configuration_compatibility(expected_configuration)

    if test_cards is not None or test_points is not None:
        profile.validate_test_operations_compatibility(test_cards, test_points)

    if parameter_registry is not None:
        profile.validate_registry_compatibility(parameter_registry)

    return profile


# Semantic aliases for convenience and domain readability
load_procedure_requirements_from_yaml = load_procedure_requirement_profile_from_yaml
load_sop_requirements_from_yaml = load_procedure_requirement_profile_from_yaml
