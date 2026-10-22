"""
AERIS Parameter Registry Configuration Loader.

Loads parameter definitions from YAML configuration files into an authoritative
in-memory registry. Guarantees atomicity and structural validation.
"""

import pathlib

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from aeris.core.registry.exceptions import (
    InvalidParameterDefinitionError,
)
from aeris.core.registry.models import ParameterDefinition
from aeris.core.registry.registry import ParameterRegistry


def load_registry_from_yaml(file_path: str | pathlib.Path) -> ParameterRegistry:
    """Load parameter definitions from a YAML file into a ParameterRegistry.

    Guarantees atomicity: if the file contains any malformed parameters,
    structural violations, or logical collisions (e.g. duplicate aliases),
    the entire load operation fails and no registry is returned.

    Parameters
    ----------
    file_path : str | pathlib.Path
        Path to the YAML configuration file.

    Returns
    -------
    ParameterRegistry
        A fully populated, validated in-memory registry.

    Raises
    ------
    InvalidParameterDefinitionError
        If the YAML structure is malformed or missing required keys.
    RegistryError
        If parameter definitions collide (duplicate IDs, duplicate aliases).
    """
    path = pathlib.Path(file_path)

    try:
        with path.open("r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise InvalidParameterDefinitionError(f"Malformed YAML syntax: {exc}") from exc
    except OSError as exc:
        raise InvalidParameterDefinitionError(f"Cannot read configuration file: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise InvalidParameterDefinitionError(
            "Configuration must be a dictionary at the top level."
        )

    if "parameters" not in raw_data:
        raise InvalidParameterDefinitionError(
            "Configuration is missing the required top-level 'parameters' key."
        )

    parameters_list = raw_data["parameters"]
    if not isinstance(parameters_list, list):
        raise InvalidParameterDefinitionError(
            "The 'parameters' key must contain a list of parameter definitions."
        )

    # Use a fresh registry to ensure atomicity.
    # If any registration fails, this instance is discarded.
    registry = ParameterRegistry()

    for idx, raw_param in enumerate(parameters_list):
        if not isinstance(raw_param, dict):
            raise InvalidParameterDefinitionError(f"Parameter at index {idx} is not a dictionary.")

        try:
            definition = ParameterDefinition(**raw_param)
        except ValidationError as exc:
            param_id = raw_param.get("parameter_id", f"index_{idx}")
            raise InvalidParameterDefinitionError(
                f"Validation failed for parameter '{param_id}': {exc}"
            ) from exc

        # register() will automatically throw DuplicateParameterError,
        # DuplicateAliasError, or InvalidParameterDefinitionError if
        # there are collisions.
        registry.register(definition)

    return registry
