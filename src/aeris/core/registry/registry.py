"""
AERIS Parameter Registry Service.

Maintains the authoritative, in-memory dictionary of parameter definitions.
"""

from aeris.core.registry.exceptions import (
    DuplicateAliasError,
    DuplicateParameterError,
    InvalidParameterDefinitionError,
    ParameterNotFoundError,
)
from aeris.core.registry.models import ParameterDefinition


class ParameterRegistry:
    """In-memory authoritative registry for semantic parameter definitions."""

    def __init__(self) -> None:
        self._parameters: dict[str, ParameterDefinition] = {}
        self._aliases: dict[str, str] = {}

    def register(self, definition: ParameterDefinition) -> None:
        """Register a new parameter definition.

        Parameters
        ----------
        definition : ParameterDefinition
            The immutable semantic definition to store.

        Raises
        ------
        DuplicateParameterError
            If a parameter with the same ID is already registered.
        DuplicateAliasError
            If any of the parameter's aliases clash with an existing registered ID
            or an alias belonging to another parameter.
        InvalidParameterDefinitionError
            If an alias matches the parameter's own ID (redundant but cleanly handled).
        """
        param_id = definition.parameter_id

        if param_id in self._parameters:
            raise DuplicateParameterError(f"Parameter ID '{param_id}' is already registered.")

        # ID cannot conflict with an existing alias of ANOTHER parameter
        if param_id in self._aliases:
            raise DuplicateAliasError(
                f"Cannot register parameter ID '{param_id}'; "
                f"it is already claimed as an alias by '{self._aliases[param_id]}'."
            )

        # Pre-validate all new aliases for conflicts
        for alias in definition.aliases:
            if alias == param_id:
                raise InvalidParameterDefinitionError(
                    f"Alias '{alias}' matches the parameter ID itself. Aliases must be distinct."
                )
            if alias in self._parameters:
                raise DuplicateAliasError(
                    f"Alias '{alias}' for parameter '{param_id}' conflicts "
                    "with an existing parameter ID."
                )
            if alias in self._aliases:
                raise DuplicateAliasError(
                    f"Alias '{alias}' for '{param_id}' is already claimed "
                    f"by '{self._aliases[alias]}'."
                )

        # Commit to registry
        self._parameters[param_id] = definition
        for alias in definition.aliases:
            self._aliases[alias] = param_id

    def get(self, parameter_id: str) -> ParameterDefinition:
        """Retrieve a parameter definition exactly by its ID.

        Does not resolve aliases.

        Parameters
        ----------
        parameter_id : str
            The exact primary identifier of the parameter.

        Returns
        -------
        ParameterDefinition
            The registered immutable definition.

        Raises
        ------
        ParameterNotFoundError
            If the exact parameter ID does not exist in the registry.
        """
        if parameter_id not in self._parameters:
            raise ParameterNotFoundError(f"Parameter ID '{parameter_id}' not found in registry.")
        return self._parameters[parameter_id]

    def contains(self, parameter_id: str) -> bool:
        """Check if an exact parameter ID is registered.

        Parameters
        ----------
        parameter_id : str
            The exact primary identifier of the parameter.

        Returns
        -------
        bool
            True if registered, False otherwise.
        """
        return parameter_id in self._parameters

    def resolve_alias(self, alias_or_id: str) -> str:
        """Deterministically resolve a string to its primary parameter ID.

        Rules:
        1. Exact parameter ID takes precedence (returns itself).
        2. Known alias maps to its parent parameter ID.
        3. Unknown strings raise a ParameterNotFoundError.

        Parameters
        ----------
        alias_or_id : str
            The string to resolve.

        Returns
        -------
        str
            The resolved primary parameter ID.

        Raises
        ------
        ParameterNotFoundError
            If the string matches neither an ID nor an alias.
        """
        if alias_or_id in self._parameters:
            return alias_or_id
        if alias_or_id in self._aliases:
            return self._aliases[alias_or_id]

        raise ParameterNotFoundError(
            f"Identifier '{alias_or_id}' could not be resolved to any "
            "registered parameter or alias."
        )

    def list(self) -> list[ParameterDefinition]:
        """List all registered parameter definitions.

        Returns
        -------
        List[ParameterDefinition]
            A list of all immutable parameter definitions in the registry.
        """
        return list(self._parameters.values())

    def unregister(self, parameter_id: str) -> None:
        """Safely remove a parameter definition and all its aliases.

        Parameters
        ----------
        parameter_id : str
            The exact primary identifier of the parameter to remove.

        Raises
        ------
        ParameterNotFoundError
            If the exact parameter ID does not exist in the registry.
        """
        if parameter_id not in self._parameters:
            raise ParameterNotFoundError(
                f"Cannot unregister unknown parameter ID '{parameter_id}'."
            )

        definition = self._parameters.pop(parameter_id)
        for alias in definition.aliases:
            if alias in self._aliases:
                del self._aliases[alias]
