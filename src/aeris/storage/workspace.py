"""
AERIS Project Workspace Manager.

Manages application workspace layout, directory confinement, safe path resolution,
identifier validation, and path traversal defense.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import re
import shutil
import uuid

from aeris.storage.exceptions import (
    FileAlreadyExistsError,
    InvalidIdentifierError,
    PathTraversalError,
    WorkspaceError,
)

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
MAX_IDENTIFIER_LENGTH = 64


class WorkspaceManager:
    """
    Manages the physical filesystem layout and security boundaries for AERIS.

    Directory Layout
    ----------------
    {workspace_root}/
    ├── aeris_metadata.db
    └── projects/
        └── {project_id}/
            ├── datasets/
            ├── configs/
            ├── runs/
            └── reports/
    """

    def __init__(self, workspace_root: str | pathlib.Path | None = None) -> None:
        """
        Initialize the WorkspaceManager.

        Parameters
        ----------
        workspace_root : str | pathlib.Path | None
            Root directory for all AERIS managed projects. If None, defaults to
            '~/.aeris_workspace' or the environment variable AERIS_WORKSPACE_DIR.
        """
        if workspace_root is None:
            env_dir = os.environ.get("AERIS_WORKSPACE_DIR")
            if env_dir:
                self.workspace_root = pathlib.Path(env_dir).resolve()
            else:
                self.workspace_root = (pathlib.Path.home() / ".aeris_workspace").resolve()
        else:
            self.workspace_root = pathlib.Path(workspace_root).resolve()

        self.projects_dir = self.workspace_root / "projects"
        self.db_path = self.workspace_root / "aeris_metadata.db"

    def init_workspace(self) -> None:
        """Create the workspace root and projects directories if they do not exist."""
        try:
            self.workspace_root.mkdir(parents=True, exist_ok=True)
            self.projects_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise WorkspaceError(
                f"Failed to initialize workspace root at '{self.workspace_root}': {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Identifier Validation & Path Safety
    # -----------------------------------------------------------------------

    @staticmethod
    def validate_identifier(identifier: str, field_name: str = "identifier") -> str:
        """
        Validate that an entity identifier is safe, non-empty, and correctly formatted.

        Parameters
        ----------
        identifier : str
            The identifier to validate (e.g., project_id, dataset_id).
        field_name : str
            Descriptive name for error messages.

        Returns
        -------
        str
            The validated identifier stripped of leading/trailing whitespace.

        Raises
        ------
        InvalidIdentifierError
            If identifier is empty, exceeds maximum length, or contains illegal characters.
        """
        if not isinstance(identifier, str):
            raise InvalidIdentifierError(f"{field_name} must be a string, got {type(identifier)}")

        clean_id = identifier.strip()
        if not clean_id:
            raise InvalidIdentifierError(f"{field_name} cannot be empty or solely whitespace.")

        if len(clean_id) > MAX_IDENTIFIER_LENGTH:
            raise InvalidIdentifierError(
                f"{field_name} length ({len(clean_id)}) exceeds "
                f"maximum allowed ({MAX_IDENTIFIER_LENGTH})."
            )

        if not IDENTIFIER_PATTERN.match(clean_id):
            raise InvalidIdentifierError(
                f"{field_name} '{clean_id}' contains invalid characters. "
                "Only alphanumeric characters, dashes, and underscores are permitted."
            )

        return clean_id

    def resolve_safe_path(
        self,
        target_path: str | pathlib.Path,
        base_dir: pathlib.Path | None = None,
    ) -> pathlib.Path:
        """
        Resolve target_path and verify that it remains strictly confined within base_dir.

        Parameters
        ----------
        target_path : str | pathlib.Path
            Path to resolve (can be relative or absolute).
        base_dir : pathlib.Path | None
            Base directory boundary. Defaults to workspace_root.

        Returns
        -------
        pathlib.Path
            Resolved absolute Path guaranteed to be inside base_dir.

        Raises
        ------
        PathTraversalError
            If target_path escapes base_dir or contains traversal attacks.
        """
        effective_base = (base_dir or self.workspace_root).resolve()

        # Reject path strings explicitly containing parent traversal tokens
        str_path = str(target_path)
        if ".." in str_path.split("/") or ".." in str_path.split("\\"):
            raise PathTraversalError(
                f"Path traversal token ('..') detected in path: '{target_path}'"
            )

        target = pathlib.Path(target_path)
        if not target.is_absolute():
            resolved = (effective_base / target).resolve()
        else:
            resolved = target.resolve()

        if not resolved.is_relative_to(effective_base):
            raise PathTraversalError(
                f"Resolved path '{resolved}' escapes base directory '{effective_base}'."
            )

        return resolved

    # -----------------------------------------------------------------------
    # Project Workspace Layout Management
    # -----------------------------------------------------------------------

    def get_project_dir(self, project_id: str) -> pathlib.Path:
        """Return the managed project directory path for project_id."""
        valid_id = self.validate_identifier(project_id, "project_id")
        return self.resolve_safe_path(self.projects_dir / valid_id, self.projects_dir)

    def get_dataset_dir(self, project_id: str) -> pathlib.Path:
        """Return the managed datasets subdirectory for project_id."""
        return self.get_project_dir(project_id) / "datasets"

    def get_config_dir(self, project_id: str) -> pathlib.Path:
        """Return the managed configs subdirectory for project_id."""
        return self.get_project_dir(project_id) / "configs"

    def get_runs_dir(self, project_id: str) -> pathlib.Path:
        """Return the managed runs subdirectory for project_id."""
        return self.get_project_dir(project_id) / "runs"

    def get_reports_dir(self, project_id: str) -> pathlib.Path:
        """Return the managed reports subdirectory for project_id."""
        return self.get_project_dir(project_id) / "reports"

    def create_project_workspace(self, project_id: str) -> pathlib.Path:
        """
        Create the full directory structure for a project.

        Returns
        -------
        pathlib.Path
            The root directory of the created project workspace.
        """
        self.init_workspace()
        proj_dir = self.get_project_dir(project_id)
        try:
            proj_dir.mkdir(parents=True, exist_ok=True)
            self.get_dataset_dir(project_id).mkdir(exist_ok=True)
            self.get_config_dir(project_id).mkdir(exist_ok=True)
            self.get_runs_dir(project_id).mkdir(exist_ok=True)
            self.get_reports_dir(project_id).mkdir(exist_ok=True)
            return proj_dir
        except OSError as exc:
            raise WorkspaceError(
                f"Failed to create workspace directory structure for project '{project_id}': {exc}"
            ) from exc

    def purge_project_workspace(self, project_id: str) -> None:
        """
        Safely delete the entire directory tree of a project workspace.

        Parameters
        ----------
        project_id : str
            Identifier of the project to remove.
        """
        proj_dir = self.get_project_dir(project_id)
        if proj_dir.exists():
            try:
                shutil.rmtree(proj_dir)
            except OSError as exc:
                raise WorkspaceError(
                    f"Failed to purge project workspace '{proj_dir}': {exc}"
                ) from exc

    # -----------------------------------------------------------------------
    # Safe Atomic File Operations
    # -----------------------------------------------------------------------

    def create_safe_file(
        self,
        target_path: str | pathlib.Path,
        content_bytes: bytes,
        overwrite: bool = False,
        base_dir: pathlib.Path | None = None,
    ) -> pathlib.Path:
        """
        Atomically write content_bytes to target_path within base_dir.

        Parameters
        ----------
        target_path : str | pathlib.Path
            Destination path.
        content_bytes : bytes
            Payload to write.
        overwrite : bool
            Whether to overwrite an existing file.
        base_dir : pathlib.Path | None
            Confining base directory. Defaults to workspace_root.

        Returns
        -------
        pathlib.Path
            Resolved final path of the written file.
        """
        resolved_dest = self.resolve_safe_path(target_path, base_dir)

        if resolved_dest.exists() and not overwrite:
            raise FileAlreadyExistsError(
                f"Destination file already exists and overwrite is False: '{resolved_dest}'"
            )

        resolved_dest.parent.mkdir(parents=True, exist_ok=True)

        staging_name = f".aeris_tmp_{uuid.uuid4().hex}"
        staging_path = resolved_dest.parent / staging_name

        try:
            with staging_path.open("wb") as f:
                f.write(content_bytes)
                f.flush()
                os.fsync(f.fileno())

            os.replace(staging_path, resolved_dest)
            return resolved_dest
        except Exception as exc:
            if staging_path.exists():
                with contextlib.suppress(OSError):
                    staging_path.unlink()
            raise WorkspaceError(f"Failed writing safe file to '{resolved_dest}': {exc}") from exc
