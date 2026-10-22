"""
AERIS Storage Subsystem Exceptions.

Defines the authoritative error hierarchy for storage, workspace path resolution,
database operations, dataset persistence, and integrity checks.
"""

from __future__ import annotations


class StorageError(Exception):
    """Base exception for all storage, workspace, and persistence errors."""


class WorkspaceError(StorageError):
    """Raised when application workspace initialization or layout fails."""


class PathTraversalError(StorageError):
    """Raised when a path escapes the managed workspace or contains illegal traversal tokens."""


class InvalidIdentifierError(StorageError):
    """Raised when a project, dataset, or entity identifier violates format constraints."""


class EntityNotFoundError(StorageError):
    """Base exception when an entity is not found in the storage repository."""


class ProjectNotFoundError(EntityNotFoundError):
    """Raised when the specified project ID does not exist."""


class DatasetNotFoundError(EntityNotFoundError):
    """Raised when the specified dataset ID does not exist."""


class DuplicateEntityError(StorageError):
    """Raised when attempting to create an entity that already exists."""


class FileAlreadyExistsError(StorageError):
    """Raised when target file already exists and overwrite is False."""


class FileIntegrityError(StorageError):
    """Raised when a file's content fails cryptographic checksum verification or is corrupted."""


class OversizedFileError(StorageError):
    """Raised when an ingested file exceeds the configured maximum size limit."""


class DatabaseMigrationError(StorageError):
    """Raised when database schema migrations fail to apply."""


class DatabaseError(StorageError):
    """Raised for unexpected SQLite database operational or execution failures."""


class ArtifactNotFoundError(EntityNotFoundError):
    """Raised when an analysis run artifact is missing from metadata or filesystem."""


class ArtifactCorruptError(FileIntegrityError):
    """Raised when an analysis artifact fails checksum verification or JSON parsing."""


class ArtifactIncompatibleError(StorageError):
    """Raised when an analysis artifact format or version is unsupported."""


class ProjectMismatchError(StorageError):
    """Raised when an analysis run or artifact does not match the expected project."""
