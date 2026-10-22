"""
AERIS Storage Subsystem.

Provides persistent workspace directory management, SQLite metadata repository,
byte-preserving dataset storage, and crash reconciliation.
"""

from aeris.storage.artifact_store import (
    ARTIFACT_FORMAT,
    CURRENT_FORMAT_VERSION,
    DEFAULT_ARTIFACT_FILENAME,
    AnalysisArtifactStore,
    deserialize_pipeline_result,
    sanitize_pipeline_result_for_storage,
    serialize_pipeline_result,
)
from aeris.storage.dataset_store import DEFAULT_MAX_UPLOAD_SIZE_BYTES, ManagedDatasetStore
from aeris.storage.db import get_db_connection, transaction
from aeris.storage.exceptions import (
    ArtifactCorruptError,
    ArtifactIncompatibleError,
    ArtifactNotFoundError,
    DatabaseError,
    DatabaseMigrationError,
    DatasetNotFoundError,
    DuplicateEntityError,
    EntityNotFoundError,
    FileAlreadyExistsError,
    FileIntegrityError,
    InvalidIdentifierError,
    OversizedFileError,
    PathTraversalError,
    ProjectMismatchError,
    ProjectNotFoundError,
    StorageError,
    WorkspaceError,
)
from aeris.storage.migrations import apply_migrations, get_current_schema_version
from aeris.storage.models import (
    AnalysisRunRecord,
    BackgroundJobRecord,
    ConfigurationProfileRecord,
    DatasetRecord,
    ExportedReportRecord,
    ParameterMappingRecord,
    ProjectRecord,
    ProjectStatus,
    WorkspaceIntegrityReport,
)
from aeris.storage.reconciliation import WorkspaceReconciler
from aeris.storage.repository import SQLiteMetadataRepository
from aeris.storage.workspace import WorkspaceManager

__all__ = [
    "ARTIFACT_FORMAT",
    "CURRENT_FORMAT_VERSION",
    "DEFAULT_ARTIFACT_FILENAME",
    "DEFAULT_MAX_UPLOAD_SIZE_BYTES",
    "AnalysisArtifactStore",
    "AnalysisRunRecord",
    "ArtifactCorruptError",
    "ArtifactIncompatibleError",
    "ArtifactNotFoundError",
    "BackgroundJobRecord",
    "ConfigurationProfileRecord",
    "DatabaseError",
    "DatabaseMigrationError",
    "DatasetNotFoundError",
    "DatasetRecord",
    "DuplicateEntityError",
    "EntityNotFoundError",
    "ExportedReportRecord",
    "FileAlreadyExistsError",
    "FileIntegrityError",
    "InvalidIdentifierError",
    "ManagedDatasetStore",
    "OversizedFileError",
    "ParameterMappingRecord",
    "PathTraversalError",
    "ProjectMismatchError",
    "ProjectNotFoundError",
    "ProjectRecord",
    "ProjectStatus",
    "SQLiteMetadataRepository",
    "StorageError",
    "WorkspaceError",
    "WorkspaceIntegrityReport",
    "WorkspaceManager",
    "WorkspaceReconciler",
    "apply_migrations",
    "deserialize_pipeline_result",
    "get_current_schema_version",
    "get_db_connection",
    "sanitize_pipeline_result_for_storage",
    "serialize_pipeline_result",
    "transaction",
]
