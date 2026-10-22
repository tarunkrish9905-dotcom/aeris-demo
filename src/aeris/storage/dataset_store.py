"""
AERIS Managed Dataset Store.

Handles byte-preserving ingestion, cryptographic checksum calculation, upload size limits,
atomic staging and replacement, and synchronization between managed files and metadata.
"""

from __future__ import annotations

import contextlib
import csv
import datetime
import hashlib
import os
import pathlib
import uuid
from typing import BinaryIO

from aeris.storage.exceptions import (
    DatasetNotFoundError,
    FileAlreadyExistsError,
    FileIntegrityError,
    OversizedFileError,
    PathTraversalError,
    ProjectNotFoundError,
    StorageError,
)
from aeris.storage.models import DatasetRecord
from aeris.storage.repository import SQLiteMetadataRepository
from aeris.storage.workspace import WorkspaceManager

DEFAULT_MAX_UPLOAD_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB
CHUNK_SIZE_BYTES = 64 * 1024  # 64 KB


class ManagedDatasetStore:
    """
    Manages physical storage, integrity verification, and metadata registration
    for raw telemetry datasets.
    """

    def __init__(
        self,
        workspace: WorkspaceManager,
        repository: SQLiteMetadataRepository,
        max_upload_size_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES,
    ) -> None:
        """
        Initialize ManagedDatasetStore.

        Parameters
        ----------
        workspace : WorkspaceManager
            The authoritative workspace directory manager.
        repository : SQLiteMetadataRepository
            The authoritative SQLite metadata repository.
        max_upload_size_bytes : int
            Maximum allowed file size in bytes. Defaults to 200MB.
        """
        self.workspace = workspace
        self.repository = repository
        self.max_upload_size_bytes = max_upload_size_bytes

    def store_dataset(
        self,
        project_id: str,
        source: str | pathlib.Path | BinaryIO | bytes,
        original_filename: str,
        dataset_id: str | None = None,
        overwrite: bool = False,
    ) -> tuple[DatasetRecord, pathlib.Path]:
        """
        Ingest a telemetry CSV file into managed workspace storage.

        Parameters
        ----------
        project_id : str
            Parent project identifier. Must exist in metadata repository.
        source : str | pathlib.Path | BinaryIO | bytes
            Source data to copy into managed storage.
        original_filename : str
            Original filename as provided by the user (must end with .csv).
        dataset_id : str | None
            Optional explicit dataset identifier. If None, generated as 'ds_{uuid12}'.
        overwrite : bool
            Whether to overwrite an existing managed file and metadata.

        Returns
        -------
        tuple[DatasetRecord, pathlib.Path]
            The registered DatasetRecord and the absolute path to the managed CSV file.

        Raises
        ------
        ProjectNotFoundError
            If project_id does not exist in repository.
        InvalidIdentifierError
            If project_id or dataset_id is malformed.
        PathTraversalError
            If filename contains path traversal tokens.
        FileAlreadyExistsError
            If destination file exists and overwrite is False.
        OversizedFileError
            If source data exceeds max_upload_size_bytes.
        FileIntegrityError
            If file fails inspection or verification.
        """
        # 1. Verify project exists
        valid_project_id = self.workspace.validate_identifier(project_id, "project_id")
        project = self.repository.get_project(valid_project_id)
        if project is None:
            raise ProjectNotFoundError(f"Project '{valid_project_id}' does not exist.")

        # 2. Assign / validate dataset ID
        if dataset_id is None:
            clean_dataset_id = f"ds_{uuid.uuid4().hex[:12]}"
        else:
            clean_dataset_id = self.workspace.validate_identifier(dataset_id, "dataset_id")

        # 3. Sanitize filename and verify CSV extension
        clean_filename = self._sanitize_filename(original_filename)

        # 4. Resolve destination path
        dataset_dir = self.workspace.get_dataset_dir(valid_project_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dest_filename = f"{clean_dataset_id}.csv"
        dest_path = dataset_dir / dest_filename

        if dest_path.exists() and not overwrite:
            raise FileAlreadyExistsError(
                f"Managed dataset file already exists: '{dest_path}'. "
                "Use overwrite=True to replace."
            )

        # 5. Atomic staging: Write to temporary file
        staging_filename = f".aeris_tmp_{uuid.uuid4().hex}.csv"
        staging_path = dataset_dir / staging_filename

        hasher = hashlib.sha256()
        total_bytes = 0

        try:
            with staging_path.open("wb") as out_f:
                if isinstance(source, (str, pathlib.Path)):
                    src_path = pathlib.Path(source)
                    if not src_path.exists() or not src_path.is_file():
                        raise StorageError(f"Source file does not exist: '{src_path}'")

                    with src_path.open("rb") as in_f:
                        while chunk := in_f.read(CHUNK_SIZE_BYTES):
                            total_bytes += len(chunk)
                            if total_bytes > self.max_upload_size_bytes:
                                raise OversizedFileError(
                                    f"File size exceeds maximum allowed "
                                    f"({self.max_upload_size_bytes} bytes)."
                                )
                            hasher.update(chunk)
                            out_f.write(chunk)
                elif isinstance(source, bytes):
                    total_bytes = len(source)
                    if total_bytes > self.max_upload_size_bytes:
                        raise OversizedFileError(
                            f"File size exceeds maximum allowed "
                            f"({self.max_upload_size_bytes} bytes)."
                        )
                    hasher.update(source)
                    out_f.write(source)
                elif hasattr(source, "read"):
                    while chunk := source.read(CHUNK_SIZE_BYTES):
                        total_bytes += len(chunk)
                        if total_bytes > self.max_upload_size_bytes:
                            raise OversizedFileError(
                                f"File size exceeds maximum allowed "
                                f"({self.max_upload_size_bytes} bytes)."
                            )
                        hasher.update(chunk)
                        out_f.write(chunk)
                else:
                    raise StorageError(f"Unsupported source type: {type(source)}")

                out_f.flush()
                os.fsync(out_f.fileno())

            sha256_digest = hasher.hexdigest()

            # 6. Inspect tabular structure (row count and column names)
            row_count, columns = self._inspect_csv_structure(staging_path)

            # 7. Atomically replace destination file
            os.replace(staging_path, dest_path)

            # Relative path from workspace root
            rel_path = str(dest_path.relative_to(self.workspace.workspace_root)).replace("\\", "/")

            # 8. Create database record
            now_dt = datetime.datetime.now(datetime.UTC)
            record = DatasetRecord(
                dataset_id=clean_dataset_id,
                project_id=valid_project_id,
                original_filename=clean_filename,
                storage_relative_path=rel_path,
                sha256_checksum=sha256_digest,
                file_size_bytes=total_bytes,
                row_count=row_count,
                column_count=len(columns),
                detected_columns=columns,
                ingested_at=now_dt,
            )

            # Insert into database (if overwrite, delete old record first)
            if overwrite and self.repository.get_dataset(clean_dataset_id) is not None:
                self.repository.delete_dataset(clean_dataset_id)

            try:
                self.repository.create_dataset(record)
            except Exception as db_exc:
                # If database insertion fails, remove newly moved file to prevent inconsistency
                if dest_path.exists():
                    with contextlib.suppress(OSError):
                        dest_path.unlink()
                raise StorageError(
                    f"Failed to record dataset metadata in database: {db_exc}"
                ) from db_exc

            return record, dest_path

        except Exception:
            # Clean up temporary staging file if still present
            if staging_path.exists():
                with contextlib.suppress(OSError):
                    staging_path.unlink()
            raise

    def get_dataset_path(self, dataset_id: str) -> pathlib.Path:
        """
        Resolve the managed filesystem path for a registered dataset.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.

        Returns
        -------
        pathlib.Path
            Absolute path to managed CSV file on disk.

        Raises
        ------
        DatasetNotFoundError
            If dataset is not found in repository or file is missing on disk.
        """
        valid_id = self.workspace.validate_identifier(dataset_id, "dataset_id")
        record = self.repository.get_dataset(valid_id)
        if record is None:
            raise DatasetNotFoundError(f"Dataset '{valid_id}' not found in metadata repository.")

        target = self.workspace.resolve_safe_path(
            record.storage_relative_path, self.workspace.workspace_root
        )
        if not target.exists() or not target.is_file():
            raise DatasetNotFoundError(
                f"Managed dataset file for '{valid_id}' missing on disk at '{target}'."
            )
        return target

    def verify_dataset_integrity(self, dataset_id: str) -> bool:
        """
        Verify that the physical managed file matches its recorded SHA-256 digest.

        Returns
        -------
        bool
            True if file exists and SHA-256 matches. False otherwise.
        """
        try:
            path = self.get_dataset_path(dataset_id)
            record = self.repository.get_dataset(dataset_id)
            if record is None:
                return False

            hasher = hashlib.sha256()
            with path.open("rb") as f:
                while chunk := f.read(CHUNK_SIZE_BYTES):
                    hasher.update(chunk)

            return hasher.hexdigest() == record.sha256_checksum
        except (DatasetNotFoundError, OSError):
            return False

    def delete_dataset(self, dataset_id: str, remove_file: bool = True) -> bool:
        """
        Delete a dataset from repository and optionally unlink physical file.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.
        remove_file : bool
            Whether to delete the managed file on disk.

        Returns
        -------
        bool
            True if dataset existed and was deleted. False otherwise.
        """
        record = self.repository.get_dataset(dataset_id)
        if record is None:
            return False

        if remove_file:
            try:
                target = self.workspace.resolve_safe_path(
                    record.storage_relative_path, self.workspace.workspace_root
                )
                if target.exists():
                    target.unlink()
            except (PathTraversalError, OSError):
                pass

        return self.repository.delete_dataset(dataset_id)

    # -----------------------------------------------------------------------
    # Helper Methods
    # -----------------------------------------------------------------------

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent directory traversal and enforce CSV extension."""
        if not filename or not filename.strip():
            raise PathTraversalError("Filename cannot be empty.")

        raw = filename.strip()
        if ".." in raw or "/" in raw or "\\" in raw:
            raise PathTraversalError(f"Invalid characters in filename: '{filename}'")

        clean = pathlib.Path(raw).name.strip()
        if not clean.lower().endswith(".csv"):
            raise StorageError(f"Only CSV files are supported, got '{clean}'.")

        return clean

    @staticmethod
    def _inspect_csv_structure(csv_path: pathlib.Path) -> tuple[int, list[str]]:
        """
        Inspect CSV file to extract header column names and count data rows
        without modifying content.

        Returns
        -------
        tuple[int, list[str]]
            (row_count, list_of_column_names)
        """
        try:
            with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    return 0, []

                columns = [col.strip() for col in header]
                row_count = sum(1 for _ in reader)
                return row_count, columns
        except Exception as exc:
            raise FileIntegrityError(f"Failed inspecting CSV structure: {exc}") from exc
