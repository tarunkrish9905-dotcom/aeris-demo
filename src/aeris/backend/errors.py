"""
AERIS Backend Exception Handlers and Error Formatting.

Provides RFC 7807 structured problem details for all HTTP and domain errors
without leaking internal filesystem paths, stack traces, or environment secrets.
"""

from __future__ import annotations

import datetime
import re
import uuid

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from aeris.storage.exceptions import (
    DatasetNotFoundError,
    DuplicateEntityError,
    FileAlreadyExistsError,
    FileIntegrityError,
    InvalidIdentifierError,
    OversizedFileError,
    PathTraversalError,
    ProjectNotFoundError,
    StorageError,
    WorkspaceError,
)

_PATH_PATTERN = re.compile(r"([A-Za-z]:[/\\][^\s'\",]+|/[^\s'\",]+)")


def sanitize_error_message(message: str) -> str:
    """Strip internal local filesystem paths from user-facing error text."""
    return _PATH_PATTERN.sub("[managed_path]", message)


def make_error_payload(
    status_code: int,
    error_code: str,
    message: str,
    detail: str | None = None,
    request_id: str | None = None,
) -> dict[str, str | None]:
    """Build structured RFC 7807 problem details dictionary."""
    clean_message = sanitize_error_message(message)
    clean_detail = sanitize_error_message(detail) if detail else None
    return {
        "error_code": error_code,
        "message": clean_message,
        "detail": clean_detail,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "request_id": request_id or f"req_{uuid.uuid4().hex[:8]}",
    }


def register_exception_handlers(app: FastAPI) -> None:
    """Register domain and HTTP exception handlers on FastAPI application."""

    @app.exception_handler(ProjectNotFoundError)
    async def handle_project_not_found(request: Request, exc: ProjectNotFoundError) -> JSONResponse:
        payload = make_error_payload(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="ERR_PROJECT_NOT_FOUND",
            message=str(exc),
        )
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=payload)

    @app.exception_handler(DatasetNotFoundError)
    async def handle_dataset_not_found(request: Request, exc: DatasetNotFoundError) -> JSONResponse:
        payload = make_error_payload(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="ERR_DATASET_NOT_FOUND",
            message=str(exc),
        )
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=payload)

    @app.exception_handler(DuplicateEntityError)
    async def handle_duplicate_entity(request: Request, exc: DuplicateEntityError) -> JSONResponse:
        payload = make_error_payload(
            status_code=status.HTTP_409_CONFLICT,
            error_code="ERR_DUPLICATE_ENTITY",
            message=str(exc),
        )
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content=payload)

    @app.exception_handler(FileAlreadyExistsError)
    async def handle_file_already_exists(
        request: Request, exc: FileAlreadyExistsError
    ) -> JSONResponse:
        payload = make_error_payload(
            status_code=status.HTTP_409_CONFLICT,
            error_code="ERR_FILE_ALREADY_EXISTS",
            message=str(exc),
        )
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content=payload)

    @app.exception_handler(PathTraversalError)
    async def handle_path_traversal(request: Request, exc: PathTraversalError) -> JSONResponse:
        payload = make_error_payload(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="ERR_PATH_TRAVERSAL_DETECTED",
            message="Invalid or unsafe path characters detected in request.",
        )
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=payload)

    @app.exception_handler(InvalidIdentifierError)
    async def handle_invalid_identifier(
        request: Request, exc: InvalidIdentifierError
    ) -> JSONResponse:
        payload = make_error_payload(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="ERR_INVALID_IDENTIFIER",
            message=str(exc),
        )
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=payload)

    @app.exception_handler(OversizedFileError)
    async def handle_oversized_file(request: Request, exc: OversizedFileError) -> JSONResponse:
        payload = make_error_payload(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            error_code="ERR_FILE_TOO_LARGE",
            message=str(exc),
        )
        return JSONResponse(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, content=payload)

    @app.exception_handler(FileIntegrityError)
    async def handle_file_integrity(request: Request, exc: FileIntegrityError) -> JSONResponse:
        payload = make_error_payload(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="ERR_FILE_INTEGRITY_FAILED",
            message=str(exc),
        )
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=payload)

    @app.exception_handler(WorkspaceError)
    async def handle_workspace_error(request: Request, exc: WorkspaceError) -> JSONResponse:
        payload = make_error_payload(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="ERR_WORKSPACE_ERROR",
            message=str(exc),
        )
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=payload)

    @app.exception_handler(StorageError)
    async def handle_storage_error(request: Request, exc: StorageError) -> JSONResponse:
        payload = make_error_payload(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="ERR_STORAGE_ERROR",
            message=str(exc),
        )
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=payload)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Simplify validation error messages for client
        errors_summary = "; ".join(
            f"{'.'.join(str(loc) for loc in err.get('loc', []))}: {err.get('msg', '')}"
            for err in exc.errors()
        )
        payload = make_error_payload(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="ERR_REQUEST_VALIDATION_FAILED",
            message="Request body or parameter validation failed.",
            detail=errors_summary,
        )
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=payload)

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        detail_msg = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        payload = make_error_payload(
            status_code=exc.status_code,
            error_code=f"ERR_HTTP_{exc.status_code}",
            message=detail_msg,
        )
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.exception_handler(Exception)
    async def handle_generic_exception(request: Request, exc: Exception) -> JSONResponse:
        # Never leak traceback or raw exception internals in production API
        payload = make_error_payload(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="ERR_INTERNAL_SERVER_ERROR",
            message="An unexpected server error occurred. Please contact the system administrator.",
        )
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=payload)
