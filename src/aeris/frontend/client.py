"""
AERIS Frontend REST API Client.

Provides a synchronous, strongly-typed HTTP client for Dash UI callbacks to
interact with the AERIS FastAPI backend service.
"""

from __future__ import annotations

import os
import re
from typing import Any

import httpx

_PATH_PATTERN = re.compile(r"([A-Za-z]:\\[^\s:;,]+|/[^\s:;,]+)")
_TRACEBACK_PATTERN = re.compile(r"Traceback \(most recent call last\):.*", re.DOTALL)


def sanitize_ui_error(message: str) -> str:
    """
    Remove internal filesystem paths, stack traces, and sensitive diagnostic tokens
    from error strings before presenting to the analyst UI.
    """
    lines = []
    for line in message.splitlines():
        trimmed = line.strip()
        if "Traceback (most recent call last):" in line or trimmed.startswith("File "):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    cleaned = _PATH_PATTERN.sub("[internal-path]", cleaned)
    return cleaned or "An unexpected server error occurred."


class AerisApiError(Exception):
    """Exception raised when an API request to the AERIS backend fails."""

    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        detail: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.message = sanitize_ui_error(message)
        self.detail = sanitize_ui_error(detail) if detail else None
        super().__init__(f"[{status_code}] {self.error_code}: {self.message}")


class AerisApiClient:
    """
    Synchronous HTTP client for the AERIS REST API.

    Designed for use within Plotly Dash callbacks with configurable base URL,
    bounded timeouts, and clean error transformation.
    """

    def __init__(
        self,
        base_url: str = os.environ.get("AERIS_API_BASE_URL", "http://127.0.0.1:8000/api/v1"),
        timeout: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = client or httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def close(self) -> None:
        """Close the underlying HTTP client session."""
        self._client.close()

    def __enter__(self) -> AerisApiClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Verify response status and parse JSON payload or raise AerisApiError."""
        if response.is_success:
            if not response.content:
                return {}
            try:
                res = response.json()
                if isinstance(res, dict):
                    return res
                return {"data": res}
            except Exception as exc:
                raise AerisApiError(
                    status_code=response.status_code,
                    error_code="ERR_JSON_DECODE",
                    message="Failed to decode response from AERIS backend.",
                    detail=str(exc),
                ) from exc

        # Handle RFC 7807 problem details or standard error
        try:
            err_data = response.json()
            error_code = str(err_data.get("error_code") or f"ERR_HTTP_{response.status_code}")
            message = str(err_data.get("message") or err_data.get("detail") or "Request failed.")
            detail = str(err_data.get("detail")) if err_data.get("detail") != message else None
        except Exception:
            error_code = f"ERR_HTTP_{response.status_code}"
            message = response.text or f"HTTP {response.status_code} error."
            detail = None

        raise AerisApiError(
            status_code=response.status_code,
            error_code=error_code,
            message=message,
            detail=detail,
        )

    # =========================================================================
    # Health & System
    # =========================================================================

    def get_health(self) -> dict[str, Any]:
        """Check system liveness and database connectivity."""
        try:
            resp = self._client.get("/health")
            return self._handle_response(resp)
        except httpx.RequestError as exc:
            raise AerisApiError(
                status_code=503,
                error_code="ERR_SERVICE_UNAVAILABLE",
                message="Cannot connect to AERIS backend service.",
                detail=str(exc),
            ) from exc

    def get_version(self) -> dict[str, Any]:
        """Retrieve software and API version details."""
        try:
            resp = self._client.get("/version")
            return self._handle_response(resp)
        except httpx.RequestError as exc:
            raise AerisApiError(
                status_code=503,
                error_code="ERR_SERVICE_UNAVAILABLE",
                message="Cannot connect to AERIS backend service.",
                detail=str(exc),
            ) from exc

    # =========================================================================
    # Projects
    # =========================================================================

    def list_projects(self, status: str | None = None) -> list[dict[str, Any]]:
        """List active or filtered projects."""
        params = {"status": status} if status else {}
        resp = self._client.get("/projects", params=params)
        data = self._handle_response(resp)
        projects = data.get("projects", [])
        return projects if isinstance(projects, list) else []

    def get_project(self, project_id: str) -> dict[str, Any]:
        """Retrieve single project details."""
        resp = self._client.get(f"/projects/{project_id}")
        return self._handle_response(resp)

    def create_project(
        self,
        name: str,
        description: str = "",
        analyst_name: str | None = None,
    ) -> dict[str, Any]:
        """Create a new flight analysis project."""
        payload = {
            "name": name,
            "description": description,
            "analyst_name": analyst_name,
        }
        resp = self._client.post("/projects", json=payload)
        return self._handle_response(resp)

    def update_project(
        self,
        project_id: str,
        name: str | None = None,
        description: str | None = None,
        analyst_name: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """Update existing project metadata."""
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if analyst_name is not None:
            payload["analyst_name"] = analyst_name
        if status is not None:
            payload["status"] = status
        resp = self._client.patch(f"/projects/{project_id}", json=payload)
        return self._handle_response(resp)

    def delete_project(self, project_id: str, purge: bool = False) -> dict[str, Any]:
        """Soft-delete or purge a project."""
        resp = self._client.delete(f"/projects/{project_id}", params={"purge": purge})
        return self._handle_response(resp)

    # =========================================================================
    # Datasets & Mapping
    # =========================================================================

    def list_datasets(self, project_id: str) -> list[dict[str, Any]]:
        """List telemetry datasets in a project."""
        resp = self._client.get(f"/projects/{project_id}/datasets")
        data = self._handle_response(resp)
        datasets = data.get("datasets", [])
        return datasets if isinstance(datasets, list) else []

    def upload_dataset(
        self,
        project_id: str,
        filename: str,
        content: bytes,
    ) -> dict[str, Any]:
        """Upload raw telemetry CSV file."""
        files = {"file": (filename, content, "text/csv")}
        resp = self._client.post(
            f"/projects/{project_id}/datasets/upload",
            files=files,
        )
        return self._handle_response(resp)

    def get_dataset_preview(
        self,
        project_id: str,
        dataset_id: str,
        rows: int = 50,
    ) -> dict[str, Any]:
        """Retrieve preview rows and detected column metadata for a dataset."""
        resp = self._client.get(
            f"/projects/{project_id}/datasets/{dataset_id}/preview",
            params={"rows": rows},
        )
        return self._handle_response(resp)

    def get_parameter_mappings(self, project_id: str, dataset_id: str) -> list[dict[str, Any]]:
        """Retrieve parameter mappings for a dataset."""
        resp = self._client.get(f"/projects/{project_id}/datasets/{dataset_id}/mappings")
        data = self._handle_response(resp)
        mappings = data.get("mappings", [])
        return mappings if isinstance(mappings, list) else []

    def save_parameter_mappings(
        self,
        project_id: str,
        dataset_id: str,
        mappings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Save verified column-to-parameter mappings."""
        payload = {"mappings": mappings}
        resp = self._client.put(
            f"/projects/{project_id}/datasets/{dataset_id}/mappings",
            json=payload,
        )
        return self._handle_response(resp)

    # =========================================================================
    # Configurations
    # =========================================================================

    def get_builtin_configs(self) -> list[dict[str, Any]]:
        """List built-in vehicle profiles and parameter registries."""
        resp = self._client.get("/configs/built-in")
        data = self._handle_response(resp)
        configs = data.get("builtin_configs", [])
        return configs if isinstance(configs, list) else []

    def list_project_configs(self, project_id: str) -> list[dict[str, Any]]:
        """List uploaded custom configurations for a project."""
        resp = self._client.get(f"/projects/{project_id}/configs")
        data = self._handle_response(resp)
        configs = data.get("configs", [])
        return configs if isinstance(configs, list) else []

    def upload_custom_config(
        self,
        project_id: str,
        profile_type: str,
        filename: str,
        content: bytes,
    ) -> dict[str, Any]:
        """Upload custom configuration YAML profile."""
        files = {"file": (filename, content, "application/x-yaml")}
        data = {"profile_type": profile_type}
        resp = self._client.post(
            f"/projects/{project_id}/configs/upload",
            files=files,
            data=data,
        )
        return self._handle_response(resp)

    # =========================================================================
    # Analyses & Background Jobs
    # =========================================================================

    def start_analysis(
        self,
        project_id: str,
        dataset_id: str,
        vehicle_profile_id: str | None = None,
        limit_profile_id: str | None = None,
        procedure_profile_id: str | None = None,
        report_title: str | None = None,
    ) -> dict[str, Any]:
        """Submit a new analysis run as a background job."""
        payload: dict[str, Any] = {"dataset_id": dataset_id}
        if vehicle_profile_id:
            payload["vehicle_profile_id"] = vehicle_profile_id
        if limit_profile_id:
            payload["limit_profile_id"] = limit_profile_id
        if procedure_profile_id:
            payload["procedure_profile_id"] = procedure_profile_id
        if report_title:
            payload["report_title"] = report_title

        resp = self._client.post(
            f"/projects/{project_id}/analyses",
            json=payload,
        )
        return self._handle_response(resp)

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Poll real-time execution progress of a background job."""
        resp = self._client.get(f"/jobs/{job_id}")
        return self._handle_response(resp)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Signal cooperative cancellation to a background job."""
        resp = self._client.post(f"/jobs/{job_id}/cancel")
        return self._handle_response(resp)

    def get_analysis_detail(self, project_id: str, analysis_id: str) -> dict[str, Any]:
        """Retrieve completed analysis summary and diagnostic metrics."""
        resp = self._client.get(f"/projects/{project_id}/analyses/{analysis_id}")
        return self._handle_response(resp)

    # =========================================================================
    # Plots & Visualizations
    # =========================================================================

    def get_time_series_plot(
        self,
        project_id: str,
        analysis_id: str,
        parameter_id: str,
        overlays: list[str] | None = None,
    ) -> dict[str, Any]:
        """Retrieve single-signal time-series Plotly figure structure."""
        params: dict[str, Any] = {"parameter_id": parameter_id}
        if overlays:
            params["overlays"] = overlays
        resp = self._client.get(
            f"/projects/{project_id}/analyses/{analysis_id}/plots/time-series",
            params=params,
        )
        return self._handle_response(resp)

    def get_multi_signal_plot(
        self,
        project_id: str,
        analysis_id: str,
        signal_ids: list[str],
        layout_mode: str = "STACKED",
        show_limits: bool = True,
        show_events: bool = True,
        dark_mode: bool = False,
    ) -> dict[str, Any]:
        """Generate multi-signal panel visualization figure."""
        payload = {
            "signal_ids": signal_ids,
            "layout_mode": layout_mode,
            "show_limits": show_limits,
            "show_events": show_events,
            "dark_mode": dark_mode,
        }
        resp = self._client.post(
            f"/projects/{project_id}/analyses/{analysis_id}/plots/multi-signal",
            json=payload,
        )
        return self._handle_response(resp)

    def get_available_signals(
        self,
        project_id: str,
        analysis_id: str,
    ) -> dict[str, Any]:
        """Retrieve plottable signals (mapped canonical and raw columns) and defaults."""
        resp = self._client.get(f"/projects/{project_id}/analyses/{analysis_id}/plots/signals")
        return self._handle_response(resp)

    # =========================================================================
    # Reports
    # =========================================================================

    def generate_report(
        self,
        project_id: str,
        analysis_id: str,
        report_format: str = "HTML",
        report_title: str | None = None,
        include_interactive_plots: bool = True,
    ) -> dict[str, Any]:
        """Trigger generation of an offline engineering report."""
        payload = {
            "report_format": report_format,
            "report_title": report_title,
            "include_interactive_plots": include_interactive_plots,
        }
        resp = self._client.post(
            f"/projects/{project_id}/analyses/{analysis_id}/reports",
            json=payload,
        )
        return self._handle_response(resp)

    def list_reports(self, project_id: str) -> list[dict[str, Any]]:
        """List generated engineering reports in a project."""
        resp = self._client.get(f"/projects/{project_id}/reports")
        data = self._handle_response(resp)
        reports = data.get("reports", [])
        return reports if isinstance(reports, list) else []

    def download_report(self, report_id: str) -> tuple[bytes, str]:
        """Download raw report file bytes and content type."""
        resp = self._client.get(f"/reports/{report_id}/download")
        if not resp.is_success:
            self._handle_response(resp)
        content_type = resp.headers.get("content-type", "application/octet-stream")
        return resp.content, content_type

