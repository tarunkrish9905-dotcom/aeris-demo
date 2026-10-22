"""
Tests for AERIS Backend Plots, Visualizations, Configurations, and Reports Endpoints.
"""

from __future__ import annotations

import io
import pathlib
import tempfile
import time
from collections.abc import Generator

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from aeris.backend.app import create_app
from aeris.backend.config import BackendConfig

SAMPLE_TELEMETRY = (
    b"timestamp,altitude_ft,airspeed_kts,pitch_deg\n"
    b"2026-01-01T12:00:00Z,5000.0,150.2,2.5\n"
    b"2026-01-01T12:00:01Z,5010.0,151.0,2.6\n"
    b"2026-01-01T12:00:02Z,5020.0,151.8,2.7\n"
    b"2026-01-01T12:00:03Z,5030.0,152.5,2.8\n"
)


@pytest.fixture
def executed_setup() -> Generator[tuple[TestClient, str, str, str, pathlib.Path], None, None]:
    """Provide TestClient with a completed analysis run ready for plotting and reporting."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = pathlib.Path(tmp_dir)
        config = BackendConfig(workspace_root=root)
        app = create_app(config)

        with TestClient(app) as client:
            # 1. Project
            p_resp = client.post("/api/v1/projects", json={"name": "Viz Project"})
            project_id = p_resp.json()["project_id"]

            # 2. Dataset
            files = {"file": ("flight.csv", io.BytesIO(SAMPLE_TELEMETRY), "text/csv")}
            d_resp = client.post(f"/api/v1/projects/{project_id}/datasets/upload", files=files)
            dataset_id = d_resp.json()["dataset_id"]

            # 3. Submit analysis and wait for completion
            s_resp = client.post(
                f"/api/v1/projects/{project_id}/analyses",
                json={"dataset_id": dataset_id},
            )
            job_id = s_resp.json()["job_id"]
            analysis_id = s_resp.json()["analysis_id"]

            # Wait for job completion
            for _ in range(50):
                st = client.get(f"/api/v1/jobs/{job_id}").json()
                if st["state"] in ("COMPLETED", "FAILED"):
                    break
                time.sleep(0.1)

            yield client, project_id, dataset_id, analysis_id, root


def test_builtin_and_custom_configs(
    executed_setup: tuple[TestClient, str, str, str, pathlib.Path],
) -> None:
    """Verify built-in configurations list and custom profile upload."""
    client, project_id, _dataset_id, _analysis_id, _root = executed_setup

    # 1. List built-in configs
    resp = client.get("/api/v1/configs/built-in")
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert len(data["builtin_configs"]) >= 3
    types = {c["profile_type"] for c in data["builtin_configs"]}
    assert "VEHICLE" in types
    assert "LIMIT" in types

    # 2. Upload custom YAML configuration
    custom_yaml = b"vehicle_name: Custom Fighter\nmax_mach: 2.2\n"
    files = {"file": ("custom_envelope.yaml", io.BytesIO(custom_yaml), "application/x-yaml")}
    upload_resp = client.post(
        f"/api/v1/projects/{project_id}/configs/upload",
        data={"profile_type": "VEHICLE"},
        files=files,
    )
    assert upload_resp.status_code == status.HTTP_201_CREATED
    cfg_data = upload_resp.json()
    assert cfg_data["profile_type"] == "VEHICLE"
    assert cfg_data["profile_identifier"] == "custom_envelope.yaml"

    # 3. List project configs
    list_resp = client.get(f"/api/v1/projects/{project_id}/configs")
    assert list_resp.status_code == status.HTTP_200_OK
    assert len(list_resp.json()["configs"]) == 1


def test_time_series_plot_and_missing_param(
    executed_setup: tuple[TestClient, str, str, str, pathlib.Path],
) -> None:
    """Verify time-series Plotly JSON generation and missing parameter error."""
    client, project_id, _dataset_id, analysis_id, _root = executed_setup

    # Valid parameter
    resp = client.get(
        f"/api/v1/projects/{project_id}/analyses/{analysis_id}/plots/time-series?parameter_id=altitude_ft"
    )
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["parameter_id"] == "altitude_ft"
    assert "figure_json" in data
    assert "data" in data["figure_json"]
    assert "layout" in data["figure_json"]

    # Invalid parameter
    err_resp = client.get(
        f"/api/v1/projects/{project_id}/analyses/{analysis_id}/plots/time-series?parameter_id=nonexistent_signal"
    )
    assert err_resp.status_code == status.HTTP_400_BAD_REQUEST


def test_multi_signal_plot(
    executed_setup: tuple[TestClient, str, str, str, pathlib.Path],
) -> None:
    """Verify multi-signal coordinated panel figure generation with raw columns."""
    client, project_id, _dataset_id, analysis_id, _root = executed_setup

    payload = {
        "signal_ids": ["altitude_ft", "airspeed_kts"],
        "layout_mode": "STACKED",
        "show_limits": True,
        "show_events": False,
        "dark_mode": True,
    }
    resp = client.post(
        f"/api/v1/projects/{project_id}/analyses/{analysis_id}/plots/multi-signal",
        json=payload,
    )
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert "figure_json" in data
    assert len(data["figure_json"]["data"]) >= 2


def test_available_signals_discovery_and_auto_suggest(
    executed_setup: tuple[TestClient, str, str, str, pathlib.Path],
) -> None:
    """Verify signal discovery endpoint and canonical auto-suggestion for datasets."""
    client, project_id, dataset_id, analysis_id, _root = executed_setup

    # 1. Check auto-suggested mappings on dataset
    m_resp = client.get(f"/api/v1/projects/{project_id}/datasets/{dataset_id}/mappings")
    assert m_resp.status_code == status.HTTP_200_OK
    mapping_items = m_resp.json()["mappings"]
    airspeed_item = next((m for m in mapping_items if m["raw_column_name"] == "airspeed_kts"), None)
    assert airspeed_item is not None
    assert airspeed_item["target_parameter_id"] == "indicated_airspeed"
    assert airspeed_item["target_unit"] == "knots"

    # 2. Check signal discovery endpoint before explicit mappings
    sig_resp = client.get(f"/api/v1/projects/{project_id}/analyses/{analysis_id}/plots/signals")
    assert sig_resp.status_code == status.HTTP_200_OK
    sig_data = sig_resp.json()
    assert sig_data["dataset_id"] == dataset_id
    assert len(sig_data["signals"]) >= 3
    # defaults to available columns in dataset
    assert len(sig_data["default_signals"]) >= 1
    assert "altitude_ft" in sig_data["raw_columns"]


def test_parameter_mapping_resolution_in_plots(
    executed_setup: tuple[TestClient, str, str, str, pathlib.Path],
) -> None:
    """Verify that mapped canonical signals (IAS, Altitude) resolve and plot cleanly."""
    client, project_id, dataset_id, analysis_id, _root = executed_setup

    # 1. Save explicit parameter mappings
    mappings_payload = {
        "mappings": [
            {
                "raw_column_name": "airspeed_kts",
                "target_parameter_id": "indicated_airspeed",
                "source_unit": "knots",
                "target_unit": "knots",
                "is_timestamp": False,
                "data_type": "float",
            },
            {
                "raw_column_name": "altitude_ft",
                "target_parameter_id": "pressure_altitude",
                "source_unit": "ft",
                "target_unit": "ft",
                "is_timestamp": False,
                "data_type": "float",
            },
            {
                "raw_column_name": "timestamp",
                "target_parameter_id": "timestamp",
                "source_unit": "s",
                "target_unit": "s",
                "is_timestamp": True,
                "data_type": "datetime",
            },
        ]
    }
    save_resp = client.put(
        f"/api/v1/projects/{project_id}/datasets/{dataset_id}/mappings",
        json=mappings_payload,
    )
    assert save_resp.status_code == status.HTTP_200_OK

    # 2. Signal discovery now reflects mapped parameters and prefers IAS + Altitude
    sig_resp = client.get(f"/api/v1/projects/{project_id}/analyses/{analysis_id}/plots/signals")
    assert sig_resp.status_code == status.HTTP_200_OK
    sig_data = sig_resp.json()
    assert sig_data["default_signals"] == ["indicated_airspeed", "pressure_altitude"]

    # 3. Generate multi-signal plot with canonical parameters
    plot_payload = {
        "signal_ids": ["indicated_airspeed", "pressure_altitude"],
        "layout_mode": "STACKED",
        "show_limits": True,
        "show_events": True,
        "dark_mode": False,
    }
    plot_resp = client.post(
        f"/api/v1/projects/{project_id}/analyses/{analysis_id}/plots/multi-signal",
        json=plot_payload,
    )
    assert plot_resp.status_code == status.HTTP_200_OK
    plot_data = plot_resp.json()
    assert "figure_json" in plot_data
    assert len(plot_data["figure_json"]["data"]) == 2

    # 4. Generate single time-series plot with canonical parameter
    ts_resp = client.get(
        f"/api/v1/projects/{project_id}/analyses/{analysis_id}/plots/time-series?parameter_id=indicated_airspeed"
    )
    assert ts_resp.status_code == status.HTTP_200_OK
    ts_data = ts_resp.json()
    assert ts_data["parameter_id"] == "indicated_airspeed"
    assert ts_data["unit"] == "knots"


def test_missing_signal_returns_actionable_error(
    executed_setup: tuple[TestClient, str, str, str, pathlib.Path],
) -> None:
    """Verify that unmapped missing signals return clear, actionable error details."""
    client, project_id, _dataset_id, analysis_id, _root = executed_setup

    payload = {
        "signal_ids": ["unmapped_airspeed", "nonexistent_signal"],
        "layout_mode": "STACKED",
    }
    resp = client.post(
        f"/api/v1/projects/{project_id}/analyses/{analysis_id}/plots/multi-signal",
        json=payload,
    )
    assert resp.status_code == status.HTTP_400_BAD_REQUEST
    err_body = resp.json()
    err_text = err_body.get("message") or err_body.get("detail") or ""
    assert "unmapped_airspeed" in err_text
    assert "Available dataset columns" in err_text
    assert "Configure parameter mappings on the Datasets page" in err_text


def test_report_generation_listing_and_download(
    executed_setup: tuple[TestClient, str, str, str, pathlib.Path],
) -> None:
    """Verify generating HTML report, listing reports, and downloading standalone file."""
    client, project_id, _dataset_id, analysis_id, _root = executed_setup

    # 1. Generate HTML report
    gen_payload = {
        "report_format": "HTML",
        "include_interactive_plots": True,
        "report_title": "Certified Flight Test Report",
    }
    gen_resp = client.post(
        f"/api/v1/projects/{project_id}/analyses/{analysis_id}/reports",
        json=gen_payload,
    )
    assert gen_resp.status_code == status.HTTP_201_CREATED
    r_data = gen_resp.json()
    report_id = r_data["report_id"]
    assert r_data["report_format"] == "HTML"
    assert r_data["file_size_bytes"] > 0

    # 2. List reports
    list_resp = client.get(f"/api/v1/projects/{project_id}/reports")
    assert list_resp.status_code == status.HTTP_200_OK
    assert list_resp.json()["total_count"] == 1

    # 3. Download report file (attachment disposition)
    dl_resp = client.get(f"/api/v1/reports/{report_id}/download")
    assert dl_resp.status_code == status.HTTP_200_OK
    assert "text/html" in dl_resp.headers["content-type"]
    assert "attachment" in dl_resp.headers.get("content-disposition", "")
    assert len(dl_resp.content) > 0
    assert b"<!DOCTYPE html>" in dl_resp.content or b"<html" in dl_resp.content

    # 4. Stream report file for sandboxed preview (inline disposition)
    preview_resp = client.get(f"/api/v1/reports/{report_id}/download?inline=true")
    assert preview_resp.status_code == status.HTTP_200_OK
    assert "text/html" in preview_resp.headers["content-type"]
    assert "inline" in preview_resp.headers.get("content-disposition", "")
    assert len(preview_resp.content) == len(dl_resp.content)
