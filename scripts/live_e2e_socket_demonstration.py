"""
Live Network Socket End-to-End Demonstration for AERIS.

Spawns a real Uvicorn ASGI server bound to 127.0.0.1 on an ephemeral port,
connects via an independent HTTP client (httpx), and executes the entire
analytical workflow across live network sockets:
1. Health and Version Verification
2. Dash UI Root & Layout Verification
3. Project Creation
4. Raw Telemetry Ingestion (Streaming Multi-part)
5. Parameter Semantic Mapping
6. 15-Stage Pipeline Analysis & Polling
7. Interactive Plotly Figure Retrieval
8. Engineering Report Export & SHA-256 Download Verification
9. Clean Server Shutdown
"""

from __future__ import annotations

import hashlib
import io
import pathlib
import sys
import tempfile
import threading
import time
from typing import Any

import httpx
import uvicorn

from aeris.backend.app import create_app
from aeris.backend.config import BackendConfig
from aeris.frontend.app import create_dash_app, mount_dash_app

TELEMETRY_CSV = (
    b"timestamp,indicated_airspeed,pressure_altitude,pitch_rate,roll_rate,normal_acceleration\n"
    b"2026-04-01T14:00:00Z,220.0,15000.0,0.1,0.0,1.00\n"
    b"2026-04-01T14:00:01Z,221.5,15030.0,0.5,0.1,1.02\n"
    b"2026-04-01T14:00:02Z,223.0,15060.0,1.0,0.2,1.05\n"
    b"2026-04-01T14:00:03Z,224.2,15090.0,1.4,0.1,1.07\n"
    b"2026-04-01T14:00:04Z,225.0,15120.0,0.9,-0.1,1.04\n"
    b"2026-04-01T14:00:05Z,225.5,15150.0,0.2,-0.2,1.01\n"
)


class ServerThread(threading.Thread):
    """Run Uvicorn server in a background thread."""

    def __init__(self, app: Any, host: str, port: int) -> None:
        super().__init__(daemon=True)
        self.server_config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="error",
        )
        self.server = uvicorn.Server(self.server_config)

    def run(self) -> None:
        self.server.run()

    def stop(self) -> None:
        self.server.should_exit = True


def run_live_demonstration() -> int:
    """Execute complete live socket integration verification."""
    print("=" * 72)
    print(" AERIS LIVE END-TO-END NETWORK SOCKET INTEGRATION AUDIT")
    print("=" * 72)

    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_dir = pathlib.Path(tmp_dir).resolve()
        host = "127.0.0.1"
        port = 8765
        base_url = f"http://{host}:{port}"

        print(f"[*] Workspace Root : {workspace_dir}")
        print(f"[*] Bind Address   : {base_url} (Loopback only)")

        # 1. Build application with mounted Dash frontend
        config = BackendConfig(
            workspace_root=workspace_dir,
            host=host,
            port=port,
        )
        fastapi_app = create_app(config=config, include_frontend=False)
        dash_app = create_dash_app()
        mount_dash_app(fastapi_app, dash_app=dash_app)

        # 2. Start Uvicorn in background thread
        server_thread = ServerThread(fastapi_app, host, port)
        server_thread.start()

        # Wait for socket to become available
        client = httpx.Client(base_url=base_url, timeout=30.0)
        healthy = False
        for _ in range(50):
            try:
                r = client.get("/api/v1/health")
                if r.status_code == 200:
                    healthy = True
                    break
            except Exception:
                time.sleep(0.1)

        if not healthy:
            print("[FAIL] Server failed to bind to socket within 5 seconds.")
            server_thread.stop()
            return 1

        print("[PASS] Server successfully bound and healthy on loopback socket.")

        try:
            # 3. Step 1: Health & Version
            v_resp = client.get("/api/v1/version")
            assert v_resp.status_code == 200, f"Version returned {v_resp.status_code}"
            print(f"[PASS] Version Verified: {v_resp.json()}")

            # 4. Step 2: Dash Root & Layout
            dash_resp = client.get("/")
            assert dash_resp.status_code == 200
            assert "html" in dash_resp.text.lower()
            layout_resp = client.get("/_dash-layout")
            assert layout_resp.status_code == 200
            print("[PASS] Dash Frontend Root & Layout Verified.")

            # 5. Step 3: Project Creation
            p_resp = client.post(
                "/api/v1/projects",
                json={
                    "name": "Live Socket Flight Demonstration",
                    "description": "Real-time network socket end-to-end integration test",
                    "analyst_name": "Chief Flight Test Engineer",
                },
            )
            assert p_resp.status_code == 201, f"Project create failed: {p_resp.text}"
            project_id = p_resp.json()["project_id"]
            print(f"[PASS] Project Created: {project_id}")

            # 6. Step 4: Telemetry Upload
            files = {"file": ("flight_log.csv", io.BytesIO(TELEMETRY_CSV), "text/csv")}
            up_resp = client.post(f"/api/v1/projects/{project_id}/datasets/upload", files=files)
            assert up_resp.status_code == 201, f"Upload failed: {up_resp.text}"
            d_data = up_resp.json()
            dataset_id = d_data["dataset_id"]
            print(
                f"[PASS] Telemetry Uploaded: {dataset_id} "
                f"({d_data['row_count']} rows, SHA-256: {d_data['sha256_checksum'][:16]}...)"
            )

            # 7. Step 5: Parameter Mapping
            mapping_payload = {
                "mappings": [
                    {
                        "raw_column_name": "indicated_airspeed",
                        "target_parameter_id": "indicated_airspeed",
                        "is_timestamp": False,
                        "data_type": "float",
                    },
                    {
                        "raw_column_name": "pressure_altitude",
                        "target_parameter_id": "pressure_altitude",
                        "is_timestamp": False,
                        "data_type": "float",
                    },
                    {
                        "raw_column_name": "pitch_rate",
                        "target_parameter_id": "pitch_rate",
                        "is_timestamp": False,
                        "data_type": "float",
                    },
                ]
            }
            map_resp = client.put(
                f"/api/v1/projects/{project_id}/datasets/{dataset_id}/mappings",
                json=mapping_payload,
            )
            assert map_resp.status_code == 200, f"Mapping failed: {map_resp.text}"
            print("[PASS] Semantic Parameter Mappings Configured.")

            # 8. Step 6: Trigger 15-Stage Analysis Pipeline
            an_resp = client.post(
                f"/api/v1/projects/{project_id}/analyses",
                json={
                    "dataset_id": dataset_id,
                    "report_title": "Live Socket Test Report",
                },
            )
            assert an_resp.status_code == 202, f"Analysis trigger failed: {an_resp.text}"
            job_id = an_resp.json()["job_id"]
            analysis_id = an_resp.json()["analysis_id"]
            print(f"[PASS] Analysis Job Queued: {job_id} (Run: {analysis_id})")

            # 9. Step 7: Poll Job Status over Socket
            print("[*] Polling background worker stages...")
            final_job_data = None
            for _ in range(60):
                poll_resp = client.get(f"/api/v1/jobs/{job_id}")
                assert poll_resp.status_code == 200
                final_job_data = poll_resp.json()
                st = final_job_data["state"]
                pct = final_job_data["progress_percent"]
                stage = final_job_data["current_stage"]
                print(f"    -> Progress: {pct}% | Stage: {stage} | State: {st}")
                if st in ("COMPLETED", "FAILED", "CANCELLED"):
                    break
                time.sleep(0.15)

            assert final_job_data and final_job_data["state"] == "COMPLETED"
            assert final_job_data["completed_stages"] == 15
            print("[PASS] 15-Stage Pipeline Completed Successfully.")

            # 10. Step 8: Multi-Signal Plotly Figure
            multi_req = {
                "signal_ids": ["indicated_airspeed", "pressure_altitude"],
                "layout_mode": "STACKED",
                "show_limits": True,
                "show_events": True,
            }
            plot_resp = client.post(
                f"/api/v1/projects/{project_id}/analyses/{analysis_id}/plots/multi-signal",
                json=multi_req,
            )
            assert plot_resp.status_code == 200, f"Plot generation failed: {plot_resp.text}"
            fig_data = plot_resp.json()["figure_json"]
            assert "data" in fig_data and len(fig_data["data"]) >= 2
            print(f"[PASS] Plotly Figure Retrieved: {len(fig_data['data'])} trace panels.")

            # 11. Step 9: Report Generation & Download
            rep_gen = client.post(
                f"/api/v1/projects/{project_id}/analyses/{analysis_id}/reports",
                json={"report_format": "HTML", "report_title": "Live Flight Demonstration"},
            )
            assert rep_gen.status_code == 201, f"Report gen failed: {rep_gen.text}"
            rep_meta = rep_gen.json()
            report_id = rep_meta["report_id"]
            print(f"[PASS] Engineering Report Generated: {report_id}")

            dl_resp = client.get(f"/api/v1/reports/{report_id}/download")
            assert dl_resp.status_code == 200
            dl_bytes = dl_resp.content
            assert len(dl_bytes) == rep_meta["file_size_bytes"]
            dl_hash = hashlib.sha256(dl_bytes).hexdigest()
            assert dl_hash == rep_meta["sha256_checksum"]
            print(
                f"[PASS] Report Downloaded & SHA-256 Checksum Verified: "
                f"{len(dl_bytes):,} bytes | {dl_hash[:16]}..."
            )

            print("=" * 72)
            print(" ALL LIVE NETWORK SOCKET WORKFLOWS DEMONSTRABLE AND VERIFIED")
            print("=" * 72)
            return 0

        finally:
            server_thread.stop()
            server_thread.join(timeout=3.0)
            client.close()


if __name__ == "__main__":
    sys.exit(run_live_demonstration())
