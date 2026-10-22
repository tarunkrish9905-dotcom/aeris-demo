"""
AERIS Backend Configuration Profiles Router.

Manages built-in and project-specific vehicle profiles, engineering limits, and SOP configurations.
"""

from __future__ import annotations

import datetime
import hashlib
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from aeris.backend.deps import get_repository, get_workspace
from aeris.backend.models import (
    BuiltinConfigItem,
    BuiltinConfigsResponse,
    ConfigUploadResponse,
    ProjectConfigsResponse,
)
from aeris.storage.exceptions import ProjectNotFoundError
from aeris.storage.models import ConfigurationProfileRecord
from aeris.storage.repository import SQLiteMetadataRepository
from aeris.storage.workspace import WorkspaceManager

router = APIRouter(tags=["Configurations"])

BUILTIN_PROFILES: list[BuiltinConfigItem] = [
    BuiltinConfigItem(
        profile_type="REGISTRY",
        profile_id="default_flight_telemetry_registry",
        name="Standard Flight Test Parameter Registry",
        description="Standard aero, propulsion, dynamics, and atmospheric flight parameters.",
    ),
    BuiltinConfigItem(
        profile_type="VEHICLE",
        profile_id="f16_flight_envelope_profile",
        name="F-16 High-Performance Fighter Aircraft Profile",
        description="Aerodynamic reference areas, weight limits, and flight envelope boundaries.",
    ),
    BuiltinConfigItem(
        profile_type="LIMIT",
        profile_id="mil_std_structural_limits",
        name="MIL-STD Structural & Dynamics Engineering Limits",
        description="Structural g-limits, airspeed dynamic pressure thresholds, and stall margins.",
    ),
    BuiltinConfigItem(
        profile_type="SOP",
        profile_id="flight_test_safety_sop_01",
        name="Standard Operating Procedure — Flight Flutter & Envelope Expansion",
        description="Pre-flight briefing, test card boundaries, and abort execution criteria.",
    ),
]


@router.get("/configs/built-in", response_model=BuiltinConfigsResponse)
async def list_builtin_configs() -> BuiltinConfigsResponse:
    """List built-in default parameter registries, vehicle profiles, and limits."""
    return BuiltinConfigsResponse(builtin_configs=BUILTIN_PROFILES)


@router.post(
    "/projects/{project_id}/configs/upload",
    response_model=ConfigUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_config_profile(
    project_id: str,
    profile_type: str = Form(...),
    file: UploadFile = File(...),
    workspace: WorkspaceManager = Depends(get_workspace),
    repository: SQLiteMetadataRepository = Depends(get_repository),
) -> ConfigUploadResponse:
    """Upload custom YAML configuration (Vehicle, Limit, SOP) into project workspace."""
    project = repository.get_project(project_id)
    if project is None:
        raise ProjectNotFoundError(f"Project '{project_id}' not found.")

    clean_filename = file.filename or "config.yaml"
    if not (clean_filename.lower().endswith(".yaml") or clean_filename.lower().endswith(".yml")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only YAML files (.yaml, .yml) are supported for configuration profiles.",
        )

    content = await file.read()
    sha256_digest = hashlib.sha256(content).hexdigest()
    config_id = f"cfg_{uuid.uuid4().hex[:12]}"

    # Save to project configs directory
    project_dir = workspace.get_project_dir(project_id)
    configs_dir = project_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    dest_file = configs_dir / f"{config_id}_{clean_filename}"
    dest_file.write_bytes(content)

    rel_path = str(dest_file.relative_to(workspace.workspace_root)).replace("\\", "/")
    now_dt = datetime.datetime.now(datetime.UTC)

    record = ConfigurationProfileRecord(
        config_id=config_id,
        project_id=project_id,
        profile_type=profile_type.upper().strip(),
        profile_identifier=clean_filename,
        storage_relative_path=rel_path,
        sha256_checksum=sha256_digest,
        loaded_at=now_dt,
    )
    repository.create_configuration_profile(record)

    return ConfigUploadResponse(
        config_id=record.config_id,
        project_id=record.project_id,
        profile_type=record.profile_type,
        profile_identifier=record.profile_identifier,
        sha256_checksum=record.sha256_checksum,
        loaded_at=record.loaded_at,
    )


@router.get("/projects/{project_id}/configs", response_model=ProjectConfigsResponse)
async def list_project_configs(
    project_id: str,
    repository: SQLiteMetadataRepository = Depends(get_repository),
) -> ProjectConfigsResponse:
    """List custom configuration profiles uploaded to a project."""
    project = repository.get_project(project_id)
    if project is None:
        raise ProjectNotFoundError(f"Project '{project_id}' not found.")

    configs = repository.list_configuration_profiles(project_id)
    items = [
        ConfigUploadResponse(
            config_id=c.config_id,
            project_id=c.project_id,
            profile_type=c.profile_type,
            profile_identifier=c.profile_identifier,
            sha256_checksum=c.sha256_checksum,
            loaded_at=c.loaded_at,
        )
        for c in configs
    ]
    return ProjectConfigsResponse(configs=items)
