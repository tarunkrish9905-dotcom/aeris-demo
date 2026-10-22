"""
Tests for AERIS Analysis Run Artifact Storage Subsystem.

Verifies:
- Safe persistence under projects/<project_id>/runs/<analysis_id>/
- Cryptographic SHA-256 checksum generation and verification
- Faithful round-trip reconstruction of DataPipelineResult
- Representation of skipped stages as None (zero false reporting)
- Stripping of redundant telemetry DataFrames and private host paths
- Explicit exception handling for missing, corrupt, or incompatible artifacts
- Strict project/run ownership validation
"""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import tempfile
from collections.abc import Generator

import pandas as pd
import pytest

from aeris.core.evidence.pipeline import EvidenceSourceSelection, PipelineEvidenceRequest
from aeris.core.limits.models import (
    EngineeringLimitProfile,
    LimitSeverity,
    LimitType,
    ParameterLimitDefinition,
)
from aeris.core.pipeline.orchestrator import run_csv_pipeline
from aeris.core.recommendation.generation_models import RecommendationGenerationPolicy
from aeris.core.recommendation.pipeline import PipelineRecommendationRequest
from aeris.core.sop.models import ProcedureRequirementProfile
from aeris.core.vehicle.envelope_models import ConfigurationEnvelope
from aeris.core.vehicle.profile_models import VehicleProfile
from aeris.core.vehicle_evaluation.pipeline import PipelineVehicleEvaluationRequest
from aeris.storage.artifact_store import (
    ARTIFACT_FORMAT,
    CURRENT_FORMAT_VERSION,
    DEFAULT_ARTIFACT_FILENAME,
    AnalysisArtifactStore,
    sanitize_pipeline_result_for_storage,
)
from aeris.storage.exceptions import (
    ArtifactCorruptError,
    ArtifactIncompatibleError,
    ArtifactNotFoundError,
    EntityNotFoundError,
    ProjectMismatchError,
)
from aeris.storage.models import AnalysisRunRecord, DatasetRecord, ProjectRecord
from aeris.storage.repository import SQLiteMetadataRepository
from aeris.storage.workspace import WorkspaceManager


@pytest.fixture
def isolated_storage() -> (
    Generator[tuple[WorkspaceManager, SQLiteMetadataRepository, AnalysisArtifactStore], None, None]
):
    """Provide an isolated WorkspaceManager, SQLiteMetadataRepository, and AnalysisArtifactStore."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = WorkspaceManager(workspace_root=tmp_dir)
        workspace.init_workspace()

        db_path = pathlib.Path(tmp_dir) / "aeris_metadata.db"
        repository = SQLiteMetadataRepository(db_path=db_path, workspace=workspace)
        store = AnalysisArtifactStore(workspace=workspace, repository=repository)

        yield workspace, repository, store


@pytest.fixture
def seeded_run(
    isolated_storage: tuple[WorkspaceManager, SQLiteMetadataRepository, AnalysisArtifactStore],
) -> tuple[ProjectRecord, DatasetRecord, AnalysisRunRecord]:
    """Create and persist a valid Project, Dataset, and AnalysisRun in the repository."""
    _workspace, repository, _store = isolated_storage

    now = datetime.datetime.now(datetime.UTC)
    project = repository.create_project(
        ProjectRecord(
            project_id="proj_alpha",
            name="Project Alpha",
            description="Alpha test project",
            created_at=now,
            updated_at=now,
        )
    )

    dataset = repository.create_dataset(
        DatasetRecord(
            dataset_id="ds_flight_01",
            project_id=project.project_id,
            original_filename="flight_data.csv",
            storage_relative_path="datasets/flight_data.csv",
            file_size_bytes=1024,
            sha256_checksum="abc123def456",
            row_count=60,
            column_count=6,
            detected_columns=["timestamp", "altitude_ft", "airspeed_kts"],
            ingested_at=now,
        )
    )

    run = repository.create_analysis_run(
        AnalysisRunRecord(
            analysis_id="an_run_001",
            project_id=project.project_id,
            dataset_id=dataset.dataset_id,
            run_status="SUCCESS",
            triggered_at=datetime.datetime.now(datetime.UTC),
            completed_at=datetime.datetime.now(datetime.UTC),
            execution_duration_ms=450,
            total_issues_count=0,
            error_count=0,
            warning_count=0,
            result_summary={"stages": {"ingestion": "SUCCESS", "validation": "SUCCESS"}},
        )
    )

    return project, dataset, run


def test_sanitize_pipeline_result_strips_dataframes_and_host_paths(
    csv_fixtures_dir: pathlib.Path,
) -> None:
    """Verify serializer sanitizer strips in-memory DataFrames and private absolute host paths."""
    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    res = run_csv_pipeline(clean_csv)

    assert isinstance(res.ingestion.data, pd.DataFrame)
    assert res.ingestion.source_path is not None
    assert (
        str(clean_csv) in res.ingestion.source_path
        or "/" in res.ingestion.source_path
        or "\\" in res.ingestion.source_path
    )

    sanitized = sanitize_pipeline_result_for_storage(res)

    # In-memory DataFrame must be removed
    assert sanitized.ingestion.data is None
    # Private directory tree must be stripped, leaving only the clean filename
    assert sanitized.ingestion.source_path == "clean_flight.csv"
    # Status and validation structures must be preserved
    assert sanitized.status == res.status
    assert sanitized.validation is not None
    assert sanitized.validation.is_valid is True
    assert sanitized.validation.validated_row_count == 60


def test_artifact_store_and_load_round_trip(
    isolated_storage: tuple[WorkspaceManager, SQLiteMetadataRepository, AnalysisArtifactStore],
    seeded_run: tuple[ProjectRecord, DatasetRecord, AnalysisRunRecord],
    csv_fixtures_dir: pathlib.Path,
) -> None:
    """Verify round-trip fidelity between pipeline execution, disk persistence, and loading."""
    _workspace, repository, store = isolated_storage
    project, _dataset, run = seeded_run

    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    original_result = run_csv_pipeline(clean_csv)

    # 1. Store artifact via store
    updated_run, artifact_path = store.store_artifact(
        analysis_id=run.analysis_id,
        result=original_result,
        project_id=project.project_id,
    )

    assert artifact_path.exists()
    assert artifact_path.name == DEFAULT_ARTIFACT_FILENAME
    assert str(run.analysis_id) in str(artifact_path)

    # 2. Verify database record contains artifact linkage metadata
    assert updated_run.result_summary is not None
    artifact_meta = updated_run.result_summary.get("artifact")
    assert artifact_meta is not None
    expected_rel_path = f"runs/{run.analysis_id}/{DEFAULT_ARTIFACT_FILENAME}"
    assert artifact_meta["storage_relative_path"] == expected_rel_path
    assert artifact_meta["format"] == ARTIFACT_FORMAT
    assert artifact_meta["format_version"] == CURRENT_FORMAT_VERSION
    assert artifact_meta["file_size_bytes"] > 0
    assert len(artifact_meta["sha256_checksum"]) == 64

    # 3. Load artifact via store
    loaded_result = store.load_artifact(analysis_id=run.analysis_id, project_id=project.project_id)

    assert loaded_result.status == original_result.status
    assert original_result.validation is not None
    assert loaded_result.validation is not None
    assert loaded_result.validation.is_valid == original_result.validation.is_valid
    assert (
        loaded_result.validation.validated_row_count
        == original_result.validation.validated_row_count
    )
    assert original_result.quality is not None
    assert loaded_result.quality is not None
    assert loaded_result.quality.quality_score == original_result.quality.quality_score

    # 4. Load artifact via repository convenience method
    repo_loaded_result = repository.get_analysis_run_artifact(
        analysis_id=run.analysis_id,
        project_id=project.project_id,
    )
    assert repo_loaded_result.status == original_result.status
    assert repo_loaded_result.quality is not None
    assert repo_loaded_result.quality.quality_score == original_result.quality.quality_score


def test_artifact_preserves_skipped_stages_as_none(
    isolated_storage: tuple[WorkspaceManager, SQLiteMetadataRepository, AnalysisArtifactStore],
    seeded_run: tuple[ProjectRecord, DatasetRecord, AnalysisRunRecord],
    csv_fixtures_dir: pathlib.Path,
) -> None:
    """Verify stages that did not execute are reconstructed strictly as None without fabrication."""
    _workspace, _repository, store = isolated_storage
    project, _dataset, run = seeded_run

    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    res = run_csv_pipeline(clean_csv)

    # Default run executes only stages 1-3
    assert res.normalization is None
    assert res.parameter_validation is None
    assert res.association is None
    assert res.engineering is None
    assert res.statistics is None
    assert res.time_series is None
    assert res.anomaly is None
    assert res.expected_observed is None
    assert res.evidence is None
    assert res.contributor_analysis is None
    assert res.recommendation is None
    assert res.vehicle_evaluation is None

    store.store_artifact(run.analysis_id, res, project_id=project.project_id)
    reconstructed = store.load_artifact(run.analysis_id, project_id=project.project_id)

    # All unexecuted stages must remain None
    assert reconstructed.normalization is None
    assert reconstructed.parameter_validation is None
    assert reconstructed.association is None
    assert reconstructed.engineering is None
    assert reconstructed.statistics is None
    assert reconstructed.time_series is None
    assert reconstructed.anomaly is None
    assert reconstructed.expected_observed is None
    assert reconstructed.evidence is None
    assert reconstructed.contributor_analysis is None
    assert reconstructed.recommendation is None
    assert reconstructed.vehicle_evaluation is None


def test_artifact_checksum_verification_and_corruption_detection(
    isolated_storage: tuple[WorkspaceManager, SQLiteMetadataRepository, AnalysisArtifactStore],
    seeded_run: tuple[ProjectRecord, DatasetRecord, AnalysisRunRecord],
    csv_fixtures_dir: pathlib.Path,
) -> None:
    """Verify that tampering with an artifact on disk triggers ArtifactCorruptError."""
    _workspace, _repository, store = isolated_storage
    project, _dataset, run = seeded_run

    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    res = run_csv_pipeline(clean_csv)

    _updated_run, artifact_path = store.store_artifact(
        analysis_id=run.analysis_id,
        result=res,
        project_id=project.project_id,
    )

    # Tamper with the artifact content on disk (flip bytes)
    content = artifact_path.read_bytes()
    tampered_content = content + b"\n/* corrupted */"
    artifact_path.write_bytes(tampered_content)

    with pytest.raises(ArtifactCorruptError) as exc_info:
        store.load_artifact(run.analysis_id, project_id=project.project_id)

    assert "checksum verification failed" in str(exc_info.value).lower()


def test_artifact_syntax_corruption_detection(
    isolated_storage: tuple[WorkspaceManager, SQLiteMetadataRepository, AnalysisArtifactStore],
    seeded_run: tuple[ProjectRecord, DatasetRecord, AnalysisRunRecord],
    csv_fixtures_dir: pathlib.Path,
) -> None:
    """Verify that malformed JSON triggers ArtifactCorruptError."""
    _workspace, repository, store = isolated_storage
    project, _dataset, run = seeded_run

    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    res = run_csv_pipeline(clean_csv)

    _updated_run, artifact_path = store.store_artifact(
        analysis_id=run.analysis_id,
        result=res,
        project_id=project.project_id,
    )

    # Overwrite with malformed JSON and update checksum so checksum matches but JSON parsing fails
    corrupt_bytes = b"{ unquoted_key: invalid_json, "
    corrupt_checksum = hashlib.sha256(corrupt_bytes).hexdigest()
    artifact_path.write_bytes(corrupt_bytes)

    # Update database checksum to match corrupt bytes to isolate JSON parser check
    run_record = repository.get_analysis_run(run.analysis_id)
    assert run_record is not None
    summary = dict(run_record.result_summary or {})
    summary["artifact"]["sha256_checksum"] = corrupt_checksum
    repository.update_analysis_run(
        analysis_id=run.analysis_id,
        run_status=run_record.run_status,
        result_summary=summary,
    )

    with pytest.raises(ArtifactCorruptError) as exc_info:
        store.load_artifact(run.analysis_id, project_id=project.project_id)

    assert "json parsing failed" in str(exc_info.value).lower()


def test_artifact_incompatible_format_or_version(
    isolated_storage: tuple[WorkspaceManager, SQLiteMetadataRepository, AnalysisArtifactStore],
    seeded_run: tuple[ProjectRecord, DatasetRecord, AnalysisRunRecord],
    csv_fixtures_dir: pathlib.Path,
) -> None:
    """Verify unsupported envelope format or version raises ArtifactIncompatibleError."""
    _workspace, repository, store = isolated_storage
    project, _dataset, run = seeded_run

    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    res = run_csv_pipeline(clean_csv)

    _updated_run, artifact_path = store.store_artifact(
        run.analysis_id, res, project_id=project.project_id
    )

    # Load envelope and change format_version
    envelope = json.loads(artifact_path.read_text(encoding="utf-8"))
    envelope["format_version"] = "99.0"
    new_bytes = json.dumps(envelope).encode("utf-8")
    new_checksum = hashlib.sha256(new_bytes).hexdigest()
    artifact_path.write_bytes(new_bytes)

    # Update recorded checksum to match
    run_record = repository.get_analysis_run(run.analysis_id)
    assert run_record is not None
    summary = dict(run_record.result_summary or {})
    summary["artifact"]["sha256_checksum"] = new_checksum
    repository.update_analysis_run(
        analysis_id=run.analysis_id,
        run_status=run_record.run_status,
        result_summary=summary,
    )

    with pytest.raises(ArtifactIncompatibleError) as exc_info:
        store.load_artifact(run.analysis_id, project_id=project.project_id)

    assert "unsupported artifact version '99.0'" in str(exc_info.value).lower()


def test_artifact_missing_file_raises_artifact_not_found(
    isolated_storage: tuple[WorkspaceManager, SQLiteMetadataRepository, AnalysisArtifactStore],
    seeded_run: tuple[ProjectRecord, DatasetRecord, AnalysisRunRecord],
    csv_fixtures_dir: pathlib.Path,
) -> None:
    """Verify that deleting the physical artifact file raises ArtifactNotFoundError."""
    _workspace, _repository, store = isolated_storage
    project, _dataset, run = seeded_run

    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    res = run_csv_pipeline(clean_csv)

    _updated_run, artifact_path = store.store_artifact(
        run.analysis_id, res, project_id=project.project_id
    )
    artifact_path.unlink()

    with pytest.raises(ArtifactNotFoundError) as exc_info:
        store.load_artifact(run.analysis_id, project_id=project.project_id)

    assert "artifact file not found on disk" in str(exc_info.value).lower()


def test_artifact_unregistered_run_raises_artifact_not_found(
    isolated_storage: tuple[WorkspaceManager, SQLiteMetadataRepository, AnalysisArtifactStore],
    seeded_run: tuple[ProjectRecord, DatasetRecord, AnalysisRunRecord],
) -> None:
    """Verify querying run with no stored artifact metadata raises ArtifactNotFoundError."""
    _workspace, _repository, store = isolated_storage
    project, _dataset, run = seeded_run

    # run has no artifact registered in result_summary
    with pytest.raises(ArtifactNotFoundError) as exc_info:
        store.load_artifact(run.analysis_id, project_id=project.project_id)

    assert "no artifact metadata registered" in str(exc_info.value).lower()


def test_artifact_nonexistent_run_raises_entity_not_found(
    isolated_storage: tuple[WorkspaceManager, SQLiteMetadataRepository, AnalysisArtifactStore],
) -> None:
    """Verify querying an unknown analysis_id raises EntityNotFoundError."""
    _workspace, _repository, store = isolated_storage

    with pytest.raises(EntityNotFoundError) as exc_info:
        store.load_artifact("an_does_not_exist")

    assert "not found in metadata repository" in str(exc_info.value).lower()


def test_artifact_project_ownership_verification(
    isolated_storage: tuple[WorkspaceManager, SQLiteMetadataRepository, AnalysisArtifactStore],
    seeded_run: tuple[ProjectRecord, DatasetRecord, AnalysisRunRecord],
    csv_fixtures_dir: pathlib.Path,
) -> None:
    """Verify cross-project artifact access attempts are strictly rejected."""
    _workspace, repository, store = isolated_storage
    project, _dataset, run = seeded_run

    # Create a second project
    now = datetime.datetime.now(datetime.UTC)
    other_project = repository.create_project(
        ProjectRecord(
            project_id="proj_beta",
            name="Project Beta",
            created_at=now,
            updated_at=now,
        )
    )

    clean_csv = csv_fixtures_dir / "clean_flight.csv"
    res = run_csv_pipeline(clean_csv)
    store.store_artifact(run.analysis_id, res, project_id=project.project_id)

    # Attempt to load using mismatched project_id
    with pytest.raises(ProjectMismatchError) as exc_info:
        store.load_artifact(run.analysis_id, project_id=other_project.project_id)

    assert "belongs to project 'proj_alpha', not 'proj_beta'" in str(exc_info.value)


def test_artifact_full_pipeline_round_trip_with_advanced_stages(
    isolated_storage: tuple[WorkspaceManager, SQLiteMetadataRepository, AnalysisArtifactStore],
    seeded_run: tuple[ProjectRecord, DatasetRecord, AnalysisRunRecord],
    csv_fixtures_dir: pathlib.Path,
) -> None:
    """Verify multi-stage pipeline result reconstructs faithfully with advanced stages."""
    _workspace, _repository, store = isolated_storage
    project, _dataset, run = seeded_run

    clean_csv = csv_fixtures_dir / "clean_flight.csv"

    # Assemble request models for advanced stages
    veh_profile = VehicleProfile(
        profile_id="PROF-ADV-01",
        vehicle_id="AC-ADV-01",
        profile_version="1.0.0",
        name="Advanced Airframe",
        supported_parameter_ids=("airspeed_kts", "altitude_ft"),
    )
    envelope = ConfigurationEnvelope(
        configuration_id="CRUISE_ADV",
        vehicle_id="AC-ADV-01",
        profile_id="PROF-ADV-01",
        profile_version="1.0.0",
        name="Cruise Envelope",
    )
    limit_profile = EngineeringLimitProfile(
        limit_profile_id="LIM-ADV-01",
        vehicle_id="AC-ADV-01",
        profile_id="PROF-ADV-01",
        profile_version="1.0.0",
        limits=(
            ParameterLimitDefinition(
                limit_id="LIM-VNE",
                parameter_id="airspeed_kts",
                limit_type=LimitType.UPPER_BOUND,
                upper_bound=200.0,
                unit="kts",
                severity=LimitSeverity.WARNING,
                vehicle_id="AC-ADV-01",
                profile_id="PROF-ADV-01",
                profile_version="1.0.0",
                source_reference="AFM",
                description="Vne Limit",
            ),
        ),
    )
    proc_profile = ProcedureRequirementProfile(
        profile_id="PROC-ADV-01",
        document_id="SOP-ADV-01",
        document_version="1.0",
        vehicle_id="AC-ADV-01",
        profile_version="1.0.0",
        requirements=(),
    )

    veh_req = PipelineVehicleEvaluationRequest(
        vehicle_profile=veh_profile,
        configuration_envelope=envelope,
        limit_profile=limit_profile,
        procedure_profile=proc_profile,
        auto_extract_observations=True,
    )
    ev_req = PipelineEvidenceRequest(
        collection_id="EV-COLL-ADV",
        selections=EvidenceSourceSelection(include_vehicle_evaluation=True),
    )
    rec_req = PipelineRecommendationRequest(
        policy=RecommendationGenerationPolicy(enable_data_verification_rule=True),
        use_pipeline_evidence=True,
        use_pipeline_validation=True,
    )

    full_result = run_csv_pipeline(
        clean_csv,
        vehicle_evaluation_request=veh_req,
        evidence_request=ev_req,
        recommendation_request=rec_req,
    )

    assert full_result.vehicle_evaluation is not None
    assert full_result.evidence is not None
    assert full_result.recommendation is not None

    # Persist and reload
    store.store_artifact(run.analysis_id, full_result, project_id=project.project_id)
    loaded = store.load_artifact(run.analysis_id, project_id=project.project_id)

    assert loaded.status == full_result.status
    assert loaded.vehicle_evaluation is not None
    assert loaded.evidence is not None
    assert loaded.vehicle_evaluation.profile_id == "PROF-ADV-01"
    assert loaded.vehicle_evaluation.vehicle_id == "AC-ADV-01"
    assert loaded.vehicle_evaluation.configuration_id == "CRUISE_ADV"
