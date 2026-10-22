"""
AERIS Background Job Infrastructure.

Implements an in-process, thread-pool-based background task runner for long-running
analytical pipelines with authoritative stage request assembly, cooperative cancellation,
stage tracking, startup crash recovery, and non-leaking diagnostic error reporting.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import datetime
import pathlib
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

from aeris.core.evidence.pipeline import EvidenceSourceSelection, PipelineEvidenceRequest
from aeris.core.limits import load_limit_profile_from_yaml
from aeris.core.normalization.engine import NormalizationRequest
from aeris.core.parameter_validation.validator import ParameterValidationRequest
from aeris.core.pipeline.orchestrator import DataPipelineOrchestrator
from aeris.core.recommendation.pipeline import PipelineRecommendationRequest
from aeris.core.registry import ParameterRegistry, load_registry_from_yaml
from aeris.core.results.types import IssueSeverity, ResultStatus
from aeris.core.sop import load_procedure_requirement_profile_from_yaml
from aeris.core.vehicle import load_vehicle_profile_from_yaml
from aeris.core.vehicle_evaluation.pipeline import PipelineVehicleEvaluationRequest
from aeris.storage.dataset_store import ManagedDatasetStore
from aeris.storage.exceptions import DatasetNotFoundError
from aeris.storage.models import AnalysisRunRecord, BackgroundJobRecord
from aeris.storage.repository import SQLiteMetadataRepository
from aeris.storage.workspace import WorkspaceManager

if TYPE_CHECKING:
    from aeris.backend.models import StartAnalysisRequest
    from aeris.core.limits.models import EngineeringLimitProfile
    from aeris.core.sop.models import ProcedureRequirementProfile
    from aeris.core.vehicle.profile_models import VehicleProfile


class CancellationToken:
    """Thread-safe cooperative cancellation token."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self._event = threading.Event()

    def cancel(self) -> None:
        """Signal cancellation request."""
        self._event.set()

    def is_cancelled(self) -> bool:
        """Check whether cancellation was requested."""
        return self._event.is_set()


class JobRunner:
    """Authoritative background analysis job executor."""

    def __init__(
        self,
        workspace: WorkspaceManager,
        repository: SQLiteMetadataRepository,
        dataset_store: ManagedDatasetStore,
        max_workers: int = 2,
    ) -> None:
        """Initialize background job runner."""
        self.workspace = workspace
        self.repository = repository
        if getattr(self.repository, "workspace", None) is None:
            self.repository.workspace = self.workspace
        self.dataset_store = dataset_store
        self.max_workers = max_workers
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="aeris_worker",
        )
        self._cancellation_tokens: dict[str, CancellationToken] = {}
        self._lock = threading.Lock()

    def recover_interrupted_jobs(self) -> list[str]:
        """
        Recover jobs interrupted by previous system shutdown or crash.

        Transitions any jobs left in PENDING, QUEUED, or RUNNING state to FAILED
        with the authoritative error code ERR_SYSTEM_RESTARTED_DURING_EXECUTION.
        """
        interrupted_job_ids: list[str] = []
        now_dt = datetime.datetime.now(datetime.UTC)

        with self.repository._connection() as conn:
            cursor = conn.execute(
                """
                SELECT job_id, analysis_id, state
                FROM background_jobs
                WHERE state IN ('PENDING', 'QUEUED', 'RUNNING')
                """
            )
            rows = cursor.fetchall()

        for row in rows:
            job_id = row[0]
            analysis_id = row[1]
            interrupted_job_ids.append(job_id)

            self.repository.update_background_job(
                job_id=job_id,
                state="FAILED",
                current_stage="FAILED",
                error_message=(
                    "ERR_SYSTEM_RESTARTED_DURING_EXECUTION: System restarted during execution."
                ),
            )

            with self.repository._connection() as conn:
                conn.execute(
                    """
                    UPDATE analysis_runs
                    SET run_status = 'FAILED',
                        completed_at = ?
                    WHERE analysis_id = ?
                    """,
                    (now_dt.isoformat(), analysis_id),
                )
                conn.commit()

        return interrupted_job_ids

    def submit_analysis_job(
        self,
        project_id: str,
        request: StartAnalysisRequest,
        **custom_stage_requests: Any,
    ) -> tuple[AnalysisRunRecord, BackgroundJobRecord]:
        """
        Create persistent tracking records and submit analysis to background thread pool.

        Parameters
        ----------
        project_id : str
            Parent project identifier.
        request : StartAnalysisRequest
            Analysis configuration specifying dataset and optional profiles.
        custom_stage_requests : Any
            Optional direct stage requests for targeted pipeline execution.

        Returns
        -------
        tuple[AnalysisRunRecord, BackgroundJobRecord]
            The initialized analysis and background job records.
        """
        now_dt = datetime.datetime.now(datetime.UTC)
        analysis_id = f"run_{uuid.uuid4().hex[:12]}"
        job_id = f"job_{uuid.uuid4().hex[:12]}"

        # 1. Create AnalysisRunRecord
        run_record = AnalysisRunRecord(
            analysis_id=analysis_id,
            project_id=project_id,
            dataset_id=request.dataset_id,
            run_status="PENDING",
            triggered_at=now_dt,
            completed_at=None,
            execution_duration_ms=None,
            total_issues_count=0,
            error_count=0,
            warning_count=0,
            result_summary=None,
            provenance_id=None,
        )
        self.repository.create_analysis_run(run_record)

        # 2. Create BackgroundJobRecord
        job_record = BackgroundJobRecord(
            job_id=job_id,
            analysis_id=analysis_id,
            state="QUEUED",
            current_stage="INITIALIZING",
            completed_stages_count=0,
            total_stages_count=15,
            progress_percent=0,
            error_message=None,
            created_at=now_dt,
            updated_at=now_dt,
        )
        self.repository.create_background_job(job_record)

        # 3. Create cooperative cancellation token
        token = CancellationToken(job_id)
        with self._lock:
            self._cancellation_tokens[job_id] = token

        # 4. Dispatch to worker pool
        self._executor.submit(
            self._worker_execute_pipeline,
            job_id=job_id,
            analysis_id=analysis_id,
            project_id=project_id,
            token=token,
            request=request,
            dataset_id=request.dataset_id,
            **custom_stage_requests,
        )

        return run_record, job_record

    def cancel_job(self, job_id: str) -> bool:
        """
        Request cooperative cancellation of a queued or running background job.

        Returns
        -------
        bool
            True if cancellation was signaled, False if job does not exist or already finished.
        """
        with self._lock:
            job = self.repository.get_background_job(job_id)
            if job is None:
                return False

            if job.state in ("COMPLETED", "SUCCEEDED", "FAILED", "CANCELLED"):
                return False

            token = self._cancellation_tokens.get(job_id)
            if token:
                token.cancel()

            self.repository.update_background_job(
                job_id=job_id,
                state="CANCELLED",
                current_stage="CANCELLED",
                error_message="Job was cancelled by user request.",
            )

            now_dt = datetime.datetime.now(datetime.UTC)
            with self.repository._connection() as conn:
                conn.execute(
                    """
                    UPDATE analysis_runs
                    SET run_status = 'CANCELLED',
                        completed_at = ?
                    WHERE analysis_id = ?
                    """,
                    (now_dt.isoformat(), job.analysis_id),
                )
                conn.commit()

            return True

    def get_job(self, job_id: str) -> BackgroundJobRecord | None:
        """Retrieve background job by job_id."""
        return self.repository.get_background_job(job_id)

    def shutdown(self, wait: bool = True) -> None:
        """Cleanly shut down background thread pool."""
        with self._lock:
            for token in self._cancellation_tokens.values():
                token.cancel()
        self._executor.shutdown(wait=wait, cancel_futures=True)

    # -----------------------------------------------------------------------
    # Authoritative Request Assembly Helpers
    # -----------------------------------------------------------------------

    def _assemble_normalization_request(self, dataset_id: str) -> NormalizationRequest | None:
        """Build NormalizationRequest from dataset parameter mappings."""
        mappings = self.repository.get_parameter_mappings(dataset_id)
        if not mappings:
            return None

        norm_req = NormalizationRequest()
        for m in mappings:
            norm_req.rename_column(m.raw_column_name, m.target_parameter_id)
            if m.is_timestamp:
                norm_req.standardize_timestamp(m.target_parameter_id)
            if m.data_type and m.data_type.lower() in ("float", "numeric"):
                norm_req.convert_type(m.target_parameter_id, float)
            elif m.data_type and m.data_type.lower() == "int":
                norm_req.convert_type(m.target_parameter_id, int)
        return norm_req

    def _resolve_profile_path(
        self,
        project_id: str,
        profile_id: str,
        expected_type: str | None = None,
    ) -> pathlib.Path | None:
        """Resolve a configuration profile YAML file path safely within project workspace."""
        profiles = self.repository.list_configuration_profiles(project_id)
        for p in profiles:
            clean_name = p.profile_identifier.replace(".yaml", "").replace(".yml", "")
            if (
                p.config_id == profile_id
                or p.profile_identifier == profile_id
                or clean_name == profile_id
                or (
                    expected_type
                    and p.profile_type == expected_type
                    and clean_name.startswith(profile_id)
                )
            ):
                path = self.workspace.resolve_safe_path(
                    p.storage_relative_path, self.workspace.workspace_root
                )
                if path.exists() and path.is_file():
                    return path

        direct = pathlib.Path(profile_id)
        if direct.exists() and direct.is_file():
            return direct

        return None

    def _resolve_vehicle_profile(
        self,
        project_id: str,
        profile_id: str | None,
    ) -> VehicleProfile | None:
        if not profile_id:
            return None
        path = self._resolve_profile_path(project_id, profile_id, "VEHICLE")
        if path is None:
            return None
        try:
            return load_vehicle_profile_from_yaml(path)
        except Exception:
            return None

    def _resolve_limit_profile(
        self,
        project_id: str,
        profile_id: str | None,
    ) -> EngineeringLimitProfile | None:
        if not profile_id:
            return None
        path = self._resolve_profile_path(project_id, profile_id, "LIMIT")
        if path is None:
            return None
        try:
            return load_limit_profile_from_yaml(path)
        except Exception:
            return None

    def _resolve_procedure_profile(
        self,
        project_id: str,
        profile_id: str | None,
    ) -> ProcedureRequirementProfile | None:
        if not profile_id:
            return None
        path = self._resolve_profile_path(project_id, profile_id, "SOP")
        if path is None:
            return None
        try:
            return load_procedure_requirement_profile_from_yaml(path)
        except Exception:
            return None

    def _resolve_registry_and_validation(
        self,
        project_id: str,
        dataset_id: str,
    ) -> tuple[ParameterRegistry | None, ParameterValidationRequest | None]:
        reg_profiles = self.repository.list_configuration_profiles(
            project_id, profile_type="REGISTRY"
        )
        registry = None
        if reg_profiles:
            path = self.workspace.resolve_safe_path(
                reg_profiles[0].storage_relative_path, self.workspace.workspace_root
            )
            if path.exists():
                try:
                    registry = load_registry_from_yaml(path)
                except Exception:
                    registry = None

        param_req = None
        if registry is not None:
            mappings = self.repository.get_parameter_mappings(dataset_id)
            if mappings:
                param_req = ParameterValidationRequest(
                    requested_parameters=[m.target_parameter_id for m in mappings]
                )

        return registry, param_req

    def _assemble_vehicle_evaluation_request(
        self,
        project_id: str,
        request: StartAnalysisRequest,
        parameter_registry: ParameterRegistry | None = None,
    ) -> PipelineVehicleEvaluationRequest | None:
        has_any_profile_id = (
            request.vehicle_profile_id is not None
            or request.limit_profile_id is not None
            or request.procedure_profile_id is not None
        )
        if not has_any_profile_id:
            return None

        veh_prof = self._resolve_vehicle_profile(project_id, request.vehicle_profile_id)
        lim_prof = self._resolve_limit_profile(project_id, request.limit_profile_id)
        proc_prof = self._resolve_procedure_profile(project_id, request.procedure_profile_id)

        return PipelineVehicleEvaluationRequest(
            vehicle_profile=veh_prof,
            limit_profile=lim_prof,
            procedure_profile=proc_prof,
            parameter_registry=parameter_registry,
            auto_extract_observations=True,
        )

    def _assemble_evidence_request(
        self,
        analysis_id: str,
        vehicle_eval_req: PipelineVehicleEvaluationRequest | None,
    ) -> PipelineEvidenceRequest | None:
        if vehicle_eval_req is not None:
            return PipelineEvidenceRequest(
                collection_id=f"ev_coll_{analysis_id}",
                selections=EvidenceSourceSelection(
                    include_vehicle_evaluation=True,
                ),
            )
        return None

    def _assemble_recommendation_request(
        self,
        vehicle_eval_req: PipelineVehicleEvaluationRequest | None,
        evidence_req: PipelineEvidenceRequest | None,
    ) -> PipelineRecommendationRequest | None:
        if vehicle_eval_req is not None or evidence_req is not None:
            return PipelineRecommendationRequest(
                use_pipeline_validation=True,
                use_pipeline_quality=True,
                use_pipeline_evidence=evidence_req is not None,
                use_pipeline_vehicle_evaluation=vehicle_eval_req is not None,
            )
        return None

    # -----------------------------------------------------------------------
    # Internal Worker Pipeline Execution
    # -----------------------------------------------------------------------

    def _worker_execute_pipeline(
        self,
        job_id: str,
        analysis_id: str,
        project_id: str,
        token: CancellationToken,
        request: StartAnalysisRequest | None = None,
        dataset_id: str | None = None,
        **custom_stage_requests: Any,
    ) -> None:
        """Worker thread entry point executing the AERIS analytical pipeline."""
        start_time = time.perf_counter()

        with self._lock:
            if token.is_cancelled():
                return

            current_job = self.repository.get_background_job(job_id)
            if current_job and current_job.state == "CANCELLED":
                return

            self.repository.update_background_job(
                job_id=job_id,
                state="RUNNING",
                current_stage="INITIALIZING",
                completed_stages_count=0,
                progress_percent=5,
            )

        try:
            actual_dataset_id = request.dataset_id if request else (dataset_id or "")
            if not actual_dataset_id:
                raise DatasetNotFoundError(
                    "No dataset_id supplied for analysis pipeline execution."
                )

            # 1. Resolve dataset managed path
            dataset_path = self.dataset_store.get_dataset_path(actual_dataset_id)

            if token.is_cancelled():
                return

            # 2. Assemble Stage 4 (Normalization) request from parameter mappings
            norm_req = custom_stage_requests.get("normalization_request")
            if norm_req is None:
                norm_req = self._assemble_normalization_request(actual_dataset_id)

            # 3. Resolve Stage 5 (Parameter Registry & Semantic Validation)
            registry = custom_stage_requests.get("parameter_registry")
            param_val_req = custom_stage_requests.get("parameter_validation_request")
            if registry is None:
                registry, auto_param_val = self._resolve_registry_and_validation(
                    project_id, actual_dataset_id
                )
                if param_val_req is None:
                    param_val_req = auto_param_val

            # 4. Assemble Stage 12 (Vehicle Evaluation)
            veh_eval_req = custom_stage_requests.get("vehicle_evaluation_request")
            if veh_eval_req is None and request is not None:
                veh_eval_req = self._assemble_vehicle_evaluation_request(
                    project_id=project_id,
                    request=request,
                    parameter_registry=registry,
                )

            # 5. Assemble Stage 13 (Evidence Construction)
            ev_req = custom_stage_requests.get("evidence_request")
            if ev_req is None:
                ev_req = self._assemble_evidence_request(analysis_id, veh_eval_req)

            # 6. Assemble Stage 15 (Recommendation Generation)
            rec_req = custom_stage_requests.get("recommendation_request")
            if rec_req is None:
                rec_req = self._assemble_recommendation_request(veh_eval_req, ev_req)

            # Other custom stage requests passed explicitly
            assoc_req = custom_stage_requests.get("association_request")
            eng_req = custom_stage_requests.get("engineering_request")
            stat_req = custom_stage_requests.get("statistics_request")
            ts_req = custom_stage_requests.get("time_series_request")
            anom_req = custom_stage_requests.get("anomaly_request")
            eo_req = custom_stage_requests.get("expected_observed_request")
            contrib_req = custom_stage_requests.get("contributor_analysis_request")

            if token.is_cancelled():
                return

            # Update progress: Pipeline executing
            self.repository.update_background_job(
                job_id=job_id,
                state="RUNNING",
                current_stage="PIPELINE_EXECUTION",
                completed_stages_count=1,
                progress_percent=25,
            )

            # 7. Execute DataPipelineOrchestrator
            orchestrator = DataPipelineOrchestrator()
            result = orchestrator.run_csv_pipeline(
                file_path=dataset_path,
                normalization_request=norm_req,
                parameter_registry=registry,
                parameter_validation_request=param_val_req,
                association_request=assoc_req,
                engineering_request=eng_req,
                statistics_request=stat_req,
                time_series_request=ts_req,
                anomaly_request=anom_req,
                expected_observed_request=eo_req,
                vehicle_evaluation_request=veh_eval_req,
                evidence_request=ev_req,
                contributor_analysis_request=contrib_req,
                recommendation_request=rec_req,
            )

            if token.is_cancelled():
                return

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            end_dt = datetime.datetime.now(datetime.UTC)

            # 8. Accurately map executed stages and statuses
            stage_map: dict[str, Any] = {
                "ingestion": result.ingestion,
                "validation": result.validation,
                "quality": result.quality,
                "normalization": result.normalization,
                "parameter_validation": result.parameter_validation,
                "association": result.association,
                "engineering": result.engineering,
                "statistics": result.statistics,
                "time_series": result.time_series,
                "anomaly": result.anomaly,
                "expected_observed": result.expected_observed,
                "vehicle_evaluation": result.vehicle_evaluation,
                "evidence": result.evidence,
                "contributor_analysis": result.contributor_analysis,
                "recommendation": result.recommendation,
            }

            stage_statuses = {
                name: (res.status.value if res is not None else "SKIPPED")
                for name, res in stage_map.items()
            }

            completed_stages = [
                name
                for name, res in stage_map.items()
                if res is not None
                and res.status
                in (ResultStatus.SUCCESS, ResultStatus.WARNING, ResultStatus.PARTIAL)
            ]
            completed_stages_count = len(completed_stages)

            total_issues = len(result.issues)
            error_count = sum(1 for i in result.issues if i.severity == IssueSeverity.ERROR)
            warning_count = sum(1 for i in result.issues if i.severity == IssueSeverity.WARNING)

            # 9. Result Summary
            result_summary: dict[str, Any] = {
                "overall_status": result.status.value,
                "stages": stage_statuses,
                "completed_stages": completed_stages,
                "completed_stages_count": completed_stages_count,
                "total_stages_count": 15,
                "total_issues": total_issues,
                "error_count": error_count,
                "warning_count": warning_count,
            }

            # 10. Update AnalysisRunRecord with execution results
            terminal_run_status = (
                "FAILED" if result.status == ResultStatus.ERROR else "COMPLETED"
            )
            self.repository.update_analysis_run(
                analysis_id=analysis_id,
                run_status=terminal_run_status,
                completed_at=end_dt,
                execution_duration_ms=elapsed_ms,
                total_issues_count=total_issues,
                error_count=error_count,
                warning_count=warning_count,
                result_summary=result_summary,
            )

            # 11. Persist actual pipeline result artifact to workspace disk
            self.repository.update_background_job(
                job_id=job_id,
                state="RUNNING",
                current_stage="PERSISTING_ARTIFACT",
                progress_percent=85,
            )

            self.repository.save_analysis_run_artifact(
                analysis_id=analysis_id,
                result=result,
                project_id=project_id,
                workspace=self.workspace,
                overwrite=True,
            )

            if token.is_cancelled():
                return

            # 12. Finalize BackgroundJobRecord with accurate stage metrics
            terminal_state = (
                "FAILED" if result.status == ResultStatus.ERROR else "COMPLETED"
            )
            current_stage = (
                "FAILED" if result.status == ResultStatus.ERROR else "COMPLETED"
            )

            self.repository.update_background_job(
                job_id=job_id,
                state=terminal_state,
                current_stage=current_stage,
                completed_stages_count=completed_stages_count,
                progress_percent=100,
                error_message=None
                if result.status != ResultStatus.ERROR
                else "Pipeline execution flagged errors.",
            )

        except Exception as exc:
            if token.is_cancelled():
                return

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            end_dt = datetime.datetime.now(datetime.UTC)
            error_desc = f"{type(exc).__name__}: {exc!s}"

            # Update AnalysisRunRecord to FAILED
            with contextlib.suppress(Exception):
                self.repository.update_analysis_run(
                    analysis_id=analysis_id,
                    run_status="FAILED",
                    completed_at=end_dt,
                    execution_duration_ms=elapsed_ms,
                    total_issues_count=1,
                    error_count=1,
                    warning_count=0,
                    result_summary={"overall_status": "FAILED", "error": error_desc},
                )

            # Update BackgroundJobRecord to FAILED
            with contextlib.suppress(Exception):
                self.repository.update_background_job(
                    job_id=job_id,
                    state="FAILED",
                    current_stage="FAILED",
                    completed_stages_count=0,
                    progress_percent=100,
                    error_message=f"Pipeline execution failed: {error_desc}",
                )
        finally:
            with self._lock:
                self._cancellation_tokens.pop(job_id, None)
