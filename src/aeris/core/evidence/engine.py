from aeris.core.anomaly.models import PipelineAnomalyDetectionResult
from aeris.core.engineering.models import EngineeringCalculationResult
from aeris.core.evidence.construction import EvidenceConstructionEngine
from aeris.core.evidence.pipeline import PipelineEvidenceRequest, PipelineEvidenceResult
from aeris.core.expected_observed.models import PipelineExpectedObservedResult
from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus
from aeris.core.statistics.models import StatisticalAnalysisResult
from aeris.core.vehicle_evaluation.models import VehicleEvaluationResult


class PipelineEvidenceEngine:
    """
    Integrates explicit Evidence construction into the AERIS pipeline.
    """

    @classmethod
    def process(
        cls,
        request: PipelineEvidenceRequest,
        parent_record_ids: list[str],
        source_uri: str,
        anomaly_result: PipelineAnomalyDetectionResult | None = None,
        engineering_results: list[EngineeringCalculationResult] | None = None,
        statistical_results: list[StatisticalAnalysisResult] | None = None,
        expected_observed_result: PipelineExpectedObservedResult | None = None,
        vehicle_evaluation_result: VehicleEvaluationResult | None = None,
    ) -> PipelineEvidenceResult:
        """Process explicit evidence requests against upstream pipeline results."""
        issues = []
        records = []

        # We create a top-level provenance for the integration stage
        # (not the evidence items themselves)
        prov = create_provenance(
            activity_name="pipeline_evidence_construction",
            epistemic_level=EpistemicLevel.CALCULATED,  # The act of aggregating the collection
            parent_record_ids=parent_record_ids,
            source_uri=source_uri,
        )

        # 1. Anomalies
        if request.selections.anomaly_ids:
            if not anomaly_result:
                for a_id in request.selections.anomaly_ids:
                    issues.append(
                        Issue(
                            severity=IssueSeverity.ERROR,
                            issue_code="ERR_EVIDENCE_SOURCE_UNAVAILABLE",
                            message=(
                                f"Anomaly source {a_id} requested but anomaly detection "
                                "did not run."
                            ),
                        )
                    )
            else:
                # Find anomalies
                available_anomalies = {}
                for d_res in anomaly_result.detector_results:
                    for a in d_res.anomalies:
                        available_anomalies[a.anomaly_id] = a

                for a_id in request.selections.anomaly_ids:
                    if a_id in available_anomalies:
                        try:
                            ev = EvidenceConstructionEngine.create_from_anomaly(
                                available_anomalies[a_id]
                            )
                            records.append(ev)
                        except Exception as e:
                            issues.append(
                                Issue(
                                    severity=IssueSeverity.ERROR,
                                    issue_code="ERR_EVIDENCE_CONSTRUCTION_FAILED",
                                    message=str(e),
                                )
                            )
                    else:
                        issues.append(
                            Issue(
                                severity=IssueSeverity.ERROR,
                                issue_code="ERR_EVIDENCE_SOURCE_UNAVAILABLE",
                                message=f"Anomaly source {a_id} not found in pipeline results.",
                            )
                        )

        # 2. Engineering
        if request.selections.engineering_result_ids:
            if not engineering_results:
                for r_id in request.selections.engineering_result_ids:
                    issues.append(
                        Issue(
                            severity=IssueSeverity.ERROR,
                            issue_code="ERR_EVIDENCE_SOURCE_UNAVAILABLE",
                            message=(
                                f"Engineering source {r_id} requested but engineering did not run."
                            ),
                        )
                    )
            else:
                available_eng = {r.calculation_id: r for r in engineering_results}
                for r_id in request.selections.engineering_result_ids:
                    if r_id in available_eng:
                        try:
                            ev = EvidenceConstructionEngine.create_from_engineering_result(
                                available_eng[r_id]
                            )
                            records.append(ev)
                        except Exception as e:
                            issues.append(
                                Issue(
                                    severity=IssueSeverity.ERROR,
                                    issue_code="ERR_EVIDENCE_CONSTRUCTION_FAILED",
                                    message=str(e),
                                )
                            )
                    else:
                        issues.append(
                            Issue(
                                severity=IssueSeverity.ERROR,
                                issue_code="ERR_EVIDENCE_SOURCE_UNAVAILABLE",
                                message=f"Engineering source {r_id} not found in pipeline results.",
                            )
                        )

        # 3. Statistics
        if request.selections.statistical_metrics:
            if not statistical_results:
                for sel in request.selections.statistical_metrics:
                    issues.append(
                        Issue(
                            severity=IssueSeverity.ERROR,
                            issue_code="ERR_EVIDENCE_SOURCE_UNAVAILABLE",
                            message=(
                                f"Statistical source {sel.analysis_id}/{sel.metric_id} requested "
                                "but statistics did not run."
                            ),
                        )
                    )
            else:
                available_stat = {r.analysis_id: r for r in statistical_results}
                for sel in request.selections.statistical_metrics:
                    if sel.analysis_id in available_stat:
                        analysis = available_stat[sel.analysis_id]
                        available_metrics = {m.metric_id: m for m in analysis.metrics}
                        if sel.metric_id in available_metrics:
                            try:
                                ev = EvidenceConstructionEngine.create_from_statistical_metric(
                                    analysis, available_metrics[sel.metric_id]
                                )
                                records.append(ev)
                            except Exception as e:
                                issues.append(
                                    Issue(
                                        severity=IssueSeverity.ERROR,
                                        issue_code="ERR_EVIDENCE_CONSTRUCTION_FAILED",
                                        message=str(e),
                                    )
                                )
                        else:
                            issues.append(
                                Issue(
                                    severity=IssueSeverity.ERROR,
                                    issue_code="ERR_EVIDENCE_SOURCE_UNAVAILABLE",
                                    message=(
                                        f"Statistical metric {sel.metric_id} not found in "
                                        f"analysis {sel.analysis_id}."
                                    ),
                                )
                            )
                    else:
                        issues.append(
                            Issue(
                                severity=IssueSeverity.ERROR,
                                issue_code="ERR_EVIDENCE_SOURCE_UNAVAILABLE",
                                message=(
                                    f"Statistical analysis {sel.analysis_id} not found in "
                                    "pipeline results."
                                ),
                            )
                        )

        # 4. Expected-vs-Observed
        if request.selections.expected_observed_comparison_ids:
            if not expected_observed_result:
                for c_id in request.selections.expected_observed_comparison_ids:
                    issues.append(
                        Issue(
                            severity=IssueSeverity.ERROR,
                            issue_code="ERR_EVIDENCE_SOURCE_UNAVAILABLE",
                            message=(
                                f"Expected-vs-Observed source {c_id} requested but it did not run."
                            ),
                        )
                    )
            else:
                available_comp = {c.comparison_id: c for c in expected_observed_result.comparisons}
                for c_id in request.selections.expected_observed_comparison_ids:
                    if c_id in available_comp:
                        try:
                            ev = (
                                EvidenceConstructionEngine.create_from_expected_observed_comparison(
                                    available_comp[c_id]
                                )
                            )
                            records.append(ev)
                        except Exception as e:
                            issues.append(
                                Issue(
                                    severity=IssueSeverity.ERROR,
                                    issue_code="ERR_EVIDENCE_CONSTRUCTION_FAILED",
                                    message=str(e),
                                )
                            )
                    else:
                        issues.append(
                            Issue(
                                severity=IssueSeverity.ERROR,
                                issue_code="ERR_EVIDENCE_SOURCE_UNAVAILABLE",
                                message=(
                                    f"Expected-vs-Observed comparison {c_id} not found in "
                                    "pipeline results."
                                ),
                            )
                        )

        # 5. Vehicle Evaluation
        has_veh_req = (
            request.selections.include_vehicle_evaluation
            or bool(request.selections.limit_evaluation_ids)
            or bool(request.selections.procedure_evaluation_ids)
            or request.selections.include_configuration_discrepancies
        )
        if has_veh_req:
            if not vehicle_evaluation_result:
                issues.append(
                    Issue(
                        severity=IssueSeverity.ERROR,
                        issue_code="ERR_EVIDENCE_SOURCE_UNAVAILABLE",
                        message=(
                            "Vehicle evaluation evidence requested but vehicle evaluation "
                            "did not run."
                        ),
                    )
                )
            else:
                try:
                    if request.selections.include_vehicle_evaluation:
                        veh_coll = EvidenceConstructionEngine.create_from_vehicle_evaluation_result(
                            vehicle_evaluation_result
                        )
                        for r in veh_coll.evidence:
                            records.append(r)
                    else:
                        if request.selections.limit_evaluation_ids:
                            available_lims = {
                                lim.evaluation_id: lim
                                for lim in vehicle_evaluation_result.limit_evaluations
                            }
                            for l_id in request.selections.limit_evaluation_ids:
                                if l_id in available_lims:
                                    ev = EvidenceConstructionEngine.create_from_limit_evaluation(
                                        available_lims[l_id],
                                        evaluation_result=vehicle_evaluation_result,
                                    )
                                    records.append(ev)
                                else:
                                    issues.append(
                                        Issue(
                                            severity=IssueSeverity.ERROR,
                                            issue_code="ERR_EVIDENCE_SOURCE_UNAVAILABLE",
                                            message=(
                                                f"Limit evaluation source {l_id} not found in "
                                                "vehicle evaluation results."
                                            ),
                                        )
                                    )
                        if request.selections.procedure_evaluation_ids:
                            available_procs = {
                                proc.evaluation_id: proc
                                for proc in vehicle_evaluation_result.procedure_evaluations
                            }
                            for p_id in request.selections.procedure_evaluation_ids:
                                if p_id in available_procs:
                                    ev = (
                                        EvidenceConstructionEngine.create_from_procedure_evaluation(
                                            available_procs[p_id],
                                            evaluation_result=vehicle_evaluation_result,
                                        )
                                    )
                                    records.append(ev)
                                else:
                                    issues.append(
                                        Issue(
                                            severity=IssueSeverity.ERROR,
                                            issue_code="ERR_EVIDENCE_SOURCE_UNAVAILABLE",
                                            message=(
                                                f"Procedure evaluation source {p_id} not found "
                                                "in vehicle evaluation results."
                                            ),
                                        )
                                    )
                        if request.selections.include_configuration_discrepancies:
                            create_disc_ev = (
                                EvidenceConstructionEngine.create_from_configuration_discrepancy
                            )
                            for idx, disc in enumerate(
                                vehicle_evaluation_result.configuration_discrepancies
                            ):
                                ev = create_disc_ev(
                                    disc,
                                    evaluation_result=vehicle_evaluation_result,
                                    index=idx,
                                )
                                records.append(ev)
                except Exception as e:
                    issues.append(
                        Issue(
                            severity=IssueSeverity.ERROR,
                            issue_code="ERR_EVIDENCE_CONSTRUCTION_FAILED",
                            message=str(e),
                        )
                    )

        # Deduplicate records by evidence_id deterministically preserving order
        seen_ids = set()
        deduped_records = []
        for r in records:
            if r.evidence_id not in seen_ids:
                seen_ids.add(r.evidence_id)
                deduped_records.append(r)
        records = deduped_records

        # Determine status
        total_requested = (
            len(request.selections.anomaly_ids)
            + len(request.selections.engineering_result_ids)
            + len(request.selections.statistical_metrics)
            + len(request.selections.expected_observed_comparison_ids)
            + (1 if request.selections.include_vehicle_evaluation else 0)
            + len(request.selections.limit_evaluation_ids)
            + len(request.selections.procedure_evaluation_ids)
            + (1 if request.selections.include_configuration_discrepancies else 0)
        )
        if any(i.severity == IssueSeverity.ERROR for i in issues):
            if records:
                status = ResultStatus.PARTIAL
            else:
                status = (
                    ResultStatus.NOT_AVAILABLE
                    if len(issues) == total_requested
                    else ResultStatus.ERROR
                )
        else:
            status = ResultStatus.SUCCESS

        collection = None
        if records:
            collection = EvidenceConstructionEngine.create_collection(
                request.collection_id, records
            )

        return PipelineEvidenceResult(
            collection=collection,
            status=status,
            issues=issues,
            provenance=prov,
        )
