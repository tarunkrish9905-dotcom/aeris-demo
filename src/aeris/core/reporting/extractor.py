"""
AERIS Pipeline Result Report Extractor.

Extracts, structures, and synthesizes an immutable EngineeringReport domain model
from a completed or partially completed DataPipelineResult.

Maintains strict epistemic discipline:
- Never fabricates outcomes or missing stages.
- Never infers that missing limits or unexecuted checks mean "safe" or "passed".
- Explicitly documents unexecuted or degraded stages in the limitations section.
- Does not mutate the source DataPipelineResult.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.reporting.models import (
    AnomalyAndComparisonSummary,
    AnomalyItem,
    ContributorItem,
    DataQualitySummary,
    EngineeringEvaluationSummary,
    EngineeringReport,
    EvidenceSummaryItem,
    ExecutiveSummary,
    ExpectedObservedComparisonItem,
    FlightContextSummary,
    InvestigationIntelligenceSummary,
    LimitationItem,
    LimitationsSummary,
    LimitEvaluationItem,
    ProcedureEvaluationItem,
    ProvenanceSummary,
    RecommendationItem,
    ReportMetadata,
    StageExecutionSummary,
    StageStatus,
)
from aeris.core.results.types import DataPipelineResult, IssueSeverity, ResultStatus


class ReportDataExtractor:
    """Extracts typed EngineeringReport domain models from DataPipelineResult instances."""

    @staticmethod
    def extract(
        pipeline_result: DataPipelineResult,
        report_title: str = "AERIS Flight Test Engineering Report",
        analyst_name: str | None = None,
        custom_metadata: dict[str, Any] | None = None,
    ) -> EngineeringReport:
        """
        Extract an immutable EngineeringReport from a DataPipelineResult.

        Parameters
        ----------
        pipeline_result : DataPipelineResult
            The source pipeline result container (full or partial).
        report_title : str
            Title for the generated engineering report.
        analyst_name : str | None
            Optional analyzing flight test engineer identifier.
        custom_metadata : dict[str, Any] | None
            Optional auxiliary metadata to attach to the report.

        Returns
        -------
        EngineeringReport
            A fully structured, immutable engineering report domain representation.
        """
        now_utc = datetime.datetime.now(datetime.UTC)
        meta_dict = dict(custom_metadata) if custom_metadata else {}

        # 1. Traverse and extract 15 stages in canonical defined order
        stage_summaries = ReportDataExtractor._extract_stage_summaries(pipeline_result)

        # 2. Extract Data Quality and Validation section
        data_quality = ReportDataExtractor._extract_data_quality(pipeline_result)

        # 3. Extract Flight Context section
        flight_context = ReportDataExtractor._extract_flight_context(pipeline_result)

        # 4. Extract Engineering Evaluation & Limits section
        eng_eval = ReportDataExtractor._extract_engineering_evaluation(pipeline_result)

        # 5. Extract Anomaly & Expected-Observed section
        anomaly_comp = ReportDataExtractor._extract_anomaly_and_comparison(pipeline_result)

        # 6. Extract Investigation & Recommendation Intelligence section
        intel_summary = ReportDataExtractor._extract_investigation_intelligence(pipeline_result)

        # 7. Extract Limitations, Caveats, and Data Gaps
        limitations = ReportDataExtractor._extract_limitations(
            pipeline_result=pipeline_result,
            stage_summaries=stage_summaries,
            eng_eval=eng_eval,
        )

        # 8. Extract Provenance and Audit Trail
        provenance = ReportDataExtractor._extract_provenance(pipeline_result)

        # 9. Synthesize Executive Summary
        exec_summary = ReportDataExtractor._synthesize_executive_summary(
            pipeline_result=pipeline_result,
            stage_summaries=stage_summaries,
            eng_eval=eng_eval,
            anomaly_comp=anomaly_comp,
            intel_summary=intel_summary,
        )

        # 10. Construct Report Metadata
        report_id = f"REP-{uuid.uuid4().hex[:12].upper()}"
        pipeline_run_id = (
            pipeline_result.provenance.record_id if pipeline_result.provenance else None
        )
        source_file = pipeline_result.ingestion.source_path if pipeline_result.ingestion else None
        source_hash = None
        if pipeline_result.ingestion and pipeline_result.ingestion.provenance:
            source_hash = pipeline_result.ingestion.provenance.source_checksum

        metadata = ReportMetadata(
            report_id=report_id,
            title=report_title,
            analyst_name=analyst_name,
            generated_at=now_utc,
            pipeline_run_id=pipeline_run_id,
            source_file=source_file,
            source_file_hash=source_hash,
            execution_duration_seconds=pipeline_result.metadata.get("execution_duration_seconds"),
            pipeline_version=pipeline_result.metadata.get("pipeline_version", "1.0.0"),
            custom_metadata=meta_dict,
        )

        return EngineeringReport(
            metadata=metadata,
            executive_summary=exec_summary,
            stage_summaries=tuple(stage_summaries),
            data_quality=data_quality,
            flight_context=flight_context,
            engineering_evaluation=eng_eval,
            anomaly_and_comparison=anomaly_comp,
            investigation_intelligence=intel_summary,
            limitations=limitations,
            provenance=provenance,
            all_issues=tuple(pipeline_result.issues),
        )

    # -----------------------------------------------------------------------
    # Stage Summaries Extraction
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_stage_summaries(
        res: DataPipelineResult,
    ) -> list[StageExecutionSummary]:
        """Extract summaries for all 15 stages in strictly defined sequence."""
        stages: list[tuple[str, int, Any]] = [
            ("ingestion", 1, res.ingestion),
            ("validation", 2, res.validation),
            ("quality", 3, res.quality),
            ("normalization", 4, res.normalization),
            ("parameter_validation", 5, res.parameter_validation),
            ("association", 6, res.association),
            ("engineering", 7, res.engineering),
            ("statistics", 8, res.statistics),
            ("time_series", 9, res.time_series),
            ("anomaly", 10, res.anomaly),
            ("expected_observed", 11, res.expected_observed),
            ("vehicle_evaluation", 12, res.vehicle_evaluation),
            ("evidence", 13, res.evidence),
            ("contributor_analysis", 14, res.contributor_analysis),
            ("recommendation", 15, res.recommendation),
        ]

        summaries: list[StageExecutionSummary] = []
        upstream_failed = res.ingestion.status == ResultStatus.ERROR

        for name, order, stage_obj in stages:
            if stage_obj is None:
                if upstream_failed:
                    status = StageStatus.UPSTREAM_ABORTED
                    notes = "Prerequisite upstream stage failed or encountered critical error."
                else:
                    status = StageStatus.SKIPPED
                    notes = "Stage was not explicitly requested or configured for execution."
                summaries.append(
                    StageExecutionSummary(
                        stage_name=name,
                        stage_order=order,
                        status=status,
                        is_executed=False,
                        issue_count=0,
                        error_count=0,
                        warning_count=0,
                        has_errors=False,
                        has_warnings=False,
                        notes=notes,
                    )
                )
                continue

            # Stage object is present
            is_exec = True
            raw_status = getattr(stage_obj, "status", ResultStatus.SUCCESS)
            issues = getattr(stage_obj, "issues", [])
            err_count = sum(
                1
                for i in issues
                if getattr(i, "severity", None) in (IssueSeverity.ERROR, IssueSeverity.CRITICAL)
            )
            warn_count = sum(
                1 for i in issues if getattr(i, "severity", None) == IssueSeverity.WARNING
            )
            has_err = raw_status == ResultStatus.ERROR or err_count > 0
            has_warn = raw_status == ResultStatus.WARNING or warn_count > 0

            # Map raw ResultStatus to report StageStatus
            if raw_status == ResultStatus.SUCCESS:
                st = StageStatus.SUCCESS
            elif raw_status == ResultStatus.WARNING:
                st = StageStatus.WARNING
            elif raw_status == ResultStatus.ERROR:
                st = StageStatus.ERROR
                upstream_failed = True
            elif raw_status == ResultStatus.PARTIAL:
                st = StageStatus.PARTIAL
            elif raw_status == ResultStatus.NOT_AVAILABLE:
                st = StageStatus.NOT_AVAILABLE
            else:
                st = StageStatus.SUCCESS

            notes = ReportDataExtractor._generate_stage_notes(name, stage_obj)

            summaries.append(
                StageExecutionSummary(
                    stage_name=name,
                    stage_order=order,
                    status=st,
                    is_executed=is_exec,
                    issue_count=len(issues),
                    error_count=err_count,
                    warning_count=warn_count,
                    has_errors=has_err,
                    has_warnings=has_warn,
                    notes=notes,
                )
            )

        return summaries

    @staticmethod
    def _generate_stage_notes(name: str, stage_obj: Any) -> str:
        """Deterministic operational note generator per stage."""
        if name == "ingestion":
            row_count = getattr(stage_obj, "row_count", 0)
            col_count = len(getattr(stage_obj, "column_names", []))
            return f"Ingested {row_count} rows across {col_count} columns."
        elif name == "validation":
            is_val = getattr(stage_obj, "is_valid", False)
            cols = len(getattr(stage_obj, "validated_columns", []))
            viols = getattr(stage_obj, "violation_count", 0)
            verdict = "PASSED" if is_val else "FAILED"
            return f"Schema validation {verdict}: {cols} columns evaluated, {viols} violations."
        elif name == "quality":
            score = getattr(stage_obj, "quality_score", None)
            score_str = f"{score:.2f}" if score is not None else "N/A"
            return f"Quality score: {score_str}."
        elif name == "normalization":
            tx_count = len(getattr(stage_obj, "transformations", []))
            return f"{tx_count} explicit transformations applied."
        elif name == "parameter_validation":
            val_p = len(getattr(stage_obj, "validated_parameters", []))
            fail_p = len(getattr(stage_obj, "failed_parameters", []))
            return f"{val_p} parameters validated, {fail_p} failed semantic validation."
        elif name == "association":
            tp_id = getattr(stage_obj, "test_point_id", "N/A")
            rows = getattr(stage_obj, "row_count", 0)
            return f"Associated test point {tp_id} with {rows} rows."
        elif name == "engineering":
            calc_id = getattr(stage_obj, "calculation_id", "N/A")
            return f"Calculation '{calc_id}' executed."
        elif name == "statistics":
            col = getattr(stage_obj, "target_column", "N/A")
            metrics = len(getattr(stage_obj, "metrics", []))
            return f"Statistical profile on '{col}': {metrics} metrics calculated."
        elif name == "time_series":
            col = getattr(stage_obj, "target_column", "N/A")
            metrics = len(getattr(stage_obj, "metrics", []))
            return f"Rolling time-series analysis on '{col}': {metrics} metrics calculated."
        elif name == "anomaly":
            dets = len(getattr(stage_obj, "detector_results", ()))
            total_anom = sum(
                len(getattr(d, "anomalies", ())) for d in getattr(stage_obj, "detector_results", ())
            )
            return f"{dets} anomaly detectors evaluated: {total_anom} anomalies detected."
        elif name == "expected_observed":
            comps = len(getattr(stage_obj, "comparisons", ()))
            return f"{comps} expected-vs-observed comparisons evaluated."
        elif name == "vehicle_evaluation":
            v_id = getattr(stage_obj, "vehicle_id", "N/A")
            lim_count = len(getattr(stage_obj, "limit_evaluations", ()))
            proc_count = len(getattr(stage_obj, "procedure_evaluations", ()))
            return f"Vehicle {v_id}: {lim_count} limits, {proc_count} procedures evaluated."
        elif name == "evidence":
            coll = getattr(stage_obj, "collection", None)
            ev_count = len(getattr(coll, "evidence", ())) if coll else 0
            return f"{ev_count} traceable evidence items constructed."
        elif name == "contributor_analysis":
            results = len(getattr(stage_obj, "results", ()))
            return f"{results} contributor correlation analyses evaluated."
        elif name == "recommendation":
            recs = len(getattr(stage_obj, "recommendations", ()))
            rules = len(getattr(stage_obj, "rules_triggered", ()))
            return f"{recs} advisory recommendations generated across {rules} rules."
        return f"{name} stage executed."

    # -----------------------------------------------------------------------
    # Section Extractors
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_data_quality(res: DataPipelineResult) -> DataQualitySummary:
        """Extract data quality, validation, normalization, and semantic check metrics."""
        ingest = res.ingestion
        val = res.validation
        qual = res.quality
        norm = res.normalization
        param_val = res.parameter_validation

        total_rows = ingest.row_count if ingest else 0
        total_cols = len(ingest.column_names) if ingest else 0
        cols_ingested = tuple(ingest.column_names) if ingest else ()

        is_valid = val.is_valid if val else None
        cols_eval = tuple(val.validated_columns) if val else ()
        viol_count = val.violation_count if val else 0

        qual_score = qual.quality_score if qual else None
        qual_metrics = dict(qual.metrics) if qual and qual.metrics else {}
        qual_summary = qual.summary if qual and qual.summary else ""

        norm_applied = norm is not None and norm.status in (
            ResultStatus.SUCCESS,
            ResultStatus.WARNING,
            ResultStatus.PARTIAL,
        )
        tx_count = len(norm.transformations) if norm else 0

        param_val_passed = param_val.is_successful if param_val else None
        val_params = tuple(param_val.validated_parameters) if param_val else ()
        failed_params = tuple(param_val.failed_parameters) if param_val else ()

        # Collect issues from the data quality stages
        dq_issues = []
        if ingest:
            dq_issues.extend(ingest.issues)
        if val:
            dq_issues.extend(val.issues)
        if qual:
            dq_issues.extend(qual.issues)
        if norm:
            dq_issues.extend(norm.issues)
        if param_val:
            dq_issues.extend(param_val.issues)

        return DataQualitySummary(
            is_valid=is_valid,
            quality_score=qual_score,
            total_rows_ingested=total_rows,
            total_columns_ingested=total_cols,
            columns_ingested=cols_ingested,
            columns_evaluated=cols_eval,
            violation_count=viol_count,
            quality_metrics=qual_metrics,
            quality_summary_text=qual_summary,
            normalization_applied=norm_applied,
            transformations_count=tx_count,
            parameter_validation_passed=param_val_passed,
            validated_parameters=val_params,
            failed_parameters=failed_params,
            issues=tuple(dq_issues),
        )

    @staticmethod
    def _extract_flight_context(res: DataPipelineResult) -> FlightContextSummary:
        """Extract vehicle profile, configuration, test card, and test point context."""
        veh = res.vehicle_evaluation
        assoc = res.association

        v_id = veh.vehicle_id if veh else None
        prof_id = veh.profile_id if veh else None
        prof_ver = veh.profile_version if veh else None
        cfg_id = veh.configuration_id if veh else None
        discrepancies = tuple(veh.configuration_discrepancies) if veh else ()

        tp_id = getattr(assoc, "test_point_id", None) if assoc else None
        obs_id = getattr(assoc, "observation_id", None) if assoc else None
        assoc_rows = getattr(assoc, "row_count", None) if assoc else None

        # Check vehicle evaluation metadata or test operation for test_card_id
        tc_id = None
        if veh and veh.summary:
            tc_id = veh.summary.get("test_card_id")
        if tc_id is None and assoc:
            tc_id = getattr(assoc, "metadata", {}).get("test_card_id")

        return FlightContextSummary(
            vehicle_id=v_id,
            profile_id=prof_id,
            profile_version=prof_ver,
            configuration_id=cfg_id,
            test_card_id=tc_id,
            test_point_id=tp_id,
            observation_id=obs_id,
            associated_rows_count=assoc_rows,
            configuration_discrepancies=discrepancies,
        )

    @staticmethod
    def _extract_engineering_evaluation(
        res: DataPipelineResult,
    ) -> EngineeringEvaluationSummary:
        """Extract vehicle engineering limit and procedure evaluations."""
        veh = res.vehicle_evaluation
        eng = res.engineering

        calcs_performed = []
        if eng and getattr(eng, "calculation_id", None):
            calcs_performed.append(eng.calculation_id)

        if veh is None:
            return EngineeringEvaluationSummary(
                is_evaluated=False,
                total_limits_evaluated=0,
                limit_exceedance_count=0,
                limit_inconclusive_count=0,
                total_procedures_evaluated=0,
                procedure_violation_count=0,
                procedure_inconclusive_count=0,
                limit_evaluations=(),
                procedure_evaluations=(),
                calculations_performed=tuple(calcs_performed),
            )

        limit_items: list[LimitEvaluationItem] = []
        inconclusive_lim = 0
        for rec in veh.limit_evaluations:
            if rec.outcome == "INCONCLUSIVE":
                inconclusive_lim += 1
            b_val = rec.limit_value
            if b_val is None:
                b_val = rec.upper_bound if rec.upper_bound is not None else rec.lower_bound
            limit_unit = rec.limit_unit or rec.observed_unit

            limit_items.append(
                LimitEvaluationItem(
                    evaluation_id=rec.evaluation_id,
                    limit_id=rec.limit_id,
                    parameter_id=rec.parameter_id,
                    outcome=str(rec.outcome),
                    limit_type=str(rec.limit_type) if rec.limit_type else None,
                    severity=str(rec.severity) if rec.severity else None,
                    observed_value=rec.observed_value,
                    boundary_value=b_val,
                    unit=limit_unit,
                    margin=rec.margin,
                    is_exceeded=rec.is_exceeded,
                    reason=rec.reason,
                    epistemic_level=EpistemicLevel.LIMIT,
                )
            )

        proc_items: list[ProcedureEvaluationItem] = []
        inconclusive_proc = 0
        for p_rec in veh.procedure_evaluations:
            if p_rec.outcome == "INCONCLUSIVE":
                inconclusive_proc += 1
            proc_items.append(
                ProcedureEvaluationItem(
                    evaluation_id=p_rec.evaluation_id,
                    requirement_id=p_rec.requirement_id,
                    document_id=p_rec.document_id,
                    category=str(p_rec.category),
                    verification_status=str(p_rec.verification_status),
                    outcome=str(p_rec.outcome),
                    observed_value=p_rec.observed_value,
                    target_value=p_rec.target_value,
                    unit=p_rec.observed_unit or p_rec.requirement_unit,
                    margin=p_rec.margin,
                    is_satisfied=p_rec.is_satisfied,
                    is_qualitative=p_rec.is_qualitative,
                    qualitative_state=p_rec.qualitative_state,
                    reason=p_rec.reason,
                    epistemic_level=EpistemicLevel.PROCEDURAL,
                )
            )

        return EngineeringEvaluationSummary(
            is_evaluated=True,
            total_limits_evaluated=len(limit_items),
            limit_exceedance_count=veh.limit_exceedance_count,
            limit_inconclusive_count=inconclusive_lim,
            total_procedures_evaluated=len(proc_items),
            procedure_violation_count=veh.procedure_violation_count,
            procedure_inconclusive_count=inconclusive_proc,
            limit_evaluations=tuple(limit_items),
            procedure_evaluations=tuple(proc_items),
            calculations_performed=tuple(calcs_performed),
        )

    @staticmethod
    def _extract_anomaly_and_comparison(
        res: DataPipelineResult,
    ) -> AnomalyAndComparisonSummary:
        """Extract detected anomalies and expected-vs-observed deviations."""
        anom_res = res.anomaly
        eo_res = res.expected_observed

        anomalies: list[AnomalyItem] = []
        high_cnt = 0
        med_cnt = 0
        low_cnt = 0
        info_cnt = 0

        if anom_res:
            for det in anom_res.detector_results:
                for a in getattr(det, "anomalies", ()):
                    sev_str = str(a.severity).upper()
                    if sev_str == "HIGH":
                        high_cnt += 1
                    elif sev_str == "MEDIUM":
                        med_cnt += 1
                    elif sev_str == "LOW":
                        low_cnt += 1
                    else:
                        info_cnt += 1

                    anomalies.append(
                        AnomalyItem(
                            anomaly_id=a.anomaly_id,
                            analysis_id=a.analysis_id,
                            target_column=a.target_column,
                            classification=str(a.classification),
                            severity=sev_str,
                            detection_method=str(a.detection_method),
                            timestamp=a.timestamp,
                            observed_value=a.observed_value,
                            expected_value=a.expected_value,
                            deviation_value=a.deviation_value,
                            explanation=a.explanation,
                            epistemic_level=EpistemicLevel.CALCULATED,
                        )
                    )

        comparisons: list[ExpectedObservedComparisonItem] = []
        within_tol = 0
        out_tol = 0

        if eo_res:
            for comp in eo_res.comparisons:
                if comp.within_tolerance is True:
                    within_tol += 1
                elif comp.within_tolerance is False:
                    out_tol += 1

                comparisons.append(
                    ExpectedObservedComparisonItem(
                        comparison_id=comp.comparison_id,
                        parameter_id=comp.observed.parameter_id,
                        expected_value=comp.expected.value,
                        observed_value=comp.observed.value,
                        deviation=comp.deviation.deviation_value if comp.deviation else None,
                        tolerance_evaluated=comp.tolerance_evaluated,
                        within_tolerance=comp.within_tolerance,
                        status=comp.status,
                        epistemic_level=EpistemicLevel.CALCULATED,
                    )
                )

        return AnomalyAndComparisonSummary(
            total_anomalies=len(anomalies),
            high_severity_anomalies=high_cnt,
            medium_severity_anomalies=med_cnt,
            low_severity_anomalies=low_cnt,
            info_severity_anomalies=info_cnt,
            anomalies=tuple(anomalies),
            total_comparisons=len(comparisons),
            comparisons_within_tolerance=within_tol,
            comparisons_out_of_tolerance=out_tol,
            comparisons=tuple(comparisons),
        )

    @staticmethod
    def _extract_investigation_intelligence(
        res: DataPipelineResult,
    ) -> InvestigationIntelligenceSummary:
        """Extract evidence, contributor rankings, and generated recommendations."""
        ev_res = res.evidence
        contrib_res = res.contributor_analysis
        rec_res = res.recommendation

        evidence_items: list[EvidenceSummaryItem] = []
        if ev_res and ev_res.collection:
            for e in ev_res.collection.evidence:
                evidence_items.append(
                    EvidenceSummaryItem(
                        evidence_id=e.evidence_id,
                        evidence_type=str(e.evidence_type),
                        description=e.description,
                        source_reference=e.source_reference,
                        parameter_id=e.source_parameter_id,
                        observed_value=e.observed_value,
                        margin=e.margin,
                        epistemic_level=e.epistemic_level,
                    )
                )

        contributor_items: list[ContributorItem] = []
        if contrib_res:
            for c in contrib_res.results:
                contributor_items.append(
                    ContributorItem(
                        analysis_id=c.analysis_id,
                        target_column=c.target_column,
                        factor_column=c.factor_column,
                        method=str(c.analysis_method),
                        coefficient=c.coefficient,
                        p_value=c.p_value,
                        status=c.status,
                        epistemic_level=EpistemicLevel.CALCULATED,
                    )
                )

        recommendation_items: list[RecommendationItem] = []
        rules_eval: tuple[str, ...] = ()
        rules_trig: tuple[str, ...] = ()
        if rec_res:
            rules_eval = tuple(rec_res.rules_evaluated)
            rules_trig = tuple(rec_res.rules_triggered)
            for r in rec_res.recommendations:
                # Extract rule_id from metadata or description if present
                rule_id = (
                    r.metadata.get("rule_id") if hasattr(r, "metadata") and r.metadata else None
                )
                if not rule_id:
                    # Check if rule name appears in title or description
                    for rule_candidate in rules_trig:
                        if rule_candidate in r.title or rule_candidate in r.description:
                            rule_id = rule_candidate
                            break

                recommendation_items.append(
                    RecommendationItem(
                        recommendation_id=r.recommendation_id,
                        recommendation_type=str(r.recommendation_type),
                        title=r.title,
                        description=r.description,
                        rationale=r.rationale,
                        status=str(r.status),
                        evidence_ids=tuple(r.evidence_ids),
                        rule_id=rule_id,
                        epistemic_level=EpistemicLevel.RECOMMENDATION,
                    )
                )

        return InvestigationIntelligenceSummary(
            evidence_count=len(evidence_items),
            evidence_items=tuple(evidence_items),
            contributor_rankings=tuple(contributor_items),
            recommendation_count=len(recommendation_items),
            rules_evaluated=rules_eval,
            rules_triggered=rules_trig,
            recommendations=tuple(recommendation_items),
        )

    @staticmethod
    def _extract_limitations(
        pipeline_result: DataPipelineResult,
        stage_summaries: list[StageExecutionSummary],
        eng_eval: EngineeringEvaluationSummary,
    ) -> LimitationsSummary:
        """Explicitly capture analytical limitations, data gaps, and inconclusive checks."""
        unexecuted = [s.stage_name for s in stage_summaries if not s.is_executed]
        failed = [
            s.stage_name
            for s in stage_summaries
            if s.status in (StageStatus.ERROR, StageStatus.UPSTREAM_ABORTED)
        ]
        items: list[LimitationItem] = []

        # 1. Document unexecuted stages
        for s in stage_summaries:
            if not s.is_executed:
                cat = (
                    "UPSTREAM_ABORTED"
                    if s.status == StageStatus.UPSTREAM_ABORTED
                    else "UNEXECUTED_STAGE"
                )
                items.append(
                    LimitationItem(
                        limitation_id=f"LIM-STAGE-{s.stage_name.upper()}",
                        category=cat,
                        stage_name=s.stage_name,
                        description=f"Stage '{s.stage_name}' was not executed ({s.status.value}).",
                        impact=(
                            f"Analysis results, checks, or metrics from '{s.stage_name}' "
                            "are absent from this report."
                        ),
                    )
                )

        # 2. Document failed stages
        for s in stage_summaries:
            if s.is_executed and s.status == StageStatus.ERROR:
                items.append(
                    LimitationItem(
                        limitation_id=f"LIM-FAIL-{s.stage_name.upper()}",
                        category="STAGE_EXECUTION_FAILURE",
                        stage_name=s.stage_name,
                        description=f"Stage '{s.stage_name}' failed with errors.",
                        impact=s.notes or "Partial or corrupted analysis outputs.",
                    )
                )

        # 3. Document inconclusive engineering limit evaluations
        for lim in eng_eval.limit_evaluations:
            if lim.outcome == "INCONCLUSIVE":
                items.append(
                    LimitationItem(
                        limitation_id=f"LIM-INCONCL-LIM-{lim.limit_id}",
                        category="INCONCLUSIVE_LIMIT_EVALUATION",
                        stage_name="vehicle_evaluation",
                        description=(
                            f"Limit evaluation for '{lim.parameter_id}' ({lim.limit_id}) "
                            f"was inconclusive: {lim.reason or 'Unit mismatch or missing data.'}"
                        ),
                        impact=(
                            "Engineering boundary compliance could not be verified. "
                            "Do NOT assume parameter was within allowable limits."
                        ),
                    )
                )

        # 4. Document inconclusive procedure evaluations
        for proc in eng_eval.procedure_evaluations:
            if proc.outcome == "INCONCLUSIVE":
                items.append(
                    LimitationItem(
                        limitation_id=f"LIM-INCONCL-PROC-{proc.requirement_id}",
                        category="INCONCLUSIVE_PROCEDURE_EVALUATION",
                        stage_name="vehicle_evaluation",
                        description=(
                            f"Procedure requirement '{proc.requirement_id}' was inconclusive: "
                            f"{proc.reason}"
                        ),
                        impact="Procedural compliance could not be confirmed.",
                    )
                )

        # 5. Check if vehicle evaluation was completely absent
        if not eng_eval.is_evaluated:
            items.append(
                LimitationItem(
                    limitation_id="LIM-NO-VEHICLE-EVALUATION",
                    category="MISSING_ENGINEERING_LIMITS",
                    stage_name="vehicle_evaluation",
                    description=(
                        "No vehicle profile or engineering limit evaluation "
                        "was supplied or executed."
                    ),
                    impact=(
                        "Report contains empirical telemetry and calculations only. "
                        "No vehicle boundary safety assertions can be made."
                    ),
                )
            )

        total_inconclusive = (
            eng_eval.limit_inconclusive_count + eng_eval.procedure_inconclusive_count
        )
        has_lim = bool(unexecuted or failed or total_inconclusive > 0 or items)

        return LimitationsSummary(
            has_limitations=has_lim,
            unexecuted_stages=tuple(unexecuted),
            failed_stages=tuple(failed),
            inconclusive_checks_count=total_inconclusive,
            items=tuple(items),
        )

    @staticmethod
    def _extract_provenance(res: DataPipelineResult) -> ProvenanceSummary:
        """Extract provenance records and audit trail mapping."""
        root_prov = res.provenance
        root_id = root_prov.record_id if root_prov else None
        src_uri = root_prov.source_uri if root_prov else res.ingestion.source_path
        act_name = root_prov.activity_name if root_prov else "csv_data_pipeline"
        ep_lvl = root_prov.epistemic_level if root_prov else EpistemicLevel.CALCULATED
        parent_ids = tuple(root_prov.parent_record_ids) if root_prov else ()

        stage_prov_map: dict[str, str] = {}
        for stage_name in (
            "ingestion",
            "validation",
            "quality",
            "normalization",
            "parameter_validation",
            "association",
            "engineering",
            "statistics",
            "time_series",
            "anomaly",
            "expected_observed",
            "vehicle_evaluation",
            "evidence",
            "contributor_analysis",
            "recommendation",
        ):
            stage_obj = getattr(res, stage_name, None)
            if stage_obj and getattr(stage_obj, "provenance", None):
                p = stage_obj.provenance
                if hasattr(p, "record_id"):
                    stage_prov_map[stage_name] = p.record_id

        return ProvenanceSummary(
            root_provenance_id=root_id,
            source_uri=src_uri,
            activity_name=act_name,
            epistemic_level=ep_lvl,
            parent_record_ids=parent_ids,
            stage_provenance_records=stage_prov_map,
        )

    @staticmethod
    def _synthesize_executive_summary(
        pipeline_result: DataPipelineResult,
        stage_summaries: list[StageExecutionSummary],
        eng_eval: EngineeringEvaluationSummary,
        anomaly_comp: AnomalyAndComparisonSummary,
        intel_summary: InvestigationIntelligenceSummary,
    ) -> ExecutiveSummary:
        """Synthesize overall executive statistics and deterministic narrative."""
        total_stages = len(stage_summaries)
        exec_count = sum(1 for s in stage_summaries if s.is_executed)
        fail_count = sum(
            1
            for s in stage_summaries
            if s.has_errors or s.status in (StageStatus.ERROR, StageStatus.UPSTREAM_ABORTED)
        )
        warn_count = sum(
            1 for s in stage_summaries if s.has_warnings or s.status == StageStatus.WARNING
        )
        skip_count = sum(1 for s in stage_summaries if not s.is_executed)

        total_issues = len(pipeline_result.issues)
        total_crit_err = sum(
            1
            for i in pipeline_result.issues
            if getattr(i, "severity", None) in (IssueSeverity.ERROR, IssueSeverity.CRITICAL)
        )

        # Build narrative
        status_name = pipeline_result.status.value
        ingest_rows = pipeline_result.ingestion.row_count if pipeline_result.ingestion else 0
        val_status = "PASSED"
        if pipeline_result.validation:
            val_status = "PASSED" if pipeline_result.validation.is_valid else "FAILED"
        elif pipeline_result.ingestion and pipeline_result.ingestion.status == ResultStatus.ERROR:
            val_status = "ABORTED"
        else:
            val_status = "NOT EXECUTED"

        narrative_parts = [
            f"Post-flight analysis completed with overall pipeline status {status_name}.",
            (
                f"Executed {exec_count} of {total_stages} stages; "
                f"{skip_count} skipped, {fail_count} failed."
            ),
            (
                f"Data ingestion processed {ingest_rows} telemetry rows "
                f"with validation status {val_status}."
            ),
        ]

        if eng_eval.is_evaluated:
            narrative_parts.append(
                f"Engineering evaluation evaluated {eng_eval.total_limits_evaluated} limits "
                f"({eng_eval.limit_exceedance_count} exceedances) and "
                f"{eng_eval.total_procedures_evaluated} procedures "
                f"({eng_eval.procedure_violation_count} violations)."
            )
        else:
            narrative_parts.append(
                "Vehicle engineering evaluation was not executed; no limit assertions made."
            )

        if anomaly_comp.total_anomalies > 0:
            narrative_parts.append(
                f"Anomaly detection flagged {anomaly_comp.total_anomalies} anomalies "
                f"({anomaly_comp.high_severity_anomalies} HIGH severity)."
            )
        else:
            narrative_parts.append("Zero anomalies detected.")

        if intel_summary.recommendation_count > 0:
            narrative_parts.append(
                f"Generated {intel_summary.recommendation_count} advisory recommendations "
                "requiring engineering review."
            )
        else:
            narrative_parts.append("No advisory recommendations generated.")

        narrative = " ".join(narrative_parts)

        return ExecutiveSummary(
            overall_status=pipeline_result.status,
            total_stages=total_stages,
            executed_stages_count=exec_count,
            failed_stages_count=fail_count,
            warning_stages_count=warn_count,
            skipped_stages_count=skip_count,
            total_issues_count=total_issues,
            total_critical_or_error_count=total_crit_err,
            total_limit_exceedances=eng_eval.limit_exceedance_count,
            total_procedure_violations=eng_eval.procedure_violation_count,
            total_anomalies_detected=anomaly_comp.total_anomalies,
            total_recommendations_generated=intel_summary.recommendation_count,
            high_level_narrative=narrative,
        )


def extract_engineering_report(
    pipeline_result: DataPipelineResult,
    report_title: str = "AERIS Flight Test Engineering Report",
    analyst_name: str | None = None,
    custom_metadata: dict[str, Any] | None = None,
) -> EngineeringReport:
    """Convenience functional interface for engineering report extraction."""
    return ReportDataExtractor.extract(
        pipeline_result=pipeline_result,
        report_title=report_title,
        analyst_name=analyst_name,
        custom_metadata=custom_metadata,
    )
