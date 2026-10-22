from aeris.core.anomaly.models import AnomalyRecord
from aeris.core.engineering.models import EngineeringCalculationResult
from aeris.core.evidence.models import EvidenceCollection, EvidenceRecord, EvidenceType
from aeris.core.expected_observed.models import ExpectedObservedComparison
from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.results.types import Issue
from aeris.core.statistics.models import StatisticalAnalysisResult, StatisticalMetric
from aeris.core.vehicle_evaluation.models import (
    LimitEvaluationRecord,
    ProcedureEvaluationRecord,
    VehicleEvaluationResult,
)


class EvidenceConstructionEngine:
    """
    Deterministic mechanism for explicitly constructing EvidenceRecord objects
    from existing AERIS result objects.
    """

    @classmethod
    def create_from_anomaly(cls, anomaly: AnomalyRecord) -> EvidenceRecord:
        """Explicitly constructs an EvidenceRecord from an AnomalyRecord."""
        source_ref = anomaly.source_reference or anomaly.analysis_id
        row_idx = None
        if anomaly.evidence_reference and anomaly.evidence_reference.row_index is not None:
            row_idx = anomaly.evidence_reference.row_index

        desc = (
            f"Anomaly {anomaly.anomaly_id} detected on {anomaly.target_column} "
            f"via {anomaly.detection_method.name}"
        )

        return EvidenceRecord(
            evidence_id=f"EVD-ANOMALY-{anomaly.anomaly_id}",
            evidence_type=EvidenceType.ANOMALY,
            description=desc,
            source_reference=source_ref,
            source_record_id=anomaly.anomaly_id,
            source_parameter_id=anomaly.target_column,
            source_row_index=row_idx,
            timestamp=anomaly.timestamp,
            epistemic_level=anomaly.epistemic_level,
            provenance=anomaly.provenance,
        )

    @classmethod
    def create_from_engineering_result(cls, result: EngineeringCalculationResult) -> EvidenceRecord:
        """Explicitly constructs an EvidenceRecord from an EngineeringCalculationResult."""
        if not result.provenance:
            raise ValueError(
                "ERR_REQUIRED_TRACEABILITY_MISSING: Engineering result lacks provenance."
            )

        source_ref = result.calculation_id
        desc = f"Engineering calculation {result.calculation_id} produced {result.output_value}"

        return EvidenceRecord(
            evidence_id=f"EVD-ENG-{result.result_id}",
            evidence_type=EvidenceType.CALCULATION,
            description=desc,
            source_reference=source_ref,
            source_record_id=result.result_id,
            unit=result.unit,
            epistemic_level=result.epistemic_level,
            provenance=result.provenance,
        )

    @classmethod
    def create_from_statistical_metric(
        cls, analysis: StatisticalAnalysisResult, metric: StatisticalMetric
    ) -> EvidenceRecord:
        """Explicitly constructs an EvidenceRecord from a StatisticalAnalysisResult and metric."""
        if not analysis.provenance:
            raise ValueError(
                "ERR_REQUIRED_TRACEABILITY_MISSING: Statistical analysis lacks provenance."
            )

        source_ref = analysis.analysis_id
        desc = (
            f"Statistical {metric.metric_type.name} on {analysis.target_column} was {metric.value}"
        )

        return EvidenceRecord(
            evidence_id=f"EVD-STAT-{analysis.result_id}-{metric.metric_id}",
            evidence_type=EvidenceType.STATISTICAL_RESULT,
            description=desc,
            source_reference=source_ref,
            source_record_id=analysis.result_id,
            source_parameter_id=analysis.target_column,
            epistemic_level=analysis.epistemic_level,
            provenance=analysis.provenance,
        )

    @classmethod
    def create_from_expected_observed_comparison(
        cls, comparison: ExpectedObservedComparison
    ) -> EvidenceRecord:
        """Explicitly constructs an EvidenceRecord from an ExpectedObservedComparison."""
        source_ref = comparison.comparison_id
        desc = (
            f"Expected vs Observed comparison {comparison.comparison_id} "
            f"yielded {comparison.status.name}"
        )

        # Determine the timestamp and parameter id if possible from expected value
        # Or from the comparison object itself if there's a way.
        param_id = getattr(comparison.expected, "parameter_id", None)
        timestamp = getattr(comparison.expected, "timestamp", None)
        unit = getattr(comparison.expected, "unit", None)

        return EvidenceRecord(
            evidence_id=f"EVD-EXOBS-{comparison.comparison_id}",
            evidence_type=EvidenceType.EXPECTED_OBSERVED_COMPARISON,
            description=desc,
            source_reference=source_ref,
            source_record_id=comparison.comparison_id,
            source_parameter_id=param_id,
            timestamp=timestamp,
            unit=unit,
            epistemic_level=comparison.provenance.epistemic_level,
            provenance=comparison.provenance,
        )

    @classmethod
    def create_from_limit_evaluation(
        cls,
        evaluation: LimitEvaluationRecord,
        evaluation_result: VehicleEvaluationResult | None = None,
    ) -> EvidenceRecord:
        """
        Explicitly constructs an EvidenceRecord from a LimitEvaluationRecord.

        Preserves limit boundary evaluation outcome, mathematical margin,
        epistemic level (LIMIT), and full vehicle/profile/limit lineage.
        """
        source_ref = evaluation.source_reference
        rec_id = evaluation.observation_id or evaluation.evaluation_id

        limit_val = evaluation.limit_value
        if limit_val is None:
            if evaluation.upper_bound is not None:
                limit_val = evaluation.upper_bound
            elif evaluation.lower_bound is not None:
                limit_val = evaluation.lower_bound

        desc = (
            f"Limit evaluation {evaluation.evaluation_id} on parameter "
            f"'{evaluation.parameter_id}' against limit '{evaluation.limit_id}': "
            f"outcome={evaluation.outcome.value}, margin={evaluation.margin} ({evaluation.reason})"
        )

        parent_ids: list[str] = []
        if evaluation_result is not None and evaluation_result.provenance is not None:
            parent_ids.append(str(evaluation_result.provenance.record_id))

        prov = create_provenance(
            activity_name="evidence_construction_from_limit_evaluation",
            source_uri=f"aeris://evidence/limit/{evaluation.evaluation_id}",
            epistemic_level=evaluation.epistemic_level,
            parent_record_ids=parent_ids,
            parameters={
                "evaluation_id": evaluation.evaluation_id,
                "limit_id": evaluation.limit_id,
                "parameter_id": evaluation.parameter_id,
                "outcome": evaluation.outcome.value,
                "is_exceeded": evaluation.is_exceeded,
            },
        )

        eval_result_id = (
            evaluation_result.evaluation_id
            if evaluation_result is not None
            else evaluation.evaluation_id
        )
        cfg_id = evaluation.configuration_id or (
            evaluation_result.configuration_id if evaluation_result is not None else None
        )

        return EvidenceRecord(
            evidence_id=f"EVD-LIM-{evaluation.evaluation_id}",
            evidence_type=EvidenceType.LIMIT_REFERENCE,
            description=desc,
            source_reference=source_ref,
            source_record_id=rec_id,
            source_parameter_id=evaluation.parameter_id,
            source_row_index=evaluation.row_index,
            timestamp=evaluation.timestamp,
            unit=evaluation.limit_unit,
            epistemic_level=evaluation.epistemic_level,
            provenance=prov,
            evaluation_id=eval_result_id,
            vehicle_id=evaluation.vehicle_id,
            profile_id=evaluation.profile_id,
            profile_version=evaluation.profile_version,
            configuration_id=cfg_id,
            limit_id=evaluation.limit_id,
            evaluation_outcome=evaluation.outcome.value,
            observed_value=evaluation.observed_value,
            limit_value=limit_val,
            margin=evaluation.margin,
            metadata=dict(evaluation.metadata),
        )

    @classmethod
    def create_from_procedure_evaluation(
        cls,
        evaluation: ProcedureEvaluationRecord,
        evaluation_result: VehicleEvaluationResult | None = None,
    ) -> EvidenceRecord:
        """
        Explicitly constructs an EvidenceRecord from a ProcedureEvaluationRecord.

        Preserves procedure verification outcome, target criteria or qualitative state,
        epistemic level (PROCEDURAL), and document/revision lineage.
        """
        source_ref = evaluation.source_reference
        rec_id = evaluation.observation_id or evaluation.evaluation_id

        desc = (
            f"Procedure evaluation {evaluation.evaluation_id} on requirement "
            f"'{evaluation.requirement_id}' ({evaluation.document_id} rev "
            f"{evaluation.document_version}): outcome={evaluation.outcome.value}, "
            f"status={evaluation.verification_status.value} ({evaluation.reason})"
        )

        parent_ids: list[str] = []
        if evaluation_result is not None and evaluation_result.provenance is not None:
            parent_ids.append(str(evaluation_result.provenance.record_id))

        prov = create_provenance(
            activity_name="evidence_construction_from_procedure_evaluation",
            source_uri=f"aeris://evidence/procedure/{evaluation.evaluation_id}",
            epistemic_level=evaluation.epistemic_level,
            parent_record_ids=parent_ids,
            parameters={
                "evaluation_id": evaluation.evaluation_id,
                "requirement_id": evaluation.requirement_id,
                "outcome": evaluation.outcome.value,
                "verification_status": evaluation.verification_status.value,
            },
        )

        eval_result_id = (
            evaluation_result.evaluation_id
            if evaluation_result is not None
            else evaluation.evaluation_id
        )
        cfg_id = evaluation.configuration_id or (
            evaluation_result.configuration_id if evaluation_result is not None else None
        )

        meta = dict(evaluation.metadata)
        if evaluation.is_qualitative:
            meta["is_qualitative"] = True
            meta["qualitative_state"] = evaluation.qualitative_state

        return EvidenceRecord(
            evidence_id=f"EVD-PROC-{evaluation.evaluation_id}",
            evidence_type=EvidenceType.PROCEDURAL_REFERENCE,
            description=desc,
            source_reference=source_ref,
            source_record_id=rec_id,
            source_parameter_id=evaluation.parameter_id,
            source_row_index=evaluation.row_index,
            timestamp=evaluation.timestamp,
            unit=evaluation.requirement_unit or evaluation.observed_unit,
            epistemic_level=evaluation.epistemic_level,
            provenance=prov,
            evaluation_id=eval_result_id,
            vehicle_id=evaluation.vehicle_id,
            profile_version=evaluation.profile_version,
            configuration_id=cfg_id,
            document_id=evaluation.document_id,
            document_version=evaluation.document_version,
            requirement_id=evaluation.requirement_id,
            evaluation_outcome=evaluation.outcome.value,
            observed_value=evaluation.observed_value,
            limit_value=evaluation.target_value,
            margin=evaluation.margin,
            metadata=meta,
        )

    @classmethod
    def create_from_configuration_discrepancy(
        cls,
        discrepancy: str,
        evaluation_result: VehicleEvaluationResult,
        index: int = 0,
    ) -> EvidenceRecord:
        """
        Explicitly constructs an EvidenceRecord for a detected configuration discrepancy.

        Preserves discrepancy description, active configuration context, and traceability
        to the evaluation result.
        """
        parent_ids: list[str] = []
        if evaluation_result.provenance is not None:
            parent_ids.append(str(evaluation_result.provenance.record_id))

        prov = create_provenance(
            activity_name="evidence_construction_from_configuration_discrepancy",
            source_uri=(
                f"aeris://evidence/configuration_discrepancy/"
                f"{evaluation_result.evaluation_id}/{index}"
            ),
            epistemic_level=EpistemicLevel.OBSERVED,
            parent_record_ids=parent_ids,
            parameters={
                "evaluation_id": evaluation_result.evaluation_id,
                "vehicle_id": evaluation_result.vehicle_id,
                "configuration_id": evaluation_result.configuration_id,
                "discrepancy_index": index,
            },
        )

        return EvidenceRecord(
            evidence_id=f"EVD-CFG-DISC-{evaluation_result.evaluation_id}-{index:02d}",
            evidence_type=EvidenceType.CONFIGURATION_DISCREPANCY,
            description=f"Configuration discrepancy: {discrepancy}",
            source_reference=f"urn:aeris:evaluation:{evaluation_result.evaluation_id}",
            source_record_id=evaluation_result.evaluation_id,
            epistemic_level=EpistemicLevel.OBSERVED,
            provenance=prov,
            evaluation_id=evaluation_result.evaluation_id,
            vehicle_id=evaluation_result.vehicle_id,
            profile_id=evaluation_result.profile_id,
            profile_version=evaluation_result.profile_version,
            configuration_id=evaluation_result.configuration_id,
            evaluation_outcome="DISCREPANCY",
            metadata={"discrepancy": discrepancy, "discrepancy_index": index},
        )

    @classmethod
    def create_from_evaluation_issue(
        cls,
        issue: Issue,
        evaluation_result: VehicleEvaluationResult,
        index: int = 0,
    ) -> EvidenceRecord:
        """
        Explicitly constructs an EvidenceRecord from an evaluation Issue.
        """
        parent_ids: list[str] = []
        if evaluation_result.provenance is not None:
            parent_ids.append(str(evaluation_result.provenance.record_id))

        prov = create_provenance(
            activity_name="evidence_construction_from_evaluation_issue",
            source_uri=f"aeris://evidence/issue/{evaluation_result.evaluation_id}/{index}",
            epistemic_level=EpistemicLevel.LIMIT,
            parent_record_ids=parent_ids,
            parameters={
                "evaluation_id": evaluation_result.evaluation_id,
                "issue_code": issue.issue_code,
                "severity": issue.severity.value,
            },
        )

        if issue.issue_code == "VEH_CONFIG_DISCREPANCY":
            ev_type = EvidenceType.CONFIGURATION_DISCREPANCY
        elif "LIMIT" in issue.issue_code:
            ev_type = EvidenceType.LIMIT_REFERENCE
        elif "PROCEDURE" in issue.issue_code:
            ev_type = EvidenceType.PROCEDURAL_REFERENCE
        else:
            ev_type = EvidenceType.DATA_QUALITY

        row_idx = issue.location if isinstance(issue.location, int) else None

        return EvidenceRecord(
            evidence_id=f"EVD-ISSUE-{evaluation_result.evaluation_id}-{index:02d}",
            evidence_type=ev_type,
            description=f"Evaluation issue [{issue.issue_code}]: {issue.message}",
            source_reference=f"urn:aeris:evaluation:{evaluation_result.evaluation_id}",
            source_record_id=evaluation_result.evaluation_id,
            source_parameter_id=issue.field_name,
            source_row_index=row_idx,
            epistemic_level=EpistemicLevel.LIMIT,
            provenance=prov,
            evaluation_id=evaluation_result.evaluation_id,
            vehicle_id=evaluation_result.vehicle_id,
            profile_id=evaluation_result.profile_id,
            profile_version=evaluation_result.profile_version,
            configuration_id=evaluation_result.configuration_id,
            evaluation_outcome=issue.issue_code,
            metadata={"issue_code": issue.issue_code, "severity": issue.severity.value},
        )

    @classmethod
    def create_from_vehicle_evaluation_result(
        cls,
        result: VehicleEvaluationResult,
        collection_id: str | None = None,
    ) -> EvidenceCollection:
        """
        Constructs an immutable EvidenceCollection encompassing all limit evaluations,
        procedure evaluations, and configuration discrepancies from a VehicleEvaluationResult.

        Guarantees deterministic ordering and deduplication by evidence_id.
        """
        records: list[EvidenceRecord] = []
        seen_ids: set[str] = set()

        for lim in result.limit_evaluations:
            rec = cls.create_from_limit_evaluation(lim, evaluation_result=result)
            if rec.evidence_id not in seen_ids:
                seen_ids.add(rec.evidence_id)
                records.append(rec)

        for proc in result.procedure_evaluations:
            rec = cls.create_from_procedure_evaluation(proc, evaluation_result=result)
            if rec.evidence_id not in seen_ids:
                seen_ids.add(rec.evidence_id)
                records.append(rec)

        for idx, disc in enumerate(result.configuration_discrepancies):
            rec = cls.create_from_configuration_discrepancy(
                disc, evaluation_result=result, index=idx
            )
            if rec.evidence_id not in seen_ids:
                seen_ids.add(rec.evidence_id)
                records.append(rec)

        cid = collection_id or f"EVD-COLL-{result.evaluation_id}"
        return EvidenceCollection(collection_id=cid, evidence=tuple(records))

    @classmethod
    def create_collection(
        cls, collection_id: str, records: list[EvidenceRecord]
    ) -> EvidenceCollection:
        """
        Creates an immutable EvidenceCollection.
        Rejects duplicates as enforced by EvidenceCollection itself.
        """
        return EvidenceCollection(collection_id=collection_id, evidence=tuple(records))
