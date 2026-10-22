"""
Vehicle Engineering Evaluation Engine.

Provides deterministic, policy-controlled evaluation of telemetry observations
and calculated engineering results against authoritative vehicle engineering
limits and SOP procedure requirements. Preserves strict separation between
comparison outcomes, configured severities, and downstream engineering judgments.
"""

from __future__ import annotations

import math
import uuid

from aeris.core.limits.models import LimitSeverity, LimitType
from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus
from aeris.core.sop.models import (
    RequirementVerificationStatus,
)
from aeris.core.vehicle_evaluation.models import (
    EvaluationObservation,
    EvaluationOutcome,
    LimitEvaluationRecord,
    ProcedureEvaluationRecord,
    VehicleEvaluationRequest,
    VehicleEvaluationResult,
)


class VehicleEngineeringEvaluationEngine:
    """
    Deterministic evaluation engine for vehicle limits and procedure requirements.

    Evaluates observations against configured limits and SOP requirements,
    producing immutable evaluation records without mutating caller inputs.
    """

    def evaluate(self, request: VehicleEvaluationRequest) -> VehicleEvaluationResult:
        """
        Evaluate supplied observations against configured limits and SOP requirements.

        Parameters
        ----------
        request : VehicleEvaluationRequest
            Explicit request specifying vehicle profile, limits, SOP requirements,
            and observations to evaluate.

        Returns
        -------
        VehicleEvaluationResult
            Immutable container holding all individual evaluation records,
            detected configuration discrepancies, and overall execution status.
        """
        issues: list[Issue] = []
        configuration_discrepancies: list[str] = []
        limit_evaluations: list[LimitEvaluationRecord] = []
        procedure_evaluations: list[ProcedureEvaluationRecord] = []

        vehicle_profile = request.vehicle_profile
        envelope = request.configuration_envelope
        active_config_id = envelope.configuration_id if envelope else None

        # ---------------------------------------------------------------------
        # 1. Profile and Configuration Pre-Validation
        # ---------------------------------------------------------------------
        if envelope is not None:
            if envelope.vehicle_id != vehicle_profile.vehicle_id:
                issues.append(
                    Issue(
                        issue_code="ERR_VEHICLE_PROFILE_MISMATCH",
                        severity=IssueSeverity.ERROR,
                        message=(
                            f"ConfigurationEnvelope vehicle_id '{envelope.vehicle_id}' "
                            f"does not match VehicleProfile vehicle_id "
                            f"'{vehicle_profile.vehicle_id}'."
                        ),
                    )
                )
            if envelope.profile_version != vehicle_profile.profile_version:
                issues.append(
                    Issue(
                        issue_code="ERR_VEHICLE_PROFILE_MISMATCH",
                        severity=IssueSeverity.ERROR,
                        message=(
                            f"ConfigurationEnvelope profile_version '{envelope.profile_version}' "
                            f"does not match VehicleProfile profile_version "
                            f"'{vehicle_profile.profile_version}'."
                        ),
                    )
                )

        if request.limit_profile is not None:
            if request.limit_profile.vehicle_id != vehicle_profile.vehicle_id:
                issues.append(
                    Issue(
                        issue_code="ERR_LIMIT_PROFILE_MISMATCH",
                        severity=IssueSeverity.ERROR,
                        message=(
                            f"EngineeringLimitProfile vehicle_id "
                            f"'{request.limit_profile.vehicle_id}' does not match "
                            f"VehicleProfile vehicle_id '{vehicle_profile.vehicle_id}'."
                        ),
                    )
                )
            if request.limit_profile.profile_version != vehicle_profile.profile_version:
                issues.append(
                    Issue(
                        issue_code="ERR_LIMIT_PROFILE_MISMATCH",
                        severity=IssueSeverity.ERROR,
                        message=(
                            f"EngineeringLimitProfile profile_version "
                            f"'{request.limit_profile.profile_version}' does not match "
                            f"VehicleProfile profile_version '{vehicle_profile.profile_version}'."
                        ),
                    )
                )

        if request.procedure_profile is not None:
            if (
                request.procedure_profile.vehicle_id is not None
                and request.procedure_profile.vehicle_id != vehicle_profile.vehicle_id
            ):
                issues.append(
                    Issue(
                        issue_code="ERR_PROCEDURE_PROFILE_MISMATCH",
                        severity=IssueSeverity.ERROR,
                        message=(
                            f"ProcedureRequirementProfile vehicle_id "
                            f"'{request.procedure_profile.vehicle_id}' does not match "
                            f"VehicleProfile vehicle_id '{vehicle_profile.vehicle_id}'."
                        ),
                    )
                )
            if (
                request.procedure_profile.profile_version is not None
                and request.procedure_profile.profile_version != vehicle_profile.profile_version
            ):
                issues.append(
                    Issue(
                        issue_code="ERR_PROCEDURE_PROFILE_MISMATCH",
                        severity=IssueSeverity.ERROR,
                        message=(
                            f"ProcedureRequirementProfile profile_version "
                            f"'{request.procedure_profile.profile_version}' does not match "
                            f"VehicleProfile profile_version '{vehicle_profile.profile_version}'."
                        ),
                    )
                )

        # Fatal validation abort if top-level profile bindings conflict
        if any(i.severity in (IssueSeverity.ERROR, IssueSeverity.CRITICAL) for i in issues):
            prov = create_provenance(
                source_uri=f"aeris://vehicle_evaluation/{request.request_id}",
                activity_name="vehicle_engineering_evaluation",
                epistemic_level=EpistemicLevel.LIMIT,
                parameters={"status": "ERROR", "request_id": request.request_id},
            )
            return VehicleEvaluationResult(
                evaluation_id=f"VEH_EVAL_{uuid.uuid4().hex[:8].upper()}",
                vehicle_id=vehicle_profile.vehicle_id,
                profile_id=vehicle_profile.profile_id,
                profile_version=vehicle_profile.profile_version,
                configuration_id=active_config_id,
                limit_evaluations=(),
                procedure_evaluations=(),
                configuration_discrepancies=(),
                status=ResultStatus.ERROR,
                issues=issues,
                provenance=prov,
                summary={"error": "Fatal profile mismatch aborted evaluation."},
            )

        # ---------------------------------------------------------------------
        # 2. Configuration Discrepancy Checks on Observations
        # ---------------------------------------------------------------------
        if envelope is not None:
            for obs in request.observations:
                if (
                    obs.configuration_id is not None
                    and obs.configuration_id != envelope.configuration_id
                ):
                    discrepancy_msg = (
                        f"Observation '{obs.observation_id}' for parameter '{obs.parameter_id}' "
                        f"declared configuration '{obs.configuration_id}' which conflicts with "
                        f"active configuration envelope '{envelope.configuration_id}'."
                    )
                    configuration_discrepancies.append(discrepancy_msg)
                    issues.append(
                        Issue(
                            issue_code="VEH_CONFIG_DISCREPANCY",
                            severity=IssueSeverity.WARNING,
                            message=discrepancy_msg,
                            field_name=obs.parameter_id,
                            location=obs.row_index,
                        )
                    )

        # ---------------------------------------------------------------------
        # 3. Engineering Limit Evaluation
        # ---------------------------------------------------------------------
        if request.evaluate_limits and request.limit_profile is not None:
            for obs in request.observations:
                if obs.is_qualitative:
                    continue  # Limits apply exclusively to quantitative channels

                applicable_limits = request.limit_profile.get_limits_for_parameter(
                    obs.parameter_id,
                    configuration_id=active_config_id,
                )

                if not applicable_limits:
                    # Check whether parameter has limits for other configurations
                    all_limits = request.limit_profile.get_limits_for_parameter(
                        obs.parameter_id,
                        include_all_configurations=True,
                    )
                    if all_limits:
                        rec = LimitEvaluationRecord(
                            evaluation_id=f"LIM_EV_{uuid.uuid4().hex[:8].upper()}",
                            limit_id="NOT_CONFIGURED",
                            parameter_id=obs.parameter_id,
                            limit_type=LimitType.UPPER_BOUND,
                            severity=LimitSeverity.ADVISORY,
                            outcome=EvaluationOutcome.NOT_APPLICABLE,
                            reason=(
                                f"No limits configured for parameter '{obs.parameter_id}' "
                                f"under active configuration '{active_config_id}'."
                            ),
                            observed_value=obs.value,
                            observed_unit=obs.unit,
                            limit_unit=obs.unit if obs.unit else "unknown",
                            is_exceeded=False,
                            margin=None,
                            vehicle_id=vehicle_profile.vehicle_id,
                            profile_id=vehicle_profile.profile_id,
                            profile_version=vehicle_profile.profile_version,
                            configuration_id=active_config_id,
                            source_reference=obs.source_reference,
                            observation_id=obs.observation_id,
                            row_index=obs.row_index,
                            timestamp=obs.timestamp,
                        )
                    else:
                        rec = LimitEvaluationRecord(
                            evaluation_id=f"LIM_EV_{uuid.uuid4().hex[:8].upper()}",
                            limit_id="NOT_CONFIGURED",
                            parameter_id=obs.parameter_id,
                            limit_type=LimitType.UPPER_BOUND,
                            severity=LimitSeverity.ADVISORY,
                            outcome=EvaluationOutcome.NOT_EVALUATED,
                            reason=(
                                f"No engineering limits configured for "
                                f"parameter '{obs.parameter_id}'."
                            ),
                            observed_value=obs.value,
                            observed_unit=obs.unit,
                            limit_unit=obs.unit if obs.unit else "unknown",
                            is_exceeded=False,
                            margin=None,
                            vehicle_id=vehicle_profile.vehicle_id,
                            profile_id=vehicle_profile.profile_id,
                            profile_version=vehicle_profile.profile_version,
                            configuration_id=active_config_id,
                            source_reference=obs.source_reference,
                            observation_id=obs.observation_id,
                            row_index=obs.row_index,
                            timestamp=obs.timestamp,
                        )
                    limit_evaluations.append(rec)
                    continue

                for limit in applicable_limits:
                    eval_id = f"LIM_EV_{uuid.uuid4().hex[:8].upper()}"

                    # Value validation
                    if obs.value is None or math.isnan(obs.value) or math.isinf(obs.value):
                        rec = LimitEvaluationRecord(
                            evaluation_id=eval_id,
                            limit_id=limit.limit_id,
                            parameter_id=obs.parameter_id,
                            limit_type=limit.limit_type,
                            severity=limit.severity,
                            outcome=EvaluationOutcome.INCONCLUSIVE,
                            reason=f"Non-finite or missing observed value: {obs.value}.",
                            observed_value=obs.value,
                            observed_unit=obs.unit,
                            limit_value=limit.limit_value,
                            lower_bound=limit.lower_bound,
                            upper_bound=limit.upper_bound,
                            limit_unit=limit.unit,
                            is_exceeded=False,
                            margin=None,
                            vehicle_id=vehicle_profile.vehicle_id,
                            profile_id=vehicle_profile.profile_id,
                            profile_version=vehicle_profile.profile_version,
                            configuration_id=limit.configuration_id,
                            source_reference=limit.source_reference,
                            observation_id=obs.observation_id,
                            row_index=obs.row_index,
                            timestamp=obs.timestamp,
                        )
                        limit_evaluations.append(rec)
                        issues.append(
                            Issue(
                                issue_code="ERR_NON_FINITE_OBSERVATION",
                                severity=IssueSeverity.WARNING,
                                message=(
                                    f"Cannot evaluate non-finite value '{obs.value}' "
                                    f"for limit '{limit.limit_id}'."
                                ),
                                field_name=obs.parameter_id,
                                location=obs.row_index,
                            )
                        )
                        continue

                    # Unit validation
                    if obs.unit is None or obs.unit != limit.unit:
                        rec = LimitEvaluationRecord(
                            evaluation_id=eval_id,
                            limit_id=limit.limit_id,
                            parameter_id=obs.parameter_id,
                            limit_type=limit.limit_type,
                            severity=limit.severity,
                            outcome=EvaluationOutcome.INCONCLUSIVE,
                            reason=(
                                f"Incompatible units: observation unit '{obs.unit}' != "
                                f"limit unit '{limit.unit}'. Silent unit conversion is forbidden."
                            ),
                            observed_value=obs.value,
                            observed_unit=obs.unit,
                            limit_value=limit.limit_value,
                            lower_bound=limit.lower_bound,
                            upper_bound=limit.upper_bound,
                            limit_unit=limit.unit,
                            is_exceeded=False,
                            margin=None,
                            vehicle_id=vehicle_profile.vehicle_id,
                            profile_id=vehicle_profile.profile_id,
                            profile_version=vehicle_profile.profile_version,
                            configuration_id=limit.configuration_id,
                            source_reference=limit.source_reference,
                            observation_id=obs.observation_id,
                            row_index=obs.row_index,
                            timestamp=obs.timestamp,
                        )
                        limit_evaluations.append(rec)
                        issues.append(
                            Issue(
                                issue_code="ERR_INCOMPATIBLE_UNIT",
                                severity=IssueSeverity.WARNING,
                                message=(
                                    f"Observation unit '{obs.unit}' does not match limit unit "
                                    f"'{limit.unit}' for limit '{limit.limit_id}'."
                                ),
                                field_name=obs.parameter_id,
                                location=obs.row_index,
                            )
                        )
                        continue

                    # Boundary evaluation (Inclusive boundary equality policy)
                    is_exceeded = limit.is_exceeded(obs.value)
                    margin = limit.calculate_margin(obs.value)

                    if is_exceeded:
                        outcome = EvaluationOutcome.OUTSIDE_BOUNDS
                        reason = (
                            f"Observed value {obs.value} {obs.unit} breaches "
                            f"{limit.limit_type} boundary ({limit.severity})."
                        )
                        issues.append(
                            Issue(
                                issue_code="VEH_LIMIT_EXCEEDANCE",
                                severity=IssueSeverity.WARNING,
                                message=(
                                    f"Limit '{limit.limit_id}' exceeded by {obs.parameter_id}="
                                    f"{obs.value} {obs.unit} (margin: {margin})."
                                ),
                                field_name=obs.parameter_id,
                                location=obs.row_index,
                            )
                        )
                    else:
                        outcome = EvaluationOutcome.WITHIN_BOUNDS
                        reason = (
                            f"Observed value {obs.value} {obs.unit} is within "
                            f"{limit.limit_type} boundary (margin: {margin})."
                        )

                    rec = LimitEvaluationRecord(
                        evaluation_id=eval_id,
                        limit_id=limit.limit_id,
                        parameter_id=obs.parameter_id,
                        limit_type=limit.limit_type,
                        severity=limit.severity,
                        outcome=outcome,
                        reason=reason,
                        observed_value=obs.value,
                        observed_unit=obs.unit,
                        limit_value=limit.limit_value,
                        lower_bound=limit.lower_bound,
                        upper_bound=limit.upper_bound,
                        limit_unit=limit.unit,
                        is_exceeded=is_exceeded,
                        margin=margin,
                        vehicle_id=vehicle_profile.vehicle_id,
                        profile_id=vehicle_profile.profile_id,
                        profile_version=vehicle_profile.profile_version,
                        configuration_id=limit.configuration_id,
                        source_reference=limit.source_reference,
                        observation_id=obs.observation_id,
                        row_index=obs.row_index,
                        timestamp=obs.timestamp,
                    )
                    limit_evaluations.append(rec)

        # ---------------------------------------------------------------------
        # 4. Procedure Requirement Evaluation
        # ---------------------------------------------------------------------
        if request.evaluate_procedures and request.procedure_profile is not None:
            # Map observations for lookup
            numeric_obs_map: dict[str, list[EvaluationObservation]] = {}
            qual_obs_map: dict[str, list[EvaluationObservation]] = {}

            for obs in request.observations:
                if obs.is_qualitative:
                    qual_obs_map.setdefault(obs.parameter_id, []).append(obs)
                    qual_obs_map.setdefault(obs.observation_id, []).append(obs)
                    if "requirement_id" in obs.metadata:
                        qual_obs_map.setdefault(str(obs.metadata["requirement_id"]), []).append(obs)
                else:
                    numeric_obs_map.setdefault(obs.parameter_id, []).append(obs)

            for req in request.procedure_profile.requirements:
                # Scoping checks
                if req.vehicle_id is not None and req.vehicle_id != vehicle_profile.vehicle_id:
                    continue
                if (
                    req.configuration_id is not None
                    and active_config_id is not None
                    and req.configuration_id != active_config_id
                ):
                    continue
                if (
                    req.test_card_id is not None
                    and request.test_card_id is not None
                    and req.test_card_id != request.test_card_id
                ):
                    continue
                if (
                    req.test_point_id is not None
                    and request.test_point_id is not None
                    and req.test_point_id != request.test_point_id
                ):
                    continue

                eval_id = f"PROC_EV_{uuid.uuid4().hex[:8].upper()}"

                # 4A. Non-numeric / Qualitative Checklist Requirement
                if not req.is_numeric:
                    # Look for explicit qualitative observation matching requirement
                    matching_qual: EvaluationObservation | None = None
                    candidates = (
                        qual_obs_map.get(req.requirement_id, [])
                        + (qual_obs_map.get(req.parameter_id, []) if req.parameter_id else [])
                        + qual_obs_map.get(req.description.lower().strip(), [])
                    )
                    if candidates:
                        matching_qual = candidates[0]

                    if matching_qual is not None:
                        state_val = (matching_qual.qualitative_state or "").upper()
                        if state_val in {
                            "CONFIRMED",
                            "SATISFIED",
                            "PASS",
                            "CHECKED",
                            "TRUE",
                            "YES",
                        }:
                            rec_proc = ProcedureEvaluationRecord(
                                evaluation_id=eval_id,
                                requirement_id=req.requirement_id,
                                document_id=req.document_id,
                                document_version=req.document_version,
                                category=req.category,
                                verification_status=RequirementVerificationStatus.SATISFIED,
                                outcome=EvaluationOutcome.SATISFIED,
                                reason=(
                                    f"Qualitative requirement confirmed: "
                                    f"'{matching_qual.qualitative_state}'."
                                ),
                                is_satisfied=True,
                                is_qualitative=True,
                                qualitative_state=matching_qual.qualitative_state,
                                vehicle_id=req.vehicle_id,
                                profile_version=req.profile_version,
                                configuration_id=req.configuration_id,
                                test_card_id=req.test_card_id,
                                test_point_id=req.test_point_id,
                                source_reference=req.source_reference,
                                observation_id=matching_qual.observation_id,
                                timestamp=matching_qual.timestamp,
                            )
                        elif state_val in ("DISCREPANT", "FAILED", "VIOLATED", "FALSE", "NO"):
                            rec_proc = ProcedureEvaluationRecord(
                                evaluation_id=eval_id,
                                requirement_id=req.requirement_id,
                                document_id=req.document_id,
                                document_version=req.document_version,
                                category=req.category,
                                verification_status=RequirementVerificationStatus.VIOLATED,
                                outcome=EvaluationOutcome.VIOLATED,
                                reason=(
                                    f"Qualitative requirement failed: "
                                    f"'{matching_qual.qualitative_state}'."
                                ),
                                is_satisfied=False,
                                is_qualitative=True,
                                qualitative_state=matching_qual.qualitative_state,
                                vehicle_id=req.vehicle_id,
                                profile_version=req.profile_version,
                                configuration_id=req.configuration_id,
                                test_card_id=req.test_card_id,
                                test_point_id=req.test_point_id,
                                source_reference=req.source_reference,
                                observation_id=matching_qual.observation_id,
                                timestamp=matching_qual.timestamp,
                            )
                            issues.append(
                                Issue(
                                    issue_code="VEH_PROCEDURE_VIOLATION",
                                    severity=IssueSeverity.WARNING,
                                    message=(
                                        f"Qualitative requirement '{req.requirement_id}' "
                                        f"discrepant: '{matching_qual.qualitative_state}'."
                                    ),
                                )
                            )
                        else:
                            rec_proc = ProcedureEvaluationRecord(
                                evaluation_id=eval_id,
                                requirement_id=req.requirement_id,
                                document_id=req.document_id,
                                document_version=req.document_version,
                                category=req.category,
                                verification_status=RequirementVerificationStatus.INCONCLUSIVE,
                                outcome=EvaluationOutcome.INCONCLUSIVE,
                                reason=(
                                    f"Ambiguous qualitative state: "
                                    f"'{matching_qual.qualitative_state}'."
                                ),
                                is_satisfied=None,
                                is_qualitative=True,
                                qualitative_state=matching_qual.qualitative_state,
                                vehicle_id=req.vehicle_id,
                                profile_version=req.profile_version,
                                configuration_id=req.configuration_id,
                                test_card_id=req.test_card_id,
                                test_point_id=req.test_point_id,
                                source_reference=req.source_reference,
                                observation_id=matching_qual.observation_id,
                                timestamp=matching_qual.timestamp,
                            )
                    else:
                        rec_proc = ProcedureEvaluationRecord(
                            evaluation_id=eval_id,
                            requirement_id=req.requirement_id,
                            document_id=req.document_id,
                            document_version=req.document_version,
                            category=req.category,
                            verification_status=RequirementVerificationStatus.UNVERIFIED,
                            outcome=EvaluationOutcome.NOT_EVALUATED,
                            reason=(
                                "Qualitative procedure requirement has not received "
                                "explicit operational verification evidence."
                            ),
                            is_satisfied=None,
                            is_qualitative=True,
                            vehicle_id=req.vehicle_id,
                            profile_version=req.profile_version,
                            configuration_id=req.configuration_id,
                            test_card_id=req.test_card_id,
                            test_point_id=req.test_point_id,
                            source_reference=req.source_reference,
                        )
                    procedure_evaluations.append(rec_proc)
                    continue

                # 4B. Quantitative Requirement
                assert req.parameter_id is not None
                obs_list = numeric_obs_map.get(req.parameter_id, [])
                matching_obs: EvaluationObservation | None = None

                # Find best matching observation honoring card and point context
                for candidate in obs_list:
                    if (
                        req.test_point_id is not None
                        and candidate.test_point_id is not None
                        and candidate.test_point_id != req.test_point_id
                    ):
                        continue
                    if (
                        req.test_card_id is not None
                        and candidate.test_card_id is not None
                        and candidate.test_card_id != req.test_card_id
                    ):
                        continue
                    matching_obs = candidate
                    break

                if matching_obs is None:
                    rec_proc = ProcedureEvaluationRecord(
                        evaluation_id=eval_id,
                        requirement_id=req.requirement_id,
                        document_id=req.document_id,
                        document_version=req.document_version,
                        category=req.category,
                        verification_status=RequirementVerificationStatus.UNVERIFIED,
                        outcome=EvaluationOutcome.NOT_EVALUATED,
                        reason=(
                            f"No telemetry observation supplied for parameter '{req.parameter_id}'."
                        ),
                        is_satisfied=None,
                        is_qualitative=False,
                        target_value=req.target_value,
                        tolerance_lower=req.tolerance_lower,
                        tolerance_upper=req.tolerance_upper,
                        effective_lower_bound=req.effective_lower_bound,
                        effective_upper_bound=req.effective_upper_bound,
                        requirement_unit=req.unit,
                        vehicle_id=req.vehicle_id,
                        profile_version=req.profile_version,
                        configuration_id=req.configuration_id,
                        test_card_id=req.test_card_id,
                        test_point_id=req.test_point_id,
                        parameter_id=req.parameter_id,
                        source_reference=req.source_reference,
                    )
                    procedure_evaluations.append(rec_proc)
                    continue

                # Non-finite value check
                if (
                    matching_obs.value is None
                    or math.isnan(matching_obs.value)
                    or math.isinf(matching_obs.value)
                ):
                    rec_proc = ProcedureEvaluationRecord(
                        evaluation_id=eval_id,
                        requirement_id=req.requirement_id,
                        document_id=req.document_id,
                        document_version=req.document_version,
                        category=req.category,
                        verification_status=RequirementVerificationStatus.INCONCLUSIVE,
                        outcome=EvaluationOutcome.INCONCLUSIVE,
                        reason=f"Non-finite or missing observed value: {matching_obs.value}.",
                        observed_value=matching_obs.value,
                        observed_unit=matching_obs.unit,
                        is_satisfied=None,
                        is_qualitative=False,
                        target_value=req.target_value,
                        tolerance_lower=req.tolerance_lower,
                        tolerance_upper=req.tolerance_upper,
                        effective_lower_bound=req.effective_lower_bound,
                        effective_upper_bound=req.effective_upper_bound,
                        requirement_unit=req.unit,
                        vehicle_id=req.vehicle_id,
                        profile_version=req.profile_version,
                        configuration_id=req.configuration_id,
                        test_card_id=req.test_card_id,
                        test_point_id=req.test_point_id,
                        parameter_id=req.parameter_id,
                        source_reference=req.source_reference,
                        observation_id=matching_obs.observation_id,
                        row_index=matching_obs.row_index,
                        timestamp=matching_obs.timestamp,
                    )
                    procedure_evaluations.append(rec_proc)
                    issues.append(
                        Issue(
                            issue_code="ERR_NON_FINITE_OBSERVATION",
                            severity=IssueSeverity.WARNING,
                            message=(
                                f"Non-finite observation value '{matching_obs.value}' "
                                f"for requirement '{req.requirement_id}'."
                            ),
                            field_name=req.parameter_id,
                            location=matching_obs.row_index,
                        )
                    )
                    continue

                # Unit check
                if matching_obs.unit is None or matching_obs.unit != req.unit:
                    rec_proc = ProcedureEvaluationRecord(
                        evaluation_id=eval_id,
                        requirement_id=req.requirement_id,
                        document_id=req.document_id,
                        document_version=req.document_version,
                        category=req.category,
                        verification_status=RequirementVerificationStatus.INCONCLUSIVE,
                        outcome=EvaluationOutcome.INCONCLUSIVE,
                        reason=(
                            f"Unit mismatch: observation unit '{matching_obs.unit}' != "
                            f"requirement unit '{req.unit}'. Silent unit conversion is forbidden."
                        ),
                        observed_value=matching_obs.value,
                        observed_unit=matching_obs.unit,
                        is_satisfied=None,
                        is_qualitative=False,
                        target_value=req.target_value,
                        tolerance_lower=req.tolerance_lower,
                        tolerance_upper=req.tolerance_upper,
                        effective_lower_bound=req.effective_lower_bound,
                        effective_upper_bound=req.effective_upper_bound,
                        requirement_unit=req.unit,
                        vehicle_id=req.vehicle_id,
                        profile_version=req.profile_version,
                        configuration_id=req.configuration_id,
                        test_card_id=req.test_card_id,
                        test_point_id=req.test_point_id,
                        parameter_id=req.parameter_id,
                        source_reference=req.source_reference,
                        observation_id=matching_obs.observation_id,
                        row_index=matching_obs.row_index,
                        timestamp=matching_obs.timestamp,
                    )
                    procedure_evaluations.append(rec_proc)
                    issues.append(
                        Issue(
                            issue_code="ERR_INCOMPATIBLE_UNIT",
                            severity=IssueSeverity.WARNING,
                            message=(
                                f"Observation unit '{matching_obs.unit}' does not match "
                                f"requirement unit '{req.unit}' for requirement "
                                f"'{req.requirement_id}'."
                            ),
                            field_name=req.parameter_id,
                            location=matching_obs.row_index,
                        )
                    )
                    continue

                # Quantitative comparison
                is_sat = req.is_satisfied_by_value(matching_obs.value)
                proc_margin = req.calculate_margin(matching_obs.value)

                if is_sat:
                    v_status = RequirementVerificationStatus.SATISFIED
                    outcome = EvaluationOutcome.SATISFIED
                    reason = (
                        f"Observed value {matching_obs.value} {matching_obs.unit} satisfies "
                        f"requirement tolerance criteria (margin: {proc_margin})."
                    )
                else:
                    v_status = RequirementVerificationStatus.VIOLATED
                    outcome = EvaluationOutcome.VIOLATED
                    reason = (
                        f"Observed value {matching_obs.value} {matching_obs.unit} breaches "
                        f"requirement tolerance criteria (margin: {proc_margin})."
                    )
                    issues.append(
                        Issue(
                            issue_code="VEH_PROCEDURE_VIOLATION",
                            severity=IssueSeverity.WARNING,
                            message=(
                                f"Procedure requirement '{req.requirement_id}' breached by "
                                f"{matching_obs.parameter_id}={matching_obs.value} "
                                f"{matching_obs.unit}."
                            ),
                            field_name=req.parameter_id,
                            location=matching_obs.row_index,
                        )
                    )

                rec_proc = ProcedureEvaluationRecord(
                    evaluation_id=eval_id,
                    requirement_id=req.requirement_id,
                    document_id=req.document_id,
                    document_version=req.document_version,
                    category=req.category,
                    verification_status=v_status,
                    outcome=outcome,
                    reason=reason,
                    observed_value=matching_obs.value,
                    observed_unit=matching_obs.unit,
                    target_value=req.target_value,
                    tolerance_lower=req.tolerance_lower,
                    tolerance_upper=req.tolerance_upper,
                    effective_lower_bound=req.effective_lower_bound,
                    effective_upper_bound=req.effective_upper_bound,
                    requirement_unit=req.unit,
                    margin=proc_margin,
                    is_satisfied=is_sat,
                    is_qualitative=False,
                    vehicle_id=req.vehicle_id,
                    profile_version=req.profile_version,
                    configuration_id=req.configuration_id,
                    test_card_id=req.test_card_id,
                    test_point_id=req.test_point_id,
                    parameter_id=req.parameter_id,
                    source_reference=req.source_reference,
                    observation_id=matching_obs.observation_id,
                    row_index=matching_obs.row_index,
                    timestamp=matching_obs.timestamp,
                )
                procedure_evaluations.append(rec_proc)

        # ---------------------------------------------------------------------
        # 5. Result Status & Provenance Aggregation
        # ---------------------------------------------------------------------
        has_errors = any(
            i.severity in (IssueSeverity.ERROR, IssueSeverity.CRITICAL) for i in issues
        )
        has_warnings = any(i.severity == IssueSeverity.WARNING for i in issues)
        total_evals = len(limit_evaluations) + len(procedure_evaluations)

        if has_errors:
            status = ResultStatus.ERROR
        elif has_warnings:
            status = ResultStatus.WARNING
        elif total_evals == 0:
            status = ResultStatus.NOT_AVAILABLE
        else:
            status = ResultStatus.SUCCESS

        prov = create_provenance(
            source_uri=f"aeris://vehicle_evaluation/{request.request_id}",
            activity_name="vehicle_engineering_evaluation",
            epistemic_level=EpistemicLevel.LIMIT,
            parameters={
                "vehicle_id": vehicle_profile.vehicle_id,
                "profile_version": vehicle_profile.profile_version,
                "configuration_id": active_config_id,
                "evaluate_limits": request.evaluate_limits,
                "evaluate_procedures": request.evaluate_procedures,
                "total_evaluations": total_evals,
                "request_id": request.request_id,
            },
        )

        exceedances = sum(1 for rec in limit_evaluations if rec.is_exceeded)
        violations = sum(1 for rec in procedure_evaluations if rec.is_satisfied is False)
        inconclusives = sum(
            1 for rec in limit_evaluations if rec.outcome == EvaluationOutcome.INCONCLUSIVE
        ) + sum(1 for rec in procedure_evaluations if rec.outcome == EvaluationOutcome.INCONCLUSIVE)

        summary = {
            "total_evaluations": total_evals,
            "limit_evaluations_count": len(limit_evaluations),
            "procedure_evaluations_count": len(procedure_evaluations),
            "limit_exceedances": exceedances,
            "procedure_violations": violations,
            "inconclusive_count": inconclusives,
            "configuration_discrepancies_count": len(configuration_discrepancies),
        }

        return VehicleEvaluationResult(
            evaluation_id=f"VEH_EVAL_{uuid.uuid4().hex[:8].upper()}",
            vehicle_id=vehicle_profile.vehicle_id,
            profile_id=vehicle_profile.profile_id,
            profile_version=vehicle_profile.profile_version,
            configuration_id=active_config_id,
            limit_evaluations=tuple(limit_evaluations),
            procedure_evaluations=tuple(procedure_evaluations),
            configuration_discrepancies=tuple(configuration_discrepancies),
            status=status,
            issues=issues,
            provenance=prov,
            summary=summary,
        )
