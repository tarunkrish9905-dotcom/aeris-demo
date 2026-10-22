"""
AERIS Markdown Engineering Report Renderer.

Renders an immutable EngineeringReport domain model into publication-grade,
GitHub-Flavored Markdown (GFM).

Features:
- Deterministic, byte-identical output for identical report instances.
- Pure Python string builder (no Jinja2, zero external dependencies).
- Strict epistemic boundary labeling ([OBSERVED], [CALCULATED], [LIMIT],
  [PROCEDURAL], [HYPOTHESIS], [RECOMMENDATION]).
- Mandatory non-binding advisory disclaimers on recommendations.
- Safe escaping of table cells (pipe escaping, newline substitution) to prevent
  table layout corruption.
- Transparent reporting of unexecuted stages, inconclusive checks, and data gaps.
- Non-mutating read operations on the input EngineeringReport.
"""

from __future__ import annotations

import datetime
from typing import Any

from aeris.core.reporting.models import EngineeringReport
from aeris.core.results.types import ResultStatus


def _escape_cell(val: Any) -> str:
    """
    Escape text for safe inclusion inside a Markdown table cell.

    Replaces vertical bar separators with escaped bars (\\|) and newlines
    with HTML breaks (<br>) to prevent table structure breakage.
    """
    if val is None:
        return "-"
    s = str(val)
    s = s.replace("\r\n", "<br>").replace("\n", "<br>").replace("\r", "<br>")
    s = s.replace("|", "\\|")
    return s.strip() if s.strip() else "-"


def _fmt_float(val: float | None, precision: int = 4) -> str:
    """Format an optional float safely."""
    if val is None:
        return "-"
    return f"{val:.{precision}f}"


def _fmt_datetime(dt: datetime.datetime | None) -> str:
    """Format an optional datetime safely to ISO 8601 string."""
    if dt is None:
        return "-"
    return dt.isoformat()


class MarkdownReportRenderer:
    """Deterministic, pure-Python Markdown renderer for EngineeringReport domain models."""

    @staticmethod
    def render(report: EngineeringReport) -> str:
        """
        Render an EngineeringReport domain model as GitHub-Flavored Markdown.

        Parameters
        ----------
        report : EngineeringReport
            The authoritative, immutable post-flight engineering report.

        Returns
        -------
        str
            A complete, publication-grade Markdown document.
        """
        lines: list[str] = []

        # -------------------------------------------------------------------
        # Header & Metadata
        # -------------------------------------------------------------------
        lines.append(f"# {report.metadata.title}")
        lines.append("")
        lines.append(f"**Report ID**: `{report.metadata.report_id}`  ")
        lines.append(f"**Generated (UTC)**: `{_fmt_datetime(report.metadata.generated_at)}`  ")
        lines.append(f"**Analyst**: {report.metadata.analyst_name or 'Unassigned'}  ")
        lines.append(f"**Pipeline Run ID**: `{report.metadata.pipeline_run_id or 'N/A'}`  ")
        lines.append(f"**Source Telemetry File**: `{report.metadata.source_file or 'N/A'}`  ")
        lines.append(f"**Dataset SHA-256**: `{report.metadata.source_file_hash or 'N/A'}`  ")
        if report.metadata.execution_duration_seconds is not None:
            dur = report.metadata.execution_duration_seconds
            lines.append(f"**Pipeline Execution Duration**: {dur:.3f} s  ")
        lines.append(f"**Pipeline Version**: {report.metadata.pipeline_version}")

        if report.metadata.custom_metadata:
            lines.append("")
            lines.append("### Custom Metadata")
            for k, v in sorted(report.metadata.custom_metadata.items()):
                lines.append(f"- **{_escape_cell(k)}**: {_escape_cell(v)}")

        lines.append("")
        lines.append("---")
        lines.append("")

        # -------------------------------------------------------------------
        # 1. Executive Summary
        # -------------------------------------------------------------------
        lines.append("## 1. Executive Summary")
        lines.append("")

        # Alert banner based on overall status
        status = report.executive_summary.overall_status
        if status == ResultStatus.SUCCESS:
            lines.append("> [!NOTE]")
            lines.append(
                "> **Pipeline Execution Verdict: SUCCESS** — "
                "All analytical stages completed within normal operational margins."
            )
        elif status == ResultStatus.WARNING:
            lines.append("> [!WARNING]")
            lines.append(
                "> **Pipeline Execution Verdict: WARNING** — "
                "Analytical advisories, parameter anomalies, or margin breaches were identified."
            )
        elif status == ResultStatus.ERROR:
            lines.append("> [!CAUTION]")
            lines.append(
                "> **Pipeline Execution Verdict: ERROR** — "
                "Critical defects, validation failures, or safety boundary violations occurred."
            )
        elif status == ResultStatus.PARTIAL:
            lines.append("> [!IMPORTANT]")
            lines.append(
                "> **Pipeline Execution Verdict: PARTIAL** — "
                "Analysis executed with partial coverage, unrequested stages, or degraded datasets."
            )
        else:
            lines.append("> [!NOTE]")
            lines.append(f"> **Pipeline Execution Verdict: {status.value}**")

        lines.append("")
        lines.append("### Key Execution Metrics")
        lines.append("")
        lines.append("| Metric | Value | Epistemic Level |")
        lines.append("| :--- | :--- | :---: |")
        lines.append(f"| **Overall Pipeline Status** | `{status.value}` | `[CALCULATED]` |")
        lines.append(
            f"| **Executed / Total Stages** | "
            f"{report.executive_summary.executed_stages_count} / "
            f"{report.executive_summary.total_stages} | `[CALCULATED]` |"
        )
        lines.append(
            f"| **Failed Stages** | "
            f"{report.executive_summary.failed_stages_count} | `[CALCULATED]` |"
        )
        lines.append(
            f"| **Warning Stages** | "
            f"{report.executive_summary.warning_stages_count} | `[CALCULATED]` |"
        )
        lines.append(
            f"| **Skipped / Unexecuted Stages** | "
            f"{report.executive_summary.skipped_stages_count} | `[CALCULATED]` |"
        )
        total_iss = report.executive_summary.total_issues_count
        crit_iss = report.executive_summary.total_critical_or_error_count
        lines.append(
            f"| **Total Logged Issues** | {total_iss} (Errors/Critical: {crit_iss}) | "
            "`[CALCULATED]` |"
        )
        lines.append(
            f"| **Limit Exceedances** | "
            f"{report.executive_summary.total_limit_exceedances} | `[LIMIT]` |"
        )
        lines.append(
            f"| **Procedure Violations** | "
            f"{report.executive_summary.total_procedure_violations} | `[PROCEDURAL]` |"
        )
        lines.append(
            f"| **Anomalies Detected** | "
            f"{report.executive_summary.total_anomalies_detected} | `[CALCULATED]` |"
        )
        lines.append(
            f"| **Advisory Recommendations** | "
            f"{report.executive_summary.total_recommendations_generated} | `[RECOMMENDATION]` |"
        )
        lines.append("")

        lines.append("### Operational Narrative")
        lines.append("")
        lines.append(report.executive_summary.high_level_narrative)
        lines.append("")
        lines.append("---")
        lines.append("")

        # -------------------------------------------------------------------
        # 2. Pipeline Stage Execution Status
        # -------------------------------------------------------------------
        lines.append("## 2. Pipeline Stage Execution Status")
        lines.append("")
        if report.stage_summaries:
            lines.append(
                "| Order | Pipeline Stage | Status | Executed | "
                "Issues | Errors | Warnings | Operational Notes |"
            )
            lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
            for st in sorted(report.stage_summaries, key=lambda s: s.stage_order):
                exec_str = "Yes" if st.is_executed else "No"
                notes_str = _escape_cell(st.notes) if st.notes else "-"
                lines.append(
                    f"| {st.stage_order} | {_escape_cell(st.stage_name)} | `{st.status.value}` | "
                    f"{exec_str} | {st.issue_count} | {st.error_count} | {st.warning_count} | "
                    f"{notes_str} |"
                )
        else:
            lines.append("*No pipeline stage execution records available.*")

        lines.append("")
        lines.append("---")
        lines.append("")

        # -------------------------------------------------------------------
        # 3. Flight Context & Vehicle Configuration
        # -------------------------------------------------------------------
        lines.append("## 3. Flight Context & Test Operational Boundaries")
        lines.append("")
        ctx = report.flight_context
        lines.append("| Parameter | Value | Description |")
        lines.append("| :--- | :--- | :--- |")
        lines.append(
            f"| **Vehicle Identifier** | `{_escape_cell(ctx.vehicle_id)}` | "
            "Active airframe identifier |"
        )
        prof_info = f"{ctx.profile_id} (v{ctx.profile_version})" if ctx.profile_id else "-"
        lines.append(
            f"| **Vehicle Profile** | `{_escape_cell(prof_info)}` | "
            "Aerodynamic & subsystem definition |"
        )
        lines.append(
            f"| **Configuration Envelope** | `{_escape_cell(ctx.configuration_id)}` | "
            "Flaps, gear, loading envelope |"
        )
        lines.append(
            f"| **Flight Test Card** | `{_escape_cell(ctx.test_card_id)}` | "
            "Planned flight card identifier |"
        )
        lines.append(
            f"| **Test Point Identifier** | `{_escape_cell(ctx.test_point_id)}` | "
            "Specific maneuver point under test |"
        )
        lines.append(
            f"| **Observation Identifier** | `{_escape_cell(ctx.observation_id)}` | "
            "Telemetry temporal slice observation |"
        )
        rows_str = str(ctx.associated_rows_count) if ctx.associated_rows_count is not None else "-"
        lines.append(
            f"| **Associated Telemetry Rows** | {rows_str} | Data rows correlated to test point |"
        )
        lines.append("")

        if ctx.configuration_discrepancies:
            lines.append("> [!WARNING]")
            lines.append("> **Configuration Discrepancies Detected**:")
            for disc in ctx.configuration_discrepancies:
                lines.append(f"> - {_escape_cell(disc)}")
            lines.append("")
        else:
            lines.append("*No vehicle configuration discrepancies detected.*")

        lines.append("")
        lines.append("---")
        lines.append("")

        # -------------------------------------------------------------------
        # 4. Data Quality & Telemetry Validation
        # -------------------------------------------------------------------
        lines.append("## 4. Data Quality & Telemetry Validation")
        lines.append("")
        dq = report.data_quality
        val_str = "VALID" if dq.is_valid is True else ("INVALID" if dq.is_valid is False else "N/A")
        score_str = f"{dq.quality_score * 100.0:.1f}%" if dq.quality_score is not None else "N/A"
        if dq.parameter_validation_passed is True:
            param_val_str = "PASSED"
        elif dq.parameter_validation_passed is False:
            param_val_str = "FAILED"
        else:
            param_val_str = "N/A"

        lines.append("| Metric | Assessment | Details |")
        lines.append("| :--- | :---: | :--- |")
        lines.append(
            f"| **Validation Verdict** | `{val_str}` | Telemetry structure and schema integrity |"
        )
        lines.append(
            f"| **Data Quality Score** | `{score_str}` | Aggregate sensor health & completeness |"
        )
        lines.append(
            f"| **Ingested Dimensions** | {dq.total_rows_ingested} rows x "
            f"{dq.total_columns_ingested} cols | Telemetry file raw matrix |"
        )
        lines.append(
            f"| **Schema / Range Violations** | {dq.violation_count} | "
            "Detected format or type defects |"
        )
        if dq.normalization_applied:
            norm_str = f"Yes ({dq.transformations_count} transforms applied)"
        else:
            norm_str = "No"
        lines.append(
            f"| **Normalization Applied** | {norm_str} | Unit conversions & timestamp alignments |"
        )
        lines.append(
            f"| **Parameter Validation** | `{param_val_str}` | "
            "Semantic range checks against registry |"
        )
        lines.append("")

        if dq.quality_summary_text:
            lines.append("### Quality Summary Description")
            lines.append("")
            lines.append(dq.quality_summary_text)
            lines.append("")

        if dq.issues:
            lines.append("### Data Quality & Validation Issues")
            lines.append("")
            lines.append("| Severity | Code | Field | Location | Message |")
            lines.append("| :---: | :--- | :--- | :---: | :--- |")
            for iss in dq.issues:
                loc_str = str(iss.location) if iss.location is not None else "-"
                lines.append(
                    f"| `{iss.severity.value}` | `{_escape_cell(iss.issue_code)}` | "
                    f"`{_escape_cell(iss.field_name)}` | {loc_str} | "
                    f"{_escape_cell(iss.message)} |"
                )
            lines.append("")

        lines.append("---")
        lines.append("")

        # -------------------------------------------------------------------
        # 5. Vehicle Engineering Evaluations
        # -------------------------------------------------------------------
        lines.append("## 5. Vehicle Engineering Evaluations")
        lines.append("")
        ee = report.engineering_evaluation

        lines.append(
            f"**Limits Evaluated**: {ee.total_limits_evaluated} | "
            f"**Exceedances**: {ee.limit_exceedance_count} | "
            f"**Inconclusive**: {ee.limit_inconclusive_count}  "
        )
        lines.append(
            f"**Procedures Evaluated**: {ee.total_procedures_evaluated} | "
            f"**Violations**: {ee.procedure_violation_count} | "
            f"**Inconclusive**: {ee.procedure_inconclusive_count}"
        )
        lines.append("")

        # 5.1 Limits
        lines.append("### 5.1 Engineering Limit Evaluations `[LIMIT]`")
        lines.append("")
        if ee.limit_evaluations:
            lines.append(
                "| Limit ID | Parameter | Type | Severity | Boundary | "
                "Observed | Margin | Outcome | Reason |"
            )
            lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
            for lim in ee.limit_evaluations:
                unit_str = f" {lim.unit}" if lim.unit else ""
                if lim.boundary_value is not None:
                    bound_str = f"{_fmt_float(lim.boundary_value)}{unit_str}"
                else:
                    bound_str = "-"
                if lim.observed_value is not None:
                    obs_str = f"{_fmt_float(lim.observed_value)}{unit_str}"
                else:
                    obs_str = "-"
                if lim.margin is not None:
                    margin_str = f"{_fmt_float(lim.margin)}{unit_str}"
                else:
                    margin_str = "-"
                outcome_fmt = f"**`{lim.outcome}`**" if lim.is_exceeded else f"`{lim.outcome}`"
                lines.append(
                    f"| `{_escape_cell(lim.limit_id)}` | `{_escape_cell(lim.parameter_id)}` | "
                    f"{_escape_cell(lim.limit_type)} | `{_escape_cell(lim.severity)}` | "
                    f"{bound_str} | {obs_str} | {margin_str} | {outcome_fmt} | "
                    f"{_escape_cell(lim.reason)} |"
                )
        else:
            lines.append("*No engineering limit evaluations performed or attached.*")

        lines.append("")

        # 5.2 Procedures
        lines.append("### 5.2 SOP Procedure Requirement Evaluations `[PROCEDURAL]`")
        lines.append("")
        if ee.procedure_evaluations:
            lines.append(
                "| Requirement ID | Document ID | Category | Target | Observed | "
                "Margin | Satisfied | Reason |"
            )
            lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
            for pr in ee.procedure_evaluations:
                unit_str = f" {pr.unit}" if pr.unit else ""
                if pr.target_value is not None:
                    target_str = f"{_fmt_float(pr.target_value)}{unit_str}"
                else:
                    target_str = "-"
                if pr.is_qualitative:
                    obs_str = f"`{_escape_cell(pr.qualitative_state)}`"
                elif pr.observed_value is not None:
                    obs_str = f"{_fmt_float(pr.observed_value)}{unit_str}"
                else:
                    obs_str = "-"
                margin_str = f"{_fmt_float(pr.margin)}{unit_str}" if pr.margin is not None else "-"
                if pr.is_satisfied is True:
                    sat_str = "PASSED"
                elif pr.is_satisfied is False:
                    sat_str = "VIOLATED"
                else:
                    sat_str = "INCONCLUSIVE"
                sat_fmt = f"**`{sat_str}`**" if pr.is_satisfied is False else f"`{sat_str}`"
                lines.append(
                    f"| `{_escape_cell(pr.requirement_id)}` | `{_escape_cell(pr.document_id)}` | "
                    f"`{_escape_cell(pr.category)}` | {target_str} | {obs_str} | {margin_str} | "
                    f"{sat_fmt} | {_escape_cell(pr.reason)} |"
                )
        else:
            lines.append("*No procedural requirements evaluated or attached.*")

        lines.append("")

        # 5.3 Derived Engineering Calculations
        lines.append("### 5.3 Derived Engineering Calculations `[CALCULATED]`")
        lines.append("")
        if ee.calculations_performed:
            for calc in ee.calculations_performed:
                lines.append(f"- `{_escape_cell(calc)}`")
        else:
            lines.append("*No derived engineering calculations performed.*")

        lines.append("")
        lines.append("---")
        lines.append("")

        # -------------------------------------------------------------------
        # 6. Telemetry Anomalies & Comparisons
        # -------------------------------------------------------------------
        lines.append("## 6. Telemetry Anomalies & Comparisons")
        lines.append("")
        ac = report.anomaly_and_comparison
        lines.append(
            f"**Total Anomalies**: {ac.total_anomalies} "
            f"(HIGH: {ac.high_severity_anomalies}, "
            f"MEDIUM: {ac.medium_severity_anomalies}, "
            f"LOW: {ac.low_severity_anomalies}, "
            f"INFO: {ac.info_severity_anomalies}) | "
            f"**Comparisons**: {ac.total_comparisons} "
            f"(Within Tolerance: {ac.comparisons_within_tolerance}, "
            f"Breached: {ac.comparisons_out_of_tolerance})"
        )
        lines.append("")

        # 6.1 Anomalies
        lines.append("### 6.1 Telemetry Anomalies `[CALCULATED]`")
        lines.append("")
        if ac.anomalies:
            lines.append(
                "| Anomaly ID | Parameter | Classification | Severity | Method | "
                "Timestamp | Observed | Expected | Deviation | Explanation |"
            )
            lines.append(
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
            )
            for an in ac.anomalies:
                ts_str = _fmt_datetime(an.timestamp)
                obs_str = _escape_cell(an.observed_value)
                exp_str = _escape_cell(an.expected_value)
                dev_str = _fmt_float(an.deviation_value) if an.deviation_value is not None else "-"
                lines.append(
                    f"| `{_escape_cell(an.anomaly_id)}` | `{_escape_cell(an.target_column)}` | "
                    f"`{_escape_cell(an.classification)}` | `{_escape_cell(an.severity)}` | "
                    f"`{_escape_cell(an.detection_method)}` | {ts_str} | {obs_str} | {exp_str} | "
                    f"{dev_str} | {_escape_cell(an.explanation)} |"
                )
        else:
            lines.append("*No telemetry anomalies detected.*")

        lines.append("")

        # 6.2 Comparisons
        lines.append("### 6.2 Expected vs Observed Comparisons `[CALCULATED]`")
        lines.append("")
        if ac.comparisons:
            lines.append(
                "| Comparison ID | Parameter | Expected | Observed | "
                "Deviation | In Tolerance | Status |"
            )
            lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
            for cmp in ac.comparisons:
                dev_str = _fmt_float(cmp.deviation) if cmp.deviation is not None else "-"
                if cmp.within_tolerance is True:
                    tol_str = "Yes"
                elif cmp.within_tolerance is False:
                    tol_str = "No"
                else:
                    tol_str = "N/A"
                lines.append(
                    f"| `{_escape_cell(cmp.comparison_id)}` | "
                    f"`{_escape_cell(cmp.parameter_id)}` | "
                    f"{_escape_cell(cmp.expected_value)} | "
                    f"{_escape_cell(cmp.observed_value)} | "
                    f"{dev_str} | {tol_str} | `{cmp.status.value}` |"
                )
        else:
            lines.append("*No expected-vs-observed comparisons evaluated.*")

        lines.append("")
        lines.append("---")
        lines.append("")

        # -------------------------------------------------------------------
        # 7. Investigation Intelligence & Recommendations
        # -------------------------------------------------------------------
        lines.append("## 7. Investigation Intelligence & Recommendations")
        lines.append("")
        ii = report.investigation_intelligence

        # 7.1 Evidence Items
        lines.append("### 7.1 Traceable Evidence Items")
        lines.append("")
        if ii.evidence_items:
            lines.append(
                "| Evidence ID | Type | Parameter | Observed | Margin | "
                "Epistemic Level | Description | Source Reference |"
            )
            lines.append("| :--- | :---: | :--- | :---: | :---: | :---: | :--- | :--- |")
            for ev in ii.evidence_items:
                obs_str = _fmt_float(ev.observed_value) if ev.observed_value is not None else "-"
                margin_str = _fmt_float(ev.margin) if ev.margin is not None else "-"
                lines.append(
                    f"| `{_escape_cell(ev.evidence_id)}` | `{_escape_cell(ev.evidence_type)}` | "
                    f"`{_escape_cell(ev.parameter_id)}` | {obs_str} | {margin_str} | "
                    f"`[{ev.epistemic_level.value}]` | {_escape_cell(ev.description)} | "
                    f"`{_escape_cell(ev.source_reference)}` |"
                )
        else:
            lines.append("*No structured evidence items recorded.*")

        lines.append("")

        # 7.2 Contributor Rankings
        lines.append("### 7.2 Contributor Correlation Rankings `[CALCULATED]`")
        lines.append("")
        if ii.contributor_rankings:
            lines.append(
                "| Analysis ID | Target Column | Candidate Factor | Method | "
                "Coefficient | p-value | Status |"
            )
            lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: |")
            for cr in ii.contributor_rankings:
                coeff_str = _fmt_float(cr.coefficient) if cr.coefficient is not None else "-"
                pval_str = _fmt_float(cr.p_value, 5) if cr.p_value is not None else "-"
                lines.append(
                    f"| `{_escape_cell(cr.analysis_id)}` | `{_escape_cell(cr.target_column)}` | "
                    f"`{_escape_cell(cr.factor_column)}` | `{_escape_cell(cr.method)}` | "
                    f"{coeff_str} | {pval_str} | `{cr.status.value}` |"
                )
        else:
            lines.append("*No contributor correlation rankings generated.*")

        lines.append("")

        # 7.3 Recommendations
        lines.append("### 7.3 Advisory Recommendations `[RECOMMENDATION]`")
        lines.append("")
        if ii.recommendations:
            for rec in ii.recommendations:
                rec_title = _escape_cell(rec.title)
                lines.append(f"#### Recommendation `{rec.recommendation_id}`: {rec_title}")
                lines.append("")
                lines.append("> [!IMPORTANT]")
                lines.append(f"> **Advisory Notice**: {rec.advisory_notice}")
                lines.append("")
                lines.append(f"- **Type**: `{rec.recommendation_type}`  ")
                lines.append(f"- **Status**: `{rec.status}`  ")
                rule_str = rec.rule_id if rec.rule_id else "N/A"
                lines.append(f"- **Triggered Rule**: `{rule_str}`  ")
                if rec.evidence_ids:
                    ev_str = ", ".join(f"`{eid}`" for eid in rec.evidence_ids)
                else:
                    ev_str = "None"
                lines.append(f"- **Traceable Evidence IDs**: {ev_str}  ")
                lines.append(f"- **Engineering Rationale**: {rec.rationale}  ")
                lines.append(f"- **Detailed Description**: {rec.description}")
                lines.append("")
        else:
            lines.append("*No advisory recommendations generated.*")
            lines.append("")

        lines.append("---")
        lines.append("")

        # -------------------------------------------------------------------
        # 8. Analytical Limitations, Caveats & Data Gaps
        # -------------------------------------------------------------------
        lines.append("## 8. Analytical Limitations, Caveats & Data Gaps")
        lines.append("")
        lims = report.limitations
        if lims.has_limitations:
            lines.append("> [!NOTE]")
            lines.append("> **Operational Limitations and Analysis Caveats are Active**")
            lines.append("")
            if lims.unexecuted_stages:
                unexec_str = ", ".join(f"`{s}`" for s in lims.unexecuted_stages)
            else:
                unexec_str = "None"
            lines.append(f"- **Unexecuted Stages**: {unexec_str}")
            if lims.failed_stages:
                fail_str = ", ".join(f"`{s}`" for s in lims.failed_stages)
            else:
                fail_str = "None"
            lines.append(f"- **Failed Stages**: {fail_str}")
            lines.append(f"- **Inconclusive Evaluations**: {lims.inconclusive_checks_count}")
            lines.append("")

            if lims.items:
                lines.append("| Limitation ID | Category | Stage | Impact | Description |")
                lines.append("| :--- | :---: | :---: | :--- | :--- |")
                for it in lims.items:
                    st_str = it.stage_name if it.stage_name else "-"
                    lines.append(
                        f"| `{_escape_cell(it.limitation_id)}` | `{_escape_cell(it.category)}` | "
                        f"`{_escape_cell(st_str)}` | {_escape_cell(it.impact)} | "
                        f"{_escape_cell(it.description)} |"
                    )
                lines.append("")
        else:
            lines.append("*No operational limitations, skipped stages, or data gaps identified.*")
            lines.append("")

        lines.append("---")
        lines.append("")

        # -------------------------------------------------------------------
        # 9. Provenance & Audit Trail
        # -------------------------------------------------------------------
        lines.append("## 9. Provenance & Audit Trail")
        lines.append("")
        prov = report.provenance
        lines.append("| Provenance Field | Value | Epistemic Level |")
        lines.append("| :--- | :--- | :---: |")
        lines.append(
            f"| **Root Provenance ID** | `{_escape_cell(prov.root_provenance_id)}` | "
            f"`[{prov.epistemic_level.value}]` |"
        )
        lines.append(
            f"| **Source Dataset URI** | `{_escape_cell(prov.source_uri)}` | `[OBSERVED]` |"
        )
        lines.append(
            f"| **Root Activity Name** | `{_escape_cell(prov.activity_name)}` | `[CALCULATED]` |"
        )
        if prov.parent_record_ids:
            parents_str = ", ".join(f"`{p}`" for p in prov.parent_record_ids)
        else:
            parents_str = "None"
        lines.append(f"| **Parent Provenance IDs** | {parents_str} | - |")
        lines.append("")

        if prov.stage_provenance_records:
            lines.append("### Stage Provenance Lineage Map")
            lines.append("")
            lines.append("| Pipeline Stage | Provenance Record ID |")
            lines.append("| :--- | :--- |")
            for st_name, rec_id in sorted(prov.stage_provenance_records.items()):
                lines.append(f"| `{_escape_cell(st_name)}` | `{_escape_cell(rec_id)}` |")
            lines.append("")

        # -------------------------------------------------------------------
        # 10. Comprehensive Issue Log
        # -------------------------------------------------------------------
        if report.all_issues:
            lines.append("---")
            lines.append("")
            lines.append("## 10. Comprehensive Pipeline Issue Log")
            lines.append("")
            lines.append("| Severity | Stage | Code | Location | Field | Message |")
            lines.append("| :---: | :--- | :--- | :---: | :--- | :--- |")
            for iss in report.all_issues:
                loc_str = str(iss.location) if iss.location is not None else "-"
                st_str = iss.stage if hasattr(iss, "stage") else "-"
                lines.append(
                    f"| `{iss.severity.value}` | `{_escape_cell(st_str)}` | "
                    f"`{_escape_cell(iss.issue_code)}` | {loc_str} | "
                    f"`{_escape_cell(iss.field_name)}` | {_escape_cell(iss.message)} |"
                )
            lines.append("")

        return "\n".join(lines).strip() + "\n"


def render_markdown_report(report: EngineeringReport) -> str:
    """
    Convenience function to render an EngineeringReport as GitHub-Flavored Markdown.

    Parameters
    ----------
    report : EngineeringReport
        The report domain model instance.

    Returns
    -------
    str
        Rendered Markdown content.
    """
    return MarkdownReportRenderer.render(report)
