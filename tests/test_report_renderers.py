"""
Tests for AERIS Engineering Report Renderers (Markdown and HTML) — Step 14f.

Validates:
1. Markdown rendering from complete and partial EngineeringReport domain models.
2. HTML rendering from complete, partial, and empty EngineeringReport domain models.
3. Preservation of epistemic distinctions ([OBSERVED], [CALCULATED], [LIMIT],
   [PROCEDURAL], [HYPOTHESIS], [RECOMMENDATION]).
4. Retention of mandatory advisory notices on all recommendations.
5. Strict escaping of Markdown table cell characters (pipes, newlines) to prevent table breakage.
6. Absolute XSS prevention: HTML-sensitive characters (<script>, <img>, event handlers)
   are escaped to harmless text.
7. Completely self-contained offline HTML: zero external script/font calls when figures=None.
8. Safe, bounded Plotly figure integration in HTML (sequences, mappings, TimeSeriesPlotResult).
9. Determinism: identical report inputs produce byte-identical rendering outputs.
10. Immutability: input EngineeringReport domain models are never mutated during rendering.
11. Accessibility and print stylesheet inclusion in HTML (@media print).
"""

from __future__ import annotations

import datetime
import pathlib

import pandas as pd
import plotly.graph_objects as go
import pytest

from aeris.core.evidence.pipeline import EvidenceSourceSelection, PipelineEvidenceRequest
from aeris.core.hypothesis.assessment_models import (
    HypothesisAssessmentConfiguration,
    HypothesisAssessmentResult,
)
from aeris.core.hypothesis.models import HypothesisAssessmentState
from aeris.core.limits.models import (
    EngineeringLimitProfile,
    LimitSeverity,
    LimitType,
    ParameterLimitDefinition,
)
from aeris.core.pipeline.orchestrator import run_csv_pipeline
from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.recommendation.generation_models import RecommendationGenerationPolicy
from aeris.core.recommendation.pipeline import PipelineRecommendationRequest
from aeris.core.reporting import (
    AnomalyAndComparisonSummary,
    DataQualitySummary,
    EngineeringEvaluationSummary,
    EngineeringReport,
    ExecutiveSummary,
    FlightContextSummary,
    HtmlReportRenderer,
    InvestigationIntelligenceSummary,
    LimitationItem,
    LimitationsSummary,
    LimitEvaluationItem,
    MarkdownReportRenderer,
    ProvenanceSummary,
    ReportMetadata,
    StageExecutionSummary,
    StageStatus,
    extract_engineering_report,
    render_html_report,
    render_markdown_report,
)
from aeris.core.results.types import Issue, IssueSeverity, ResultStatus
from aeris.core.sop.models import (
    ProcedureRequirement,
    ProcedureRequirementProfile,
    RequirementCategory,
)
from aeris.core.vehicle.envelope_models import ConfigurationEnvelope
from aeris.core.vehicle.profile_models import VehicleProfile
from aeris.core.vehicle_evaluation.pipeline import PipelineVehicleEvaluationRequest
from aeris.core.visualization.models import TimeSeriesPlotResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_vehicle_profile() -> VehicleProfile:
    """Return a standard synthetic VehicleProfile."""
    return VehicleProfile(
        profile_id="PROF-SYNTH-X1",
        vehicle_id="AC-SYNTH-01",
        profile_version="1.0.0",
        name="Synthetic Model X1",
        supported_parameter_ids=("airspeed_kts", "altitude_ft", "nz_g", "gear_handle"),
    )


@pytest.fixture
def synthetic_envelope() -> ConfigurationEnvelope:
    """Return a standard synthetic ConfigurationEnvelope."""
    return ConfigurationEnvelope(
        configuration_id="CRUISE_CLEAN",
        vehicle_id="AC-SYNTH-01",
        profile_id="PROF-SYNTH-X1",
        profile_version="1.0.0",
        name="Clean Cruise Envelope",
    )


@pytest.fixture
def synthetic_limit_profile() -> EngineeringLimitProfile:
    """Return a synthetic limit profile with upper bound airspeed limit."""
    return EngineeringLimitProfile(
        limit_profile_id="LIM-PROF-SYNTH-01",
        vehicle_id="AC-SYNTH-01",
        profile_id="PROF-SYNTH-X1",
        profile_version="1.0.0",
        limits=(
            ParameterLimitDefinition(
                limit_id="LIM-VNE-01",
                parameter_id="airspeed_kts",
                limit_type=LimitType.UPPER_BOUND,
                upper_bound=250.0,
                unit="kts",
                severity=LimitSeverity.WARNING,
                vehicle_id="AC-SYNTH-01",
                profile_id="PROF-SYNTH-X1",
                profile_version="1.0.0",
                source_reference="AFM-SEC2",
                description="Never exceed airspeed (Vne)",
            ),
        ),
    )


@pytest.fixture
def synthetic_procedure_profile() -> ProcedureRequirementProfile:
    """Return a synthetic procedure requirement profile."""
    return ProcedureRequirementProfile(
        profile_id="PROC-PROF-SYNTH-01",
        document_id="SOP-SYNTH-001",
        document_version="1.2",
        vehicle_id="AC-SYNTH-01",
        profile_version="1.0.0",
        requirements=(
            ProcedureRequirement(
                requirement_id="REQ-TARGET-ALT",
                document_id="SOP-SYNTH-001",
                document_version="1.2",
                category=RequirementCategory.TOLERANCE_BAND,
                parameter_id="altitude_ft",
                target_value=10000.0,
                tolerance=100.0,
                unit="ft",
                source_reference="SOP-SYNTH-001 Sec 3.1",
                description="Maintain cruise altitude within 100 ft tolerance",
            ),
        ),
    )


@pytest.fixture
def flight_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    """Generate a clean synthetic flight telemetry CSV."""
    csv_file = tmp_path / "flight_telemetry.csv"
    timestamps = [
        "2026-10-14T12:00:00Z",
        "2026-10-14T12:00:01Z",
        "2026-10-14T12:00:02Z",
    ]
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "airspeed_kts": [220.0, 260.0, 225.0],  # 260.0 breaches 250.0 limit
            "altitude_ft": [10000.0, 10020.0, 9990.0],
            "pitch_deg": [2.0, 2.1, 2.0],
            "roll_deg": [0.0, 0.2, -0.1],
            "engine_rpm": [2400.0, 2410.0, 2405.0],
            "nz_g": [1.0, 1.05, 0.98],
            "gear_handle": ["UP", "UP", "UP"],
            "configuration_id": ["CRUISE_CLEAN", "CRUISE_CLEAN", "CRUISE_CLEAN"],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def synthetic_hypothesis_assessment() -> HypothesisAssessmentResult:
    """Return an inconclusive hypothesis assessment to legitimately trigger engineering review."""
    prov = create_provenance(
        activity_name="HypothesisAssessment",
        source_uri="urn:aeris:assessment:hyp-synth-01",
        epistemic_level=EpistemicLevel.HYPOTHESIS,
    )
    return HypothesisAssessmentResult(
        hypothesis_id="HYP-SYNTH-01",
        assessment_state=HypothesisAssessmentState.INCONCLUSIVE,
        supporting_evidence_ids=(),
        contradicting_evidence_ids=(),
        configuration_used=HypothesisAssessmentConfiguration(),
        provenance=prov,
        status=ResultStatus.SUCCESS,
    )


@pytest.fixture
def full_engineering_report(
    flight_csv: pathlib.Path,
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_envelope: ConfigurationEnvelope,
    synthetic_limit_profile: EngineeringLimitProfile,
    synthetic_procedure_profile: ProcedureRequirementProfile,
    synthetic_hypothesis_assessment: HypothesisAssessmentResult,
) -> EngineeringReport:
    """Return an extracted full EngineeringReport from end-to-end pipeline run."""
    veh_req = PipelineVehicleEvaluationRequest(
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_envelope,
        limit_profile=synthetic_limit_profile,
        procedure_profile=synthetic_procedure_profile,
        auto_extract_observations=True,
    )
    ev_req = PipelineEvidenceRequest(
        collection_id="EV-COLL-REPORT-TEST",
        selections=EvidenceSourceSelection(include_vehicle_evaluation=True),
    )
    rec_req = PipelineRecommendationRequest(
        hypothesis_assessment=synthetic_hypothesis_assessment,
        policy=RecommendationGenerationPolicy(
            enable_data_verification_rule=True,
            enable_engineering_review_rule=True,
        ),
        use_pipeline_evidence=True,
        use_pipeline_validation=True,
    )
    pipeline_result = run_csv_pipeline(
        file_path=flight_csv,
        vehicle_evaluation_request=veh_req,
        evidence_request=ev_req,
        recommendation_request=rec_req,
    )
    return extract_engineering_report(
        pipeline_result=pipeline_result,
        report_title="AERIS Flight Test 401 Engineering Report",
        analyst_name="Maj. John Glenn",
        custom_metadata={"Flight Phase": "Cruise Decel", "Test Day": "Day 4"},
    )


@pytest.fixture
def empty_engineering_report() -> EngineeringReport:
    """Return an EngineeringReport where all evaluation collections are empty."""
    now = datetime.datetime.now(datetime.UTC)
    return EngineeringReport(
        metadata=ReportMetadata(
            report_id="REP-EMPTY-001",
            title="AERIS Empty Baseline Report",
            analyst_name=None,
            generated_at=now,
            pipeline_run_id=None,
            source_file=None,
            source_file_hash=None,
            execution_duration_seconds=0.012,
            pipeline_version="1.0.0",
        ),
        executive_summary=ExecutiveSummary(
            overall_status=ResultStatus.SUCCESS,
            total_stages=15,
            executed_stages_count=2,
            failed_stages_count=0,
            warning_stages_count=0,
            skipped_stages_count=13,
            total_issues_count=0,
            total_critical_or_error_count=0,
            total_limit_exceedances=0,
            total_procedure_violations=0,
            total_anomalies_detected=0,
            total_recommendations_generated=0,
            high_level_narrative="Baseline run without vehicle evaluations or anomalies.",
        ),
        stage_summaries=(
            StageExecutionSummary(
                stage_name="ingestion",
                stage_order=1,
                status=StageStatus.SUCCESS,
                is_executed=True,
                issue_count=0,
                error_count=0,
                warning_count=0,
            ),
            StageExecutionSummary(
                stage_name="validation",
                stage_order=2,
                status=StageStatus.SUCCESS,
                is_executed=True,
                issue_count=0,
                error_count=0,
                warning_count=0,
            ),
        ),
        data_quality=DataQualitySummary(
            is_valid=True,
            quality_score=1.0,
            total_rows_ingested=10,
            total_columns_ingested=5,
            columns_ingested=("timestamp", "airspeed_kts", "altitude_ft"),
            columns_evaluated=("airspeed_kts", "altitude_ft"),
            violation_count=0,
            quality_metrics={},
            quality_summary_text="",
            normalization_applied=False,
            transformations_count=0,
            parameter_validation_passed=True,
            validated_parameters=("airspeed_kts", "altitude_ft"),
            failed_parameters=(),
            issues=(),
        ),
        flight_context=FlightContextSummary(
            vehicle_id=None,
            profile_id=None,
            profile_version=None,
            configuration_id=None,
            test_card_id=None,
            test_point_id=None,
            observation_id=None,
            associated_rows_count=None,
            configuration_discrepancies=(),
        ),
        engineering_evaluation=EngineeringEvaluationSummary(
            is_evaluated=False,
            total_limits_evaluated=0,
            limit_exceedance_count=0,
            limit_inconclusive_count=0,
            total_procedures_evaluated=0,
            procedure_violation_count=0,
            procedure_inconclusive_count=0,
            limit_evaluations=(),
            procedure_evaluations=(),
            calculations_performed=(),
        ),
        anomaly_and_comparison=AnomalyAndComparisonSummary(
            total_anomalies=0,
            high_severity_anomalies=0,
            medium_severity_anomalies=0,
            low_severity_anomalies=0,
            info_severity_anomalies=0,
            anomalies=(),
            total_comparisons=0,
            comparisons_within_tolerance=0,
            comparisons_out_of_tolerance=0,
            comparisons=(),
        ),
        investigation_intelligence=InvestigationIntelligenceSummary(
            evidence_count=0,
            evidence_items=(),
            contributor_rankings=(),
            recommendation_count=0,
            rules_evaluated=(),
            rules_triggered=(),
            recommendations=(),
        ),
        limitations=LimitationsSummary(
            has_limitations=False,
            unexecuted_stages=(),
            failed_stages=(),
            inconclusive_checks_count=0,
            items=(),
        ),
        provenance=ProvenanceSummary(
            root_provenance_id="PROV-ROOT-EMPTY",
            source_uri=None,
            activity_name="minimal_pipeline",
            epistemic_level=EpistemicLevel.OBSERVED,
            parent_record_ids=(),
            stage_provenance_records={},
        ),
        all_issues=(),
    )


# ---------------------------------------------------------------------------
# Test Suite: Markdown Report Renderer
# ---------------------------------------------------------------------------


class TestMarkdownReportRenderer:
    """Focused test suite for MarkdownReportRenderer."""

    def test_markdown_render_complete_report(
        self,
        full_engineering_report: EngineeringReport,
    ) -> None:
        """Verify full Markdown report rendering containing all 15 stages and findings."""
        md = MarkdownReportRenderer.render(full_engineering_report)

        assert isinstance(md, str)
        assert len(md) > 500

        # Header & Metadata
        assert "# AERIS Flight Test 401 Engineering Report" in md
        assert f"**Report ID**: `{full_engineering_report.metadata.report_id}`" in md
        assert "**Analyst**: Maj. John Glenn" in md
        assert "Flight Phase" in md
        assert "Cruise Decel" in md

        # Executive Summary & Metrics
        assert "## 1. Executive Summary" in md
        assert "Executed / Total Stages" in md
        assert "Operational Narrative" in md

        # Stage Execution
        assert "## 2. Pipeline Stage Execution Status" in md
        assert "| Order | Pipeline Stage | Status | Executed |" in md
        assert "ingestion" in md
        assert "vehicle_evaluation" in md
        assert "recommendation" in md

        # Flight Context
        assert "## 3. Flight Context & Test Operational Boundaries" in md
        assert "AC-SYNTH-01" in md
        assert "CRUISE_CLEAN" in md

        # Data Quality
        assert "## 4. Data Quality & Telemetry Validation" in md
        assert "Validation Verdict" in md

        # Engineering Evaluations
        assert "## 5. Vehicle Engineering Evaluations" in md
        assert "### 5.1 Engineering Limit Evaluations `[LIMIT]`" in md
        assert "LIM-VNE-01" in md
        assert "airspeed_kts" in md
        assert "### 5.2 SOP Procedure Requirement Evaluations `[PROCEDURAL]`" in md
        assert "REQ-TARGET-ALT" in md

        # Investigation Intelligence & Recommendations
        assert "## 7. Investigation Intelligence & Recommendations" in md
        assert "FOR ENGINEERING REVIEW ONLY — NOT AN OPERATIONAL OR AIRWORTHINESS DIRECTIVE" in md
        assert "[RECOMMENDATION]" in md

        # Epistemic Level Labels
        assert "`[LIMIT]`" in md
        assert "`[PROCEDURAL]`" in md
        assert "`[CALCULATED]`" in md
        assert "`[RECOMMENDATION]`" in md

        # Limitations & Provenance
        assert "## 8. Analytical Limitations, Caveats & Data Gaps" in md
        assert "## 9. Provenance & Audit Trail" in md

    def test_markdown_render_empty_collections(
        self,
        empty_engineering_report: EngineeringReport,
    ) -> None:
        """Verify clean, crash-free rendering when report has empty collections."""
        md = MarkdownReportRenderer.render(empty_engineering_report)

        assert "# AERIS Empty Baseline Report" in md
        assert "*No vehicle configuration discrepancies detected.*" in md
        assert "*No engineering limit evaluations performed or attached.*" in md
        assert "*No procedural requirements evaluated or attached.*" in md
        assert "*No telemetry anomalies detected.*" in md
        assert "*No expected-vs-observed comparisons evaluated.*" in md
        assert "*No structured evidence items recorded.*" in md
        assert "*No contributor correlation rankings generated.*" in md
        assert "*No advisory recommendations generated.*" in md
        assert "*No operational limitations, skipped stages, or data gaps identified.*" in md

    def test_markdown_escaping_and_injection(self) -> None:
        """Verify that table cell pipes and newlines are escaped to preserve table structure."""
        now = datetime.datetime.now(datetime.UTC)
        malicious_report = EngineeringReport(
            metadata=ReportMetadata(
                report_id="REP-MALICIOUS-001",
                title="Security <script>alert(1)</script> Test",
                analyst_name="Attacker | Injected Column",
                generated_at=now,
                pipeline_run_id=None,
                source_file="telemetry|pipe.csv",
                source_file_hash=None,
                execution_duration_seconds=1.0,
                pipeline_version="1.0.0",
                custom_metadata={"Malicious|Key": "Value|With|Pipes\nAnd\nNewlines"},
            ),
            executive_summary=ExecutiveSummary(
                overall_status=ResultStatus.WARNING,
                total_stages=15,
                executed_stages_count=1,
                failed_stages_count=0,
                warning_stages_count=1,
                skipped_stages_count=14,
                total_issues_count=1,
                total_critical_or_error_count=0,
                total_limit_exceedances=1,
                total_procedure_violations=0,
                total_anomalies_detected=0,
                total_recommendations_generated=0,
                high_level_narrative="Narrative with | pipes | and unescaped markdown",
            ),
            stage_summaries=(
                StageExecutionSummary(
                    stage_name="stage|injected",
                    stage_order=1,
                    status=StageStatus.WARNING,
                    is_executed=True,
                    issue_count=1,
                    error_count=0,
                    warning_count=1,
                    notes="Note line 1\nNote line 2 | extra cell",
                ),
            ),
            data_quality=DataQualitySummary(
                is_valid=True,
                quality_score=0.9,
                total_rows_ingested=10,
                total_columns_ingested=2,
                columns_ingested=("col|1", "col|2"),
                columns_evaluated=("col|1",),
                violation_count=0,
                quality_metrics={},
                quality_summary_text="",
                normalization_applied=False,
                transformations_count=0,
                parameter_validation_passed=True,
                validated_parameters=("col|1",),
                failed_parameters=(),
                issues=(),
            ),
            flight_context=FlightContextSummary(
                vehicle_id="AC|INJECTED",
                profile_id="PROF|1",
                profile_version="1.0",
                configuration_id="ENV|1",
                test_card_id="CARD|1",
                test_point_id="PT|1",
                observation_id="OBS|1",
                associated_rows_count=10,
                configuration_discrepancies=("Discrepancy | item 1",),
            ),
            engineering_evaluation=EngineeringEvaluationSummary(
                is_evaluated=True,
                total_limits_evaluated=1,
                limit_exceedance_count=1,
                limit_inconclusive_count=0,
                total_procedures_evaluated=0,
                procedure_violation_count=0,
                procedure_inconclusive_count=0,
                limit_evaluations=(
                    LimitEvaluationItem(
                        evaluation_id="EVAL|001",
                        limit_id="LIM|VNE",
                        parameter_id="param|speed",
                        outcome="EXCEEDED",
                        limit_type="UPPER|BOUND",
                        severity="WARNING",
                        observed_value=300.0,
                        boundary_value=250.0,
                        unit="kts|extra",
                        margin=-50.0,
                        is_exceeded=True,
                        reason="Speed exceeded threshold\nInjected row | cell | extra",
                        epistemic_level=EpistemicLevel.LIMIT,
                    ),
                ),
                procedure_evaluations=(),
                calculations_performed=(),
            ),
            anomaly_and_comparison=AnomalyAndComparisonSummary(
                total_anomalies=0,
                high_severity_anomalies=0,
                medium_severity_anomalies=0,
                low_severity_anomalies=0,
                info_severity_anomalies=0,
                anomalies=(),
                total_comparisons=0,
                comparisons_within_tolerance=0,
                comparisons_out_of_tolerance=0,
                comparisons=(),
            ),
            investigation_intelligence=InvestigationIntelligenceSummary(
                evidence_count=0,
                evidence_items=(),
                contributor_rankings=(),
                recommendation_count=0,
                rules_evaluated=(),
                rules_triggered=(),
                recommendations=(),
            ),
            limitations=LimitationsSummary(
                has_limitations=True,
                unexecuted_stages=("stage|skipped",),
                failed_stages=(),
                inconclusive_checks_count=0,
                items=(
                    LimitationItem(
                        limitation_id="LIM|GAP",
                        category="DATA|GAP",
                        stage_name="stage|gap",
                        description="Description\nwith newlines | pipes",
                        impact="Impact\nwith newlines | pipes",
                    ),
                ),
            ),
            provenance=ProvenanceSummary(
                root_provenance_id="PROV|ROOT",
                source_uri="file:///path|with|pipe",
                activity_name="activity|calc",
                epistemic_level=EpistemicLevel.CALCULATED,
                parent_record_ids=("PARENT|1",),
                stage_provenance_records={"stage|ingest": "REC|001"},
            ),
            all_issues=(),
        )

        md = MarkdownReportRenderer.render(malicious_report)

        # Check that table cell pipes are escaped as \|
        assert "stage\\|injected" in md
        assert "param\\|speed" in md
        assert "LIM\\|VNE" in md
        # Check that newlines in table cells are replaced with <br>
        assert "<br>" in md

    def test_markdown_determinism_and_immutability(
        self,
        full_engineering_report: EngineeringReport,
    ) -> None:
        """Verify calling render multiple times is 100% deterministic and non-mutating."""
        md1 = MarkdownReportRenderer.render(full_engineering_report)
        md2 = MarkdownReportRenderer.render(full_engineering_report)
        md3 = render_markdown_report(full_engineering_report)

        assert md1 == md2
        assert md1 == md3
        # Assert model is frozen
        assert full_engineering_report.metadata.report_id is not None


# ---------------------------------------------------------------------------
# Test Suite: HTML Report Renderer
# ---------------------------------------------------------------------------


class TestHtmlReportRenderer:
    """Focused test suite for HtmlReportRenderer."""

    def test_html_render_complete_report(
        self,
        full_engineering_report: EngineeringReport,
    ) -> None:
        """Verify valid, complete HTML rendering with semantic sections and badges."""
        html_doc = HtmlReportRenderer.render(full_engineering_report)

        assert isinstance(html_doc, str)
        assert html_doc.startswith("<!DOCTYPE html>")
        assert '<html lang="en">' in html_doc
        assert "</html>" in html_doc

        # Head & Styling
        assert "<style>" in html_doc
        assert "@media print" in html_doc
        assert "break-inside: avoid" in html_doc
        assert "@page" in html_doc

        # Accessible status and epistemic badges
        assert any(b in html_doc for b in ("badge-warning", "badge-danger", "badge-success"))
        assert "epistemic-limit" in html_doc
        assert "epistemic-procedural" in html_doc
        assert "epistemic-recommendation" in html_doc

        # Sections
        assert '<section class="report-section" id="executive-summary">' in html_doc
        assert '<section class="report-section" id="stage-execution">' in html_doc
        assert '<section class="report-section" id="flight-context">' in html_doc
        assert '<section class="report-section" id="data-quality">' in html_doc
        assert '<section class="report-section" id="engineering-evaluations">' in html_doc
        assert '<section class="report-section" id="investigation-intelligence">' in html_doc
        assert '<section class="report-section" id="limitations">' in html_doc
        assert '<section class="report-section" id="provenance">' in html_doc

        # Mandatory Advisory Notice
        assert "advisory-notice-box" in html_doc
        advisory_msg = "FOR ENGINEERING REVIEW ONLY — NOT AN OPERATIONAL OR AIRWORTHINESS DIRECTIVE"
        assert advisory_msg in html_doc

        # Analyst and metadata
        assert "Maj. John Glenn" in html_doc
        assert "AC-SYNTH-01" in html_doc

    def test_html_render_empty_collections(
        self,
        empty_engineering_report: EngineeringReport,
    ) -> None:
        """Verify empty report renders gracefully with empty-state messages."""
        html_doc = HtmlReportRenderer.render(empty_engineering_report)

        assert "<!DOCTYPE html>" in html_doc
        assert "AERIS Empty Baseline Report" in html_doc
        assert "No engineering limit evaluations performed or attached." in html_doc
        assert "No procedural requirements evaluated or attached." in html_doc
        assert "No telemetry anomalies detected across evaluated parameters." in html_doc
        assert "No structured evidence items recorded." in html_doc
        assert "No advisory recommendations generated." in html_doc
        assert "No operational limitations, skipped stages, or data gaps identified." in html_doc

    def test_html_escaping_and_xss_prevention(self) -> None:
        """Verify that XSS injection payloads in report fields are safely escaped as inert text."""
        now = datetime.datetime.now(datetime.UTC)
        xss_report = EngineeringReport(
            metadata=ReportMetadata(
                report_id="<script>alert('xss-id')</script>",
                title="Title <img src=x onerror=alert('xss')>",
                analyst_name='Analyst " onclick="alert(1)"',
                generated_at=now,
                pipeline_run_id=None,
                source_file='javascript:alert("source")',
                source_file_hash=None,
                execution_duration_seconds=1.2,
                pipeline_version="1.0.0",
                custom_metadata={"<b>Key</b>": "<i>Value</i>"},
            ),
            executive_summary=ExecutiveSummary(
                overall_status=ResultStatus.ERROR,
                total_stages=15,
                executed_stages_count=1,
                failed_stages_count=1,
                warning_stages_count=0,
                skipped_stages_count=13,
                total_issues_count=1,
                total_critical_or_error_count=1,
                total_limit_exceedances=0,
                total_procedure_violations=0,
                total_anomalies_detected=0,
                total_recommendations_generated=0,
                high_level_narrative="Payload <script>evil()</script>",
            ),
            stage_summaries=(),
            data_quality=DataQualitySummary(
                is_valid=False,
                quality_score=0.0,
                total_rows_ingested=0,
                total_columns_ingested=0,
                columns_ingested=(),
                columns_evaluated=(),
                violation_count=1,
                quality_metrics={},
                quality_summary_text="<svg onload=alert(1)>",
                normalization_applied=False,
                transformations_count=0,
                parameter_validation_passed=False,
                validated_parameters=(),
                failed_parameters=(),
                issues=(
                    Issue(
                        issue_code="<ERR_XSS>",
                        severity=IssueSeverity.CRITICAL,
                        message="Payload <iframe src=javascript:alert(1)>",
                        field_name="<input>",
                        location=0,
                    ),
                ),
            ),
            flight_context=FlightContextSummary(
                vehicle_id="<script>",
                profile_id="<prof>",
                profile_version="1.0",
                configuration_id="<config>",
                test_card_id=None,
                test_point_id=None,
                observation_id=None,
                associated_rows_count=None,
                configuration_discrepancies=("<script>bad()</script>",),
            ),
            engineering_evaluation=EngineeringEvaluationSummary(
                is_evaluated=True,
                total_limits_evaluated=0,
                limit_exceedance_count=0,
                limit_inconclusive_count=0,
                total_procedures_evaluated=0,
                procedure_violation_count=0,
                procedure_inconclusive_count=0,
                limit_evaluations=(),
                procedure_evaluations=(),
                calculations_performed=(),
            ),
            anomaly_and_comparison=AnomalyAndComparisonSummary(
                total_anomalies=0,
                high_severity_anomalies=0,
                medium_severity_anomalies=0,
                low_severity_anomalies=0,
                info_severity_anomalies=0,
                anomalies=(),
                total_comparisons=0,
                comparisons_within_tolerance=0,
                comparisons_out_of_tolerance=0,
                comparisons=(),
            ),
            investigation_intelligence=InvestigationIntelligenceSummary(
                evidence_count=0,
                evidence_items=(),
                contributor_rankings=(),
                recommendation_count=0,
                rules_evaluated=(),
                rules_triggered=(),
                recommendations=(),
            ),
            limitations=LimitationsSummary(
                has_limitations=False,
                unexecuted_stages=(),
                failed_stages=(),
                inconclusive_checks_count=0,
                items=(),
            ),
            provenance=ProvenanceSummary(
                root_provenance_id="<ROOT>",
                source_uri=None,
                activity_name="<act>",
                epistemic_level=EpistemicLevel.OBSERVED,
                parent_record_ids=(),
                stage_provenance_records={},
            ),
            all_issues=(),
        )

        html_doc = HtmlReportRenderer.render(xss_report)

        # Assert dangerous tags are escaped
        assert "<script>alert('xss-id')</script>" not in html_doc
        assert "&lt;script&gt;alert(&#x27;xss-id&#x27;)&lt;/script&gt;" in html_doc
        assert "<img src=x onerror=alert('xss')>" not in html_doc
        assert "&lt;img src=x onerror=alert(&#x27;xss&#x27;)&gt;" in html_doc
        assert "<svg onload=alert(1)>" not in html_doc
        assert "&lt;svg onload=alert(1)&gt;" in html_doc
        assert "<iframe src=javascript:alert(1)>" not in html_doc
        assert "&lt;iframe src=javascript:alert(1)&gt;" in html_doc

    def test_html_offline_self_contained(
        self,
        full_engineering_report: EngineeringReport,
    ) -> None:
        """Verify that default HTML is strictly offline and self-contained."""
        html_doc = HtmlReportRenderer.render(full_engineering_report, figures=None)

        # No external script or stylesheet links
        assert '<script src="http' not in html_doc
        assert '<link rel="stylesheet" href="http' not in html_doc
        assert "fonts.googleapis.com" not in html_doc
        assert "cdn.plot.ly" not in html_doc

    def test_html_visualization_embedding(
        self,
        full_engineering_report: EngineeringReport,
    ) -> None:
        """Verify embedding Plotly figures into HTML reports."""
        # Create a simple valid Plotly figure
        fig = go.Figure(
            data=[go.Scatter(x=[1, 2, 3], y=[10, 20, 15], mode="lines", name="Airspeed")]
        )
        fig.update_layout(title="Test Figure")

        # 1. Embedding a single figure sequence with CDN
        html_cdn = HtmlReportRenderer.render(
            report=full_engineering_report,
            figures=[fig],
            include_plotlyjs="cdn",
        )
        assert '<section class="report-section" id="visualizations">' in html_cdn
        assert "https://cdn.plot.ly/plotly" in html_cdn
        assert "figure-container" in html_cdn

        # 2. Embedding figure with include_plotlyjs=False
        html_no_js = HtmlReportRenderer.render(
            report=full_engineering_report,
            figures=[fig],
            include_plotlyjs=False,
        )
        assert '<section class="report-section" id="visualizations">' in html_no_js
        assert "https://cdn.plot.ly/plotly" not in html_no_js
        assert "figure-container" in html_no_js

        # 3. Embedding a mapping (dictionary with custom captions)
        html_map = HtmlReportRenderer.render(
            report=full_engineering_report,
            figures={"Primary Airspeed Telemetry": fig},
            include_plotlyjs=False,
        )
        assert "Primary Airspeed Telemetry" in html_map

        # 4. Embedding TimeSeriesPlotResult object (has .figure property)
        plot_result = TimeSeriesPlotResult(
            status=ResultStatus.SUCCESS,
            parameter_id="airspeed_kts",
            figure=fig,
            point_count=3,
            non_null_count=3,
            unit="kts",
        )
        html_res = HtmlReportRenderer.render(
            report=full_engineering_report,
            figures={"Extracted Result": plot_result},
            include_plotlyjs=False,
        )
        assert "Extracted Result" in html_res
        assert "figure-container" in html_res

        # 5. Invalid non-figure object fallback
        html_inv = HtmlReportRenderer.render(
            report=full_engineering_report,
            figures=[object()],
            include_plotlyjs=False,
        )
        assert "Unable to render visualization: supplied object is not a Plotly figure" in html_inv

    def test_html_determinism_and_immutability(
        self,
        full_engineering_report: EngineeringReport,
    ) -> None:
        """Verify calling HTML render is 100% deterministic and non-mutating."""
        h1 = HtmlReportRenderer.render(full_engineering_report)
        h2 = HtmlReportRenderer.render(full_engineering_report)
        h3 = render_html_report(full_engineering_report)

        assert h1 == h2
        assert h1 == h3

    def test_convenience_functions(
        self,
        full_engineering_report: EngineeringReport,
    ) -> None:
        """Verify top-level convenience functions render_markdown_report and render_html_report."""
        md = render_markdown_report(full_engineering_report)
        html_doc = render_html_report(full_engineering_report)

        assert isinstance(md, str)
        assert isinstance(html_doc, str)
        assert "# AERIS Flight Test 401 Engineering Report" in md
        assert "<!DOCTYPE html>" in html_doc
