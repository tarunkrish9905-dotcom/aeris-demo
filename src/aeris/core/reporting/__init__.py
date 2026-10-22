"""
AERIS Core Reporting Subsystem.

Provides domain models and result extraction for post-flight engineering reports.
"""

from __future__ import annotations

from aeris.core.reporting.exporter import (
    ReportExporter,
    ReportExportFormat,
    ReportExportResult,
    export_engineering_report,
    export_pipeline_report,
    export_report,
)
from aeris.core.reporting.extractor import (
    ReportDataExtractor,
    extract_engineering_report,
)
from aeris.core.reporting.html_renderer import (
    HtmlReportRenderer,
    render_html_report,
)
from aeris.core.reporting.markdown_renderer import (
    MarkdownReportRenderer,
    render_markdown_report,
)
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

__all__ = [
    "AnomalyAndComparisonSummary",
    "AnomalyItem",
    "ContributorItem",
    "DataQualitySummary",
    "EngineeringEvaluationSummary",
    "EngineeringReport",
    "EvidenceSummaryItem",
    "ExecutiveSummary",
    "ExpectedObservedComparisonItem",
    "FlightContextSummary",
    "HtmlReportRenderer",
    "InvestigationIntelligenceSummary",
    "LimitEvaluationItem",
    "LimitationItem",
    "LimitationsSummary",
    "MarkdownReportRenderer",
    "ProcedureEvaluationItem",
    "ProvenanceSummary",
    "RecommendationItem",
    "ReportDataExtractor",
    "ReportExportFormat",
    "ReportExportResult",
    "ReportExporter",
    "ReportMetadata",
    "StageExecutionSummary",
    "StageStatus",
    "export_engineering_report",
    "export_pipeline_report",
    "export_report",
    "extract_engineering_report",
    "render_html_report",
    "render_markdown_report",
]
