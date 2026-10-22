"""
AERIS HTML Engineering Report Renderer.

Renders an immutable EngineeringReport domain model into a complete, self-contained,
accessible, and print-friendly HTML document.

Features:
- Pure Python string builder (no Jinja2, zero template injection vulnerabilities).
- Absolute XSS safety: all dynamic report values are escaped via html.escape(..., quote=True).
- Completely self-contained and offline: no external CDN fonts, scripts, or tracking by default.
- Full accessibility compliance: high contrast, semantic markup, status badges with explicit
  text and symbol markers (never color alone).
- Publication-grade @media print stylesheet optimized for browser Print-to-PDF.
- Bounded, safe Plotly visualization embedding without fabricating data.
- Non-mutating read operations on the input EngineeringReport.
"""

from __future__ import annotations

import datetime
import html
from collections.abc import Mapping, Sequence
from typing import Any

from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.reporting.models import EngineeringReport, StageStatus
from aeris.core.results.types import ResultStatus


def _e(val: Any) -> str:
    """Safely escape any report-derived value for HTML rendering."""
    if val is None:
        return "-"
    return html.escape(str(val), quote=True)


def _fmt_float(val: float | None, precision: int = 4) -> str:
    """Format an optional float safely."""
    if val is None:
        return "-"
    return f"{val:.{precision}f}"


def _fmt_datetime(dt: datetime.datetime | None) -> str:
    """Format an optional datetime safely to ISO 8601 string."""
    if dt is None:
        return "-"
    return html.escape(dt.isoformat(), quote=True)


def _render_status_badge(status: ResultStatus | StageStatus | str) -> str:
    """Render an accessible HTML status badge with symbol and text."""
    st_val = status.value if hasattr(status, "value") else str(status)
    st_upper = st_val.upper()

    if st_upper in ("SUCCESS", "PASSED", "WITHIN_LIMITS", "SATISFIED", "VALID"):
        css_cls = "badge-success"
        icon = "&#x2713;"  # Checkmark ✓
    elif st_upper in ("WARNING", "CAUTION", "MARGINAL"):
        css_cls = "badge-warning"
        icon = "&#x26A0;"  # Warning ⚠
    elif st_upper in ("ERROR", "CRITICAL", "FAILED", "VIOLATED", "EXCEEDED", "INVALID"):
        css_cls = "badge-danger"
        icon = "&#x2715;"  # Cross ✕
    elif st_upper in ("PARTIAL",):
        css_cls = "badge-partial"
        icon = "&#x25D0;"  # Circle half ◐
    elif st_upper in ("INCONCLUSIVE", "UNKNOWN"):
        css_cls = "badge-inconclusive"
        icon = "&#x003F;"  # Question mark ?
    elif st_upper in ("UPSTREAM_ABORTED", "ABORTED"):
        css_cls = "badge-danger"
        icon = "&#x25A0;"  # Square ■
    elif st_upper in ("SKIPPED", "NOT_AVAILABLE", "NOT_REQUESTED"):
        css_cls = "badge-neutral"
        icon = "&#x2013;"  # En-dash
    else:
        css_cls = "badge-neutral"
        icon = "&#x2022;"  # Bullet •

    return (
        f'<span class="badge {css_cls}">'
        f'<span class="badge-icon" aria-hidden="true">{icon}</span> '
        f'<span class="badge-text">{html.escape(st_upper)}</span>'
        f"</span>"
    )


def _render_epistemic_badge(level: EpistemicLevel | str) -> str:
    """Render an accessible HTML epistemic badge."""
    lvl_val = level.value if hasattr(level, "value") else str(level)
    lvl_upper = lvl_val.upper()
    css_cls = f"epistemic-{lvl_val.lower()}"

    return (
        f'<span class="badge badge-epistemic {css_cls}" '
        f'title="Epistemic Classification: {html.escape(lvl_upper)}">'
        f"[{html.escape(lvl_upper)}]"
        f"</span>"
    )


def _meta_item(
    label: str,
    value: str,
    is_mono: bool = False,
    title: str | None = None,
) -> str:
    """Render a metadata grid cell."""
    cls = "meta-value mono" if is_mono else "meta-value"
    title_attr = f' title="{title}"' if title else ""
    return (
        '      <div class="meta-item">'
        f'<span class="meta-label">{label}</span>'
        f'<span class="{cls}"{title_attr}>{value}</span>'
        "</div>"
    )


def _metric_card(label: str, value: str, desc: str = "") -> str:
    """Render a dashboard summary metric card."""
    desc_html = f'<span class="metric-card-desc">{desc}</span>' if desc else ""
    return (
        '      <div class="metric-card">'
        f'<span class="metric-card-label">{label}</span>'
        f'<span class="metric-card-value">{value}</span>'
        f"{desc_html}</div>"
    )


# ---------------------------------------------------------------------------
# Self-Contained Modern CSS Stylesheet
# ---------------------------------------------------------------------------

_REPORT_CSS = """
:root {
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue",
    Arial, sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono",
    "Courier New", monospace;

  --color-bg-body: #f8fafc;
  --color-bg-card: #ffffff;
  --color-text-main: #0f172a;
  --color-text-muted: #475569;
  --color-text-light: #64748b;
  --color-border: #e2e8f0;
  --color-border-dark: #cbd5e1;

  --color-success-bg: #dcfce7;
  --color-success-text: #166534;
  --color-success-border: #86efac;

  --color-warning-bg: #fef3c7;
  --color-warning-text: #92400e;
  --color-warning-border: #fcd34d;

  --color-danger-bg: #fee2e2;
  --color-danger-text: #991b1b;
  --color-danger-border: #fca5a5;

  --color-partial-bg: #ede9fe;
  --color-partial-text: #5b21b6;
  --color-partial-border: #c4b5fd;

  --color-neutral-bg: #f1f5f9;
  --color-neutral-text: #334155;
  --color-neutral-border: #cbd5e1;

  --color-primary: #0284c7;
  --color-primary-dark: #0369a1;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: var(--font-sans);
  background-color: var(--color-bg-body);
  color: var(--color-text-main);
  line-height: 1.55;
  font-size: 14px;
  padding: 24px 16px;
}

.report-container {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* Header & Banner */
header.report-header {
  background: var(--color-bg-card);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 28px 32px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.header-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 20px;
}

h1.report-title {
  font-size: 24px;
  font-weight: 700;
  color: var(--color-text-main);
  letter-spacing: -0.02em;
}

.report-meta-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px 20px;
  border-top: 1px solid var(--color-border);
  padding-top: 16px;
  font-size: 13px;
}

.meta-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.meta-label {
  color: var(--color-text-light);
  font-size: 11px;
  text-transform: uppercase;
  font-weight: 600;
  letter-spacing: 0.05em;
}

.meta-value {
  color: var(--color-text-main);
  font-weight: 500;
}

.meta-value.mono {
  font-family: var(--font-mono);
  font-size: 12px;
}

/* Section Cards */
section.report-section {
  background: var(--color-bg-card);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 24px 28px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}

h2.section-heading {
  font-size: 18px;
  font-weight: 600;
  color: var(--color-text-main);
  margin-bottom: 16px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--color-border);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

h3.sub-heading {
  font-size: 15px;
  font-weight: 600;
  color: var(--color-text-muted);
  margin: 20px 0 10px 0;
}

/* Metric Summary Dashboard Grid */
.dashboard-metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 14px;
  margin-bottom: 20px;
}

.metric-card {
  background: var(--color-bg-body);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.metric-card-label {
  font-size: 11px;
  color: var(--color-text-light);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.metric-card-value {
  font-size: 20px;
  font-weight: 700;
  color: var(--color-text-main);
}

.metric-card-desc {
  font-size: 11px;
  color: var(--color-text-muted);
}

/* Callout Boxes */
.callout {
  padding: 14px 18px;
  border-radius: 6px;
  margin-bottom: 18px;
  font-size: 13px;
  border-left: 4px solid;
}

.callout-success {
  background: var(--color-success-bg);
  color: var(--color-success-text);
  border-color: var(--color-success-border);
}

.callout-warning {
  background: var(--color-warning-bg);
  color: var(--color-warning-text);
  border-color: var(--color-warning-border);
}

.callout-danger {
  background: var(--color-danger-bg);
  color: var(--color-danger-text);
  border-color: var(--color-danger-border);
}

.callout-partial {
  background: var(--color-partial-bg);
  color: var(--color-partial-text);
  border-color: var(--color-partial-border);
}

.callout-info {
  background: var(--color-neutral-bg);
  color: var(--color-neutral-text);
  border-color: var(--color-neutral-border);
}

.advisory-notice-box {
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-left: 4px solid #f59e0b;
  padding: 12px 16px;
  border-radius: 6px;
  margin: 12px 0;
  font-size: 12px;
  font-weight: 600;
  color: #92400e;
}

/* Tables */
.table-responsive {
  width: 100%;
  overflow-x: auto;
  margin-bottom: 16px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
}

table.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  text-align: left;
}

table.data-table th {
  background: #f1f5f9;
  color: var(--color-text-muted);
  font-weight: 600;
  padding: 10px 12px;
  border-bottom: 1px solid var(--color-border-dark);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  white-space: nowrap;
}

table.data-table td {
  padding: 9px 12px;
  border-bottom: 1px solid var(--color-border);
  vertical-align: top;
}

table.data-table tbody tr:last-child td {
  border-bottom: none;
}

table.data-table tbody tr:hover {
  background-color: #f8fafc;
}

.mono {
  font-family: var(--font-mono);
}

.text-center {
  text-align: center;
}

.text-right {
  text-align: right;
}

/* Badges */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 7px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  line-height: 1.2;
  white-space: nowrap;
  border: 1px solid transparent;
}

.badge-success {
  background: var(--color-success-bg);
  color: var(--color-success-text);
  border-color: var(--color-success-border);
}

.badge-warning {
  background: var(--color-warning-bg);
  color: var(--color-warning-text);
  border-color: var(--color-warning-border);
}

.badge-danger {
  background: var(--color-danger-bg);
  color: var(--color-danger-text);
  border-color: var(--color-danger-border);
}

.badge-partial {
  background: var(--color-partial-bg);
  color: var(--color-partial-text);
  border-color: var(--color-partial-border);
}

.badge-inconclusive {
  background: #f1f5f9;
  color: #475569;
  border-color: #cbd5e1;
}

.badge-neutral {
  background: var(--color-neutral-bg);
  color: var(--color-neutral-text);
  border-color: var(--color-neutral-border);
}

.badge-epistemic {
  font-family: var(--font-mono);
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 3px;
  font-weight: 700;
}

.epistemic-observed {
  background: #e0f2fe;
  color: #0369a1;
  border-color: #7dd3fc;
}

.epistemic-calculated {
  background: #ccfbf1;
  color: #0f766e;
  border-color: #5eead4;
}

.epistemic-limit {
  background: #ede9fe;
  color: #6d28d9;
  border-color: #c4b5fd;
}

.epistemic-procedural {
  background: #f1f5f9;
  color: #334155;
  border-color: #cbd5e1;
}

.epistemic-hypothesis {
  background: #fef3c7;
  color: #b45309;
  border-color: #fcd34d;
}

.epistemic-recommendation {
  background: #fce7f3;
  color: #be185d;
  border-color: #f472b6;
}

/* Recommendations Card */
.recommendation-card {
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 16px 20px;
  margin-bottom: 16px;
  background: #fafafa;
}

.recommendation-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 10px;
}

.recommendation-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--color-text-main);
}

.recommendation-body p {
  margin-bottom: 8px;
  font-size: 13px;
}

.recommendation-metadata {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  font-size: 12px;
  color: var(--color-text-muted);
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px dashed var(--color-border);
}

/* Figures Integration */
.figure-container {
  margin: 20px 0;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  background: var(--color-bg-card);
  padding: 16px;
  overflow: hidden;
}

.figure-caption {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-muted);
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.figure-plot-wrapper {
  width: 100%;
  min-height: 400px;
}

/* Empty State */
.empty-state {
  padding: 16px;
  color: var(--color-text-light);
  font-style: italic;
  font-size: 13px;
  text-align: center;
}

/* Footer */
footer.report-footer {
  text-align: center;
  padding: 24px 0;
  color: var(--color-text-light);
  font-size: 12px;
  border-top: 1px solid var(--color-border);
}

/* Print Stylesheet */
@media print {
  @page {
    size: letter;
    margin: 12mm 15mm 15mm 15mm;
  }

  body {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-size: 9.5pt;
    line-height: 1.4;
    padding: 0 !important;
  }

  .report-container {
    max-width: 100% !important;
    gap: 14pt;
  }

  header.report-header,
  section.report-section,
  .metric-card,
  .recommendation-card,
  .table-responsive,
  .figure-container {
    box-shadow: none !important;
    border: 1px solid #cccccc !important;
    break-inside: avoid !important;
    page-break-inside: avoid !important;
  }

  h1.report-title,
  h2.section-heading,
  h3.sub-heading {
    break-after: avoid !important;
    page-break-after: avoid !important;
    color: #000000 !important;
  }

  .badge {
    border: 1px solid #888888 !important;
    background: transparent !important;
    color: #000000 !important;
    font-weight: 600 !important;
  }

  .no-print {
    display: none !important;
  }

  .page-break {
    page-break-before: always;
    break-before: always;
  }
}
"""


class HtmlReportRenderer:
    """Deterministic, pure-Python HTML renderer for EngineeringReport domain models."""

    @staticmethod
    def render(
        report: EngineeringReport,
        figures: Sequence[Any] | Mapping[str, Any] | None = None,
        include_plotlyjs: bool | str = False,
    ) -> str:
        """
        Render an EngineeringReport domain model as a complete, self-contained HTML document.

        Parameters
        ----------
        report : EngineeringReport
            The authoritative, immutable post-flight engineering report.
        figures : Sequence[Any] | Mapping[str, Any] | None
            Optional sequence or dictionary of Plotly figures or TimeSeriesPlotResult
            objects to embed within the report visualizer section.
        include_plotlyjs : bool | str
            Option passed to Plotly when figures are included:
            - False: Do not include plotly.js (preserves lightweight offline HTML).
            - "inline": Embed minified plotly.js script inline once (offline self-contained).
            - "cdn": Link plotly.js from official Plotly CDN.

        Returns
        -------
        str
            A complete, self-contained HTML document string.
        """
        html_parts: list[str] = []

        # Document Header
        html_parts.append("<!DOCTYPE html>")
        html_parts.append('<html lang="en">')
        html_parts.append("<head>")
        html_parts.append('  <meta charset="UTF-8">')
        html_parts.append(
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">'
        )
        html_parts.append(f"  <title>{_e(report.metadata.title)}</title>")
        html_parts.append(f"  <style>\n{_REPORT_CSS}\n  </style>")

        # Optional Plotly script tag if CDN mode explicitly selected with figures
        if figures and include_plotlyjs == "cdn":
            html_parts.append('  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>')

        html_parts.append("</head>")
        html_parts.append("<body>")
        html_parts.append('<div class="report-container">')

        # -------------------------------------------------------------------
        # Header & Metadata Block
        # -------------------------------------------------------------------
        html_parts.append('  <header class="report-header">')
        html_parts.append('    <div class="header-top">')
        html_parts.append("      <div>")
        html_parts.append(f'        <h1 class="report-title">{_e(report.metadata.title)}</h1>')
        html_parts.append(
            '        <div style="margin-top: 4px; color: var(--color-text-light); '
            'font-size: 13px;">AERIS Post-Flight Telemetry & Vehicle Engineering Analysis</div>'
        )
        html_parts.append("      </div>")
        st_badge = _render_status_badge(report.executive_summary.overall_status)
        html_parts.append(f"      <div>{st_badge}</div>")
        html_parts.append("    </div>")

        html_parts.append('    <div class="report-meta-grid">')
        html_parts.append(_meta_item("Report Identifier", _e(report.metadata.report_id), True))
        html_parts.append(
            _meta_item("Generated (UTC)", _fmt_datetime(report.metadata.generated_at), True)
        )
        html_parts.append(
            _meta_item("Lead Analyst", _e(report.metadata.analyst_name or "Unassigned"))
        )
        html_parts.append(
            _meta_item("Pipeline Run ID", _e(report.metadata.pipeline_run_id or "N/A"), True)
        )
        html_parts.append(
            _meta_item("Telemetry Source", _e(report.metadata.source_file or "N/A"), True)
        )

        hash_val = report.metadata.source_file_hash
        short_hash = (
            hash_val[:16] + "..." if hash_val and len(hash_val) > 16 else (hash_val or "N/A")
        )
        html_parts.append(_meta_item("Dataset SHA-256", _e(short_hash), True, title=_e(hash_val)))

        if report.metadata.execution_duration_seconds is not None:
            dur_str = f"{report.metadata.execution_duration_seconds:.3f} s"
        else:
            dur_str = "N/A"
        html_parts.append(_meta_item("Execution Duration", dur_str))
        html_parts.append(
            _meta_item("Pipeline Version", f"v{_e(report.metadata.pipeline_version)}", True)
        )
        html_parts.append("    </div>")

        if report.metadata.custom_metadata:
            html_parts.append(
                '    <div style="margin-top: 14px; padding-top: 12px; '
                "border-top: 1px dashed var(--color-border); font-size: 12px; "
                'color: var(--color-text-muted);">'
            )
            html_parts.append("      <strong>Custom Metadata:</strong> ")
            meta_items = [
                f'<span class="mono">{_e(k)}</span>: {_e(v)}'
                for k, v in sorted(report.metadata.custom_metadata.items())
            ]
            html_parts.append(" &bull; ".join(meta_items))
            html_parts.append("    </div>")

        html_parts.append("  </header>")

        # -------------------------------------------------------------------
        # 1. Executive Summary
        # -------------------------------------------------------------------
        html_parts.append('  <section class="report-section" id="executive-summary">')
        html_parts.append('    <h2 class="section-heading">1. Executive Summary</h2>')

        # Status callout banner
        st = report.executive_summary.overall_status
        if st == ResultStatus.SUCCESS:
            callout_cls = "callout-success"
            verdict_text = (
                "Analysis completed cleanly. All evaluated parameters remained within "
                "allowable operational envelopes."
            )
        elif st == ResultStatus.WARNING:
            callout_cls = "callout-warning"
            verdict_text = (
                "Analytical advisories, parameter anomalies, or procedural warnings were flagged."
            )
        elif st == ResultStatus.ERROR:
            callout_cls = "callout-danger"
            verdict_text = (
                "Critical defects, validation failures, or safety boundary exceedances detected."
            )
        elif st == ResultStatus.PARTIAL:
            callout_cls = "callout-partial"
            verdict_text = (
                "Analysis completed with partial coverage or degraded telemetry datasets."
            )
        else:
            callout_cls = "callout-info"
            verdict_text = f"Execution status: {st.value}"

        html_parts.append(f'    <div class="callout {callout_cls}">')
        html_parts.append(
            f"      <strong>Pipeline Verdict: {html.escape(st.value)}</strong> "
            f"&mdash; {verdict_text}"
        )
        html_parts.append("    </div>")

        # Metric dashboard grid
        es = report.executive_summary
        html_parts.append('    <div class="dashboard-metrics-grid">')
        html_parts.append(
            _metric_card(
                "Executed Stages",
                f"{es.executed_stages_count} / {es.total_stages}",
                f"Failed: {es.failed_stages_count}",
            )
        )
        html_parts.append(
            _metric_card(
                "Logged Issues",
                str(es.total_issues_count),
                f"Critical: {es.total_critical_or_error_count}",
            )
        )
        html_parts.append(
            _metric_card(
                "Limit Exceedances",
                str(es.total_limit_exceedances),
                _render_epistemic_badge(EpistemicLevel.LIMIT),
            )
        )
        html_parts.append(
            _metric_card(
                "Procedure Violations",
                str(es.total_procedure_violations),
                _render_epistemic_badge(EpistemicLevel.PROCEDURAL),
            )
        )
        html_parts.append(
            _metric_card(
                "Telemetry Anomalies",
                str(es.total_anomalies_detected),
                _render_epistemic_badge(EpistemicLevel.CALCULATED),
            )
        )
        html_parts.append(
            _metric_card(
                "Recommendations",
                str(es.total_recommendations_generated),
                _render_epistemic_badge(EpistemicLevel.RECOMMENDATION),
            )
        )
        html_parts.append("    </div>")

        html_parts.append('    <h3 class="sub-heading">Operational Narrative</h3>')
        narrative_html = (
            f'    <p style="font-size: 13.5px; color: var(--color-text-main); '
            f'line-height: 1.6;">{_e(es.high_level_narrative)}</p>'
        )
        html_parts.append(narrative_html)
        html_parts.append("  </section>")

        # -------------------------------------------------------------------
        # 2. Pipeline Stage Execution Status
        # -------------------------------------------------------------------
        html_parts.append('  <section class="report-section" id="stage-execution">')
        html_parts.append('    <h2 class="section-heading">2. Pipeline Stage Execution Status</h2>')
        if report.stage_summaries:
            html_parts.append('    <div class="table-responsive">')
            html_parts.append('      <table class="data-table">')
            html_parts.append("        <thead>")
            html_parts.append("          <tr>")
            html_parts.append('            <th class="text-center" style="width: 50px;">#</th>')
            html_parts.append("            <th>Stage Name</th>")
            html_parts.append('            <th class="text-center">Status</th>')
            html_parts.append('            <th class="text-center">Ran</th>')
            html_parts.append('            <th class="text-center">Issues</th>')
            html_parts.append('            <th class="text-center">Errors</th>')
            html_parts.append('            <th class="text-center">Warnings</th>')
            html_parts.append("            <th>Operational Notes</th>")
            html_parts.append("          </tr>")
            html_parts.append("        </thead>")
            html_parts.append("        <tbody>")
            for st_sum in sorted(report.stage_summaries, key=lambda s: s.stage_order):
                if st_sum.is_executed:
                    ran_str = (
                        '<span style="color: var(--color-success-text); '
                        'font-weight: 600;">Yes</span>'
                    )
                else:
                    ran_str = '<span style="color: var(--color-text-light);">No</span>'
                notes_str = _e(st_sum.notes) if st_sum.notes else "-"
                html_parts.append("          <tr>")
                html_parts.append(
                    f'            <td class="text-center mono">{st_sum.stage_order}</td>'
                )
                html_parts.append(
                    f'            <td class="mono" style="font-weight: 600;">'
                    f"{_e(st_sum.stage_name)}</td>"
                )
                st_b = _render_status_badge(st_sum.status)
                html_parts.append(f'            <td class="text-center">{st_b}</td>')
                html_parts.append(f'            <td class="text-center">{ran_str}</td>')
                html_parts.append(
                    f'            <td class="text-center mono">{st_sum.issue_count}</td>'
                )
                html_parts.append(
                    f'            <td class="text-center mono">{st_sum.error_count}</td>'
                )
                html_parts.append(
                    f'            <td class="text-center mono">{st_sum.warning_count}</td>'
                )
                html_parts.append(f"            <td>{notes_str}</td>")
                html_parts.append("          </tr>")
            html_parts.append("        </tbody>")
            html_parts.append("      </table>")
            html_parts.append("    </div>")
        else:
            html_parts.append(
                '    <div class="empty-state">No pipeline stage execution records available.</div>'
            )
        html_parts.append("  </section>")

        # -------------------------------------------------------------------
        # 3. Flight Context & Vehicle Configuration
        # -------------------------------------------------------------------
        html_parts.append('  <section class="report-section" id="flight-context">')
        html_parts.append(
            '    <h2 class="section-heading">3. Flight Context & Test Operational Boundaries</h2>'
        )
        ctx = report.flight_context
        html_parts.append('    <div class="table-responsive">')
        html_parts.append('      <table class="data-table">')
        html_parts.append(
            "        <thead><tr><th>Parameter</th><th>Value</th>"
            "<th>Context Description</th></tr></thead>"
        )
        html_parts.append("        <tbody>")
        html_parts.append(
            "          <tr><td><strong>Vehicle Identifier</strong></td>"
            f'<td class="mono">{_e(ctx.vehicle_id)}</td>'
            "<td>Active airframe under post-flight evaluation</td></tr>"
        )
        prof_info = f"{ctx.profile_id} (v{ctx.profile_version})" if ctx.profile_id else "-"
        html_parts.append(
            "          <tr><td><strong>Vehicle Profile</strong></td>"
            f'<td class="mono">{_e(prof_info)}</td>'
            "<td>Aerodynamic & subsystem physical parameters</td></tr>"
        )
        html_parts.append(
            "          <tr><td><strong>Configuration Envelope</strong></td>"
            f'<td class="mono">{_e(ctx.configuration_id)}</td>'
            "<td>Selected configuration operational corridor</td></tr>"
        )
        html_parts.append(
            "          <tr><td><strong>Flight Test Card</strong></td>"
            f'<td class="mono">{_e(ctx.test_card_id)}</td>'
            "<td>Pre-briefed test plan card reference</td></tr>"
        )
        html_parts.append(
            "          <tr><td><strong>Test Point Identifier</strong></td>"
            f'<td class="mono">{_e(ctx.test_point_id)}</td>'
            "<td>Target maneuver operational context</td></tr>"
        )
        html_parts.append(
            "          <tr><td><strong>Observation Identifier</strong></td>"
            f'<td class="mono">{_e(ctx.observation_id)}</td>'
            "<td>Temporal slice observation window</td></tr>"
        )
        rows_str = str(ctx.associated_rows_count) if ctx.associated_rows_count is not None else "-"
        html_parts.append(
            "          <tr><td><strong>Associated Data Rows</strong></td>"
            f'<td class="mono">{rows_str}</td>'
            "<td>Telemetry samples correlated to test interval</td></tr>"
        )
        html_parts.append("        </tbody>")
        html_parts.append("      </table>")
        html_parts.append("    </div>")

        if ctx.configuration_discrepancies:
            html_parts.append('    <div class="callout callout-warning">')
            html_parts.append("      <strong>Configuration Discrepancies Detected:</strong>")
            html_parts.append('      <ul style="margin-left: 20px; margin-top: 6px;">')
            for disc in ctx.configuration_discrepancies:
                html_parts.append(f"        <li>{_e(disc)}</li>")
            html_parts.append("      </ul>")
            html_parts.append("    </div>")
        html_parts.append("  </section>")

        # -------------------------------------------------------------------
        # 4. Data Quality & Telemetry Validation
        # -------------------------------------------------------------------
        html_parts.append('  <section class="report-section" id="data-quality">')
        html_parts.append(
            '    <h2 class="section-heading">4. Data Quality & Telemetry Validation</h2>'
        )
        dq = report.data_quality
        if dq.is_valid is True:
            val_status = "VALID"
        elif dq.is_valid is False:
            val_status = "INVALID"
        else:
            val_status = "N/A"
        score_str = f"{dq.quality_score * 100.0:.1f}%" if dq.quality_score is not None else "N/A"
        if dq.parameter_validation_passed is True:
            param_val_status = "PASSED"
        elif dq.parameter_validation_passed is False:
            param_val_status = "FAILED"
        else:
            param_val_status = "N/A"

        html_parts.append('    <div class="dashboard-metrics-grid">')
        html_parts.append(
            _metric_card(
                "Validation Status",
                _render_status_badge(val_status),
                "Schema conformance",
            )
        )
        html_parts.append(_metric_card("Quality Score", score_str, "Sensor integrity"))
        html_parts.append(
            _metric_card(
                "Raw Rows",
                f"{dq.total_rows_ingested:,}",
                f"Columns: {dq.total_columns_ingested}",
            )
        )
        html_parts.append(
            _metric_card("Schema Violations", str(dq.violation_count), "Format defects")
        )
        norm_desc = "Applied" if dq.normalization_applied else "Not run"
        html_parts.append(_metric_card("Normalization", str(dq.transformations_count), norm_desc))
        html_parts.append(
            _metric_card(
                "Param Validation",
                _render_status_badge(param_val_status),
                "Registry boundaries",
            )
        )
        html_parts.append("    </div>")

        if dq.quality_summary_text:
            html_parts.append(
                f'    <p style="margin-bottom: 16px; font-size: 13px; '
                f'color: var(--color-text-muted);">{_e(dq.quality_summary_text)}</p>'
            )

        if dq.issues:
            html_parts.append('    <h3 class="sub-heading">Data Quality & Validation Issues</h3>')
            html_parts.append('    <div class="table-responsive">')
            html_parts.append('      <table class="data-table">')
            html_parts.append(
                "        <thead><tr><th>Severity</th><th>Code</th><th>Field</th>"
                "<th>Location</th><th>Message</th></tr></thead>"
            )
            html_parts.append("        <tbody>")
            for iss in dq.issues:
                loc_str = str(iss.location) if iss.location is not None else "-"
                html_parts.append("          <tr>")
                iss_b = _render_status_badge(iss.severity.value)
                html_parts.append(f"            <td>{iss_b}</td>")
                html_parts.append(f'            <td class="mono">{_e(iss.issue_code)}</td>')
                html_parts.append(f'            <td class="mono">{_e(iss.field_name)}</td>')
                html_parts.append(f'            <td class="mono">{loc_str}</td>')
                html_parts.append(f"            <td>{_e(iss.message)}</td>")
                html_parts.append("          </tr>")
            html_parts.append("        </tbody>")
            html_parts.append("      </table>")
            html_parts.append("    </div>")
        html_parts.append("  </section>")

        # -------------------------------------------------------------------
        # 5. Vehicle Engineering Evaluations
        # -------------------------------------------------------------------
        html_parts.append('  <section class="report-section" id="engineering-evaluations">')
        html_parts.append('    <h2 class="section-heading">5. Vehicle Engineering Evaluations</h2>')
        ee = report.engineering_evaluation

        html_parts.append('    <div class="dashboard-metrics-grid">')
        html_parts.append(
            _metric_card(
                "Limits Evaluated",
                str(ee.total_limits_evaluated),
                _render_epistemic_badge(EpistemicLevel.LIMIT),
            )
        )
        if ee.limit_exceedance_count > 0:
            exceed_style = ' style="color: var(--color-danger-text);"'
        else:
            exceed_style = ""
        html_parts.append(
            '      <div class="metric-card">'
            '<span class="metric-card-label">Limit Exceedances</span>'
            f'<span class="metric-card-value mono"{exceed_style}>{ee.limit_exceedance_count}</span>'
            '<span class="metric-card-desc">Safety breaches</span></div>'
        )
        html_parts.append(
            _metric_card("Inconclusive Limits", str(ee.limit_inconclusive_count), "Missing data")
        )
        html_parts.append(
            _metric_card(
                "Procedures Checked",
                str(ee.total_procedures_evaluated),
                _render_epistemic_badge(EpistemicLevel.PROCEDURAL),
            )
        )
        if ee.procedure_violation_count > 0:
            viol_style = ' style="color: var(--color-danger-text);"'
        else:
            viol_style = ""
        html_parts.append(
            '      <div class="metric-card">'
            '<span class="metric-card-label">Procedure Violations</span>'
            f'<span class="metric-card-value mono"{viol_style}>'
            f"{ee.procedure_violation_count}</span>"
            '<span class="metric-card-desc">SOP non-compliance</span></div>'
        )
        html_parts.append(
            _metric_card("Inconclusive SOP", str(ee.procedure_inconclusive_count), "Unverified")
        )
        html_parts.append("    </div>")

        # 5.1 Limits Table
        lim_badge = _render_epistemic_badge(EpistemicLevel.LIMIT)
        html_parts.append(
            f'    <h3 class="sub-heading">5.1 Engineering Limit Evaluations {lim_badge}</h3>'
        )
        if ee.limit_evaluations:
            html_parts.append('    <div class="table-responsive">')
            html_parts.append('      <table class="data-table">')
            html_parts.append(
                "        <thead><tr><th>Limit ID</th><th>Parameter</th><th>Type</th>"
                "<th>Severity</th><th>Boundary</th><th>Observed</th><th>Margin</th>"
                "<th>Outcome</th><th>Reason</th></tr></thead>"
            )
            html_parts.append("        <tbody>")
            for lim in ee.limit_evaluations:
                unit_str = f" {html.escape(lim.unit)}" if lim.unit else ""
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
                html_parts.append("          <tr>")
                html_parts.append(
                    f'            <td class="mono"><strong>{_e(lim.limit_id)}</strong></td>'
                )
                html_parts.append(f'            <td class="mono">{_e(lim.parameter_id)}</td>')
                html_parts.append(f"            <td>{_e(lim.limit_type)}</td>")
                html_parts.append(f"            <td>{_e(lim.severity)}</td>")
                html_parts.append(f'            <td class="mono">{bound_str}</td>')
                html_parts.append(f'            <td class="mono">{obs_str}</td>')
                html_parts.append(f'            <td class="mono">{margin_str}</td>')
                html_parts.append(f"            <td>{_render_status_badge(lim.outcome)}</td>")
                html_parts.append(f"            <td>{_e(lim.reason)}</td>")
                html_parts.append("          </tr>")
            html_parts.append("        </tbody>")
            html_parts.append("      </table>")
            html_parts.append("    </div>")
        else:
            html_parts.append(
                '    <div class="empty-state">'
                "No engineering limit evaluations performed or attached.</div>"
            )

        # 5.2 Procedure Evaluations Table
        proc_badge = _render_epistemic_badge(EpistemicLevel.PROCEDURAL)
        html_parts.append(
            f'    <h3 class="sub-heading">5.2 SOP Procedure Requirement Evaluations '
            f"{proc_badge}</h3>"
        )
        if ee.procedure_evaluations:
            html_parts.append('    <div class="table-responsive">')
            html_parts.append('      <table class="data-table">')
            html_parts.append(
                "        <thead><tr><th>Requirement ID</th><th>SOP Doc</th><th>Category</th>"
                "<th>Target</th><th>Observed</th><th>Margin</th><th>Verification</th>"
                "<th>Reason</th></tr></thead>"
            )
            html_parts.append("        <tbody>")
            for pr in ee.procedure_evaluations:
                unit_str = f" {html.escape(pr.unit)}" if pr.unit else ""
                if pr.target_value is not None:
                    target_str = f"{_fmt_float(pr.target_value)}{unit_str}"
                else:
                    target_str = "-"
                if pr.is_qualitative:
                    obs_str = f'<span class="mono">{_e(pr.qualitative_state)}</span>'
                elif pr.observed_value is not None:
                    obs_str = f'<span class="mono">{_fmt_float(pr.observed_value)}{unit_str}</span>'
                else:
                    obs_str = "-"
                margin_str = f"{_fmt_float(pr.margin)}{unit_str}" if pr.margin is not None else "-"
                if pr.is_satisfied is True:
                    sat_status = "PASSED"
                elif pr.is_satisfied is False:
                    sat_status = "VIOLATED"
                else:
                    sat_status = "INCONCLUSIVE"

                html_parts.append("          <tr>")
                html_parts.append(
                    f'            <td class="mono"><strong>{_e(pr.requirement_id)}</strong></td>'
                )
                html_parts.append(f'            <td class="mono">{_e(pr.document_id)}</td>')
                html_parts.append(f"            <td>{_e(pr.category)}</td>")
                html_parts.append(f'            <td class="mono">{target_str}</td>')
                html_parts.append(f"            <td>{obs_str}</td>")
                html_parts.append(f'            <td class="mono">{margin_str}</td>')
                html_parts.append(f"            <td>{_render_status_badge(sat_status)}</td>")
                html_parts.append(f"            <td>{_e(pr.reason)}</td>")
                html_parts.append("          </tr>")
            html_parts.append("        </tbody>")
            html_parts.append("      </table>")
            html_parts.append("    </div>")
        else:
            html_parts.append(
                '    <div class="empty-state">'
                "No procedural requirements evaluated or attached.</div>"
            )

        # 5.3 Derived Engineering Calculations
        if ee.calculations_performed:
            calc_badge = _render_epistemic_badge(EpistemicLevel.CALCULATED)
            html_parts.append(
                f'    <h3 class="sub-heading">5.3 Derived Engineering Calculations '
                f"{calc_badge}</h3>"
            )
            html_parts.append(
                '    <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px;">'
            )
            for calc in ee.calculations_performed:
                html_parts.append(f'      <span class="badge badge-neutral mono">{_e(calc)}</span>')
            html_parts.append("    </div>")

        html_parts.append("  </section>")

        # -------------------------------------------------------------------
        # 6. Telemetry Anomalies & Comparisons
        # -------------------------------------------------------------------
        html_parts.append('  <section class="report-section" id="anomalies-comparisons">')
        html_parts.append(
            '    <h2 class="section-heading">6. Telemetry Anomalies & Comparisons</h2>'
        )
        ac = report.anomaly_and_comparison

        # 6.1 Anomalies Table
        anom_badge = _render_epistemic_badge(EpistemicLevel.CALCULATED)
        html_parts.append(f'    <h3 class="sub-heading">6.1 Telemetry Anomalies {anom_badge}</h3>')
        if ac.anomalies:
            html_parts.append('    <div class="table-responsive">')
            html_parts.append('      <table class="data-table">')
            html_parts.append(
                "        <thead><tr><th>Anomaly ID</th><th>Parameter</th><th>Class</th>"
                "<th>Severity</th><th>Method</th><th>Timestamp</th><th>Observed</th>"
                "<th>Expected</th><th>Deviation</th><th>Explanation</th></tr></thead>"
            )
            html_parts.append("        <tbody>")
            for an in ac.anomalies:
                ts_str = _fmt_datetime(an.timestamp)
                dev_str = _fmt_float(an.deviation_value) if an.deviation_value is not None else "-"
                html_parts.append("          <tr>")
                html_parts.append(
                    f'            <td class="mono"><strong>{_e(an.anomaly_id)}</strong></td>'
                )
                html_parts.append(f'            <td class="mono">{_e(an.target_column)}</td>')
                html_parts.append(f"            <td>{_e(an.classification)}</td>")
                html_parts.append(f"            <td>{_render_status_badge(an.severity)}</td>")
                html_parts.append(
                    f'            <td class="mono" style="font-size: 11px;">{ts_str}</td>'
                )
                html_parts.append(f'            <td class="mono">{_e(an.observed_value)}</td>')
                html_parts.append(f'            <td class="mono">{_e(an.expected_value)}</td>')
                html_parts.append(f'            <td class="mono">{dev_str}</td>')
                html_parts.append(f"            <td>{_e(an.explanation)}</td>")
                html_parts.append("          </tr>")
            html_parts.append("        </tbody>")
            html_parts.append("      </table>")
            html_parts.append("    </div>")
        else:
            html_parts.append(
                '    <div class="empty-state">'
                "No telemetry anomalies detected across evaluated parameters.</div>"
            )

        # 6.2 Expected vs Observed Comparisons Table
        comp_badge = _render_epistemic_badge(EpistemicLevel.CALCULATED)
        html_parts.append(
            f'    <h3 class="sub-heading">6.2 Expected vs Observed Comparisons {comp_badge}</h3>'
        )
        if ac.comparisons:
            html_parts.append('    <div class="table-responsive">')
            html_parts.append('      <table class="data-table">')
            html_parts.append(
                "        <thead><tr><th>Comparison ID</th><th>Parameter</th>"
                "<th>Expected</th><th>Observed</th><th>Deviation</th>"
                "<th>In Tolerance</th><th>Status</th></tr></thead>"
            )
            html_parts.append("        <tbody>")
            for cmp in ac.comparisons:
                dev_str = _fmt_float(cmp.deviation) if cmp.deviation is not None else "-"
                if cmp.within_tolerance is True:
                    tol_str = "Yes"
                elif cmp.within_tolerance is False:
                    tol_str = "No"
                else:
                    tol_str = "N/A"
                html_parts.append("          <tr>")
                html_parts.append(
                    f'            <td class="mono"><strong>{_e(cmp.comparison_id)}</strong></td>'
                )
                html_parts.append(f'            <td class="mono">{_e(cmp.parameter_id)}</td>')
                html_parts.append(f'            <td class="mono">{_e(cmp.expected_value)}</td>')
                html_parts.append(f'            <td class="mono">{_e(cmp.observed_value)}</td>')
                html_parts.append(f'            <td class="mono">{dev_str}</td>')
                html_parts.append(f"            <td>{tol_str}</td>")
                html_parts.append(f"            <td>{_render_status_badge(cmp.status)}</td>")
                html_parts.append("          </tr>")
            html_parts.append("        </tbody>")
            html_parts.append("      </table>")
            html_parts.append("    </div>")
        else:
            html_parts.append(
                '    <div class="empty-state">No expected-vs-observed comparisons evaluated.</div>'
            )

        html_parts.append("  </section>")

        # -------------------------------------------------------------------
        # 7. Investigation Intelligence & Recommendations
        # -------------------------------------------------------------------
        html_parts.append('  <section class="report-section" id="investigation-intelligence">')
        html_parts.append(
            '    <h2 class="section-heading">7. Investigation Intelligence & Recommendations</h2>'
        )
        ii = report.investigation_intelligence

        # 7.1 Evidence Items Table
        html_parts.append('    <h3 class="sub-heading">7.1 Traceable Evidence Items</h3>')
        if ii.evidence_items:
            html_parts.append('    <div class="table-responsive">')
            html_parts.append('      <table class="data-table">')
            html_parts.append(
                "        <thead><tr><th>Evidence ID</th><th>Type</th><th>Parameter</th>"
                "<th>Observed</th><th>Margin</th><th>Epistemic Level</th><th>Description</th>"
                "<th>Source URI</th></tr></thead>"
            )
            html_parts.append("        <tbody>")
            for ev in ii.evidence_items:
                obs_str = _fmt_float(ev.observed_value) if ev.observed_value is not None else "-"
                margin_str = _fmt_float(ev.margin) if ev.margin is not None else "-"
                html_parts.append("          <tr>")
                html_parts.append(
                    f'            <td class="mono"><strong>{_e(ev.evidence_id)}</strong></td>'
                )
                html_parts.append(f"            <td>{_e(ev.evidence_type)}</td>")
                html_parts.append(f'            <td class="mono">{_e(ev.parameter_id)}</td>')
                html_parts.append(f'            <td class="mono">{obs_str}</td>')
                html_parts.append(f'            <td class="mono">{margin_str}</td>')
                ev_ep = _render_epistemic_badge(ev.epistemic_level)
                html_parts.append(f"            <td>{ev_ep}</td>")
                html_parts.append(f"            <td>{_e(ev.description)}</td>")
                html_parts.append(
                    f'            <td class="mono" style="font-size: 11px;">'
                    f"{_e(ev.source_reference)}</td>"
                )
                html_parts.append("          </tr>")
            html_parts.append("        </tbody>")
            html_parts.append("      </table>")
            html_parts.append("    </div>")
        else:
            html_parts.append(
                '    <div class="empty-state">No structured evidence items recorded.</div>'
            )

        # 7.2 Contributor Correlation Rankings Table
        cr_badge = _render_epistemic_badge(EpistemicLevel.CALCULATED)
        html_parts.append(
            f'    <h3 class="sub-heading">7.2 Contributor Correlation Rankings {cr_badge}</h3>'
        )
        if ii.contributor_rankings:
            html_parts.append('    <div class="table-responsive">')
            html_parts.append('      <table class="data-table">')
            html_parts.append(
                "        <thead><tr><th>Analysis ID</th><th>Target Signal</th>"
                "<th>Candidate Factor</th><th>Method</th><th>Coefficient</th>"
                "<th>p-value</th><th>Status</th></tr></thead>"
            )
            html_parts.append("        <tbody>")
            for cr in ii.contributor_rankings:
                coeff_str = _fmt_float(cr.coefficient) if cr.coefficient is not None else "-"
                pval_str = _fmt_float(cr.p_value, 5) if cr.p_value is not None else "-"
                html_parts.append("          <tr>")
                html_parts.append(
                    f'            <td class="mono"><strong>{_e(cr.analysis_id)}</strong></td>'
                )
                html_parts.append(f'            <td class="mono">{_e(cr.target_column)}</td>')
                html_parts.append(f'            <td class="mono">{_e(cr.factor_column)}</td>')
                html_parts.append(f"            <td>{_e(cr.method)}</td>")
                html_parts.append(f'            <td class="mono">{coeff_str}</td>')
                html_parts.append(f'            <td class="mono">{pval_str}</td>')
                html_parts.append(f"            <td>{_render_status_badge(cr.status)}</td>")
                html_parts.append("          </tr>")
            html_parts.append("        </tbody>")
            html_parts.append("      </table>")
            html_parts.append("    </div>")
        else:
            html_parts.append(
                '    <div class="empty-state">No contributor correlation rankings generated.</div>'
            )

        # 7.3 Recommendations Cards
        rec_badge = _render_epistemic_badge(EpistemicLevel.RECOMMENDATION)
        html_parts.append(
            f'    <h3 class="sub-heading">7.3 Advisory Recommendations {rec_badge}</h3>'
        )
        if ii.recommendations:
            for rec in ii.recommendations:
                html_parts.append('    <div class="recommendation-card">')
                html_parts.append('      <div class="recommendation-header">')
                html_parts.append(
                    f'        <div class="recommendation-title">{_e(rec.title)} '
                    f'<span class="mono" style="font-size: 12px; color: var(--color-text-light);">'
                    f"({_e(rec.recommendation_id)})</span></div>"
                )
                html_parts.append(
                    f"        <div>{_render_status_badge(rec.status)} "
                    f"{_render_epistemic_badge(EpistemicLevel.RECOMMENDATION)}</div>"
                )
                html_parts.append("      </div>")

                # Mandatory non-binding advisory notice
                html_parts.append(
                    f'      <div class="advisory-notice-box">{_e(rec.advisory_notice)}</div>'
                )

                html_parts.append('      <div class="recommendation-body">')
                html_parts.append(f"        <p><strong>Rationale:</strong> {_e(rec.rationale)}</p>")
                html_parts.append(
                    f"        <p><strong>Actionable Description:</strong> {_e(rec.description)}</p>"
                )
                html_parts.append("      </div>")

                html_parts.append('      <div class="recommendation-metadata">')
                html_parts.append(
                    f'        <div><strong>Type:</strong> <span class="mono">'
                    f"{_e(rec.recommendation_type)}</span></div>"
                )
                rule_str = _e(rec.rule_id) if rec.rule_id else "N/A"
                html_parts.append(
                    "        <div><strong>Rule:</strong> "
                    f'<span class="mono">{rule_str}</span></div>'
                )
                if rec.evidence_ids:
                    ev_str = ", ".join(
                        f'<span class="mono">{_e(eid)}</span>' for eid in rec.evidence_ids
                    )
                else:
                    ev_str = "None"
                html_parts.append(f"        <div><strong>Evidence Links:</strong> {ev_str}</div>")
                html_parts.append("      </div>")
                html_parts.append("    </div>")
        else:
            html_parts.append(
                '    <div class="empty-state">No advisory recommendations generated.</div>'
            )

        html_parts.append("  </section>")

        # -------------------------------------------------------------------
        # 8. Analytical Limitations, Caveats & Data Gaps
        # -------------------------------------------------------------------
        html_parts.append('  <section class="report-section" id="limitations">')
        html_parts.append(
            '    <h2 class="section-heading">8. Analytical Limitations, Caveats & Data Gaps</h2>'
        )
        lims = report.limitations
        if lims.has_limitations:
            html_parts.append('    <div class="callout callout-warning">')
            html_parts.append(
                "      <strong>Operational Limitations Active:</strong> Unexecuted analytical "
                "stages or inconclusive checks limit analytical scope. Missing checks must never "
                "be presumed safe or compliant."
            )
            html_parts.append("    </div>")

            if lims.unexecuted_stages:
                unexec_str = ", ".join(
                    f'<span class="mono">{_e(s)}</span>' for s in lims.unexecuted_stages
                )
            else:
                unexec_str = "None"
            if lims.failed_stages:
                failed_str = ", ".join(
                    f'<span class="mono">{_e(s)}</span>' for s in lims.failed_stages
                )
            else:
                failed_str = "None"

            html_parts.append('    <div style="font-size: 13px; margin-bottom: 16px;">')
            html_parts.append(f"      <div><strong>Unexecuted Stages:</strong> {unexec_str}</div>")
            html_parts.append(f"      <div><strong>Failed Stages:</strong> {failed_str}</div>")
            html_parts.append(
                f"      <div><strong>Inconclusive Checks:</strong> "
                f"{lims.inconclusive_checks_count}</div>"
            )
            html_parts.append("    </div>")

            if lims.items:
                html_parts.append('    <div class="table-responsive">')
                html_parts.append('      <table class="data-table">')
                html_parts.append(
                    "        <thead><tr><th>Limitation ID</th><th>Category</th><th>Stage</th>"
                    "<th>Impact</th><th>Description</th></tr></thead>"
                )
                html_parts.append("        <tbody>")
                for it in lims.items:
                    st_str = it.stage_name if it.stage_name else "-"
                    html_parts.append("          <tr>")
                    html_parts.append(
                        f'            <td class="mono"><strong>{_e(it.limitation_id)}</strong></td>'
                    )
                    html_parts.append(f"            <td>{_e(it.category)}</td>")
                    html_parts.append(f'            <td class="mono">{_e(st_str)}</td>')
                    html_parts.append(f"            <td>{_e(it.impact)}</td>")
                    html_parts.append(f"            <td>{_e(it.description)}</td>")
                    html_parts.append("          </tr>")
                html_parts.append("        </tbody>")
                html_parts.append("      </table>")
                html_parts.append("    </div>")
        else:
            html_parts.append(
                '    <div class="empty-state">'
                "No operational limitations, skipped stages, or data gaps identified.</div>"
            )
        html_parts.append("  </section>")

        # -------------------------------------------------------------------
        # 9. Provenance & Audit Trail
        # -------------------------------------------------------------------
        html_parts.append('  <section class="report-section" id="provenance">')
        html_parts.append('    <h2 class="section-heading">9. Provenance & Audit Trail</h2>')
        prov = report.provenance
        html_parts.append('    <div class="table-responsive">')
        html_parts.append('      <table class="data-table">')
        html_parts.append(
            "        <thead><tr><th>Lineage Field</th><th>Value</th>"
            "<th>Epistemic Standing</th></tr></thead>"
        )
        html_parts.append("        <tbody>")
        html_parts.append(
            "          <tr><td><strong>Root Provenance ID</strong></td>"
            f'<td class="mono">{_e(prov.root_provenance_id)}</td>'
            f"<td>{_render_epistemic_badge(prov.epistemic_level)}</td></tr>"
        )
        html_parts.append(
            "          <tr><td><strong>Source Telemetry URI</strong></td>"
            f'<td class="mono">{_e(prov.source_uri)}</td>'
            f"<td>{_render_epistemic_badge(EpistemicLevel.OBSERVED)}</td></tr>"
        )
        html_parts.append(
            "          <tr><td><strong>Root Activity Name</strong></td>"
            f'<td class="mono">{_e(prov.activity_name)}</td>'
            f"<td>{_render_epistemic_badge(EpistemicLevel.CALCULATED)}</td></tr>"
        )
        if prov.parent_record_ids:
            parent_str = ", ".join(
                f'<span class="mono">{_e(p)}</span>' for p in prov.parent_record_ids
            )
        else:
            parent_str = "None"
        html_parts.append(
            f"          <tr><td><strong>Parent Record IDs</strong></td>"
            f"<td>{parent_str}</td><td>-</td></tr>"
        )
        html_parts.append("        </tbody>")
        html_parts.append("      </table>")
        html_parts.append("    </div>")

        if prov.stage_provenance_records:
            html_parts.append('    <h3 class="sub-heading">Stage Provenance Lineage Map</h3>')
            html_parts.append('    <div class="table-responsive">')
            html_parts.append('      <table class="data-table">')
            html_parts.append(
                "        <thead><tr><th>Pipeline Stage</th>"
                "<th>Cryptographic Record ID</th></tr></thead>"
            )
            html_parts.append("        <tbody>")
            for st_name, rec_id in sorted(prov.stage_provenance_records.items()):
                html_parts.append("          <tr>")
                html_parts.append(f'            <td class="mono">{_e(st_name)}</td>')
                html_parts.append(f'            <td class="mono">{_e(rec_id)}</td>')
                html_parts.append("          </tr>")
            html_parts.append("        </tbody>")
            html_parts.append("      </table>")
            html_parts.append("    </div>")
        html_parts.append("  </section>")

        # -------------------------------------------------------------------
        # 10. Visualization Figures (if supplied)
        # -------------------------------------------------------------------
        if figures:
            html_parts.append('  <section class="report-section" id="visualizations">')
            html_parts.append(
                '    <h2 class="section-heading">10. Interactive Telemetry Visualizations</h2>'
            )

            # Determine items: mapping or sequence
            fig_items: list[tuple[str, Any]] = []
            if isinstance(figures, Mapping):
                for caption, fig_obj in figures.items():
                    fig_items.append((caption, fig_obj))
            elif isinstance(figures, Sequence):
                for idx, fig_obj in enumerate(figures, start=1):
                    caption = f"Figure {idx}"
                    fig_items.append((caption, fig_obj))

            first_figure = True
            for caption, fig_obj in fig_items:
                actual_fig = getattr(fig_obj, "figure", fig_obj)
                html_parts.append('    <div class="figure-container">')
                caption_span = f"<span>{_e(caption)}</span>"
                html_parts.append(f'      <div class="figure-caption">{caption_span}</div>')
                html_parts.append('      <div class="figure-plot-wrapper">')

                if actual_fig is not None and hasattr(actual_fig, "to_html"):
                    # For self-contained offline embedding: include JS once if "inline"
                    if include_plotlyjs == "inline":
                        fig_js_opt: bool | str = "inline" if first_figure else False
                    elif include_plotlyjs == "cdn":
                        fig_js_opt = False  # Script was already added in <head>
                    else:
                        fig_js_opt = False

                    fig_html = str(
                        actual_fig.to_html(
                            full_html=False,
                            include_plotlyjs=fig_js_opt,
                        )
                    )
                    html_parts.append(fig_html)
                    first_figure = False
                else:
                    html_parts.append(
                        '        <div class="empty-state">Unable to render visualization: '
                        "supplied object is not a Plotly figure.</div>"
                    )

                html_parts.append("      </div>")
                html_parts.append("    </div>")

            html_parts.append("  </section>")

        # -------------------------------------------------------------------
        # Footer
        # -------------------------------------------------------------------
        html_parts.append('  <footer class="report-footer">')
        html_parts.append(
            f"    <div>AERIS Analytical Core &bull; Post-Flight Engineering Analysis Pipeline "
            f"&bull; Report ID: {_e(report.metadata.report_id)}</div>"
        )
        html_parts.append(
            '    <div style="margin-top: 4px; font-size: 11px;">'
            "FOR ENGINEERING EVALUATION AND FLIGHT TEST ANALYSIS ONLY</div>"
        )
        html_parts.append("  </footer>")

        html_parts.append("</div>")  # .report-container
        html_parts.append("</body>")
        html_parts.append("</html>")

        return "\n".join(html_parts)


def render_html_report(
    report: EngineeringReport,
    figures: Sequence[Any] | Mapping[str, Any] | None = None,
    include_plotlyjs: bool | str = False,
) -> str:
    """
    Convenience function to render an EngineeringReport as self-contained HTML.

    Parameters
    ----------
    report : EngineeringReport
        The report domain model instance.
    figures : Sequence[Any] | Mapping[str, Any] | None
        Optional sequence or dictionary of figures to embed.
    include_plotlyjs : bool | str
        Plotly.js inclusion mode (False, 'inline', or 'cdn').

    Returns
    -------
    str
        Rendered HTML document string.
    """
    return HtmlReportRenderer.render(
        report=report,
        figures=figures,
        include_plotlyjs=include_plotlyjs,
    )
