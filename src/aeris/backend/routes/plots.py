"""
AERIS Backend Plots & Visualization Router.

Generates interactive Plotly figure JSON structures for single parameters and multi-signal panels.
Resolves canonical flight test parameters to dataset columns using validated parameter mappings.
"""

from __future__ import annotations

import json

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status

from aeris.backend.deps import get_dataset_store, get_repository
from aeris.backend.models import (
    AvailableSignalItem,
    AvailableSignalsResponse,
    MultiSignalPlotRequest,
    PlotlyFigureResponse,
)
from aeris.core.registry.canonical import (
    CANONICAL_BY_ID,
    CANONICAL_LABELS,
    infer_unit_from_column_name,
)
from aeris.core.visualization.models import (
    MultiSignalPanelConfig,
    PanelLayoutMode,
    SignalSpec,
    TimeSeriesPlotConfig,
)
from aeris.core.visualization.panels import build_multi_signal_panel
from aeris.core.visualization.time_series import generate_time_series_plot
from aeris.storage.dataset_store import ManagedDatasetStore
from aeris.storage.models import ParameterMappingRecord
from aeris.storage.repository import SQLiteMetadataRepository

router = APIRouter(
    prefix="/projects/{project_id}/analyses/{analysis_id}/plots",
    tags=["Visualizations"],
)


def _detect_time_column(df: pd.DataFrame, mappings: list[ParameterMappingRecord]) -> str | None:
    """Identify the primary timestamp reference column from mappings or column heuristics."""
    for m in mappings:
        if m.is_timestamp and m.raw_column_name in df.columns:
            return m.raw_column_name
    for col in df.columns:
        col_lower = col.lower()
        if "time" in col_lower or "stamp" in col_lower:
            return col
    return "timestamp" if "timestamp" in df.columns else None


def resolve_signals_for_plotting(
    df: pd.DataFrame,
    requested_signal_ids: list[str],
    mappings: list[ParameterMappingRecord],
) -> tuple[pd.DataFrame, list[SignalSpec], list[str]]:
    """
    Resolve requested signal identifiers against dataset columns and parameter mappings.

    Preserves strict unit awareness:
    - If mapped canonically, binds canonical parameter ID, unit, and display name.
    - If raw column name, preserves raw column name and infers unit from column suffix.
    - Never fabricates data or substitutes unrelated parameters.
    - Missing signals are returned in the missing list for explicit error reporting.
    """
    working_df = df.copy()
    target_to_mapping = {
        m.target_parameter_id: m for m in mappings if m.raw_column_name in df.columns
    }
    raw_to_mapping = {m.raw_column_name: m for m in mappings}

    resolved_specs: list[SignalSpec] = []
    missing_signals: list[str] = []

    for sig in requested_signal_ids:
        if sig in target_to_mapping:
            m = target_to_mapping[sig]
            raw_col = m.raw_column_name
            working_df[sig] = df[raw_col]
            canonical_meta = CANONICAL_BY_ID.get(sig)
            display_name = (
                canonical_meta.display_name if canonical_meta else CANONICAL_LABELS.get(sig, sig)
            )
            unit = (
                m.target_unit
                or m.source_unit
                or (canonical_meta.canonical_unit if canonical_meta else None)
                or infer_unit_from_column_name(raw_col)
            )
            resolved_specs.append(
                SignalSpec(
                    parameter_id=sig,
                    display_name=display_name,
                    unit=unit,
                )
            )
        elif sig in df.columns:
            raw_m = raw_to_mapping.get(sig)
            unit = (
                (raw_m.target_unit or raw_m.source_unit)
                if raw_m
                else infer_unit_from_column_name(sig)
            )
            display_name = (
                CANONICAL_LABELS.get(raw_m.target_parameter_id, sig)
                if (raw_m and raw_m.target_parameter_id in CANONICAL_LABELS)
                else sig
            )
            resolved_specs.append(
                SignalSpec(
                    parameter_id=sig,
                    display_name=display_name,
                    unit=unit,
                )
            )
        else:
            missing_signals.append(sig)

    return working_df, resolved_specs, missing_signals


@router.get("/signals", response_model=AvailableSignalsResponse)
async def get_available_signals(
    project_id: str,
    analysis_id: str,
    repository: SQLiteMetadataRepository = Depends(get_repository),
    dataset_store: ManagedDatasetStore = Depends(get_dataset_store),
) -> AvailableSignalsResponse:
    """Retrieve plottable signals (mapped canonical and raw columns) and defaults."""
    run = repository.get_analysis_run(analysis_id)
    if run is None or run.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis run '{analysis_id}' not found in project '{project_id}'.",
        )

    file_path = dataset_store.get_dataset_path(run.dataset_id)
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    mappings = repository.get_parameter_mappings(run.dataset_id)
    time_col = _detect_time_column(df, mappings)

    signals: list[AvailableSignalItem] = []
    mapped_raw_cols: set[str] = set()

    # 1. Add mapped signals
    for m in mappings:
        if m.is_timestamp or m.raw_column_name not in df.columns:
            continue
        target_id = m.target_parameter_id
        raw_col = m.raw_column_name
        mapped_raw_cols.add(raw_col)
        canonical_meta = CANONICAL_BY_ID.get(target_id)
        disp_name = (
            canonical_meta.display_name
            if canonical_meta
            else CANONICAL_LABELS.get(target_id, target_id)
        )
        unit = (
            m.target_unit
            or m.source_unit
            or (canonical_meta.canonical_unit if canonical_meta else None)
            or infer_unit_from_column_name(raw_col)
        )
        label = f"{disp_name} ({raw_col})"
        signals.append(
            AvailableSignalItem(
                signal_id=target_id,
                label=label,
                unit=unit,
                is_mapped=True,
                raw_column_name=raw_col,
            )
        )

    # 2. Add raw dataset columns (excluding time column)
    for col in df.columns:
        if col == time_col:
            continue
        unit = infer_unit_from_column_name(col)
        label = f"{col} [{unit}]" if unit else col
        signals.append(
            AvailableSignalItem(
                signal_id=col,
                label=label,
                unit=unit,
                is_mapped=col in mapped_raw_cols,
                raw_column_name=col,
            )
        )

    # Compute default signals
    available_ids = [s.signal_id for s in signals]
    preferred = ["indicated_airspeed", "pressure_altitude"]
    selected_preferred = [p for p in preferred if p in available_ids]

    if selected_preferred:
        default_signals = selected_preferred
    elif signals:
        default_signals = [s.signal_id for s in signals[:2]]
    else:
        default_signals = []

    return AvailableSignalsResponse(
        dataset_id=run.dataset_id,
        signals=signals,
        default_signals=default_signals,
        raw_columns=list(df.columns),
    )


@router.get("/time-series", response_model=PlotlyFigureResponse)
async def get_time_series_plot(
    project_id: str,
    analysis_id: str,
    parameter_id: str = Query(..., description="Target parameter identifier to plot."),
    dark_mode: bool = Query(default=False, description="Apply dark theme."),
    repository: SQLiteMetadataRepository = Depends(get_repository),
    dataset_store: ManagedDatasetStore = Depends(get_dataset_store),
) -> PlotlyFigureResponse:
    """Generate publication-grade interactive Plotly figure JSON for a parameter time-series."""
    run = repository.get_analysis_run(analysis_id)
    if run is None or run.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis run '{analysis_id}' not found in project '{project_id}'.",
        )

    file_path = dataset_store.get_dataset_path(run.dataset_id)
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    mappings = repository.get_parameter_mappings(run.dataset_id)
    time_col = _detect_time_column(df, mappings)
    if not time_col:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No timestamp column found in dataset columns: {list(df.columns)}",
        )

    working_df, resolved_specs, missing = resolve_signals_for_plotting(
        df=df,
        requested_signal_ids=[parameter_id],
        mappings=mappings,
    )
    if missing:
        available_cols = [c for c in df.columns if c != time_col]
        mapped_targets = [m.target_parameter_id for m in mappings if not m.is_timestamp]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Requested signal '{parameter_id}' not found in dataset columns {available_cols} "
                f"or configured parameter mappings {mapped_targets}. "
                f"Configure parameter mappings on the Datasets page or select an available column."
            ),
        )

    spec = resolved_specs[0]
    config = TimeSeriesPlotConfig(
        dark_mode=dark_mode,
        title=f"{spec.display_name or parameter_id} Time-Series",
        unit=spec.unit,
        display_name=spec.display_name,
    )
    result = generate_time_series_plot(
        data=working_df,
        parameter_id=spec.parameter_id,
        timestamp_column=time_col,
        config=config,
    )

    figure_dict = json.loads(result.figure.to_json())

    return PlotlyFigureResponse(
        figure_json=figure_dict,
        parameter_id=parameter_id,
        unit=spec.unit,
        epistemic_level="OBSERVED",
    )


@router.post("/multi-signal", response_model=PlotlyFigureResponse)
async def generate_multi_signal_plot(
    project_id: str,
    analysis_id: str,
    request: MultiSignalPlotRequest,
    repository: SQLiteMetadataRepository = Depends(get_repository),
    dataset_store: ManagedDatasetStore = Depends(get_dataset_store),
) -> PlotlyFigureResponse:
    """Generate coordinated multi-signal Plotly panel figure (stacked or overlay)."""
    run = repository.get_analysis_run(analysis_id)
    if run is None or run.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis run '{analysis_id}' not found in project '{project_id}'.",
        )

    file_path = dataset_store.get_dataset_path(run.dataset_id)
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    mappings = repository.get_parameter_mappings(run.dataset_id)
    time_col = _detect_time_column(df, mappings)
    if not time_col:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No timestamp column found in dataset columns: {list(df.columns)}",
        )

    working_df, resolved_specs, missing = resolve_signals_for_plotting(
        df=df,
        requested_signal_ids=request.signal_ids,
        mappings=mappings,
    )
    if missing:
        available_cols = [c for c in df.columns if c != time_col]
        mapped_targets = [m.target_parameter_id for m in mappings if not m.is_timestamp]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Requested signals not found in dataset: {missing}. "
                f"Available dataset columns: {available_cols}. "
                f"Configured mapped parameters: {mapped_targets}. "
                f"Configure parameter mappings on the Datasets page or select available columns."
            ),
        )

    layout_enum = PanelLayoutMode.STACKED
    if request.layout_mode.upper() == "OVERLAY":
        layout_enum = PanelLayoutMode.OVERLAY
    elif request.layout_mode.upper() == "UNIT_GROUPED":
        layout_enum = PanelLayoutMode.UNIT_GROUPED

    config = MultiSignalPanelConfig(
        layout_mode=layout_enum,
        dark_mode=request.dark_mode,
        timestamp_column=time_col,
    )

    fig = build_multi_signal_panel(
        data=working_df,
        signals=resolved_specs,
        timestamp_column=time_col,
        config=config,
    )
    figure_dict = json.loads(fig.to_json())

    return PlotlyFigureResponse(
        figure_json=figure_dict,
        parameter_id="multi_signal",
        unit=None,
        epistemic_level="OBSERVED",
    )
