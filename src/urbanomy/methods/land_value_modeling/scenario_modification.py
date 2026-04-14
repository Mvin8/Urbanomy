"""Scenario tools for adjusting block indicators and analysing impact."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import random
from geopandas import GeoDataFrame
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy
from blocksnet.analysis.indicators import calculate_density_indicators
from blocksnet.analysis.morphotypes import get_strelka_morphotypes
from blocksnet.enums import LandUse

from .constants import (
    BlockColumn,
    DEFAULT_SQM_PER_PERSON,
    ScenarioResultKey,
)


class ScenarioTEPModifier:
    """Apply scenario changes to a single block.

    Parameters
    ----------
    blocks : geopandas.GeoDataFrame
        Source blocks dataset that will be copied during each scenario
        application.
    """

    def __init__(self, blocks: GeoDataFrame) -> None:
        """Store a reference blocks dataset used as the scenario baseline.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Source blocks that will be copied and modified per scenario.
        """
        self._blocks = blocks

    def apply(
        self,
        target_id: Any,
        changes: Mapping[str, Any],
        *,
        target_id_column: str = BlockColumn.ID.value,
    ) -> GeoDataFrame:
        """Return a modified copy of the blocks with updated TEP values.

        Parameters
        ----------
        target_id : Any
            Value from ``target_id_column`` identifying the block to modify.
        changes : Mapping[str, object]
            Dictionary of field updates (e.g. ``{"land_use": LandUse.RESIDENTIAL}``
            or an equivalent string token).
        target_id_column : str, optional
            Name of the identifier column used to locate the target block.

        Returns
        -------
        geopandas.GeoDataFrame
            Modified copy of the original blocks DataFrame.
        """

        df = self._blocks.copy()
        resolved_index = self._resolve_target_index(df, target_id, target_id_column)

        original_row = df.loc[resolved_index].copy()
        row = original_row.to_dict()
        normalised_changes = dict(changes)

        land_use_key = BlockColumn.LAND_USE.value
        if land_use_key in normalised_changes:
            normalised_changes[land_use_key] = self._coerce_land_use(normalised_changes[land_use_key])

        row.update(normalised_changes)

        site_area_key = BlockColumn.SITE_AREA.value
        build_key = BlockColumn.BUILD_FLOOR_AREA.value
        living_key = BlockColumn.LIVING_AREA.value
        non_living_key = BlockColumn.NON_LIVING_AREA.value
        mxi_key = BlockColumn.MXI.value
        footprint_key = BlockColumn.FOOTPRINT_AREA.value
        population_key = BlockColumn.POPULATION.value
        share_key = BlockColumn.SHARE.value
        residential_key = BlockColumn.RESIDENTIAL.value
        site_area = float(row.get(site_area_key, df.at[resolved_index, site_area_key]))

        build = float(row.get(build_key, df.at[resolved_index, build_key]))
        live = float(row.get(living_key, df.at[resolved_index, living_key]))
        non = row.get(non_living_key, np.nan)

        build = max(build, 0.0)
        if living_key not in changes and ((mxi_key in changes) or (mxi_key in row)):
            mxi = float(row.get(mxi_key, df.at[resolved_index, mxi_key]))
            mxi = float(np.clip(mxi, 0.0, 1.0))
            live = mxi * build
        live = float(np.clip(live, 0.0, build))

        if not np.isfinite(non):
            non = max(build - live, 0.0)
        else:
            non = max(float(non), 0.0)

        if (non_living_key in changes and living_key in changes) and (build_key not in changes):
            build = live + float(non)

        row[build_key] = build
        row[living_key] = live
        row[non_living_key] = float(non)

        footprint = float(row.get(footprint_key, df.at[resolved_index, footprint_key]))
        row[population_key] = live / DEFAULT_SQM_PER_PERSON if DEFAULT_SQM_PER_PERSON > 0 else 0.0

        row[BlockColumn.FSI.value] = build / site_area if site_area > 0 else np.nan
        row[BlockColumn.GSI.value] = footprint / site_area if site_area > 0 else np.nan
        row[mxi_key] = (row[living_key] / build) if build > 0 else np.nan
        row[BlockColumn.L.value] = (build / footprint) if footprint > 0 else np.nan
        # row[BlockColumn.OSR.value] = (site_area - footprint) / site_area if site_area > 0 else np.nan

        if residential_key in normalised_changes and share_key not in normalised_changes:
            row[share_key] = float(row[residential_key])

        # Preserve pre-scenario values needed for demolition-cost estimates.
        row["land_use_before"] = original_row.get("land_use_before", original_row.get(land_use_key))
        row["build_floor_area_before"] = original_row.get(
            "build_floor_area_before",
            original_row.get(build_key),
        )

        for key, value in row.items():
            df.at[resolved_index, key] = value

        # Keep morphotype consistent with the updated block TEP.
        try:
            morphotypes = get_strelka_morphotypes(df)
            if "morphotype" in morphotypes.columns and resolved_index in morphotypes.index:
                df.at[resolved_index, "morphotype"] = morphotypes.at[resolved_index, "morphotype"]
        except Exception:
            # Do not break scenario application if morphotype recomputation fails.
            pass

        return df

    @staticmethod
    def _resolve_target_index(df: GeoDataFrame, target_id: Any, target_id_column: str) -> Any:
        if target_id_column not in df.columns:
            raise KeyError(f"Target id column {target_id_column!r} is not present in the dataset.")

        matches = df.index[df[target_id_column] == target_id]
        if len(matches) == 0:
            raise KeyError(
                f"Block with {target_id_column}={target_id!r} is not present in the dataset."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple blocks found for {target_id_column}={target_id!r}; the identifier must be unique."
            )
        return matches[0]

    @staticmethod
    def _coerce_land_use(value: Any) -> LandUse:
        """Convert user-provided land-use tokens to ``LandUse`` enum values."""
        if isinstance(value, LandUse):
            return value
        if value is None:
            raise ValueError("land_use cannot be None")

        text = str(value).strip()
        if not text:
            raise ValueError("land_use cannot be empty")

        try:
            return LandUse(text)
        except ValueError:
            pass

        upper = text.upper()
        if upper.startswith("LANDUSE."):
            upper = upper.split(".", 1)[1]

        try:
            return LandUse[upper]
        except KeyError as exc:
            raise ValueError(f"Unknown land_use value: {value!r}") from exc

def plot_scenario_impact(
    *,
    blocks: GeoDataFrame,
    target_idx: Any,
    target_id_column: str = "id",
    before_column: str = "land_value_before",
    after_column: str = "land_value",
    pct_column: str = "land_value_delta_pct",
    delta_column: str = "d_rub",
    eps: float = 1e-9,
    buffer_radius: float = 4000.0,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    print_summary: bool = True,
    print_quarter_stats: bool = True,
) -> Dict[str, object]:
    """Визуализировать и вывести статистику по уже рассчитанным изменениям цен.

    Ожидается, что ``blocks`` уже содержит колонки с ценой до/после сценария.
    Функция ничего не предсказывает — только строит карту изменений и печатает
    сводную статистику. На карте используется процентное изменение полной цены
    ``land_value_delta_pct``. ``target_idx`` ожидает
    идентификатор квартала (значение в колонке ``target_id_column``).
    """

    missing = [col for col in (before_column, after_column) if col not in blocks.columns]
    if missing:
        raise KeyError(f"blocks is missing required columns: {', '.join(sorted(missing))}")

    combined = blocks.copy()
    combined["land_value_before"] = combined[before_column].astype(float)
    combined["land_value_after"] = combined[after_column].astype(float)
    combined = _ensure_valid_geoms(combined)

    if target_id_column not in combined.columns:
        raise KeyError(f"blocks is missing target_id_column: {target_id_column}")
    match = combined[combined[target_id_column] == target_idx]
    if match.empty:
        raise KeyError(f"target_id {target_idx!r} not found in column {target_id_column!r}")
    resolved_target_idx = match.index[0]

    if delta_column in combined.columns:
        combined["d_rub"] = combined[delta_column].astype(float)
    else:
        combined["d_rub"] = combined["land_value_after"] - combined["land_value_before"]

    if pct_column in combined.columns:
        combined["d_pct"] = combined[pct_column].astype(float)
    else:
        combined["d_pct"] = (
            (combined["land_value_after"] / combined["land_value_before"] - 1.0) * 100
        )
    combined["d_pct"] = combined["d_pct"].replace([np.inf, -np.inf], np.nan)

    buffer_gdf = _build_buffer(combined, resolved_target_idx, buffer_radius)
    buffer_gdf = _ensure_valid_geoms(buffer_gdf)
    clipped = gpd.clip(combined, buffer_gdf) if len(combined) else combined

    changed = clipped[np.abs(clipped["d_rub"].astype(float)) > eps].copy()
    unchanged = clipped[np.abs(clipped["d_rub"].astype(float)) <= eps].copy()

    plot_column = pct_column
    if plot_column not in combined.columns:
        raise KeyError(f"blocks is missing plot_column: {plot_column}")

    # Use a fixed symmetric percentage scale for consistent maps
    vmin, vmax = -20.0, 20.0

    fig = _plot_change_map(
        changed=changed,
        unchanged=unchanged,
        target_geometry=combined.loc[resolved_target_idx, "geometry"],
        plot_column=plot_column,
        vmin=vmin,
        vmax=vmax,
        figsize=figsize,
        show=show,
    )

    summary_all = _summarise_changes(clipped)
    summary_changed = _summarise_changes(changed)

    if print_summary:
        _print_summary(
            changed,
            summary_changed,
            eps=eps,
            total_summary=summary_all,
            total_count=int(len(clipped)),
            target_idx=resolved_target_idx,
            print_quarter_stats=print_quarter_stats,
        )

    return {
        ScenarioResultKey.MAP.value: changed if len(changed) else clipped.iloc[0:0].copy(),
        ScenarioResultKey.MAP_ALL.value: clipped,
        ScenarioResultKey.FIGURE.value: fig,
        ScenarioResultKey.SUMMARY.value: summary_changed,
        ScenarioResultKey.SUMMARY_ALL.value: summary_all,
    }


def _build_buffer(blocks: GeoDataFrame, target_idx: int, radius_m: float) -> GeoDataFrame:
    """Create a buffer around the target block geometry.

    Parameters
    ----------
    blocks : geopandas.GeoDataFrame
        Source dataset containing the geometry.
    target_idx : int
        Index of the block serving as the buffer centre.
    radius_m : float
        Buffer radius expressed in metres.

    Returns
    -------
    geopandas.GeoDataFrame
        Single-row GeoDataFrame with the buffered geometry.
    """
    tgt_geom = blocks.loc[target_idx, "geometry"]
    buf = gpd.GeoSeries([tgt_geom], crs=blocks.crs)
    if buf.crs and buf.crs.is_geographic:
        utm = buf.estimate_utm_crs()
        buf = buf.to_crs(utm).buffer(radius_m).to_crs(blocks.crs)
    else:
        buf = buf.buffer(radius_m)
    return gpd.GeoDataFrame(geometry=buf, crs=blocks.crs)


def _ensure_valid_geoms(gdf: GeoDataFrame) -> GeoDataFrame:
    """Fix invalid geometries to avoid GEOS TopologyException during overlays/clipping."""
    if gdf.empty or gdf.geometry is None:
        return gdf

    geom = gdf.geometry
    try:
        invalid_mask = ~geom.is_valid
    except Exception:
        invalid_mask = pd.Series(False, index=gdf.index)

    if not invalid_mask.any():
        return gdf

    fixed = geom.copy()
    try:
        from shapely.validation import make_valid  # type: ignore
        fixed.loc[invalid_mask] = geom.loc[invalid_mask].apply(make_valid)
    except Exception:
        fixed.loc[invalid_mask] = geom.loc[invalid_mask].buffer(0)

    return gdf.set_geometry(fixed)


def _plot_change_map(
    *,
    changed: GeoDataFrame,
    unchanged: GeoDataFrame,
    target_geometry,
    plot_column: str,
    vmin: float,
    vmax: float,
    figsize: tuple[float, float] | None,
    show: bool,
) -> plt.Figure:
    """Plot price changes around the target block using the selected metric.

    Parameters
    ----------
    changed : geopandas.GeoDataFrame
        Blocks whose price change magnitude exceeds ``eps``.
    unchanged : geopandas.GeoDataFrame
        Blocks within the buffer that remain below the change threshold.
    target_geometry : shapely.geometry.base.BaseGeometry
        Geometry of the focus block to outline.
    vmin, vmax : float
        Color scale bounds for percentage change.
    plot_column : str
        Column in ``changed``/``unchanged`` to visualise (e.g., ``d_pct`` or ``d_rub``).
    figsize : tuple[float, float] | None
        Optional figure size passed to ``plt.subplots``.
    show : bool
        Whether to render the Matplotlib figure immediately.

    Returns
    -------
    matplotlib.figure.Figure
        Figure displaying the scenario impact map.
    """
    fig, ax = plt.subplots(figsize=figsize or (35, 30))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plot_changed = changed.copy()
    plot_unchanged = unchanged.copy()
    has_colorbar = False

    color_label = "Изменение цены, %"
    formatter = _fmt_pct
    if plot_column.lower().endswith("rub") or plot_column.lower().endswith("value") or plot_column == "d_rub":
        color_label = "Изменение цены, ₽"
        formatter = lambda val: _fmt_rub(val, signed=True)  # noqa: E731

    base_crs = plot_changed.crs or plot_unchanged.crs
    target_gdf = gpd.GeoDataFrame(geometry=[target_geometry], crs=base_crs)
    plotting_crs = plot_changed.crs or plot_unchanged.crs or target_gdf.crs

    if base_crs is not None:
        if plot_changed.crs is None:
            plot_changed = plot_changed.set_crs(base_crs, allow_override=True)
        if plot_unchanged.crs is None:
            plot_unchanged = plot_unchanged.set_crs(base_crs, allow_override=True)
        if target_gdf.crs is None:
            target_gdf = target_gdf.set_crs(base_crs, allow_override=True)

    target_plot_gdf = target_gdf

    if len(plot_unchanged):
        plot_unchanged.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.3, zorder=1)
    if len(plot_changed):
        plot_changed.plot(
            ax=ax,
            column=plot_column,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            legend=True,
            legend_kwds={
                "label": color_label,
                "orientation": "vertical",
                "pad": 0.02,
                "shrink": 0.7,
            },
            edgecolor="black",
            linewidth=0.4,
            zorder=2,
        )

        if len(fig.axes) > 1:
            cbar_ax = fig.axes[-1]
            cbar_ax.set_ylabel(color_label, fontsize=20)
            cbar_ax.tick_params(labelsize=14)
            cbar_ax.set_position([0.92, 0.15, 0.025, 0.7])
            has_colorbar = True

        for _, row in plot_changed.iterrows():
            x, y = row.geometry.centroid.coords[0]
            value_text = formatter(row.get(plot_column))
            ax.text(
                x,
                y,
                value_text,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="black",
                zorder=4,
            )
            ax.text(
                x,
                y,
                _format_land_use(row.get("land_use")),
                ha="center",
                va="center",
                fontsize=6,
                color="black",
                transform=offset_copy(ax.transData, fig=fig, y=-10, units="points"),
                zorder=4,
            )

    target_plot_gdf.boundary.plot(
        ax=ax,
        edgecolor="red",
        linewidth=2.0,
        zorder=3,
    )

    legend_handles: list[object] = []
    if len(plot_unchanged):
        legend_handles.append(
            Patch(
                facecolor="lightgrey",
                edgecolor="white",
                linewidth=0.3,
                label="Кварталы без существующих изменений",
            )
        )
    if len(plot_changed):
        legend_handles.append(
            Patch(
                facecolor="none",
                edgecolor="black",
                linewidth=0.4,
                label="Кварталы с изменением цены",
            )
        )
    legend_handles.append(
        Line2D([0], [0], color="red", linewidth=2.0, label="Границы изменяемого квартала")
    )
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        title="Обозначения",
        title_fontsize=18,
        prop={"size": 16},
    )

    if has_colorbar:
        ax.set_position([0.02, 0.02, 0.88, 0.96])
    else:
        ax.set_position([0.02, 0.02, 0.96, 0.96])

    _add_scale_bar(
        ax=ax,
        crs=plotting_crs,
        label="Масштаб",
    )

    ax.set_title("Изменение цены в процентах (after − before)", pad=12, fontsize=30)
    ax.axis("off")
    if show:
        plt.show()
    return fig


def _add_scale_bar(
    *,
    ax: plt.Axes,
    crs,
    label: str,
    units: str = "м",
    length: float | None = None,
    location: tuple[float, float] = (0.08, 0.08),
    linewidth: float = 4.0,
) -> None:
    """Draw a simple scale bar in the lower corner of the map."""
    if not ax:
        return

    if hasattr(crs, "is_geographic") and crs.is_geographic:
        ax.text(
            0.02,
            0.02,
            "Масштаб недоступен (географическая проекция)",
            transform=ax.transAxes,
            fontsize=10,
            va="bottom",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        return

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    width = float(x_max - x_min)
    height = float(y_max - y_min)
    if width <= 0 or height <= 0:
        return

    candidates = (50, 100, 200, 250, 500, 1000, 2000, 5000, 10000, 20000)
    target = width / 5.0
    if not length:
        length = max((cand for cand in candidates if cand <= target), default=candidates[0])
        if length > target and len(candidates):
            length = min(candidates, key=lambda cand: abs(cand - target))

    x_start = x_min + width * location[0]
    y_start = y_min + height * location[1]
    segment_length = length / 2
    baseline = y_start
    tick_height = height * 0.02

    ax.plot(
        [x_start, x_start + length],
        [baseline, baseline],
        color="black",
        linewidth=linewidth,
        solid_capstyle="butt",
        zorder=5,
    )

    tick_positions = (x_start, x_start + segment_length, x_start + length)
    tick_labels = (
        "0",
        f"{int(round(segment_length))}",
        f"{int(round(length))}",
    )
    for x_pos in tick_positions:
        ax.plot(
            [x_pos, x_pos],
            [baseline, baseline + tick_height],
            color="black",
            linewidth=linewidth / 1.5,
            zorder=6,
        )

    for label_text, x_pos in zip(tick_labels, tick_positions):
        suffix = f" {units}" if x_pos == tick_positions[-1] else ""
        ax.text(
            x_pos,
            baseline + tick_height * 1.4,
            f"{label_text}{suffix}",
            ha="center",
            va="bottom",
            fontsize=13,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            zorder=7,
        )

    ax.text(
        x_start,
        baseline - tick_height * 1.8,
        label,
        ha="left",
        va="top",
        fontsize=13,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        zorder=6,
    )


def _summarise_changes(gdf: GeoDataFrame) -> Dict[str, float]:
    """Compute aggregate before/after price statistics for a subset of blocks.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Dataset containing ``land_value_before`` and ``land_value_after`` columns.

    Returns
    -------
    dict[str, float]
        Aggregated sums, deltas, and count information.
    """
    if gdf is None or gdf.empty:
        return {
            "sum_before": 0.0,
            "sum_after": 0.0,
            "delta": 0.0,
            "delta_pct": np.nan,
            "count": 0,
        }

    land_value_before = float(gdf["land_value_before"].sum())
    land_value_after = float(gdf["land_value_after"].sum())
    delta = land_value_after - land_value_before
    delta_pct = (
        (land_value_after / land_value_before - 1.0) * 100 if land_value_before > 0 else np.nan
    )
    return {
        "sum_before": land_value_before,
        "sum_after": land_value_after,
        "delta": delta,
        "delta_pct": delta_pct,
        "count": int(len(gdf)),
    }


def _format_land_use(value: Any) -> str:
    """Render land-use labels for plotting."""
    if isinstance(value, LandUse):
        return value.value
    if value is None:
        return "missing"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "missing"
    try:
        return LandUse(text).value
    except ValueError:
        pass
    upper = text.upper()
    if upper.startswith("LANDUSE."):
        upper = upper.split(".", 1)[1]
    return upper


def _print_summary(
    gdf: GeoDataFrame,
    summary: Dict[str, float],
    *,
    eps: float,
    total_summary: Dict[str, float] | None = None,
    total_count: int | None = None,
    target_idx: int | None = None,
    project_column: str = BlockColumn.IS_PROJECT.value,
    print_quarter_stats: bool = True,
) -> None:
    """Print human-readable summaries of scenario-induced price changes.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Blocks whose changes are being highlighted.
    summary : dict[str, float]
        Aggregated metrics for ``gdf`` from :func:`_summarise_changes`.
    eps : float
        Price-change threshold used to filter ``gdf``.
    total_summary : dict[str, float], optional
        Aggregated metrics for the entire buffer.
    total_count : int, optional
        Total number of buffered blocks.
    target_idx : int, optional
        Identifier of the modified block to highlight separately.
    project_column : str, optional
        Column indicating whether a block belongs to the project area.
    """
    def _print_summary_block(title: str, stats: Dict[str, float] | None, *, count_override: int | None = None) -> None:
        print(f"\n{title}:")
        if not stats or stats["count"] == 0:
            print(" Нет данных.")
            return
        print(" • Сумма до:   ", _fmt_rub(stats["sum_before"]))
        print(" • Сумма после:", _fmt_rub(stats["sum_after"]))
        print(" • Изм., ₽:    ", _fmt_rub(stats["delta"], signed=True))
        print(" • Изм., %:    ", _fmt_pct(stats["delta_pct"]))
        print(" • Территорий: ", count_override if count_override is not None else stats["count"])

    def _print_row(idx, row) -> None:
        land_use = _format_land_use(row.get(land_use_key))
        print(
            f"{idx} | {land_use} | {_fmt_rub(row['land_value_before'])} → "
            f"{_fmt_rub(row['land_value_after'])} | {_fmt_rub(row['d_rub'], signed=True)} | "
            f"{_fmt_pct(row['d_pct'])}"
        )

    land_use_key = BlockColumn.LAND_USE.value
    project_mask = pd.Series(False, index=gdf.index, dtype=bool)

    if len(gdf):
        project_mask = pd.Series(gdf.get(project_column, False), index=gdf.index).fillna(False).astype(bool)

    if print_quarter_stats:
        threshold_note = f" (|Δ₽| > {eps:g})" if eps > 0 else ""
        print(f"\nСтатистика по кварталам в контексте{threshold_note}:")

        if len(gdf):
            print("\nИзменяемый квартал:")
            print("index | land_use | до (₽) → после (₽) | Δ₽ | Δ%")
            if target_idx is not None and target_idx in gdf.index:
                _print_row(target_idx, gdf.loc[target_idx])
            else:
                print("—")

            project_rows = gdf[project_mask].drop(index=target_idx, errors="ignore")
            context_rows = gdf[~project_mask].drop(index=target_idx, errors="ignore")

            print("\nКварталы в границах проекта:")
            if len(project_rows):
                print("index | land_use | до (₽) → после (₽) | Δ₽ | Δ%")
                for idx, row in project_rows.iterrows():
                    _print_row(idx, row)
            else:
                print("Нет кварталов проекта с изменениями выше порога.")

            print("\nКварталы в контексте:")
            if len(context_rows):
                print("index | land_use | до (₽) → после (₽) | Δ₽ | Δ%")
                for idx, row in context_rows.iterrows():
                    _print_row(idx, row)
            else:
                print("Нет кварталов контекста с изменениями выше порога.")
        else:
            print("Нет кварталов, у которых стоимость изменилась выше порога.")

    target_rows = gdf.loc[[target_idx]] if target_idx is not None and target_idx in gdf.index else gdf.iloc[0:0]
    project_rows = gdf[project_mask].drop(index=target_idx, errors="ignore")
    context_rows = gdf[~project_mask].drop(index=target_idx, errors="ignore")

    target_summary = _summarise_changes(target_rows)
    project_summary = _summarise_changes(project_rows)
    context_summary = _summarise_changes(context_rows)

    _print_summary_block("Сводка по изменяемому кварталу", target_summary)
    _print_summary_block("Сводка по проектной территории (изменения)", project_summary)
    _print_summary_block("Сводка по контексту (изменения)", context_summary)
    _print_summary_block("Сводка по всем изменениям", summary)



def _fmt_rub(value: float, *, signed: bool = False, digits: int = 0) -> str:
    """Format numeric values as Russian roubles with thin-space separators.

    Parameters
    ----------
    value : float
        Numeric value to format.
    signed : bool, optional
        Display the sign explicitly when ``True``.
    digits : int, optional
        Number of decimal digits to display.

    Returns
    -------
    str
        Formatted currency string or an em dash if value is not finite.
    """
    try:
        value = float(value)
        if not np.isfinite(value):
            return "—"
        formatted = f"{value:+,.{digits}f}" if signed else f"{value:,.{digits}f}"
        return formatted.replace(",", " ") + " ₽"
    except Exception:
        return "—"


def _fmt_pct(value: float, digits: int = 1) -> str:
    """Format percentage values with an explicit sign when finite.

    Parameters
    ----------
    value : float
        Percentage value to display.
    digits : int, optional
        Number of decimal digits to display.

    Returns
    -------
    str
        Signed percentage string or an em dash if value is not finite.
    """
    try:
        value = float(value)
        if not np.isfinite(value):
            return "—"
        return f"{value:+.{digits}f}%"
    except Exception:
        return "—"
