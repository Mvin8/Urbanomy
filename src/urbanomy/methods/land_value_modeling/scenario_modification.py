"""Scenario tools for adjusting block indicators and analysing impact."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from geopandas import GeoDataFrame
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy

from blocksnet.enums import LandUse

from .constants import (
    BlockColumn,
    CATEGORICAL_FEATURES,
    ORIGINAL_FEATURES,
    RADIUS_LIST,
    ScenarioResultKey,
)
from .land_price_estimation import LandPriceEstimator


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

    def apply(self, target_idx: int, changes: Mapping[str, Any]) -> GeoDataFrame:
        """Return a modified copy of the blocks with updated TEP values.

        Parameters
        ----------
        target_idx : int
            Index of the block that should be modified.
        changes : Mapping[str, object]
            Dictionary of field updates (e.g. ``{"land_use": LandUse.RESIDENTIAL}``
            or an equivalent string token).

        Returns
        -------
        geopandas.GeoDataFrame
            Modified copy of the original blocks DataFrame.
        """

        df = self._blocks.copy()
        if target_idx not in df.index:
            raise KeyError(f"Block index {target_idx} is not present in the dataset.")

        row = df.loc[target_idx].to_dict()
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
        share_living_key = BlockColumn.SHARE_LIVING.value
        share_non_living_key = BlockColumn.SHARE_NON_LIVING.value

        site_area = float(row.get(site_area_key, df.at[target_idx, site_area_key]))

        build = float(row.get(build_key, df.at[target_idx, build_key]))
        live = float(row.get(living_key, df.at[target_idx, living_key]))
        non = row.get(non_living_key, np.nan)

        if not np.isfinite(non):
            if build_key in changes and living_key in changes:
                non = max(build - live, 0.0)
            elif (mxi_key in changes) or (mxi_key in row):
                mxi = float(row.get(mxi_key, df.at[target_idx, mxi_key]))
                non = max(mxi * build, 0.0)
            else:
                non = float(df.at[target_idx, non_living_key])

        if (non_living_key in changes and living_key in changes) and (build_key not in changes):
            build = live + float(non)

        row[build_key] = build
        row[living_key] = live
        row[non_living_key] = float(non)

        footprint = float(row.get(footprint_key, df.at[target_idx, footprint_key]))
        population = float(row.get(population_key, df.at[target_idx, population_key]))

        row[BlockColumn.FSI.value] = build / site_area if site_area > 0 else np.nan
        row[BlockColumn.GSI.value] = footprint / site_area if site_area > 0 else np.nan
        row[mxi_key] = (row[non_living_key] / build) if build > 0 else 0.0
        row[BlockColumn.L.value] = (build / population) if population > 0 else np.nan
        row[BlockColumn.OSR.value] = (site_area - footprint) / site_area if site_area > 0 else np.nan
        row[share_living_key] = live / site_area if site_area > 0 else np.nan
        row[share_non_living_key] = row[non_living_key] / site_area if site_area > 0 else np.nan

        if residential_key in normalised_changes and share_key not in normalised_changes:
            row[share_key] = float(row[residential_key])

        for key, value in row.items():
            df.at[target_idx, key] = value

        return df

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
    blocks_before: GeoDataFrame,
    blocks_after: GeoDataFrame,
    model,
    target_idx: int,
    orig_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    radius_list: Sequence[float] | None = None,
    eps: float = 1e-9,
    buffer_radius: float = 4000.0,
    figsize: tuple[float, float] | None = None,
    add_basemap: bool = True,
    basemap_provider: str | None = "CartoDB.Positron",
    show: bool = True,
    print_summary: bool = True,
) -> Dict[str, object]:
    """Visualise and report price changes introduced by a scenario.

    Parameters
    ----------
    blocks_before : geopandas.GeoDataFrame
        Baseline blocks prior to scenario changes.
    blocks_after : geopandas.GeoDataFrame
        Blocks after applying the scenario modifications.
    model : object
        Fitted regression model used for price prediction (log-scale output).
    target_idx : int
        Index of the target block that was modified.
    orig_features : Sequence[str], optional
        Feature list passed to the model. Defaults to
        :data:`~urbanomy.methods.land_value_modeling.constants.ORIGINAL_FEATURES`.
    categorical_features : Sequence[str], optional
        Categorical feature names. Defaults to
        :data:`~urbanomy.methods.land_value_modeling.constants.CATEGORICAL_FEATURES`.
    radius_list : Sequence[float], optional
        Distance thresholds for spatial lags. Defaults to
        :data:`~urbanomy.methods.land_value_modeling.constants.RADIUS_LIST`.
    eps : float, default=1e-9
        Threshold to decide whether the price change is significant.
    buffer_radius : float, default=4000.0
        Buffer radius (in metres) around the target block used to clip the map.
    figsize : tuple[float, float], optional
        Figure size passed to Matplotlib ``plt.subplots``.
    add_basemap : bool, default=True
        Whether to draw a cartographic basemap under the scenario layers.
    basemap_provider : str, optional
        Name of the Contextily provider to use. Defaults to ``\"CartoDB.Positron\"``.
    show : bool, default=True
        Display the generated figure using Matplotlib.
    print_summary : bool, default=True
        Whether to print textual statistics.

    Returns
    -------
    dict[str, object]
        Mapping keyed by :class:`ScenarioResultKey` values with GeoDataFrames,
        figures, and summary statistics for the scenario impact analysis.
    """

    features = tuple(orig_features) if orig_features is not None else ORIGINAL_FEATURES
    cats = tuple(categorical_features) if categorical_features is not None else CATEGORICAL_FEATURES
    radii = tuple(radius_list) if radius_list is not None else RADIUS_LIST

    before_estimator = LandPriceEstimator(
        model=model,
        blocks=blocks_before,
        radius_list=radii,
        orig_features=features,
        categorical_features=cats,
    )
    after_estimator = LandPriceEstimator(
        model=model,
        blocks=blocks_after,
        radius_list=radii,
        orig_features=features,
        categorical_features=cats,
    )

    before_pred = before_estimator.predict()[["y_log_pred", "price_pred"]]
    after_pred = after_estimator.predict()[["y_log_pred", "price_pred"]]

    combined = blocks_after.copy()
    combined["price_before"] = before_pred["price_pred"].astype(float)
    combined["price_after"] = after_pred["price_pred"].astype(float)
    combined["d_rub"] = combined["price_after"] - combined["price_before"]
    combined["d_pct"] = (
        (combined["price_after"] / combined["price_before"] - 1.0) * 100
    ).replace([np.inf, -np.inf], np.nan)

    buffer_gdf = _build_buffer(blocks_before, target_idx, buffer_radius)
    clipped = gpd.clip(combined, buffer_gdf) if len(combined) else combined

    changed = clipped[np.abs(clipped["d_rub"].astype(float)) > eps].copy()
    unchanged = clipped[np.abs(clipped["d_rub"].astype(float)) <= eps].copy()

    if len(changed):
        q_low, q_high = changed["d_pct"].quantile([0.02, 0.98]).astype(float)
        lim = float(max(abs(q_low), abs(q_high)))
        vmin, vmax = -lim, lim
    else:
        vmin, vmax = -1.0, 1.0

    fig = _plot_change_map(
        changed=changed,
        unchanged=unchanged,
        target_geometry=blocks_before.loc[target_idx, "geometry"],
        vmin=vmin,
        vmax=vmax,
        figsize=figsize,
        add_basemap=add_basemap,
        basemap_provider=basemap_provider,
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


def _resolve_basemap_provider(ctx_module, provider_name: str | None):
    """Return a Contextily provider object compatible with the installed version."""
    default = None
    cartodb = getattr(ctx_module.providers, "CartoDB", None)
    if cartodb is not None and hasattr(cartodb, "Positron"):
        default = getattr(cartodb, "Positron")
    if default is None:
        try:
            default = ctx_module.providers["CartoDB.Positron"]
        except Exception:
            default = None

    if provider_name in (None, "", "default"):
        return default

    try:
        return ctx_module.providers.normalize_provider(provider_name)
    except AttributeError:
        pass
    except Exception:
        if default is not None:
            warnings.warn(
                f"Не удалось использовать провайдера {provider_name!r}; использую CartoDB.Positron.",
                stacklevel=3,
            )
            return default
        raise

    current = ctx_module.providers
    for token in provider_name.split("."):
        if hasattr(current, token):
            current = getattr(current, token)
        elif isinstance(current, dict) and token in current:
            current = current[token]
        else:
            if default is not None:
                warnings.warn(
                    f"Не удалось распознать провайдера {provider_name!r}; использую CartoDB.Positron.",
                    stacklevel=3,
                )
                return default
            raise KeyError(provider_name)
    return current


def _plot_change_map(
    *,
    changed: GeoDataFrame,
    unchanged: GeoDataFrame,
    target_geometry,
    vmin: float,
    vmax: float,
    figsize: tuple[float, float] | None,
    add_basemap: bool,
    basemap_provider: str | None,
    show: bool,
) -> plt.Figure:
    """Plot percentage price changes around the target block.

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
    figsize : tuple[float, float] | None
        Optional figure size passed to ``plt.subplots``.
    add_basemap : bool
        Whether to draw a cartographic basemap under the map layers.
    basemap_provider : str | None
        Name of the Contextily provider to use for the basemap.
    show : bool
        Whether to render the Matplotlib figure immediately.

    Returns
    -------
    matplotlib.figure.Figure
        Figure displaying the scenario impact map.
    """
    fig, ax = plt.subplots(figsize=figsize or (35, 30))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    source_changed = changed.copy()
    source_unchanged = unchanged.copy()

    plot_changed = source_changed
    plot_unchanged = source_unchanged
    has_colorbar = False

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

    ctx_module = None
    provider_obj = None
    basemap_enabled = False
    if add_basemap:
        if plotting_crs is None:
            warnings.warn("Basemap disabled: CRS is undefined for the plot.", stacklevel=2)
        else:
            try:
                import contextily as ctx

                ctx_module = ctx
                provider_obj = _resolve_basemap_provider(ctx, basemap_provider)
                basemap_enabled = provider_obj is not None
            except Exception as exc:
                warnings.warn(f"Не удалось подготовить подложку карты: {exc}", stacklevel=2)

    if basemap_enabled:
        target_epsg = 3857
        try:
            if plot_changed.crs is not None:
                plot_changed = plot_changed.to_crs(epsg=target_epsg)
            if plot_unchanged.crs is not None:
                plot_unchanged = plot_unchanged.to_crs(epsg=target_epsg)
            if target_gdf.crs is not None:
                target_plot_gdf = target_gdf.to_crs(epsg=target_epsg)
            plotting_crs = (
                plot_changed.crs or plot_unchanged.crs or target_plot_gdf.crs or plotting_crs
            )
        except Exception as exc:
            warnings.warn(f"Не удалось перепроектировать данные для подложки: {exc}", stacklevel=2)
            basemap_enabled = False
            plot_changed = source_changed
            plot_unchanged = source_unchanged
            target_plot_gdf = gpd.GeoDataFrame(geometry=[target_geometry], crs=plotting_crs)
            plotting_crs = plot_changed.crs or plot_unchanged.crs or target_plot_gdf.crs

    if len(plot_unchanged):
        plot_unchanged.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.3, zorder=1)
    if len(plot_changed):
        plot_changed.plot(
            ax=ax,
            column="d_pct",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            legend=True,
            legend_kwds={
                "label": "Изменение цены, %",
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
            cbar_ax.set_ylabel("Изменение цены, %", fontsize=20)
            cbar_ax.tick_params(labelsize=14)
            cbar_ax.set_position([0.92, 0.15, 0.025, 0.7])
            has_colorbar = True

        for _, row in plot_changed.iterrows():
            x, y = row.geometry.centroid.coords[0]
            ax.text(
                x,
                y,
                f"{row['d_pct']:+.1f}%",
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

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if basemap_enabled and ctx_module is not None and provider_obj is not None:
        try:
            ctx_module.add_basemap(
                ax,
                source=provider_obj,
                crs=plotting_crs,
                attribution=False,
                zorder=0,
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        except Exception as exc:
            warnings.warn(f"Не удалось наложить подложку карты: {exc}", stacklevel=2)

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
        Dataset containing ``price_before`` and ``price_after`` columns.

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

    price_before = float(gdf["price_before"].sum())
    price_after = float(gdf["price_after"].sum())
    delta = price_after - price_before
    delta_pct = (price_after / price_before - 1.0) * 100 if price_before > 0 else np.nan
    return {
        "sum_before": price_before,
        "sum_after": price_after,
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
    """
    threshold_note = f" (|Δ₽| > {eps:g})" if eps > 0 else ""
    print(f"\nСтатистика по кварталам в пределах буфера{threshold_note}:")
    if len(gdf):
        print("index | до (₽) → после (₽) | Δ₽ | Δ%")
        for idx, row in gdf.iterrows():
            print(
                f"{idx} | {_fmt_rub(row['price_before'])} → {_fmt_rub(row['price_after'])} | "
                f"{_fmt_rub(row['d_rub'], signed=True)} | {_fmt_pct(row['d_pct'])}"
            )
    else:
        print("Нет кварталов, у которых стоимость изменилась выше порога.")

    print("\nСводка по кварталам с изменениями:")
    print(" • Сумма до:   ", _fmt_rub(summary["sum_before"]))
    print(" • Сумма после:", _fmt_rub(summary["sum_after"]))
    print(" • Изм., ₽:    ", _fmt_rub(summary["delta"], signed=True))
    print(" • Изм., %:    ", _fmt_pct(summary["delta_pct"]))
    print(" • Территорий: ", summary["count"])

    if total_summary is not None:
        print("\nСводка по буферу (все кварталы):")
        print(" • Сумма до:   ", _fmt_rub(total_summary["sum_before"]))
        print(" • Сумма после:", _fmt_rub(total_summary["sum_after"]))
        print(" • Изм., ₽:    ", _fmt_rub(total_summary["delta"], signed=True))
        print(" • Изм., %:    ", _fmt_pct(total_summary["delta_pct"]))
        count = total_count if total_count is not None else total_summary.get("count")
        if count is not None:
            print(" • Территорий: ", count)


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
