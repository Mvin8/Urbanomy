"""Scenario tools for adjusting block indicators and analysing impact."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from matplotlib.transforms import offset_copy

from .constants import CATEGORICAL_FEATURES, ORIGINAL_FEATURES, RADIUS_LIST
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
        self._blocks = blocks

    def apply(self, target_idx: int, changes: Mapping[str, object]) -> GeoDataFrame:
        """Return a modified copy of the blocks with updated TEP values.

        Parameters
        ----------
        target_idx : int
            Index of the block that should be modified.
        changes : Mapping[str, object]
            Dictionary of field updates (e.g. ``{"land_use": "RESIDENTIAL"}``).

        Returns
        -------
        geopandas.GeoDataFrame
            Modified copy of the original blocks DataFrame.
        """

        df = self._blocks.copy()
        if target_idx not in df.index:
            raise KeyError(f"Block index {target_idx} is not present in the dataset.")

        row = df.loc[target_idx].to_dict()
        row.update(changes)

        site_area = float(row.get("site_area", df.at[target_idx, "site_area"]))

        build = float(row.get("build_floor_area", df.at[target_idx, "build_floor_area"]))
        live = float(row.get("living_area", df.at[target_idx, "living_area"]))
        non = row.get("non_living_area", np.nan)

        if not np.isfinite(non):
            if "build_floor_area" in changes and "living_area" in changes:
                non = max(build - live, 0.0)
            elif ("mxi" in changes) or ("mxi" in row):
                mxi = float(row.get("mxi", df.at[target_idx, "mxi"]))
                non = max(mxi * build, 0.0)
            else:
                non = float(df.at[target_idx, "non_living_area"])

        if ("non_living_area" in changes and "living_area" in changes) and ("build_floor_area" not in changes):
            build = live + float(non)

        row["build_floor_area"] = build
        row["living_area"] = live
        row["non_living_area"] = float(non)

        footprint = float(row.get("footprint_area", df.at[target_idx, "footprint_area"]))
        pop = float(row.get("population", df.at[target_idx, "population"]))

        row["fsi"] = build / site_area if site_area > 0 else np.nan
        row["gsi"] = footprint / site_area if site_area > 0 else np.nan
        row["mxi"] = (row["non_living_area"] / build) if build > 0 else 0.0
        row["l"] = (build / pop) if pop > 0 else np.nan
        row["osr"] = (site_area - footprint) / site_area if site_area > 0 else np.nan
        row["share_living"] = live / site_area if site_area > 0 else np.nan
        row["share_non_living"] = row["non_living_area"] / site_area if site_area > 0 else np.nan

        if "residential" in changes and "share" not in changes:
            row["share"] = float(row["residential"])

        for key, value in row.items():
            df.at[target_idx, key] = value

        return df


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
    show : bool, default=True
        Display the generated figure using Matplotlib.
    print_summary : bool, default=True
        Whether to print textual statistics.

    Returns
    -------
    dict
        Dictionary containing ``'map'`` with the clipped GeoDataFrame,
        ``'fig'`` with the Matplotlib figure, and ``'summary'`` with aggregate
        statistics.
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
        show=show,
    )

    summary = _summarise_changes(clipped)
    if print_summary:
        _print_summary(clipped, summary)

    return {"map": clipped, "fig": fig, "summary": summary}


def _build_buffer(blocks: GeoDataFrame, target_idx: int, radius_m: float) -> GeoDataFrame:
    tgt_geom = blocks.loc[target_idx, "geometry"]
    buf = gpd.GeoSeries([tgt_geom], crs=blocks.crs)
    if buf.crs and buf.crs.is_geographic:
        utm = buf.estimate_utm_crs()
        buf = buf.to_crs(utm).buffer(radius_m).to_crs(blocks.crs)
    else:
        buf = buf.buffer(radius_m)
    return gpd.GeoDataFrame(geometry=buf, crs=blocks.crs)


def _plot_change_map(
    *,
    changed: GeoDataFrame,
    unchanged: GeoDataFrame,
    target_geometry,
    vmin: float,
    vmax: float,
    show: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(25, 20))
    if len(unchanged):
        unchanged.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.3, zorder=1)
    if len(changed):
        changed.plot(
            ax=ax,
            column="d_pct",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            legend=True,
            edgecolor="black",
            linewidth=0.4,
            zorder=2,
        )

        for _, row in changed.iterrows():
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
                str(row.get("land_use", "missing")),
                ha="center",
                va="center",
                fontsize=6,
                color="black",
                transform=offset_copy(ax.transData, fig=fig, y=-10, units="points"),
                zorder=4,
            )

    gpd.GeoDataFrame(geometry=[target_geometry], crs=changed.crs or unchanged.crs).boundary.plot(
        ax=ax,
        edgecolor="red",
        linewidth=2.0,
        zorder=3,
    )

    ax.set_title("Изменение цены в процентах (after − before)", pad=12)
    ax.axis("off")
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def _summarise_changes(gdf: GeoDataFrame) -> Dict[str, float]:
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


def _print_summary(gdf: GeoDataFrame, summary: Dict[str, float]) -> None:
    print("\nСтатистика по кварталам в пределах буфера:")
    print("index | до (₽) → после (₽) | Δ₽ | Δ%")
    for idx, row in gdf.iterrows():
        print(
            f"{idx} | {_fmt_rub(row['price_before'])} → {_fmt_rub(row['price_after'])} | "
            f"{_fmt_rub(row['d_rub'], signed=True)} | {_fmt_pct(row['d_pct'])}"
        )

    print("\nСводка по буферу:")
    print(" • Сумма до:   ", _fmt_rub(summary["sum_before"]))
    print(" • Сумма после:", _fmt_rub(summary["sum_after"]))
    print(" • Изм., ₽:    ", _fmt_rub(summary["delta"], signed=True))
    print(" • Изм., %:    ", _fmt_pct(summary["delta_pct"]))
    print(" • Территорий: ", summary["count"])


def _fmt_rub(value: float, *, signed: bool = False, digits: int = 0) -> str:
    try:
        value = float(value)
        if not np.isfinite(value):
            return "—"
        formatted = f"{value:+,.{digits}f}" if signed else f"{value:,.{digits}f}"
        return formatted.replace(",", " ") + " ₽"
    except Exception:
        return "—"


def _fmt_pct(value: float, digits: int = 1) -> str:
    try:
        value = float(value)
        if not np.isfinite(value):
            return "—"
        return f"{value:+.{digits}f}%"
    except Exception:
        return "—"
