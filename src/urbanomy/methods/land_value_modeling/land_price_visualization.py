"""Visualization utilities for land price scenarios."""

from __future__ import annotations

from typing import Dict, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from geopandas import GeoDataFrame

from .land_data_preparation import LandDataPreparator


def plot_land_price_maps(
    *,
    blocks_pred: GeoDataFrame,
    scenario_blocks: GeoDataFrame,
    price_column: str = "price_pred",
    log_price_column: str = "y_log_pred",
    area_column: str = "site_area",
    buffer_radius_m: float = 2000.0,
    show: bool = True,
    print_summary: bool = True,
) -> Dict[str, object]:
    """Plot land price maps for scenario and context blocks.

    Parameters
    ----------
    blocks_pred : geopandas.GeoDataFrame
        Dataset containing price predictions and block geometries. The
        DataFrame is copied internally to avoid mutating the original object.
    scenario_blocks : geopandas.GeoDataFrame
        Subset of blocks that represent the project scenario. Used to identify
        the scenario footprint and compute the buffer around it.
    price_column : str, default='price_pred'
        Name of the column with price predictions in the original scale. If the
        column is missing but ``log_price_column`` is available, prices are
        derived by exponentiating the logarithmic predictions.
    log_price_column : str, default='y_log_pred'
        Column containing logarithmic prices. Used as a fallback to build
        ``price_column`` when it is absent.
    area_column : str, default='site_area'
        Column with block areas in square metres. Required for computing price
        per sotka (100 m²).
    buffer_radius_m : float, default=2000.0
        Radius of the buffer (in metres) around the scenario footprint used to
        clip context blocks.
    show : bool, default=True
        Whether to display the generated figures using ``plt.show``.
    print_summary : bool, default=True
        Whether to print aggregate price statistics to stdout.

    Returns
    -------
    dict
        Dictionary with keys ``'totals'`` and ``'figures'``. ``totals`` stores
        aggregate prices for scenario, context, and all blocks. ``figures`` is
        a tuple of Matplotlib figures corresponding to the project, context, and
        combined maps.
    """

    blocks = blocks_pred.copy()
    scenario = scenario_blocks.copy()

    if price_column not in blocks.columns:
        if log_price_column in blocks.columns:
            blocks[price_column] = np.exp(blocks[log_price_column])
        else:
            raise ValueError(
                f"Neither '{price_column}' nor '{log_price_column}' is available in blocks_pred."
            )

    if area_column not in blocks.columns:
        raise ValueError(f"The column '{area_column}' is required to compute per-sotka prices.")

    area_series = blocks[area_column].astype(float)
    blocks["price_per_sotka"] = np.where(area_series > 0, blocks[price_column] / area_series * 100, np.nan)

    if scenario.crs != blocks.crs:
        scenario = scenario.to_crs(blocks.crs)

    if "is_scn" not in blocks.columns:
        blocks["is_scn"] = LandDataPreparator.mark_scenario_blocks(blocks, scenario)

    scenario_subset = blocks[blocks["is_scn"]].copy()
    context_subset = blocks[~blocks["is_scn"]].copy()

    buffer_gdf = _make_buffer_m(scenario_subset, buffer_radius_m, target_crs=blocks.crs)
    context_clipped = gpd.clip(context_subset, buffer_gdf) if len(context_subset) else context_subset

    totals = {
        "scenario": float(scenario_subset[price_column].sum()),
        "context": float(context_subset[price_column].sum()),
        "all": float(blocks[price_column].sum()),
    }

    if print_summary:
        _print_totals(totals)

    figures: Tuple[plt.Figure, ...] = _plot_maps(
        blocks=blocks,
        scenario=scenario_subset,
        context=context_clipped,
        price_column="price_per_sotka",
        quantile_bounds=(0.05, 0.95),
        show=show,
    )

    return {"totals": totals, "figures": figures}


def _make_buffer_m(scenario: GeoDataFrame, radius_m: float, *, target_crs) -> GeoDataFrame:
    """Generate a buffer around the scenario footprint in metres.

    Parameters
    ----------
    scenario : geopandas.GeoDataFrame
        Scenario blocks whose union defines the buffer centre.
    radius_m : float
        Buffer radius in metres applied to the unified geometry.
    target_crs : Any
        Coordinate reference system for the returned geometry.

    Returns
    -------
    geopandas.GeoDataFrame
        Single-row GeoDataFrame representing the buffered scenario extent.
    """
    if scenario.empty:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)

    geometry_union = scenario.geometry.unary_union
    buffer_gdf = gpd.GeoDataFrame(geometry=[geometry_union], crs=scenario.crs)

    if buffer_gdf.crs and buffer_gdf.crs.is_geographic:
        projected = buffer_gdf.to_crs(buffer_gdf.estimate_utm_crs())
        projected["geometry"] = projected.buffer(radius_m)
        buffer_gdf = projected.to_crs(target_crs)
    else:
        buffer_gdf["geometry"] = buffer_gdf.buffer(radius_m)
        if buffer_gdf.crs != target_crs:
            buffer_gdf = buffer_gdf.to_crs(target_crs)

    return buffer_gdf


def _plot_maps(
    *,
    blocks: GeoDataFrame,
    scenario: GeoDataFrame,
    context: GeoDataFrame,
    price_column: str,
    quantile_bounds: Tuple[float, float],
    show: bool,
) -> Tuple[plt.Figure, ...]:
    """Create individual scenario, context, and combined price maps.

    Parameters
    ----------
    blocks : geopandas.GeoDataFrame
        Dataset containing price-per-sotka values.
    scenario : geopandas.GeoDataFrame
        Scenario subset used for highlighting.
    context : geopandas.GeoDataFrame
        Context blocks within the buffer.
    price_column : str
        Name of the column visualised via colour mapping.
    quantile_bounds : tuple[float, float]
        Lower and upper quantiles used to clamp the colour scale.
    show : bool
        Whether to display the figures via ``plt.show``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, ...]
        Figures for scenario vs context, context-only, and combined views.
    """
    figures = []
    vmin, vmax = _compute_color_bounds(blocks, price_column, quantile_bounds)

    figures.append(
        _plot_context_vs_scenario(
            context=context,
            scenario=scenario,
            price_column=price_column,
            vmin=vmin,
            vmax=vmax,
            title="Цена за сотку: проект (контекст ≤6 км серым)",
            show=show,
        )
    )
    figures.append(
        _plot_context_map(
            blocks=blocks,
            scenario=scenario,
            price_column=price_column,
            vmin=vmin,
            vmax=vmax,
            title="Цена за сотку: контекст ≤6 км (сценические серым)",
            show=show,
        )
    )
    figures.append(
        _plot_combined_map(
            blocks=blocks,
            scenario=scenario,
            price_column=price_column,
            vmin=vmin,
            vmax=vmax,
            title="Цена за сотку: вся территория (сценические + контекст ≤6 км)",
            show=show,
        )
    )
    return tuple(figures)


def _compute_color_bounds(
    blocks: GeoDataFrame,
    price_column: str,
    quantile_bounds: Tuple[float, float],
) -> Tuple[float, float]:
    """Derive robust colour scale bounds from price quantiles.

    Parameters
    ----------
    blocks : geopandas.GeoDataFrame
        Dataset providing the price distribution.
    price_column : str
        Column containing price values to evaluate.
    quantile_bounds : tuple[float, float]
        Lower and upper quantiles (0-1 range) used to compute bounds.

    Returns
    -------
    tuple[float, float]
        Minimum and maximum values for the colour scale.
    """
    if len(blocks):
        low, high = quantile_bounds
        vmin = blocks[price_column].quantile(low)
        vmax = blocks[price_column].quantile(high)
    else:
        vmin = np.nan
        vmax = np.nan
    return vmin, vmax


def _print_totals(totals: Dict[str, float]) -> None:
    """Display aggregate scenario and context prices in the console.

    Parameters
    ----------
    totals : dict[str, float]
        Mapping containing ``scenario``, ``context``, and ``all`` totals.
    """
    print("Суммарная стоимость:")
    print(" • Сценические кварталы:", _format_currency(totals["scenario"]))
    print(" • Контекст:", _format_currency(totals["context"]))
    print(" • Все кварталы:", _format_currency(totals["all"]))


def _format_currency(value: float) -> str:
    """Return a formatted currency string with thin-space separators.

    Parameters
    ----------
    value : float
        Numeric value to format.

    Returns
    -------
    str
        Formatted currency string expressed in roubles.
    """
    return f"{value:,.0f} ₽".replace(",", " ")


def _format_colorbar(ax) -> None:
    """Apply thousands separators to a Matplotlib colourbar axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Colorbar axes receiving the formatter.
    """
    formatter = mticker.FuncFormatter(lambda value, _: f"{int(value):,}".replace(",", " "))
    ax.yaxis.set_major_formatter(formatter)


def _plot_context_vs_scenario(
    *,
    context: GeoDataFrame,
    scenario: GeoDataFrame,
    price_column: str,
    vmin: float,
    vmax: float,
    title: str,
    show: bool,
) -> plt.Figure:
    """Plot scenario blocks atop context blocks using a shared colour scale.

    Parameters
    ----------
    context : geopandas.GeoDataFrame
        Context blocks to plot in grey.
    scenario : geopandas.GeoDataFrame
        Scenario blocks coloured by ``price_column``.
    price_column : str
        Column used for colouring scenario blocks.
    vmin, vmax : float
        Colour scale limits shared across plots.
    title : str
        Plot title.
    show : bool
        Whether to execute ``plt.show``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the scenario vs context view.
    """
    fig, ax = plt.subplots(figsize=(18, 14))
    context.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.2, zorder=1)
    if len(scenario):
        scenario.plot(
            ax=ax,
            column=price_column,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            legend=True,
            edgecolor="black",
            linewidth=0.3,
            zorder=2,
        )
    ax.set_title(title)
    ax.axis("off")

    if len(fig.axes) > 1:
        _format_colorbar(fig.axes[-1])
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def _plot_context_map(
    *,
    blocks: GeoDataFrame,
    scenario: GeoDataFrame,
    price_column: str,
    vmin: float,
    vmax: float,
    title: str,
    show: bool,
) -> plt.Figure:
    """Plot the full context map with scenario blocks highlighted in grey.

    Parameters
    ----------
    blocks : geopandas.GeoDataFrame
        Context blocks coloured by ``price_column``.
    scenario : geopandas.GeoDataFrame
        Scenario overlay shown in neutral colour.
    price_column : str
        Column used for colour mapping.
    vmin, vmax : float
        Colour scale limits shared across plots.
    title : str
        Plot title.
    show : bool
        Whether to execute ``plt.show``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the context map.
    """
    fig, ax = plt.subplots(figsize=(18, 14))
    if len(blocks):
        blocks.plot(
            ax=ax,
            column=price_column,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            legend=True,
            edgecolor="black",
            linewidth=0.3,
            zorder=1,
        )
    if len(scenario):
        scenario.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.5, zorder=2)
    ax.set_title(title, pad=10)
    ax.axis("off")

    if len(fig.axes) > 1:
        _format_colorbar(fig.axes[-1])
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def _plot_combined_map(
    *,
    blocks: GeoDataFrame,
    scenario: GeoDataFrame,
    price_column: str,
    vmin: float,
    vmax: float,
    title: str,
    show: bool,
) -> plt.Figure:
    """Plot the combined scenario and context price map.

    Parameters
    ----------
    blocks : geopandas.GeoDataFrame
        All blocks coloured by ``price_column``.
    scenario : geopandas.GeoDataFrame
        Scenario subset rendered using the same colour map.
    price_column : str
        Column used for colour mapping.
    vmin, vmax : float
        Colour scale limits shared across plots.
    title : str
        Plot title.
    show : bool
        Whether to execute ``plt.show``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the combined view.
    """
    fig, ax = plt.subplots(figsize=(18, 14))
    if len(blocks):
        blocks.plot(
            ax=ax,
            column=price_column,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            legend=True,
            edgecolor="black",
            linewidth=0.3,
            zorder=1,
        )
    if len(scenario):
        scenario.plot(
            ax=ax,
            column=price_column,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )
    ax.set_title(title, pad=10)
    ax.axis("off")

    if len(fig.axes) > 1:
        _format_colorbar(fig.axes[-1])
    plt.tight_layout()
    if show:
        plt.show()
    return fig
