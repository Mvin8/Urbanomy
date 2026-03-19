"""Shared plotting helpers for Urbanomy agent tools."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from ...models import LandValueVisualizationArtifact, TargetBlockVisualizationArtifact


def render_land_value_map(
    *,
    baseline_blocks: gpd.GeoDataFrame,
    price_column: str,
    metric_kind: str,
    tool_name: str,
    title: str,
    figsize: tuple[float, float] = (20.0, 20.0),
    cmap: str = "coolwarm",
    edgecolor: str = "black",
    linewidth: float = 0.2,
    legend: bool = True,
    axis_off: bool = True,
    show_plot: bool = True,
) -> LandValueVisualizationArtifact:
    """Render a choropleth in the same style as the notebook workflow."""
    if not isinstance(baseline_blocks, gpd.GeoDataFrame):
        raise TypeError("baseline_blocks must be a geopandas.GeoDataFrame.")
    if baseline_blocks.empty:
        raise ValueError("baseline_blocks is empty. Nothing to plot.")
    if baseline_blocks.geometry is None:
        raise ValueError("baseline_blocks must provide a geometry column.")
    if price_column not in baseline_blocks.columns:
        raise KeyError(
            f"Column '{price_column}' is missing in baseline_blocks. "
            f"Available columns include: {', '.join(map(str, baseline_blocks.columns[:20]))}"
        )

    working = baseline_blocks.copy()
    working[price_column] = pd.to_numeric(working[price_column], errors="coerce")
    if working[price_column].notna().sum() == 0:
        raise ValueError(f"Column '{price_column}' does not contain numeric values suitable for plotting.")

    ax = working.plot(
        column=price_column,
        legend=legend,
        figsize=figsize,
        cmap=cmap,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    if axis_off:
        ax.set_axis_off()
    plt.title(title, fontsize=16)
    if show_plot:
        plt.show()
    fig = ax.get_figure()

    return LandValueVisualizationArtifact(
        tool_name=tool_name,
        metric_kind=metric_kind,
        price_column=price_column,
        title=title,
        rows_plotted=len(working),
        figure=fig,
        axis=ax,
    )


def resolve_tool_title(*, title: str | None, default_title: str) -> str:
    """Return a cleaned title with a sensible default."""
    if title is None:
        return default_title
    cleaned = str(title).strip()
    return cleaned or default_title


def build_tool_payload(artifact: LandValueVisualizationArtifact) -> dict[str, Any]:
    """Convert the full artifact to a compact payload suitable for tool output."""
    return artifact.tool_payload()


def render_target_block_map(
    *,
    baseline_blocks: gpd.GeoDataFrame,
    target_id: int,
    id_column: str = "id",
    title: str,
    show_plot: bool = True,
) -> TargetBlockVisualizationArtifact:
    """Render a highlighted target block in the same style as the notebook workflow."""
    if not isinstance(baseline_blocks, gpd.GeoDataFrame):
        raise TypeError("baseline_blocks must be a geopandas.GeoDataFrame.")
    if baseline_blocks.empty:
        raise ValueError("baseline_blocks is empty. Nothing to plot.")
    if baseline_blocks.geometry is None:
        raise ValueError("baseline_blocks must provide a geometry column.")
    if id_column not in baseline_blocks.columns:
        raise KeyError(f"Column '{id_column}' is missing in baseline_blocks.")

    target_block = baseline_blocks.loc[baseline_blocks[id_column] == target_id]
    if target_block.empty:
        raise ValueError(f"Block with {id_column}={target_id} was not found in baseline_blocks.")

    fig, ax = plt.subplots(figsize=(25, 35))
    baseline_blocks.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.6)
    target_block.plot(ax=ax, color="none", edgecolor="gold", linewidth=5.5)
    target_block.centroid.plot(ax=ax, color="red", markersize=30, zorder=3)
    ax.set_title(title)
    ax.axis("off")
    if show_plot:
        plt.show()

    return TargetBlockVisualizationArtifact(
        tool_name="plot_target_block_map",
        target_id=int(target_id),
        title=title,
        rows_plotted=len(baseline_blocks),
        figure=fig,
        axis=ax,
    )


def build_target_block_tool_payload(artifact: TargetBlockVisualizationArtifact) -> dict[str, Any]:
    """Convert a target-block artifact to a compact payload suitable for tool output."""
    return artifact.tool_payload()
