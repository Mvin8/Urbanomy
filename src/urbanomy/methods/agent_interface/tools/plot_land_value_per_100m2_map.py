"""Tool for plotting land value per 100 m2."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
from langchain_core.tools import tool

from .internal.plotting import build_tool_payload, render_land_value_map


def make_plot_land_value_per_100m2_map_tool(
    *,
    baseline_blocks: gpd.GeoDataFrame,
    artifact_store: dict[str, Any],
    default_title: str = "Карта стоимости земельных участков за сотку (руб.)",
    default_show_plot: bool = True,
    default_figsize: tuple[float, float] = (20.0, 20.0),
    default_cmap: str = "coolwarm",
    default_edgecolor: str = "black",
    default_linewidth: float = 0.2,
    default_legend: bool = True,
    default_axis_off: bool = True,
):
    """Create a LangChain tool bound to a concrete baseline GeoDataFrame."""

    @tool("plot_land_value_per_100m2_map")
    def plot_land_value_per_100m2_map() -> dict[str, Any]:
        """What the tool does.

        Builds a choropleth map of land value per 100 square meters for all
        blocks in ``baseline_blocks`` using the fixed Urbanomy notebook style.

        When to use this tool.

        Use this tool when the user asks to show, plot, draw, or visualize land
        value per 100 m2, per sotka, or unit land value.

        Args:
            None.

        Returns:
            dict[str, Any]: Compact metadata about the created plot, including
            tool name, metric kind, price column, title, and number of plotted
            rows.

        Restrictions:
            Uses the fixed ``land_value_per_100m2`` column and fixed
            notebook-style formatting. Does not accept custom styling or
            filtering arguments.
        """
        artifact = render_land_value_map(
            baseline_blocks=baseline_blocks,
            price_column="land_value_per_100m2",
            metric_kind="land_value_per_100m2",
            tool_name="plot_land_value_per_100m2_map",
            title=default_title,
            figsize=default_figsize,
            cmap=default_cmap,
            edgecolor=default_edgecolor,
            linewidth=default_linewidth,
            legend=default_legend,
            axis_off=default_axis_off,
            show_plot=default_show_plot,
        )
        artifact_store["plot_land_value_per_100m2_map"] = artifact
        return build_tool_payload(artifact)

    return plot_land_value_per_100m2_map
