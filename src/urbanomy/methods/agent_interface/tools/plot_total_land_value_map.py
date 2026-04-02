"""Tool for plotting total land value by parcel."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
from langchain_core.tools import tool

from .internal.plotting import build_tool_payload, render_land_value_map


def make_plot_total_land_value_map_tool(
    *,
    baseline_blocks: gpd.GeoDataFrame,
    artifact_store: dict[str, Any],
    default_title: str = "Карта стоимости земельных участков (руб.)",
    default_show_plot: bool = True,
    default_figsize: tuple[float, float] = (20.0, 20.0),
    default_cmap: str = "coolwarm",
    default_edgecolor: str = "black",
    default_linewidth: float = 0.2,
    default_legend: bool = True,
    default_axis_off: bool = True,
):
    """Create a LangChain tool bound to a concrete baseline GeoDataFrame."""

    @tool("plot_total_land_value_map")
    def plot_total_land_value_map() -> dict[str, Any]:
        """What the tool does.

        Builds a choropleth map of total land value for all blocks in
        ``baseline_blocks`` using the fixed Urbanomy notebook style.

        When to use this tool.

        Use this tool when the user asks to show, plot, draw, or visualize the
        total value of land parcels or the full value of each parcel.

        Args:
            None.

        Returns:
            dict[str, Any]: Compact metadata about the created plot, including
            tool name, metric kind, price column, title, and number of plotted
            rows.

        Restrictions:
            Uses the fixed ``land_value`` column and fixed notebook-style
            formatting. Does not accept custom styling or filtering arguments.
        """
        artifact = render_land_value_map(
            baseline_blocks=baseline_blocks,
            price_column="land_value",
            metric_kind="total_land_value",
            tool_name="plot_total_land_value_map",
            title=default_title,
            figsize=default_figsize,
            cmap=default_cmap,
            edgecolor=default_edgecolor,
            linewidth=default_linewidth,
            legend=default_legend,
            axis_off=default_axis_off,
            show_plot=default_show_plot,
        )
        artifact_store["plot_total_land_value_map"] = artifact
        return build_tool_payload(artifact)

    return plot_total_land_value_map
