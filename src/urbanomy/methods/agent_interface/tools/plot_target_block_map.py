"""Tool for highlighting a target block by id."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ._plotting import build_target_block_tool_payload, render_target_block_map


class PlotTargetBlockMapInput(BaseModel):
    """Input schema for the target block map tool."""

    model_config = ConfigDict(extra="forbid")

    target_id: int = Field(description="Block id from baseline_blocks['id'] to highlight on the map.")

    @field_validator("target_id", mode="before")
    @classmethod
    def _coerce_target_id(cls, value: Any) -> int:
        if value is None:
            raise ValueError("target_id is required")
        return int(value)


def make_plot_target_block_map_tool(
    *,
    baseline_blocks: gpd.GeoDataFrame,
    artifact_store: dict[str, Any],
    id_column: str = "id",
    title_template: str = "Изменяемый квартал  (id={target_id})",
    show_plot: bool = True,
):
    """Create a LangChain tool bound to a concrete baseline GeoDataFrame."""

    @tool("plot_target_block_map", args_schema=PlotTargetBlockMapInput)
    def plot_target_block_map(target_id: int) -> dict[str, Any]:
        """What the tool does.

        Highlights one target block on the map by its ``target_id`` using the
        fixed Urbanomy notebook style: grey background blocks, gold outline for
        the selected block, and a red centroid marker.

        When to use this tool.

        Use this tool when the user asks to show, plot, draw, or visualize a
        specific block, quarter, or scenario target identified by ``id`` or
        ``target_id``.

        Args:
            target_id: Integer block identifier from ``baseline_blocks[id_column]``.

        Returns:
            dict[str, Any]: Compact metadata about the created plot, including
            tool name, target id, title, and number of plotted rows.

        Restrictions:
            Requires that the requested ``target_id`` exists in
            ``baseline_blocks``. Uses fixed notebook-style formatting and does
            not support custom styling arguments.
        """
        artifact = render_target_block_map(
            baseline_blocks=baseline_blocks,
            target_id=target_id,
            id_column=id_column,
            title=title_template.format(target_id=target_id),
            show_plot=show_plot,
        )
        artifact_store["plot_target_block_map"] = artifact
        return build_target_block_tool_payload(artifact)

    return plot_target_block_map
