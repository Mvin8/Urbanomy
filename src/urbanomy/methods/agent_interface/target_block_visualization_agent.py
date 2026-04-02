"""Legacy wrapper around the unified visualization agent for target-block maps."""

from __future__ import annotations

from typing import Any

import geopandas as gpd

from .models import TargetBlockVisualizationResult
from .visualization_agent import VisualizationAgent, create_visualization_agent


class TargetBlockVisualizationAgent:
    """Backward-compatible wrapper exposing only target-block visualization."""

    def __init__(
        self,
        *,
        llm: Any,
        baseline_blocks: gpd.GeoDataFrame,
        id_column: str = "id",
        show_plot: bool = True,
        title_template: str = "Изменяемый квартал  (id={target_id})",
    ) -> None:
        self._agent = create_visualization_agent(
            llm=llm,
            baseline_blocks=baseline_blocks,
            id_column=id_column,
            show_plot=show_plot,
            target_block_title_template=title_template,
        )

    def invoke(self, user_request: str) -> TargetBlockVisualizationResult:
        """Execute a target-block visualization request through the unified agent."""
        result = self._agent.invoke(user_request)
        if result.route != "plot_target_block_map" or result.target_id is None:
            raise ValueError("TargetBlockVisualizationAgent supports only target-block requests.")
        return TargetBlockVisualizationResult(
            user_request=result.user_request,
            route="plot_target_block_map",
            target_id=result.target_id,
            title=result.title,
            reasoning=result.reasoning,
            agent_message=result.agent_message,
            used_tool_fallback=result.used_tool_fallback,
            tool_payload=result.tool_payload,
            artifact=result.artifact,
        )

    def run(self, user_request: str) -> TargetBlockVisualizationResult:
        return self.invoke(user_request)

    def ask(self, user_request: str) -> TargetBlockVisualizationResult:
        return self.invoke(user_request)

    def __call__(self, user_request: str) -> TargetBlockVisualizationResult:
        return self.invoke(user_request)

    @property
    def graph(self):
        return self._agent.graph

    @property
    def unified_agent(self) -> VisualizationAgent:
        return self._agent


def create_target_block_visualization_agent(
    *,
    llm: Any,
    baseline_blocks: gpd.GeoDataFrame,
    id_column: str = "id",
    show_plot: bool = True,
    title_template: str = "Изменяемый квартал  (id={target_id})",
) -> TargetBlockVisualizationAgent:
    """Factory for the legacy target-block visualization wrapper."""
    return TargetBlockVisualizationAgent(
        llm=llm,
        baseline_blocks=baseline_blocks,
        id_column=id_column,
        show_plot=show_plot,
        title_template=title_template,
    )
