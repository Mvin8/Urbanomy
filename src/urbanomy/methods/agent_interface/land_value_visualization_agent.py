"""Legacy wrapper around the unified visualization agent for land-value maps."""

from __future__ import annotations

from typing import Any

import geopandas as gpd

from .models import LandValuePredictionConfig, LandValueVisualizationResult
from .visualization_agent import VisualizationAgent, create_visualization_agent


class LandValueVisualizationRoutingAgent:
    """Backward-compatible wrapper exposing only land-value visualization routes."""

    def __init__(
        self,
        *,
        llm: Any,
        baseline_blocks: gpd.GeoDataFrame,
        prediction_config: LandValuePredictionConfig | None = None,
        show_plot: bool = True,
        figsize: tuple[float, float] = (20.0, 20.0),
        cmap: str = "coolwarm",
        edgecolor: str = "black",
        linewidth: float = 0.2,
        legend: bool = True,
        axis_off: bool = True,
        default_total_title: str = "Карта стоимости земельных участков (руб.)",
        default_unit_title: str = "Карта стоимости земельных участков за сотку (руб.)",
    ) -> None:
        self._agent = create_visualization_agent(
            llm=llm,
            baseline_blocks=baseline_blocks,
            prediction_config=prediction_config,
            show_plot=show_plot,
            figsize=figsize,
            cmap=cmap,
            edgecolor=edgecolor,
            linewidth=linewidth,
            legend=legend,
            axis_off=axis_off,
            default_total_title=default_total_title,
            default_unit_title=default_unit_title,
        )

    def invoke(self, user_request: str) -> LandValueVisualizationResult:
        """Execute a land-value visualization request through the unified agent."""
        result = self._agent.invoke(user_request)
        if result.route in {"predict_land_value", "plot_target_block_map"}:
            raise ValueError("LandValueVisualizationRoutingAgent supports only land-value maps.")
        return LandValueVisualizationResult(
            user_request=result.user_request,
            route=result.route,
            metric_kind=result.metric_kind,
            price_column=result.price_column,
            title=result.title,
            reasoning=result.reasoning,
            agent_message=result.agent_message,
            used_tool_fallback=result.used_tool_fallback,
            tool_payload=result.tool_payload,
            artifact=result.artifact,
        )

    def run(self, user_request: str) -> LandValueVisualizationResult:
        return self.invoke(user_request)

    def ask(self, user_request: str) -> LandValueVisualizationResult:
        return self.invoke(user_request)

    def __call__(self, user_request: str) -> LandValueVisualizationResult:
        return self.invoke(user_request)

    @property
    def graph(self):
        return self._agent.graph

    @property
    def unified_agent(self) -> VisualizationAgent:
        return self._agent


def create_land_value_visualization_agent(
    *,
    llm: Any,
    baseline_blocks: gpd.GeoDataFrame,
    prediction_config: LandValuePredictionConfig | None = None,
    show_plot: bool = True,
    figsize: tuple[float, float] = (20.0, 20.0),
    cmap: str = "coolwarm",
    edgecolor: str = "black",
    linewidth: float = 0.2,
    legend: bool = True,
    axis_off: bool = True,
    default_total_title: str = "Карта стоимости земельных участков (руб.)",
    default_unit_title: str = "Карта стоимости земельных участков за сотку (руб.)",
) -> LandValueVisualizationRoutingAgent:
    """Factory for the legacy land-value visualization wrapper."""
    return LandValueVisualizationRoutingAgent(
        llm=llm,
        baseline_blocks=baseline_blocks,
        prediction_config=prediction_config,
        show_plot=show_plot,
        figsize=figsize,
        cmap=cmap,
        edgecolor=edgecolor,
        linewidth=linewidth,
        legend=legend,
        axis_off=axis_off,
        default_total_title=default_total_title,
        default_unit_title=default_unit_title,
    )


def visualize_land_value_from_request(
    *,
    llm: Any,
    baseline_blocks: gpd.GeoDataFrame,
    user_request: str,
    prediction_config: LandValuePredictionConfig | None = None,
    show_plot: bool = True,
    figsize: tuple[float, float] = (20.0, 20.0),
    cmap: str = "coolwarm",
    edgecolor: str = "black",
    linewidth: float = 0.2,
    legend: bool = True,
    axis_off: bool = True,
    default_total_title: str = "Карта стоимости земельных участков (руб.)",
    default_unit_title: str = "Карта стоимости земельных участков за сотку (руб.)",
) -> LandValueVisualizationResult:
    """Build the legacy wrapper and execute one request."""
    agent = create_land_value_visualization_agent(
        llm=llm,
        baseline_blocks=baseline_blocks,
        prediction_config=prediction_config,
        show_plot=show_plot,
        figsize=figsize,
        cmap=cmap,
        edgecolor=edgecolor,
        linewidth=linewidth,
        legend=legend,
        axis_off=axis_off,
        default_total_title=default_total_title,
        default_unit_title=default_unit_title,
    )
    return agent.invoke(user_request)
