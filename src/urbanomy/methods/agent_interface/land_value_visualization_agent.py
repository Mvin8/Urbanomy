"""Routing tool agent for land-value visualization tools."""

from __future__ import annotations

from typing import Any, TypedDict

import geopandas as gpd
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from ._agent_utils import build_tool_agent, extract_last_ai_message_text, invoke_structured_router
from .models import (
    LandValueVisualizationArtifact,
    LandValueVisualizationRequest,
    LandValueVisualizationResult,
    VisualizationRouteDecision,
)
from .prompts import ROUTER_SYSTEM_PROMPT, TOTAL_VALUE_AGENT_PROMPT, UNIT_VALUE_AGENT_PROMPT
from .tools import (
    make_plot_land_value_per_100m2_map_tool,
    make_plot_total_land_value_map_tool,
)


class LandValueVisualizationState(TypedDict, total=False):
    """State passed across the routing graph."""

    user_request: str
    route_decision: VisualizationRouteDecision
    agent_result: dict[str, Any]
    used_tool_fallback: bool
    result: LandValueVisualizationResult


class LandValueVisualizationRoutingAgent:
    """Route land-value visualization requests to the correct plotting tool."""

    def __init__(
        self,
        *,
        llm: Any,
        baseline_blocks: gpd.GeoDataFrame,
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
        if llm is None:
            raise ValueError("llm is required to build LandValueVisualizationRoutingAgent.")

        self.llm = llm
        self.baseline_blocks = baseline_blocks
        self.default_request_options = {
            "show_plot": show_plot,
            "figsize": figsize,
            "cmap": cmap,
            "edgecolor": edgecolor,
            "linewidth": linewidth,
            "legend": legend,
            "axis_off": axis_off,
            "default_total_title": default_total_title,
            "default_unit_title": default_unit_title,
        }
        self._artifact_store: dict[str, LandValueVisualizationArtifact] = {}
        self._plot_total_tool = make_plot_total_land_value_map_tool(
            baseline_blocks=baseline_blocks,
            artifact_store=self._artifact_store,
            default_title=default_total_title,
            default_show_plot=show_plot,
            default_figsize=figsize,
            default_cmap=cmap,
            default_edgecolor=edgecolor,
            default_linewidth=linewidth,
            default_legend=legend,
            default_axis_off=axis_off,
        )
        self._plot_unit_tool = make_plot_land_value_per_100m2_map_tool(
            baseline_blocks=baseline_blocks,
            artifact_store=self._artifact_store,
            default_title=default_unit_title,
            default_show_plot=show_plot,
            default_figsize=figsize,
            default_cmap=cmap,
            default_edgecolor=edgecolor,
            default_linewidth=linewidth,
            default_legend=legend,
            default_axis_off=axis_off,
        )
        self._total_agent = build_tool_agent(
            llm=self.llm,
            tools=[self._plot_total_tool],
            system_prompt=TOTAL_VALUE_AGENT_PROMPT,
        )
        self._unit_agent = build_tool_agent(
            llm=self.llm,
            tools=[self._plot_unit_tool],
            system_prompt=UNIT_VALUE_AGENT_PROMPT,
        )
        self.graph = self._build_graph()

    def invoke(self, user_request: str) -> LandValueVisualizationResult:
        """Execute the routing graph for a natural-language visualization request."""
        request = self._coerce_request(user_request)
        self._artifact_store.clear()
        output = self.graph.invoke({"user_request": request.user_request})
        return output["result"]

    def run(self, user_request: str) -> LandValueVisualizationResult:
        return self.invoke(user_request)

    def ask(self, user_request: str) -> LandValueVisualizationResult:
        return self.invoke(user_request)

    def __call__(self, user_request: str) -> LandValueVisualizationResult:
        return self.invoke(user_request)

    def _coerce_request(self, user_request: str) -> LandValueVisualizationRequest:
        return LandValueVisualizationRequest(user_request=user_request, **self.default_request_options)

    def _build_graph(self):
        graph = StateGraph(LandValueVisualizationState)
        graph.add_node("router", self._router_node)
        graph.add_node("plot_total", self._plot_total_node)
        graph.add_node("plot_unit", self._plot_unit_node)
        graph.add_node("finalize", self._finalize_node)
        graph.add_edge(START, "router")
        graph.add_conditional_edges(
            "router",
            self._route_after_router,
            {
                "plot_total": "plot_total",
                "plot_unit": "plot_unit",
            },
        )
        graph.add_edge("plot_total", "finalize")
        graph.add_edge("plot_unit", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def _router_node(self, state: LandValueVisualizationState) -> dict[str, Any]:
        user_request = state["user_request"]
        heuristic = self._heuristic_route(user_request)
        if heuristic is not None:
            return {"route_decision": heuristic}

        decision = invoke_structured_router(
            llm=self.llm,
            schema=VisualizationRouteDecision,
            system_prompt=ROUTER_SYSTEM_PROMPT,
            user_request=user_request,
        )
        if decision is not None:
            return {"route_decision": decision}

        return {
            "route_decision": VisualizationRouteDecision(
                route="plot_total_land_value_map",
                metric_kind="total_land_value",
                price_column="land_value",
                title=self.default_request_options["default_total_title"],
                reasoning="Точный маршрут не распознан, выбрана визуализация полной стоимости участка.",
            )
        }

    def _heuristic_route(self, user_request: str) -> VisualizationRouteDecision | None:
        text = user_request.lower()
        per_100m2_markers = (
            "за сотку",
            "сотк",
            "100 м2",
            "100м2",
            "100 м²",
            "100м²",
            "удельн",
            "per 100",
            "per_100",
        )
        total_markers = (
            "полную стоимость",
            "общую стоимость",
            "всего участка",
            "стоимость участка",
            "стоимость земельных участков",
            "total land value",
            "полная стоимость",
        )

        if any(marker in text for marker in per_100m2_markers):
            return VisualizationRouteDecision(
                route="plot_land_value_per_100m2_map",
                metric_kind="land_value_per_100m2",
                price_column="land_value_per_100m2",
                title=self.default_request_options["default_unit_title"],
                reasoning="В запросе упомянута стоимость за сотку или за 100 м2.",
            )
        if any(marker in text for marker in total_markers):
            return VisualizationRouteDecision(
                route="plot_total_land_value_map",
                metric_kind="total_land_value",
                price_column="land_value",
                title=self.default_request_options["default_total_title"],
                reasoning="В запросе упомянута полная стоимость участка.",
            )
        return None

    def _route_after_router(self, state: LandValueVisualizationState) -> str:
        return "plot_unit" if state["route_decision"].route == "plot_land_value_per_100m2_map" else "plot_total"

    def _plot_total_node(self, state: LandValueVisualizationState) -> dict[str, Any]:
        agent_result, used_tool_fallback = self._run_subagent(
            agent=self._total_agent,
            tool_name="plot_total_land_value_map",
            fallback_input={},
            user_request=state["user_request"],
        )
        return {"agent_result": agent_result, "used_tool_fallback": used_tool_fallback}

    def _plot_unit_node(self, state: LandValueVisualizationState) -> dict[str, Any]:
        agent_result, used_tool_fallback = self._run_subagent(
            agent=self._unit_agent,
            tool_name="plot_land_value_per_100m2_map",
            fallback_input={},
            user_request=state["user_request"],
        )
        return {"agent_result": agent_result, "used_tool_fallback": used_tool_fallback}

    def _run_subagent(
        self,
        *,
        agent: Any,
        tool_name: str,
        fallback_input: dict[str, Any],
        user_request: str,
    ) -> tuple[dict[str, Any], bool]:
        self._artifact_store.pop(tool_name, None)
        try:
            agent_result = agent.invoke({"messages": [HumanMessage(content=user_request)]})
        except Exception:
            agent_result = {}
        used_tool_fallback = False
        if tool_name not in self._artifact_store:
            used_tool_fallback = True
            if tool_name == "plot_total_land_value_map":
                self._plot_total_tool.invoke(fallback_input)
            else:
                self._plot_unit_tool.invoke(fallback_input)
        return agent_result, used_tool_fallback

    def _finalize_node(self, state: LandValueVisualizationState) -> dict[str, Any]:
        decision = state["route_decision"]
        artifact = self._artifact_store.get(decision.route)
        result = LandValueVisualizationResult(
            user_request=state["user_request"],
            route=decision.route,
            metric_kind=decision.metric_kind,
            price_column=decision.price_column,
            title=artifact.title if artifact is not None else decision.title,
            reasoning=decision.reasoning,
            agent_message=extract_last_ai_message_text(state.get("agent_result", {})),
            used_tool_fallback=bool(state.get("used_tool_fallback", False)),
            tool_payload=artifact.tool_payload() if artifact is not None else {},
            artifact=artifact,
        )
        return {"result": result}


def create_land_value_visualization_agent(
    *,
    llm: Any,
    baseline_blocks: gpd.GeoDataFrame,
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
    """Factory for the routing visualization agent."""
    return LandValueVisualizationRoutingAgent(
        llm=llm,
        baseline_blocks=baseline_blocks,
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
    """Build the agent and execute a single free-form visualization request."""
    agent = create_land_value_visualization_agent(
        llm=llm,
        baseline_blocks=baseline_blocks,
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
