"""Unified tool-calling agent for Urbanomy visualization requests."""

from __future__ import annotations

import re
from typing import Any, TypedDict

import geopandas as gpd
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from .internal.agent_utils import (
    build_tool_agent,
    extract_last_ai_message_text,
    invoke_structured_router,
)
from .models import VisualizationResult, VisualizationRouteDecision
from .prompts import VISUALIZATION_AGENT_PROMPT, VISUALIZATION_ROUTER_SYSTEM_PROMPT
from .tools import (
    make_plot_land_value_per_100m2_map_tool,
    make_plot_target_block_map_tool,
    make_plot_total_land_value_map_tool,
)


class VisualizationState(TypedDict, total=False):
    """State passed through the unified visualization graph."""

    user_request: str
    route_decision: VisualizationRouteDecision
    agent_result: dict[str, Any]
    used_tool_fallback: bool
    result: VisualizationResult


class VisualizationAgent:
    """Handle all visualization requests with one routing and tool-calling agent."""

    def __init__(
        self,
        *,
        llm: Any,
        baseline_blocks: gpd.GeoDataFrame,
        id_column: str = "id",
        show_plot: bool = True,
        figsize: tuple[float, float] = (20.0, 20.0),
        cmap: str = "coolwarm",
        edgecolor: str = "black",
        linewidth: float = 0.2,
        legend: bool = True,
        axis_off: bool = True,
        default_total_title: str = "Карта стоимости земельных участков (руб.)",
        default_unit_title: str = "Карта стоимости земельных участков за сотку (руб.)",
        target_block_title_template: str = "Изменяемый квартал  (id={target_id})",
    ) -> None:
        if llm is None:
            raise ValueError("llm is required to build VisualizationAgent.")

        self.llm = llm
        self.id_column = id_column
        self.default_total_title = default_total_title
        self.default_unit_title = default_unit_title
        self.target_block_title_template = target_block_title_template
        self._artifact_store: dict[str, Any] = {}
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
        self._plot_target_block_tool = make_plot_target_block_map_tool(
            baseline_blocks=baseline_blocks,
            artifact_store=self._artifact_store,
            id_column=id_column,
            title_template=target_block_title_template,
            show_plot=show_plot,
        )
        self._agent = build_tool_agent(
            llm=self.llm,
            tools=[
                self._plot_total_tool,
                self._plot_unit_tool,
                self._plot_target_block_tool,
            ],
            system_prompt=VISUALIZATION_AGENT_PROMPT,
        )
        self.graph = self._build_graph()

    def invoke(self, user_request: str) -> VisualizationResult:
        """Execute the visualization graph for a free-form request."""
        text = str(user_request).strip()
        if not text:
            raise ValueError("user_request cannot be empty")
        self._artifact_store.clear()
        output = self.graph.invoke({"user_request": text})
        return output["result"]

    def run(self, user_request: str) -> VisualizationResult:
        return self.invoke(user_request)

    def ask(self, user_request: str) -> VisualizationResult:
        return self.invoke(user_request)

    def __call__(self, user_request: str) -> VisualizationResult:
        return self.invoke(user_request)

    def _build_graph(self):
        graph = StateGraph(VisualizationState)
        graph.add_node("router", self._router_node)
        graph.add_node("execute_visualization", self._execute_visualization_node)
        graph.add_node("finalize", self._finalize_node)
        graph.add_edge(START, "router")
        graph.add_edge("router", "execute_visualization")
        graph.add_edge("execute_visualization", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def _router_node(self, state: VisualizationState) -> dict[str, Any]:
        user_request = state["user_request"]
        heuristic = self._heuristic_route(user_request)
        if heuristic is not None:
            return {"route_decision": heuristic}

        decision = invoke_structured_router(
            llm=self.llm,
            schema=VisualizationRouteDecision,
            system_prompt=VISUALIZATION_ROUTER_SYSTEM_PROMPT,
            user_request=user_request,
        )
        if decision is not None:
            if decision.route == "plot_target_block_map" and decision.target_id is None:
                target_id = self._extract_target_id(user_request)
                if target_id is not None:
                    decision = decision.model_copy(
                        update={
                            "target_id": target_id,
                            "price_column": self.id_column,
                            "title": self.target_block_title_template.format(target_id=target_id),
                        }
                    )
                else:
                    decision = VisualizationRouteDecision(
                        route="plot_total_land_value_map",
                        metric_kind="total_land_value",
                        price_column="land_value",
                        title=self.default_total_title,
                        reasoning="Маршрут квартала без target_id недоопределён, выбрана карта полной стоимости участка.",
                    )
            return {"route_decision": decision}

        return {
            "route_decision": VisualizationRouteDecision(
                route="plot_total_land_value_map",
                metric_kind="total_land_value",
                price_column="land_value",
                title=self.default_total_title,
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
        target_markers = (
            "target_id",
            "квартал",
            "блок",
            "block id",
            "изменяемый квартал",
            "выдели квартал",
            "покажи квартал",
        )
        target_id = self._extract_target_id(user_request)

        if any(marker in text for marker in target_markers) and target_id is not None:
            return VisualizationRouteDecision(
                route="plot_target_block_map",
                metric_kind="target_block",
                price_column=self.id_column,
                title=self.target_block_title_template.format(target_id=target_id),
                target_id=target_id,
                reasoning="В запросе указан квартал по id или target_id.",
            )
        if any(marker in text for marker in per_100m2_markers):
            return VisualizationRouteDecision(
                route="plot_land_value_per_100m2_map",
                metric_kind="land_value_per_100m2",
                price_column="land_value_per_100m2",
                title=self.default_unit_title,
                reasoning="В запросе упомянута стоимость за сотку или за 100 м2.",
            )
        if any(marker in text for marker in total_markers):
            return VisualizationRouteDecision(
                route="plot_total_land_value_map",
                metric_kind="total_land_value",
                price_column="land_value",
                title=self.default_total_title,
                reasoning="В запросе упомянута полная стоимость участка.",
            )
        return None

    def _execute_visualization_node(self, state: VisualizationState) -> dict[str, Any]:
        decision = state["route_decision"]
        self._artifact_store.pop(decision.route, None)
        execution_request = self._build_execution_request(
            user_request=state["user_request"],
            route_decision=decision,
        )
        try:
            agent_result = self._agent.invoke({"messages": [HumanMessage(content=execution_request)]})
        except Exception:
            agent_result = {}
        used_tool_fallback = False
        if decision.route not in self._artifact_store:
            used_tool_fallback = True
            self._invoke_tool_fallback(decision)
        return {"agent_result": agent_result, "used_tool_fallback": used_tool_fallback}

    def _build_execution_request(
        self,
        *,
        user_request: str,
        route_decision: VisualizationRouteDecision,
    ) -> str:
        lines = [
            user_request,
            f"Маршрут уже выбран: {route_decision.route}.",
            f"Вызови только инструмент {route_decision.route}.",
        ]
        if route_decision.target_id is not None:
            lines.append(f"Используй target_id={route_decision.target_id}.")
        return "\n\n".join(lines)

    def _invoke_tool_fallback(self, route_decision: VisualizationRouteDecision) -> None:
        if route_decision.route == "plot_total_land_value_map":
            self._plot_total_tool.invoke({})
            return
        if route_decision.route == "plot_land_value_per_100m2_map":
            self._plot_unit_tool.invoke({})
            return
        if route_decision.target_id is None:
            raise ValueError("plot_target_block_map requires target_id.")
        self._plot_target_block_tool.invoke({"target_id": route_decision.target_id})

    def _finalize_node(self, state: VisualizationState) -> dict[str, Any]:
        decision = state["route_decision"]
        artifact = self._artifact_store.get(decision.route)
        target_id = decision.target_id if decision.route == "plot_target_block_map" else None
        if artifact is not None and hasattr(artifact, "target_id"):
            target_id = int(getattr(artifact, "target_id"))
        result = VisualizationResult(
            user_request=state["user_request"],
            route=decision.route,
            metric_kind=decision.metric_kind,
            price_column=decision.price_column,
            title=artifact.title if artifact is not None else decision.title,
            reasoning=decision.reasoning,
            target_id=target_id,
            agent_message=extract_last_ai_message_text(state.get("agent_result", {})),
            used_tool_fallback=bool(state.get("used_tool_fallback", False)),
            tool_payload=artifact.tool_payload() if artifact is not None else {},
            artifact=artifact,
        )
        return {"result": result}

    @staticmethod
    def _extract_target_id(user_request: str) -> int | None:
        patterns = (
            r"target_id\s*[:=]?\s*(\d+)",
            r"\bid\s*[:=]?\s*(\d+)\b",
            r"\bквартал(?:а|у|ом)?\s*(?:с\s*)?(?:id\s*)?[:=]?\s*(\d+)\b",
            r"\bблок(?:а|у|ом)?\s*(?:с\s*)?(?:id\s*)?[:=]?\s*(\d+)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, user_request, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None


def create_visualization_agent(
    *,
    llm: Any,
    baseline_blocks: gpd.GeoDataFrame,
    id_column: str = "id",
    show_plot: bool = True,
    figsize: tuple[float, float] = (20.0, 20.0),
    cmap: str = "coolwarm",
    edgecolor: str = "black",
    linewidth: float = 0.2,
    legend: bool = True,
    axis_off: bool = True,
    default_total_title: str = "Карта стоимости земельных участков (руб.)",
    default_unit_title: str = "Карта стоимости земельных участков за сотку (руб.)",
    target_block_title_template: str = "Изменяемый квартал  (id={target_id})",
) -> VisualizationAgent:
    """Factory for the unified visualization agent."""
    return VisualizationAgent(
        llm=llm,
        baseline_blocks=baseline_blocks,
        id_column=id_column,
        show_plot=show_plot,
        figsize=figsize,
        cmap=cmap,
        edgecolor=edgecolor,
        linewidth=linewidth,
        legend=legend,
        axis_off=axis_off,
        default_total_title=default_total_title,
        default_unit_title=default_unit_title,
        target_block_title_template=target_block_title_template,
    )


def visualize_from_request(
    *,
    llm: Any,
    baseline_blocks: gpd.GeoDataFrame,
    user_request: str,
    id_column: str = "id",
    show_plot: bool = True,
    figsize: tuple[float, float] = (20.0, 20.0),
    cmap: str = "coolwarm",
    edgecolor: str = "black",
    linewidth: float = 0.2,
    legend: bool = True,
    axis_off: bool = True,
    default_total_title: str = "Карта стоимости земельных участков (руб.)",
    default_unit_title: str = "Карта стоимости земельных участков за сотку (руб.)",
    target_block_title_template: str = "Изменяемый квартал  (id={target_id})",
) -> VisualizationResult:
    """Build the agent and execute a single visualization request."""
    agent = create_visualization_agent(
        llm=llm,
        baseline_blocks=baseline_blocks,
        id_column=id_column,
        show_plot=show_plot,
        figsize=figsize,
        cmap=cmap,
        edgecolor=edgecolor,
        linewidth=linewidth,
        legend=legend,
        axis_off=axis_off,
        default_total_title=default_total_title,
        default_unit_title=default_unit_title,
        target_block_title_template=target_block_title_template,
    )
    return agent.invoke(user_request)
