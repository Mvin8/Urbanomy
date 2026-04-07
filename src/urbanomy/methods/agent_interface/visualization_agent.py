"""Unified tool-calling agent for Urbanomy visualization requests."""

from __future__ import annotations

import json
from typing import Any, TypedDict

import geopandas as gpd
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from .internal.common.agent_utils import (
    build_tool_agent,
    extract_last_ai_message_text,
    extract_last_tool_message,
    invoke_structured_router,
)
from .internal.common.domain_contracts import ToolDescriptor, describe_tool
from .internal.common.request_parsing import extract_target_id
from .internal.visualization.metadata import VISUALIZATION_CAPABILITY_LINES
from .models import LandValuePredictionConfig, VisualizationResult, VisualizationRouteDecision
from .prompts import VISUALIZATION_AGENT_PROMPT, VISUALIZATION_ROUTER_SYSTEM_PROMPT
from .tools import (
    make_predict_land_value_tool,
    make_plot_land_value_per_100m2_map_tool,
    make_plot_target_block_map_tool,
    make_plot_total_land_value_map_tool,
)


class VisualizationState(TypedDict, total=False):
    """State passed through the unified visualization graph."""

    user_request: str
    route_decision: dict[str, Any]
    agent_message: str
    used_tool_fallback: bool
    tool_payload: dict[str, Any]
    result_payload: dict[str, Any]


class VisualizationAgent:
    """Handle all visualization requests with one routing and tool-calling agent."""

    _LLM_ROUTE_CONFIDENCE_THRESHOLD = 0.65
    _LLM_ROUTE_MIN_CONFIDENCE = 0.35

    def __init__(
        self,
        *,
        llm: Any,
        baseline_blocks: gpd.GeoDataFrame,
        prediction_config: LandValuePredictionConfig | None = None,
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
        self._prediction_config = prediction_config
        self._artifact_store: dict[str, Any] = {}
        self._predict_tool = (
            make_predict_land_value_tool(
                baseline_blocks=baseline_blocks,
                prediction_config=prediction_config,
            )
            if prediction_config is not None
            else None
        )
        self._plot_total_tool = make_plot_total_land_value_map_tool(
            baseline_blocks=baseline_blocks,
            artifact_store=self._artifact_store,
            prediction_config=prediction_config,
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
            prediction_config=prediction_config,
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
                tool
                for tool in (
                    self._predict_tool,
                    self._plot_total_tool,
                    self._plot_unit_tool,
                    self._plot_target_block_tool,
                )
                if tool is not None
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
        result_payload = dict(output["result_payload"])
        artifact = self._artifact_store.get(str(result_payload["route"]))
        return VisualizationResult(
            **result_payload,
            artifact=artifact,
        )

    def run(self, user_request: str) -> VisualizationResult:
        return self.invoke(user_request)

    def ask(self, user_request: str) -> VisualizationResult:
        return self.invoke(user_request)

    def __call__(self, user_request: str) -> VisualizationResult:
        return self.invoke(user_request)

    def available_tools(self) -> tuple[Any, ...]:
        """Return tool instances owned by the visualization domain."""
        return (
            *(() if self._predict_tool is None else (self._predict_tool,)),
            self._plot_total_tool,
            self._plot_unit_tool,
            self._plot_target_block_tool,
        )

    def tool_descriptors(self) -> list[ToolDescriptor]:
        """Return compact user-facing descriptions of domain tools."""
        return [describe_tool(tool) for tool in self.available_tools()]

    @staticmethod
    def capability_lines() -> list[str]:
        """Return user-facing capabilities exposed by this domain agent."""
        return list(VISUALIZATION_CAPABILITY_LINES)

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
        decision = invoke_structured_router(
            llm=self.llm,
            schema=VisualizationRouteDecision,
            system_prompt=VISUALIZATION_ROUTER_SYSTEM_PROMPT,
            user_request=user_request,
        )
        heuristic = self._heuristic_route(user_request)
        if decision is not None:
            decision = self._normalize_llm_decision(decision, user_request=user_request)
            if self._should_accept_llm_route(decision):
                return {"route_decision": decision.model_dump()}
            if heuristic is not None:
                return {"route_decision": heuristic.model_dump()}
            if float(decision.confidence) >= self._LLM_ROUTE_MIN_CONFIDENCE:
                return {"route_decision": decision.model_dump()}
        elif heuristic is not None:
            return {"route_decision": heuristic.model_dump()}

        if heuristic is not None:
            return {"route_decision": heuristic.model_dump()}

        return {"route_decision": self._default_route_decision().model_dump()}

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
        prediction_markers = (
            "вычисли",
            "посчитай",
            "рассчитай",
            "предскаж",
            "прогноз",
            "оцени",
            "estimate",
            "predict",
            "calculate",
            "calculate land value",
            "обнови стоимость",
        )
        visualization_markers = (
            "покажи",
            "показать",
            "визуализ",
            "построй",
            "нарисуй",
            "карта",
            "карту",
            "plot",
            "map",
            "show",
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
        target_id = extract_target_id(user_request)

        if any(marker in text for marker in target_markers) and target_id is not None:
            return VisualizationRouteDecision(
                route="plot_target_block_map",
                metric_kind="target_block",
                price_column=self.id_column,
                title=self.target_block_title_template.format(target_id=target_id),
                target_id=target_id,
                reasoning="В запросе указан квартал по id или target_id.",
                confidence=0.9,
            )
        has_visualization_signal = any(marker in text for marker in visualization_markers)
        has_prediction_signal = any(marker in text for marker in prediction_markers)
        if self._predict_tool is not None and has_prediction_signal and not has_visualization_signal:
            return VisualizationRouteDecision(
                route="predict_land_value",
                metric_kind="land_value_prediction",
                price_column="land_value",
                title="Прогноз стоимости земли рассчитан.",
                reasoning="В запросе требуется расчёт стоимости земли без карты.",
                confidence=0.88,
            )
        if any(marker in text for marker in per_100m2_markers):
            return VisualizationRouteDecision(
                route="plot_land_value_per_100m2_map",
                metric_kind="land_value_per_100m2",
                price_column="land_value_per_100m2",
                title=self.default_unit_title,
                reasoning="В запросе упомянута стоимость за сотку или за 100 м2.",
                confidence=0.86,
            )
        if any(marker in text for marker in total_markers):
            return VisualizationRouteDecision(
                route="plot_total_land_value_map",
                metric_kind="total_land_value",
                price_column="land_value",
                title=self.default_total_title,
                reasoning="В запросе упомянута полная стоимость участка.",
                confidence=0.84,
            )
        return None

    @classmethod
    def _should_accept_llm_route(cls, decision: VisualizationRouteDecision) -> bool:
        return float(decision.confidence) >= cls._LLM_ROUTE_CONFIDENCE_THRESHOLD

    def _normalize_llm_decision(
        self,
        decision: VisualizationRouteDecision,
        *,
        user_request: str,
    ) -> VisualizationRouteDecision:
        normalized_request = user_request.lower()
        has_visualization_signal = any(
            marker in normalized_request
            for marker in (
                "покажи",
                "показать",
                "визуализ",
                "построй",
                "нарисуй",
                "карта",
                "карту",
                "plot",
                "map",
                "show",
            )
        )
        has_prediction_signal = any(
            marker in normalized_request
            for marker in (
                "вычисли",
                "посчитай",
                "рассчитай",
                "предскаж",
                "прогноз",
                "оцени",
                "estimate",
                "predict",
                "calculate",
                "обнови стоимость",
            )
        )
        if (
            self._predict_tool is not None
            and has_prediction_signal
            and not has_visualization_signal
            and decision.route != "predict_land_value"
        ):
            return VisualizationRouteDecision(
                route="predict_land_value",
                metric_kind="land_value_prediction",
                price_column="land_value",
                title="Прогноз стоимости земли рассчитан.",
                reasoning="Запрос требует только вычисления стоимости земли без построения карты.",
                confidence=max(float(decision.confidence), 0.9),
            )
        if decision.route != "plot_target_block_map" or decision.target_id is not None:
            if decision.route != "predict_land_value" or self._predict_tool is not None:
                return decision
            return VisualizationRouteDecision(
                route="plot_total_land_value_map",
                metric_kind="total_land_value",
                price_column="land_value",
                title=self.default_total_title,
                reasoning="Маршрут predict_land_value недоступен без prediction_config, выбрана карта полной стоимости участка.",
                confidence=max(float(decision.confidence) - 0.2, 0.0),
            )
        target_id = extract_target_id(user_request)
        if target_id is not None:
            return decision.model_copy(
                update={
                    "target_id": target_id,
                    "price_column": self.id_column,
                    "title": self.target_block_title_template.format(target_id=target_id),
                }
            )
        return VisualizationRouteDecision(
            route="plot_total_land_value_map",
            metric_kind="total_land_value",
            price_column="land_value",
            title=self.default_total_title,
            reasoning="Маршрут квартала без target_id недоопределён, выбрана карта полной стоимости участка.",
            confidence=max(float(decision.confidence) - 0.2, 0.0),
        )

    def _default_route_decision(self) -> VisualizationRouteDecision:
        return VisualizationRouteDecision(
            route="plot_total_land_value_map",
            metric_kind="total_land_value",
            price_column="land_value",
            title=self.default_total_title,
            reasoning="Точный маршрут не распознан, выбрана визуализация полной стоимости участка.",
            confidence=0.2,
        )

    def _execute_visualization_node(self, state: VisualizationState) -> dict[str, Any]:
        decision = VisualizationRouteDecision.model_validate(state["route_decision"])
        self._artifact_store.pop(decision.route, None)
        execution_request = self._build_execution_request(
            user_request=state["user_request"],
            route_decision=decision,
        )
        try:
            agent_result = self._agent.invoke({"messages": [HumanMessage(content=execution_request)]})
        except Exception:
            agent_message = ""
            tool_name = None
            tool_payload: dict[str, Any] = {}
        else:
            agent_message = extract_last_ai_message_text(agent_result)
            tool_name, tool_payload = self._extract_last_tool_result(agent_result)
        used_tool_fallback = False
        if not self._was_route_executed(
            decision=decision,
            tool_name=tool_name,
            tool_payload=tool_payload,
        ):
            used_tool_fallback = True
            tool_payload = self._invoke_tool_fallback(decision)
        return {
            "agent_message": agent_message,
            "used_tool_fallback": used_tool_fallback,
            "tool_payload": tool_payload,
        }

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

    def _invoke_tool_fallback(self, route_decision: VisualizationRouteDecision) -> dict[str, Any]:
        if route_decision.route == "predict_land_value":
            if self._predict_tool is None:
                raise ValueError("predict_land_value requires prediction_config.")
            return self._predict_tool.invoke({})
        if route_decision.route == "plot_total_land_value_map":
            return self._plot_total_tool.invoke({})
        if route_decision.route == "plot_land_value_per_100m2_map":
            return self._plot_unit_tool.invoke({})
        if route_decision.target_id is None:
            raise ValueError("plot_target_block_map requires target_id.")
        return self._plot_target_block_tool.invoke({"target_id": route_decision.target_id})

    def _was_route_executed(
        self,
        *,
        decision: VisualizationRouteDecision,
        tool_name: str | None,
        tool_payload: dict[str, Any],
    ) -> bool:
        if decision.route == "predict_land_value":
            return tool_name == "predict_land_value" and bool(tool_payload)
        return decision.route in self._artifact_store

    @staticmethod
    def _extract_last_tool_result(raw_result: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        message = extract_last_tool_message(raw_result)
        if message is None:
            return None, {}
        content = message.content
        parsed: dict[str, Any] = {}
        if isinstance(content, dict):
            parsed = content
        elif isinstance(content, str):
            try:
                loaded = json.loads(content)
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, dict):
                parsed = loaded
        name = getattr(message, "name", None)
        return (str(name) if name else None), parsed

    def _finalize_node(self, state: VisualizationState) -> dict[str, Any]:
        decision = VisualizationRouteDecision.model_validate(state["route_decision"])
        artifact = self._artifact_store.get(decision.route)
        tool_payload = dict(state.get("tool_payload", {}) or {})
        if artifact is not None:
            tool_payload = artifact.tool_payload()
            prediction_payload = dict(state.get("tool_payload", {}) or {}).get("prediction")
            if prediction_payload is not None:
                tool_payload["prediction"] = prediction_payload
        target_id = decision.target_id if decision.route == "plot_target_block_map" else None
        if artifact is not None and hasattr(artifact, "target_id"):
            target_id = int(getattr(artifact, "target_id"))
        agent_message = str(state.get("agent_message", "")).strip()
        if decision.route == "predict_land_value":
            agent_message = (
                "Стоимость земли (`land_value`) и стоимость земли за сотку "
                "(`land_value_per_100m2`) рассчитаны для всех кварталов. "
                "Данные обновлены в наборе `baseline_blocks`."
            )
        return {
            "result_payload": {
                "user_request": state["user_request"],
                "route": decision.route,
                "metric_kind": decision.metric_kind,
                "price_column": decision.price_column,
                "title": artifact.title if artifact is not None else decision.title,
                "reasoning": decision.reasoning,
                "target_id": target_id,
                "agent_message": agent_message,
                "used_tool_fallback": bool(state.get("used_tool_fallback", False)),
                "tool_payload": tool_payload,
            }
        }

def create_visualization_agent(
    *,
    llm: Any,
    baseline_blocks: gpd.GeoDataFrame,
    prediction_config: LandValuePredictionConfig | None = None,
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
        prediction_config=prediction_config,
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
    prediction_config: LandValuePredictionConfig | None = None,
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
        prediction_config=prediction_config,
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
