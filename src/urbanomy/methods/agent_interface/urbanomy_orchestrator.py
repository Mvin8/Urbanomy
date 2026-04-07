"""Top-level Urbanomy orchestrator that delegates requests to subagents."""

from __future__ import annotations

import re
from typing import Any, TypedDict

import geopandas as gpd
from langgraph.graph import END, START, StateGraph

from .internal.common.agent_utils import invoke_structured_router
from .internal.common.domain_contracts import ToolDescriptor, format_tool_catalog
from .internal.common.request_parsing import normalize_text
from .internal.common.session_memory import (
    build_session_memory_snapshot,
    build_session_task,
    format_recent_tasks,
)
from .internal.block_parameters.glossary import looks_like_block_parameter_definition_request
from .internal.district_optimization.intents import (
    looks_like_district_optimization_request,
)
from .block_parameters_agent import (
    BlockParametersAgent,
    create_block_parameters_agent,
    looks_like_block_parameters_request,
)
from .document_rag_agent import DocumentRagAgent, create_document_rag_agent
from .general_qa_agent import GeneralQaAgent, create_general_qa_agent
from .models import (
    ConfirmationGate,
    DistrictOptimizationConfig,
    LandValuePredictionConfig,
    TargetBlockVisualizationResult,
    UrbanomyOrchestratorRequest,
    UrbanomyOrchestratorResult,
    UrbanomyOrchestratorRouteDecision,
)
from .planning_agent import PlanningAgent, create_planning_agent
from .prompts import ORCHESTRATOR_ROUTER_SYSTEM_PROMPT
from .tools.internal.district_optimization import latest_session_summary
from .verification_agent import VerificationAgent, create_verification_agent
from .visualization_agent import VisualizationAgent, create_visualization_agent

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:
    InMemorySaver = None


class UrbanomyOrchestratorState(TypedDict, total=False):
    """State passed through the top-level Urbanomy orchestrator graph."""

    user_request: str
    thread_id: str
    route: str
    reasoning: str
    latest_route: str
    latest_response: str
    latest_optimization_summary: dict[str, Any]
    latest_solution_number: int
    history: list[dict[str, str]]


class UrbanomyOrchestrator:
    """Top-level router over Urbanomy interface subagents."""

    _LLM_ROUTE_CONFIDENCE_THRESHOLD = 0.65
    _LLM_ROUTE_MIN_CONFIDENCE = 0.35

    def __init__(
        self,
        *,
        llm: Any,
        baseline_blocks: gpd.GeoDataFrame,
        district_optimization_config: DistrictOptimizationConfig | None = None,
        land_value_prediction_config: LandValuePredictionConfig | None = None,
        checkpointer: Any | None = None,
        default_thread_id: str = "default",
        checkpoint_namespace: str = "urbanomy_orchestrator_v2",
    ) -> None:
        if llm is None:
            raise ValueError("llm is required to build UrbanomyOrchestrator.")
        self.llm = llm
        self.baseline_blocks = baseline_blocks
        self.district_optimization_config = district_optimization_config
        self.land_value_prediction_config = (
            land_value_prediction_config
            or self._prediction_config_from_optimization(district_optimization_config)
        )
        self.checkpointer = checkpointer or (InMemorySaver() if InMemorySaver is not None else None)
        self.default_thread_id = str(default_thread_id).strip() or "default"
        self.active_thread_id = self.default_thread_id
        self.checkpoint_namespace = str(checkpoint_namespace).strip() or "urbanomy_orchestrator_v2"
        self._visualization_agent: VisualizationAgent | None = None
        self._district_optimization_agent: Any | None = None
        self._document_rag_agent: DocumentRagAgent | None = None
        self._block_parameters_agent: BlockParametersAgent | None = None
        self._general_qa_agent: GeneralQaAgent | None = None
        self._thread_optimization_sessions: dict[str, Any] = {}
        self._thread_algorithm_overrides: dict[str, dict[str, int]] = {}
        self._thread_runtime_outputs: dict[str, dict[str, Any]] = {}
        self._thread_task_ledger: dict[str, list[Any]] = {}
        self._thread_verification_reports: dict[str, Any] = {}
        self._thread_session_memory: dict[str, Any] = {}
        self._thread_research_briefs: dict[str, Any] = {}
        self._thread_active_plans: dict[str, Any] = {}
        self._thread_confirmation_gates: dict[str, ConfirmationGate] = {}
        self._thread_decision_documents: dict[str, Any] = {}
        self._thread_decision_document_stores: dict[str, Any] = {}
        self._thread_document_selection_results: dict[str, Any] = {}
        self._verification_agent: VerificationAgent | None = None
        self._planning_agent: PlanningAgent | None = None
        self.graph = self._build_graph()

    def invoke(self, user_request: str, *, thread_id: str | None = None) -> UrbanomyOrchestratorResult:
        """Route a free-form user request to the right Urbanomy subagent."""
        request = UrbanomyOrchestratorRequest(user_request=user_request)
        resolved_thread_id = self._resolve_thread_id(thread_id)
        pending_gate = self._thread_confirmation_gates.get(resolved_thread_id)
        if pending_gate is not None:
            decision = self._interpret_confirmation_reply(request.user_request)
            if decision == "approve":
                self._thread_confirmation_gates.pop(resolved_thread_id, None)
                output = self.graph.invoke(
                    {
                        "user_request": pending_gate.original_user_request,
                        "thread_id": resolved_thread_id,
                    },
                    config=self._graph_config(resolved_thread_id, self.checkpoint_namespace),
                )
                result = self._build_result_from_state(
                    state=dict(output),
                    thread_id=resolved_thread_id,
                )
                result.response = (
                    "Подтверждение принято. Выполняю согласованный план.\n\n"
                    f"{result.response}"
                ).strip()
                result.confirmation_gate = pending_gate.model_copy(update={"status": "approved"})
                return result
            if decision == "reject":
                self._thread_confirmation_gates.pop(resolved_thread_id, None)
                return self._build_confirmation_result(
                    user_request=request.user_request,
                    route=pending_gate.route,
                    reasoning="План снят после экспертного отказа пользователя.",
                    response=(
                        "План отменён. Сформулируй уточнение, и я соберу новый набор шагов и инструментов."
                    ),
                    thread_id=resolved_thread_id,
                    confirmation_gate=pending_gate.model_copy(update={"status": "cancelled"}),
                )
            self._thread_confirmation_gates.pop(resolved_thread_id, None)
        preview = self._router_node(
            {
                "user_request": request.user_request,
                "thread_id": resolved_thread_id,
            }
        )
        confirmation_gate = self._maybe_create_confirmation_gate(
            thread_id=resolved_thread_id,
            route=str(preview.get("route") or "general_qa"),
            user_request=request.user_request,
        )
        if confirmation_gate is not None:
            self._thread_confirmation_gates[resolved_thread_id] = confirmation_gate
            return self._build_confirmation_result(
                user_request=request.user_request,
                route=str(preview.get("route") or "general_qa"),
                reasoning=str(preview.get("reasoning") or ""),
                response=confirmation_gate.to_text(),
                thread_id=resolved_thread_id,
                confirmation_gate=confirmation_gate,
            )
        output = self.graph.invoke(
            {
                "user_request": request.user_request,
                "thread_id": resolved_thread_id,
            },
            config=self._graph_config(resolved_thread_id, self.checkpoint_namespace),
        )
        return self._build_result_from_state(
            state=dict(output),
            thread_id=resolved_thread_id,
        )

    def run(self, user_request: str, *, thread_id: str | None = None) -> UrbanomyOrchestratorResult:
        return self.invoke(user_request, thread_id=thread_id)

    def ask(self, user_request: str, *, thread_id: str | None = None) -> UrbanomyOrchestratorResult:
        return self.invoke(user_request, thread_id=thread_id)

    def __call__(self, user_request: str, *, thread_id: str | None = None) -> UrbanomyOrchestratorResult:
        return self.invoke(user_request, thread_id=thread_id)

    def set_thread(self, thread_id: str) -> str:
        """Set the active thread id used by default in subsequent calls."""
        resolved_thread_id = self._resolve_thread_id(thread_id)
        self.active_thread_id = resolved_thread_id
        return resolved_thread_id

    def get_active_thread(self) -> str:
        """Return the currently active default thread id."""
        return self.active_thread_id

    def get_thread_state(self, thread_id: str | None = None) -> dict[str, Any]:
        """Return the latest persisted state values for a thread."""
        resolved_thread_id = self._resolve_thread_id(thread_id)
        try:
            snapshot = self.graph.get_state(
                self._graph_config(resolved_thread_id, self.checkpoint_namespace)
            )
        except Exception:
            return {}
        return dict(getattr(snapshot, "values", {}) or {})

    def get_memory_summary(self, thread_id: str | None = None) -> dict[str, Any]:
        """Return a compact summary of the current thread memory."""
        resolved_thread_id = self._resolve_thread_id(thread_id)
        state = self.get_thread_state(resolved_thread_id)
        history = list(state.get("history", []))
        session_memory = self._thread_session_memory.get(resolved_thread_id)
        tasks = list(self._thread_task_ledger.get(resolved_thread_id, []))
        return {
            "thread_id": resolved_thread_id,
            "latest_route": state.get("latest_route"),
            "latest_response": state.get("latest_response"),
            "history_size": len(history),
            "recent_history": history[-6:],
            "latest_optimization_summary": state.get("latest_optimization_summary"),
            "latest_solution_number": state.get("latest_solution_number"),
            "has_optimization_session": resolved_thread_id in self._thread_optimization_sessions,
            "algorithm_overrides": self._thread_algorithm_overrides.get(resolved_thread_id),
            "task_count": len(tasks),
            "recent_tasks": tasks[-6:],
            "verification_report": self._thread_verification_reports.get(resolved_thread_id),
            "session_memory": session_memory.model_dump() if session_memory is not None else None,
            "active_decision_document": (
                self._summarize_decision_document(self._thread_decision_documents[resolved_thread_id])
                if resolved_thread_id in self._thread_decision_documents
                else None
            ),
            "latest_document_selection": (
                self._thread_document_selection_results[resolved_thread_id].model_dump()
                if resolved_thread_id in self._thread_document_selection_results
                else None
            ),
            "research_brief": (
                self._thread_research_briefs[resolved_thread_id].model_dump()
                if resolved_thread_id in self._thread_research_briefs
                else None
            ),
            "active_plan": (
                self._thread_active_plans[resolved_thread_id].model_dump()
                if resolved_thread_id in self._thread_active_plans
                else None
            ),
            "confirmation_gate": (
                self._thread_confirmation_gates[resolved_thread_id].model_dump()
                if resolved_thread_id in self._thread_confirmation_gates
                else None
            ),
        }

    def get_session_memory(self, thread_id: str | None = None) -> dict[str, Any] | None:
        """Return the latest durable session-memory snapshot for one thread."""
        resolved_thread_id = self._resolve_thread_id(thread_id)
        snapshot = self._thread_session_memory.get(resolved_thread_id)
        return snapshot.model_dump() if snapshot is not None else None

    def get_capabilities(self) -> list[str]:
        """Return user-facing capabilities currently available in this runtime."""
        lines: list[str] = []
        lines.extend(self._get_visualization_agent().capability_lines())
        lines.extend(self._get_block_parameters_agent().capability_lines())
        lines.extend(self._get_document_rag_agent().capability_lines())
        lines.extend(GeneralQaAgent.capability_lines())
        if self.district_optimization_config is not None:
            lines.extend(self._get_district_optimization_agent().capability_lines())
        return list(dict.fromkeys(lines))

    def get_capabilities_text(self) -> str:
        """Return a formatted capabilities block for notebook/chat display."""
        return "\n".join(self.get_capabilities())

    def get_tool_catalog(self) -> list[ToolDescriptor]:
        """Return structured descriptors of tools available in this runtime."""
        descriptors: list[ToolDescriptor] = []
        descriptors.extend(self._get_visualization_agent().tool_descriptors())
        descriptors.extend(self._get_block_parameters_agent().tool_descriptors())
        descriptors.extend(self._get_document_rag_agent().tool_descriptors())
        descriptors.extend(self._get_general_qa_agent().tool_descriptors())
        if self.district_optimization_config is not None:
            descriptors.extend(self._get_district_optimization_agent().tool_descriptors())
        deduped: dict[str, ToolDescriptor] = {}
        for descriptor in descriptors:
            deduped.setdefault(descriptor.name, descriptor)
        return list(deduped.values())

    def get_tool_catalog_text(self) -> str:
        """Return a formatted tool catalog for notebook/chat display."""
        return format_tool_catalog(self.get_tool_catalog())

    def clear_thread_memory(self, thread_id: str | None = None) -> None:
        """Clear persisted memory for one thread in the in-memory runtime stores."""
        resolved_thread_id = self._resolve_thread_id(thread_id)
        if self.checkpointer is not None:
            self.graph.update_state(
                self._graph_config(resolved_thread_id, self.checkpoint_namespace),
                {
                    "history": [],
                    "latest_response": "",
                    "latest_route": "",
                    "latest_optimization_summary": None,
                    "latest_solution_number": None,
                },
                as_node="finalize",
            )
        self._thread_optimization_sessions.pop(resolved_thread_id, None)
        self._thread_algorithm_overrides.pop(resolved_thread_id, None)
        self._thread_runtime_outputs.pop(resolved_thread_id, None)
        self._thread_task_ledger.pop(resolved_thread_id, None)
        self._thread_verification_reports.pop(resolved_thread_id, None)
        self._thread_session_memory.pop(resolved_thread_id, None)
        self._thread_research_briefs.pop(resolved_thread_id, None)
        self._thread_active_plans.pop(resolved_thread_id, None)
        self._thread_confirmation_gates.pop(resolved_thread_id, None)
        self._thread_decision_documents.pop(resolved_thread_id, None)
        self._thread_decision_document_stores.pop(resolved_thread_id, None)
        self._thread_document_selection_results.pop(resolved_thread_id, None)

    def reset_memory(self, thread_id: str | None = None) -> None:
        """Alias for clear_thread_memory with notebook-friendly naming."""
        self.clear_thread_memory(thread_id)

    def _build_graph(self):
        graph = StateGraph(UrbanomyOrchestratorState)
        graph.add_node("router", self._router_node)
        graph.add_node("visualization", self._visualization_node)
        graph.add_node("document_rag", self._document_rag_node)
        graph.add_node("district_optimization", self._district_optimization_node)
        graph.add_node("block_parameters", self._block_parameters_node)
        graph.add_node("general_qa", self._general_qa_node)
        graph.add_node("unsupported", self._unsupported_node)
        graph.add_node("finalize", self._finalize_node)
        graph.add_edge(START, "router")
        graph.add_conditional_edges(
            "router",
            self._route_after_router,
            {
                "visualization": "visualization",
                "document_rag": "document_rag",
                "district_optimization": "district_optimization",
                "block_parameters": "block_parameters",
                "general_qa": "general_qa",
                "unsupported": "unsupported",
            },
        )
        graph.add_edge("visualization", "finalize")
        graph.add_edge("document_rag", "finalize")
        graph.add_edge("district_optimization", "finalize")
        graph.add_edge("block_parameters", "finalize")
        graph.add_edge("general_qa", "finalize")
        graph.add_edge("unsupported", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile(checkpointer=self.checkpointer)

    def _router_node(self, state: UrbanomyOrchestratorState) -> dict[str, Any]:
        user_request = state["user_request"]
        thread_id = state["thread_id"]
        if looks_like_block_parameter_definition_request(user_request):
            return self._build_route_update(
                thread_id=thread_id,
                route="general_qa",
                reasoning="Запрос объясняет термин параметра квартала, поэтому направлен в general_qa.",
                user_request=user_request,
            )
        if self._looks_like_document_rag_request(user_request):
            return self._build_route_update(
                thread_id=thread_id,
                route="document_rag",
                reasoning=(
                    "Запрос явно относится к активному документу, retrieval по документу "
                    "или выбору решения на основе документа."
                ),
                user_request=user_request,
            )
        decision = invoke_structured_router(
            llm=self.llm,
            schema=self._route_decision_schema(),
            system_prompt=ORCHESTRATOR_ROUTER_SYSTEM_PROMPT,
            user_request=user_request,
        )
        heuristic = self._heuristic_route(user_request)
        if decision is not None:
            if self._should_accept_llm_route(decision):
                return self._build_route_update(
                    thread_id=thread_id,
                    route=str(decision.route),
                    reasoning=str(decision.reasoning),
                    user_request=user_request,
                )
            if heuristic is not None:
                return self._build_route_update(
                    thread_id=thread_id,
                    route=str(heuristic.route),
                    reasoning=str(heuristic.reasoning),
                    user_request=user_request,
                )
            if float(decision.confidence) >= self._LLM_ROUTE_MIN_CONFIDENCE:
                return self._build_route_update(
                    thread_id=thread_id,
                    route=str(decision.route),
                    reasoning=str(decision.reasoning),
                    user_request=user_request,
                )
        if heuristic is not None:
            return self._build_route_update(
                thread_id=thread_id,
                route=str(heuristic.route),
                reasoning=str(heuristic.reasoning),
                user_request=user_request,
            )

        return self._build_route_update(
            thread_id=thread_id,
            route="general_qa",
            reasoning="Точный маршрут не определён, запрос передан в разговорную ветку.",
            user_request=user_request,
        )

    def _heuristic_route(self, user_request: str) -> UrbanomyOrchestratorRouteDecision | None:
        text = normalize_text(user_request)
        stripped = text.strip()
        visualization_markers = (
            "покажи",
            "показать",
            "карта",
            "карту",
            "визуализ",
            "построй",
            "нарисуй",
            "plot",
            "map",
            "show",
        )
        land_value_markers = (
            "стоимость земли",
            "стоимость земельных участков",
            "стоимость участка",
            "стоимость за сотку",
            "за сотку",
            "land value",
            "land_value",
            "land_value_per_100m2",
            "100м2",
            "100 м2",
            "100 м²",
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
        )
        target_block_markers = (
            "target_id",
            "квартал",
            "блок",
            "block id",
            "изменяемый квартал",
            "выдели квартал",
            "покажи квартал",
        )
        general_qa_markers = (
            "что такое",
            "как работает",
            "что делает",
            "как устро",
            "что ты умеешь",
            "что умеешь",
            "какие инструменты",
            "какие возможности",
            "что можешь",
            "объясни",
            "расскажи",
            "как ты работаешь",
        )
        has_target_id = bool(
            re.search(r"(target_id|id)\s*[:=]?\s*\d+", text)
            or re.search(r"(квартал|блок)\D{0,20}\d+", text)
        )
        is_arithmetic_expression = bool(re.fullmatch(r"[\d\s+\-*/().=]+", stripped))
        is_short_general_query = len(stripped) <= 32 and not any(
            marker in text
            for marker in (
                *visualization_markers,
                *land_value_markers,
                *target_block_markers,
            )
        ) and not looks_like_district_optimization_request(text)

        if any(marker in text for marker in general_qa_markers) and not has_target_id:
            return UrbanomyOrchestratorRouteDecision(
                route="general_qa",
                reasoning="Запрос относится к объяснению работы системы или её веток.",
                confidence=0.88,
            )
        if looks_like_district_optimization_request(text):
            return UrbanomyOrchestratorRouteDecision(
                route="district_optimization",
                reasoning="Запрос относится к optimization-домену: запуску, анализу решений или настройкам оптимизатора.",
                confidence=0.92,
            )
        if self._looks_like_block_parameters_request(user_request):
            return UrbanomyOrchestratorRouteDecision(
                route="block_parameters",
                reasoning="Запрос относится к получению параметров квартала по id.",
                confidence=0.9,
            )
        if any(marker in text for marker in target_block_markers) and has_target_id:
            return UrbanomyOrchestratorRouteDecision(
                route="visualization",
                reasoning="Запрос относится к визуализации квартала по target_id.",
                confidence=0.88,
            )
        if any(marker in text for marker in prediction_markers) and any(
            marker in text for marker in land_value_markers
        ):
            return UrbanomyOrchestratorRouteDecision(
                route="visualization",
                reasoning="Запрос относится к расчёту или прогнозу стоимости земли.",
                confidence=0.86,
            )
        if any(marker in text for marker in visualization_markers) and any(
            marker in text for marker in land_value_markers
        ):
            return UrbanomyOrchestratorRouteDecision(
                route="visualization",
                reasoning="Запрос относится к визуализации стоимости земли.",
                confidence=0.84,
            )
        if any(marker in text for marker in general_qa_markers):
            return UrbanomyOrchestratorRouteDecision(
                route="general_qa",
                reasoning="Запрос относится к разговорной или справочной ветке.",
                confidence=0.86,
            )
        if is_arithmetic_expression or is_short_general_query:
            return UrbanomyOrchestratorRouteDecision(
                route="general_qa",
                reasoning="Короткий общий запрос направлен в разговорную ветку.",
                confidence=0.9,
            )
        return None

    @staticmethod
    def _looks_like_document_rag_request(user_request: str) -> bool:
        text = normalize_text(user_request)
        document_markers = (
            "найди в документе",
            "что в документе сказано",
            "покажи фрагменты",
            "покажи выдержки",
            "контекст документа",
            "согласно документ",
            "по документу",
            "на основе документа",
            "активный документ",
            "какой документ",
            "докмент",
            "генплан",
            "стратег",
            "document_text",
            "document_path",
        )
        optimization_markers = (
            "оптимиз",
            "pareto front",
            "парето фронт",
            "pareto-front",
            "инвестиционные метрики",
            "метрики решения",
            "параметры решения",
            "визуализируй решение",
            "покажи pareto",
        )
        has_document_signal = any(marker in text for marker in document_markers)
        has_optimization_signal = any(marker in text for marker in optimization_markers)
        return has_document_signal and not has_optimization_signal

    @classmethod
    def _should_accept_llm_route(cls, decision: UrbanomyOrchestratorRouteDecision) -> bool:
        return float(decision.confidence) >= cls._LLM_ROUTE_CONFIDENCE_THRESHOLD

    @staticmethod
    def _route_update_from_decision(
        decision: UrbanomyOrchestratorRouteDecision,
    ) -> dict[str, Any]:
        return {
            "route": str(decision.route),
            "reasoning": str(decision.reasoning),
        }

    def _build_route_update(
        self,
        *,
        thread_id: str,
        route: str,
        reasoning: str,
        user_request: str,
    ) -> dict[str, Any]:
        self._record_request_analysis(
            thread_id=thread_id,
            route=route,
            user_request=user_request,
        )
        return {
            "route": route,
            "reasoning": reasoning,
        }

    def _route_after_router(self, state: UrbanomyOrchestratorState) -> str:
        route = state["route"]
        if route == "district_optimization" and self.district_optimization_config is not None:
            return "district_optimization"
        if route == "document_rag":
            return "document_rag"
        if route == "block_parameters":
            return "block_parameters"
        if route in {"visualization", "target_block_visualization", "land_value_visualization"}:
            return "visualization"
        if route == "general_qa":
            return "general_qa"
        return "unsupported"

    def _visualization_node(self, state: UrbanomyOrchestratorState) -> dict[str, Any]:
        thread_id = state["thread_id"]
        result = self._get_visualization_agent().invoke(state["user_request"])
        self._thread_verification_reports[thread_id] = self._get_verification_agent().verify(
            route="visualization",
            result=result,
        )
        self._set_runtime_output(
            thread_id,
            visualization_result=result,
            target_block_visualization_result=self._build_legacy_target_block_result(result),
            block_parameters_result=None,
            document_rag_result=None,
            district_optimization_result=None,
        )
        return {
            "latest_route": "visualization",
            "latest_response": result.agent_message or self._default_visualization_response(result.route),
        }

    def _document_rag_node(self, state: UrbanomyOrchestratorState) -> dict[str, Any]:
        thread_id = state["thread_id"]
        agent = self._get_document_rag_agent()
        active_document = self._thread_decision_documents.get(thread_id)
        if active_document is not None:
            agent.session_store["active_decision_document"] = active_document
        else:
            agent.session_store.pop("active_decision_document", None)
        active_document_store = self._thread_decision_document_stores.get(thread_id)
        if active_document_store is not None:
            agent.session_store["active_decision_document_store"] = active_document_store
        else:
            agent.session_store.pop("active_decision_document_store", None)
        latest_selection = self._thread_document_selection_results.get(thread_id)
        if latest_selection is not None:
            agent.session_store["latest_document_selection_result"] = latest_selection
        else:
            agent.session_store.pop("latest_document_selection_result", None)
        latest_optimization_session = self._thread_optimization_sessions.get(thread_id)
        if latest_optimization_session is not None:
            agent.session_store["latest_district_optimization_session"] = latest_optimization_session
        else:
            agent.session_store.pop("latest_district_optimization_session", None)
        result = agent.invoke(state["user_request"])
        latest_document = agent.session_store.get("active_decision_document")
        if latest_document is not None:
            self._thread_decision_documents[thread_id] = latest_document
        else:
            self._thread_decision_documents.pop(thread_id, None)
        latest_document_store = agent.session_store.get("active_decision_document_store")
        if latest_document_store is not None:
            self._thread_decision_document_stores[thread_id] = latest_document_store
        else:
            self._thread_decision_document_stores.pop(thread_id, None)
        latest_selection = agent.session_store.get("latest_document_selection_result")
        if latest_selection is not None:
            self._thread_document_selection_results[thread_id] = latest_selection
        else:
            self._thread_document_selection_results.pop(thread_id, None)
        self._set_runtime_output(
            thread_id,
            visualization_result=None,
            target_block_visualization_result=None,
            block_parameters_result=None,
            document_rag_result=result,
            district_optimization_result=None,
        )
        self._thread_verification_reports[thread_id] = self._get_verification_agent().verify(
            route="document_rag",
            result=result,
        )
        update: dict[str, Any] = {
            "latest_route": "document_rag",
            "latest_response": str(result.get("response", "Документный запрос обработан.")),
        }
        tool_output = result.get("tool_output")
        if isinstance(tool_output, dict):
            solution_number = tool_output.get("selected_solution_number") or tool_output.get(
                "solution_number"
            )
            if isinstance(solution_number, int):
                update["latest_solution_number"] = solution_number
        return update

    def _district_optimization_node(self, state: UrbanomyOrchestratorState) -> dict[str, Any]:
        thread_id = state["thread_id"]
        agent = self._get_district_optimization_agent()
        session = self._thread_optimization_sessions.get(thread_id)
        overrides = self._thread_algorithm_overrides.get(thread_id)
        if session is None:
            agent.session_store.pop("latest_district_optimization_session", None)
        else:
            agent.session_store["latest_district_optimization_session"] = session
        if overrides:
            agent.session_store["algorithm_overrides"] = dict(overrides)
        else:
            agent.session_store.pop("algorithm_overrides", None)
        active_document = self._thread_decision_documents.get(thread_id)
        if active_document is not None:
            agent.session_store["active_decision_document"] = active_document
        else:
            agent.session_store.pop("active_decision_document", None)
        active_document_store = self._thread_decision_document_stores.get(thread_id)
        if active_document_store is not None:
            agent.session_store["active_decision_document_store"] = active_document_store
        else:
            agent.session_store.pop("active_decision_document_store", None)
        latest_selection = self._thread_document_selection_results.get(thread_id)
        if latest_selection is not None:
            agent.session_store["latest_document_selection_result"] = latest_selection
        else:
            agent.session_store.pop("latest_document_selection_result", None)
        result = agent.invoke(state["user_request"])
        latest_session = agent.session_store.get("latest_district_optimization_session")
        if latest_session is not None:
            self._thread_optimization_sessions[thread_id] = latest_session
        latest_overrides = dict(agent.session_store.get("algorithm_overrides") or {})
        if latest_overrides:
            self._thread_algorithm_overrides[thread_id] = latest_overrides
        else:
            self._thread_algorithm_overrides.pop(thread_id, None)
        latest_document = agent.session_store.get("active_decision_document")
        if latest_document is not None:
            self._thread_decision_documents[thread_id] = latest_document
        else:
            self._thread_decision_documents.pop(thread_id, None)
        latest_document_store = agent.session_store.get("active_decision_document_store")
        if latest_document_store is not None:
            self._thread_decision_document_stores[thread_id] = latest_document_store
        else:
            self._thread_decision_document_stores.pop(thread_id, None)
        latest_selection = agent.session_store.get("latest_document_selection_result")
        if latest_selection is not None:
            self._thread_document_selection_results[thread_id] = latest_selection
        else:
            self._thread_document_selection_results.pop(thread_id, None)
        summary = self._build_latest_optimization_summary(thread_id=thread_id)
        self._set_runtime_output(
            thread_id,
            visualization_result=None,
            target_block_visualization_result=None,
            block_parameters_result=None,
            document_rag_result=None,
            district_optimization_result=result,
        )
        self._thread_verification_reports[thread_id] = self._get_verification_agent().verify(
            route="district_optimization",
            result=result,
            optimization_summary=summary,
        )
        update: dict[str, Any] = {
            "latest_route": "district_optimization",
            "latest_response": str(result.get("response", "Запрос обработан.")),
        }
        if summary is not None:
            update["latest_optimization_summary"] = summary
        tool_output = result.get("tool_output")
        if isinstance(tool_output, dict):
            solution_number = tool_output.get("solution_number")
            if isinstance(solution_number, int):
                update["latest_solution_number"] = solution_number
        return update

    def _general_qa_node(self, state: UrbanomyOrchestratorState) -> dict[str, Any]:
        context = self._build_general_qa_context(state)
        response = self._get_general_qa_agent().invoke(state["user_request"], context=context)
        self._thread_verification_reports[state["thread_id"]] = self._get_verification_agent().verify(
            route="general_qa",
            result=response,
        )
        self._set_runtime_output(
            state["thread_id"],
            visualization_result=None,
            target_block_visualization_result=None,
            block_parameters_result=None,
            document_rag_result=None,
            district_optimization_result=None,
        )
        return {
            "latest_route": "general_qa",
            "latest_response": response,
        }

    def _block_parameters_node(self, state: UrbanomyOrchestratorState) -> dict[str, Any]:
        result = self._get_block_parameters_agent().invoke(state["user_request"])
        self._thread_verification_reports[state["thread_id"]] = self._get_verification_agent().verify(
            route="block_parameters",
            result=result,
        )
        self._set_runtime_output(
            state["thread_id"],
            visualization_result=None,
            target_block_visualization_result=None,
            block_parameters_result=result,
            document_rag_result=None,
            district_optimization_result=None,
        )
        return {
            "latest_route": "block_parameters",
            "latest_response": str(result.get("response", "Параметры квартала получены.")).strip(),
        }

    def _unsupported_node(self, state: UrbanomyOrchestratorState) -> dict[str, Any]:
        capabilities = self.get_capabilities_text()
        self._thread_verification_reports[state["thread_id"]] = self._get_verification_agent().verify(
            route="unsupported",
            result=None,
        )
        self._set_runtime_output(
            state["thread_id"],
            visualization_result=None,
            target_block_visualization_result=None,
            block_parameters_result=None,
            document_rag_result=None,
            district_optimization_result=None,
        )
        return {
            "latest_route": "unsupported",
            "latest_response": (
                "Этот запрос не относится к доступным возможностям orchestrator.\n\n"
                f"Сейчас доступны:\n{capabilities}"
            ),
        }

    def _finalize_node(self, state: UrbanomyOrchestratorState) -> dict[str, Any]:
        thread_id = state["thread_id"]
        task_ledger = list(self._thread_task_ledger.get(thread_id, []))
        task_ledger.append(
            build_session_task(
                thread_id=thread_id,
                task_index=len(task_ledger) + 1,
                route=str(state.get("latest_route") or state.get("route") or "unsupported"),
                user_request=str(state.get("user_request", "")).strip(),
                response=self._result_response_text(state),
                verification_report=self._thread_verification_reports.get(thread_id),
            )
        )
        self._thread_task_ledger[thread_id] = task_ledger[-20:]
        self._thread_session_memory[thread_id] = build_session_memory_snapshot(
            thread_id=thread_id,
            latest_route=str(state.get("latest_route") or state.get("route") or "unsupported"),
            latest_response=self._result_response_text(state),
            latest_solution_number=state.get("latest_solution_number"),
            latest_optimization_summary=state.get("latest_optimization_summary"),
            algorithm_overrides=self._thread_algorithm_overrides.get(thread_id),
            runtime_output=self._thread_runtime_outputs.get(thread_id),
            recent_tasks=self._thread_task_ledger[thread_id],
            verification_report=self._thread_verification_reports.get(thread_id),
            research_brief=self._thread_research_briefs.get(thread_id),
            active_plan=self._thread_active_plans.get(thread_id),
            pending_confirmation_gate=self._thread_confirmation_gates.get(thread_id),
            active_decision_document=self._thread_decision_documents.get(thread_id),
            latest_document_selection=self._thread_document_selection_results.get(thread_id),
        )
        history = self._build_updated_history(state)
        return {"history": history}

    def _get_visualization_agent(self) -> VisualizationAgent:
        if self._visualization_agent is None:
            self._visualization_agent = create_visualization_agent(
                llm=self.llm,
                baseline_blocks=self.baseline_blocks,
                prediction_config=self.land_value_prediction_config,
            )
        return self._visualization_agent

    def _get_district_optimization_agent(self):
        if self.district_optimization_config is None:
            raise ValueError("district_optimization_config is not configured.")
        if self._district_optimization_agent is None:
            from .district_optimization_agent import create_district_optimization_agent

            self._district_optimization_agent = create_district_optimization_agent(
                llm=self.llm,
                baseline_blocks=self.baseline_blocks,
                model=self.district_optimization_config.model,
                orig_features=self.district_optimization_config.orig_features,
                categorical_features=self.district_optimization_config.categorical_features,
                constraints_template=self.district_optimization_config.constraints_template,
                target_id_column=self.district_optimization_config.target_id_column,
                pop_size=self.district_optimization_config.pop_size,
                n_gen=self.district_optimization_config.n_gen,
                seed=self.district_optimization_config.seed,
                eliminate_duplicates=self.district_optimization_config.eliminate_duplicates,
                save_history=self.district_optimization_config.save_history,
                use_history=self.district_optimization_config.use_history,
                verbose=self.district_optimization_config.verbose,
                use_service_features=self.district_optimization_config.use_service_features,
                service_features=self.district_optimization_config.service_features,
            )
        return self._district_optimization_agent

    def _get_document_rag_agent(self) -> DocumentRagAgent:
        if self._document_rag_agent is None:
            self._document_rag_agent = create_document_rag_agent(
                llm=self.llm,
                session_store={},
            )
        return self._document_rag_agent

    def _get_general_qa_agent(self) -> GeneralQaAgent:
        if self._general_qa_agent is None:
            self._general_qa_agent = create_general_qa_agent(
                llm=self.llm,
                context_provider=lambda: {},
            )
        return self._general_qa_agent

    def _get_block_parameters_agent(self) -> BlockParametersAgent:
        if self._block_parameters_agent is None:
            self._block_parameters_agent = create_block_parameters_agent(
                baseline_blocks=self.baseline_blocks,
            )
        return self._block_parameters_agent

    def _get_verification_agent(self) -> VerificationAgent:
        if self._verification_agent is None:
            self._verification_agent = create_verification_agent()
        return self._verification_agent

    def _get_planning_agent(self) -> PlanningAgent:
        if self._planning_agent is None:
            self._planning_agent = create_planning_agent()
        return self._planning_agent

    @staticmethod
    def _default_visualization_response(route: str) -> str:
        if route == "predict_land_value":
            return "Стоимость земли рассчитана и сохранена в baseline_blocks."
        if route == "plot_target_block_map":
            return "Квартал выделен на карте."
        if route == "plot_land_value_per_100m2_map":
            return "Построена карта стоимости земли за сотку."
        return "Построена карта стоимости земли."

    @staticmethod
    def _build_legacy_target_block_result(result: Any) -> TargetBlockVisualizationResult | None:
        if getattr(result, "route", None) != "plot_target_block_map":
            return None
        target_id = getattr(result, "target_id", None)
        if target_id is None:
            return None
        return TargetBlockVisualizationResult(
            user_request=str(getattr(result, "user_request", "")).strip(),
            route="plot_target_block_map",
            target_id=int(target_id),
            title=str(getattr(result, "title", "")).strip(),
            reasoning=str(getattr(result, "reasoning", "")).strip(),
            agent_message=str(getattr(result, "agent_message", "")).strip(),
            used_tool_fallback=bool(getattr(result, "used_tool_fallback", False)),
            tool_payload=dict(getattr(result, "tool_payload", {}) or {}),
            artifact=getattr(result, "artifact", None),
        )

    def _build_general_qa_context(self, state: UrbanomyOrchestratorState) -> dict[str, str]:
        thread_id = state["thread_id"]
        verification_report = self._thread_verification_reports.get(thread_id)
        session_memory = self._thread_session_memory.get(thread_id)
        recent_tasks = list(self._thread_task_ledger.get(thread_id, []))
        research_brief = self._thread_research_briefs.get(thread_id)
        active_plan = self._thread_active_plans.get(thread_id)
        confirmation_gate = self._thread_confirmation_gates.get(thread_id)
        return {
            "capabilities": self.get_capabilities_text(),
            "tool_catalog": self.get_tool_catalog_text(),
            "latest_optimization_context": self._build_latest_optimization_context(state),
            "latest_response": str(state.get("latest_response", "")).strip(),
            "recent_history": self._format_recent_history(state),
            "recent_tasks": format_recent_tasks(recent_tasks),
            "session_memory": (
                session_memory.to_text()
                if session_memory is not None
                else "Session memory пока не сформирована."
            ),
            "latest_verification": (
                verification_report.to_text()
                if verification_report is not None
                else "Проверка последнего шага пока не выполнялась."
            ),
            "research_brief": (
                research_brief.to_text()
                if research_brief is not None
                else "Research brief пока не построен."
            ),
            "active_plan": (
                active_plan.to_text()
                if active_plan is not None
                else "Активный план пока не построен."
            ),
            "pending_confirmation": (
                confirmation_gate.to_text()
                if confirmation_gate is not None
                else "Экспертное подтверждение сейчас не ожидается."
            ),
            "active_decision_document": (
                self._format_decision_document_context(self._thread_decision_documents[thread_id])
                if thread_id in self._thread_decision_documents
                else "Активный policy-документ пока не зарегистрирован."
            ),
            "latest_document_selection": (
                self._thread_document_selection_results[thread_id].to_text()
                if thread_id in self._thread_document_selection_results
                else "Выбор решения по документу пока не выполнялся."
            ),
        }

    def _build_capabilities_text(self) -> str:
        return self.get_capabilities_text()

    def _build_tool_catalog_text(self) -> str:
        return self.get_tool_catalog_text()

    def _build_latest_optimization_context(self, state: UrbanomyOrchestratorState) -> str:
        summary = state.get("latest_optimization_summary")
        if not isinstance(summary, dict):
            summary = self._build_latest_optimization_summary(thread_id=state["thread_id"])
        if not isinstance(summary, dict):
            return "Сессия оптимизации пока не запускалась."
        lines = [
            f"- target_id: {summary.get('target_id')}",
            f"- найдено решений: {summary.get('n_solutions')}",
        ]
        for item in (summary.get("solutions") or [])[:5]:
            lines.append(
                f"- решение {item.get('solution_number')}: {item.get('title')} | "
                f"NPV={item.get('investor_npv')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _summarize_decision_document(document: Any) -> dict[str, Any]:
        return {
            "document_name": getattr(document, "document_name", None),
            "document_type": getattr(document, "document_type", None),
            "summary": getattr(document, "summary", ""),
            "priorities": list(getattr(document, "extracted_priorities", []) or []),
            "retrieval_backend": getattr(document, "retrieval_backend", None),
            "chunk_count": int(getattr(document, "chunk_count", 0) or 0),
            "preview_text": document.preview_text() if hasattr(document, "preview_text") else "",
        }

    @classmethod
    def _format_decision_document_context(cls, document: Any) -> str:
        payload = cls._summarize_decision_document(document)
        lines = [
            f"Документ: {payload.get('document_name')}",
            f"Тип: {payload.get('document_type')}",
        ]
        if payload.get("retrieval_backend"):
            lines.append(f"Retrieval backend: {payload.get('retrieval_backend')}")
        lines.append(f"RAG-фрагментов: {payload.get('chunk_count')}")
        summary = str(payload.get("summary") or "").strip()
        if summary:
            lines.append(f"Сводка: {summary}")
        priorities = list(payload.get("priorities") or [])
        if priorities:
            lines.append("Приоритеты:")
            lines.extend(f"- {item}" for item in priorities[:5])
        preview = str(payload.get("preview_text") or "").strip()
        if preview:
            lines.append(f"Preview: {preview}")
        return "\n".join(lines).strip()

    def _build_latest_optimization_summary(self, *, thread_id: str) -> dict[str, Any] | None:
        session = self._thread_optimization_sessions.get(thread_id)
        if session is None:
            return None
        try:
            return latest_session_summary(session)
        except Exception:
            return None

    def _build_updated_history(self, state: UrbanomyOrchestratorState) -> list[dict[str, str]]:
        history = list(state.get("history", []))
        response = self._result_response_text(state)
        history.append(
            {
                "user": str(state.get("user_request", "")).strip(),
                "assistant": response.strip(),
            }
        )
        return history[-12:]

    def _format_recent_history(self, state: UrbanomyOrchestratorState) -> str:
        history = list(state.get("history", []))
        if not history:
            return "История диалога пока пуста."
        lines: list[str] = []
        for item in history[-6:]:
            user_text = str(item.get("user", "")).strip()
            assistant_text = str(item.get("assistant", "")).strip()
            if user_text:
                lines.append(f"Пользователь: {user_text}")
            if assistant_text:
                lines.append(f"Ассистент: {assistant_text}")
        return "\n".join(lines)

    def _result_response_text(self, state: UrbanomyOrchestratorState) -> str:
        return str(state.get("latest_response", "")).strip()

    def _build_result_from_state(
        self,
        *,
        state: dict[str, Any],
        thread_id: str,
    ) -> UrbanomyOrchestratorResult:
        runtime_output = self._thread_runtime_outputs.get(thread_id, {})
        route = str(state.get("latest_route") or state.get("route") or "unsupported")
        response = str(state.get("latest_response") or "").strip()
        reasoning = str(state.get("reasoning") or "").strip()
        return UrbanomyOrchestratorResult(
            user_request=str(state.get("user_request", "")).strip(),
            route=route,
            reasoning=reasoning,
            response=response,
            visualization_result=runtime_output.get("visualization_result"),
            target_block_visualization_result=runtime_output.get("target_block_visualization_result"),
            block_parameters_result=runtime_output.get("block_parameters_result"),
            document_rag_result=runtime_output.get("document_rag_result"),
            district_optimization_result=runtime_output.get("district_optimization_result"),
            verification_report=self._thread_verification_reports.get(thread_id),
            session_memory=self._thread_session_memory.get(thread_id),
            research_brief=self._thread_research_briefs.get(thread_id),
            execution_plan=self._thread_active_plans.get(thread_id),
            confirmation_gate=self._thread_confirmation_gates.get(thread_id),
        )

    @staticmethod
    def _looks_like_block_parameters_request(user_request: str) -> bool:
        return looks_like_block_parameters_request(user_request)

    def _set_runtime_output(
        self,
        thread_id: str,
        *,
        visualization_result: Any,
        target_block_visualization_result: Any,
        block_parameters_result: Any,
        document_rag_result: Any,
        district_optimization_result: Any,
    ) -> None:
        self._thread_runtime_outputs[thread_id] = {
            "visualization_result": visualization_result,
            "target_block_visualization_result": target_block_visualization_result,
            "block_parameters_result": block_parameters_result,
            "document_rag_result": document_rag_result,
            "district_optimization_result": district_optimization_result,
        }

    def _record_request_analysis(
        self,
        *,
        thread_id: str,
        route: str,
        user_request: str,
    ) -> None:
        latest_summary = self._build_latest_optimization_summary(thread_id=thread_id)
        active_target_id = None
        if isinstance(latest_summary, dict):
            target_id = latest_summary.get("target_id")
            if isinstance(target_id, int):
                active_target_id = target_id
        latest_solution_number = None
        session_memory = self._thread_session_memory.get(thread_id)
        if session_memory is not None:
            latest_solution_number = getattr(session_memory, "latest_solution_number", None)
        research_brief = self._get_planning_agent().research(
            user_request=user_request,
            route=route,
            active_target_id=active_target_id,
            latest_solution_number=latest_solution_number,
        )
        self._thread_research_briefs[thread_id] = research_brief
        self._thread_active_plans[thread_id] = self._get_planning_agent().plan(
            research_brief=research_brief
        )

    def _maybe_create_confirmation_gate(
        self,
        *,
        thread_id: str,
        route: str,
        user_request: str,
    ) -> ConfirmationGate | None:
        research_brief = self._thread_research_briefs.get(thread_id)
        active_plan = self._thread_active_plans.get(thread_id)
        if research_brief is None or active_plan is None:
            return None
        if not self._requires_expert_confirmation(route=route, complexity=research_brief.complexity):
            return None
        tools = self._get_planning_agent().involved_tools(research_brief=research_brief)
        prompt = (
            "Перед выполнением нужен экспертный confirm-gate. "
            "Проверь, что я правильно понял запрос, шаги и инструменты."
        )
        return ConfirmationGate(
            status="pending",
            route=route,
            original_user_request=user_request,
            prompt=prompt,
            involved_tools=tools,
            research_brief=research_brief,
            execution_plan=active_plan,
        )

    @staticmethod
    def _requires_expert_confirmation(*, route: str, complexity: str) -> bool:
        return complexity == "multi_step" and route in {"district_optimization", "visualization"}

    @staticmethod
    def _interpret_confirmation_reply(user_request: str) -> str:
        text = normalize_text(user_request).strip()
        approve_markers = ("да", "ага", "ок", "окей", "верно", "подтверждаю", "yes")
        reject_markers = ("нет", "неа", "no", "отмена", "неверно")
        if any(
            text == marker or text.startswith(marker + " ") or text.startswith(marker + ",")
            for marker in approve_markers
        ):
            return "approve"
        for marker in reject_markers:
            if text == marker:
                return "reject"
            if text.startswith(marker + " ") or text.startswith(marker + ","):
                return "refine"
        return "refine"

    def _build_confirmation_result(
        self,
        *,
        user_request: str,
        route: str,
        reasoning: str,
        response: str,
        thread_id: str,
        confirmation_gate: ConfirmationGate,
    ) -> UrbanomyOrchestratorResult:
        return UrbanomyOrchestratorResult(
            user_request=user_request,
            route=route,
            reasoning=reasoning,
            response=response,
            verification_report=self._thread_verification_reports.get(thread_id),
            session_memory=self._thread_session_memory.get(thread_id),
            research_brief=self._thread_research_briefs.get(thread_id),
            execution_plan=self._thread_active_plans.get(thread_id),
            confirmation_gate=confirmation_gate,
        )

    @staticmethod
    def _route_decision_schema():
        from .models import UrbanomyOrchestratorRouteDecision

        return UrbanomyOrchestratorRouteDecision

    @staticmethod
    def _prediction_config_from_optimization(
        config: DistrictOptimizationConfig | None,
    ) -> LandValuePredictionConfig | None:
        if config is None:
            return None
        return LandValuePredictionConfig(
            model=config.model,
            orig_features=list(config.orig_features),
            categorical_features=list(config.categorical_features),
            use_service_features=config.use_service_features,
            service_features=(
                list(config.service_features) if config.service_features is not None else None
            ),
        )

    def _resolve_thread_id(self, thread_id: str | None) -> str:
        return str(thread_id or self.active_thread_id).strip() or self.default_thread_id

    @staticmethod
    def _graph_config(thread_id: str, checkpoint_ns: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}}


def create_urbanomy_orchestrator(
    *,
    llm: Any,
    baseline_blocks: gpd.GeoDataFrame,
    district_optimization_config: DistrictOptimizationConfig | None = None,
    land_value_prediction_config: LandValuePredictionConfig | None = None,
    checkpointer: Any | None = None,
    default_thread_id: str = "default",
    checkpoint_namespace: str = "urbanomy_orchestrator_v2",
) -> UrbanomyOrchestrator:
    """Factory for the top-level Urbanomy orchestrator."""
    return UrbanomyOrchestrator(
        llm=llm,
        baseline_blocks=baseline_blocks,
        district_optimization_config=district_optimization_config,
        land_value_prediction_config=land_value_prediction_config,
        checkpointer=checkpointer,
        default_thread_id=default_thread_id,
        checkpoint_namespace=checkpoint_namespace,
    )
