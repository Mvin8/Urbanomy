"""Tool-calling agent for district optimization and Pareto-solution analysis."""

from __future__ import annotations

import json
from typing import Any

import geopandas as gpd
from langchain_core.messages import HumanMessage

from .internal.common.agent_utils import (
    build_tool_agent,
    extract_last_ai_message_text,
    extract_last_tool_message,
)
from .internal.common.domain_contracts import ToolDescriptor, describe_tool
from .internal.district_optimization.formatting import (
    compact_tool_output,
    optimization_parameter_response,
    short_response_from_tool,
)
from .internal.district_optimization.intents import parse_district_optimization_intent
from .internal.district_optimization.metadata import DISTRICT_OPTIMIZATION_CAPABILITY_LINES
from .models import DistrictOptimizationConfig
from .pareto_document_agent import create_pareto_document_agent
from .prompts import DISTRICT_OPTIMIZATION_AGENT_PROMPT
from .tools.internal.district_optimization import latest_session_or_error, latest_session_summary
from .tools.calculate_pareto_solution_investment_metrics import (
    make_calculate_pareto_solution_investment_metrics_tool,
)
from .tools.get_pareto_solution_parameters import make_get_pareto_solution_parameters_tool
from .tools.get_district_optimization_problem_statement import (
    make_get_district_optimization_problem_statement_tool,
)
from .tools.get_district_optimization_constraints import (
    make_get_district_optimization_constraints_tool,
)
from .tools.plot_district_optimization_pareto_front import (
    make_plot_district_optimization_pareto_front_tool,
)
from .tools.plot_pareto_solution_impact import make_plot_pareto_solution_impact_tool
from .tools.run_district_optimization import make_run_district_optimization_tool


class DistrictOptimizationAgent:
    """Stateful agent over district optimization tools."""

    def __init__(
        self,
        *,
        llm: Any,
        baseline_blocks: gpd.GeoDataFrame,
        optimization_config: DistrictOptimizationConfig,
    ) -> None:
        if llm is None:
            raise ValueError("llm is required to build DistrictOptimizationAgent.")
        self.llm = llm
        self.optimization_config = optimization_config
        self.session_store: dict[str, Any] = {}
        self._pareto_document_agent = create_pareto_document_agent(
            llm=self.llm,
            session_store=self.session_store,
        )
        self._run_tool = make_run_district_optimization_tool(
            baseline_blocks=baseline_blocks,
            optimization_config=optimization_config,
            session_store=self.session_store,
        )
        self._plot_tool = make_plot_pareto_solution_impact_tool(session_store=self.session_store)
        self._investment_tool = make_calculate_pareto_solution_investment_metrics_tool(
            session_store=self.session_store
        )
        self._parameters_tool = make_get_pareto_solution_parameters_tool(
            session_store=self.session_store
        )
        self._problem_statement_tool = make_get_district_optimization_problem_statement_tool(
            baseline_blocks=baseline_blocks,
            optimization_config=optimization_config,
            session_store=self.session_store,
        )
        self._constraints_tool = make_get_district_optimization_constraints_tool(
            baseline_blocks=baseline_blocks,
            optimization_config=optimization_config,
            session_store=self.session_store,
        )
        self._pareto_front_tool = make_plot_district_optimization_pareto_front_tool(
            session_store=self.session_store
        )
        self._tools = [
            self._run_tool,
            self._plot_tool,
            self._investment_tool,
            self._parameters_tool,
            self._problem_statement_tool,
            self._constraints_tool,
            self._pareto_front_tool,
            *self._pareto_document_agent.available_tools(),
        ]
        self._tools_by_name = {tool.name: tool for tool in self._tools}
        self.agent = build_tool_agent(
            llm=self.llm,
            tools=self._tools,
            system_prompt=DISTRICT_OPTIMIZATION_AGENT_PROMPT,
        )

    def invoke(self, user_request: str) -> dict[str, Any]:
        """Execute the optimization agent for a free-form user request."""
        text = str(user_request).strip()
        if not text:
            raise ValueError("user_request cannot be empty")
        fallback = self._recover_missing_tool_call(text)
        if fallback is not None:
            return fallback
        try:
            raw_result = self.agent.invoke({"messages": [HumanMessage(content=text)]})
        except Exception as exc:
            return self._error(f"Не удалось обработать optimization-запрос: {exc}")

        tool_name, tool_output = self._extract_last_tool_result(raw_result)
        if tool_output is not None:
            agent_text = extract_last_ai_message_text(raw_result)
            return {
                "status": "ok",
                "response": self._build_response_from_tool(
                    tool_name=tool_name or "",
                    tool_output=tool_output,
                    agent_text=agent_text,
                    user_request=text,
                ),
                "tool_name": tool_name,
                "tool_output": compact_tool_output(
                    tool_name=tool_name,
                    tool_output=tool_output,
                ),
                "used_tool_fallback": False,
            }

        agent_text = extract_last_ai_message_text(raw_result)
        return {
            "status": "ok",
            "response": agent_text or "Запрос обработан.",
            "tool_name": None,
            "tool_output": None,
            "used_tool_fallback": False,
        }

    def run(self, user_request: str) -> dict[str, Any]:
        return self.invoke(user_request)

    def ask(self, user_request: str) -> dict[str, Any]:
        return self.invoke(user_request)

    def __call__(self, user_request: str) -> dict[str, Any]:
        return self.invoke(user_request)

    def available_tools(self) -> tuple[Any, ...]:
        """Return tool instances owned by the optimization domain."""
        return tuple(self._tools)

    def tool_descriptors(self) -> list[ToolDescriptor]:
        """Return compact user-facing descriptions of domain tools."""
        return [describe_tool(tool) for tool in self.available_tools()]

    @staticmethod
    def capability_lines() -> list[str]:
        """Return user-facing capabilities exposed by this domain agent."""
        return list(DISTRICT_OPTIMIZATION_CAPABILITY_LINES)

    def _recover_missing_tool_call(self, user_request: str) -> dict[str, Any] | None:
        intent = parse_district_optimization_intent(user_request)

        if intent.kind == "unknown":
            return None
        if intent.kind == "session_summary":
            return self._session_summary_response()
        if intent.kind in {
            "register_decision_document",
            "active_decision_document",
            "retrieve_document_context",
            "select_solution_by_document",
        }:
            return self._pareto_document_agent.invoke(user_request)
        if intent.kind == "set_algorithm_parameter":
            return self._set_algorithm_parameters_response(intent)
        if intent.kind == "algorithm_parameter":
            return self._algorithm_parameter_response(intent.parameter_name)
        if intent.kind == "problem_statement":
            return self._invoke_named_tool(
                tool_name="get_district_optimization_problem_statement",
                payload={"target_id": intent.target_id},
                used_tool_fallback=True,
                user_request=user_request,
            )
        if intent.kind == "constraints":
            return self._invoke_named_tool(
                tool_name="get_district_optimization_constraints",
                payload={"target_id": intent.target_id},
                used_tool_fallback=True,
                user_request=user_request,
            )
        if intent.kind == "plot_pareto_front":
            return self._invoke_named_tool(
                tool_name="plot_district_optimization_pareto_front",
                payload={},
                used_tool_fallback=True,
                user_request=user_request,
            )
        if intent.kind == "run_optimization":
            resolved_target_id = intent.target_id
            used_latest_target = False
            if resolved_target_id is None:
                latest_session = self.session_store.get("latest_district_optimization_session")
                if latest_session is not None:
                    resolved_target_id = int(latest_session.target_id)
                    used_latest_target = True
                else:
                    return self._error(
                        "Не найден target_id и нет активной сессии. Укажите, например, target_id=86."
                    )
            response = self._invoke_named_tool(
                tool_name="run_district_optimization",
                payload={
                    "target_id": resolved_target_id,
                    "pop_size": intent.pop_size,
                    "n_gen": intent.n_gen,
                    "seed": intent.seed,
                },
                used_tool_fallback=True,
                user_request=user_request,
            )
            if used_latest_target and response.get("status") == "ok":
                response["response"] = (
                    f"Использую target_id={resolved_target_id} из текущей сессии.\n\n"
                    f"{response.get('response', '').strip()}"
                ).strip()
            return response
        if intent.solution_number is None:
            return self._error("Не найден номер решения. Укажите, например: 'решение 2'.")
        if intent.kind == "plot_solution":
            return self._invoke_named_tool(
                tool_name="plot_pareto_solution_impact",
                payload={"solution_number": intent.solution_number},
                used_tool_fallback=True,
                user_request=user_request,
            )
        if intent.kind == "investment_metrics":
            return self._invoke_named_tool(
                tool_name="calculate_pareto_solution_investment_metrics",
                payload={"solution_number": intent.solution_number},
                used_tool_fallback=True,
                user_request=user_request,
            )
        if intent.kind == "solution_parameters":
            return self._invoke_named_tool(
                tool_name="get_pareto_solution_parameters",
                payload={"solution_number": intent.solution_number},
                used_tool_fallback=True,
                user_request=user_request,
            )
        return None

    def _set_algorithm_parameters_response(
        self,
        intent,
    ) -> dict[str, Any]:
        updates = {
            key: value
            for key, value in {
                "pop_size": intent.pop_size,
                "n_gen": intent.n_gen,
                "seed": intent.seed,
            }.items()
            if value is not None
        }
        if not updates:
            return self._error("Не удалось определить новое значение параметра алгоритма.")
        current_overrides = dict(self.session_store.get("algorithm_overrides") or {})
        current_overrides.update(updates)
        self.session_store["algorithm_overrides"] = current_overrides
        settings = self._effective_algorithm_settings()
        updated = ", ".join(f"{key}={value}" for key, value in updates.items())
        return {
            "status": "ok",
            "response": (
                f"Параметры алгоритма обновлены: {updated}. "
                f"Теперь будут использоваться pop_size={settings['pop_size']}, "
                f"n_gen={settings['n_gen']}, seed={settings['seed']}."
            ),
            "tool_name": "set_algorithm_parameter",
            "tool_output": {
                "updated": updates,
                "effective_algorithm_settings": settings,
            },
            "used_tool_fallback": True,
        }

    def _algorithm_parameter_response(self, parameter_name: str | None) -> dict[str, Any]:
        if not parameter_name:
            return self._error("Не удалось определить параметр алгоритма.")
        try:
            tool_output = self._tools_by_name["get_district_optimization_problem_statement"].invoke({})
        except Exception as exc:
            return self._error(str(exc))
        return {
            "status": "ok",
            "response": optimization_parameter_response(
                tool_output=tool_output,
                parameter_name=parameter_name,
            ),
            "tool_name": "get_district_optimization_problem_statement",
            "tool_output": compact_tool_output(
                tool_name="get_district_optimization_problem_statement",
                tool_output=tool_output,
            ),
            "used_tool_fallback": True,
        }

    def _session_summary_response(self) -> dict[str, Any]:
        try:
            summary = latest_session_summary(latest_session_or_error(self.session_store))
        except Exception as exc:
            return self._error(str(exc))
        return {
            "status": "ok",
            "response": short_response_from_tool(tool_name="session_summary", tool_output=summary),
            "tool_name": "session_summary",
            "tool_output": compact_tool_output(tool_name="session_summary", tool_output=summary),
            "used_tool_fallback": True,
        }

    def _effective_algorithm_settings(self) -> dict[str, int]:
        overrides = dict(self.session_store.get("algorithm_overrides") or {})
        return {
            "pop_size": int(overrides.get("pop_size", self.optimization_config.pop_size)),
            "n_gen": int(overrides.get("n_gen", self.optimization_config.n_gen)),
            "seed": int(overrides.get("seed", self.optimization_config.seed)),
        }

    def _invoke_named_tool(
        self,
        *,
        tool_name: str,
        payload: dict[str, Any],
        used_tool_fallback: bool,
        user_request: str = "",
    ) -> dict[str, Any]:
        tool = self._tools_by_name[tool_name]
        clean_payload = {key: value for key, value in payload.items() if value is not None}
        try:
            tool_output = tool.invoke(clean_payload)
        except Exception as exc:
            return {
                "status": "error",
                "response": str(exc),
                "tool_name": tool_name,
                "tool_output": None,
                "used_tool_fallback": used_tool_fallback,
            }
        return {
            "status": "ok",
            "response": self._build_response_from_tool(
                tool_name=tool_name,
                tool_output=tool_output,
                agent_text="",
                user_request=user_request,
            ),
            "tool_name": tool_name,
            "tool_output": compact_tool_output(tool_name=tool_name, tool_output=tool_output),
            "used_tool_fallback": used_tool_fallback,
        }

    @staticmethod
    def _build_response_from_tool(
        *,
        tool_name: str,
        tool_output: dict[str, Any],
        agent_text: str,
        user_request: str,
    ) -> str:
        base = short_response_from_tool(
            tool_name=tool_name,
            tool_output=tool_output,
            user_request=user_request,
        )
        ai_text = str(agent_text).strip()
        return base or ai_text or "Запрос обработан."

    def _extract_last_tool_result(self, raw_result: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
        message = extract_last_tool_message(raw_result)
        if message is None:
            return None, None
        name = getattr(message, "name", None)
        parsed = self._parse_tool_message_content(message.content)
        return str(name) if name else None, parsed

    @staticmethod
    def _parse_tool_message_content(content: Any) -> dict[str, Any] | None:
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    @staticmethod
    def _error(message: str) -> dict[str, Any]:
        return {
            "status": "error",
            "response": message,
            "tool_name": None,
            "tool_output": None,
            "used_tool_fallback": False,
        }


def create_district_optimization_agent(
    *,
    llm: Any,
    baseline_blocks: gpd.GeoDataFrame,
    model: Any,
    orig_features: list[str] | tuple[str, ...],
    categorical_features: list[str] | tuple[str, ...],
    constraints_template: dict[str, dict[str, Any]] | None = None,
    target_id_column: str = "id",
    pop_size: int = 10,
    n_gen: int = 15,
    seed: int = 42,
    eliminate_duplicates: bool = True,
    save_history: bool = True,
    use_history: bool = False,
    verbose: bool = True,
    use_service_features: bool | None = None,
    service_features: list[str] | tuple[str, ...] | None = None,
) -> DistrictOptimizationAgent:
    """Factory for the district optimization agent."""
    optimization_config = DistrictOptimizationConfig(
        model=model,
        orig_features=list(orig_features),
        categorical_features=list(categorical_features),
        constraints_template=constraints_template,
        target_id_column=target_id_column,
        pop_size=pop_size,
        n_gen=n_gen,
        seed=seed,
        eliminate_duplicates=eliminate_duplicates,
        save_history=save_history,
        use_history=use_history,
        verbose=verbose,
        use_service_features=use_service_features,
        service_features=list(service_features) if service_features is not None else None,
    )
    return DistrictOptimizationAgent(
        llm=llm,
        baseline_blocks=baseline_blocks,
        optimization_config=optimization_config,
    )
