"""Deterministic planning and research layer for Urbanomy requests."""

from __future__ import annotations

import re

from .models import ExecutionPlan, ExecutionPlanStep, ResearchBrief, ResearchEntity


class PlanningAgent:
    """Extract entities and build a compact execution plan for one request."""

    def research(
        self,
        *,
        user_request: str,
        route: str,
        active_target_id: int | None = None,
        latest_solution_number: int | None = None,
    ) -> ResearchBrief:
        text = str(user_request).strip()
        normalized = text.lower()
        requested_outputs = self._requested_outputs(normalized=normalized, route=route)
        requested_solution_number = self._extract_solution_number(text) or latest_solution_number
        resolved_target_id = self._extract_target_id(text) or active_target_id
        complexity = "multi_step" if len(requested_outputs) > 1 or self._looks_multi_step(normalized) else "simple"
        entities: list[ResearchEntity] = [ResearchEntity(kind="route", value=route)]
        if resolved_target_id is not None:
            entities.append(ResearchEntity(kind="target_id", value=str(resolved_target_id)))
        if requested_solution_number is not None:
            entities.append(
                ResearchEntity(kind="solution_number", value=str(requested_solution_number))
            )
        for item in requested_outputs:
            entities.append(ResearchEntity(kind="requested_output", value=item))
        summary = self._build_summary(
            route=route,
            complexity=complexity,
            target_id=resolved_target_id,
            requested_outputs=requested_outputs,
        )
        return ResearchBrief(
            route=route,
            complexity=complexity,
            summary=summary,
            active_target_id=resolved_target_id,
            requested_solution_number=requested_solution_number,
            requested_outputs=requested_outputs,
            entities=entities,
        )

    def plan(self, *, research_brief: ResearchBrief) -> ExecutionPlan:
        steps = self._steps_for_route(research_brief)
        return ExecutionPlan(
            route=research_brief.route,
            summary=research_brief.summary,
            complexity=research_brief.complexity,
            steps=steps,
        )

    def involved_tools(self, *, research_brief: ResearchBrief) -> list[str]:
        """Return the expected tool set for this researched request."""
        route = research_brief.route
        outputs = set(research_brief.requested_outputs)
        if route == "district_optimization":
            tools: list[str] = []
            if "запуск оптимизации" in outputs:
                tools.append("run_district_optimization")
            if "регистрация документа" in outputs:
                tools.append("register_decision_document")
            if "активный документ" in outputs:
                tools.append("get_active_decision_document")
            if "retrieval по документу" in outputs or "выбор решения по документу" in outputs:
                tools.append("retrieve_decision_document_context")
            if "объяснение ограничений" in outputs:
                tools.append("get_district_optimization_constraints")
            if "объяснение постановки задачи" in outputs:
                tools.append("get_district_optimization_problem_statement")
            if "инвестиционные метрики" in outputs:
                tools.append("calculate_pareto_solution_investment_metrics")
            if "параметры решения" in outputs:
                tools.append("get_pareto_solution_parameters")
            if "визуализация решения" in outputs:
                tools.append("plot_pareto_solution_impact")
            if "pareto front" in outputs:
                tools.append("plot_district_optimization_pareto_front")
            if "выбор решения по документу" in outputs:
                tools.append("select_pareto_solution_by_document")
            return list(dict.fromkeys(tools))
        if route == "document_rag":
            tools: list[str] = []
            if "регистрация документа" in outputs:
                tools.append("register_decision_document")
            if "активный документ" in outputs:
                tools.append("get_active_decision_document")
            if "retrieval по документу" in outputs:
                tools.append("retrieve_decision_document_context")
            if "выбор решения по документу" in outputs:
                tools.append("retrieve_decision_document_context")
                tools.append("select_pareto_solution_by_document")
            return list(dict.fromkeys(tools))
        if route == "visualization":
            if "карта квартала" in outputs:
                return ["plot_target_block_map"]
            if "карта стоимости за сотку" in outputs and "карта полной стоимости" in outputs:
                return ["plot_land_value_per_100m2_map", "plot_total_land_value_map"]
            if "карта стоимости за сотку" in outputs:
                return ["plot_land_value_per_100m2_map"]
            return ["plot_total_land_value_map"]
        if route == "block_parameters":
            return ["block_parameters"]
        return ["get_orchestrator_context", "get_orchestrator_tool_catalog"]

    @staticmethod
    def _extract_target_id(text: str) -> int | None:
        match = re.search(r"(?:target_id|id)\s*[:=]?\s*(\d+)", text, flags=re.IGNORECASE)
        if match is None:
            match = re.search(r"(?:квартал|блок)\D{0,20}(\d+)", text, flags=re.IGNORECASE)
        return int(match.group(1)) if match is not None else None

    @staticmethod
    def _extract_solution_number(text: str) -> int | None:
        match = re.search(r"(?:решени[ея]|solution)\D{0,12}(\d+)", text, flags=re.IGNORECASE)
        return int(match.group(1)) if match is not None else None

    @staticmethod
    def _looks_multi_step(normalized: str) -> bool:
        markers = ("сначала", "потом", "затем", "и", "также", "после этого", "а затем")
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _requested_outputs(*, normalized: str, route: str) -> list[str]:
        requested: list[str] = []
        if route == "visualization":
            if any(marker in normalized for marker in ("квартал", "блок", "target_id")):
                requested.append("карта квартала")
            if any(marker in normalized for marker in ("за сотку", "100 м2", "100м2", "удельн")):
                requested.append("карта стоимости за сотку")
            if any(marker in normalized for marker in ("общую стоимость", "полную стоимость", "стоимость участка")):
                requested.append("карта полной стоимости")
        elif route == "district_optimization":
            if any(marker in normalized for marker in ("оптимиз", "pareto", "парето")):
                requested.append("запуск оптимизации")
            if any(
                marker in normalized
                for marker in (
                    "document_text",
                    "document_path",
                    "добавь документ",
                    "сохрани документ",
                    "зарегистрируй документ",
                    "загрузи документ",
                )
            ):
                requested.append("регистрация документа")
            if any(
                marker in normalized
                for marker in ("активный документ", "какой документ", "что за документ")
            ):
                requested.append("активный документ")
            if any(
                marker in normalized
                for marker in (
                    "фрагмент",
                    "фрагменты",
                    "контекст документа",
                    "найди в документе",
                    "покажи выдержки",
                    "rag",
                    "retrieval",
                )
            ):
                requested.append("retrieval по документу")
            if any(marker in normalized for marker in ("огранич", "constraints")):
                requested.append("объяснение ограничений")
            if any(marker in normalized for marker in ("постановк", "problem statement", "целев")):
                requested.append("объяснение постановки задачи")
            if any(marker in normalized for marker in ("метрик", "npv", "инвести")):
                requested.append("инвестиционные метрики")
            if any(marker in normalized for marker in ("параметр", "params_repaired")):
                requested.append("параметры решения")
            if any(marker in normalized for marker in ("визуализ", "покажи", "impact", "карта")):
                requested.append("визуализация решения")
            if any(marker in normalized for marker in ("front", "фронт", "график")):
                requested.append("pareto front")
            if any(
                marker in normalized
                for marker in (
                    "лучшее решение",
                    "лучший сценарий",
                    "на основе документа",
                    "по документу",
                    "по генплану",
                    "по стратегии",
                    "рекомендуй решение",
                )
            ):
                requested.append("выбор решения по документу")
        elif route == "document_rag":
            if any(
                marker in normalized
                for marker in (
                    "document_text",
                    "document_path",
                    "добавь документ",
                    "сохрани документ",
                    "зарегистрируй документ",
                    "загрузи документ",
                    "новый документ",
                )
            ):
                requested.append("регистрация документа")
            if any(
                marker in normalized
                for marker in ("активный документ", "какой документ", "что за документ")
            ):
                requested.append("активный документ")
            if any(
                marker in normalized
                for marker in (
                    "фрагмент",
                    "фрагменты",
                    "контекст документа",
                    "найди в документе",
                    "что в документе сказано",
                    "покажи выдержки",
                    "достань из документа",
                    "согласно документ",
                    "rag",
                    "retrieval",
                )
            ):
                requested.append("retrieval по документу")
            if any(
                marker in normalized
                for marker in (
                    "лучшее решение",
                    "лучший сценарий",
                    "на основе документа",
                    "по документу",
                    "по генплану",
                    "по стратегии",
                    "рекомендуй решение",
                )
            ):
                requested.append("выбор решения по документу")
        elif route == "block_parameters":
            requested.append("параметры квартала")
        elif route == "general_qa":
            requested.append("объяснение")
            if any(marker in normalized for marker in ("что уме", "возможност", "инструмент")):
                requested.append("обзор возможностей")
            if any(marker in normalized for marker in ("контекст", "последн", "истори")):
                requested.append("обзор runtime контекста")
        return list(dict.fromkeys(requested or [route]))

    @staticmethod
    def _build_summary(
        *,
        route: str,
        complexity: str,
        target_id: int | None,
        requested_outputs: list[str],
    ) -> str:
        outputs = ", ".join(requested_outputs) if requested_outputs else route
        if target_id is not None:
            return (
                f"Нужно обработать {complexity} запрос по маршруту {route} для target_id={target_id}: "
                f"{outputs}."
            )
        return f"Нужно обработать {complexity} запрос по маршруту {route}: {outputs}."

    @staticmethod
    def _steps_for_route(research_brief: ResearchBrief) -> list[ExecutionPlanStep]:
        route = research_brief.route
        requested_outputs = list(research_brief.requested_outputs)
        if route == "district_optimization":
            steps = [
                ExecutionPlanStep(
                    step_id="1",
                    title="Уточнить optimization context",
                    rationale="Определить target_id, активную сессию и нужный тип результата.",
                )
            ]
            for index, item in enumerate(requested_outputs, start=2):
                rationale = "Выполнить соответствующий tool call в optimization-домене."
                if item == "регистрация документа":
                    rationale = "Сохранить документ, разбить его на чанки и построить внутренний RAG-индекс."
                elif item == "retrieval по документу":
                    rationale = "Достать релевантные фрагменты active policy-документа через RAG retrieval."
                elif item == "выбор решения по документу":
                    rationale = "Использовать retrieved фрагменты документа как evidence и выбрать лучшее Pareto-решение."
                steps.append(
                    ExecutionPlanStep(
                        step_id=str(index),
                        title=item.capitalize(),
                        rationale=rationale,
                    )
                )
            return steps
        if route == "document_rag":
            steps: list[ExecutionPlanStep] = []
            for index, item in enumerate(requested_outputs, start=1):
                rationale = "Выполнить соответствующий document-RAG tool call."
                if item == "регистрация документа":
                    rationale = "Сохранить документ, разбить его на чанки и построить RAG-индекс."
                elif item == "активный документ":
                    rationale = "Вернуть metadata активного документа и параметры индекса."
                elif item == "retrieval по документу":
                    rationale = "Достать релевантные фрагменты активного документа по смыслу запроса."
                elif item == "выбор решения по документу":
                    rationale = "Сопоставить retrieved evidence с Pareto-кандидатами и вернуть лучший вариант."
                steps.append(
                    ExecutionPlanStep(
                        step_id=str(index),
                        title=item.capitalize(),
                        rationale=rationale,
                    )
                )
            return steps or [
                ExecutionPlanStep(
                    step_id="1",
                    title="Обработать запрос по документу",
                    rationale="Выбрать нужный document-RAG инструмент и вернуть grounded-ответ.",
                )
            ]
        if route == "visualization":
            return [
                ExecutionPlanStep(
                    step_id="1",
                    title="Определить тип карты",
                    rationale="Понять, нужна ли карта квартала, полной стоимости или стоимости за сотку.",
                ),
                ExecutionPlanStep(
                    step_id="2",
                    title="Построить визуализацию",
                    rationale="Вызвать согласованный plotting tool и сохранить артефакт в session memory.",
                ),
            ]
        if route == "block_parameters":
            return [
                ExecutionPlanStep(
                    step_id="1",
                    title="Определить квартал и список полей",
                    rationale="Найти target_id и понять, нужны все параметры или только часть.",
                ),
                ExecutionPlanStep(
                    step_id="2",
                    title="Вернуть baseline параметры",
                    rationale="Собрать компактный ответ без geometry и лишних полей.",
                ),
            ]
        return [
            ExecutionPlanStep(
                step_id="1",
                title="Собрать релевантный контекст",
                rationale="Подтянуть capabilities, session memory и историю, если это нужно вопросу.",
            ),
            ExecutionPlanStep(
                step_id="2",
                title="Сформировать ответ",
                rationale="Дать краткое объяснение без выдуманных возможностей.",
            ),
        ]


def create_planning_agent() -> PlanningAgent:
    """Factory for the deterministic planning layer."""
    return PlanningAgent()
