"""Session memory and task-ledger helpers for the Urbanomy orchestrator."""

from __future__ import annotations

from typing import Any

from ...models import (
    ConfirmationGate,
    ExecutionPlan,
    ParetoDocumentSelectionResult,
    ResearchBrief,
    SessionArtifact,
    SessionMemorySnapshot,
    SessionTask,
    StrategyDecisionDocument,
    VerificationReport,
)


def build_session_task(
    *,
    thread_id: str,
    task_index: int,
    route: str,
    user_request: str,
    response: str,
    verification_report: VerificationReport | None,
) -> SessionTask:
    """Create one compact task entry for the finished user turn."""
    verification_status = verification_report.status if verification_report is not None else "ok"
    status = "failed" if verification_status == "error" else "completed"
    return SessionTask(
        task_id=f"{thread_id}-{task_index}",
        route=route,
        title=_build_task_title(route=route, user_request=user_request),
        status=status,
        user_request=str(user_request).strip(),
        response_summary=_shorten_text(response, limit=220),
        verification_status=verification_status,
    )


def build_session_memory_snapshot(
    *,
    thread_id: str,
    latest_route: str,
    latest_response: str,
    latest_solution_number: int | None,
    latest_optimization_summary: dict[str, Any] | None,
    algorithm_overrides: dict[str, int] | None,
    runtime_output: dict[str, Any] | None,
    recent_tasks: list[SessionTask],
    verification_report: VerificationReport | None,
    research_brief: ResearchBrief | None,
    active_plan: ExecutionPlan | None,
    pending_confirmation_gate: ConfirmationGate | None,
    active_decision_document: StrategyDecisionDocument | None,
    latest_document_selection: ParetoDocumentSelectionResult | None,
) -> SessionMemorySnapshot:
    """Build a durable snapshot of the current thread context."""
    runtime_output = dict(runtime_output or {})
    artifacts = _extract_artifacts(
        latest_solution_number=latest_solution_number,
        latest_optimization_summary=latest_optimization_summary,
        runtime_output=runtime_output,
    )
    active_target_id = _resolve_active_target_id(
        latest_optimization_summary=latest_optimization_summary,
        runtime_output=runtime_output,
    )
    latest_visualization_route = None
    visualization_result = runtime_output.get("visualization_result")
    if visualization_result is not None:
        latest_visualization_route = str(getattr(visualization_result, "route", "")).strip() or None

    durable_facts: list[str] = []
    if active_target_id is not None:
        durable_facts.append(f"В фокусе находится квартал target_id={active_target_id}.")
    if latest_solution_number is not None:
        durable_facts.append(f"Последним упоминалось решение Pareto №{latest_solution_number}.")
    recommended_solution = None
    if latest_document_selection is not None:
        recommended_solution = latest_document_selection.selected_solution_number
        durable_facts.append(
            f"По активному документу последним рекомендовано решение Pareto №{recommended_solution}."
        )
    if latest_visualization_route:
        durable_facts.append(f"Последний визуальный артефакт получен через {latest_visualization_route}.")
    if latest_optimization_summary:
        n_solutions = latest_optimization_summary.get("n_solutions")
        if isinstance(n_solutions, int):
            durable_facts.append(f"В активной optimization session доступно решений: {n_solutions}.")
    if algorithm_overrides:
        overrides = ", ".join(f"{key}={value}" for key, value in sorted(algorithm_overrides.items()))
        durable_facts.append(f"Для алгоритма заданы overrides: {overrides}.")
    if active_decision_document is not None:
        durable_facts.append(
            f"Активный policy-документ: {active_decision_document.document_name}."
        )

    return SessionMemorySnapshot(
        thread_id=thread_id,
        latest_route=latest_route or None,
        active_target_id=active_target_id,
        latest_solution_number=latest_solution_number,
        latest_recommended_solution_number=recommended_solution,
        latest_visualization_route=latest_visualization_route,
        active_decision_document_name=(
            active_decision_document.document_name if active_decision_document is not None else None
        ),
        latest_response=_shorten_text(latest_response, limit=280),
        algorithm_overrides=dict(algorithm_overrides or {}) or None,
        recent_tasks=list(recent_tasks[-6:]),
        recent_artifacts=artifacts[-6:],
        latest_verification_report=verification_report,
        latest_research_brief=research_brief,
        active_plan=active_plan,
        pending_confirmation_gate=pending_confirmation_gate,
        durable_facts=durable_facts,
    )


def format_recent_tasks(tasks: list[SessionTask]) -> str:
    """Render recent task ledger entries for QA context."""
    if not tasks:
        return "Задачи сессии пока не накоплены."
    return "\n".join(
        f"- [{task.status}] {task.title} | route={task.route} | verify={task.verification_status}"
        for task in tasks[-6:]
    )


def _resolve_active_target_id(
    *,
    latest_optimization_summary: dict[str, Any] | None,
    runtime_output: dict[str, Any],
) -> int | None:
    if latest_optimization_summary:
        target_id = latest_optimization_summary.get("target_id")
        if isinstance(target_id, int):
            return target_id
    block_result = runtime_output.get("block_parameters_result")
    if isinstance(block_result, dict):
        target_id = block_result.get("target_id")
        if isinstance(target_id, int):
            return target_id
    visualization_result = runtime_output.get("visualization_result")
    if visualization_result is not None:
        target_id = getattr(visualization_result, "target_id", None)
        if isinstance(target_id, int):
            return target_id
    return None


def _extract_artifacts(
    *,
    latest_solution_number: int | None,
    latest_optimization_summary: dict[str, Any] | None,
    runtime_output: dict[str, Any],
) -> list[SessionArtifact]:
    artifacts: list[SessionArtifact] = []
    visualization_result = runtime_output.get("visualization_result")
    if visualization_result is not None:
        route = str(getattr(visualization_result, "route", "")).strip()
        payload = dict(getattr(visualization_result, "tool_payload", {}) or {})
        artifacts.append(
            SessionArtifact(
                kind="visualization",
                label=_shorten_text(str(getattr(visualization_result, "title", route)), limit=90),
                route=route,
                target_id=getattr(visualization_result, "target_id", None),
                payload=payload,
            )
        )
    block_result = runtime_output.get("block_parameters_result")
    if isinstance(block_result, dict) and block_result.get("status") == "ok":
        target_id = block_result.get("target_id")
        artifacts.append(
            SessionArtifact(
                kind="block_parameters",
                label=f"Параметры квартала id={target_id}",
                route="block_parameters",
                target_id=target_id if isinstance(target_id, int) else None,
                payload=dict(block_result.get("parameters") or {}),
            )
        )
    optimization_result = runtime_output.get("district_optimization_result")
    if isinstance(optimization_result, dict):
        tool_name = str(optimization_result.get("tool_name") or "district_optimization").strip()
        payload = optimization_result.get("tool_output")
        if isinstance(payload, dict):
            artifacts.append(
                SessionArtifact(
                    kind="optimization",
                    label=f"Optimization: {tool_name}",
                    route="district_optimization",
                    target_id=_safe_int((latest_optimization_summary or {}).get("target_id")),
                    solution_number=_safe_int(payload.get("solution_number"))
                    if latest_solution_number is not None
                    else _safe_int(payload.get("solution_number")),
                    payload=dict(payload),
                )
            )
            if tool_name == "register_decision_document":
                artifacts.append(
                    SessionArtifact(
                        kind="policy_document",
                        label=f"Policy document: {payload.get('document_name')}",
                        route="district_optimization",
                        payload=dict(payload),
                    )
                )
            if tool_name == "select_pareto_solution_by_document":
                artifacts.append(
                    SessionArtifact(
                        kind="document_selection",
                        label=f"Document-selected solution #{payload.get('selected_solution_number')}",
                        route="district_optimization",
                        target_id=_safe_int((latest_optimization_summary or {}).get("target_id")),
                        solution_number=_safe_int(payload.get("selected_solution_number")),
                        payload=dict(payload),
                    )
                )
    document_rag_result = runtime_output.get("document_rag_result")
    if isinstance(document_rag_result, dict):
        tool_name = str(document_rag_result.get("tool_name") or "document_rag").strip()
        payload = document_rag_result.get("tool_output")
        if isinstance(payload, dict):
            artifacts.append(
                SessionArtifact(
                    kind="document_rag",
                    label=f"Document RAG: {tool_name}",
                    route="document_rag",
                    solution_number=_safe_int(payload.get("selected_solution_number"))
                    or _safe_int(payload.get("solution_number")),
                    payload=dict(payload),
                )
            )
            if tool_name == "register_decision_document":
                artifacts.append(
                    SessionArtifact(
                        kind="policy_document",
                        label=f"Policy document: {payload.get('document_name')}",
                        route="document_rag",
                        payload=dict(payload),
                    )
                )
            if tool_name == "select_pareto_solution_by_document":
                artifacts.append(
                    SessionArtifact(
                        kind="document_selection",
                        label=f"Document-selected solution #{payload.get('selected_solution_number')}",
                        route="document_rag",
                        solution_number=_safe_int(payload.get("selected_solution_number")),
                        payload=dict(payload),
                    )
                )
    return artifacts


def _build_task_title(*, route: str, user_request: str) -> str:
    prefix = {
        "visualization": "Визуализация",
        "district_optimization": "Оптимизация",
        "document_rag": "Документы",
        "block_parameters": "Параметры квартала",
        "general_qa": "Вопрос",
        "unsupported": "Неподдерживаемый запрос",
    }.get(route, "Запрос")
    return f"{prefix}: {_shorten_text(user_request, limit=80)}"


def _shorten_text(text: str, *, limit: int) -> str:
    value = " ".join(str(text).split()).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _safe_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None
