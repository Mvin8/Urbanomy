"""Formatting helpers for district optimization agent responses."""

from __future__ import annotations

from typing import Any


def optimization_parameter_response(
    *,
    tool_output: dict[str, Any],
    parameter_name: str,
) -> str:
    """Build a short direct answer for one algorithm parameter."""
    algorithm = dict((tool_output.get("tunable_parameters") or {}).get("algorithm") or {})
    values = {
        "pop_size": algorithm.get("pop_size_default"),
        "n_gen": algorithm.get("n_gen_default"),
        "seed": algorithm.get("seed_default"),
    }
    descriptions = {
        "pop_size": "размер популяции генетического алгоритма",
        "n_gen": "число поколений оптимизации",
        "seed": "seed генератора случайности",
    }
    value = values.get(parameter_name)
    description = descriptions.get(parameter_name, "параметр алгоритма")
    if value is None:
        return f"Не удалось определить текущее значение `{parameter_name}`."
    return f"`{parameter_name}` = {value}. Это {description}."


def wants_detailed_problem_statement(user_request: str) -> bool:
    """Return whether the user explicitly asks for the full optimization setup."""
    text = str(user_request).lower().replace("ё", "е")
    detail_markers = (
        "все параметры",
        "все настройки",
        "полные параметры",
        "полная постановка",
        "подробно",
        "подробнее",
        "целиком",
        "полностью",
        "все ограничения",
        "все переменные",
    )
    return any(marker in text for marker in detail_markers)


def short_response_from_tool(
    *,
    tool_name: str,
    tool_output: dict[str, Any],
    user_request: str = "",
) -> str:
    """Build a short user-facing response from a tool output payload."""
    if tool_name == "run_district_optimization":
        base = f"Оптимизация завершена. Найдено {format_solution_count(tool_output['n_solutions'])}."
        front_text = str(tool_output.get("pareto_front_summary_text", "")).strip()
        return f"{base}\n\n{front_text}" if front_text else base
    if tool_name == "session_summary":
        return f"Найдено {format_solution_count(tool_output['n_solutions'])}."
    if tool_name == "plot_pareto_solution_impact":
        text = str(tool_output.get("summary_text", "")).strip()
        return text or (
            f"Построена визуализация решения {tool_output['solution_number']} "
            f"для target_id={tool_output['target_id']}."
        )
    if tool_name == "calculate_pareto_solution_investment_metrics":
        text = str(tool_output.get("project_totals_text", "")).strip()
        return text or (
            f"Посчитаны инвестиционные метрики для решения {tool_output['solution_number']} "
            f"для target_id={tool_output['target_id']}."
        )
    if tool_name == "get_pareto_solution_parameters":
        text = str(tool_output.get("params_text", "")).strip()
        return text or (
            f"Показаны параметры квартала для решения {tool_output['solution_number']} "
            f"для target_id={tool_output['target_id']}."
        )
    if tool_name == "plot_district_optimization_pareto_front":
        text = str(tool_output.get("summary_text", "")).strip()
        return text or "Построен график Парето-фронта."
    if tool_name == "get_district_optimization_problem_statement":
        if wants_detailed_problem_statement(user_request):
            text = str(tool_output.get("problem_statement_text", "")).strip()
            if text:
                return text
        algorithm = dict((tool_output.get("tunable_parameters") or {}).get("algorithm") or {})
        variables = list(tool_output.get("decision_variables") or [])
        variable_names = ", ".join(str(item.get("name")) for item in variables[:4] if item.get("name"))
        target_id = tool_output.get("target_id")
        parts = [
            "Постановка задачи оптимизации готова.",
            f"target_id={target_id}." if target_id is not None else "target_id пока не задан.",
            f"Переменных: {len(variables)}.",
        ]
        if variable_names:
            parts.append(f"Основные переменные: {variable_names}.")
        if algorithm:
            parts.append(
                "Параметры алгоритма: "
                f"pop_size={algorithm.get('pop_size_default')}, "
                f"n_gen={algorithm.get('n_gen_default')}, "
                f"seed={algorithm.get('seed_default')}."
            )
        return " ".join(parts)
    return "Запрос обработан."


def compact_tool_output(
    *,
    tool_name: str | None,
    tool_output: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Reduce heavy tool outputs to notebook-friendly compact payloads."""
    if tool_name is None or tool_output is None:
        return tool_output
    if tool_name in {"run_district_optimization", "session_summary"}:
        return {
            "target_id": tool_output.get("target_id"),
            "site_area": tool_output.get("site_area"),
            "n_solutions": tool_output.get("n_solutions"),
            "pareto_front_summary_text": tool_output.get("pareto_front_summary_text"),
            "pareto_front_plotted": tool_output.get("pareto_front_plotted"),
        }
    if tool_name == "calculate_pareto_solution_investment_metrics":
        return {
            "solution_number": tool_output.get("solution_number"),
            "target_id": tool_output.get("target_id"),
            "project_totals_text": tool_output.get("project_totals_text"),
        }
    if tool_name == "get_pareto_solution_parameters":
        return {
            "solution_number": tool_output.get("solution_number"),
            "target_id": tool_output.get("target_id"),
            "params_repaired": tool_output.get("params_repaired"),
            "params_text": tool_output.get("params_text"),
        }
    if tool_name == "plot_pareto_solution_impact":
        return {
            "solution_number": tool_output.get("solution_number"),
            "target_id": tool_output.get("target_id"),
            "summary_text": tool_output.get("summary_text"),
            "figure_created": tool_output.get("figure_created"),
        }
    if tool_name == "plot_district_optimization_pareto_front":
        return {
            "target_id": tool_output.get("target_id"),
            "n_points": tool_output.get("n_points"),
            "n_pareto_points": tool_output.get("n_pareto_points"),
            "land_use_labels": tool_output.get("land_use_labels"),
            "summary_text": tool_output.get("summary_text"),
            "figure_created": tool_output.get("figure_created"),
        }
    if tool_name == "get_district_optimization_problem_statement":
        return {
            "target_id": tool_output.get("target_id"),
            "site_area": tool_output.get("site_area"),
            "source": tool_output.get("source"),
            "variable_names": tool_output.get("variable_names"),
            "decision_variables": tool_output.get("decision_variables"),
            "objectives": tool_output.get("objectives"),
            "repair_rules": tool_output.get("repair_rules"),
            "tunable_parameters": tool_output.get("tunable_parameters"),
            "runtime_overrides_active": tool_output.get("runtime_overrides_active"),
        }
    return tool_output


def format_solution_count(value: Any) -> str:
    """Format solution count with a grammatically correct Russian suffix."""
    count = int(value)
    last_two = count % 100
    last_one = count % 10
    if 11 <= last_two <= 14:
        suffix = "оптимальных решений"
    elif last_one == 1:
        suffix = "оптимальное решение"
    elif 2 <= last_one <= 4:
        suffix = "оптимальных решения"
    else:
        suffix = "оптимальных решений"
    return f"{count} {suffix}"
