"""Formatting helpers for district optimization agent responses."""

from __future__ import annotations

from typing import Any


def short_response_from_tool(*, tool_name: str, tool_output: dict[str, Any]) -> str:
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
