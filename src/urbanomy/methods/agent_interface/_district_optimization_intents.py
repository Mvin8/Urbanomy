"""Intent parsing helpers for district optimization requests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DistrictOptimizationIntent:
    """Parsed district optimization intent extracted from a user request."""

    kind: Literal[
        "run_optimization",
        "session_summary",
        "plot_pareto_front",
        "plot_solution",
        "investment_metrics",
        "solution_parameters",
        "unknown",
    ]
    target_id: int | None = None
    solution_number: int | None = None
    pop_size: int | None = None
    n_gen: int | None = None
    seed: int | None = None


def parse_district_optimization_intent(user_request: str) -> DistrictOptimizationIntent:
    """Parse a free-form district optimization request into a structured intent."""
    text = str(user_request).strip()
    lowered = text.lower()

    if _is_optimization_request(lowered):
        return DistrictOptimizationIntent(
            kind="run_optimization",
            target_id=extract_target_id(text),
            pop_size=extract_optional_int(r"pop_size\s*[:=]?\s*(\d+)", text),
            n_gen=extract_optional_int(r"n_gen\s*[:=]?\s*(\d+)", text),
            seed=extract_optional_int(r"seed\s*[:=]?\s*(\d+)", text),
        )

    if _is_summary_request(lowered):
        return DistrictOptimizationIntent(kind="session_summary")

    if _is_front_plot_request(lowered):
        return DistrictOptimizationIntent(kind="plot_pareto_front")

    if _is_parameters_request(lowered):
        return DistrictOptimizationIntent(
            kind="solution_parameters",
            solution_number=extract_solution_number(text),
        )

    if _is_plot_request(lowered):
        return DistrictOptimizationIntent(
            kind="plot_solution",
            solution_number=extract_solution_number(text),
        )

    if _is_investment_request(lowered):
        return DistrictOptimizationIntent(
            kind="investment_metrics",
            solution_number=extract_solution_number(text),
        )

    return DistrictOptimizationIntent(kind="unknown")


def extract_target_id(user_request: str) -> int | None:
    match = re.search(r"(target_id|id)\s*[:=]?\s*(\d+)", user_request, flags=re.IGNORECASE)
    if match:
        return int(match.group(2))
    match = re.search(r"(квартал|блок)\D{0,20}(\d+)", user_request, flags=re.IGNORECASE)
    if match:
        return int(match.group(2))
    return None


def extract_solution_number(user_request: str) -> int | None:
    match = re.search(r"решени(?:е|я|ю)\s*(\d+)", user_request, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"solution\s*(\d+)", user_request, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def extract_optional_int(pattern: str, user_request: str) -> int | None:
    match = re.search(pattern, user_request, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _is_optimization_request(text: str) -> bool:
    return any(marker in text for marker in ("оптимиз", "pareto", "парето", "nsga"))


def _is_summary_request(text: str) -> bool:
    return any(marker in text for marker in ("решени", "сценари")) and any(
        marker in text for marker in ("сколько", "какие", "какое", "какая", "какие есть", "покажи список", "список")
    )


def _is_parameters_request(text: str) -> bool:
    return any(
        marker in text
        for marker in ("параметр", "params_repaired", "params repaired", "параметры квартала")
    ) and ("решени" in text or "solution" in text)


def _is_plot_request(text: str) -> bool:
    return not _is_parameters_request(text) and not _is_front_plot_request(text) and any(
        marker in text for marker in ("визуализ", "покажи", "нарисуй", "построй")
    ) and ("решени" in text or "solution" in text)


def _is_investment_request(text: str) -> bool:
    return any(
        marker in text
        for marker in ("инвестицион", "экономическ", "эффективност", "npv", "irr", "pi", "метрик")
    ) and ("решени" in text or "solution" in text)


def _is_front_plot_request(text: str) -> bool:
    graph_markers = ("график", "диаграмм", "scatter", "фронт", "plot")
    optimization_markers = ("парето", "pareto", "оптимизац", "решени", "front")
    return any(marker in text for marker in graph_markers) and any(
        marker in text for marker in optimization_markers
    )
