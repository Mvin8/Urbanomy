"""Deterministic intent classification for district optimization requests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ..common.request_parsing import extract_target_id, normalize_text


AlgorithmParameterName = Literal["pop_size", "n_gen", "seed"]

OPTIMIZATION_DOMAIN_ALIASES = (
    "оптимиз",
    "оптимизац",
    "оптимизатор",
    "алгоритм",
    "генетическ",
    "nsga",
    "pareto",
    "парето",
    "фронт",
)

SET_PARAMETER_ALIASES = (
    "измени",
    "поменяй",
    "установи",
    "обнови",
    "задай",
    "поставь",
    "сделай",
)

RUN_OPTIMIZATION_ALIASES = (
    "запусти оптимизацию",
    "запустить оптимизацию",
    "запусти оптимизатор",
    "оптимизируй",
    "оптимизировать",
    "найди pareto",
    "найди парето",
)

CHANGE_PARAMETER_ALIASES = (
    "измени",
    "поменяй",
    "смени",
    "установи",
    "задать",
    "задай",
    "поставь",
    "сделай",
)

PROBLEM_STATEMENT_ALIASES = (
    "постановк",
    "целевая функц",
    "переменн",
    "что изменить",
    "что поменять",
    "как настроить",
    "настроить оптимизацию",
    "параметры оптимизации",
    "параметры алгоритма",
    "параметры оптимизатора",
    "настройки оптимизатора",
    "настройки алгоритма",
    "параметры поиска",
)

CONSTRAINTS_ALIASES = (
    "ограничени",
    "ограничения оптимизации",
    "покажи ограничения",
    "диапазоны переменных",
    "границы переменных",
    "bounds",
    "constraints",
)

QUESTION_ALIASES = (
    "?",
    "какой",
    "какая",
    "какое",
    "какие",
    "сколько",
    "чему равно",
    "значени",
    "что такое",
    "что значит",
    "объясни",
    "поясни",
    "зачем",
    "для чего",
)

SUMMARY_ALIASES = (
    "сколько решений",
    "сколько сценариев",
    "какие решения",
    "какие сценарии",
    "покажи список решений",
    "список решений",
    "сколько найдено",
)

PARETO_FRONT_ALIASES = (
    "парето фронт",
    "pareto front",
    "pareto-front",
    "график парето",
    "диаграмм парето",
    "фронт парето",
    "scatter",
)

SOLUTION_PLOT_ALIASES = (
    "визуализируй решение",
    "покажи решение",
    "нарисуй решение",
    "построй решение",
    "визуализация решения",
)

INVESTMENT_ALIASES = (
    "инвестицион",
    "экономическ",
    "эффективност",
    "npv",
    "irr",
    "pi",
    "метрик",
)

SOLUTION_PARAMETER_ALIASES = (
    "параметры решения",
    "параметры квартала",
    "params_repaired",
    "params repaired",
)

DOCUMENT_REGISTRATION_ALIASES = (
    "добавь документ",
    "сохрани документ",
    "зарегистрируй документ",
    "загрузи документ",
    "добавь генплан",
    "добавь стратегию",
    "сохрани генплан",
    "сохрани стратегию",
    "текст документа",
    "document_text",
    "document_path",
)

ACTIVE_DOCUMENT_ALIASES = (
    "активный документ",
    "какой документ",
    "что за документ",
    "какой сейчас документ",
)

DOCUMENT_SELECTION_ALIASES = (
    "лучшее решение",
    "лучший сценарий",
    "рекомендуй решение",
    "на основе документа",
    "по документу",
    "по генплану",
    "по стратегии",
)

DOCUMENT_RETRIEVAL_ALIASES = (
    "фрагмент документа",
    "фрагменты документа",
    "найди в документе",
    "что в документе сказано",
    "покажи фрагменты",
    "покажи выдержки",
    "контекст документа",
    "rag",
    "retrieval",
)

ALGORITHM_PARAMETER_ALIASES: dict[AlgorithmParameterName, tuple[str, ...]] = {
    "pop_size": (
        "pop_size",
        "размер популяции",
        "популяц",
        "число особей",
        "количество особей",
        "population size",
    ),
    "n_gen": (
        "n_gen",
        "число поколений",
        "количество поколений",
        "поколен",
        "сколько поколений",
        "generations",
    ),
    "seed": (
        "seed",
        "сид",
        "зерно",
        "случайное зерно",
        "random seed",
    ),
}

ALGORITHM_PARAMETER_PATTERNS: dict[AlgorithmParameterName, tuple[str, ...]] = {
    "pop_size": (
        r"популяц\w*.*?\bна\s*(\d+)",
        r"pop_size\s*[:=]?\s*(\d+)",
        r"размер\w*\s+популяц\w*\D{0,12}(\d+)",
        r"числ\w*\s+особ\w*\D{0,12}(\d+)",
        r"количеств\w*\s+особ\w*\D{0,12}(\d+)",
    ),
    "n_gen": (
        r"поколен\w*.*?\bна\s*(\d+)",
        r"n_gen\s*[:=]?\s*(\d+)",
        r"(?:числ\w*|количеств\w*)\s+поколен\w*\D{0,12}(\d+)",
    ),
    "seed": (
        r"\bseed\s*[:=]?\s*(\d+)",
        r"\bсид\D{0,12}(\d+)",
        r"(?:случайн\w*\s+)?зерн\w*\D{0,12}(\d+)",
    ),
}


@dataclass(frozen=True)
class DistrictOptimizationIntent:
    """Parsed district optimization intent extracted from a user request."""

    kind: Literal[
        "run_optimization",
        "set_algorithm_parameter",
        "algorithm_parameter",
        "problem_statement",
        "constraints",
        "session_summary",
        "plot_pareto_front",
        "plot_solution",
        "investment_metrics",
        "solution_parameters",
        "register_decision_document",
        "active_decision_document",
        "retrieve_document_context",
        "select_solution_by_document",
        "unknown",
    ]
    target_id: int | None = None
    solution_number: int | None = None
    pop_size: int | None = None
    n_gen: int | None = None
    seed: int | None = None
    parameter_name: AlgorithmParameterName | None = None


def parse_district_optimization_intent(user_request: str) -> DistrictOptimizationIntent:
    """Parse a free-form district optimization request into a structured intent."""
    raw_text = str(user_request).strip()
    text = normalize_text(raw_text)

    parameter_name = detect_algorithm_parameter(text)
    if parameter_name is not None and is_algorithm_parameter_question(text):
        return DistrictOptimizationIntent(
            kind="algorithm_parameter",
            target_id=extract_target_id(raw_text),
            parameter_name=parameter_name,
        )

    if is_algorithm_parameter_update_request(text):
        updates = extract_algorithm_parameter_updates(raw_text)
        if updates:
            parameter_names = list(updates.keys())
            return DistrictOptimizationIntent(
                kind="set_algorithm_parameter",
                target_id=extract_target_id(raw_text),
                pop_size=updates.get("pop_size"),
                n_gen=updates.get("n_gen"),
                seed=updates.get("seed"),
                parameter_name=parameter_names[0] if len(parameter_names) == 1 else None,
            )

    if is_problem_statement_request(text):
        return DistrictOptimizationIntent(
            kind="constraints",
            target_id=extract_target_id(raw_text),
        ) if is_constraints_request(text) else DistrictOptimizationIntent(
            kind="problem_statement",
            target_id=extract_target_id(raw_text),
        )

    if is_constraints_request(text):
        return DistrictOptimizationIntent(
            kind="constraints",
            target_id=extract_target_id(raw_text),
        )

    if is_active_document_request(text):
        return DistrictOptimizationIntent(kind="active_decision_document")

    if is_document_registration_request(raw_text, text):
        return DistrictOptimizationIntent(kind="register_decision_document")

    if is_document_selection_request(text):
        return DistrictOptimizationIntent(
            kind="select_solution_by_document",
            solution_number=extract_solution_number(raw_text),
        )

    if is_document_retrieval_request(text):
        return DistrictOptimizationIntent(kind="retrieve_document_context")

    if is_run_optimization_request(raw_text, text):
        return DistrictOptimizationIntent(
            kind="run_optimization",
            target_id=extract_target_id(raw_text),
            pop_size=extract_algorithm_parameter_value("pop_size", raw_text),
            n_gen=extract_algorithm_parameter_value("n_gen", raw_text),
            seed=extract_algorithm_parameter_value("seed", raw_text),
        )

    if is_session_summary_request(text):
        return DistrictOptimizationIntent(kind="session_summary")

    if is_pareto_front_request(text):
        return DistrictOptimizationIntent(kind="plot_pareto_front")

    if is_solution_parameter_request(text):
        return DistrictOptimizationIntent(
            kind="solution_parameters",
            solution_number=extract_solution_number(raw_text),
        )

    if is_solution_plot_request(text):
        return DistrictOptimizationIntent(
            kind="plot_solution",
            solution_number=extract_solution_number(raw_text),
        )

    if is_investment_metrics_request(text):
        return DistrictOptimizationIntent(
            kind="investment_metrics",
            solution_number=extract_solution_number(raw_text),
        )

    return DistrictOptimizationIntent(kind="unknown")


def looks_like_district_optimization_request(user_request: str) -> bool:
    """Return whether the request appears to belong to district optimization."""
    text = normalize_text(user_request)
    intent = parse_district_optimization_intent(user_request)
    if intent.kind != "unknown":
        return True
    return has_any_alias(text, OPTIMIZATION_DOMAIN_ALIASES)


def has_any_alias(text: str, aliases: tuple[str, ...]) -> bool:
    """Check whether any normalized alias is present in text."""
    return any(alias in text for alias in aliases)
def extract_solution_number(user_request: str) -> int | None:
    match = re.search(r"решени(?:е|я|ю)\s*(\d+)", user_request, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"solution\s*(\d+)", user_request, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def extract_algorithm_parameter_value(
    parameter_name: AlgorithmParameterName,
    user_request: str,
) -> int | None:
    """Extract an explicit numeric override for one algorithm parameter."""
    for pattern in ALGORITHM_PARAMETER_PATTERNS[parameter_name]:
        match = re.search(pattern, user_request, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def extract_algorithm_parameter_updates(
    user_request: str,
) -> dict[AlgorithmParameterName, int]:
    """Extract all explicit algorithm parameter updates from a request."""
    updates: dict[AlgorithmParameterName, int] = {}
    for parameter_name in ALGORITHM_PARAMETER_PATTERNS:
        value = extract_algorithm_parameter_value(parameter_name, user_request)
        if value is not None:
            updates[parameter_name] = value
    return updates


def detect_algorithm_parameter(text: str) -> AlgorithmParameterName | None:
    """Return the first algorithm parameter alias found in a normalized text."""
    for parameter_name, aliases in ALGORITHM_PARAMETER_ALIASES.items():
        if has_any_alias(text, aliases):
            return parameter_name
    return None


def is_algorithm_parameter_question(text: str) -> bool:
    """Return whether the normalized text asks about one algorithm parameter."""
    return has_any_alias(text, QUESTION_ALIASES)


def is_algorithm_parameter_update_request(text: str) -> bool:
    """Return whether the normalized text updates algorithm settings."""
    return has_any_alias(text, CHANGE_PARAMETER_ALIASES) or has_any_alias(text, SET_PARAMETER_ALIASES)


def is_problem_statement_request(text: str) -> bool:
    """Return whether the normalized text asks for optimization setup details."""
    return has_any_alias(text, PROBLEM_STATEMENT_ALIASES)


def is_constraints_request(text: str) -> bool:
    """Return whether the normalized text explicitly asks for constraints/bounds."""
    return has_any_alias(text, CONSTRAINTS_ALIASES)


def is_run_optimization_request(raw_text: str, text: str) -> bool:
    """Return whether the request asks to run optimization."""
    return has_any_alias(text, RUN_OPTIMIZATION_ALIASES) or (
        has_any_alias(text, OPTIMIZATION_DOMAIN_ALIASES) and extract_target_id(raw_text) is not None
    )


def is_session_summary_request(text: str) -> bool:
    """Return whether the request asks about available optimization solutions."""
    return has_any_alias(text, SUMMARY_ALIASES)


def is_pareto_front_request(text: str) -> bool:
    """Return whether the request asks for a Pareto-front plot."""
    return has_any_alias(text, PARETO_FRONT_ALIASES)


def is_solution_plot_request(text: str) -> bool:
    """Return whether the request asks to visualize one specific solution."""
    return has_any_alias(text, SOLUTION_PLOT_ALIASES)


def is_investment_metrics_request(text: str) -> bool:
    """Return whether the request asks for investment metrics."""
    return has_any_alias(text, INVESTMENT_ALIASES)


def is_solution_parameter_request(text: str) -> bool:
    """Return whether the request asks for parameters of one specific solution."""
    return has_any_alias(text, SOLUTION_PARAMETER_ALIASES)


def is_document_registration_request(raw_text: str, text: str) -> bool:
    """Return whether the request appears to register a planning document."""
    return has_any_alias(text, DOCUMENT_REGISTRATION_ALIASES) and (
        ":" in raw_text or "\n" in raw_text or "document_path" in text
    )


def is_active_document_request(text: str) -> bool:
    """Return whether the request asks which planning document is active."""
    return has_any_alias(text, ACTIVE_DOCUMENT_ALIASES)


def is_document_selection_request(text: str) -> bool:
    """Return whether the request asks to choose a Pareto solution by document."""
    return has_any_alias(text, DOCUMENT_SELECTION_ALIASES)


def is_document_retrieval_request(text: str) -> bool:
    """Return whether the request asks for document-grounded evidence."""
    return has_any_alias(text, DOCUMENT_RETRIEVAL_ALIASES)
