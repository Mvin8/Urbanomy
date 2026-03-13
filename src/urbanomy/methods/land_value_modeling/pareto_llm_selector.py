"""Helpers for sending Pareto-front scenarios to an LLM."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np


DEFAULT_SELECTION_GOAL = (
    "Выбери лучший сценарий развития территории из Pareto-фронта. "
    "Нужно сохранить баланс между выгодой для города и инвестиционной привлекательностью. "
    "Ответ должен быть коротким и прикладным."
)

DEFAULT_MULTIAGENT_GOAL = (
    "Выбери лучший сценарий развития территории из Pareto-фронта. "
    "Рассматривай в первую очередь сценарии, где NPV инвестора положительный "
    "и прирост суммарной стоимости всей земли положительный. "
    "Учитывай также предполагаемую значимость сценария для города."
)


def collect_pareto_scenarios(
    problem: Any,
    res: Any,
    baseline_land_value: float | None = None,
) -> list[dict[str, Any]]:
    """
    Convert Pareto-front solutions from pymoo result into repaired parameter sets.

    Parameters
    ----------
    problem
        Optimization problem with ``var_names`` (or ``constraints``) and
        ``_repair_genome``.
    res
        Result returned by ``pymoo.optimize.minimize``.
    baseline_land_value
        Optional baseline land value. If omitted, only Pareto objectives are used.
    """
    X = np.atleast_2d(np.asarray(res.X, dtype=float))
    F = np.atleast_2d(np.asarray(res.F, dtype=float))
    var_names = list(getattr(problem, "var_names", None) or problem.constraints.keys())

    scenarios: list[dict[str, Any]] = []
    for idx, (genome_vec, objectives) in enumerate(zip(X, F)):
        params_raw = {name: float(genome_vec[j]) for j, name in enumerate(var_names)}
        params_repaired = _to_jsonable(problem._repair_genome(params_raw.copy()))

        land_value_total = float(-objectives[0])
        investor_npv = float(-objectives[1])
        admin_gain = None if baseline_land_value is None else float(land_value_total - baseline_land_value)
        growth_pct = None
        if baseline_land_value is not None and baseline_land_value > 0:
            growth_pct = float(admin_gain / baseline_land_value * 100.0)

        scenarios.append(
            {
                "scenario_id": idx,
                "objectives": {
                    "land_value_total": land_value_total,
                    "investor_npv": investor_npv,
                    "admin_gain_vs_baseline": admin_gain,
                    "land_value_growth_total_pct": growth_pct,
                },
                "params_raw": _to_jsonable(params_raw),
                "params_repaired": params_repaired,
            }
        )

    return scenarios


def select_best_pareto_scenario(
    problem: Any,
    res: Any,
    llm: Any,
    baseline_land_value: float | None = None,
    selection_goal: str = DEFAULT_SELECTION_GOAL,
) -> dict[str, Any]:
    """
    Ask an LLM to choose the best scenario from the Pareto-front.

    Returns a dictionary with the selected scenario, reason, raw LLM answer and
    the full list of prepared scenarios.
    """
    scenarios = collect_pareto_scenarios(
        problem=problem,
        res=res,
        baseline_land_value=baseline_land_value,
    )
    if not scenarios:
        raise ValueError("Pareto-front is empty: there are no scenarios to send to the LLM.")

    prompt = _build_selection_prompt(scenarios=scenarios, selection_goal=selection_goal)
    raw_response = _invoke_llm(llm=llm, prompt=prompt)
    parsed = _parse_json_object(raw_response)

    scenario_id = _resolve_scenario_id(parsed, scenarios)
    if scenario_id is None:
        scenario_id = _fallback_balanced_choice(scenarios)

    best_scenario = next(item for item in scenarios if item["scenario_id"] == scenario_id)
    reason = _extract_reason(parsed, raw_response)

    return {
        "best_scenario": best_scenario,
        "reason": reason,
        "selected_scenario_id": scenario_id,
        "llm_response_raw": raw_response,
        "all_scenarios": scenarios,
    }


def select_best_pareto_scenario_multiagent(
    problem: Any,
    res: Any,
    llm: Any,
    baseline_land_value: float,
    selection_goal: str = DEFAULT_MULTIAGENT_GOAL,
) -> dict[str, Any]:
    """
    Multi-agent Pareto scenario selection using resident, investor and city-administration roles.

    The agents first review only scenarios with positive investor NPV and positive
    total land-value growth. Then an arbiter selects the final scenario.
    """
    scenarios = collect_pareto_scenarios(
        problem=problem,
        res=res,
        baseline_land_value=baseline_land_value,
    )
    if not scenarios:
        raise ValueError("Pareto-front is empty: there are no scenarios to send to the LLM.")

    candidate_scenarios = _filter_positive_scenarios(scenarios)
    if not candidate_scenarios:
        candidate_scenarios = scenarios

    resident_prompt = _build_role_prompt(
        role_name="житель",
        role_goal=(
            "Оцени сценарии с позиции качества городской среды, удобства для жителей, "
            "сбалансированности функций, уместной плотности и полезности для повседневной жизни."
        ),
        scenarios=candidate_scenarios,
        selection_goal=selection_goal,
    )
    investor_prompt = _build_role_prompt(
        role_name="инвестор",
        role_goal=(
            "Оцени сценарии с позиции финансовой реализуемости. "
            "Смотри на положительный NPV, устойчивость параметров и практичность сценария."
        ),
        scenarios=candidate_scenarios,
        selection_goal=selection_goal,
    )
    admin_prompt = _build_role_prompt(
        role_name="городской администратор",
        role_goal=(
            "Оцени сценарии с позиции интересов города. "
            "Смотри на положительный прирост суммарной стоимости всей земли, "
            "процентный рост этой стоимости и предполагаемую значимость сценария для города: "
            "новые функции, деловая активность, рекреация, качество среды и стратегическая полезность."
        ),
        scenarios=candidate_scenarios,
        selection_goal=selection_goal,
    )

    resident_raw = _invoke_llm(llm=llm, prompt=resident_prompt)
    investor_raw = _invoke_llm(llm=llm, prompt=investor_prompt)
    admin_raw = _invoke_llm(llm=llm, prompt=admin_prompt)

    resident_opinion = _parse_agent_opinion(resident_raw)
    investor_opinion = _parse_agent_opinion(investor_raw)
    admin_opinion = _parse_agent_opinion(admin_raw)

    arbiter_prompt = _build_arbiter_prompt(
        scenarios=candidate_scenarios,
        selection_goal=selection_goal,
        resident_opinion=resident_opinion,
        investor_opinion=investor_opinion,
        admin_opinion=admin_opinion,
    )
    arbiter_raw = _invoke_llm(llm=llm, prompt=arbiter_prompt)
    arbiter_parsed = _parse_json_object(arbiter_raw)

    scenario_id = _resolve_scenario_id(arbiter_parsed, candidate_scenarios)
    if scenario_id is None:
        scenario_id = _fallback_positive_choice(candidate_scenarios)

    best_scenario = next(item for item in candidate_scenarios if item["scenario_id"] == scenario_id)
    reason = _extract_reason(arbiter_parsed, arbiter_raw)

    return {
        "best_scenario": best_scenario,
        "reason": reason,
        "selected_scenario_id": scenario_id,
        "agent_opinions": {
            "resident": resident_opinion,
            "investor": investor_opinion,
            "administrator": admin_opinion,
        },
        "llm_response_raw": {
            "resident": resident_raw,
            "investor": investor_raw,
            "administrator": admin_raw,
            "arbiter": arbiter_raw,
        },
        "candidate_scenarios": candidate_scenarios,
        "all_scenarios": scenarios,
    }


def _build_selection_prompt(scenarios: list[dict[str, Any]], selection_goal: str) -> str:
    llm_scenarios = [_to_llm_scenario(item) for item in scenarios]
    return f"""
Ты помогаешь выбрать лучший сценарий развития территории.

Задача:
{selection_goal}

Ниже список сценариев из Pareto-фронта. У каждого сценария есть:
- scenario_id
- objectives: целевые метрики
- params_repaired: уже отремонтированные параметры, которые можно применять дальше

Сценарии:
{json.dumps(llm_scenarios, ensure_ascii=False, indent=2)}

Выбери ровно один лучший сценарий и верни только JSON такого вида:
{{
  "scenario_id": <целое число>,
  "reason": "краткое объяснение на русском, 1-3 предложения"
}}

Без markdown.
Без текста до или после JSON.
""".strip()


def _build_role_prompt(
    *,
    role_name: str,
    role_goal: str,
    scenarios: list[dict[str, Any]],
    selection_goal: str,
) -> str:
    llm_scenarios = [_to_llm_scenario(item) for item in scenarios]
    return f"""
Ты выступаешь в роли: {role_name}.

Общая задача:
{selection_goal}

Оцени только эти сценарии. Это уже отфильтрованные кандидаты, где приоритетно:
- investor_npv > 0
- admin_gain_vs_baseline > 0

Твоя роль:
{role_goal}

Сценарии:
{json.dumps(llm_scenarios, ensure_ascii=False, indent=2)}

Верни только JSON такого вида:
{{
  "preferred_scenario_id": <целое число>,
  "reason": "краткое объяснение 1-3 предложения"
}}

Без markdown.
Без текста до или после JSON.
""".strip()


def _build_arbiter_prompt(
    *,
    scenarios: list[dict[str, Any]],
    selection_goal: str,
    resident_opinion: dict[str, Any],
    investor_opinion: dict[str, Any],
    admin_opinion: dict[str, Any],
) -> str:
    llm_scenarios = [_to_llm_scenario(item) for item in scenarios]
    return f"""
Ты агент-арбитр и должен выбрать один лучший сценарий развития территории.

Общая задача:
{selection_goal}

Кандидатные сценарии:
{json.dumps(llm_scenarios, ensure_ascii=False, indent=2)}

Мнения агентов:
{json.dumps(
    {
        "resident": resident_opinion,
        "investor": investor_opinion,
        "administrator": admin_opinion,
    },
    ensure_ascii=False,
    indent=2,
)}

Правила выбора:
- не выбирай сценарий с отрицательным investor_npv, если есть положительные альтернативы
- не выбирай сценарий с отрицательным приростом общей стоимости земли, если есть положительные альтернативы
- среди допустимых сценариев ищи лучший компромисс между инвестором, жителями и интересами города
- отдельно учитывай предполагаемую значимость сценария для города

Верни только JSON:
{{
  "scenario_id": <целое число>,
  "reason": "кратко почему выбран этот сценарий, 1-3 предложения"
}}

Без markdown.
Без текста до или после JSON.
""".strip()


def _invoke_llm(llm: Any, prompt: str) -> str:
    if hasattr(llm, "invoke"):
        response = llm.invoke(prompt)
        return getattr(response, "content", response)

    if hasattr(llm, "generate"):
        return llm.generate(prompt)

    raise TypeError("LLM object must provide either `invoke(prompt)` or `generate(prompt)`.")


def _parse_json_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _resolve_scenario_id(
    parsed: dict[str, Any] | None,
    scenarios: list[dict[str, Any]],
) -> int | None:
    if not parsed:
        return None

    raw_value = parsed.get("scenario_id", parsed.get("preferred_scenario_id"))
    if raw_value is None:
        return None

    try:
        scenario_id = int(raw_value)
    except (TypeError, ValueError):
        return None

    valid_ids = {item["scenario_id"] for item in scenarios}
    return scenario_id if scenario_id in valid_ids else None


def _extract_reason(parsed: dict[str, Any] | None, raw_response: Any) -> str:
    if parsed and isinstance(parsed.get("reason"), str) and parsed["reason"].strip():
        return parsed["reason"].strip()

    if isinstance(raw_response, str) and raw_response.strip():
        return raw_response.strip()

    return "LLM не вернула структурированное объяснение, поэтому выбран запасной сценарий."


def _fallback_balanced_choice(scenarios: list[dict[str, Any]]) -> int:
    land_values = np.asarray([item["objectives"]["land_value_total"] for item in scenarios], dtype=float)
    investor_npvs = np.asarray([item["objectives"]["investor_npv"] for item in scenarios], dtype=float)

    land_score = _minmax_scale(land_values)
    npv_score = _minmax_scale(investor_npvs)
    balanced_score = 0.5 * land_score + 0.5 * npv_score

    best_idx = int(np.argmax(balanced_score))
    return int(scenarios[best_idx]["scenario_id"])


def _fallback_positive_choice(scenarios: list[dict[str, Any]]) -> int:
    growth_pct = np.asarray(
        [
            item["objectives"].get("land_value_growth_total_pct", 0.0) or 0.0
            for item in scenarios
        ],
        dtype=float,
    )
    investor_npvs = np.asarray([item["objectives"]["investor_npv"] for item in scenarios], dtype=float)
    combined_score = 0.5 * _minmax_scale(growth_pct) + 0.5 * _minmax_scale(investor_npvs)
    best_idx = int(np.argmax(combined_score))
    return int(scenarios[best_idx]["scenario_id"])


def _filter_positive_scenarios(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for item in scenarios:
        objectives = item.get("objectives", {})
        investor_npv = float(objectives.get("investor_npv", 0.0) or 0.0)
        admin_gain = objectives.get("admin_gain_vs_baseline")
        if admin_gain is None:
            continue
        if investor_npv > 0 and float(admin_gain) > 0:
            filtered.append(item)
    return filtered


def _parse_agent_opinion(raw_response: Any) -> dict[str, Any]:
    parsed = _parse_json_object(raw_response)
    if parsed:
        return parsed
    text = raw_response.strip() if isinstance(raw_response, str) else ""
    return {
        "preferred_scenario_id": None,
        "reason": text or "Агент не вернул структурированное мнение.",
    }


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if np.isclose(min_value, max_value):
        return np.ones_like(values, dtype=float)
    return (values - min_value) / (max_value - min_value)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]

    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]

    if isinstance(value, np.generic):
        return value.item()

    if hasattr(value, "name") and isinstance(getattr(value, "name"), str):
        return value.name

    return value


def _to_llm_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": scenario["scenario_id"],
        "objectives": scenario["objectives"],
        "params_repaired": scenario["params_repaired"],
    }
