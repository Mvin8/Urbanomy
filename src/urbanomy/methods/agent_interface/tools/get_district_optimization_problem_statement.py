"""Tool for inspecting the current district optimization problem statement."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..models import DistrictOptimizationConfig
from .internal.district_optimization import (
    build_default_constraints,
    latest_session_or_error,
    resolve_constraints,
)
from .internal.district_optimization_formatting import (
    clean_text_block,
    format_float,
    json_mapping,
    json_value,
)


class GetDistrictOptimizationProblemStatementInput(BaseModel):
    """Input schema for the optimization-problem statement tool."""

    model_config = ConfigDict(extra="forbid")

    target_id: int | None = Field(
        default=None,
        description="Optional block id. If omitted, use the latest optimization session.",
    )

    @field_validator("target_id", mode="before")
    @classmethod
    def _coerce_target_id(cls, value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(value)


def make_get_district_optimization_problem_statement_tool(
    *,
    baseline_blocks: gpd.GeoDataFrame,
    optimization_config: DistrictOptimizationConfig,
    session_store: dict[str, Any],
):
    """Create a LangChain tool that describes the optimization setup."""

    @tool(
        "get_district_optimization_problem_statement",
        args_schema=GetDistrictOptimizationProblemStatementInput,
    )
    def get_district_optimization_problem_statement(target_id: int | None = None) -> dict[str, Any]:
        """What the tool does.

        Returns the current optimization problem statement: target block,
        decision variables with bounds, optimization objectives, repair rules,
        and tunable algorithm settings.

        When to use this tool.

        Use this tool when the user asks to show the optimization setup,
        explain the variables or constraints, or asks what should be changed
        in the problem statement before running optimization.

        Args:
            target_id: Optional block identifier. If omitted, the tool uses the
                latest optimization session if available.

        Returns:
            dict[str, Any]: JSON-safe problem-statement payload plus a compact
            human-readable text block for the agent.

        Restrictions:
            If there is no active optimization session, ``target_id`` must be
            provided so the tool can resolve site area and constraints.
        """
        source = "config_template"
        session = None
        resolved_target_id = target_id
        if resolved_target_id is None:
            try:
                session = latest_session_or_error(session_store)
            except Exception:
                session = None
                source = "config_defaults"
            else:
                source = "active_session"
                resolved_target_id = int(session.target_id)

        site_area: float | None = None
        if resolved_target_id is not None:
            site_area = _resolve_site_area(
                baseline_blocks=baseline_blocks,
                target_id_column=optimization_config.target_id_column,
                target_id=int(resolved_target_id),
            )

        if session is not None and resolved_target_id is not None and int(session.target_id) == int(resolved_target_id):
            constraints = dict(session.problem.constraints)
            variable_names = list(getattr(session.problem, "var_names", constraints.keys()))
        elif site_area is not None:
            constraints = (
                resolve_constraints(optimization_config.constraints_template, site_area=site_area)
                if optimization_config.constraints_template
                else build_default_constraints(site_area)
            )
            variable_names = list(constraints.keys())
        else:
            constraints = {}
            variable_names = []

        decision_variables = [
            {
                "name": str(name),
                "type": str(spec.get("type", "float")),
                "min": json_value(spec.get("min")),
                "max": json_value(spec.get("max")),
            }
            for name, spec in constraints.items()
        ]
        objectives = [
            {
                "name": "total_land_value",
                "goal": "maximize",
                "description": "Суммарная стоимость земли после сценарного изменения квартала.",
            },
            {
                "name": "investor_npv",
                "goal": "maximize",
                "description": "NPV проекта для изменяемого квартала по инвестиционной модели.",
            },
        ]
        repair_rules = _repair_rules(site_area=site_area)
        runtime_overrides = dict(session_store.get("algorithm_overrides") or {})
        tunable_parameters = {
            "algorithm": {
                "pop_size_default": int(runtime_overrides.get("pop_size", optimization_config.pop_size)),
                "n_gen_default": int(runtime_overrides.get("n_gen", optimization_config.n_gen)),
                "seed_default": int(runtime_overrides.get("seed", optimization_config.seed)),
                "eliminate_duplicates": bool(optimization_config.eliminate_duplicates),
                "save_history": bool(optimization_config.save_history),
                "use_history": bool(optimization_config.use_history),
                "verbose": bool(optimization_config.verbose),
            },
            "runtime_overrides_supported": ["pop_size", "n_gen", "seed"],
            "constraints_editable_via_config": True,
        }
        change_examples = [
            "Если нужно шире исследовать поиск, увеличьте pop_size и n_gen.",
            "Если нужны более плотные сценарии, увеличьте верхнюю границу для footprint_area или l.",
            "Если нужен другой баланс жилья и коммерции, измените диапазоны mxi и долей land_use.",
            "Если ограничения слишком жёсткие, пересмотрите constraints_template до запуска оптимизации.",
        ]
        problem_statement_text = _build_problem_statement_text(
            target_id=int(resolved_target_id) if resolved_target_id is not None else None,
            site_area=site_area,
            source=source,
            decision_variables=decision_variables,
            objectives=objectives,
            repair_rules=repair_rules,
            tunable_parameters=tunable_parameters,
            change_examples=change_examples,
        )
        return {
            "target_id": int(resolved_target_id) if resolved_target_id is not None else None,
            "site_area": site_area,
            "source": source,
            "decision_variables": decision_variables,
            "variable_names": variable_names,
            "constraints": json_mapping(constraints),
            "objectives": objectives,
            "repair_rules": repair_rules,
            "tunable_parameters": json_mapping(tunable_parameters),
            "runtime_overrides_active": json_mapping(runtime_overrides),
            "change_examples": change_examples,
            "problem_statement_text": problem_statement_text,
        }

    return get_district_optimization_problem_statement


def _resolve_site_area(
    *,
    baseline_blocks: gpd.GeoDataFrame,
    target_id_column: str,
    target_id: int,
) -> float:
    site_area_series = pd.to_numeric(
        baseline_blocks.loc[baseline_blocks[target_id_column] == target_id, "site_area"],
        errors="coerce",
    )
    if site_area_series.empty or pd.isna(site_area_series.iloc[0]):
        raise ValueError(f"Не удалось определить site_area для target_id={target_id}.")
    return float(site_area_series.iloc[0])


def _repair_rules(*, site_area: float | None) -> list[str]:
    rules = [
        "Доли land_use нормализуются так, чтобы их сумма была равна 1.0.",
        "Если все доли land_use нулевые, сценарий принудительно становится residential=1.0.",
        "l ограничивается снизу значением 1.0.",
        "mxi ограничивается диапазоном [0.0, 1.0].",
        "Доминирующая доля land_use определяет итоговый land_use квартала.",
        "Производные показатели build_floor_area, living_area, non_living_area, population, fsi и gsi пересчитываются автоматически.",
    ]
    if site_area is not None:
        footprint_cap = 0.8 * float(site_area)
        rules.insert(
            2,
            f"footprint_area ограничивается сверху значением 0.8 * site_area = {format_float(footprint_cap)}.",
        )
    else:
        rules.insert(
            2,
            "footprint_area ограничивается сверху значением 0.8 * site_area после выбора конкретного квартала.",
        )
    return rules


def _build_problem_statement_text(
    *,
    target_id: int | None,
    site_area: float | None,
    source: str,
    decision_variables: list[dict[str, Any]],
    objectives: list[dict[str, Any]],
    repair_rules: list[str],
    tunable_parameters: dict[str, Any],
    change_examples: list[str],
) -> str:
    lines = [
        "Постановка задачи оптимизации:",
        f" • target_id: {target_id if target_id is not None else 'не задан'}",
        f" • source: {source}",
        f" • site_area: {format_float(site_area) if site_area is not None else 'будет определена после выбора квартала'}",
        "",
        "Целевые функции:",
    ]
    for item in objectives:
        lines.append(
            f" • {item['name']} ({item['goal']}): {item['description']}"
        )
    lines.append("")
    lines.append("Переменные решения и диапазоны:")
    if decision_variables:
        for item in decision_variables:
            min_value = item.get("min")
            max_value = item.get("max")
            lines.append(
                f" • {item['name']} [{item['type']}]: min={min_value}, max={max_value}"
            )
    else:
        lines.append(" • Для точных диапазонов укажи target_id или сначала запусти оптимизацию квартала.")
    lines.append("")
    lines.append("Правила repair / нормализации:")
    for rule in repair_rules:
        lines.append(f" • {rule}")
    lines.append("")
    lines.append("Параметры алгоритма по умолчанию:")
    algorithm = tunable_parameters["algorithm"]
    lines.append(
        f" • pop_size={algorithm['pop_size_default']}, n_gen={algorithm['n_gen_default']}, seed={algorithm['seed_default']}"
    )
    lines.append("")
    lines.append("Что можно менять:")
    for item in change_examples:
        lines.append(f" • {item}")
    return clean_text_block("\n".join(lines))
