"""Tool for inspecting district-optimization constraints."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..models import DistrictOptimizationConfig
from .internal.district_optimization import (
    build_default_constraints,
    latest_session_or_error,
    resolve_constraints,
)
from .internal.district_optimization_formatting import clean_text_block, format_float, json_mapping
from .get_district_optimization_problem_statement import _resolve_site_area


class GetDistrictOptimizationConstraintsInput(BaseModel):
    """Input schema for the optimization-constraints tool."""

    model_config = ConfigDict(extra="forbid")

    target_id: int | None = Field(
        default=None,
        description="Optional block id. If omitted, use active session or symbolic defaults.",
    )

    @field_validator("target_id", mode="before")
    @classmethod
    def _coerce_target_id(cls, value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(value)


def make_get_district_optimization_constraints_tool(
    *,
    baseline_blocks: gpd.GeoDataFrame,
    optimization_config: DistrictOptimizationConfig,
    session_store: dict[str, Any],
):
    """Create a LangChain tool that returns optimization constraints."""

    @tool(
        "get_district_optimization_constraints",
        args_schema=GetDistrictOptimizationConstraintsInput,
    )
    def get_district_optimization_constraints(target_id: int | None = None) -> dict[str, Any]:
        """What the tool does.

        Returns the optimization constraints and variable bounds currently used
        by the district-optimization setup.

        When to use this tool.

        Use this tool when the user explicitly asks to show, list, or explain
        optimization constraints or decision-variable bounds.

        Args:
            target_id: Optional block identifier. If omitted, the tool uses the
                latest session if available, otherwise returns symbolic defaults.

        Returns:
            dict[str, Any]: JSON-safe constraints payload plus a readable text block.
        """
        source = "config_template"
        session = None
        resolved_target_id = target_id
        symbolic = False

        if resolved_target_id is None:
            try:
                session = latest_session_or_error(session_store)
            except Exception:
                session = None
                source = "config_defaults_symbolic"
                symbolic = True
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
        elif site_area is not None:
            constraints = (
                resolve_constraints(optimization_config.constraints_template, site_area=site_area)
                if optimization_config.constraints_template
                else build_default_constraints(site_area)
            )
        elif optimization_config.constraints_template:
            constraints = dict(optimization_config.constraints_template)
        else:
            constraints = _build_symbolic_default_constraints()

        constraints_text = _build_constraints_text(
            target_id=int(resolved_target_id) if resolved_target_id is not None else None,
            site_area=site_area,
            source=source,
            constraints=constraints,
            symbolic=symbolic,
        )
        return {
            "target_id": int(resolved_target_id) if resolved_target_id is not None else None,
            "site_area": site_area,
            "source": source,
            "symbolic": symbolic,
            "constraints": json_mapping(constraints),
            "constraint_names": list(constraints.keys()),
            "constraints_text": constraints_text,
        }

    return get_district_optimization_constraints


def _build_symbolic_default_constraints() -> dict[str, dict[str, Any]]:
    return {
        "footprint_area": {"type": "float", "min": 0.0, "max": "0.1 * site_area"},
        "l": {"type": "float", "min": 1.0, "max": 10.0},
        "mxi": {"type": "float", "min": 0.0, "max": 1.0},
        "residential": {"type": "float", "min": 0.0, "max": 1.0},
        "business": {"type": "float", "min": 0.0, "max": 1.0},
        "recreation": {"type": "float", "min": 0.0, "max": 1.0},
        "industrial": {"type": "float", "min": 0.0, "max": 1.0},
        "transport": {"type": "float", "min": 0.0, "max": 1.0},
        "special": {"type": "float", "min": 0.0, "max": 1.0},
        "agriculture": {"type": "float", "min": 0.0, "max": 1.0},
    }


def _build_constraints_text(
    *,
    target_id: int | None,
    site_area: float | None,
    source: str,
    constraints: dict[str, dict[str, Any]],
    symbolic: bool,
) -> str:
    lines = [
        "Ограничения оптимизации:",
        f" • target_id: {target_id if target_id is not None else 'не задан'}",
        f" • source: {source}",
    ]
    if site_area is None:
        lines.append(" • site_area: не задана, поэтому формулы могут быть символическими")
    else:
        lines.append(f" • site_area: {format_float(site_area)}")
    lines.append("")
    lines.append("Диапазоны переменных:")
    for name, spec in constraints.items():
        lines.append(
            f" • {name} [{spec.get('type', 'float')}]: min={spec.get('min')}, max={spec.get('max')}"
        )
    if symbolic:
        lines.append("")
        lines.append("Примечание: без target_id показаны шаблонные ограничения до подстановки site_area.")
    return clean_text_block("\n".join(lines))
