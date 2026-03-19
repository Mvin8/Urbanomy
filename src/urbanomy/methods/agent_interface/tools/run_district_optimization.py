"""Tool for running district optimization for a target block."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..models import DistrictOptimizationConfig
from .internal.district_optimization import (
    latest_session_summary,
    plot_optimization_pareto_front,
    run_optimization_session,
)


class RunDistrictOptimizationInput(BaseModel):
    """Input schema for the district optimization tool."""

    model_config = ConfigDict(extra="forbid")

    target_id: int = Field(description="Block id to optimize.")
    pop_size: int | None = Field(default=None, ge=2, description="Optional NSGA2 population size override.")
    n_gen: int | None = Field(default=None, ge=1, description="Optional number of generations override.")
    seed: int | None = Field(default=None, description="Optional random seed override.")

    @field_validator("target_id", mode="before")
    @classmethod
    def _coerce_target_id(cls, value: Any) -> int:
        return int(value)


def make_run_district_optimization_tool(
    *,
    baseline_blocks: gpd.GeoDataFrame,
    optimization_config: DistrictOptimizationConfig,
    session_store: dict[str, Any],
):
    """Create a LangChain tool bound to the district optimization context."""

    @tool("run_district_optimization", args_schema=RunDistrictOptimizationInput)
    def run_district_optimization(
        target_id: int,
        pop_size: int | None = None,
        n_gen: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """What the tool does.

        Runs NSGA2-based district optimization for the block identified by
        ``target_id`` and builds a Pareto-front table for the resulting
        scenarios.

        When to use this tool.

        Use this tool when the user asks to optimize a block, select a
        development scenario for a block, or generate Pareto solutions for a
        specific target квартал.

        Args:
            target_id: Integer block identifier from ``baseline_blocks['id']``.
            pop_size: Optional override for NSGA2 population size.
            n_gen: Optional override for the number of generations.
            seed: Optional override for the random seed.

        Returns:
            dict[str, Any]: Compact summary of the created optimization session,
            including target id, site area, number of Pareto solutions, and
            per-solution metadata.

        Restrictions:
            Requires a valid optimization context with model and feature lists.
            Stores the created session as the latest optimization session and
            replaces the previous one.
        """
        session = run_optimization_session(
            blocks=baseline_blocks,
            config=optimization_config,
            target_id=target_id,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
        )
        session_store["latest_district_optimization_session"] = session
        summary = latest_session_summary(session)
        pareto_front_plot = plot_optimization_pareto_front(session)
        summary["pareto_front_summary_text"] = pareto_front_plot.get("summary_text")
        summary["pareto_front_plotted"] = bool(pareto_front_plot.get("figure_created"))
        return summary

    return run_district_optimization
