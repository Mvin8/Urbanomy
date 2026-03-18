"""Tool for visualizing land-value impact of one Pareto solution."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ._district_optimization import latest_session_or_error, plot_solution_impact


class PlotParetoSolutionImpactInput(BaseModel):
    """Input schema for the Pareto solution impact plot tool."""

    model_config = ConfigDict(extra="forbid")

    solution_number: int = Field(description="1-based Pareto solution number to visualize.")

    @field_validator("solution_number", mode="before")
    @classmethod
    def _coerce_solution_number(cls, value: Any) -> int:
        return int(value)


def make_plot_pareto_solution_impact_tool(*, session_store: dict[str, Any]):
    """Create a LangChain tool bound to the latest district optimization session."""

    @tool("plot_pareto_solution_impact", args_schema=PlotParetoSolutionImpactInput)
    def plot_pareto_solution_impact(solution_number: int) -> dict[str, Any]:
        """What the tool does.

        Reconstructs a selected Pareto solution, computes land value before and
        after the scenario, and visualizes the land-value impact map for that
        solution.

        When to use this tool.

        Use this tool when the user asks to visualize, plot, or inspect a
        specific Pareto solution, for example "visualize solution 2".

        Args:
            solution_number: 1-based number of the Pareto solution to visualize.

        Returns:
            dict[str, Any]: Compact payload with total and target-block land
            value changes for the selected solution. The figure is created as a
            side effect.

        Restrictions:
            Requires that a district optimization session has already been run
            in the current agent instance.
        """
        session = latest_session_or_error(session_store)
        return plot_solution_impact(session, solution_number=solution_number)

    return plot_pareto_solution_impact
