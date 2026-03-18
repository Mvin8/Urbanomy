"""Tool for calculating investment metrics for one Pareto solution."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ._district_optimization import (
    calculate_solution_investment_summary,
    latest_session_or_error,
)


class CalculateParetoSolutionInvestmentMetricsInput(BaseModel):
    """Input schema for the Pareto solution investment metrics tool."""

    model_config = ConfigDict(extra="forbid")

    solution_number: int = Field(description="1-based Pareto solution number to analyze.")

    @field_validator("solution_number", mode="before")
    @classmethod
    def _coerce_solution_number(cls, value: Any) -> int:
        return int(value)


def make_calculate_pareto_solution_investment_metrics_tool(*, session_store: dict[str, Any]):
    """Create a LangChain tool bound to the latest district optimization session."""

    @tool(
        "calculate_pareto_solution_investment_metrics",
        args_schema=CalculateParetoSolutionInvestmentMetricsInput,
    )
    def calculate_pareto_solution_investment_metrics(solution_number: int) -> dict[str, Any]:
        """What the tool does.

        Reconstructs a selected Pareto solution and calculates investment
        metrics for that scenario using the Urbanomy investment module.

        When to use this tool.

        Use this tool when the user asks for investment metrics, economic
        efficiency, NPV, IRR, PI, or investment summary for a specific Pareto
        solution.

        Args:
            solution_number: 1-based number of the Pareto solution to analyze.

        Returns:
            dict[str, Any]: Serialized investment summary table for the selected
            solution, along with the repaired scenario parameters.

        Restrictions:
            Requires that a district optimization session has already been run
            in the current agent instance.
        """
        session = latest_session_or_error(session_store)
        return calculate_solution_investment_summary(session, solution_number=solution_number)

    return calculate_pareto_solution_investment_metrics
