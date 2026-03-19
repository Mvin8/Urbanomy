"""Tool for returning repaired parameters of one Pareto solution."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .internal.district_optimization import get_solution_parameters, latest_session_or_error


class GetParetoSolutionParametersInput(BaseModel):
    """Input schema for the Pareto solution parameters tool."""

    model_config = ConfigDict(extra="forbid")

    solution_number: int = Field(description="1-based Pareto solution number to inspect.")

    @field_validator("solution_number", mode="before")
    @classmethod
    def _coerce_solution_number(cls, value: Any) -> int:
        return int(value)


def make_get_pareto_solution_parameters_tool(*, session_store: dict[str, Any]):
    """Create a LangChain tool bound to the latest district optimization session."""

    @tool("get_pareto_solution_parameters", args_schema=GetParetoSolutionParametersInput)
    def get_pareto_solution_parameters(solution_number: int) -> dict[str, Any]:
        """What the tool does.

        Returns the repaired optimization parameters for a selected Pareto
        solution.

        When to use this tool.

        Use this tool when the user asks to show, print, or inspect the
        parameters of a specific Pareto solution or explicitly asks for
        ``params_repaired``.

        Args:
            solution_number: 1-based number of the Pareto solution to inspect.

        Returns:
            dict[str, Any]: Repaired parameters for the selected solution and a
            preformatted text block for user display.

        Restrictions:
            Requires that a district optimization session has already been run
            in the current agent instance.
        """
        session = latest_session_or_error(session_store)
        return get_solution_parameters(session, solution_number=solution_number)

    return get_pareto_solution_parameters
