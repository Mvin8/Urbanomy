"""Tool for selecting the best Pareto solution against an active document."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from ..models import ParetoDocumentSelectionResult
from ..internal.district_optimization.document_selection import select_solution_by_document


class SelectParetoSolutionByDocumentInput(BaseModel):
    """Input schema for document-based Pareto solution selection."""

    model_config = ConfigDict(extra="forbid")

    selection_goal: str = Field(
        default="Выбрать лучшее решение по активному документу.",
        description="How the selection should be interpreted from the user's perspective.",
    )


def make_select_pareto_solution_by_document_tool(*, llm: Any, session_store: dict[str, Any]):
    """Create a LangChain tool for document-based solution selection."""

    @tool(
        "select_pareto_solution_by_document",
        args_schema=SelectParetoSolutionByDocumentInput,
    )
    def select_pareto_solution_by_document_tool(
        selection_goal: str = "Выбрать лучшее решение по активному документу.",
    ) -> dict[str, Any]:
        """What the tool does.

        Selects the best Pareto solution from the latest optimization session
        using the active planning or strategy document as the policy context.

        When to use this tool.

        Use this tool when the user asks to choose, recommend, rank, or justify
        the best Pareto solution according to a general plan, development
        strategy, or any other previously registered document.
        """
        result = select_solution_by_document(
            llm=llm,
            session_store=session_store,
            selection_goal=selection_goal,
        )
        return _selection_payload(result)

    return select_pareto_solution_by_document_tool


def _selection_payload(result: ParetoDocumentSelectionResult) -> dict[str, Any]:
    return {
        "document_name": result.document_name,
        "document_type": result.document_type,
        "selection_goal": result.selection_goal,
        "document_summary": result.document_summary,
        "selected_solution_number": result.selected_solution_number,
        "solution_number": result.selected_solution_number,
        "selected_scenario_id": result.selected_scenario_id,
        "selected_title": result.selected_title,
        "rationale": result.rationale,
        "criteria_used": list(result.criteria_used),
        "retrieved_chunks": [item.model_dump() for item in result.retrieved_chunks],
        "candidate_reviews": [item.model_dump() for item in result.candidate_reviews],
        "risks": list(result.risks),
        "selection_text": result.to_text(),
    }
