"""Tool for retrieving relevant RAG context from the active decision document."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from ..internal.district_optimization.document_selection import retrieve_document_context


class RetrieveDecisionDocumentContextInput(BaseModel):
    """Input schema for active-document RAG retrieval."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Question or retrieval query used to search the active document.")
    top_k: int = Field(default=4, ge=1, le=8, description="How many relevant chunks to retrieve.")


def make_retrieve_decision_document_context_tool(*, session_store: dict[str, Any]):
    """Create a LangChain tool for querying the active decision-document store."""

    @tool(
        "retrieve_decision_document_context",
        args_schema=RetrieveDecisionDocumentContextInput,
    )
    def retrieve_decision_document_context_tool(
        query: str,
        top_k: int = 4,
    ) -> dict[str, Any]:
        """What the tool does.

        Retrieves the most relevant fragments from the active planning document
        using the in-memory RAG index built during document registration.

        When to use this tool.

        Use this tool when the user asks which document fragments justify a
        recommendation, wants evidence from the active strategy, or needs
        document-grounded context before selecting a Pareto solution.
        """
        chunks = retrieve_document_context(
            session_store=session_store,
            query=query,
            top_k=top_k,
        )
        return {
            "query": str(query).strip(),
            "top_k": int(top_k),
            "chunks": [item.model_dump() for item in chunks],
        }

    return retrieve_decision_document_context_tool
