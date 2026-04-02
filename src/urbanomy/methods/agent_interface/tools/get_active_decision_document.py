"""Tool for inspecting the active planning document used for selection."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from ..models import StrategyDecisionDocument
from ..internal.district_optimization.document_selection import get_active_decision_document


def make_get_active_decision_document_tool(*, session_store: dict[str, Any]):
    """Create a LangChain tool for inspecting the active planning document."""

    @tool("get_active_decision_document")
    def get_active_decision_document_tool() -> dict[str, Any]:
        """What the tool does.

        Returns the currently active planning or strategy document that is
        stored in the optimization session.

        When to use this tool.

        Use this tool when the user asks which document is active, wants to
        inspect its summary, or wants to verify what policy context will be
        used for Pareto-solution selection.
        """
        document = get_active_decision_document(session_store)
        return _document_payload(document)

    return get_active_decision_document_tool


def _document_payload(document: StrategyDecisionDocument) -> dict[str, Any]:
    return {
        "document_name": document.document_name,
        "document_type": document.document_type,
        "source": document.source,
        "source_ref": document.source_ref,
        "summary": document.summary,
        "priorities": list(document.extracted_priorities),
        "retrieval_backend": document.retrieval_backend,
        "chunk_count": document.chunk_count,
        "preview_text": document.preview_text(),
    }
