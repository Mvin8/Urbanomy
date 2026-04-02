"""Dedicated agent for document registration, retrieval, and document-grounded choices."""

from __future__ import annotations

from typing import Any

from .pareto_document_agent import ParetoDocumentAgent

DOCUMENT_RAG_CAPABILITY_LINES = [
    "- загрузка PDF/DOCX/TXT документа как отдельного RAG-контекста",
    "- retrieval сведений и фрагментов из активного документа",
    "- проверка, какой документ сейчас активен",
    "- document-grounded выбор лучшего Pareto-решения по активному документу",
]


class DocumentRagAgent(ParetoDocumentAgent):
    """Thin dedicated wrapper around the document RAG tool layer."""

    @staticmethod
    def capability_lines() -> list[str]:
        """Return user-facing capabilities of the document route."""
        return list(DOCUMENT_RAG_CAPABILITY_LINES)


def create_document_rag_agent(*, llm: Any, session_store: dict[str, Any]) -> DocumentRagAgent:
    """Factory for the dedicated document RAG agent."""
    return DocumentRagAgent(llm=llm, session_store=session_store)
