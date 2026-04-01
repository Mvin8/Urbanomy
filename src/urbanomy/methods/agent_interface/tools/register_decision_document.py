"""Tool for registering an active planning document for Pareto selection."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..models import StrategyDecisionDocument
from ..internal.district_optimization.document_selection import register_decision_document


class RegisterDecisionDocumentInput(BaseModel):
    """Input schema for the planning-document registration tool."""

    model_config = ConfigDict(extra="forbid")

    document_name: str | None = Field(
        default=None,
        description="Optional human-readable name of the document.",
    )
    document_text: str | None = Field(
        default=None,
        description="Inline text of the planning or strategy document.",
    )
    document_path: str | None = Field(
        default=None,
        description="Optional local path to a supported text file.",
    )
    document_type: str | None = Field(
        default=None,
        description="Optional document type: master_plan, development_strategy, or custom.",
    )

    @model_validator(mode="after")
    def _ensure_source_present(self) -> "RegisterDecisionDocumentInput":
        if str(self.document_text or "").strip() or str(self.document_path or "").strip():
            return self
        raise ValueError("Передайте document_text или document_path.")


def make_register_decision_document_tool(*, llm: Any, session_store: dict[str, Any]):
    """Create a LangChain tool for registering the active planning document."""

    @tool("register_decision_document", args_schema=RegisterDecisionDocumentInput)
    def register_decision_document_tool(
        document_name: str | None = None,
        document_text: str | None = None,
        document_path: str | None = None,
        document_type: str | None = None,
    ) -> dict[str, Any]:
        """What the tool does.

        Registers one planning document, master plan, or development strategy
        that will be used later to choose the best Pareto solution.

        When to use this tool.

        Use this tool when the user provides a planning document, strategy
        text, general plan excerpt, or a path to a supported text file and
        wants the optimizer to remember it for later decision-making.

        Returns:
            dict[str, Any]: Compact metadata about the active document,
            including its summary and extracted priorities.
        """
        document = register_decision_document(
            llm=llm,
            session_store=session_store,
            document_name=document_name,
            document_text=document_text,
            document_path=document_path,
            document_type=document_type,
        )
        return _document_payload(document)

    return register_decision_document_tool


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
