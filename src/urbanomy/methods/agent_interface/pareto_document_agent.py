"""Agent for policy-document-aware Pareto solution selection."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .internal.common.agent_utils import extract_message_text
from .internal.common.domain_contracts import ToolDescriptor, describe_tool
from .tools.get_active_decision_document import make_get_active_decision_document_tool
from .tools.register_decision_document import make_register_decision_document_tool
from .tools.retrieve_decision_document_context import (
    make_retrieve_decision_document_context_tool,
)
from .tools.select_pareto_solution_by_document import make_select_pareto_solution_by_document_tool

PARETO_DOCUMENT_CAPABILITY_LINES = [
    "- регистрация генерального плана или стратегии как RAG policy-контекста",
    "- retrieval релевантных фрагментов из активного документа",
    "- выбор лучшего Pareto-решения по активному документу",
    "- объяснение, почему выбранное решение лучше согласуется с документом",
]

_DOCUMENT_MARKERS = (
    "документ",
    "генплан",
    "генеральный план",
    "стратег",
    "master plan",
)
_SELECTION_MARKERS = (
    "лучшее решение",
    "лучший сценарий",
    "рекомендуй",
    "на основе документа",
    "по документу",
    "по генплану",
    "по стратегии",
)
_RETRIEVAL_MARKERS = (
    "фрагмент",
    "фрагменты",
    "контекст документа",
    "rag",
    "retrieval",
    "найди в документе",
    "что в документе сказано",
    "покажи выдержки",
    "достань из документа",
    "согласно документ",
    "согласно документу",
    "в документе",
    "из документа",
    "по данным документа",
)


class ParetoDocumentAgent:
    """Small deterministic agent over policy-document selection tools."""

    def __init__(
        self,
        *,
        llm: Any,
        session_store: dict[str, Any],
    ) -> None:
        self.llm = llm
        self.session_store = session_store
        self._register_tool = make_register_decision_document_tool(
            llm=llm,
            session_store=session_store,
        )
        self._active_document_tool = make_get_active_decision_document_tool(
            session_store=session_store
        )
        self._retrieval_tool = make_retrieve_decision_document_context_tool(
            session_store=session_store
        )
        self._selection_tool = make_select_pareto_solution_by_document_tool(
            llm=llm,
            session_store=session_store,
        )
        self._tools = [
            self._register_tool,
            self._active_document_tool,
            self._retrieval_tool,
            self._selection_tool,
        ]
        self._tools_by_name = {tool.name: tool for tool in self._tools}

    def invoke(self, user_request: str) -> dict[str, Any]:
        """Handle a document-oriented Pareto selection request."""
        text = str(user_request).strip()
        if not text:
            raise ValueError("user_request cannot be empty")
        if self._looks_like_active_document_request(text):
            return self._invoke_named_tool("get_active_decision_document", {})
        document_payload = self._extract_document_payload(text)
        if document_payload is not None:
            return self._invoke_named_tool("register_decision_document", document_payload)
        if self._looks_like_retrieval_request(text):
            return self._invoke_named_tool(
                "retrieve_decision_document_context",
                {"query": text, "top_k": 4},
            )
        if self._looks_like_selection_request(text):
            return self._invoke_named_tool(
                "select_pareto_solution_by_document",
                {"selection_goal": text},
            )
        return {
            "status": "error",
            "response": (
                "Не удалось определить document-selection запрос. "
                "Передайте текст документа или попросите выбрать лучшее решение по активному документу."
            ),
            "tool_name": None,
            "tool_output": None,
            "used_tool_fallback": True,
        }

    def available_tools(self) -> tuple[Any, ...]:
        """Return tools exposed by the document-selection layer."""
        return tuple(self._tools)

    def tool_descriptors(self) -> list[ToolDescriptor]:
        """Return compact user-facing descriptions of document-selection tools."""
        return [describe_tool(tool) for tool in self._tools]

    @staticmethod
    def capability_lines() -> list[str]:
        """Return user-facing capabilities exposed by the document-selection layer."""
        return list(PARETO_DOCUMENT_CAPABILITY_LINES)

    def _invoke_named_tool(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools_by_name[tool_name]
        try:
            tool_output = tool.invoke(payload)
        except Exception as exc:
            message = str(exc).strip() or f"{exc.__class__.__name__}: document tool завершился с ошибкой."
            return {
                "status": "error",
                "response": message,
                "tool_name": tool_name,
                "tool_output": None,
                "used_tool_fallback": True,
            }
        response = self._build_response_from_tool(tool_name=tool_name, tool_output=tool_output)
        return {
            "status": "ok",
            "response": response,
            "tool_name": tool_name,
            "tool_output": tool_output,
            "used_tool_fallback": True,
        }

    def _build_response_from_tool(self, *, tool_name: str, tool_output: dict[str, Any]) -> str:
        if tool_name == "register_decision_document":
            priorities = list(tool_output.get("priorities") or [])
            priority_text = "\n".join(f"- {item}" for item in priorities[:4])
            base = (
                f"Документ '{tool_output.get('document_name')}' зарегистрирован "
                f"как active RAG context."
            )
            summary = str(tool_output.get("summary", "")).strip()
            chunk_count = tool_output.get("chunk_count")
            retrieval_backend = str(tool_output.get("retrieval_backend") or "").strip()
            if chunk_count:
                base = f"{base}\nИндексировано фрагментов: {chunk_count}."
            if retrieval_backend:
                base = f"{base}\nRetrieval backend: {retrieval_backend}."
            if summary:
                base = f"{base}\n\nКраткая сводка: {summary}"
            if priority_text:
                base = f"{base}\n\nИзвлечённые приоритеты:\n{priority_text}"
            return base
        if tool_name == "get_active_decision_document":
            priorities = list(tool_output.get("priorities") or [])
            lines = [
                f"Активный документ: {tool_output.get('document_name')}",
                f"Тип: {tool_output.get('document_type')}",
            ]
            chunk_count = tool_output.get("chunk_count")
            retrieval_backend = str(tool_output.get("retrieval_backend") or "").strip()
            if chunk_count:
                lines.append(f"RAG-фрагментов: {chunk_count}")
            if retrieval_backend:
                lines.append(f"Retrieval backend: {retrieval_backend}")
            summary = str(tool_output.get("summary", "")).strip()
            if summary:
                lines.append(f"Сводка: {summary}")
            if priorities:
                lines.append("Приоритеты:")
                lines.extend(f"- {item}" for item in priorities[:5])
            return "\n".join(lines).strip()
        if tool_name == "retrieve_decision_document_context":
            chunks = list(tool_output.get("chunks") or [])
            if not chunks:
                return "Для запроса не нашлось релевантных RAG-фрагментов в активном документе."
            query = str(tool_output.get("query") or "").strip()
            if self._prefers_evidence_only_response(query):
                return self._render_retrieved_chunks(chunks)
            synthesized = self._synthesize_grounded_answer(query=query, chunks=chunks)
            if synthesized:
                return synthesized
            return self._render_retrieved_chunks(chunks)
        if tool_name == "select_pareto_solution_by_document":
            return str(tool_output.get("selection_text") or "Решение выбрано по документу.").strip()
        return "Запрос обработан."

    @staticmethod
    def _render_retrieved_chunks(chunks: list[dict[str, Any]]) -> str:
        """Render retrieved chunks as a compact evidence list."""
        lines = ["Нашёл релевантные фрагменты из активного документа:"]
        for chunk in chunks[:4]:
            chunk_id = str(chunk.get("chunk_id") or "").strip() or "chunk"
            score = float(chunk.get("score") or 0.0)
            text = " ".join(str(chunk.get("text") or "").split()).strip()
            suffix = "…" if len(text) > 180 else ""
            lines.append(f"- {chunk_id} (score={score:.2f}): {text[:180]}{suffix}")
        return "\n".join(lines).strip()

    @staticmethod
    def _prefers_evidence_only_response(query: str) -> bool:
        normalized = str(query).lower().replace("ё", "е")
        return any(
            marker in normalized
            for marker in (
                "покажи фрагмент",
                "покажи фрагменты",
                "покажи выдержки",
                "контекст документа",
                "rag",
                "retrieval",
            )
        )

    def _synthesize_grounded_answer(self, *, query: str, chunks: list[dict[str, Any]]) -> str:
        """Try to answer the user question strictly from retrieved chunks."""
        if not hasattr(self.llm, "invoke"):
            return ""
        evidence_lines: list[str] = []
        for chunk in chunks[:4]:
            chunk_id = str(chunk.get("chunk_id") or "").strip() or "chunk"
            text = " ".join(str(chunk.get("text") or "").split()).strip()
            if text:
                evidence_lines.append(f"[{chunk_id}] {text}")
        if not evidence_lines:
            return ""
        prompt = (
            "Ты document-RAG subagent Urbanomy.\n"
            "Отвечай только на основе переданных фрагментов документа.\n"
            "Если точного ответа в них нет, прямо скажи об этом.\n"
            "Если ответ найден, дай короткий ответ по-русски и в конце укажи ссылки на chunk_id в квадратных скобках.\n"
            "Не используй внешние знания."
        )
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(
                        content=(
                            f"Вопрос: {query.strip()}\n\n"
                            "Фрагменты документа:\n"
                            f"{chr(10).join(evidence_lines)}"
                        )
                    ),
                ]
            )
        except Exception:
            return ""
        text = extract_message_text(response)
        return text.strip()

    @staticmethod
    def _looks_like_selection_request(text: str) -> bool:
        normalized = text.lower().replace("ё", "е")
        return any(marker in normalized for marker in _SELECTION_MARKERS)

    @staticmethod
    def _looks_like_active_document_request(text: str) -> bool:
        normalized = text.lower().replace("ё", "е")
        return any(
            marker in normalized
            for marker in (
                "какой документ",
                "активный документ",
                "что за документ",
                "какой сейчас документ",
            )
        )

    @staticmethod
    def _looks_like_retrieval_request(text: str) -> bool:
        normalized = text.lower().replace("ё", "е")
        if any(marker in normalized for marker in _RETRIEVAL_MARKERS):
            return True
        question_markers = (
            "какое",
            "какой",
            "какая",
            "какие",
            "сколько",
            "когда",
            "где",
            "почему",
            "зачем",
            "кто",
            "что",
        )
        has_document_reference = any(
            marker in normalized
            for marker in ("документ", "документу", "документе", "из документа", "согласно")
        )
        has_question_shape = any(marker in normalized for marker in question_markers)
        return has_document_reference and has_question_shape

    @staticmethod
    def _extract_document_payload(text: str) -> dict[str, Any] | None:
        normalized = text.lower().replace("ё", "е")
        if not any(marker in normalized for marker in _DOCUMENT_MARKERS):
            return None
        path_match = re.search(r"(?:document_path|path)\s*[:=]\s*([^\n]+)", text, flags=re.IGNORECASE)
        name_match = re.search(r"(?:document_name|name)\s*[:=]\s*([^\n]+)", text, flags=re.IGNORECASE)
        type_match = re.search(r"(?:document_type|type)\s*[:=]\s*([^\n]+)", text, flags=re.IGNORECASE)
        explicit_text_match = re.search(
            r"(?:document_text|текст документа)\s*[:=]\s*(.+)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        document_text: str | None = None
        if explicit_text_match is not None:
            document_text = explicit_text_match.group(1).strip()
        elif ":" in text and any(
            marker in normalized
            for marker in ("добавь документ", "сохрани документ", "зарегистрируй документ", "генплан", "стратег")
        ):
            document_text = text.split(":", 1)[1].strip()
        elif "\n" in text and any(marker in normalized for marker in ("добавь документ", "сохрани документ", "зарегистрируй документ")):
            document_text = text.split("\n", 1)[1].strip()
        if not str(document_text or "").strip() and path_match is None:
            return None
        return {
            "document_name": name_match.group(1).strip() if name_match is not None else None,
            "document_text": document_text,
            "document_path": path_match.group(1).strip() if path_match is not None else None,
            "document_type": type_match.group(1).strip() if type_match is not None else None,
        }


def create_pareto_document_agent(*, llm: Any, session_store: dict[str, Any]) -> ParetoDocumentAgent:
    """Factory for the Pareto document-selection agent."""
    return ParetoDocumentAgent(llm=llm, session_store=session_store)
