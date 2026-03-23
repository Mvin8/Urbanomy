"""General conversational subagent for the Urbanomy orchestrator."""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from .internal.agent_utils import build_tool_agent, extract_last_ai_message_text
from .prompts import GENERAL_QA_AGENT_SYSTEM_PROMPT, GENERAL_QA_DIRECT_SYSTEM_PROMPT


class GeneralQaAgent:
    """Conversational subagent that can explain orchestrator capabilities."""

    def __init__(
        self,
        *,
        llm: Any,
        context_provider: Callable[[], dict[str, str]],
    ) -> None:
        if llm is None:
            raise ValueError("llm is required to build GeneralQaAgent.")
        self.llm = llm
        self._context_provider = context_provider
        self._active_context: dict[str, str] = {}
        self._context_tool = self._make_context_tool()
        self._tool_catalog_tool = self._make_tool_catalog_tool()
        self.agent = build_tool_agent(
            llm=self.llm,
            tools=[self._context_tool, self._tool_catalog_tool],
            system_prompt=GENERAL_QA_AGENT_SYSTEM_PROMPT,
        )

    def invoke(self, user_request: str, *, context: dict[str, str] | None = None) -> str:
        """Answer a free-form user question in Russian."""
        text = str(user_request).strip()
        if not text:
            raise ValueError("user_request cannot be empty")
        self._active_context = dict(context or self._context_provider())
        if self._should_use_tool_agent(text):
            response = self._invoke_with_context_tool(text)
            if response:
                return response
        response = self._invoke_direct(text)
        if response:
            return response
        if self._should_use_tool_agent(text):
            response = self._invoke_with_context_tool(text)
            if response:
                return response
        return self._fallback_answer(user_request=text)

    def _invoke_with_context_tool(self, user_request: str) -> str:
        """Try the tool-calling agent path for context-aware Q&A."""
        try:
            raw_result = self.agent.invoke({"messages": [HumanMessage(content=user_request)]})
        except Exception:
            return ""
        response = extract_last_ai_message_text(raw_result)
        return "" if self._looks_like_tool_artifact(response) else response

    def _invoke_direct(self, user_request: str) -> str:
        """Invoke the base chat model directly for ordinary questions."""
        prompt = GENERAL_QA_DIRECT_SYSTEM_PROMPT
        context_block = self._build_inline_context(user_request)
        if context_block:
            prompt = f"{prompt}\n\n{context_block}"
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content=user_request),
                ]
            )
        except Exception:
            return ""
        text = (
            getattr(response, "content", "")
            if isinstance(getattr(response, "content", ""), str)
            else str(getattr(response, "content", "")).strip()
        )
        return "" if self._looks_like_tool_artifact(text) else text

    def run(self, user_request: str, *, context: dict[str, str] | None = None) -> str:
        return self.invoke(user_request, context=context)

    def ask(self, user_request: str, *, context: dict[str, str] | None = None) -> str:
        return self.invoke(user_request, context=context)

    def __call__(self, user_request: str, *, context: dict[str, str] | None = None) -> str:
        return self.invoke(user_request, context=context)

    def _make_context_tool(self):
        context_provider = self._context_provider

        @tool("get_orchestrator_context")
        def get_orchestrator_context() -> dict[str, str]:
            """What the tool does.

            Returns the current orchestrator capabilities and the latest
            optimization-session context in a compact textual form.

            When to use this tool.

            Use this tool when the user asks what the orchestrator can do,
            which tools are available, or asks follow-up questions about the
            latest optimization results.

            Args:
                None.

            Returns:
                dict[str, str]: Text blocks with current capabilities and the
                latest optimization-session context.
            """

            context = self._active_context or context_provider()
            return {
                "capabilities": str(context.get("capabilities", "")).strip(),
                "latest_optimization_context": str(
                    context.get("latest_optimization_context", "")
                ).strip(),
                "recent_history": str(context.get("recent_history", "")).strip(),
                "latest_response": str(context.get("latest_response", "")).strip(),
            }

        return get_orchestrator_context

    def _make_tool_catalog_tool(self):
        context_provider = self._context_provider

        @tool("get_orchestrator_tool_catalog")
        def get_orchestrator_tool_catalog() -> dict[str, str]:
            """What the tool does.

            Returns a compact catalog of available tools and route branches
            with their descriptions.

            When to use this tool.

            Use this tool when the user asks how the system works, what a
            specific tool or branch does, or how optimization/visualization is
            organized.

            Returns:
                dict[str, str]: Human-readable catalog of tools and branches.
            """

            context = self._active_context or context_provider()
            return {
                "tool_catalog": str(context.get("tool_catalog", "")).strip(),
                "capabilities": str(context.get("capabilities", "")).strip(),
            }

        return get_orchestrator_tool_catalog

    def _build_inline_context(self, user_request: str) -> str:
        """Build an inline context block only when the question needs runtime context."""
        if not self._should_use_tool_agent(user_request):
            return ""
        context = self._active_context or self._context_provider()
        parts: list[str] = []
        capabilities = str(context.get("capabilities", "")).strip()
        tool_catalog = str(context.get("tool_catalog", "")).strip()
        latest = str(context.get("latest_optimization_context", "")).strip()
        recent_history = str(context.get("recent_history", "")).strip()
        latest_response = str(context.get("latest_response", "")).strip()
        if tool_catalog:
            parts.append(f"Каталог инструментов:\n{tool_catalog}")
        if capabilities:
            parts.append(f"Актуальные возможности orchestrator:\n{capabilities}")
        if latest:
            parts.append(f"Контекст последней оптимизации:\n{latest}")
        if recent_history:
            parts.append(f"Недавний диалог:\n{recent_history}")
        if latest_response:
            parts.append(f"Последний ответ orchestrator:\n{latest_response}")
        return "\n\n".join(parts)

    @staticmethod
    def _should_use_context_tool(user_request: str) -> bool:
        text = str(user_request).lower()
        markers = (
            "что ты умеешь",
            "что умеет",
            "какие возможности",
            "какие инструменты",
            "что доступно",
            "orchestrator",
            "urbanomy",
            "последн",
            "предыдущ",
            "истори",
            "оптимизац",
            "pareto",
            "парето",
            "решени",
            "контекст",
        )
        return any(marker in text for marker in markers)

    @classmethod
    def _should_use_tool_agent(cls, user_request: str) -> bool:
        text = str(user_request).lower()
        tool_markers = (
            "как работает",
            "как устро",
            "что делает",
            "что происходит",
            "какой инструмент",
            "какие инструменты",
            "ветк",
            "маршрут",
            "роут",
            "оптимизац",
            "визуализац",
            "pareto",
            "парето",
        )
        return cls._should_use_context_tool(user_request) or any(marker in text for marker in tool_markers)

    def _fallback_answer(self, *, user_request: str) -> str:
        context = self._active_context or self._context_provider()
        capabilities = str(context.get("capabilities", "")).strip()
        tool_catalog = str(context.get("tool_catalog", "")).strip()
        latest = str(context.get("latest_optimization_context", "")).strip()
        recent_history = str(context.get("recent_history", "")).strip()
        if not self._should_use_context_tool(user_request):
            return (
                "Не удалось получить ответ от модели. "
                "Проверь ключ, `base_url` и доступ к LLM."
            )
        parts = ["Я могу объяснять возможности Urbanomy и отвечать на вопросы по текущему контексту."]
        if tool_catalog:
            parts.append(f"Каталог инструментов:\n{tool_catalog}")
        if capabilities:
            parts.append(f"Доступные возможности:\n{capabilities}")
        if latest:
            parts.append(f"Контекст последней оптимизации:\n{latest}")
        if recent_history:
            parts.append(f"Последние сообщения:\n{recent_history}")
        return "\n\n".join(parts)

    @staticmethod
    def _looks_like_tool_artifact(text: str) -> bool:
        normalized = str(text).strip().lower()
        if not normalized:
            return False
        return (
            "get_orchestrator_context" in normalized
            or '"tool"' in normalized
            or '"arguments"' in normalized
        )


def create_general_qa_agent(
    *,
    llm: Any,
    context_provider: Callable[[], dict[str, str]],
) -> GeneralQaAgent:
    """Factory for the general conversational subagent."""
    return GeneralQaAgent(llm=llm, context_provider=context_provider)
