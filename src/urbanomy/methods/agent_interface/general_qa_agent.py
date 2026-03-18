"""General conversational subagent for the Urbanomy orchestrator."""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from ._agent_utils import build_tool_agent, extract_last_ai_message_text
from .prompts import GENERAL_QA_AGENT_SYSTEM_PROMPT


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
        self.agent = build_tool_agent(
            llm=self.llm,
            tools=[self._context_tool],
            system_prompt=GENERAL_QA_AGENT_SYSTEM_PROMPT,
        )

    def invoke(self, user_request: str, *, context: dict[str, str] | None = None) -> str:
        """Answer a free-form user question in Russian."""
        text = str(user_request).strip()
        if not text:
            raise ValueError("user_request cannot be empty")
        self._active_context = dict(context or self._context_provider())
        try:
            raw_result = self.agent.invoke({"messages": [HumanMessage(content=text)]})
        except Exception:
            return self._fallback_answer()
        response = extract_last_ai_message_text(raw_result)
        return response or self._fallback_answer()

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

    def _fallback_answer(self) -> str:
        context = self._active_context or self._context_provider()
        capabilities = str(context.get("capabilities", "")).strip()
        latest = str(context.get("latest_optimization_context", "")).strip()
        recent_history = str(context.get("recent_history", "")).strip()
        parts = ["Я могу отвечать на общие вопросы и объяснять возможности Urbanomy."]
        if capabilities:
            parts.append(f"Доступные возможности:\n{capabilities}")
        if latest:
            parts.append(f"Контекст последней оптимизации:\n{latest}")
        if recent_history:
            parts.append(f"Последние сообщения:\n{recent_history}")
        return "\n\n".join(parts)


def create_general_qa_agent(
    *,
    llm: Any,
    context_provider: Callable[[], dict[str, str]],
) -> GeneralQaAgent:
    """Factory for the general conversational subagent."""
    return GeneralQaAgent(llm=llm, context_provider=context_provider)
