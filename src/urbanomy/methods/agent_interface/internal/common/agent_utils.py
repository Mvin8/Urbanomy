"""Shared helpers for LangChain/LangGraph-based agents in Urbanomy."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

try:
    from langchain.agents import create_agent
except ImportError as exc:
    raise ImportError(
        "Urbanomy agent interface requires LangChain agents support. "
        "Install Urbanomy with the 'llm' extra."
    ) from exc


def build_tool_agent(*, llm: Any, tools: list[Any], system_prompt: str) -> Any:
    """Create a LangChain agent with a consistent Urbanomy setup."""
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt)


def invoke_structured_router(
    *,
    llm: Any,
    schema: Any,
    system_prompt: str,
    user_request: str,
) -> Any | None:
    """Try structured routing and return ``None`` if it fails."""
    if not hasattr(llm, "with_structured_output"):
        return None
    try:
        router = llm.with_structured_output(schema)
        decision = router.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_request),
            ]
        )
    except Exception:
        return None
    return schema.model_validate(decision)


def extract_message_text(message: Any) -> str:
    """Convert a LangChain message content payload to plain text."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    parts.append(text)
                continue
            if isinstance(item, dict):
                item_type = str(item.get("type", "")).strip().lower()
                text_value = item.get("text")
                if isinstance(text_value, str) and (
                    item_type in {"", "text", "output_text", "input_text"}
                ):
                    text = text_value.strip()
                    if text:
                        parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def extract_last_ai_message_text(agent_result: dict[str, Any]) -> str:
    """Return the last non-empty AI message text from an agent result."""
    messages = agent_result.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = extract_message_text(message)
            if text:
                return text
    return ""


def extract_last_tool_message(agent_result: dict[str, Any]) -> ToolMessage | None:
    """Return the last tool message from an agent result."""
    messages = agent_result.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            return message
    return None

