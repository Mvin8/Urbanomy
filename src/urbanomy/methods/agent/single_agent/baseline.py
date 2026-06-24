from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from .graph import SingleAgentState, build_single_agent_graph


class SingleAgentBaseline:
    """Small wrapper around a one-node LangGraph baseline."""

    def __init__(self, llm: BaseChatModel) -> None:
        self.graph = build_single_agent_graph(llm=llm)

    def invoke_state(self, text: str, config: dict[str, Any] | None = None) -> SingleAgentState:
        return self.graph.invoke({"input": text, "output": "", "log": []}, config=config)

    def stream(self, text: str, config: dict[str, Any] | None = None):
        return self.graph.stream({"input": text, "output": "", "log": []}, config=config)


__all__ = ["SingleAgentBaseline"]
