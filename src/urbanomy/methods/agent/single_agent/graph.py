from typing import Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph


class SingleAgentState(TypedDict):
    input: str
    output: str
    log: list[BaseMessage]


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item if isinstance(item, str) else item.get("text", "")
            for item in content
            if isinstance(item, str) or isinstance(item, dict)
        )
    return str(content)


def build_single_agent_graph(llm: BaseChatModel) -> CompiledStateGraph:
    """Build a LangGraph baseline with one LLM node."""

    def single_agent_node(state: SingleAgentState) -> SingleAgentState:
        messages = [HumanMessage(content=state["input"])]
        response = llm.invoke(messages)
        return {
            "input": state["input"],
            "output": _message_content_to_text(response.content),
            "log": [*state.get("log", []), *messages, response],
        }

    graph = StateGraph(SingleAgentState)
    graph.add_node("single_agent", single_agent_node)
    graph.add_edge(START, "single_agent")
    graph.add_edge("single_agent", END)
    return graph.compile()


__all__ = ["SingleAgentState", "build_single_agent_graph"]
