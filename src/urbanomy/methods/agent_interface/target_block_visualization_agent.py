"""Tool-calling agent for highlighting a target block by id."""

from __future__ import annotations

import re
from typing import Any, TypedDict

import geopandas as gpd
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from ._agent_utils import build_tool_agent, extract_last_ai_message_text
from .models import TargetBlockVisualizationArtifact, TargetBlockVisualizationResult
from .prompts import TARGET_BLOCK_AGENT_PROMPT
from .tools import make_plot_target_block_map_tool


class TargetBlockVisualizationState(TypedDict, total=False):
    """State passed across the target-block visualization graph."""

    user_request: str
    target_id: int
    agent_result: dict[str, Any]
    used_tool_fallback: bool
    result: TargetBlockVisualizationResult


class TargetBlockVisualizationAgent:
    """Highlight a target block on the map using a target_id from the prompt."""

    def __init__(
        self,
        *,
        llm: Any,
        baseline_blocks: gpd.GeoDataFrame,
        id_column: str = "id",
        show_plot: bool = True,
        title_template: str = "Изменяемый квартал  (id={target_id})",
    ) -> None:
        if llm is None:
            raise ValueError("llm is required to build TargetBlockVisualizationAgent.")
        self.llm = llm
        self.id_column = id_column
        self._artifact_store: dict[str, TargetBlockVisualizationArtifact] = {}
        self._plot_target_block_tool = make_plot_target_block_map_tool(
            baseline_blocks=baseline_blocks,
            artifact_store=self._artifact_store,
            id_column=id_column,
            title_template=title_template,
            show_plot=show_plot,
        )
        self._agent = build_tool_agent(
            llm=self.llm,
            tools=[self._plot_target_block_tool],
            system_prompt=TARGET_BLOCK_AGENT_PROMPT,
        )
        self.graph = self._build_graph()

    def invoke(self, user_request: str) -> TargetBlockVisualizationResult:
        """Visualize the target block mentioned in a free-form request."""
        text = str(user_request).strip()
        if not text:
            raise ValueError("user_request cannot be empty")
        self._artifact_store.clear()
        output = self.graph.invoke({"user_request": text})
        return output["result"]

    def run(self, user_request: str) -> TargetBlockVisualizationResult:
        return self.invoke(user_request)

    def ask(self, user_request: str) -> TargetBlockVisualizationResult:
        return self.invoke(user_request)

    def __call__(self, user_request: str) -> TargetBlockVisualizationResult:
        return self.invoke(user_request)

    def _build_graph(self):
        graph = StateGraph(TargetBlockVisualizationState)
        graph.add_node("extract_target_id", self._extract_target_id_node)
        graph.add_node("visualize_target_block", self._visualize_target_block_node)
        graph.add_node("finalize", self._finalize_node)
        graph.add_edge(START, "extract_target_id")
        graph.add_edge("extract_target_id", "visualize_target_block")
        graph.add_edge("visualize_target_block", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def _extract_target_id_node(self, state: TargetBlockVisualizationState) -> dict[str, Any]:
        return {"target_id": self._extract_target_id(state["user_request"])}

    def _visualize_target_block_node(self, state: TargetBlockVisualizationState) -> dict[str, Any]:
        self._artifact_store.pop("plot_target_block_map", None)
        try:
            agent_result = self._agent.invoke({"messages": [HumanMessage(content=state["user_request"])]})
        except Exception:
            agent_result = {}
        used_tool_fallback = False
        if "plot_target_block_map" not in self._artifact_store:
            used_tool_fallback = True
            self._plot_target_block_tool.invoke({"target_id": state["target_id"]})
        return {"agent_result": agent_result, "used_tool_fallback": used_tool_fallback}

    def _finalize_node(self, state: TargetBlockVisualizationState) -> dict[str, Any]:
        artifact = self._artifact_store.get("plot_target_block_map")
        if artifact is None:
            raise RuntimeError("Не удалось построить артефакт визуализации квартала.")
        result = TargetBlockVisualizationResult(
            user_request=state["user_request"],
            route="plot_target_block_map",
            target_id=artifact.target_id,
            title=artifact.title,
            reasoning=f"Из запроса извлечён target_id={artifact.target_id}, квартал выделен на карте.",
            agent_message=extract_last_ai_message_text(state.get("agent_result", {})),
            used_tool_fallback=bool(state.get("used_tool_fallback", False)),
            tool_payload=artifact.tool_payload(),
            artifact=artifact,
        )
        return {"result": result}

    @staticmethod
    def _extract_target_id(user_request: str) -> int:
        patterns = (
            r"target_id\s*[:=]?\s*(\d+)",
            r"\bid\s*[:=]?\s*(\d+)\b",
            r"\bквартал(?:а|у|ом)?\s*(?:с\s*)?(?:id\s*)?[:=]?\s*(\d+)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, user_request, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
        raise ValueError("Не удалось извлечь target_id из запроса.")


def create_target_block_visualization_agent(
    *,
    llm: Any,
    baseline_blocks: gpd.GeoDataFrame,
    id_column: str = "id",
    show_plot: bool = True,
    title_template: str = "Изменяемый квартал  (id={target_id})",
) -> TargetBlockVisualizationAgent:
    """Factory for the target-block visualization agent."""
    return TargetBlockVisualizationAgent(
        llm=llm,
        baseline_blocks=baseline_blocks,
        id_column=id_column,
        show_plot=show_plot,
        title_template=title_template,
    )
