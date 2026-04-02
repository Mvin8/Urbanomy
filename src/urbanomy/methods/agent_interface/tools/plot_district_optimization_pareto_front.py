"""Tool for plotting the Pareto front of the latest optimization session."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from .internal.district_optimization import latest_session_or_error, plot_optimization_pareto_front


def make_plot_district_optimization_pareto_front_tool(*, session_store: dict[str, Any]):
    """Create a LangChain tool bound to the latest district optimization session."""

    @tool("plot_district_optimization_pareto_front")
    def plot_district_optimization_pareto_front() -> dict[str, Any]:
        """What the tool does.

        Builds a scatter plot of optimization solutions and overlays the Pareto
        front for the latest district optimization session.

        When to use this tool.

        Use this tool when the user asks to show the Pareto front, optimization
        graph, scatter plot of solutions, or the graph after optimization.

        Args:
            None.

        Returns:
            dict[str, Any]: Compact summary of the plotted Pareto front,
            including counts of plotted points and LANDUSE categories. The
            figure is created as a side effect.

        Restrictions:
            Requires that a district optimization session has already been run
            in the current agent instance.
        """
        session = latest_session_or_error(session_store)
        return plot_optimization_pareto_front(session)

    return plot_district_optimization_pareto_front
