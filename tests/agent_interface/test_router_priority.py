from urbanomy.methods.agent_interface.models import (
    UrbanomyOrchestratorRouteDecision,
    VisualizationRouteDecision,
)
from urbanomy.methods.agent_interface.urbanomy_orchestrator import UrbanomyOrchestrator
from urbanomy.methods.agent_interface.visualization_agent import VisualizationAgent


def test_orchestrator_prefers_high_confidence_llm_route(monkeypatch):
    monkeypatch.setattr(
        "urbanomy.methods.agent_interface.urbanomy_orchestrator.invoke_structured_router",
        lambda **kwargs: UrbanomyOrchestratorRouteDecision(
            route="block_parameters",
            reasoning="LLM semantic route.",
            confidence=0.91,
        ),
    )
    orchestrator = UrbanomyOrchestrator(llm=object(), baseline_blocks=None)

    result = orchestrator._router_node({"user_request": "что известно про квартал 22 по его атрибутам?"})

    assert result["route"] == "block_parameters"


def test_orchestrator_uses_heuristic_when_llm_confidence_is_low(monkeypatch):
    monkeypatch.setattr(
        "urbanomy.methods.agent_interface.urbanomy_orchestrator.invoke_structured_router",
        lambda **kwargs: UrbanomyOrchestratorRouteDecision(
            route="general_qa",
            reasoning="LLM is unsure.",
            confidence=0.2,
        ),
    )
    orchestrator = UrbanomyOrchestrator(llm=object(), baseline_blocks=None)

    result = orchestrator._router_node({"user_request": "Оптимизируй квартал 22"})

    assert result["route"] == "district_optimization"


def test_visualization_uses_heuristic_when_llm_confidence_is_low(monkeypatch):
    monkeypatch.setattr(
        "urbanomy.methods.agent_interface.visualization_agent.invoke_structured_router",
        lambda **kwargs: VisualizationRouteDecision(
            route="plot_total_land_value_map",
            metric_kind="total_land_value",
            price_column="land_value",
            title="Карта стоимости земельных участков (руб.)",
            reasoning="LLM is unsure.",
            confidence=0.25,
        ),
    )
    agent = object.__new__(VisualizationAgent)
    agent.id_column = "id"
    agent.default_total_title = "Карта стоимости земельных участков (руб.)"
    agent.default_unit_title = "Карта стоимости земельных участков за сотку (руб.)"
    agent.target_block_title_template = "Изменяемый квартал  (id={target_id})"

    result = agent._router_node({"user_request": "покажи квартал 12"})

    assert result["route_decision"]["route"] == "plot_target_block_map"
    assert result["route_decision"]["target_id"] == 12
