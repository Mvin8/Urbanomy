from urbanomy.methods.agent_interface.planning_agent import PlanningAgent
from urbanomy.methods.agent_interface.urbanomy_orchestrator import UrbanomyOrchestrator


def test_planning_agent_extracts_multi_step_optimization_request():
    agent = PlanningAgent()

    research = agent.research(
        user_request="Оптимизируй квартал 86, покажи Pareto front и посчитай метрики решения 2",
        route="district_optimization",
    )
    plan = agent.plan(research_brief=research)

    assert research.complexity == "multi_step"
    assert research.active_target_id == 86
    assert research.requested_solution_number == 2
    assert "запуск оптимизации" in research.requested_outputs
    assert len(plan.steps) >= 3
    assert "run_district_optimization" in agent.involved_tools(research_brief=research)
    assert "plot_district_optimization_pareto_front" in agent.involved_tools(research_brief=research)
    assert "calculate_pareto_solution_investment_metrics" in agent.involved_tools(
        research_brief=research
    )


def test_orchestrator_router_records_research_and_plan():
    orchestrator = UrbanomyOrchestrator(llm=object(), baseline_blocks=None)

    result = orchestrator._router_node(
        {
            "thread_id": "demo",
            "user_request": "Покажи квартал 12 и стоимость за сотку",
        }
    )

    assert result["route"] == "visualization"
    summary = orchestrator.get_memory_summary("demo")
    assert summary["research_brief"] is not None
    assert summary["active_plan"] is not None
    assert summary["research_brief"]["complexity"] == "multi_step"


def test_planning_agent_builds_document_rag_plan_without_registration_for_retrieval():
    agent = PlanningAgent()

    research = agent.research(
        user_request="Найди в документе сведения о населении Гатчины",
        route="document_rag",
    )
    plan = agent.plan(research_brief=research)

    assert research.route == "document_rag"
    assert research.requested_outputs == ["retrieval по документу"]
    assert agent.involved_tools(research_brief=research) == ["retrieve_decision_document_context"]
    assert len(plan.steps) == 1
    assert "retrieval" in plan.steps[0].title.lower()
