from types import SimpleNamespace

from urbanomy.methods.agent_interface.urbanomy_orchestrator import UrbanomyOrchestrator


def test_multi_step_request_returns_pending_confirmation_gate():
    orchestrator = UrbanomyOrchestrator(llm=object(), baseline_blocks=None)
    graph_calls: list[dict] = []
    orchestrator.graph = SimpleNamespace(
        invoke=lambda state, config: graph_calls.append(state) or state
    )

    result = orchestrator.invoke(
        "Оптимизируй квартал 86, покажи Pareto front и посчитай метрики решения 2",
        thread_id="demo",
    )

    assert result.confirmation_gate is not None
    assert result.confirmation_gate.status == "pending"
    assert "run_district_optimization" in result.response
    assert "plot_district_optimization_pareto_front" in result.response
    assert "calculate_pareto_solution_investment_metrics" in result.response
    assert graph_calls == []


def test_confirmation_yes_executes_original_request():
    orchestrator = UrbanomyOrchestrator(llm=object(), baseline_blocks=None)
    graph_calls: list[dict] = []

    def _invoke(state, config):
        graph_calls.append(dict(state))
        return {
            "user_request": state["user_request"],
            "latest_route": "district_optimization",
            "reasoning": "Выполняем optimization flow.",
            "latest_response": "Оптимизация завершена.",
        }

    orchestrator.graph = SimpleNamespace(invoke=_invoke)
    first = orchestrator.invoke(
        "Оптимизируй квартал 86, покажи Pareto front и посчитай метрики решения 2",
        thread_id="demo",
    )

    assert first.confirmation_gate is not None
    second = orchestrator.invoke("да", thread_id="demo")

    assert len(graph_calls) == 1
    assert graph_calls[0]["user_request"].startswith("Оптимизируй квартал 86")
    assert second.confirmation_gate is not None
    assert second.confirmation_gate.status == "approved"
    assert second.response.startswith("Подтверждение принято.")


def test_confirmation_no_cancels_pending_plan():
    orchestrator = UrbanomyOrchestrator(llm=object(), baseline_blocks=None)
    orchestrator.graph = SimpleNamespace(
        invoke=lambda state, config: {
            "user_request": state["user_request"],
            "latest_route": "district_optimization",
            "reasoning": "unused",
            "latest_response": "unused",
        }
    )

    orchestrator.invoke(
        "Оптимизируй квартал 86, покажи Pareto front и посчитай метрики решения 2",
        thread_id="demo",
    )
    result = orchestrator.invoke("нет", thread_id="demo")

    assert result.confirmation_gate is not None
    assert result.confirmation_gate.status == "cancelled"
    assert "План отменён" in result.response


def test_document_query_bypasses_confirmation_gate():
    orchestrator = UrbanomyOrchestrator(llm=object(), baseline_blocks=None)
    graph_calls: list[dict] = []

    def _invoke(state, config):
        graph_calls.append(dict(state))
        return {
            "user_request": state["user_request"],
            "latest_route": "document_rag",
            "reasoning": "Выполняем document RAG flow.",
            "latest_response": "Нашёл фрагменты документа.",
        }

    orchestrator.graph = SimpleNamespace(invoke=_invoke)

    result = orchestrator.invoke(
        "Найди в документе сведения о населении Гатчины",
        thread_id="demo",
    )

    assert result.confirmation_gate is None
    assert result.route == "document_rag"
    assert graph_calls == [
        {
            "user_request": "Найди в документе сведения о населении Гатчины",
            "thread_id": "demo",
        }
    ]
