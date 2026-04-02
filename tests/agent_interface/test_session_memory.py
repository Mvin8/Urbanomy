from urbanomy.methods.agent_interface.models import VerificationReport
from urbanomy.methods.agent_interface.urbanomy_orchestrator import UrbanomyOrchestrator


def test_finalize_node_builds_session_memory_snapshot():
    orchestrator = UrbanomyOrchestrator(llm=object(), baseline_blocks=None)
    orchestrator._record_request_analysis(
        thread_id="demo",
        route="visualization",
        user_request="Покажи квартал 12 и стоимость за сотку",
    )
    orchestrator._thread_verification_reports["demo"] = VerificationReport(
        status="ok",
        summary="Проверка пройдена.",
        verified_facts=["Ответ согласован."],
    )
    orchestrator._thread_runtime_outputs["demo"] = {}

    result = orchestrator._finalize_node(
        {
            "thread_id": "demo",
            "user_request": "Покажи квартал 12",
            "latest_route": "visualization",
            "latest_response": "Квартал выделен на карте.",
        }
    )

    assert "history" in result
    snapshot = orchestrator.get_session_memory("demo")
    assert snapshot is not None
    assert snapshot["latest_route"] == "visualization"
    assert snapshot["latest_verification_report"]["status"] == "ok"
    assert snapshot["recent_tasks"][0]["route"] == "visualization"
    assert "Покажи квартал 12" in snapshot["recent_tasks"][0]["title"]
    assert snapshot["latest_research_brief"]["complexity"] == "multi_step"
    assert len(snapshot["active_plan"]["steps"]) >= 2
