from types import SimpleNamespace

from urbanomy.methods.agent_interface.verification_agent import VerificationAgent


def test_verification_agent_accepts_consistent_target_block_visualization():
    agent = VerificationAgent()

    report = agent.verify(
        route="visualization",
        result=SimpleNamespace(
            route="plot_target_block_map",
            title="Квартал 12",
            tool_payload={"target_id": 12},
            target_id=12,
            artifact=object(),
        ),
    )

    assert report.status == "ok"
    assert any("id=12" in item for item in report.verified_facts)


def test_verification_agent_flags_invalid_solution_number():
    agent = VerificationAgent()

    report = agent.verify(
        route="district_optimization",
        result={
            "status": "ok",
            "response": "Решение 7 показано.",
            "tool_name": "get_pareto_solution_parameters",
            "tool_output": {"solution_number": 7},
        },
        optimization_summary={"target_id": 86, "n_solutions": 3},
    )

    assert report.status == "error"
    assert any("solution_number=7" in issue.message for issue in report.issues)
