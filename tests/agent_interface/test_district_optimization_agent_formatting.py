from urbanomy.methods.agent_interface.internal.district_optimization_agent_formatting import (
    short_response_from_tool,
)


def test_problem_statement_returns_full_text_for_detailed_request():
    tool_output = {
        "target_id": None,
        "decision_variables": [],
        "tunable_parameters": {
            "algorithm": {
                "pop_size_default": 10,
                "n_gen_default": 15,
                "seed_default": 42,
            }
        },
        "problem_statement_text": "ПОЛНЫЙ ТЕКСТ ПОСТАНОВКИ",
    }

    response = short_response_from_tool(
        tool_name="get_district_optimization_problem_statement",
        tool_output=tool_output,
        user_request="Покажи все параметры оптимизации",
    )

    assert response == "ПОЛНЫЙ ТЕКСТ ПОСТАНОВКИ"


def test_problem_statement_returns_compact_text_by_default():
    tool_output = {
        "target_id": None,
        "decision_variables": [{"name": "footprint_area"}],
        "tunable_parameters": {
            "algorithm": {
                "pop_size_default": 10,
                "n_gen_default": 15,
                "seed_default": 42,
            }
        },
        "problem_statement_text": "ПОЛНЫЙ ТЕКСТ ПОСТАНОВКИ",
    }

    response = short_response_from_tool(
        tool_name="get_district_optimization_problem_statement",
        tool_output=tool_output,
        user_request="Какие параметры оптимизации?",
    )

    assert "Постановка задачи оптимизации готова." in response
    assert "ПОЛНЫЙ ТЕКСТ ПОСТАНОВКИ" not in response
