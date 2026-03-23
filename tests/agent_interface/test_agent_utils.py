from langchain_core.messages import AIMessage

from urbanomy.methods.agent_interface.internal.agent_utils import extract_message_text


def test_extract_message_text_ignores_tool_blocks():
    message = AIMessage(
        content=[
            {"type": "tool_use", "name": "get_orchestrator_context", "input": {}},
            {"type": "text", "text": "Получаю текущий контекст."},
        ]
    )

    assert extract_message_text(message) == "Получаю текущий контекст."
