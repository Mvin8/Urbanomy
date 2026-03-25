from types import SimpleNamespace

from urbanomy.methods.agent_interface.general_qa_agent import GeneralQaAgent
from urbanomy.methods.agent_interface.tools.get_block_parameter_definition import (
    make_get_block_parameter_definition_tool,
)


class _StubLlm:
    def __init__(self):
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return SimpleNamespace(content="Обычный ответ")


def test_general_qa_direct_without_context_uses_freeform_prompt():
    agent = object.__new__(GeneralQaAgent)
    agent.llm = _StubLlm()
    agent._context_provider = lambda: {}
    agent._active_context = {}

    response = agent._invoke_direct("что такое город?")

    assert response == "Обычный ответ"
    system_prompt = agent.llm.last_messages[0].content
    assert "отвечай как обычный ассистент" in system_prompt.lower()
    assert "данных в контексте недостаточно" not in system_prompt.lower()


def test_general_qa_answers_block_parameter_glossary_without_llm():
    agent = object.__new__(GeneralQaAgent)
    agent.llm = _StubLlm()
    agent._context_provider = lambda: {}
    agent._active_context = {}
    agent._parameter_definition_tool = make_get_block_parameter_definition_tool()

    response = agent._maybe_answer_block_parameter_definition("Что такое FSI в параметрах кварталов?")

    assert "Floor Space Index" in response
    assert "fsi = build_floor_area / site_area" in response
