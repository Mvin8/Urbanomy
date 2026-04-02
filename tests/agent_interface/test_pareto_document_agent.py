from types import SimpleNamespace

import pandas as pd

from urbanomy.methods.agent_interface.models import DecisionDocumentChunk, RetrievedDocumentChunk
from urbanomy.methods.agent_interface.pareto_document_agent import ParetoDocumentAgent
from urbanomy.methods.agent_interface.planning_agent import PlanningAgent


class FakeDecisionDocumentStore:
    def __init__(self) -> None:
        self._chunks: list[DecisionDocumentChunk] = []

    def add_document(
        self,
        *,
        path: str | None,
        document_name: str,
        document_text: str | None,
        document_type: str,
    ) -> list[DecisionDocumentChunk]:
        text = str(document_text or "").strip() or document_name
        self._chunks = [
            DecisionDocumentChunk(
                chunk_id="chunk-1",
                text=text,
                char_start=0,
                char_end=len(text),
                keywords=["экология", "рекреация", "жилье"],
            ),
            DecisionDocumentChunk(
                chunk_id="chunk-2",
                text="Приоритеты: парки, озеленение, снижение промышленной нагрузки.",
                char_start=max(0, len(text) - 20),
                char_end=max(len(text), len(text) + 60),
                keywords=["парки", "озеленение", "промышленной"],
            ),
        ]
        return list(self._chunks)

    def retrieve(self, *, query: str, top_k: int = 4) -> list[RetrievedDocumentChunk]:
        base = [
            RetrievedDocumentChunk(
                chunk_id="chunk-2",
                score=0.92,
                text="Приоритеты: парки, озеленение, снижение промышленной нагрузки.",
                rationale="Совпали термины: парки, озеленение",
            ),
            RetrievedDocumentChunk(
                chunk_id="chunk-1",
                score=0.81,
                text="Генеральный план развития. Приоритеты: жилье, парки, экология.",
                rationale="Совпали термины: экология",
            ),
        ]
        return base[:top_k]


def test_pareto_document_agent_registers_active_document_from_text():
    session_store: dict[str, object] = {
        "decision_document_store_factory": FakeDecisionDocumentStore,
    }
    agent = ParetoDocumentAgent(llm=object(), session_store=session_store)

    result = agent.invoke(
        "Добавь документ: Генеральный план развития. Приоритеты: жилье, парки, экология."
    )

    assert result["status"] == "ok"
    assert result["tool_name"] == "register_decision_document"
    assert result["tool_output"]["document_name"]
    assert result["tool_output"]["chunk_count"] >= 1
    assert result["tool_output"]["retrieval_backend"] == "ollama_inmemory_vectorstore"
    assert "active_decision_document" in session_store


def test_pareto_document_agent_selects_recreation_friendly_solution():
    session_store: dict[str, object] = {
        "decision_document_store_factory": FakeDecisionDocumentStore,
        "latest_district_optimization_session": SimpleNamespace(
            target_id=86,
            pareto_front_df=pd.DataFrame(
                [
                    {
                        "scenario_id": "scenario_0",
                        "title": "scenario_0 | business",
                        "summary": "Деловой сценарий",
                        "land_use": "business",
                        "land_value_gain": 100.0,
                        "investor_npv": 90.0,
                        "params_repaired": {
                            "land_use": "business",
                            "residential": 0.2,
                            "business": 0.7,
                            "recreation": 0.1,
                            "industrial": 0.0,
                        },
                    },
                    {
                        "scenario_id": "scenario_1",
                        "title": "scenario_1 | recreation",
                        "summary": "Экологичный сценарий с парками",
                        "land_use": "recreation",
                        "land_value_gain": 80.0,
                        "investor_npv": 55.0,
                        "params_repaired": {
                            "land_use": "recreation",
                            "residential": 0.3,
                            "business": 0.2,
                            "recreation": 0.5,
                            "industrial": 0.0,
                        },
                    },
                    {
                        "scenario_id": "scenario_2",
                        "title": "scenario_2 | industrial",
                        "summary": "Промышленный сценарий",
                        "land_use": "industrial",
                        "land_value_gain": 120.0,
                        "investor_npv": 95.0,
                        "params_repaired": {
                            "land_use": "industrial",
                            "residential": 0.1,
                            "business": 0.1,
                            "recreation": 0.1,
                            "industrial": 0.7,
                        },
                    },
                ]
            ),
        )
    }
    agent = ParetoDocumentAgent(llm=object(), session_store=session_store)
    agent.invoke(
        "Сохрани документ: Стратегия развития. Приоритеты: экология, озеленение, рекреация, снижение промышленной нагрузки."
    )

    result = agent.invoke("Выбери лучшее решение по документу.")

    assert result["status"] == "ok"
    assert result["tool_name"] == "select_pareto_solution_by_document"
    assert result["tool_output"]["selected_solution_number"] == 2
    assert result["tool_output"]["retrieved_chunks"]
    assert "эколог" in result["response"].lower() or "рекреац" in result["response"].lower()


def test_pareto_document_agent_retrieves_relevant_document_chunks():
    session_store: dict[str, object] = {
        "decision_document_store_factory": FakeDecisionDocumentStore,
    }
    agent = ParetoDocumentAgent(llm=object(), session_store=session_store)
    agent.invoke(
        "Сохрани документ: Стратегия развития. Приоритеты: экология, озеленение, рекреация."
    )

    result = agent.invoke("Покажи фрагменты документа про экологию и озеленение")

    assert result["status"] == "ok"
    assert result["tool_name"] == "retrieve_decision_document_context"
    assert len(result["tool_output"]["chunks"]) >= 1
    assert "фрагмент" in result["response"].lower()


def test_pareto_document_agent_treats_natural_document_question_as_retrieval():
    session_store: dict[str, object] = {
        "decision_document_store_factory": FakeDecisionDocumentStore,
    }
    agent = ParetoDocumentAgent(llm=object(), session_store=session_store)
    agent.invoke(
        "Сохрани документ: Стратегия развития. Население Гатчины составляет 92 937 человек."
    )

    result = agent.invoke("Какое точное число населения Гатчины согласно документу?")

    assert result["status"] == "ok"
    assert result["tool_name"] == "retrieve_decision_document_context"
    assert len(result["tool_output"]["chunks"]) >= 1


def test_planning_agent_includes_document_selection_tools():
    planning = PlanningAgent()

    research = planning.research(
        user_request="Оптимизируй квартал 86 и выбери лучшее решение по стратегии развития",
        route="district_optimization",
    )

    tools = planning.involved_tools(research_brief=research)

    assert "run_district_optimization" in tools
    assert "select_pareto_solution_by_document" in tools
    assert "retrieve_decision_document_context" in tools
