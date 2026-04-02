from urbanomy.methods.agent_interface.block_parameters_agent import BlockParametersAgent
from urbanomy.methods.agent_interface.internal.common.domain_contracts import ToolDescriptor
from urbanomy.methods.agent_interface.urbanomy_orchestrator import UrbanomyOrchestrator


def test_block_parameters_agent_exposes_public_catalog_descriptor():
    descriptors = BlockParametersAgent.tool_descriptors()

    assert descriptors == [
        ToolDescriptor(
            name="block_parameters",
            description=(
                "Возвращает baseline-параметры квартала по id или target_id "
                "без построения карты."
            ),
        )
    ]


def test_orchestrator_build_tool_catalog_uses_domain_descriptors():
    orchestrator = UrbanomyOrchestrator(llm=object(), baseline_blocks=None)
    orchestrator.district_optimization_config = object()

    class _DummyDomain:
        def __init__(self, descriptors):
            self._descriptors = descriptors

        def tool_descriptors(self):
            return list(self._descriptors)

        def capability_lines(self):
            return []

    orchestrator._visualization_agent = _DummyDomain(
        [ToolDescriptor(name="plot_x", description="Рисует X.")]
    )
    orchestrator._block_parameters_agent = _DummyDomain(
        [ToolDescriptor(name="block_parameters", description="Показывает параметры блока.")]
    )
    orchestrator._district_optimization_agent = _DummyDomain(
        [ToolDescriptor(name="run_opt", description="Запускает оптимизацию.")]
    )

    catalog = orchestrator._build_tool_catalog_text()

    assert "- plot_x: Рисует X." in catalog
    assert "- block_parameters: Показывает параметры блока." in catalog
    assert "- run_opt: Запускает оптимизацию." in catalog


def test_unsupported_node_reports_dynamic_capabilities():
    orchestrator = UrbanomyOrchestrator(llm=object(), baseline_blocks=None)
    orchestrator._build_capabilities_text = lambda: "- тестовая возможность"

    result = orchestrator._unsupported_node({"thread_id": "demo"})

    assert result["latest_route"] == "unsupported"
    assert "тестовая возможность" in result["latest_response"]
