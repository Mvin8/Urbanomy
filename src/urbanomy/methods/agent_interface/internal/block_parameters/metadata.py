"""User-facing metadata for the block-parameters domain."""

from __future__ import annotations

from ..common.domain_contracts import ToolDescriptor

BLOCK_PARAMETERS_CAPABILITY_LINES = [
    "- получение параметров квартала по id или target_id из baseline-данных",
]

BLOCK_PARAMETERS_TOOL_DESCRIPTORS = [
    ToolDescriptor(
        name="block_parameters",
        description="Возвращает baseline-параметры квартала по id или target_id без построения карты.",
    )
]
