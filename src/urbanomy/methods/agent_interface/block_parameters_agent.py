"""Deterministic agent for returning baseline block parameters by id."""

from __future__ import annotations

from typing import Any

import geopandas as gpd

from .internal.block_parameters.logic import (
    build_block_parameters_response,
    extract_requested_parameters,
    serialize_block_row,
)
from .internal.block_parameters.metadata import (
    BLOCK_PARAMETERS_CAPABILITY_LINES,
    BLOCK_PARAMETERS_TOOL_DESCRIPTORS,
)
from .internal.common.domain_contracts import ToolDescriptor
from .internal.common.request_parsing import extract_target_id, normalize_text


def looks_like_block_parameters_request(user_request: str) -> bool:
    """Return whether the request asks for one block's parameters by id."""
    text = normalize_text(user_request)
    if extract_target_id(user_request) is None:
        return False
    parameter_markers = (
        "параметр",
        "характеристик",
        "атрибут",
        "данные",
        "свойств",
    )
    block_markers = (
        "квартал",
        "блок",
        "target_id",
        " id",
        "id=",
    )
    return any(marker in text for marker in parameter_markers) and any(
        marker in text for marker in block_markers
    )


class BlockParametersAgent:
    """Return compact baseline-block parameters for one block id."""

    def __init__(
        self,
        *,
        baseline_blocks: gpd.GeoDataFrame,
        id_column: str = "id",
    ) -> None:
        self.baseline_blocks = baseline_blocks
        self.id_column = str(id_column).strip() or "id"

    def invoke(self, user_request: str) -> dict[str, Any]:
        text = str(user_request).strip()
        if not text:
            raise ValueError("user_request cannot be empty")
        requested_parameters = self._extract_requested_parameters(text)
        target_id = extract_target_id(text)
        if target_id is None:
            return {
                "status": "error",
                "response": "Не найден id квартала. Укажите, например: 'Какие параметры квартала 20?'",
                "target_id": None,
                "parameters": None,
            }
        if self.id_column not in self.baseline_blocks.columns:
            return {
                "status": "error",
                "response": f"В baseline-данных нет колонки `{self.id_column}`.",
                "target_id": target_id,
                "parameters": None,
            }
        matches = self.baseline_blocks.loc[self.baseline_blocks[self.id_column] == target_id]
        if matches.empty:
            return {
                "status": "error",
                "response": f"Квартал с id={target_id} не найден.",
                "target_id": target_id,
                "parameters": None,
            }
        row = matches.iloc[0]
        parameters = self._serialize_row(row)
        if requested_parameters:
            filtered_parameters = {
                key: value for key, value in parameters.items() if key in requested_parameters
            }
            if filtered_parameters:
                parameters = filtered_parameters
        return {
            "status": "ok",
            "response": self._build_response(target_id=target_id, parameters=parameters),
            "target_id": target_id,
            "parameters": parameters,
        }

    def run(self, user_request: str) -> dict[str, Any]:
        return self.invoke(user_request)

    def ask(self, user_request: str) -> dict[str, Any]:
        return self.invoke(user_request)

    def __call__(self, user_request: str) -> dict[str, Any]:
        return self.invoke(user_request)

    @staticmethod
    def capability_lines() -> list[str]:
        """Return user-facing capabilities exposed by this domain agent."""
        return list(BLOCK_PARAMETERS_CAPABILITY_LINES)

    @staticmethod
    def tool_descriptors() -> list[ToolDescriptor]:
        """Return a synthetic tool descriptor for catalog rendering."""
        return list(BLOCK_PARAMETERS_TOOL_DESCRIPTORS)

    @staticmethod
    def _build_response(*, target_id: int, parameters: dict[str, Any]) -> str:
        return build_block_parameters_response(target_id=target_id, parameters=parameters)

    @staticmethod
    def _extract_requested_parameters(user_request: str) -> list[str]:
        return extract_requested_parameters(user_request)

    def _serialize_row(self, row: Any) -> dict[str, Any]:
        return serialize_block_row(row)


def create_block_parameters_agent(
    *,
    baseline_blocks: gpd.GeoDataFrame,
    id_column: str = "id",
) -> BlockParametersAgent:
    """Factory for the deterministic block-parameters agent."""
    return BlockParametersAgent(baseline_blocks=baseline_blocks, id_column=id_column)
