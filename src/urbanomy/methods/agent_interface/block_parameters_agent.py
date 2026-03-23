"""Deterministic agent for returning baseline block parameters by id."""

from __future__ import annotations

from typing import Any

import geopandas as gpd

from .internal.district_optimization_intents import extract_target_id, normalize_text
from .tools.internal.district_optimization_formatting import format_float, json_value

PREFERRED_BLOCK_PARAMETER_ORDER = (
    "id",
    "site_area",
    "land_use",
    "land_value",
    "land_value_per_100m2",
    "build_floor_area",
    "footprint_area",
    "living_area",
    "non_living_area",
    "population",
    "fsi",
    "gsi",
    "mxi",
    "residential",
    "business",
    "recreation",
    "industrial",
    "transport",
    "special",
    "agriculture",
)


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
    def _build_response(*, target_id: int, parameters: dict[str, Any]) -> str:
        lines = [f"Параметры квартала id={target_id}:"]
        for key, value in parameters.items():
            if isinstance(value, float):
                value_text = format_float(value)
            else:
                value_text = str(value)
            lines.append(f" • {key}: {value_text}")
        return "\n".join(lines)

    def _serialize_row(self, row: Any) -> dict[str, Any]:
        preferred: dict[str, Any] = {}
        remaining: dict[str, Any] = {}
        ordered_columns = [col for col in PREFERRED_BLOCK_PARAMETER_ORDER if col in row.index]
        other_columns = [
            str(col)
            for col in row.index
            if str(col) not in ordered_columns and str(col).lower() != "geometry"
        ]
        for column in ordered_columns + sorted(other_columns):
            if str(column).lower() == "geometry":
                continue
            value = json_value(row[column])
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if column in ordered_columns:
                preferred[str(column)] = value
            else:
                remaining[str(column)] = value
        return {**preferred, **remaining}


def create_block_parameters_agent(
    *,
    baseline_blocks: gpd.GeoDataFrame,
    id_column: str = "id",
) -> BlockParametersAgent:
    """Factory for the deterministic block-parameters agent."""
    return BlockParametersAgent(baseline_blocks=baseline_blocks, id_column=id_column)
