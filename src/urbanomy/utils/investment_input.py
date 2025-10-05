"""Предобработка входных данных для расчёта инвестиционных метрик."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import geopandas as gpd

from urbanomy.methods.investment_potential.constants import DEFAULT_IP_VALUE


INVESTMENT_NUMERIC_COLUMNS: tuple[str, ...] = (
    "price_pred",
    "price_per_sotka",
    "site_area",
    "living_area",
    "non_living_area",
    "build_floor_area",
    "share",
    DEFAULT_IP_VALUE,
)


@dataclass(frozen=True)
class InvestmentInputSpec:
    """Описание колонок, необходимых для расчёта инвестиционной привлекательности."""

    required: Sequence[str]
    optional: Sequence[str]
    defaults: Mapping[str, float]
    geometry_column: str = "geometry"

    def enforce(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError("Ожидается GeoDataFrame")

        if self.geometry_column not in gdf.columns:
            raise ValueError(f"Не найдена колонка геометрии '{self.geometry_column}'")

        missing = [col for col in self.required if col not in gdf.columns]
        if missing:
            raise ValueError(f"Отсутствуют обязательные поля: {missing}")

        ordered_columns: list[str] = [self.geometry_column]
        ordered_columns.extend(self.required)
        ordered_columns.extend([col for col in self.optional if col in gdf.columns])

        trimmed = gdf.loc[:, ordered_columns].copy()
        for col in self.optional:
            default_value = self.defaults.get(col, 0.0)
            if col not in trimmed.columns:
                trimmed[col] = default_value
            else:
                trimmed[col] = trimmed[col].fillna(default_value)

        return trimmed


INPUT_SPEC = InvestmentInputSpec(
    required=("land_use", "price_pred"),
    optional=(
        "site_area",
        "living_area",
        "non_living_area",
        "build_floor_area",
        "share",
        "price_per_sotka",
        DEFAULT_IP_VALUE,
    ),
    defaults={
        "site_area": 0.0,
        "living_area": 0.0,
        "non_living_area": 0.0,
        "build_floor_area": 0.0,
        "share": 1.0,
        "price_per_sotka": 0.0,
        DEFAULT_IP_VALUE: 0.0,
    },
)


def prepare_investment_input(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Фильтрует входной GeoDataFrame и заполняет отсутствующие поля дефолтами."""

    return INPUT_SPEC.enforce(gdf)


__all__ = [
    "INVESTMENT_NUMERIC_COLUMNS",
    "InvestmentInputSpec",
    "INPUT_SPEC",
    "prepare_investment_input",
]
