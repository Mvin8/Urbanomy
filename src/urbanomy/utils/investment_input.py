"""Предобработка входных данных для расчёта инвестиционных метрик."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import geopandas as gpd
import pandas as pd

# Дублируем значение, определённое в
# ``urbanomy.methods.investment_potential.constants.DEFAULT_IP_VALUE``
# ("ip_value"), чтобы избежать циклического импорта при загрузке модуля.
DEFAULT_IP_VALUE: str = "ip_value"


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

DEFAULT_SCENARIO_KEEP_COLUMNS: tuple[str, ...] = (
    "y_log_pred",
    "price_pred",
    "price_per_sotka",
    "is_scn",
    "residential",
    "business",
    "recreation",
    "industrial",
    "transport",
    "special",
    "agriculture",
    "land_use",
    "share",
    "footprint_area",
    "build_floor_area",
    "living_area",
    "non_living_area",
    "population",
    "site_area",
    "fsi",
    "gsi",
    "mxi",
    "l",
    "morphotype",
    "area_accessibility",
    "geometry",
)

DEFAULT_ALLOWED_IP_USES: tuple[str, ...] = (
    "residential",
    "business",
    "recreation",
    "industrial",
    "transport",
    "special",
    "agriculture",
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


def _ensure_geodataframe(data: gpd.GeoDataFrame | pd.DataFrame) -> gpd.GeoDataFrame:
    if isinstance(data, gpd.GeoDataFrame):
        return data
    if isinstance(data, pd.DataFrame):
        if "geometry" not in data.columns:
            raise ValueError("Ожидается GeoDataFrame или DataFrame с колонкой 'geometry'")
        return gpd.GeoDataFrame(data, geometry="geometry", crs=getattr(data, "crs", None))
    raise TypeError("Ожидается GeoDataFrame или DataFrame")


def _build_ip_value_lookup(
    base_gdf: gpd.GeoDataFrame,
    allowed_uses: Iterable[str],
    *,
    ip_type_column: str,
    ip_value_column: str,
) -> pd.DataFrame:
    if ip_type_column not in base_gdf.columns:
        raise ValueError(f"В base_gdf отсутствует колонка '{ip_type_column}'")
    if ip_value_column not in base_gdf.columns:
        raise ValueError(f"В base_gdf отсутствует колонка '{ip_value_column}'")

    working = base_gdf.copy()
    working[ip_type_column] = working[ip_type_column].astype(str).str.lower()
    allowed = tuple(allowed_uses)
    if allowed:
        working = working[working[ip_type_column].isin(allowed)]

    return (
        working.groupby(ip_type_column, as_index=False)[ip_value_column]
        .mean()
        .rename(columns={ip_value_column: "ip_value_from_base"})
    )


def _prepare_with_base(
    polygon_gdf: gpd.GeoDataFrame,
    base_gdf: gpd.GeoDataFrame,
    *,
    keep_columns: Sequence[str] | None,
    allowed_uses: Iterable[str],
    land_use_column: str,
    ip_type_column: str,
    scenario_flag_column: str,
    land_use_prefix_pattern: str,
    ip_value_column: str,
) -> gpd.GeoDataFrame:
    geometry_column = polygon_gdf.geometry.name
    keep_columns = tuple(keep_columns or DEFAULT_SCENARIO_KEEP_COLUMNS)
    existing_keep = [col for col in keep_columns if col in polygon_gdf.columns]

    if land_use_column not in polygon_gdf.columns:
        raise ValueError(f"В polygon_gdf отсутствует колонка '{land_use_column}'")

    ordered_columns: list[str] = [geometry_column]
    ordered_columns.extend(
        col for col in existing_keep if col not in {geometry_column, land_use_column}
    )
    if land_use_column != geometry_column:
        ordered_columns.insert(1, land_use_column)

    working = polygon_gdf.loc[:, ordered_columns].copy()

    working[land_use_column] = (
        working[land_use_column]
        .astype(str)
        .str.replace(land_use_prefix_pattern, "", regex=True)
    )

    if scenario_flag_column in working.columns:
        mask = working[scenario_flag_column].fillna(False).astype(bool)
        working = working.loc[mask].reset_index(drop=True)

    working[ip_type_column] = working[land_use_column].astype(str).str.lower()
    working = working[
        working[ip_type_column].notna() & (working[ip_type_column] != "none")
    ]

    base_lookup = _build_ip_value_lookup(
        base_gdf,
        allowed_uses=allowed_uses,
        ip_type_column=ip_type_column,
        ip_value_column=ip_value_column,
    )

    working = working.merge(base_lookup, on=ip_type_column, how="left")
    working[ip_value_column] = working.pop("ip_value_from_base").fillna(0.0)

    return working


def prepare_investment_input(
    gdf: gpd.GeoDataFrame,
    project_potential: gpd.GeoDataFrame | pd.DataFrame | None = None,
    *,
    keep_columns: Sequence[str] | None = None,
    allowed_uses: Iterable[str] | None = None,
    land_use_column: str = "land_use",
    ip_type_column: str = "ip_type",
    scenario_flag_column: str = "is_scn",
    land_use_prefix_pattern: str = r"^LandUse\.",
    ip_value_column: str = DEFAULT_IP_VALUE,
) -> gpd.GeoDataFrame:
    """Готовит входной GeoDataFrame для расчёта инвестиционных метрик.

    При наличии ``base_gdf`` функция повторяет сценарную предобработку из нотбука:

    - оставляет нужные колонки и нормализует ``land_use``;
    - фильтрует сценические полигоны по ``is_scn``;
    - объединяет данные с базовыми значениями ``ip_value`` по типу землепользования;
    - заполняет отсутствующие значения нулями.

    После предобработки колонки приводятся к требуемому набору ``INPUT_SPEC``.
    """

    polygon_gdf = _ensure_geodataframe(gdf)
    if project_potential is not None:
        base_ready = _ensure_geodataframe(project_potential)
        polygon_gdf = _prepare_with_base(
            polygon_gdf,
            base_ready,
            keep_columns=keep_columns,
            allowed_uses=tuple(allowed_uses or DEFAULT_ALLOWED_IP_USES),
            land_use_column=land_use_column,
            ip_type_column=ip_type_column,
            scenario_flag_column=scenario_flag_column,
            land_use_prefix_pattern=land_use_prefix_pattern,
            ip_value_column=ip_value_column,
        )

    return INPUT_SPEC.enforce(polygon_gdf)


__all__ = [
    "INVESTMENT_NUMERIC_COLUMNS",
    "DEFAULT_SCENARIO_KEEP_COLUMNS",
    "DEFAULT_ALLOWED_IP_USES",
    "InvestmentInputSpec",
    "INPUT_SPEC",
    "prepare_investment_input",
]
