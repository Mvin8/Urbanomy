from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Mapping

from blocksnet.enums import LandUse


@dataclass(frozen=True)
class LandUseConfig:
    """Immutable container for land-use related parameters."""

    potential_column: str
    indicator_weights: Mapping[str, float] = field(default_factory=dict)
    investment_weights: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        # Ensure indicator weights stay read-only after construction.
        object.__setattr__(
            self,
            "indicator_weights",
            MappingProxyType(dict(self.indicator_weights)),
        )


LAND_USE_CONFIGS: Final[dict[LandUse, LandUseConfig]] = {
    LandUse.RESIDENTIAL: LandUseConfig(
        potential_column="Потенциал развития жилой застройки",
        indicator_weights={
            "Население": 1.3,
            "Социальное обеспечение": 1.4,
            "Экологическая ситуация": 1.5,
            "Средняя доступность до близлежащего крупного населенного пункта": 1.2,
            "Транспортное обеспечение": 1.1,
            "default": 1.0,
        },
        investment_weights=(0.4, 0.6),
    ),
    LandUse.BUSINESS: LandUseConfig(
        potential_column=(
            "Потенциал развития застройки общественно-деловой зоны"
        ),
        indicator_weights={
            "Транспортное обеспечение": 1.5,
            "Население": 1.4,
            "Социальное обеспечение (комфорт)": 1.3,
            "Средняя доступность до близлежащего крупного населенного пункта": 1.2,
            "default": 1.0,
        },
        investment_weights=(0.3, 0.7),
    ),
    LandUse.RECREATION: LandUseConfig(
        potential_column="Потенциал развития застройки рекреационной зоны",
        indicator_weights={
            "Экологическая ситуация": 1.5,
            "Социальное обеспечение (комфорт)": 1.4,
            "Транспортное обеспечение": 1.2,
            "Население": 0.8,
            "default": 1.0,
        },
        investment_weights=(0.6, 0.4),
    ),
    LandUse.SPECIAL: LandUseConfig(
        potential_column=(
            "Потенциал развития застройки зоны специального назначения"
        ),
        indicator_weights={
            "Потенциал размещения порта": 1.5,
            "Транспортное обеспечение": 1.4,
            "Потенциал размещения логистического, складского комплекса": 1.3,
            "default": 1.0,
        },
        investment_weights=(0.3, 0.7),
    ),
    LandUse.INDUSTRIAL: LandUseConfig(
        potential_column="Потенциал развития застройки промышленной зоны",
        indicator_weights={
            "Потенциал размещения логистического, складского комплекса": 1.5,
            "Транспортное обеспечение": 1.4,
            "Экологическая ситуация": 0.8,
            "Население": 0.9,
            "default": 1.0,
        },
        investment_weights=(0.35, 0.65),
    ),
    LandUse.AGRICULTURE: LandUseConfig(
        potential_column="Потенциал развития застройки сельскохозяйственной зоны",
        indicator_weights={
            "Экологическая ситуация": 1.5,
            "Население": 0.8,
            "Транспортное обеспечение": 1.2,
            "Средняя доступность до близлежащего крупного населенного пункта": 1.1,
            "default": 1.0,
        },
        investment_weights=(0.6, 0.4),
    ),
    LandUse.TRANSPORT: LandUseConfig(
        potential_column="Потенциал развития застройки транспортной зоны",
        indicator_weights={
            "Потенциал размещения логистического, складского комплекса": 1.5,
            "Количество аэропортов местного значения": 1.4,
            "Средняя доступность до близлежащего крупного населенного пункта": 1.3,
            "default": 1.0,
        },
        investment_weights=(0.35, 0.65),
    ),
}


def _as_str_keyed_mapping(
    source: Mapping[LandUse, LandUseConfig],
) -> dict[str, LandUseConfig]:
    """Utility to create a dict keyed by enum values."""
    return {land_use.value: config for land_use, config in source.items()}


_CONFIGS_BY_KEY = _as_str_keyed_mapping(LAND_USE_CONFIGS)

LAND_USE_TO_POTENTIAL_COLUMN: Final[dict[str, str]] = {
    key: config.potential_column for key, config in _CONFIGS_BY_KEY.items()
}

LAND_USE_WEIGHTS: Final[dict[str, dict[str, float]]] = {
    key: dict(config.indicator_weights) for key, config in _CONFIGS_BY_KEY.items()
}

INVESTMENT_WEIGHTS: Final[dict[str, tuple[float, float]]] = {
    key: config.investment_weights for key, config in _CONFIGS_BY_KEY.items()
}


DEFAULT_ECON_METRIC: str = "EI"
DEFAULT_DISCOUNT_RATE: float = 0.18
DEFAULT_AREA_COL: str = "Площадь территории"
DEFAULT_IP_TYPE: str = "ip_type"
DEFAULT_IP_VALUE: str = "spatial_potential"


SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "land_use",
    "land_area",
    "built_area",
    "land_value",
    "demolition_cost",
    "construction_cost",
    "investment_need",
    "NPV",
    "IRR",
    "PI",
    "PP_years",
    "EI",
)
