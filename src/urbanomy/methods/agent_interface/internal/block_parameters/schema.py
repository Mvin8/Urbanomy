"""Constants and schema-like mappings for block-parameter requests."""

from __future__ import annotations

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

PARAMETER_REQUEST_MARKERS: dict[str, tuple[str, ...]] = {
    "id": ("идентификатор", "номер квартала"),
    "site_area": ("site_area", "площадь участка", "площадь квартала", "размер участка"),
    "land_use": ("land_use", "тип использования", "назначение", "землепользование"),
    "land_value": ("land_value", "стоимость земли", "стоимость участка", "полная стоимость"),
    "land_value_per_100m2": (
        "land_value_per_100m2",
        "стоимость за сотку",
        "за сотку",
        "100 м2",
        "100м2",
        "100 м²",
    ),
    "build_floor_area": ("build_floor_area", "поэтажная площадь", "общая площадь"),
    "footprint_area": ("footprint_area", "пятно застройки", "площадь пятна"),
    "living_area": ("living_area", "жилая площадь", "жилая"),
    "non_living_area": ("non_living_area", "нежилая площадь", "нежилая"),
    "population": ("population", "население", "жителей", "численность"),
    "fsi": ("fsi",),
    "gsi": ("gsi",),
    "mxi": ("mxi",),
}

PARAMETER_DISPLAY_NAMES: dict[str, str] = {
    "id": "Идентификатор",
    "site_area": "Площадь участка",
    "land_use": "Назначение",
    "land_value": "Стоимость земли",
    "land_value_per_100m2": "Стоимость земли за сотку",
    "build_floor_area": "Поэтажная площадь",
    "footprint_area": "Пятно застройки",
    "living_area": "Жилая площадь",
    "non_living_area": "Нежилая площадь",
    "population": "Население",
    "fsi": "FSI",
    "gsi": "GSI",
    "mxi": "MXI",
}
