"""Glossary helpers for Urbanomy block parameters."""

from __future__ import annotations

from typing import Any

from ..common.request_parsing import extract_target_id


BLOCK_PARAMETER_DEFINITIONS: dict[str, dict[str, Any]] = {
    "fsi": {
        "label": "FSI",
        "definition": "Floor Space Index: отношение суммарной поэтажной площади к площади участка.",
        "formula": "fsi = build_floor_area / site_area",
        "interpretation": "Показывает интенсивность использования территории.",
    },
    "gsi": {
        "label": "GSI",
        "definition": "Ground Space Index: отношение пятна застройки к площади участка.",
        "formula": "gsi = footprint_area / site_area",
        "interpretation": "Показывает, какая доля участка занята застройкой по земле.",
    },
    "mxi": {
        "label": "MXI",
        "definition": "Mixed-use Index: доля жилой площади в общей поэтажной площади.",
        "formula": "mxi = living_area / build_floor_area",
        "interpretation": "Показывает баланс между жилой и нежилой функцией.",
    },
    "l": {
        "label": "L",
        "definition": "Количество этажей или коэффициент этажности в параметризации сценария.",
        "formula": "build_floor_area = footprint_area * l",
        "interpretation": "Управляет высотностью/этажностью в оптимизации.",
    },
    "site_area": {
        "label": "Площадь участка",
        "definition": "Площадь квартала или участка, на которой считаются остальные показатели.",
        "formula": "site_area = площадь территории квартала",
        "interpretation": "Базовый нормирующий параметр для FSI и GSI.",
    },
    "footprint_area": {
        "label": "Пятно застройки",
        "definition": "Площадь застройки по земле.",
        "formula": "footprint_area = площадь основания зданий",
        "interpretation": "Используется для расчёта GSI и build_floor_area.",
    },
    "build_floor_area": {
        "label": "Поэтажная площадь",
        "definition": "Суммарная площадь всех этажей зданий.",
        "formula": "build_floor_area = footprint_area * l",
        "interpretation": "Используется для расчёта FSI и MXI.",
    },
    "living_area": {
        "label": "Жилая площадь",
        "definition": "Часть поэтажной площади, относящаяся к жилой функции.",
        "formula": "living_area = build_floor_area * mxi",
        "interpretation": "Вместе с non_living_area определяет функциональный баланс квартала.",
    },
    "non_living_area": {
        "label": "Нежилая площадь",
        "definition": "Часть поэтажной площади, относящаяся к нежилым функциям.",
        "formula": "non_living_area = build_floor_area - living_area",
        "interpretation": "Дополняет жилую площадь до общей поэтажной площади.",
    },
    "population": {
        "label": "Население",
        "definition": "Оценка численности населения квартала.",
        "formula": "population = living_area / sqm_per_person",
        "interpretation": "Производный показатель, пересчитывается из жилой площади.",
    },
}

BLOCK_PARAMETER_DEFINITION_MARKERS = (
    "что такое",
    "что значит",
    "что означает",
    "объясни",
    "поясни",
    "расшифруй",
    "в чем смысл",
    "смысл",
    "зачем нужен",
    "зачем нужна",
    "для чего нужен",
    "для чего нужна",
)


def detect_block_parameter_term(user_request: str) -> str | None:
    """Detect a known block-parameter term in a free-form request."""
    text = str(user_request).lower().replace("ё", "е")
    aliases = {
        "fsi": ("fsi",),
        "gsi": ("gsi",),
        "mxi": ("mxi",),
        "l": (" параметр l", " коэффициент l", " l ", " l?", " l."),
        "site_area": ("site_area", "площадь участка", "площадь квартала"),
        "footprint_area": ("footprint_area", "пятно застройки"),
        "build_floor_area": ("build_floor_area", "поэтажная площадь", "общая площадь застройки"),
        "living_area": ("living_area", "жилая площадь"),
        "non_living_area": ("non_living_area", "нежилая площадь"),
        "population": ("population", "население", "численность"),
    }
    for term, markers in aliases.items():
        if any(marker in text for marker in markers):
            return term
    return None


def looks_like_block_parameter_definition_request(user_request: str) -> bool:
    """Return whether the request asks to explain a parameter term, not a block value."""
    if extract_target_id(user_request) is not None:
        return False
    text = str(user_request).lower().replace("ё", "е")
    if detect_block_parameter_term(text) is None:
        return False
    return any(marker in text for marker in BLOCK_PARAMETER_DEFINITION_MARKERS)


def build_block_parameter_definition_text(term: str) -> str:
    """Render a user-facing glossary answer for one parameter term."""
    item = BLOCK_PARAMETER_DEFINITIONS[term]
    lines = [
        f"{item['label']}: {item['definition']}",
        f"Формула: {item['formula']}",
        f"Смысл: {item['interpretation']}",
    ]
    return "\n".join(lines)
