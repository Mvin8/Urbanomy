"""Shared configuration for land value modeling workflows."""

from __future__ import annotations

from enum import Enum
from typing import Sequence


ORIGINAL_FEATURES: Sequence[str] = (
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
)
"""Ordered list of base features used by the land price model."""


CATEGORICAL_FEATURES: Sequence[str] = ("land_use", "morphotype")
"""Subset of features that should be interpreted as categorical."""


RADIUS_LIST: Sequence[float] = (300, 500, 1000, 2000, 3000)
"""Default distance thresholds (in metres) for spatial lag computation."""


DEFAULT_OUTPUT_COLUMNS: Sequence[str] = ORIGINAL_FEATURES
"""Default set of columns produced by land data preparation."""


DEFAULT_ADJACENCY_RADIUS: int = 10
"""Default adjacency radius (metres) used to build block graphs."""


DEFAULT_SQM_PER_PERSON: float = 20.0
"""Default number of square metres per person when estimating population."""


ACCESSIBILITY_SPEED: float = 5 * 1_000 / 60
"""Walking speed (metres per minute) assumed when computing accessibility."""


class BlockColumn(str, Enum):
    """Canonical column identifiers used across land value workflows."""

    LAND_USE = "land_use"
    SHARE = "share"
    FOOTPRINT_AREA = "footprint_area"
    BUILD_FLOOR_AREA = "build_floor_area"
    LIVING_AREA = "living_area"
    NON_LIVING_AREA = "non_living_area"
    POPULATION = "population"
    SITE_AREA = "site_area"
    FSI = "fsi"
    GSI = "gsi"
    MXI = "mxi"
    L = "l"
    OSR = "osr"
    SHARE_LIVING = "share_living"
    SHARE_NON_LIVING = "share_non_living"
    RESIDENTIAL = "residential"
    IS_PROJECT = "is_project"


class ScenarioResultKey(str, Enum):
    """Named keys returned by scenario impact helpers."""

    MAP = "map"
    MAP_ALL = "map_all"
    FIGURE = "fig"
    SUMMARY = "summary"
    SUMMARY_ALL = "summary_all"
