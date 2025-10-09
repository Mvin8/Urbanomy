"""Shared configuration for land value modeling workflows."""

from __future__ import annotations

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


DEFAULT_ADJACENCY_RADIUS: float = 10.0
"""Default adjacency radius (metres) used to build block graphs."""


DEFAULT_SQM_PER_PERSON: float = 20.0
"""Default number of square metres per person when estimating population."""
