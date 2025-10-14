"""Configuration defaults for SER estimation routines."""

from __future__ import annotations

from typing import Final, Mapping

from blocksnet.enums import LandUse


def _value_map(
    values: Mapping[LandUse, float],
    default: float,
) -> dict[str, float]:
    """Return a string-keyed mapping backed by LandUse values."""
    mapping = {lu.value: float(val) for lu, val in values.items()}
    mapping["default"] = float(default)
    return mapping


DEFAULT_SER_PARAMETERS: Final[dict[str, object]] = {
    "build_years": 3,
    "avg_wage_base": 70_430,
    "tax_rates": {"pit": 0.13, "cit": 0.17, "prop": 0.02, "land": 0.015},
    "va_coeff_build": _value_map(
        {
            LandUse.BUSINESS: 0.50,
            LandUse.RESIDENTIAL: 0.45,
            LandUse.TRANSPORT: 0.55,
            LandUse.AGRICULTURE: 0.45,
            LandUse.SPECIAL: 0.50,
        },
        default=0.50,
    ),
    "va_per_m2_ops": _value_map(
        {
            LandUse.BUSINESS: 12_000.0,
            LandUse.RESIDENTIAL: 2_000.0,
            LandUse.TRANSPORT: 5_000.0,
            LandUse.AGRICULTURE: 3_000.0,
            LandUse.SPECIAL: 7_000.0,
        },
        default=0.0,
    ),
    "jobs_per_m2": _value_map(
        {
            LandUse.BUSINESS: 1 / 18,
            LandUse.RESIDENTIAL: 0.0,
            LandUse.TRANSPORT: 1 / 90,
            LandUse.AGRICULTURE: 1 / 400,
            LandUse.SPECIAL: 1 / 30,
        },
        default=0.0,
    ),
    "wage_by_use": _value_map(
        {
            LandUse.BUSINESS: 85_000,
            LandUse.RESIDENTIAL: 0,
            LandUse.TRANSPORT: 60_000,
            LandUse.AGRICULTURE: 45_000,
            LandUse.SPECIAL: 75_000,
        },
        default=0,
    ),
    "profit_share_ops": _value_map(
        {
            LandUse.BUSINESS: 0.18,
            LandUse.TRANSPORT: 0.10,
            LandUse.AGRICULTURE: 0.08,
            LandUse.SPECIAL: 0.15,
            LandUse.RESIDENTIAL: 0.00,
        },
        default=0.12,
    ),
    "capex_capitalizable_share": _value_map(
        {
            LandUse.BUSINESS: 0.95,
            LandUse.RESIDENTIAL: 0.95,
            LandUse.TRANSPORT: 0.90,
            LandUse.AGRICULTURE: 0.90,
            LandUse.SPECIAL: 0.95,
        },
        default=0.95,
    ),
    "amortization_rates": _value_map(
        {
            LandUse.BUSINESS: 0.03,
            LandUse.RESIDENTIAL: 0.03,
            LandUse.TRANSPORT: 0.06,
            LandUse.AGRICULTURE: 0.05,
            LandUse.SPECIAL: 0.04,
        },
        default=0.03,
    ),
    "build_wage_share": 0.25,
    "build_profit_margin": 0.05,
}
