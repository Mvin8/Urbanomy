"""Formatting and JSON-safe serialization helpers for district optimization."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a dataframe to JSON-safe row records."""
    records: list[dict[str, Any]] = []
    for _, row in df.reset_index(drop=True).iterrows():
        records.append({str(col): json_value(value) for col, value in row.items()})
    return records


def json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an arbitrary mapping to a JSON-safe dict."""
    return {str(key): json_value(item) for key, item in value.items()}


def json_value(value: Any) -> Any:
    """Convert a scalar or container to a JSON-safe value."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.floating, np.integer)):
        return json_value(value.item())
    if isinstance(value, Mapping):
        return json_mapping(value)
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return str(value)


def clean_text_block(text: str) -> str:
    """Strip empty lines and trailing whitespace from a text block."""
    lines = [line.rstrip() for line in str(text).splitlines() if line.strip()]
    return "\n".join(lines)


def build_pareto_front_text(
    *,
    n_points: int,
    n_pareto_points: int,
    land_use_labels: list[str],
) -> str:
    """Build a short textual summary for a Pareto-front plot."""
    labels = ", ".join(land_use_labels) if land_use_labels else "н/д"
    return "\n".join(
        [
            "Построен график Парето-фронта.",
            f" • Всего точек: {n_points}",
            f" • Точек на фронте: {n_pareto_points}",
            f" • Категории LANDUSE: {labels}",
        ]
    )


def build_plot_solution_summary_text(
    *,
    sum_before: float,
    sum_after: float,
    sum_delta: float,
    sum_delta_pct: float,
    target_before: float,
    target_after: float,
    target_delta_rub: float,
    target_delta_pct: float,
) -> str:
    """Build the textual summary shown after solution visualization."""
    return "\n".join(
        [
            "Изменение стоимости земли по всем кварталам:",
            f" • Сумма до:    {format_rub(sum_before)}",
            f" • Сумма после: {format_rub(sum_after)}",
            f" • Изм., ₽:     {format_signed_rub(sum_delta)}",
            f" • Изм., %:     {format_pct(sum_delta_pct, signed=True)}",
            "",
            "Изменяемый квартал:",
            (
                f" • До: {format_rub(target_before)} | "
                f"После: {format_rub(target_after)} | "
                f"Δ: {format_signed_rub(target_delta_rub)} | "
                f"Δ%: {format_pct(target_delta_pct, signed=True)}"
            ),
        ]
    )


def build_params_text(*, solution_number: int, params_repaired: Mapping[str, Any]) -> str:
    """Build a textual summary for repaired solution parameters."""
    lines = [f"Параметры квартала для решения {solution_number}:"]
    for key, value in params_repaired.items():
        if isinstance(value, float):
            value_text = format_float(value)
        else:
            value_text = str(value)
        lines.append(f" • {key}: {value_text}")
    return "\n".join(lines)


def format_rub(value: float) -> str:
    """Format ruble values for user-facing output."""
    if not np.isfinite(value):
        return "н/д"
    return f"{value:,.0f} ₽".replace(",", " ")


def format_signed_rub(value: float) -> str:
    """Format signed ruble values for user-facing output."""
    if not np.isfinite(value):
        return "н/д"
    return f"{value:+,.0f} ₽".replace(",", " ")


def format_pct(value: float, *, signed: bool) -> str:
    """Format percentage values for user-facing output."""
    if not np.isfinite(value):
        return "н/д"
    pattern = "{:+.2f}%" if signed else "{:.2f}%"
    return pattern.format(value)


def format_float(value: float) -> str:
    """Format a generic float compactly."""
    if not np.isfinite(value):
        return "н/д"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def to_float(value: Any) -> float:
    """Coerce a value to float or return nan."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan
