"""Pure helper logic for block-parameter selection and formatting."""

from __future__ import annotations

from typing import Any

from ..common.request_parsing import normalize_text
from ...tools.internal.district_optimization_formatting import format_float, json_value
from .schema import (
    PARAMETER_DISPLAY_NAMES,
    PARAMETER_REQUEST_MARKERS,
    PREFERRED_BLOCK_PARAMETER_ORDER,
)


def extract_requested_parameters(user_request: str) -> list[str]:
    """Return parameter column names explicitly mentioned in the request."""
    text = normalize_text(user_request)
    requested: list[str] = []
    for column, markers in PARAMETER_REQUEST_MARKERS.items():
        if any(marker in text for marker in markers):
            requested.append(column)
    return requested


def build_block_parameters_response(*, target_id: int, parameters: dict[str, Any]) -> str:
    """Render a compact user-facing response from selected block parameters."""
    if len(parameters) == 1:
        key, value = next(iter(parameters.items()))
        value_text = format_float(value) if isinstance(value, float) else str(value)
        label = PARAMETER_DISPLAY_NAMES.get(key, key)
        return f"{label} квартала id={target_id}: {value_text}"
    lines = [f"Параметры квартала id={target_id}:"]
    for key, value in parameters.items():
        value_text = format_float(value) if isinstance(value, float) else str(value)
        lines.append(f" • {key}: {value_text}")
    return "\n".join(lines)


def serialize_block_row(row: Any) -> dict[str, Any]:
    """Serialize one GeoDataFrame row into a stable ordered parameter dict."""
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
