"""Shared runtime helpers for baseline land-value prediction."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from ....land_value_modeling import LandPriceEstimator
from ...models import LandValuePredictionConfig

_CACHE_ATTR = "_urbanomy_land_value_prediction"


def ensure_land_value_predictions(
    *,
    baseline_blocks: gpd.GeoDataFrame,
    prediction_config: LandValuePredictionConfig,
) -> dict[str, Any]:
    """Estimate land value once and write predictions back into baseline blocks."""
    if not isinstance(baseline_blocks, gpd.GeoDataFrame):
        raise TypeError("baseline_blocks must be a geopandas.GeoDataFrame.")
    if baseline_blocks.empty:
        raise ValueError("baseline_blocks is empty. Nothing to estimate.")
    if baseline_blocks.geometry is None:
        raise ValueError("baseline_blocks must provide a geometry column.")

    total_column = prediction_config.total_price_column
    unit_column = prediction_config.unit_price_column
    area_column = prediction_config.area_column
    id_column = prediction_config.id_column

    if _has_valid_cache(baseline_blocks=baseline_blocks, prediction_config=prediction_config):
        return {
            "status": "ok",
            "tool_name": "predict_land_value",
            "used_cache": True,
            "rows_updated": 0,
            "id_column": id_column,
            "area_column": area_column,
            "total_price_column": total_column,
            "unit_price_column": unit_column,
            "n_rows": int(len(baseline_blocks)),
        }

    working = baseline_blocks.copy()
    existing_columns = set(baseline_blocks.columns)
    created_columns: list[str] = []

    if id_column not in working.columns:
        working[id_column] = working.index
        baseline_blocks.loc[:, id_column] = working[id_column].to_numpy()
        created_columns.append(id_column)

    area_stats = _ensure_area_column(
        working=working,
        baseline_blocks=baseline_blocks,
        area_column=area_column,
    )
    if area_stats["created"]:
        created_columns.append(area_column)

    for feature in prediction_config.orig_features:
        if feature in prediction_config.categorical_features or feature not in working.columns:
            continue
        working[feature] = pd.to_numeric(working[feature], errors="coerce")

    estimator = LandPriceEstimator(
        model=prediction_config.model,
        blocks=working,
        radius_list=prediction_config.radius_list,
        orig_features=prediction_config.orig_features,
        categorical_features=prediction_config.categorical_features,
        use_service_features=bool(prediction_config.use_service_features),
        service_features=prediction_config.service_features,
    )
    predicted = estimator.predict_prices(
        total_price_column=total_column,
        include_unit_price=False,
        predictions_in_log_scale=prediction_config.predictions_in_log_scale,
    )

    cleaned = pd.DataFrame(
        {
            area_column: pd.to_numeric(predicted[area_column], errors="coerce"),
            total_column: pd.to_numeric(predicted[total_column], errors="coerce"),
        },
        index=predicted.index,
    )
    cleaned[unit_column] = cleaned[total_column] / cleaned[area_column] * 100.0
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if len(cleaned.index) > 0:
        upper_quantile = float(pd.to_numeric(cleaned[unit_column], errors="coerce").quantile(0.95))
    else:
        upper_quantile = 0.0
    if not np.isfinite(upper_quantile):
        upper_quantile = 0.0
    outlier_mask = cleaned[unit_column] > upper_quantile
    if bool(outlier_mask.any()):
        cleaned.loc[outlier_mask, unit_column] = upper_quantile
        cleaned.loc[outlier_mask, total_column] = (
            cleaned.loc[outlier_mask, unit_column] * cleaned.loc[outlier_mask, area_column] / 100.0
        )

    area_values = pd.to_numeric(cleaned[area_column], errors="coerce")
    total_values = pd.to_numeric(cleaned[total_column], errors="coerce")
    unit_values = pd.to_numeric(cleaned[unit_column], errors="coerce")

    baseline_blocks.loc[:, area_column] = area_values.to_numpy()
    baseline_blocks.loc[:, total_column] = total_values.to_numpy()
    baseline_blocks.loc[:, unit_column] = unit_values.to_numpy()

    if total_column not in existing_columns and total_column not in created_columns:
        created_columns.append(total_column)
    if unit_column not in existing_columns and unit_column not in created_columns:
        created_columns.append(unit_column)

    baseline_blocks.attrs[_CACHE_ATTR] = {
        "signature": _config_signature(prediction_config),
        "row_count": int(len(baseline_blocks)),
        "total_price_column": total_column,
        "unit_price_column": unit_column,
        "area_column": area_column,
        "id_column": id_column,
    }

    return {
        "status": "ok",
        "tool_name": "predict_land_value",
        "used_cache": False,
        "rows_updated": int(len(baseline_blocks)),
        "id_column": id_column,
        "area_column": area_column,
        "total_price_column": total_column,
        "unit_price_column": unit_column,
        "n_rows": int(len(baseline_blocks)),
        "created_columns": created_columns,
        "area_recomputed_rows": int(area_stats["recomputed_rows"]),
        "unit_price_upper_quantile": upper_quantile,
        "outliers_clipped_rows": int(outlier_mask.sum()),
    }


def _has_valid_cache(
    *,
    baseline_blocks: gpd.GeoDataFrame,
    prediction_config: LandValuePredictionConfig,
) -> bool:
    cache = baseline_blocks.attrs.get(_CACHE_ATTR)
    if not isinstance(cache, dict):
        return False
    if cache.get("signature") != _config_signature(prediction_config):
        return False
    if int(cache.get("row_count", -1)) != int(len(baseline_blocks)):
        return False

    required_columns = (
        prediction_config.area_column,
        prediction_config.total_price_column,
        prediction_config.unit_price_column,
    )
    for column in required_columns:
        if column not in baseline_blocks.columns:
            return False
        if pd.to_numeric(baseline_blocks[column], errors="coerce").notna().sum() == 0:
            return False
    return True


def _config_signature(prediction_config: LandValuePredictionConfig) -> tuple[Any, ...]:
    return (
        id(prediction_config.model),
        tuple(prediction_config.orig_features),
        tuple(prediction_config.categorical_features),
        tuple(prediction_config.radius_list or ()),
        bool(prediction_config.use_service_features),
        tuple(prediction_config.service_features or ()),
        prediction_config.id_column,
        prediction_config.area_column,
        prediction_config.total_price_column,
        prediction_config.unit_price_column,
        bool(prediction_config.predictions_in_log_scale),
    )


def _ensure_area_column(
    *,
    working: gpd.GeoDataFrame,
    baseline_blocks: gpd.GeoDataFrame,
    area_column: str,
) -> dict[str, Any]:
    created = area_column not in working.columns
    if created:
        area_values = np.full(len(working), np.nan, dtype=float)
    else:
        area_values = pd.to_numeric(working[area_column], errors="coerce").to_numpy(copy=True)

    invalid_mask = ~np.isfinite(area_values) | (area_values <= 0)
    recomputed_rows = int(invalid_mask.sum())
    if invalid_mask.any():
        geometry_area = _metric_geometry_area(working)
        area_values[invalid_mask] = geometry_area[invalid_mask]

    working[area_column] = area_values
    baseline_blocks.loc[:, area_column] = area_values
    return {
        "created": created,
        "recomputed_rows": recomputed_rows,
    }


def _metric_geometry_area(blocks: gpd.GeoDataFrame) -> np.ndarray:
    try:
        utm_crs = blocks.estimate_utm_crs()
    except Exception:
        utm_crs = None
    if utm_crs is not None:
        try:
            return blocks.to_crs(utm_crs).geometry.area.to_numpy()
        except Exception:
            pass
    return blocks.geometry.area.to_numpy()
