"""Tools for estimating land prices using spatial lag features."""

from __future__ import annotations

from typing import Mapping, Sequence, Any

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import DistanceBand, lag_spatial

from blocksnet.enums import LandUse

from .constants import CATEGORICAL_FEATURES, ORIGINAL_FEATURES, RADIUS_LIST


class LandPriceEstimator:
    """Estimate land prices for blocks using a pretrained regression model.

    Parameters
    ----------
    model : object
        Fitted regression model exposing a ``predict`` method that accepts a
        pandas DataFrame and returns price predictions in logarithmic scale.
    blocks : geopandas.GeoDataFrame
        Blocks dataset containing the geometry and all required base features.
    radius_list : Sequence[float], optional
        Distance thresholds used to build spatial weights. Defaults to
        ``(300, 500, 1000, 2000, 3000)``.
    orig_features : Sequence[str], optional
        Names of the base features that will be passed to the model. Defaults to
        the feature list used in the original notebook workflow.
    categorical_features : Sequence[str], optional
        Subset of ``orig_features`` that should be treated as categorical.
    """

    DEFAULT_FEATURES: Sequence[str] = ORIGINAL_FEATURES

    DEFAULT_CATEGORICAL: Sequence[str] = CATEGORICAL_FEATURES

    DEFAULT_RADII: Sequence[float] = RADIUS_LIST

    def __init__(
        self,
        *,
        model,
        blocks: gpd.GeoDataFrame,
        radius_list: Sequence[float] | None = None,
        orig_features: Sequence[str] | None = None,
        categorical_features: Sequence[str] | None = None,
    ) -> None:
        """Initialise the estimator with model, data and feature configuration.

        Parameters
        ----------
        model : object
            Trained estimator exposing a ``predict`` method that accepts a
            pandas DataFrame and returns logarithmic prices.
        blocks : geopandas.GeoDataFrame
            Dataset containing geometries and base features required by
            ``orig_features``.
        radius_list : Sequence[float], optional
            Distance thresholds for spatial lag computation.
        orig_features : Sequence[str], optional
            Feature names supplied to the estimator.
        categorical_features : Sequence[str], optional
            Subset of features that should be treated as categorical.
        """
        self.model = model
        self._blocks = blocks.copy()
        self._radii = tuple(radius_list) if radius_list is not None else tuple(self.DEFAULT_RADII)
        self._orig_features = tuple(orig_features) if orig_features is not None else tuple(self.DEFAULT_FEATURES)
        self._categorical_features = (
            tuple(categorical_features)
            if categorical_features is not None
            else tuple(self.DEFAULT_CATEGORICAL)
        )
        self._numeric_features = tuple(
            feature for feature in self._orig_features if feature not in self._categorical_features
        )
        self._validate_inputs()
        self._weights = self._build_distance_weights()

    def predict(self) -> gpd.GeoDataFrame:
        """Generate price predictions for each block.

        Returns
        -------
        geopandas.GeoDataFrame
            Copy of the original blocks with two additional columns:
            ``y_log_pred`` (logarithmic price) and ``price_pred`` (price in the
            original scale).
        """

        design_matrix = self._design_matrix(self._blocks)
        y_log = self.model.predict(design_matrix)

        blocks_pred = self._blocks.copy()
        blocks_pred["y_log_pred"] = y_log
        blocks_pred["price_pred"] = np.exp(y_log)
        return blocks_pred

    def _build_distance_weights(self) -> Mapping[float, DistanceBand]:
        """Construct distance-band spatial weights for each configured radius.

        Returns
        -------
        Mapping[float, libpysal.weights.DistanceBand]
            Dictionary mapping radius values to ``DistanceBand`` instances.
        """
        weights = {}
        for radius in self._radii:
            weight = DistanceBand.from_dataframe(
                self._blocks,
                threshold=radius,
                binary=True,
                silence_warnings=True,
            )
            weight.transform = "r"
            weights[radius] = weight
        return weights

    def _design_matrix(self, blocks: gpd.GeoDataFrame) -> pd.DataFrame:
        """Assemble the model design matrix including spatial lag features.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Dataset for which predictions will be generated.

        Returns
        -------
        pandas.DataFrame
            Feature matrix aligned with ``blocks.index``.
        """
        base = blocks[list(self._orig_features)].copy()

        for column in self._categorical_features:
            if column in base.columns:
                base[column] = (
                    base[column]
                    .apply(self._categorical_token)
                    .astype("string")
                )

        lag_features = self._compute_lag_features(blocks)
        return pd.concat([base, lag_features], axis=1)
    
    @staticmethod
    def _categorical_token(value: Any) -> str:
        """Convert categorical values to stable tokens for model input."""
        if value is None:
            return "missing"
        if isinstance(value, float) and np.isnan(value):
            return "missing"
        if isinstance(value, LandUse):
            return value.name

        text = str(value).strip()
        if not text or text.lower() == "nan":
            return "missing"

        try:
            return LandUse(text).name
        except ValueError:
            pass

        upper = text.upper()
        if upper.startswith("LANDUSE."):
            upper = upper.split(".", 1)[1]
        if upper in LandUse.__members__:
            return upper
        return upper

    def _compute_lag_features(self, blocks: gpd.GeoDataFrame) -> pd.DataFrame:
        """Compute spatial lag features for numeric columns and neighbour counts.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Dataset whose numeric columns are used to compute lags.

        Returns
        -------
        pandas.DataFrame
            Lag feature frame indexed by ``blocks.index``.
        """
        lag_parts: list[pd.Series] = []

        if self._numeric_features:
            numeric = blocks[list(self._numeric_features)]
            global_mean = numeric.mean(numeric_only=True)

            for feature in self._numeric_features:
                filled = numeric[feature].fillna(global_mean.get(feature, 0.0))
                for radius, weight in self._weights.items():
                    series = pd.Series(
                        lag_spatial(weight, filled),
                        index=blocks.index,
                        name=f"lag{radius}_{feature}",
                    )
                    lag_parts.append(series)

        for radius, weight in self._weights.items():
            neighbors = pd.Series(
                {index: len(neigh) for index, neigh in weight.neighbors.items()},
                name=f"n_neighbors_{radius}",
            )
            lag_parts.append(neighbors.reindex(blocks.index).fillna(0))

        if not lag_parts:
            return pd.DataFrame(index=blocks.index)
        return pd.concat(lag_parts, axis=1)

    def _validate_inputs(self) -> None:
        """Ensure that all required features are present in the blocks dataset.

        Raises
        ------
        ValueError
            If any of ``orig_features`` are missing from ``self._blocks``.
        """
        missing_columns = [feature for feature in self._orig_features if feature not in self._blocks.columns]
        if missing_columns:
            raise ValueError(
                "The blocks dataset is missing required features: " + ", ".join(sorted(missing_columns))
            )


def transfer_baseline_prices(
    after_blocks: gpd.GeoDataFrame,
    before_blocks: gpd.GeoDataFrame,
    *,
    id_column: str = "id",
    price_column: str = "price_pred",
    scenario_column: str = "is_scn",
    output_column: str = "price_pred_before",
    area_column: str = "site_area",
) -> gpd.GeoDataFrame:
    """Project baseline land prices from historical blocks onto scenario blocks.

    The function computes a weighted average unit price for each scenario block
    based on the proportional overlap with historical blocks. The resulting
    baseline price is added as a new column to a copy of ``after_blocks``.

    Parameters
    ----------
    after_blocks : geopandas.GeoDataFrame
        Blocks describing the post-development scenario. Must contain geometry,
        ``id_column`` and ``scenario_column``.
    before_blocks : geopandas.GeoDataFrame
        Baseline blocks with historical prices. Must contain geometry,
        ``price_column`` and ``scenario_column``.
    id_column : str, optional
        Unique polygon identifier present in ``after_blocks``. Defaults to ``"id"``.
    price_column : str, optional
        Column in ``before_blocks`` containing baseline total prices. Defaults to ``"price_pred"``.
    scenario_column : str, optional
        Boolean column indicating scenario polygons. Defaults to ``"is_scn"``.
    output_column : str, optional
        Name of the column to store the transferred baseline price inside the
        returned GeoDataFrame. Defaults to ``"price_pred_before"``.
    area_column : str, optional
        Column describing block areas in square metres. When missing or
        containing non-positive values, geometry-derived areas are used.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of ``after_blocks`` with an additional ``output_column`` describing
        the weighted baseline price for each polygon.

    Raises
    ------
    KeyError
        If required columns are missing from the input GeoDataFrames.
    """
    for frame_name, frame, required in (
        ("after_blocks", after_blocks, {id_column, scenario_column}),
        ("before_blocks", before_blocks, {id_column, price_column, scenario_column}),
    ):
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise KeyError(
                f"{frame_name} is missing required columns: {', '.join(sorted(missing))}"
            )
        if frame.geometry is None:
            raise KeyError(f"{frame_name} must provide a geometry column")

    after_geom_col = after_blocks.geometry.name
    before_geom_col = before_blocks.geometry.name

    after_mask = after_blocks[scenario_column].fillna(False).astype(bool)
    before_mask = before_blocks[scenario_column].fillna(False).astype(bool)

    result = after_blocks.copy()
    baseline_column = "_baseline_price"
    baseline_mapping = (
        before_blocks[[id_column, price_column]]
        .drop_duplicates(subset=id_column)
        .rename(columns={price_column: baseline_column})
    )

    def _resolve_area(df: gpd.GeoDataFrame) -> np.ndarray:
        if area_column in df.columns:
            area_series = pd.to_numeric(df[area_column], errors="coerce")
        else:
            area_series = pd.Series(np.nan, index=df.index, dtype=float)
        area_values = area_series.to_numpy(copy=True)
        invalid_mask = ~np.isfinite(area_values) | (area_values <= 0)
        if invalid_mask.any():
            geom_area = df.geometry.area.to_numpy()
            area_values[invalid_mask] = geom_area[invalid_mask]
        return area_values

    def _apply_baseline(df: pd.DataFrame) -> pd.DataFrame:
        merged = df.merge(baseline_mapping, on=id_column, how="left")
        if output_column not in merged.columns:
            merged[output_column] = np.nan
        non_scenario_mask = ~merged[scenario_column].fillna(False).astype(bool)
        merged.loc[non_scenario_mask, output_column] = merged.loc[non_scenario_mask, output_column].fillna(
            merged.loc[non_scenario_mask, baseline_column]
        )
        return merged.drop(columns=[baseline_column])

    after_scn = result.loc[after_mask].copy()
    before_scn = before_blocks.loc[before_mask].copy()

    if after_scn.empty or before_scn.empty:
        return gpd.GeoDataFrame(
            _apply_baseline(result),
            geometry=after_geom_col,
            crs=after_blocks.crs,
        )

    before_scn["area_before"] = before_scn.geometry.area
    before_scn["unit_price_before"] = np.where(
        before_scn["area_before"] > 0,
        before_scn[price_column] / before_scn["area_before"],
        np.nan,
    )

    intersections = gpd.overlay(
        after_scn[[id_column, after_geom_col]],
        before_scn[["unit_price_before", before_geom_col]],
        how="intersection",
        keep_geom_type=False,
    )

    if intersections.empty:
        return gpd.GeoDataFrame(
            _apply_baseline(result),
            geometry=after_geom_col,
            crs=after_blocks.crs,
        )

    intersections["intersect_area"] = intersections.geometry.area
    area_sum = (
        intersections.groupby(id_column, as_index=False)["intersect_area"]
        .sum()
        .rename(columns={"intersect_area": "id_total_area"})
    )
    intersections = intersections.merge(area_sum, on=id_column, how="left")

    intersections["weight"] = np.divide(
        intersections["intersect_area"],
        intersections["id_total_area"],
        out=np.zeros_like(intersections["intersect_area"]),
        where=intersections["id_total_area"] > 0,
    )
    intersections["contrib"] = intersections["unit_price_before"] * intersections["weight"]

    price_transfer = (
        intersections.groupby(id_column, as_index=False)["contrib"]
        .sum()
        .rename(columns={"contrib": "unit_price_before_weighted"})
    )

    after_scn = after_scn.merge(price_transfer, on=id_column, how="left")
    scenario_area = _resolve_area(after_scn)
    after_scn[output_column] = after_scn["unit_price_before_weighted"] * scenario_area
    merged = result.merge(after_scn[[id_column, output_column]], on=id_column, how="left")

    return gpd.GeoDataFrame(
        _apply_baseline(merged),
        geometry=after_geom_col,
        crs=after_blocks.crs,
    )
