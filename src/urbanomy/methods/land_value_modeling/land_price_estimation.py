"""Tools for estimating land prices using spatial lag features."""

from __future__ import annotations

from typing import Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import DistanceBand, lag_spatial

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
        base = blocks[list(self._orig_features)].copy()

        for column in self._categorical_features:
            if column in base.columns:
                base[column] = base[column].astype("string").fillna("missing")

        lag_features = self._compute_lag_features(blocks)
        return pd.concat([base, lag_features], axis=1)

    def _compute_lag_features(self, blocks: gpd.GeoDataFrame) -> pd.DataFrame:
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
        missing_columns = [feature for feature in self._orig_features if feature not in self._blocks.columns]
        if missing_columns:
            raise ValueError(
                "The blocks dataset is missing required features: " + ", ".join(sorted(missing_columns))
            )
