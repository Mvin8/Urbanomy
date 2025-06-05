from __future__ import annotations
import math
from typing import Dict, Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pandera import check_types
import json

from ...utils.validation import LandUseDF
from .constants import LAND_USE_TO_POTENTIAL_COLUMN, LAND_USE_WEIGHTS


class LandUseScoreAnalyzer:
    """
    Compute spatial investment attractiveness scores for land-use types.

    Takes a GeoDataFrame with raw land-use attributes and outputs
    both wide- and long-format GeoDataFrames of investment scores.
    """

    def __init__(
        self,
        weights: dict[str, dict[str, float]] | None = None,
        weights_path: str | None = None
    ):
        """
        Initialize the analyzer.

        Parameters
        ----------
        weights : dict[str, dict[str, float]] or None
            Custom per-land-use weighting factors for attributes.
            If None, defaults to LAND_USE_WEIGHTS.
        weights_path : str or None
            Path to JSON file containing weights. Used if `weights` is None.

        Raises
        ------
        FileNotFoundError
            If `weights` is None and `weights_path` is provided but file is not found.
        ValueError
            If JSON at `weights_path` is invalid or not a dict of dicts.
        """
        if weights is not None:
            self.weights = weights
        elif weights_path:
            with open(weights_path, "r", encoding="utf-8") as f:
                self.weights = json.load(f)
        else:
            self.weights = LAND_USE_WEIGHTS
        self.land_use_to_potential: dict[str, str] = LAND_USE_TO_POTENTIAL_COLUMN

    def _compute_wide(self, polygon_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Compute wide-format investment scores (no 'ИП_' prefix).

        For each land-use key, computes a weighted average of numeric attributes
        scaled by the corresponding potential column.

        Parameters
        ----------
        polygon_gdf : geopandas.GeoDataFrame
            Input GeoDataFrame containing:
            - numeric attribute columns
            - potential columns as specified in LAND_USE_TO_POTENTIAL_COLUMN
            - geometry column

        Returns
        -------
        geopandas.GeoDataFrame
            Copy of input with additional score columns named by land-use keys,
            each containing a float score or None where not applicable.
        """
        gdf = polygon_gdf.copy()
        pot_cols = list(self.land_use_to_potential.values())
        attrs = [
            col for col in gdf.select_dtypes("number").columns
            if col not in pot_cols and ((gdf[col].between(-5, 5) & (gdf[col] != 0)).any())
        ]

        for lu, pot_col in self.land_use_to_potential.items():
            score_col = lu

            def calc(row: pd.Series) -> float | None:
                pot = row.get(pot_col)
                if pd.isna(pot):
                    return None
                vals = [
                    row[attr] * self.weights.get(lu, {}).get(attr, self.weights[lu]["default"])
                    for attr in attrs
                    if pd.notna(row[attr])
                ]
                if not vals:
                    return None
                return round(sum(vals) / len(vals) * (pot / 5), 1)

            gdf[score_col] = gdf.apply(calc, axis=1)

        return gdf

    def compute_scores_long(self, polygon_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Convert wide-format scores to long format.

        Takes the output of `_compute_wide` and melts score columns
        into `ip_type`, `ip_value`, preserving geometry.

        Parameters
        ----------
        polygon_gdf : geopandas.GeoDataFrame
            Input GeoDataFrame with wide-format score columns.

        Returns
        -------
        geopandas.GeoDataFrame
            Long-format GeoDataFrame with columns:
            - ip_type : str, land-use key
            - ip_value: float, computed score
            - geometry: Polygon geometry
        """
        wide = self._compute_wide(polygon_gdf)
        score_cols = list(self.land_use_to_potential.keys())

        df_long = (
            wide[["geometry", *score_cols]]
            .melt(
                id_vars="geometry",
                value_vars=score_cols,
                var_name="ip_type",
                value_name="ip_value"
            )
            .dropna(subset=["ip_value"])
            .reset_index(drop=True)
        )

        return gpd.GeoDataFrame(df_long, geometry="geometry", crs=wide.crs)
