"""Tool for estimating baseline land value and caching the result."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
from langchain_core.tools import tool

from ..models import LandValuePredictionConfig
from .internal.land_value_prediction import ensure_land_value_predictions


def make_predict_land_value_tool(
    *,
    baseline_blocks: gpd.GeoDataFrame,
    prediction_config: LandValuePredictionConfig,
):
    """Create a LangChain tool bound to one baseline GeoDataFrame."""

    @tool("predict_land_value")
    def predict_land_value() -> dict[str, Any]:
        """What the tool does.

        Estimates baseline land value for all blocks using the configured
        regression model and writes the result back into ``baseline_blocks``.

        When to use this tool.

        Use this tool when the user asks to calculate, predict, estimate, or
        refresh land value before further analysis or visualization.

        Args:
            None.

        Returns:
            dict[str, Any]: Prediction status, cache usage, output column names,
            and number of updated rows.

        Restrictions:
            Works on the full baseline dataset and overwrites cached land-value
            columns in the same GeoDataFrame object only once per runtime unless
            the prediction configuration changes.
        """
        return ensure_land_value_predictions(
            baseline_blocks=baseline_blocks,
            prediction_config=prediction_config,
        )

    return predict_land_value
