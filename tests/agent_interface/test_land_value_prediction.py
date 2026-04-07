import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from urbanomy.methods.agent_interface.models import LandValuePredictionConfig
from urbanomy.methods.agent_interface.tools.internal.land_value_prediction import (
    ensure_land_value_predictions,
)


def _make_blocks() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "residential": [1.0, 2.0],
            "land_use": ["residential", "business"],
            "morphotype": ["midrise", "mixed"],
        },
        geometry=[
            Polygon([(30.0, 59.0), (30.001, 59.0), (30.001, 59.001), (30.0, 59.001)]),
            Polygon([(30.002, 59.0), (30.003, 59.0), (30.003, 59.001), (30.002, 59.001)]),
        ],
        crs=4326,
    )


def test_ensure_land_value_predictions_mutates_blocks_once(monkeypatch):
    blocks = _make_blocks()
    calls = {"init": 0, "predict": 0}

    class _StubEstimator:
        def __init__(self, *, blocks, **kwargs):
            calls["init"] += 1
            self._blocks = blocks.copy()

        def predict_prices(self, *, total_price_column, **kwargs):
            calls["predict"] += 1
            result = self._blocks.copy()
            result[total_price_column] = [100.0, 250.0]
            return result

    monkeypatch.setattr(
        "urbanomy.methods.agent_interface.tools.internal.land_value_prediction.LandPriceEstimator",
        _StubEstimator,
    )

    config = LandValuePredictionConfig(
        model=object(),
        orig_features=["residential", "land_use", "morphotype"],
        categorical_features=["land_use", "morphotype"],
    )

    first = ensure_land_value_predictions(
        baseline_blocks=blocks,
        prediction_config=config,
    )
    second = ensure_land_value_predictions(
        baseline_blocks=blocks,
        prediction_config=config,
    )

    assert first["used_cache"] is False
    assert second["used_cache"] is True
    assert calls == {"init": 1, "predict": 1}
    assert "id" in blocks.columns
    assert "site_area" in blocks.columns
    assert "land_value" in blocks.columns
    assert "land_value_per_100m2" in blocks.columns
    assert blocks["land_value"].tolist() == [100.0, 250.0]
    assert pytest.approx(
        blocks.loc[0, "land_value_per_100m2"],
        rel=1e-6,
    ) == blocks.loc[0, "land_value"] / blocks.loc[0, "site_area"] * 100.0


def test_ensure_land_value_predictions_cleans_nan_inf_and_clips_outliers(monkeypatch):
    blocks = _make_blocks()

    class _StubEstimator:
        def __init__(self, *, blocks, **kwargs):
            self._blocks = blocks.copy()

        def predict_prices(self, *, total_price_column, **kwargs):
            result = self._blocks.copy()
            result["site_area"] = [100.0, 0.0]
            result[total_price_column] = [100.0, float("inf")]
            return result

    monkeypatch.setattr(
        "urbanomy.methods.agent_interface.tools.internal.land_value_prediction.LandPriceEstimator",
        _StubEstimator,
    )

    config = LandValuePredictionConfig(
        model=object(),
        orig_features=["residential", "land_use", "morphotype"],
        categorical_features=["land_use", "morphotype"],
    )

    payload = ensure_land_value_predictions(
        baseline_blocks=blocks,
        prediction_config=config,
    )

    assert blocks["land_value"].replace([float("inf"), float("-inf")], 0).notna().all()
    assert blocks["land_value_per_100m2"].replace([float("inf"), float("-inf")], 0).notna().all()
    assert (blocks["land_value_per_100m2"] <= payload["unit_price_upper_quantile"]).all()
