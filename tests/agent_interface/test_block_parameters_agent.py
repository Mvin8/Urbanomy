import geopandas as gpd
from shapely.geometry import Point

from urbanomy.methods.agent_interface.block_parameters_agent import (
    BlockParametersAgent,
    looks_like_block_parameters_request,
)


def test_block_parameters_agent_returns_parameters_for_target_id():
    baseline_blocks = gpd.GeoDataFrame(
        [
            {
                "id": 20,
                "site_area": 1500.0,
                "land_use": "residential",
                "land_value": 1250000.0,
                "population": 180,
                "custom_flag": True,
                "geometry": Point(30.0, 60.0),
            }
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    agent = BlockParametersAgent(baseline_blocks=baseline_blocks)

    result = agent.invoke("Какие параметры квартала 20?")

    assert result["status"] == "ok"
    assert result["target_id"] == 20
    assert result["parameters"]["id"] == 20
    assert result["parameters"]["site_area"] == 1500.0
    assert "geometry" not in result["parameters"]
    assert "Параметры квартала id=20:" in result["response"]


def test_block_parameters_request_accepts_natural_word_order():
    assert looks_like_block_parameters_request("Какие параметры квартала 22?")
    assert looks_like_block_parameters_request("Какие у квартала 22 параметры?")
