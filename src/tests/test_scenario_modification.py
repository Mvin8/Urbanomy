import os
import sys

import geopandas as gpd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from urbanomy.methods.land_value_modeling.constants import BlockColumn
from urbanomy.methods.land_value_modeling.scenario_modification import _print_summary


def test_print_summary_handles_empty_gdf_without_unboundlocalerror(capsys):
    gdf = gpd.GeoDataFrame(
        {
            BlockColumn.IS_PROJECT.value: [],
            BlockColumn.LAND_USE.value: [],
            "land_value_before": [],
            "land_value_after": [],
            "d_rub": [],
            "d_pct": [],
        },
        geometry=[],
    )

    _print_summary(
        gdf,
        {"sum_before": 0.0, "sum_after": 0.0, "delta": 0.0, "delta_pct": 0.0, "count": 0},
        eps=0.0,
        target_idx=1,
    )

    stdout = capsys.readouterr().out
    assert "Нет кварталов, у которых стоимость изменилась выше порога." in stdout
