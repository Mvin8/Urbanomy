"""Scenario tool for Pareto-front solutions."""

from __future__ import annotations
from urbanomy.methods.land_value_modeling import (
    ScenarioTEPModifier,
    plot_scenario_impact,
)
from typing import Any, Dict, Mapping, Sequence
from catboost import CatBoostRegressor
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import random
from geopandas import GeoDataFrame
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy
from blocksnet.analysis.indicators import calculate_density_indicators
from blocksnet.analysis.morphotypes import get_strelka_morphotypes
from blocksnet.enums import LandUse
from urbanomy.methods.investment_potential import prepare_investment_input, InvestmentAttractivenessAnalyzer
from .constants import (
    BlockColumn,
    CATEGORICAL_FEATURES,
    DEFAULT_SQM_PER_PERSON,
    ORIGINAL_FEATURES,
    RADIUS_LIST,
    ScenarioResultKey,
)

from pymoo.algorithms.moo.nsga2 import NSGA2 # Импорт алгоритма
from pymoo.problems import get_problem      # Для получения тестовых задач
from pymoo.optimize import minimize         # Функция для запуска оптимизации
from pymoo.visualization.scatter import Scatter # Для визуализации
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem

from .land_price_estimation import LandPriceEstimator

class DistrictProblem(Problem):
    def __init__(self, blocks: GeoDataFrame, model: CatBoostRegressor, estimator_kwargs: Dict[str, Any], constraints: Dict[str, Dict[str, Any]],benchmarks: Any, potential_df: pd.DataFrame, target_idx: int):
        self.blocks = blocks
        self.model = model
        self.estimator_kwargs = estimator_kwargs
        self.constraints = constraints
        self.benchmarks = benchmarks
        self.potential_df = potential_df
        self.target_idx = target_idx
        self.var_names = list(constraints.keys())
        # Список переменных, которые должны суммироваться в 1
        land_use_vars = ['residential', 'business', 'recreation', 
                         'industrial', 'transport', 'special', 'agriculture']
        self.land_use_indices = [i for i, name in enumerate(self.var_names) if name in land_use_vars]

        super().__init__(
            n_var=len(constraints),
            n_obj=2,
            n_constr=0,
            xl=np.array([c["min"] for c in constraints.values()]),
            xu=np.array([c["max"] for c in constraints.values()]),
            type_var=np.float64
        )
    
    def _repair_genome(self, genome: dict) -> dict:
        if self.target_idx in self.blocks.index:
            row = self.blocks.loc[self.target_idx]
        else:
            row = self.blocks.iloc[self.target_idx - 1]
        site_area = float(row["site_area"])
        footprint_max = 0.8 * site_area
        land_use_keys = [
            "residential",
            "business",
            "recreation",
            "industrial",
            "transport",
            "special",
            "agriculture",
        ]

        # --- Нормализуем доли land_use ---
        cleaned_shares = {k: max(float(genome.get(k, 0.0)), 0.0) for k in land_use_keys}
        total = sum(cleaned_shares.values())
        if total > 0:
            for k in land_use_keys:
                genome[k] = cleaned_shares[k] / total
        else:
            genome["residential"] = 1.0
            for k in land_use_keys[1:]:
                genome[k] = 0.0

        # Обновляем категорию землепользования и residential share.
        genome["share"] = float(genome["residential"])

        land_use_map = {
            "residential": LandUse.RESIDENTIAL,
            "business": LandUse.BUSINESS,
            "recreation": LandUse.RECREATION,
            "industrial": LandUse.INDUSTRIAL,
            "transport": LandUse.TRANSPORT,
            "special": LandUse.SPECIAL,
            "agriculture": LandUse.AGRICULTURE,
        }
        dominant_key = max(land_use_keys, key=lambda k: genome[k])
        genome["land_use"] = land_use_map[dominant_key]

        # footprint_area <= 0.8 * site_area
        footprint = float(genome.get("footprint_area", row.get("footprint_area", 0.0)))
        footprint = float(np.clip(footprint, 0.0, footprint_max))
        genome["footprint_area"] = footprint

        # Предпочитаем параметризацию через (footprint_area, l, mxi).
        if "l" in self.var_names:
            l_value = float(genome.get("l", row.get("l", 1.0)))
            l_value = max(l_value, 1.0)
            build = footprint * l_value
            genome["l"] = l_value
        else:
            build = max(float(genome.get("build_floor_area", row.get("build_floor_area", 0.0))), footprint)
            genome["l"] = (build / footprint) if footprint > 0 else 0.0

        if "mxi" in self.var_names:
            mxi_value = float(genome.get("mxi", row.get("mxi", 0.0)))
            mxi_value = float(np.clip(mxi_value, 0.0, 1.0))
            live = build * mxi_value
            genome["mxi"] = mxi_value
        else:
            live = float(genome.get("living_area", row.get("living_area", 0.0)))
            live = float(np.clip(live, 0.0, build))
            genome["mxi"] = (live / build) if build > 0 else 0.0

        non_live = max(build - live, 0.0)

        genome["build_floor_area"] = build
        genome["living_area"] = live
        genome["non_living_area"] = non_live
        genome["population"] = live / DEFAULT_SQM_PER_PERSON if DEFAULT_SQM_PER_PERSON > 0 else 0.0

        genome["fsi"] = build / site_area if site_area > 0 else 0.0
        genome["gsi"] = footprint / site_area if site_area > 0 else 0.0

        return genome

    def evaluate_catboost(self, geonome: GeoDataFrame, 
                        model: CatBoostRegressor,
                        orig_features: Sequence[str] | None = None,
                        cat_features: Sequence[str] | None = None,
                        radius_list: Sequence[float] | None = None,
                        ):
        features = tuple(orig_features) if orig_features is not None else ORIGINAL_FEATURES
        cats = tuple(cat_features) if cat_features is not None else CATEGORICAL_FEATURES
        radii = tuple(radius_list) if radius_list is not None else RADIUS_LIST

        estimator = LandPriceEstimator(
            model=model,
            blocks=geonome,
            radius_list=radii,
            orig_features=features,
            categorical_features=cats,
        )
        
        estimated = estimator.predict()

        return estimated['land_value'].sum()

    def evaluate_investment_potential(
        self,
        geonome: GeoDataFrame,
        benchmarks: Mapping[LandUse, Dict[str, Any]],
        potential_df: pd.DataFrame,
        discount_rate: float = 0.18,
    ):
        
        geonome["is_project"] = False
        if self.target_idx in geonome.index:
            geonome.loc[self.target_idx, "is_project"] = True
        elif "id" in geonome.columns:
            geonome.loc[geonome["id"] == self.target_idx, "is_project"] = True
        else:
            raise KeyError(
                f"Cannot set is_project for target_idx={self.target_idx}: index and 'id' column mismatch."
            )
        investment_input = prepare_investment_input(
            gdf = geonome,
            project_potential = potential_df,
        )

        an = InvestmentAttractivenessAnalyzer(benchmarks=benchmarks)
        summary = an.calculate_investment_metrics(investment_input, discount_rate=discount_rate)
        
        return summary['NPV'].astype(float).iloc[0]    

    def _evaluate(self, X, out, *args, **kwargs):
        n = X.shape[0]
        F = np.zeros((n, 2))
        
        # Вычисление целевых функций (оставляем цикл, так как оценка индивидуальна)
        for i in range(n):
            genome = X[i]
            changes = {name: genome[j] for j, name in enumerate(self.var_names)}
            changes = self._repair_genome(changes)  # Ремонтируем геном, чтобы соблюсти сумму долей
            modifier = ScenarioTEPModifier(self.blocks)
            blocks_after = modifier.apply(self.target_idx, changes)

            land_value = self.evaluate_catboost(
                geonome=blocks_after,
                model=self.model,
                orig_features=self.estimator_kwargs["orig_features"],
                cat_features=self.estimator_kwargs["categorical_features"],
                radius_list=RADIUS_LIST
            )

            investment_potential = self.evaluate_investment_potential(
                geonome=blocks_after,
                benchmarks=self.benchmarks,
                potential_df=self.potential_df,
                discount_rate=0.18,
            )

            # Если цели нужно максимизировать – ставим минус
            F[i, 0] = -land_value
            F[i, 1] = -investment_potential

        

        out["F"] = F


class NSGA2GenerationStatsCallback(Callback):
    """Print compact per-generation objective stats instead of per-individual logs."""

    def __init__(self, every: int = 1) -> None:
        super().__init__()
        self.every = max(1, int(every))
        self.history: list[dict[str, float]] = []

    def notify(self, algorithm) -> None:
        gen = int(algorithm.n_gen)
        if gen % self.every != 0:
            return

        F = algorithm.pop.get("F")
        if F is None or len(F) == 0:
            return

        f_land = -np.asarray(F[:, 0], dtype=float)
        f_npv = -np.asarray(F[:, 1], dtype=float)
        stats = {
            "gen": gen,
            "land_min": float(np.min(f_land)),
            "land_median": float(np.median(f_land)),
            "land_max": float(np.max(f_land)),
            "npv_min": float(np.min(f_npv)),
            "npv_median": float(np.median(f_npv)),
            "npv_max": float(np.max(f_npv)),
        }
        self.history.append(stats)
        print(
            "[gen {gen:03d}] "
            "land_value min/med/max={land_min:,.0f}/{land_median:,.0f}/{land_max:,.0f} | "
            "NPV min/med/max={npv_min:,.0f}/{npv_median:,.0f}/{npv_max:,.0f}".format(**stats)
        )
