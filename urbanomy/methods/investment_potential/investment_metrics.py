"""
Module for computing investment attractiveness metrics for land-use polygons.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple

import geopandas as gpd
import pandas as pd
import numpy as np

from .constants import (
    DEFAULT_ECON_METRIC,
    DEFAULT_DISCOUNT_RATE,
    DEFAULT_AREA_COL,
    DEFAULT_IP_TYPE,
    DEFAULT_IP_VALUE,
    INVESTMENT_WEIGHTS,
)
from .utils_metrics import (
    make_cashflow,
    nanminmax,
    normalize_series,
    to_numeric,
    economic_index,
    aggregate_project_cashflows,
)
from .utils_metrics import npv, irr, payback_period, quantize  # если эти финансовые утилиты тоже в utils


class InvestmentAttractivenessAnalyzer:
    """
    Compute economic and combined investment attractiveness metrics
    for each polygon based on its land-use type.

    Uses per-profile benchmarks and weights to calculate:
    - Economic metrics: NPV, IRR, ROI, payback period, economic index.
    - Spatial-economic combined metric INV.

    Attributes
    ----------
    benchmarks : dict[str, dict[str, Any]]
        Mapping from land-use profile name to its cashflow parameters.
    weights : dict[str, tuple[float, float]]
        Weights for combining spatial (ip_value) and economic metrics for each profile.
    metric : str
        Economic metric used for combination, e.g., "NPV" or "IRR".
    discount_rate : float
        Default discount rate for cashflow calculations if not specified in benchmarks.
    """

    def __init__(
        self,
        benchmarks: dict[str, dict[str, Any]],
        weights_dict: dict[str, tuple[float, float]] | None = None,
        econ_metric: str = DEFAULT_ECON_METRIC,
        discount_rate: float | None = None,
    ) -> None:
        """
        Initialize the analyzer with benchmarks and weights.

        Parameters
        ----------
        benchmarks : dict[str, dict[str, Any]]
            Per-profile cashflow parameters, keyed by profile name.
        weights_dict : dict[str, tuple[float, float]], optional
            Spatial vs economic weights for each profile (w_spatial, w_economic).
            Defaults to INVESTMENT_WEIGHTS.
        econ_metric : str, default DEFAULT_ECON_METRIC
            Economic metric to use when combining (e.g., "NPV", "IRR").
        discount_rate : float, optional
            Default discount rate to apply if not present in profile parameters.

        Returns
        -------
        None
        """
        self.benchmarks = benchmarks
        self.weights = weights_dict or INVESTMENT_WEIGHTS
        self.metric = econ_metric.upper()
        self.discount_rate = (
            discount_rate if discount_rate is not None else DEFAULT_DISCOUNT_RATE
        )

    def calculate_investment_metrics(
        self,
        gdf: gpd.GeoDataFrame,
    ) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """
        Compute row-level economic metrics and combined INV for each polygon.

        This method:
        1. Ensures an area column exists (geometry.area).
        2. Validates 'ip_type' and 'ip_value' columns in long format.
        3. Builds a summary of economic metrics per profile (and optional project).
        4. Maps economic metrics back to each row in gdf as ECON_* columns.
        5. Normalizes spatial and economic values and computes INV row-wise.
        6. Extends summary with INV per profile and project.
        7. Cleans up any temporary area column and returns results.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input GeoDataFrame with columns:
            - geometry: Polygon geometries.
            - ip_type : land-use profile identifier.
            - ip_value: numerical spatial attractiveness score.

        Returns
        -------
        tuple[gpd.GeoDataFrame, pd.DataFrame]
            - enriched_gdf : GeoDataFrame
                Original gdf supplemented with columns:
                ECON_NPV, ECON_IRR, ECON_ROI, ECON_PP_years, ECON_EI, INV.
            - summary : pandas.DataFrame
                Profile-level (and optional project-level) metrics indexed by profile.
        """
        # 1) Add area if missing
        area_col = DEFAULT_AREA_COL if DEFAULT_AREA_COL in gdf.columns else "__area_tmp"
        if area_col == "__area_tmp":
            gdf[area_col] = gdf.geometry.area

        # 2) Check for long format ip_type/ip_value
        has_ip = {DEFAULT_IP_TYPE, DEFAULT_IP_VALUE}.issubset(gdf.columns)
        ip_col = DEFAULT_IP_TYPE if has_ip else None
        val_col = DEFAULT_IP_VALUE if has_ip else None
        if ip_col is None:
            raise ValueError("GeoDataFrame must contain 'ip_type' and 'ip_value' columns")

        # 3) Build profile summary (economic metrics + investment_attractiveness)
        summary = self._build_profile_summary(gdf, area_col, ip_col)

        # 4) Map summary economic metrics back to each row
        for econ in ("NPV", "IRR", "ROI", "PP_years", "EI"):
            gdf[f"ECON_{econ}"] = gdf[ip_col].map(summary[econ])

        # 5) Normalize and compute INV per row
        s_raw = gdf[val_col].astype(float)
        e_raw = gdf[f"ECON_{self.metric}"].astype(float)

        s_min, s_max = s_raw.min(), s_raw.max()
        e_min, e_max = e_raw.min(), e_raw.max()
        if self.metric in ("NPV", "ROI"):
            e_abs = max(abs(e_min), abs(e_max))
            e_min, e_max = -e_abs, e_abs

        s_norm = normalize_series(s_raw, s_min, s_max)
        e_norm = normalize_series(e_raw, e_min, e_max)

        ws = pd.Series({lu: w_s for lu, (w_s, _) in self.weights.items()})
        we = pd.Series({lu: w_e for lu, (_, w_e) in self.weights.items()})
        w_s_row = gdf[ip_col].map(ws)
        w_e_row = gdf[ip_col].map(we)

        gdf["INV"] = (w_s_row * s_norm + w_e_row * e_norm).round(2)

        # 6) Extend summary with INV per profile/project
        summary = self._compute_summary_inv(summary)

        # 7) Drop temporary area column
        if area_col == "__area_tmp":
            gdf.drop(columns=[area_col], inplace=True)

        # 8) Select output columns
        out_cols = [
            "geometry", ip_col, val_col
        ] + [f"ECON_{e}" for e in ("NPV", "IRR", "ROI", "PP_years", "EI")] + ["INV"]
        return gdf[out_cols], summary

    def _build_profile_summary(
        self,
        gdf: gpd.GeoDataFrame,
        area_col: str,
        ip_col: str
    ) -> pd.DataFrame:
        """
        Build a summary of economic metrics per land-use profile and optional project.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input GeoDataFrame containing rows with ip_type and area.
        area_col : str
            Name of the column containing polygon areas.
        ip_col : str
            Name of the column containing land-use profile identifiers.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by profile name with columns:
            - NPV          : quantized net present value.
            - IRR          : internal rate of return.
            - ROI          : return on investment.
            - PP_years     : payback period in years.
            - EI           : economic index.
            - investment_attractiveness : mean ip_value per profile.
            Optionally includes a "project" row if multiple distinct geometries exist.
        """
        rows = []
        for lu, prof in self.benchmarks.items():
            mask = gdf[ip_col] == lu
            total_area = gdf.loc[mask, area_col].sum()

            cf = make_cashflow(lu, total_area, prof)
            rate = prof.get("discount_rate", self.discount_rate)

            raw_npv = npv(rate, cf)
            npv_v = quantize(raw_npv)
            irr_v = irr(cf)
            roi_v = (sum(cf[1:]) / -cf[0] if cf and cf[0] < 0 else np.nan)
            pp_v = payback_period(rate, cf)
            ei_v = economic_index(raw_npv, irr_v, rate)

            inv_attr = gdf.loc[mask, DEFAULT_IP_VALUE].mean()

            rows.append({
                "profile": lu,
                "NPV": npv_v,
                "IRR": irr_v,
                "ROI": roi_v,
                "PP_years": pp_v,
                "EI": ei_v,
                "investment_attractiveness": inv_attr,
            })

        summary = pd.DataFrame(rows).set_index("profile")

        # Add project-level metrics only if more than one unique geometry
        unique_geom_count = gdf.geometry.apply(lambda geom: geom.wkt).nunique()
        if unique_geom_count > 1:
            project_cf = aggregate_project_cashflows(
                [
                    {"area": r[area_col], "ip_type": r[ip_col]}
                    for _, r in gdf.iterrows()
                ],
                self.benchmarks
            )
            raw_npv_p = npv(self.discount_rate, project_cf)
            print(f"raw_npv_p: {raw_npv_p}")
            irr_p = irr(project_cf)
            roi_p = (sum(project_cf[1:]) / -project_cf[0]
                     if project_cf and project_cf[0] < 0 else np.nan)
            pp_p = payback_period(self.discount_rate, project_cf)
            ei_p = economic_index(raw_npv_p, irr_p, self.discount_rate)
            inv_attr_p = gdf[DEFAULT_IP_VALUE].mean()

            summary.loc["project"] = {
                "NPV": raw_npv_p,
                "IRR": irr_p,
                "ROI": roi_p,
                "PP_years": pp_p,
                "EI": ei_p,
                "investment_attractiveness": inv_attr_p,
            }

        summary[["IRR", "ROI", "PP_years", "EI"]] = summary[
            ["IRR", "ROI", "PP_years", "EI"]
        ].round(2)

        return summary

    def _compute_summary_inv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize summary metrics and compute INV per profile and project.

        Parameters
        ----------
        df : pandas.DataFrame
            Summary DataFrame with index 'profile' and columns:
            'investment_attractiveness' and the chosen econ metric (e.g. 'NPV').

        Returns
        -------
        pandas.DataFrame
            Summary DataFrame with added column:
            - INV : combined investment attractiveness metric.
        """
        df = df.copy()
        s = to_numeric(df["investment_attractiveness"])
        e = to_numeric(df[self.metric])

        s_min, s_max = s.min(), s.max()
        e_min, e_max = e.min(), e.max()
        if self.metric in ("NPV", "ROI"):
            e_abs = max(abs(e_min), abs(e_max))
            e_min, e_max = -e_abs, e_abs

        s_norm = normalize_series(s, s_min, s_max)
        e_norm = normalize_series(e, e_min, e_max)

        ws = pd.Series({lu: w_s for lu, (w_s, _) in self.weights.items()})
        we = pd.Series({lu: w_e for lu, (_, w_e) in self.weights.items()})
        ws["project"], we["project"] = ws.mean(), we.mean()

        df["INV"] = (ws * s_norm + we * e_norm).round(2)
        return df
