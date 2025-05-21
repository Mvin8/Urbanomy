# investment_attractiveness.py

from __future__ import annotations
from typing import Dict, Any, Tuple

import geopandas as gpd
import pandas as pd

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
    Синтез «пространственных» оценок (ИП_*) и экономических метрик
    (NPV, IRR, EI…) в итоговый показатель **INV_<land_use>**.
    """

    def __init__(
        self,
        benchmarks: Dict[str, Dict[str, Any]],
        weights_dict: Dict[str, Tuple[float, float]] | None = None,
        econ_metric: str = DEFAULT_ECON_METRIC,
        discount_rate: float | None = None,
    ) -> None:
        self.benchmarks = benchmarks
        self.weights = weights_dict or INVESTMENT_WEIGHTS
        self.metric = econ_metric.upper()
        self.discount_rate = (
            discount_rate if discount_rate is not None else DEFAULT_DISCOUNT_RATE
        )

    def calculate_investment_metrics(
        self,
        gdf: gpd.GeoDataFrame,
    ) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
        # 1) Определяем колонку с площадями
        area_col = (
            DEFAULT_AREA_COL
            if DEFAULT_AREA_COL in gdf.columns
            else "__area_tmp"
        )
        if area_col == "__area_tmp":
            gdf[area_col] = gdf.geometry.area

        # 2) Проверяем «плоскую» таблицу ИП
        has_flat_ip = {DEFAULT_IP_TYPE, DEFAULT_IP_VALUE}.issubset(gdf.columns)
        ip_type_col = DEFAULT_IP_TYPE if has_flat_ip else None
        ip_value_col = DEFAULT_IP_VALUE if has_flat_ip else None

        # 3) Считаем экономику
        gdf, summary = self._compute_economic(gdf, area_col, ip_type_col, ip_value_col)

        # 4) Синтез «пространственного» + «экономического»
        gdf = self._compute_spatial_economic_combined(gdf, ip_type_col, ip_value_col)

        # 5) Итоговый INV в summary
        summary = self._compute_summary_inv(summary)

        if area_col == "__area_tmp":
            gdf.drop(columns=[area_col], inplace=True)

        return gdf, summary

    def _compute_economic(
        self,
        gdf: gpd.GeoDataFrame,
        area_col: str,
        ip_type_col: str | None,
        ip_value_col: str | None,
    ) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
        rows = []
        for lu, prof in self.benchmarks.items():
            ip_full = f"ИП_{lu}"
            land_area = (
                gdf.loc[gdf[ip_type_col] == ip_full, area_col].sum()
                if ip_type_col
                else gdf[area_col].sum()
            )

            cf_raw = make_cashflow(lu, land_area, prof)
            rate = prof.get("discount_rate", self.discount_rate)

            rawnpv = npv(rate, cf_raw)
            npv_val = quantize(rawnpv)
            irr_val = irr(cf_raw)
            roi_val = (
                sum(cf_raw[1:]) / -cf_raw[0]
                if cf_raw and cf_raw[0] < 0
                else float("nan")
            )
            pp_val = payback_period(rate, cf_raw)
            ei_val = economic_index(rawnpv, irr_val, rate)

            # добавляем в gdf константные колонки
            for name, val in (("NPV", npv_val), ("IRR", irr_val),
                              ("ROI", roi_val), ("PP", pp_val), ("EI", ei_val)):
                gdf[f"ECON_{name}_{lu}"] = val

            inv_attr = (
                gdf[ip_full].mean()
                if ip_full in gdf.columns
                else (gdf.loc[gdf[ip_type_col] == ip_full, ip_value_col].mean()
                      if ip_type_col else float("nan"))
            )

            rows.append({
                "profile": lu,
                "NPV": npv_val,
                "IRR": irr_val,
                "ROI": roi_val,
                "PP_years": pp_val,
                "EI": ei_val,
                "investment_attractiveness": inv_attr,
            })

        summary = pd.DataFrame(rows).set_index("profile")

        # проектный уровень
        if len(gdf) > 1:
            # готовим строки-словарики с area/ip_type для aggregate_project_cashflows
            row_dicts = [
                {"area": row[area_col], "ip_type": row.get(ip_type_col, "")}
                for _, row in gdf.iterrows()
            ]
            agg_cf = aggregate_project_cashflows(row_dicts, self.benchmarks)
            rawnpv_proj = npv(self.discount_rate, agg_cf)
            irr_proj = irr(agg_cf)
            roi_proj = (sum(agg_cf[1:]) / -agg_cf[0] if agg_cf and agg_cf[0] < 0 else float("nan"))
            pp_proj = payback_period(self.discount_rate, agg_cf)
            ei_proj = economic_index(rawnpv_proj, irr_proj, self.discount_rate)
            inv_attr_proj = (gdf[DEFAULT_IP_VALUE].mean() if ip_value_col else float("nan"))

            summary.loc["project"] = {
                "NPV": quantize(rawnpv_proj),
                "IRR": irr_proj,
                "ROI": roi_proj,
                "PP_years": pp_proj,
                "EI": ei_proj,
                "investment_attractiveness": inv_attr_proj,
            }

        # округляем
        summary[["IRR", "ROI", "PP_years", "EI"]] = summary[
            ["IRR", "ROI", "PP_years", "EI"]
        ].round(2)

        return gdf, summary

    def _compute_spatial_economic_combined(
        self,
        gdf: gpd.GeoDataFrame,
        ip_type_col: str | None,
        ip_value_col: str | None,
    ) -> gpd.GeoDataFrame:
        # собираем все сырые spatial/economic для глобального min/max
        spat_vals, econ_vals = [], []
        def spatial_series(lu: str) -> pd.Series:
            col = f"ИП_{lu}"
            if col in gdf:
                return gdf[col].astype(float)
            if ip_type_col and ip_value_col:
                mask = gdf[ip_type_col] == col
                out = pd.Series(0.0, index=gdf.index)
                out[mask] = gdf.loc[mask, ip_value_col].astype(float)
                return out
            return pd.Series(0.0, index=gdf.index)

        for lu in self.weights:
            spat = spatial_series(lu)
            econ = gdf[f"ECON_{self.metric}_{lu}"].astype(float)
            spat_vals.extend(spat.tolist())
            econ_vals.extend(econ.tolist())

        s_min, s_max = nanminmax(spat_vals)
        e_min, e_max = nanminmax(econ_vals)
        if self.metric in ("NPV", "ROI"):
            e_abs = max(abs(e_min), abs(e_max))
            e_min, e_max = -e_abs, e_abs

        for lu, (w_s, w_e) in self.weights.items():
            s_raw = spatial_series(lu)
            e_raw = gdf[f"ECON_{self.metric}_{lu}"].astype(float)
            s_norm = normalize_series(s_raw, s_min, s_max)
            e_norm = normalize_series(e_raw, e_min, e_max)
            gdf[f"INV_{lu}"] = (w_s * s_norm + w_e * e_norm).round(2)

        return gdf

    def _compute_summary_inv(self, df: pd.DataFrame) -> pd.DataFrame:
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
