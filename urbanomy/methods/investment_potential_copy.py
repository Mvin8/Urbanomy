import json
import math
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import geopandas as gpd


class InvestmentAnalysisModule:
    """
    Объединённый модуль для расчёта инвестиционной привлекательности:
      1) spatial scores (ИП_<lu>)
      2) economic metrics (ECON_*)
      3) synthesis (INV_<lu> и итоговый INV в summary)
    """

    # --- Spatial defaults ---
    LAND_USE_TO_POTENTIAL_COLUMN: Dict[str, str] = {
        'residential_individual': 'Потенциал развития жилой застройки типа ИЖС',
        'residential_lowrise': 'Потенциал развития малоэтажной жилой застройки',
        'residential_midrise': 'Потенциал развития среднеэтажной жилой застройки',
        'residential_multistorey': 'Потенциал развития многоэтажной жилой застройки',
        'business': 'Потенциал развития застройки общественно-деловой зоны',
        'recreation': 'Потенциал развития застройки рекреационной зоны',
        'special': 'Потенциал развития застройки зоны специального назначения',
        'industrial': 'Потенциал развития застройки промышленной зоны',
        'agriculture': 'Потенциал развития застройки сельскохозяйственной зоны',
        'transport': 'Потенциал развития застройки транспортной зоны'
    }

    DEFAULT_SPATIAL_WEIGHTS: Dict[str, Dict[str, float]] = {
        'residential_individual': {
            'Население': 1.3, 'Социальное обеспечение': 1.4,
            'Экологическая ситуация': 1.5,
            'Средняя доступность до близлежащего крупного населенного пункта': 1.2,
            'Транспортное обеспечение': 1.1, 'default': 1.0
        },
        'residential_lowrise': {
            'Население': 1.4, 'Социальное обеспечение': 1.3,
            'Экологическая ситуация': 1.4,
            'Транспортное обеспечение': 1.2, 'default': 1.0
        },
        'residential_midrise': {
            'Средняя этажность': 1.5, 'Население': 1.4,
            'Социальное обеспечение': 1.3,
            'Транспортное обеспечение': 1.2, 'default': 1.0
        },
        'residential_multistorey': {
            'Средняя этажность': 1.5, 'Население': 1.4,
            'Транспортное обеспечение': 1.3,
            'Социальное обеспечение': 1.2, 'default': 1.0
        },
        'business': {
            'Транспортное обеспечение': 1.5, 'Население': 1.4,
            'Социальное обеспечение (комфорт)': 1.3,
            'Средняя доступность до близлежащего крупного населенного пункта': 1.2,
            'default': 1.0
        },
        'recreation': {
            'Экологическая ситуация': 1.5,
            'Социальное обеспечение (комфорт)': 1.4,
            'Транспортное обеспечение': 1.2, 'Население': 0.8,
            'default': 1.0
        },
        'special': {
            'Потенциал размещения порта': 1.5,
            'Транспортное обеспечение': 1.4,
            'Потенциал размещения логистического, складского комплекса': 1.3,
            'default': 1.0
        },
        'industrial': {
            'Потенциал размещения логистического, складского комплекса': 1.5,
            'Транспортное обеспечение': 1.4,
            'Экологическая ситуация': 0.8, 'Население': 0.9,
            'default': 1.0
        },
        'agriculture': {
            'Экологическая ситуация': 1.5, 'Население': 0.8,
            'Транспортное обеспечение': 1.2,
            'Средняя доступность до близлежащего крупного населенного пункта': 1.1,
            'default': 1.0
        },
        'transport': {
            'Потенциал размещения логистического, складского комплекса': 1.5,
            'Количество аэропортов местного значения': 1.4,
            'Средняя доступность до близлежащего крупного населенного пункта': 1.3,
            'default': 1.0
        }
    }

    # --- Synthesis defaults ---
    DEFAULT_SYNTH_WEIGHTS: Dict[str, Tuple[float, float]] = {
        "residential_individual": (0.4, 0.6),
        "residential_lowrise"   : (0.4, 0.6),
        "residential_midrise"   : (0.5, 0.5),
        "residential_multistorey": (0.5, 0.5),
        "business"              : (0.3, 0.7),
        "recreation"            : (0.6, 0.4),
        "special"               : (0.3, 0.7),
        "industrial"            : (0.35, 0.65),
        "agriculture"           : (0.6, 0.4),
        "transport"             : (0.35, 0.65)
    }

    def __init__(
        self,
        benchmarks: Dict[str, Dict[str, Any]],
        base_discount_rate: float = 0.12,
        spatial_weights: Dict[str, Dict[str, float]] | None = None,
        spatial_weights_path: str | None = None,
        synth_weights: Dict[str, Tuple[float, float]] | None = None,
        econ_metric: str = "EI",
        area_col: str = "Площадь территории",
        ip_type_col: str = "ip_type",
        ip_value_col: str = "ip_value"
    ):
        # Spatial weights
        if spatial_weights is not None:
            self.spatial_weights = spatial_weights
        elif spatial_weights_path:
            with open(spatial_weights_path, 'r', encoding='utf-8') as f:
                self.spatial_weights = json.load(f)
        else:
            self.spatial_weights = self.DEFAULT_SPATIAL_WEIGHTS

        # Economics
        self.benchmarks = benchmarks
        self.base_discount_rate = base_discount_rate

        # Synthesis
        self.synth_weights = synth_weights or self.DEFAULT_SYNTH_WEIGHTS
        self.econ_metric = econ_metric.upper()

        # Common cols
        self.area_col = area_col
        self.ip_type_col = ip_type_col
        self.ip_value_col = ip_value_col

    # ------------------ STEP 1: Spatial ------------------ #
    def compute_spatial_scores(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        pot_cols = list(self.LAND_USE_TO_POTENTIAL_COLUMN.values())
        # numeric attrs excluding potentials, filter on reasonable range
        attrs = [
            c for c in gdf.columns
            if gdf[c].dtype.kind in 'if'
            and c not in pot_cols
            and (((gdf[c] >= -5) & (gdf[c] <= 5) & (gdf[c] != 0)).any())
        ]

        for lu, pot_col in self.LAND_USE_TO_POTENTIAL_COLUMN.items():
            score_col = f'ИП_{lu}'

            def calc(row):
                pot = row.get(pot_col)
                if pd.isna(pot):
                    return None
                vals = []
                for attr in attrs:
                    v = row.get(attr)
                    if pd.notna(v):
                        w = self.spatial_weights.get(lu, {}).get(attr,
                            self.spatial_weights[lu]['default'])
                        vals.append(v * w)
                if not vals:
                    return None
                return round(sum(vals) / len(vals) * (pot / 5), 1)

            gdf[score_col] = gdf.apply(calc, axis=1)

        return gdf

    # ------------------ STEP 2: Economics ------------------ #
    def _make_cashflow(
        self, profile: str, land_area: float, b: Dict[str, Any]
    ) -> List[float]:
        density = b["density"]
        gfa = land_area * density
        land_cost_total = land_area * b.get("land_cost", 0)
        build_cost_total = gfa * b["cost_build"]
        years_build = b.get("construction_years", 2)
        capex_per_year = build_cost_total / years_build

        cash = [-land_cost_total - capex_per_year]
        for _ in range(1, years_build):
            cash.append(-capex_per_year)

        opex = b.get("opex_rate", 0)
        if "price_sale" in b:
            yrs = b.get("sale_years", 3)
            rev = gfa * b["price_sale"]
            per = rev / yrs
            for _ in range(yrs):
                cash.append(per - gfa * opex)
        else:
            yrs = b.get("rent_years", 10)
            occ = b.get("occupancy", 0.9)
            per = gfa * b["rent_annual"] * occ
            for _ in range(yrs):
                cash.append(per - gfa * opex)

        return cash

    @staticmethod
    def _npv(rate: float, cf: List[float]) -> float:
        return sum(v / (1 + rate) ** i for i, v in enumerate(cf))

    @staticmethod
    def _irr(cf: List[float], guess=0.1, tol=1e-6, max_iter=100) -> float | None:
        rate = guess
        for _ in range(max_iter):
            d_npv = sum(-i * v / (1 + rate) ** (i + 1)
                        for i, v in enumerate(cf))
            if d_npv == 0:
                break
            rate_new = rate - InvestmentAnalysisModule._npv(rate, cf) / d_npv
            if abs(rate_new - rate) < tol:
                return rate_new
            rate = rate_new
        return None

    @staticmethod
    def _payback_period(rate: float, cf: List[float]) -> float | None:
        cum = 0.0
        for i, v in enumerate(cf):
            cum += v / (1 + rate) ** i
            if cum >= 0:
                prev = cum - v / (1 + rate) ** i
                return i - (prev / (v / (1 + rate) ** i)) if v else i
        return None
    def compute_economic_metrics(
        self, gdf: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
        if self.area_col not in gdf.columns:
            gdf[self.area_col] = gdf.geometry.area

        summary = []
        for prof, b in self.benchmarks.items():
            ip_col = f'ИП_{prof}'

            # площадь под этот профиль
            if self.ip_type_col in gdf.columns:
                mask = gdf[self.ip_type_col] == ip_col
                land_area = gdf.loc[mask, self.area_col].sum()
            else:
                land_area = gdf[self.area_col].sum()

            # cashflows и базовые метрики
            cf  = self._make_cashflow(prof, land_area, b)
            r   = b.get("discount_rate", self.base_discount_rate)
            npv = self._npv(r, cf)
            irr = self._irr(cf)
            roi = (sum(cf[1:]) / -cf[0]) if cf and cf[0] < 0 else float('nan')
            pp  = self._payback_period(r, cf)

            # Economic Index EI
            ei = 0.0
            if npv is not None:
                arg = -npv / 1e8
                sig = (1.0 / (1.0 + math.exp(arg))) if abs(arg) < 700 else (0.0 if arg > 700 else 1.0)
                ei += 50 * sig
            if irr is not None:
                ei += 50 * max(0.0, irr - r) / (0.3 - r)
            ei = min(max(ei, 0), 100)

            # пишем в gdf
            for col_name, val in [
                (f"ECON_NPV_{prof}", npv),
                (f"ECON_IRR_{prof}", irr),
                (f"ECON_ROI_{prof}", roi),
                (f"ECON_PP_{prof}", pp),
                (f"ECON_EI_{prof}", ei)
            ]:
                gdf[col_name] = val

            # ИНВЕСТИЦИОННАЯ ПРИВЛЕКАТЕЛЬНОСТЬ (spatial) — просто среднее по ИП_<profile>
            inv_attr = gdf[ip_col].mean()

            summary.append({
                "profile": prof,
                "NPV": npv,
                "IRR": irr,
                "ROI": roi,
                "PP_years": pp,
                "EI": ei,
                "investment_attractiveness": inv_attr
            })

        summary_df = pd.DataFrame(summary).set_index("profile")

        # ------------------------------
        # 1) Округляем все числовые поля
        numeric_cols = ["NPV", "IRR", "ROI", "PP_years", "EI", "investment_attractiveness"]
        summary_df[numeric_cols] = summary_df[numeric_cols].astype(float).round(2)

        # 2) Форматируем NPV с разделителями тысяч (строкой)
        summary_df["NPV"] = summary_df["NPV"].map(lambda x: f"{x:,.0f}")

        # При агрегации по проекту (если в gdf >1 зоны) — аналогично считаем и добавляем строку "project"
        if len(gdf) > 1:
            all_cfs = []
            for _, row in gdf.iterrows():
                prof = row[self.ip_type_col].replace("ИП_", "")
                all_cfs.append(self._make_cashflow(prof, row[self.area_col], self.benchmarks[prof]))

            max_len = max(len(cf) for cf in all_cfs)
            agg_cf = [ sum(cf[i] if i < len(cf) else 0.0 for cf in all_cfs)
                    for i in range(max_len) ]

            proj_npv = self._npv(self.base_discount_rate, agg_cf)
            proj_irr = self._irr(agg_cf)
            proj_roi = (sum(agg_cf[1:]) / -agg_cf[0]) if agg_cf and agg_cf[0] < 0 else float('nan')
            proj_pp  = self._payback_period(self.base_discount_rate, agg_cf)

            # Project EI
            proj_ei = 0.0
            arg = -proj_npv / 1e8
            sig = (1.0 / (1.0 + math.exp(arg))) if abs(arg) < 700 else (0.0 if arg > 700 else 1.0)
            proj_ei += 50 * sig
            proj_ei += 50 * max(0.0, proj_irr - self.base_discount_rate) / (0.3 - self.base_discount_rate)
            proj_ei = min(max(proj_ei, 0), 100)

            proj_inv_attr = gdf["investment_attractiveness"].mean() \
                if "investment_attractiveness" in gdf.columns else float('nan')

            df_proj = pd.DataFrame([{
                "profile": "project",
                "NPV": proj_npv,
                "IRR": proj_irr,
                "ROI": proj_roi,
                "PP_years": proj_pp,
                "EI": proj_ei,
                "investment_attractiveness": proj_inv_attr
            }]).set_index("profile")

            # Округление и форматирование NPV для project
            df_proj[numeric_cols] = df_proj[numeric_cols].astype(float).round(2)
            df_proj["NPV"] = df_proj["NPV"].map(lambda x: f"{x:,.0f}")

            summary_df = pd.concat([summary_df, df_proj])

        return gdf, summary_df
    
    def compute_economic_metrics(
        self, gdf: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
        if self.area_col not in gdf.columns:
            gdf[self.area_col] = gdf.geometry.area

        summary = []
        for prof, b in self.benchmarks.items():
            ip_col = f'ИП_{prof}'

            # площадь под этот профиль
            if self.ip_type_col in gdf.columns:
                mask = gdf[self.ip_type_col] == ip_col
                land_area = gdf.loc[mask, self.area_col].sum()
            else:
                land_area = gdf[self.area_col].sum()

            # cashflows и базовые метрики
            cf  = self._make_cashflow(prof, land_area, b)
            r   = b.get("discount_rate", self.base_discount_rate)
            npv = self._npv(r, cf)
            irr = self._irr(cf)
            roi = (sum(cf[1:]) / -cf[0]) if cf and cf[0] < 0 else float('nan')
            pp  = self._payback_period(r, cf)

            # Economic Index EI
            ei = 0.0
            if npv is not None:
                arg = -npv / 1e8
                sig = (1.0 / (1.0 + math.exp(arg))) if abs(arg) < 700 else (0.0 if arg > 700 else 1.0)
                ei += 50 * sig
            if irr is not None:
                ei += 50 * max(0.0, irr - r) / (0.3 - r)
            ei = min(max(ei, 0), 100)

            # пишем в gdf
            for col_name, val in [
                (f"ECON_NPV_{prof}", npv),
                (f"ECON_IRR_{prof}", irr),
                (f"ECON_ROI_{prof}", roi),
                (f"ECON_PP_{prof}", pp),
                (f"ECON_EI_{prof}", ei)
            ]:
                gdf[col_name] = val

            # ИНВЕСТИЦИОННАЯ ПРИВЛЕКАТЕЛЬНОСТЬ (spatial) — просто среднее по ИП_<profile>
            inv_attr = gdf[ip_col].mean()

            summary.append({
                "profile": prof,
                "NPV": npv,
                "IRR": irr,
                "ROI": roi,
                "PP_years": pp,
                "EI": ei,
                "investment_attractiveness": inv_attr
            })

        summary_df = pd.DataFrame(summary).set_index("profile")

        # ------------------------------
        # 1) Округляем все числовые поля
        numeric_cols = ["NPV", "IRR", "ROI", "PP_years", "EI", "investment_attractiveness"]
        summary_df[numeric_cols] = summary_df[numeric_cols].astype(float).round(2)

        # 2) Форматируем NPV с разделителями тысяч (строкой)
        summary_df["NPV"] = summary_df["NPV"].map(lambda x: f"{x:,.0f}")

        # При агрегации по проекту (если в gdf >1 зоны) — аналогично считаем и добавляем строку "project"
        if len(gdf) > 1:
            all_cfs = []
            for _, row in gdf.iterrows():
                prof = row[self.ip_type_col].replace("ИП_", "")
                all_cfs.append(self._make_cashflow(prof, row[self.area_col], self.benchmarks[prof]))

            max_len = max(len(cf) for cf in all_cfs)
            agg_cf = [ sum(cf[i] if i < len(cf) else 0.0 for cf in all_cfs)
                    for i in range(max_len) ]

            proj_npv = self._npv(self.base_discount_rate, agg_cf)
            proj_irr = self._irr(agg_cf)
            proj_roi = (sum(agg_cf[1:]) / -agg_cf[0]) if agg_cf and agg_cf[0] < 0 else float('nan')
            proj_pp  = self._payback_period(self.base_discount_rate, agg_cf)

            # Project EI
            proj_ei = 0.0
            arg = -proj_npv / 1e8
            sig = (1.0 / (1.0 + math.exp(arg))) if abs(arg) < 700 else (0.0 if arg > 700 else 1.0)
            proj_ei += 50 * sig
            proj_ei += 50 * max(0.0, proj_irr - self.base_discount_rate) / (0.3 - self.base_discount_rate)
            proj_ei = min(max(proj_ei, 0), 100)

            proj_inv_attr = gdf["investment_attractiveness"].mean() \
                if "investment_attractiveness" in gdf.columns else float('nan')

            df_proj = pd.DataFrame([{
                "profile": "project",
                "NPV": proj_npv,
                "IRR": proj_irr,
                "ROI": proj_roi,
                "PP_years": proj_pp,
                "EI": proj_ei,
                "investment_attractiveness": proj_inv_attr
            }]).set_index("profile")

            # Округление и форматирование NPV для project
            df_proj[numeric_cols] = df_proj[numeric_cols].astype(float).round(2)
            df_proj["NPV"] = df_proj["NPV"].map(lambda x: f"{x:,.0f}")

            summary_df = pd.concat([summary_df, df_proj])

        return gdf, summary_df


    # ------------------ STEP 3: Synthesis ------------------ #
    def synthesize(
        self, gdf: gpd.GeoDataFrame, summary_df: pd.DataFrame
    ) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
        # собираем все spatial и econ для нормировки
        s_vals, e_vals = [], []
        for lu in self.synth_weights:
            s_vals += list(gdf[f'ИП_{lu}'].astype(float))
            e_vals += list(gdf[f'ECON_{self.econ_metric}_{lu}'].astype(float))

        s_min, s_max = np.nanmin(s_vals), np.nanmax(s_vals)
        e_min, e_max = np.nanmin(e_vals), np.nanmax(e_vals)
        if self.econ_metric in ("NPV", "ROI"):
            m = max(abs(e_min), abs(e_max))
            e_min, e_max = -m, m

        # enrich gdf
        for lu, (ws, we) in self.synth_weights.items():
            s = gdf[f'ИП_{lu}'].astype(float)
            e = gdf[f'ECON_{self.econ_metric}_{lu}'].astype(float)

            s_n = 100 * (s - s_min) / (s_max - s_min) if s_max > s_min else 0
            e_n = 100 * (e - e_min) / (e_max - e_min) if e_max > e_min else 0

            gdf[f'INV_{lu}'] = (ws * s_n + we * e_n).round(2)

        # enrich summary_df
        df = summary_df.copy()
        s0 = pd.to_numeric(df["investment_attractiveness"]
                           .astype(str)
                           .str.replace(r"[^0-9\.\-]+", "", regex=True),
                           errors='coerce')
        e0 = pd.to_numeric(df[self.econ_metric]
                           .astype(str)
                           .str.replace(r"[^0-9\.\-]+", "", regex=True),
                           errors='coerce')

        smin, smax = s0.min(), s0.max()
        emin, emax = e0.min(), e0.max()
        if self.econ_metric in ("NPV", "ROI"):
            m = max(abs(emin), abs(emax))
            emin, emax = -m, m

        s_n = 100 * (s0 - smin) / (smax - smin) if smax > smin else 0
        e_n = 100 * (e0 - emin) / (emax - emin) if emax > emin else 0

        ws = pd.Series({lu: w for lu, (w, _) in self.synth_weights.items()})
        we = pd.Series({lu: e for lu, (_, e) in self.synth_weights.items()})
        avg_ws, avg_we = ws.mean(), we.mean()
        ws.loc["project"] = avg_ws
        we.loc["project"] = avg_we

        df["INV"] = (ws * s_n + we * e_n).round(2)
        return gdf, df

    # -------------- Orchestrator ---------------- #
    def analyze(self, gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """
        Единственная точка входа.
        """
        gdf = gdf.copy()
        if self.area_col not in gdf.columns:
            gdf[self.area_col] = gdf.geometry.area

        # 1) spatial
        sp = self.compute_spatial_scores(gdf)
        # 2) economic
        eco_gdf, summary = self.compute_economic_metrics(sp)
        # 3) synth
        final_gdf, final_summary = self.synthesize(eco_gdf, summary)
        return final_gdf, final_summary


def run_investment_analysis(
    gdf: gpd.GeoDataFrame,
    benchmarks: Dict[str, Dict[str, Any]],
    spatial_weights: Dict[str, Dict[str, float]] | None = None,
    spatial_weights_path: str | None = None,
    synth_weights: Dict[str, Tuple[float, float]] | None = None,
    econ_metric: str = "EI",
    discount_rate: float = 0.12,
    area_col: str = "Площадь территории",
    ip_type_col: str = "ip_type",
    ip_value_col: str = "ip_value"
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Модульный вызов: Instantiates InvestmentAnalysisModule и возвращает (gdf, summary_df).
    """
    analyzer = InvestmentAnalysisModule(
        benchmarks=benchmarks,
        base_discount_rate=discount_rate,
        spatial_weights=spatial_weights,
        spatial_weights_path=spatial_weights_path,
        synth_weights=synth_weights,
        econ_metric=econ_metric,
        area_col=area_col,
        ip_type_col=ip_type_col,
        ip_value_col=ip_value_col
    )
    return analyzer.analyze(gdf)
