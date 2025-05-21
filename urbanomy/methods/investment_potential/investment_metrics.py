from __future__ import annotations

import math
from decimal import Decimal
from typing import Dict, Any, List, Tuple, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd


from .utils import npv, irr, payback_period, quantize
from .constants import (
    DEFAULT_ECON_METRIC,
    DEFAULT_DISCOUNT_RATE,
    DEFAULT_AREA_COL,
    DEFAULT_IP_TYPE,
    DEFAULT_IP_VALUE,
    INVESTMENT_WEIGHTS
)

class InvestmentAttractivenessAnalyzer:
    """
    Синтез «пространственных» оценок (ИП_*) и экономических метрик
    (NPV, IRR / EI…) в итоговый показатель **INV_<land_use>**.

    Публичные методы
    ----------------
    * ``fit_transform(gdf)`` — добавляет в GeoDataFrame ECON_* и INV_*,
      а также возвращает таблицу-сводку по профилям.
    """

    # ------------------------------------------------------------------ #
    # инициализация
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        benchmarks: Dict[str, Dict[str, Any]],
        weights_dict: Dict[str, Tuple[float, float]] | None = None,
        econ_metric: str = DEFAULT_ECON_METRIC,
        discount_rate: float | None = None,
    ) -> None:
        """
        Parameters
        ----------
        benchmarks
            Словарь ``{<land_use>: {...econ-профиль...}}`` — см. README.
        weights_dict
            Словарь ``{<land_use>: (w_spatial, w_economic)}``.
            Если ``None`` → ``weights_inv.DEFAULT_WEIGHTS``.
        econ_metric
            Метрика для нормировки экономики (``"EI"``, ``"NPV"``, ``"IRR"``…).
        discount_rate
            Ставка дисконтирования по умолчанию (если профили не задали свою).
        """
        self.benchmarks = benchmarks
        self.weights = weights_dict or INVESTMENT_WEIGHTS
        self.metric = econ_metric.upper()
        self.discount_rate = (
            discount_rate if discount_rate is not None else DEFAULT_DISCOUNT_RATE
        )

    # ------------------------------------------------------------------ #
    # публичный API
    # ------------------------------------------------------------------ #

    def calculate_investment_metrics(
        self,
        gdf,  # type: ignore[type-var]
    ) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """
        Добавляет колонки:

        * ``ECON_<метрика>_<lu>`` – экономический расчёт по каждому профилю.
        * ``INV_<lu>``            – синтез spatial + economic.
        * summary-таблица         – общие показатели по профилям + «project».

        Returns
        -------
        Tuple[gpd.GeoDataFrame, pandas.DataFrame]
            Обновлённый GeoDataFrame **и** DataFrame-сводка.
        """
        # --- 1. какая колонка содержит площадь?
        area_col = (
            DEFAULT_AREA_COL
            if DEFAULT_AREA_COL in gdf.columns
            else "__area_tmp"
        )
        if area_col == "__area_tmp":
            gdf[area_col] = gdf.geometry.area

        # --- 2. есть ли «плоская» таблица с ip_type/ip_value?
        has_flat_ip = {
            DEFAULT_IP_TYPE,
            DEFAULT_IP_VALUE,
        }.issubset(gdf.columns)
        ip_type_col = DEFAULT_IP_TYPE if has_flat_ip else None
        ip_value_col = DEFAULT_IP_VALUE if has_flat_ip else None

        # --- 3. экономика → ECON_*
        gdf, summary = self._compute_economic(
            gdf, area_col, ip_type_col, ip_value_col
        )

        # --- 4. synth spatial+economic → INV_*
        gdf = self._compute_spatial_economic_combined(
            gdf, ip_type_col, ip_value_col
        )

        # --- 5. итоговый INV в summary
        summary = self._compute_summary_inv(summary)

        if area_col == "__area_tmp":
            gdf.drop(columns=[area_col], inplace=True)

        return gdf, summary

    # ------------------------------------------------------------------ #
    # в нутренние методы — расчёт экономических показателей
    # ------------------------------------------------------------------ #
    # ↓↓↓ отдельными функциями-утилитами, чтобы легче тестировать ↓↓↓
    def _make_cashflow(
        self, lu: str, land_area: float, profile: Dict[str, Any]
    ) -> List[float]:
        """
        Строит дисконтированный cash-flow по профилю ``lu`` на площадь ``land_area``.
        """
        density: float = profile["density"]
        gfa = land_area * density

        # CAPEX
        land_cost = land_area * profile.get("land_cost", 0)
        build_cost = gfa * profile["cost_build"]
        years_build = profile.get("construction_years", 2)
        capex_per_year = build_cost / years_build

        cf: List[float] = [-land_cost - capex_per_year] + [
            -capex_per_year
        ] * (years_build - 1)

        # OPEX + revenue
        opex = profile.get("opex_rate", 0) * gfa
        if "price_sale" in profile:
            # продажа
            yrs = profile.get("sale_years", 3)
            rev_total = gfa * profile["price_sale"]
            rev_per_year = rev_total / yrs
            cf.extend(rev_per_year - opex for _ in range(yrs))
        elif "rent_annual" in profile:
            # аренда
            yrs = profile.get("rent_years", 10)
            occ = profile.get("occupancy", 0.9)
            rev_per_year = gfa * profile["rent_annual"] * occ
            cf.extend(rev_per_year - opex for _ in range(yrs))
        else:
            raise ValueError(f"Profile '{lu}' needs price_sale or rent_annual")
        return cf

    # ------------------------------------------------------------------ #
    def _compute_economic(
        self,
        gdf: gpd.GeoDataFrame,
        area_col: str,
        ip_type_col: str | None,
        ip_value_col: str | None,
    ) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """
        Добавляет ECON_* колонки в gdf и формирует DataFrame summary.
        """
        rows: list[dict[str, Any]] = []

        for lu, prof in self.benchmarks.items():
            ip_full = f"ИП_{lu}"

            # --- какая площадь попадает под профиль lu?
            land_area = (
                gdf.loc[gdf[ip_type_col] == ip_full, area_col].sum()  # type: ignore[index]
                if ip_type_col
                else gdf[area_col].sum()
            )

            cf_raw = self._make_cashflow(lu, land_area, prof)
            rate = prof.get("discount_rate", self.discount_rate)

            rawnpv = npv(rate, cf_raw)
            npv_val = quantize(rawnpv)
            irr_val = irr(cf_raw)
            roi_val = (
                sum(cf_raw[1:]) / -cf_raw[0] if cf_raw and cf_raw[0] < 0 else np.nan
            )
            pp_val = payback_period(rate, cf_raw)

            # Economic Index (EI)
            ei_val = self._economic_index(rawnpv, irr_val, rate)

            for k, v in (
                ("NPV", npv_val),
                ("IRR", irr_val),
                ("ROI", roi_val),
                ("PP", pp_val),
                ("EI", ei_val),
            ):
                gdf[f"ECON_{k}_{lu}"] = v
            inv_attr = (
                gdf[ip_full].mean()
                if ip_full in gdf.columns
                else gdf.loc[gdf[ip_type_col] == ip_full, ip_value_col].mean()  # type: ignore[index]
                if ip_type_col
                else np.nan
            )

            rows.append(
                {
                    "profile": lu,
                    "NPV": npv_val,
                    "IRR": irr_val,
                    "ROI": roi_val,
                    "PP_years": pp_val,
                    "EI": ei_val,
                    "investment_attractiveness": inv_attr,
                }
            )

        summary = pd.DataFrame(rows).set_index("profile")

        # -------------------------------------------------------------- #
        # Project-level агрегирование
        # -------------------------------------------------------------- #
        if len(gdf) > 1:
            agg_cf = self._aggregate_project_cashflows(
                gdf, area_col, ip_type_col, self.benchmarks
            )
            rawnpv_proj = npv(self.discount_rate, agg_cf)
            irr_proj = irr(agg_cf)
            roi_proj = (
                sum(agg_cf[1:]) / -agg_cf[0] if agg_cf and agg_cf[0] < 0 else np.nan
            )
            pp_proj = payback_period(self.discount_rate, agg_cf)
            ei_proj = self._economic_index(rawnpv_proj, irr_proj, self.discount_rate)
            inv_attr_proj = (
                gdf[DEFAULT_IP_VALUE].mean() if ip_value_col else np.nan
            )

            summary.loc["project"] = {
                "NPV": quantize(rawnpv_proj),
                "IRR": irr_proj,
                "ROI": roi_proj,
                "PP_years": pp_proj,
                "EI": ei_proj,
                "investment_attractiveness": inv_attr_proj,
            }

        summary[["IRR", "ROI", "PP_years", "EI"]] = summary[
            ["IRR", "ROI", "PP_years", "EI"]
        ].round(2)
        return gdf, summary

    # ------------------------------------------------------------------ #
    # spatial + economic synthesis
    # ------------------------------------------------------------------ #
    def _compute_spatial_economic_combined(
        self,
        gdf: gpd.GeoDataFrame,
        ip_type_col: str | None,
        ip_value_col: str | None,
    ) -> gpd.GeoDataFrame:
        """
        Расчёт ``INV_<lu>`` для каждого land_use.
        """
        # соберём «сырые» величины для нормировки
        spat_vals: list[float] = []
        econ_vals: list[float] = []

        def _spatial_series(lu: str) -> pd.Series:
            ip_col = f"ИП_{lu}"
            if ip_col in gdf.columns:
                return gdf[ip_col].astype(float)
            if ip_type_col and ip_value_col:
                mask = gdf[ip_type_col] == ip_col  # type: ignore[index]
                out = pd.Series(0.0, index=gdf.index)
                out[mask] = gdf.loc[mask, ip_value_col].astype(float)  # type: ignore[index]
                return out
            return pd.Series(0.0, index=gdf.index)

        for lu in self.weights:
            s_raw = _spatial_series(lu)
            e_raw = gdf[f"ECON_{self.metric}_{lu}"].astype(float)
            spat_vals.extend(s_raw.tolist())
            econ_vals.extend(e_raw.tolist())

        # глобальные min/max
        s_min, s_max = _nanminmax(spat_vals)
        e_min, e_max = _nanminmax(econ_vals)

        # для метрик с плюсом/минусом возьмём симметрию
        if self.metric in ("NPV", "ROI"):
            e_abs = max(abs(e_min), abs(e_max))
            e_min, e_max = -e_abs, e_abs

        # --- нормировка + свёртка
        for lu, (w_s, w_e) in self.weights.items():
            s_raw = _spatial_series(lu)
            e_raw = gdf[f"ECON_{self.metric}_{lu}"].astype(float)

            s_norm = _normalize_series(s_raw, s_min, s_max)
            e_norm = _normalize_series(e_raw, e_min, e_max)

            gdf[f"INV_{lu}"] = (w_s * s_norm + w_e * e_norm).round(2)
        return gdf

    # ------------------------------------------------------------------ #
    # таблица summary: общий INV
    # ------------------------------------------------------------------ #
    def _compute_summary_inv(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Итоговая колонка ``INV`` в таблице summary (по профилям и проекту).
        """
        df = summary_df.copy()

        s = _to_numeric(df["investment_attractiveness"])
        e = _to_numeric(df[self.metric])

        s_min, s_max = s.min(), s.max()
        e_min, e_max = e.min(), e.max()
        if self.metric in ("NPV", "ROI"):
            e_abs = max(abs(e_min), abs(e_max))
            e_min, e_max = -e_abs, e_abs

        s_norm = _normalize_series(s, s_min, s_max)
        e_norm = _normalize_series(e, e_min, e_max)

        ws = pd.Series({lu: w_s for lu, (w_s, _) in self.weights.items()})
        we = pd.Series({lu: w_e for lu, (_, w_e) in self.weights.items()})
        # веса для «project» — средние
        ws["project"], we["project"] = ws.mean(), we.mean()

        df["INV"] = (ws * s_norm + we * e_norm).round(2)
        return df

    # ------------------------------------------------------------------ #
    # вспомогательные «маленькие» методы
    # ------------------------------------------------------------------ #
    @staticmethod
    def _economic_index(
        npv_val: float | None, irr_val: float | None, rate: float
    ) -> float:
        """
        Переводим (NPV, IRR) → EI ∈ [0; 100].
        """
        ei = 0.0
        if npv_val is not None:
            arg = -npv_val / 1e8
            # защищаемся от переполнения в exp()
            sig = (
                0.0
                if arg > 700
                else 1.0
                if arg < -700
                else 1.0 / (1 + math.exp(arg))
            )
            ei += 50 * sig
        if irr_val is not None:
            ei += 50 * max(0.0, irr_val - rate) / (0.3 - rate)
        return round(min(max(ei, 0.0), 100.0), 4)

    @staticmethod
    def _aggregate_project_cashflows(
        gdf: gpd.GeoDataFrame,
        area_col: str,
        ip_type_col: str | None,
        benchmarks: Dict[str, Dict[str, Any]],
    ) -> List[float]:
        """
        Складываем CF всех участков проекта в один общий список.
        """
        all_cfs: list[list[float]] = []
        for _, row in gdf.iterrows():
            lu = (
                row.get(ip_type_col, "").replace("ИП_", "")  # type: ignore[arg-type]
                if ip_type_col
                else ""
            )
            if lu not in benchmarks:
                continue
            cf = InvestmentAttractivenessAnalyzer._make_cashflow(
                InvestmentAttractivenessAnalyzer,
                lu,
                float(row[area_col]),
                benchmarks[lu],
            )
            all_cfs.append(cf)

        max_len = max(len(cf) for cf in all_cfs)
        agg_cf = [
            sum(cf[t] if t < len(cf) else 0.0 for cf in all_cfs) for t in range(max_len)
        ]
        return agg_cf


# ====================================================================== #
#                    --- вспомогательные функции ---                    #
# ====================================================================== #
def _nanminmax(values: Iterable[float]) -> tuple[float, float]:
    """Безопасный (nan-aware) min/max."""
    arr = np.array(list(values), dtype=float)
    return float(np.nanmin(arr)), float(np.nanmax(arr))


def _normalize_series(
    s: pd.Series | np.ndarray | list[float], vmin: float, vmax: float
) -> pd.Series:
    """Нормировка в 0 – 100; если диапазон «сжат», возвращаем 0."""
    if vmax > vmin:
        return 100 * (pd.Series(s, dtype=float) - vmin) / (vmax - vmin)
    return pd.Series(0.0, index=pd.Index(range(len(s))))


def _to_numeric(obj: pd.Series) -> pd.Series:
    """«Мягко» переводит строковые значения (с пробелами, знаками %) в float."""
    return pd.to_numeric(
        obj.astype(str).str.replace(r"[^0-9\.\-]+", "", regex=True), errors="coerce"
    )
