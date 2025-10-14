import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from blocksnet.enums import LandUse

from .constants import DEFAULT_SER_PARAMETERS

class SEREstimator:
    """Socio-economic results estimator with configurable defaults.

    The estimator evaluates project contribution deltas for five headline
    indicators during the construction and operational phases. Baseline
    parameters (population, employment base) must be supplied, while all other
    knobs fall back to safe expert defaults—including ``avg_wage_base``—that
    can be overridden when needed.
    """

    # --- дефолтные параметры экспертных оценок ---
    DEFAULT_PARAMETERS = DEFAULT_SER_PARAMETERS

    def __init__(self, params: Dict[str, Any]):
        """Create a new estimator with project-specific configuration.

        Parameters
        ----------
        params : dict
            Configuration dictionary. Must include the keys ``population``
            and ``employment_base``. Any default stored in
            ``DEFAULT_SER_PARAMETERS`` (including ``avg_wage_base``) can be
            overridden by providing the
            corresponding key (nested dictionaries are merged recursively).

        Raises
        ------
        ValueError
            If any of the mandatory baseline keys are absent.
        """
        need = ['population', 'employment_base']
        missing = [k for k in need if k not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
        # слить дефолты и пользовательские параметры
        defaults = self.DEFAULT_PARAMETERS
        cfg = {**defaults, **params}
        # вложенные словари ставок/коэффов тоже аккуратно слить
        for key in ['tax_rates','va_coeff_build','va_per_m2_ops','jobs_per_m2',
                    'wage_by_use','profit_share_ops','capex_capitalizable_share',
                    'amortization_rates']:
            if key in params and isinstance(params[key], dict):
                cfg[key] = {**defaults.get(key, {}), **params[key]}
        self.cfg = cfg

    @staticmethod
    def _normalise_land_use_label(value: Any) -> str:
        """Map raw land-use tokens to canonical enum values when possible."""
        if isinstance(value, LandUse):
            return value.value
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return text
        try:
            return LandUse(text).value
        except ValueError:
            pass
        name = text.upper()
        try:
            return LandUse[name].value
        except KeyError:
            pass
        if "." in name:
            suffix = name.split(".")[-1]
            try:
                return LandUse[suffix].value
            except KeyError:
                return text
        return text

    @classmethod
    def _get(cls, mp: Dict[str, float], key: Any, default: float = 0.0) -> float:
        norm_key = cls._normalise_land_use_label(key)
        return float(mp.get(norm_key, mp.get('default', default)))

    @staticmethod
    def _format_with_space_grouping(value: float, decimals: int) -> str:
        """Return a decimal string with thin-space thousand separators."""
        formatted = format(value, f"_.{decimals}f")
        return formatted.replace("_", " ")

    @staticmethod
    def _fmt(v: Optional[float]) -> str:
        if v is None or (isinstance(v, float) and (np.isnan(v))):
            return ""
        v = float(v)
        if abs(v) >= 100:
            return SEREstimator._format_with_space_grouping(v, 0)
        elif abs(v) >= 1:
            return SEREstimator._format_with_space_grouping(v, 2)
        else:
            return SEREstimator._format_with_space_grouping(v, 4)

    def compute(self, df: pd.DataFrame, pretty: bool = True) -> pd.DataFrame:
        """Calculate socio-economic deltas for construction and operation.

        Parameters
        ----------
        df : pandas.DataFrame
            Project blocks with at least ``land_use``, ``built_area`` and
            ``investment_need`` columns. Optional land valuation columns are
            recognised when present.
        pretty : bool, default ``True``
            If ``True``, format numeric results as human-readable strings.

        Returns
        -------
        pandas.DataFrame
            Table with columns ``indicator``, ``delta_build_year`` and
            ``delta_ops_year`` containing the five key metrics.
        """
        c = self.cfg
        P        = int(c['population'])
        Emp_base = int(c['employment_base'])
        W_base   = float(c['avg_wage_base'])

        T_build  = int(c.get('build_years', 1))
        pit = float(c['tax_rates'].get('pit', 0.13))
        cit = float(c['tax_rates'].get('cit', 0.17))
        prop = float(c['tax_rates'].get('prop', 0.02))
        land_tax = float(c['tax_rates'].get('land', 0.0))

        d = df.copy()
        numeric_cols = ['built_area', 'investment_need', 'land_cost', 'land_cost_before']
        for col in numeric_cols:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors='coerce').fillna(0.0)
        if 'land_cost' in d.columns and 'land_cost_before' not in d.columns:
            # если старой цены нет, считаем её равной текущей
            d['land_cost_before'] = d['land_cost']
        d['land_use'] = d['land_use'].apply(self._normalise_land_use_label)

        agg_spec = {
            'I': ('investment_need', 'sum'),
            'A': ('built_area', 'sum')
        }
        if 'land_cost' in d.columns:
            agg_spec['land_cost_after'] = ('land_cost', 'sum')
        if 'land_cost_before' in d.columns:
            agg_spec['land_cost_before'] = ('land_cost_before', 'sum')
        g = d.groupby('land_use', dropna=False).agg(**agg_spec).reset_index()

        # 1) Δ инвестиции на душу (за период стройки)
        I_total = g['I'].sum()
        delta_invcap_pc_build = I_total / max(P, 1)

        # 2) Δ ВРП/душу — стройка
        g['k_va_build'] = g['land_use'].apply(lambda u: self._get(c['va_coeff_build'], u, 0.0))
        VA_build_annual = (g['I'] * g['k_va_build']).sum() / max(T_build, 1)
        delta_grp_pc_build = VA_build_annual / max(P, 1)

        # 2) Δ ВРП/душу — эксплуатация
        g['y_m2'] = g['land_use'].apply(lambda u: self._get(c['va_per_m2_ops'], u, 0.0))
        VA_ops_annual = (g['A'] * g['y_m2']).sum()
        delta_grp_pc_ops = VA_ops_annual / max(P, 1)

        # 3) Δ Доходы бюджета — стройка
        W_build_annual   = (I_total / max(T_build,1)) * float(c['build_wage_share'])
        PIT_build_annual = W_build_annual * 12 * pit
        CIT_build_annual = (I_total / max(T_build,1)) * float(c['build_profit_margin']) * cit
        delta_budget_build = PIT_build_annual + CIT_build_annual

        # 3) Δ Доходы бюджета — эксплуатация
        g['jobs_m2'] = g['land_use'].apply(lambda u: self._get(c['jobs_per_m2'], u, 0.0))
        g['jobs']    = g['A'] * g['jobs_m2']
        g['wage']    = g['land_use'].apply(lambda u: self._get(c['wage_by_use'], u, 0.0))
        PIT_ops_annual = (g['jobs'] * g['wage'] * 12).sum() * pit

        g['profit_sh'] = g['land_use'].apply(lambda u: self._get(c['profit_share_ops'], u, 0.0))
        CIT_ops_annual = (g['A'] * g['y_m2'] * g['profit_sh']).sum() * cit

        g['cap_share'] = g['land_use'].apply(lambda u: self._get(c['capex_capitalizable_share'], u, 1.0))
        FA_add = (g['I'] * g['cap_share']).sum()
        Property_tax_annual = FA_add * prop

        land_tax_delta = 0.0
        if land_tax and 'land_cost_after' in g.columns:
            land_after_total = g['land_cost_after'].sum()
            land_before_total = g['land_cost_before'].sum() if 'land_cost_before' in g.columns else 0.0
            land_tax_delta = (land_after_total - land_before_total) * land_tax

        delta_budget_ops = PIT_ops_annual + CIT_ops_annual + Property_tax_annual
        if land_tax_delta:
            delta_budget_ops += land_tax_delta

        # 4) Δ Средняя зарплата — меняется в эксплуатации
        Jobs_new = g['jobs'].sum()
        if Emp_base + Jobs_new > 0:
            W_new = (W_base * Emp_base + (g['jobs'] * g['wage']).sum()) / (Emp_base + Jobs_new)
            delta_wage = W_new - W_base
        else:
            delta_wage = 0.0

        # 5) Δ Износ (тыс. руб.) — годовая амортизация новых ОС
        g['a'] = g['land_use'].apply(lambda u: self._get(c['amortization_rates'], u, 0.03))
        Dep_add_annual = (g['I'] * g['cap_share'] * g['a']).sum()
        delta_wear_thousand_rub = Dep_add_annual / 1_000.0

        out = pd.DataFrame([
            ["Объём инвестиций в основной капитал на душу населения",
             delta_invcap_pc_build, np.nan],
            ["Валовый региональный продукт на душу населения",
             delta_grp_pc_build,   delta_grp_pc_ops],
            ["Доходы бюджета территории",
             delta_budget_build,   delta_budget_ops],
            ["Средний уровень заработной платы",
             np.nan,               delta_wage],
            ["Износ основного фонда (тыс. руб.)",
             np.nan,               delta_wear_thousand_rub],
        ], columns=['indicator','delta_build_year','delta_ops_year'])

        if not pretty:
            return out

        for col in ['delta_build_year','delta_ops_year']:
            out[col] = out[col].map(self._fmt)

        return out
