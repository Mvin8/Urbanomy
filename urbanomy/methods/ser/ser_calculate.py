import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class SEREstimator:
    """
    Минимальная модель СЭР: считает вклад проекта (дельты) по 5 показателям.
    Нужны базовые параметры: population, employment_base, avg_wage_base.
    Остальное имеет безопасные дефолты и может быть переопределено.
    """

    # --- дефолтные параметры экспертных оценок ---
    _DEF = {
        'build_years': 3,
        'tax_rates': {'pit': 0.13, 'cit': 0.17, 'prop': 0.02, 'land': 0.015},
        'va_coeff_build': {
            'BUSINESS': 0.50, 'RESIDENTIAL': 0.45, 'TRANSPORT': 0.55,
            'AGRICULTURE': 0.45, 'SPECIAL': 0.50, 'default': 0.50
        },
        'va_per_m2_ops': {
            'BUSINESS': 12000.0, 'RESIDENTIAL': 2000.0, 'TRANSPORT': 5000.0,
            'AGRICULTURE': 3000.0, 'SPECIAL': 7000.0, 'default': 0.0
        },
        'jobs_per_m2': {
            'BUSINESS': 1/18, 'RESIDENTIAL': 0.0, 'TRANSPORT': 1/90,
            'AGRICULTURE': 1/400, 'SPECIAL': 1/30, 'default': 0.0
        },
        'wage_by_use': {
            'BUSINESS': 85_000, 'RESIDENTIAL': 0, 'TRANSPORT': 60_000,
            'AGRICULTURE': 45_000, 'SPECIAL': 75_000, 'default': 0
        },
        'profit_share_ops': {
            'BUSINESS': 0.18, 'TRANSPORT': 0.10, 'AGRICULTURE': 0.08,
            'SPECIAL': 0.15, 'RESIDENTIAL': 0.00, 'default': 0.12
        },
        'capex_capitalizable_share': {
            'BUSINESS': 0.95, 'RESIDENTIAL': 0.95, 'TRANSPORT': 0.90,
            'AGRICULTURE': 0.90, 'SPECIAL': 0.95, 'default': 0.95
        },
        'amortization_rates': {
            'BUSINESS': 0.03, 'RESIDENTIAL': 0.03, 'TRANSPORT': 0.06,
            'AGRICULTURE': 0.05, 'SPECIAL': 0.04, 'default': 0.03
        },
        'build_wage_share': 0.25,
        'build_profit_margin': 0.05
    }

    def __init__(self, params: Dict[str, Any]):
        """
        Обязательные ключи: population, employment_base, avg_wage_base.
        Любой дефолт из _DEF можно переопределить.
        """
        need = ['population', 'employment_base', 'avg_wage_base']
        missing = [k for k in need if k not in params]
        if missing:
            raise ValueError(f"Отсутствуют параметры: {missing}")
        # слить дефолты и пользовательские параметры
        cfg = {**self._DEF, **{k: v for k, v in params.items() if k in self._DEF or True}}
        # вложенные словари ставок/коэффов тоже аккуратно слить
        for key in ['tax_rates','va_coeff_build','va_per_m2_ops','jobs_per_m2',
                    'wage_by_use','profit_share_ops','capex_capitalizable_share',
                    'amortization_rates']:
            if key in params and isinstance(params[key], dict):
                cfg[key] = {**self._DEF.get(key, {}), **params[key]}
        self.cfg = cfg

    @staticmethod
    def _get(mp: Dict[str, float], key: str, default: float = 0.0) -> float:
        return float(mp.get(key, mp.get('default', default)))

    @staticmethod
    def _fmt(v: Optional[float]) -> str:
        if v is None or (isinstance(v, float) and (np.isnan(v))):
            return ""
        v = float(v)
        if abs(v) >= 100:
            return f"{v:,.0f}".replace(",", " ")
        elif abs(v) >= 1:
            return f"{v:,.2f}".replace(",", " ")
        else:
            return f"{v:,.4f}".replace(",", " ")

    def compute(self, df: pd.DataFrame, pretty: bool = True) -> pd.DataFrame:
        """
        Возвращает DataFrame:
        ['indicator','delta_build_year','delta_ops_year']
        Только дельты по 5 показателям.
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
        d['land_use'] = d['land_use'].astype(str)

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

        # человекочитаемый формат чисел
        for col in ['delta_build_year','delta_ops_year']:
            out[col] = out[col].map(self._fmt)

        return out
