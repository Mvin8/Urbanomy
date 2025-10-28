from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from blocksnet.enums import LandUse

from urbanomy.methods.socio_economic_indicators.constants import (
    DEFAULT_AMORTIZATION_RATES,
    DEFAULT_JOBS_PER_M2,
    DEFAULT_PROFIT_SHARE_OPS,
    DEFAULT_SER_PARAMETERS,
    DEFAULT_TAX_RATES,
    DEFAULT_VA_PER_M2_OPS,
    DEFAULT_WAGE_BY_USE,
    FALLBACK_CAPEX_CAPITALIZABLE_SHARE,
)

class SEREstimator:
    """Socio-economic results estimator with configurable defaults.

    Notes
    -----
    The estimator evaluates project contribution deltas for five headline
    indicators during the construction and operational phases. Baseline
    parameters (population, employment base, build years) must be supplied,
    while the remaining parameters fall back to expert defaults—including
    ``avg_wage_base``—that end users may override.
    """

    # --- default expert assessment parameters ---
    DEFAULT_PARAMETERS = DEFAULT_SER_PARAMETERS

    @classmethod
    def _merge_parameters(cls, defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user overrides into default parameters.

        Parameters
        ----------
        defaults : dict
            Baseline configuration values.
        overrides : dict
            User-specified parameters that take precedence over defaults.

        Returns
        -------
        dict
            A merged mapping containing defaults amended with overrides.
        """
        merged: Dict[str, Any] = deepcopy(defaults)
        for key, value in overrides.items():
            default_value = merged.get(key)
            if isinstance(value, dict) and isinstance(default_value, dict):
                merged[key] = {**default_value, **value}
            else:
                merged[key] = value
        return merged

    def __init__(self, params: Dict[str, Any]):
        """Create a new estimator with project-specific configuration.

        Parameters
        ----------
        params : dict
            Configuration dictionary. Must include the keys ``population``,
            ``employment_base`` and ``build_years``. Any default stored in
            ``DEFAULT_SER_PARAMETERS`` (including ``avg_wage_base``) can be
            overridden by providing the corresponding key. Nested dictionaries
            are merged recursively.

        Raises
        ------
        ValueError
            If any of the mandatory baseline keys are absent.
        """
        need = ['population', 'employment_base', 'build_years']
        missing = [k for k in need if k not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
        # merge defaults and user-specified parameters
        self.cfg = self._merge_parameters(self.DEFAULT_PARAMETERS, params)

    @staticmethod
    def _normalise_land_use_label(value: Any) -> str:
        """Convert raw land-use tokens to canonical enum values.

        Parameters
        ----------
        value : Any
            Raw land-use entry (enum, string, number, etc.).

        Returns
        -------
        str
            Canonical land-use label recognised by downstream mappings.
        """
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
        """Resolve a mapping value using a land-use key.

        Parameters
        ----------
        mp : dict
            Mapping from land-use labels to coefficients.
        key : Any
            Raw land-use identifier.
        default : float, optional
            Fallback value when the key is absent, by default 0.0.

        Returns
        -------
        float
            Coefficient retrieved from the mapping or default.
        """
        norm_key = cls._normalise_land_use_label(key)
        return float(mp.get(norm_key, mp.get('default', default)))

    def _prepare_land_use_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate project blocks by land-use class.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw block-level data with land-use descriptors and metrics.

        Returns
        -------
        pandas.DataFrame
            Aggregated land-use table with totals for areas, investments and
            optional land-cost information.
        """
        data = df.copy()
        numeric_cols = ['built_area', 'investment_need', 'land_cost', 'land_cost_before']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0.0)
        if 'land_cost' in data.columns and 'land_cost_before' not in data.columns:
            data['land_cost_before'] = data['land_cost']
        data['land_use'] = data['land_use'].apply(self._normalise_land_use_label)

        aggregation: Dict[str, Any] = {
            'I': ('investment_need', 'sum'),
            'A': ('built_area', 'sum'),
        }
        if 'land_cost' in data.columns:
            aggregation['land_cost_after'] = ('land_cost', 'sum')
        if 'land_cost_before' in data.columns:
            aggregation['land_cost_before'] = ('land_cost_before', 'sum')

        return (
            data
            .groupby('land_use', dropna=False)
            .agg(**aggregation)
            .reset_index()
        )

    def _assign_land_use_metric(
        self,
        frame: pd.DataFrame,
        target_col: str,
        mapping: Optional[Dict[str, float]],
        default_value: float,
    ) -> None:
        """Populate a per-land-use metric column.

        Parameters
        ----------
        frame : pandas.DataFrame
            Aggregated land-use table produced by :meth:`_prepare_land_use_aggregates`.
        target_col : str
            Column name that will receive resolved coefficients.
        mapping : dict or None
            Known coefficients keyed by land-use labels.
        default_value : float
            Fallback coefficient used when the label is absent in ``mapping``.
        """
        mapping = mapping or {}
        frame[target_col] = frame['land_use'].apply(
            lambda u: self._get(mapping, u, default_value)
        )

    @staticmethod
    def _land_tax_delta(aggregated: pd.DataFrame, rate: float) -> float:
        """Compute the change in land tax revenues.

        Parameters
        ----------
        aggregated : pandas.DataFrame
            Land-use aggregates containing land-cost totals.
        rate : float
            Land tax rate for the jurisdiction.

        Returns
        -------
        float
            Incremental land tax contributed by the project.
        """
        if not rate or 'land_cost_after' not in aggregated.columns:
            return 0.0
        land_after_total = aggregated['land_cost_after'].sum()
        land_before_total = aggregated['land_cost_before'].sum() if 'land_cost_before' in aggregated.columns else 0.0
        return (land_after_total - land_before_total) * rate

    @staticmethod
    def _format_with_space_grouping(value: float, decimals: int) -> str:
        """Format a number with thin-space thousands separators.

        Parameters
        ----------
        value : float
            Raw numeric value to format.
        decimals : int
            Number of decimal places to retain.

        Returns
        -------
        str
            Human-readable representation of the numeric value.
        """
        formatted = format(value, f"_.{decimals}f")
        return formatted.replace("_", " ")

    @staticmethod
    def _fmt(v: Optional[float]) -> str:
        """Render numeric values in a compact human-readable form.

        Parameters
        ----------
        v : float or None
            Raw numeric outcome.

        Returns
        -------
        str
            Formatted string or an empty string when the value is not defined.
        """
        if v is None or (isinstance(v, float) and (np.isnan(v))):
            return ""
        v = float(v)
        if abs(v) >= 100:
            return SEREstimator._format_with_space_grouping(v, 0)
        elif abs(v) >= 1:
            return SEREstimator._format_with_space_grouping(v, 2)
        else:
            return SEREstimator._format_with_space_grouping(v, 4)

    def _format_pretty(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply human-friendly formatting to result columns.

        Parameters
        ----------
        df : pandas.DataFrame
            Computed indicators in numeric form.

        Returns
        -------
        pandas.DataFrame
            Same structure with formatted string values.
        """
        formatted = df.copy()
        for col in ['delta_build_year', 'delta_ops_year']:
            formatted[col] = formatted[col].map(self._fmt)
        return formatted

    def compute(self, df: pd.DataFrame, pretty: bool = True) -> pd.DataFrame:
        """Calculate socio-economic deltas for construction and operation.

        Parameters
        ----------
        df : pandas.DataFrame
            Project blocks with at least ``land_use``, ``built_area`` and
            ``investment_need`` columns. Optional land valuation columns are
            recognised when present.
        pretty : bool, optional
            Format numeric results as human-readable strings. The default is
            ``True``.

        Returns
        -------
        pandas.DataFrame
            Table with columns ``indicator``, ``delta_build_year`` and
            ``delta_ops_year`` containing the five key metrics.
        """
        c = self.cfg
        population = int(c['population'])
        employment_base = int(c['employment_base'])
        wage_base = float(c['avg_wage_base'])

        build_years = int(c['build_years'])
        build_years_safe = build_years if build_years > 0 else 1
        population_safe = population if population > 0 else 1
        pit = float(c['tax_rates'].get('pit', DEFAULT_TAX_RATES['pit']))
        cit = float(c['tax_rates'].get('cit', DEFAULT_TAX_RATES['cit']))
        prop = float(c['tax_rates'].get('prop', DEFAULT_TAX_RATES['prop']))
        land_tax = float(c['tax_rates'].get('land', DEFAULT_TAX_RATES.get('land', 0.0)))

        g = self._prepare_land_use_aggregates(df)

        # Metric 1: Delta investment in fixed capital per capita during construction
        i_total = g['I'].sum()
        delta_invcap_pc_build = i_total / population_safe

        # Metric 2: Delta gross regional product per capita during construction
        self._assign_land_use_metric(g, 'k_va_build', c.get('va_coeff_build', {}), 0.0)
        va_build_annual = (g['I'] * g['k_va_build']).sum() / build_years_safe
        delta_grp_pc_build = va_build_annual / population_safe

        # Metric 2 continued: Delta gross regional product per capita during operations
        self._assign_land_use_metric(g, 'y_m2', c.get('va_per_m2_ops', {}), DEFAULT_VA_PER_M2_OPS['default'])
        va_ops_annual = (g['A'] * g['y_m2']).sum()
        delta_grp_pc_ops = va_ops_annual / population_safe

        # Metric 3: Delta budget revenues during construction
        investment_annual = i_total / build_years_safe
        w_build_annual = investment_annual * float(c['build_wage_share'])
        pit_build_annual = w_build_annual * 12 * pit
        cit_build_annual = investment_annual * float(c['build_profit_margin']) * cit
        delta_budget_build = pit_build_annual + cit_build_annual

        # Metric 3 continued: Delta budget revenues during operations
        self._assign_land_use_metric(g, 'jobs_m2', c.get('jobs_per_m2', {}), DEFAULT_JOBS_PER_M2['default'])
        g['jobs'] = g['A'] * g['jobs_m2']
        self._assign_land_use_metric(g, 'wage', c.get('wage_by_use', {}), DEFAULT_WAGE_BY_USE['default'])
        pit_ops_annual = (g['jobs'] * g['wage'] * 12).sum() * pit

        self._assign_land_use_metric(g, 'profit_sh', c.get('profit_share_ops', {}), DEFAULT_PROFIT_SHARE_OPS['default'])
        cit_ops_annual = (g['A'] * g['y_m2'] * g['profit_sh']).sum() * cit

        self._assign_land_use_metric(
            g,
            'cap_share',
            c.get('capex_capitalizable_share', {}),
            FALLBACK_CAPEX_CAPITALIZABLE_SHARE,
        )
        fa_add = (g['I'] * g['cap_share']).sum()
        property_tax_annual = fa_add * prop

        land_tax_delta = self._land_tax_delta(g, land_tax)
        delta_budget_ops = pit_ops_annual + cit_ops_annual + property_tax_annual + land_tax_delta

        # Metric 4: Delta average wage during operations
        jobs_new = g['jobs'].sum()
        workforce_total = employment_base + jobs_new
        if workforce_total > 0:
            w_new = (wage_base * employment_base + (g['jobs'] * g['wage']).sum()) / workforce_total
            delta_wage = w_new - wage_base
        else:
            delta_wage = 0.0

        # Metric 5: Delta depreciation expense (thousand RUB) for new fixed assets
        self._assign_land_use_metric(g, 'a', c.get('amortization_rates', {}), DEFAULT_AMORTIZATION_RATES['default'])
        dep_add_annual = (g['I'] * g['cap_share'] * g['a']).sum()
        delta_wear_thousand_rub = dep_add_annual / 1_000.0

        out = pd.DataFrame(
            [
                ["Объём инвестиций в основной капитал на душу населения", delta_invcap_pc_build, np.nan],
                ["Валовый региональный продукт на душу населения", delta_grp_pc_build, delta_grp_pc_ops],
                ["Доходы бюджета территории", delta_budget_build, delta_budget_ops],
                ["Средний уровень заработной платы", np.nan, delta_wage],
                ["Износ основного фонда (тыс. руб.)", np.nan, delta_wear_thousand_rub],
            ],
            columns=['indicator', 'delta_build_year', 'delta_ops_year'],
        )

        return self._format_pretty(out) if pretty else out
