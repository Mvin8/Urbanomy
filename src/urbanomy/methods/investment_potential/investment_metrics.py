"""Utilities for computing investment attractiveness metrics per polygon."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

from blocksnet.enums import LandUse

from urbanomy.utils.investment_input import (
    INVESTMENT_NUMERIC_COLUMNS,
    prepare_investment_input,
)

from .constants import (
    DEFAULT_DISCOUNT_RATE,
    DEFAULT_ECON_METRIC,
    DEFAULT_IP_VALUE,
    INVESTMENT_WEIGHTS,
    LAND_USE_CONFIGS,
    SUMMARY_COLUMNS,
)
from .utils_metrics import (
    economic_index,
    irr,
    make_cashflow,
    nanminmax,
    normalize_series,
    npv,
    payback_period,
    quantize,
)


@dataclass(frozen=True)
class InvestmentMetricsResult:
    """Calculated economic metrics for a cash-flow sequence."""

    npv: float
    irr: float
    roi: float
    payback_years: float
    economic_index: float

    @classmethod
    def from_cashflow(
        cls,
        cashflow: Sequence[float],
        discount_rate: float,
    ) -> "InvestmentMetricsResult":
        """Create an instance from a cash-flow sequence.

        Parameters
        ----------
        cashflow : Sequence[float]
            Ordered cash-flow values per period, including the initial
            investment.
        discount_rate : float
            Discount rate per period expressed as a decimal fraction.

        Returns
        -------
        InvestmentMetricsResult
            New instance populated with derived financial indicators.
        """
        cashflow = list(cashflow or [])

        raw_npv = npv(discount_rate, cashflow)
        quantized = quantize(raw_npv)
        npv_value = float(quantized) if quantized is not None else np.nan

        irr_raw = irr(cashflow)
        irr_value = float(irr_raw) if irr_raw is not None else np.nan

        roi_value = np.nan
        if cashflow and cashflow[0] < 0:
            denom = -cashflow[0]
            if denom:
                roi_value = sum(cashflow[1:]) / denom

        payback_raw = payback_period(discount_rate, cashflow)
        payback_value = float(payback_raw) if payback_raw is not None else np.nan

        ei_value = economic_index(raw_npv, irr_raw, discount_rate)

        return cls(
            npv=npv_value,
            irr=irr_value,
            roi=roi_value,
            payback_years=payback_value,
            economic_index=ei_value,
        )


@dataclass(frozen=True)
class PreparedProfile:
    """Resolved profile parameters ready for cash-flow computation."""

    params: dict[str, Any]
    land_area: float
    built_area: float
    gross_floor_area: float
    land_cost_total: float


@dataclass(frozen=True)
class RowComputation:
    """Computed financial metrics for an individual polygon."""

    index: Any
    land_use: str
    land_area: float
    built_area: float
    land_cost_total: float
    construction_cost: float
    investment_need: float
    cashflow: list[float]
    metrics: InvestmentMetricsResult


@dataclass(frozen=True)
class NormalizedMetricRange:
    """Normalized metric values alongside their source bounds."""

    values: pd.Series
    normalized: pd.Series
    minimum: float
    maximum: float

    @property
    def has_valid(self) -> bool:
        """Return True when at least one finite value is present."""
        return bool(self.values.notna().any())


class InvestmentAttractivenessAnalyzer:
    """Compute investment attractiveness metrics per polygon plus project summary."""

    def __init__(
        self,
        benchmarks: Mapping[str | LandUse, Mapping[str, Any]],
        weights_dict: Mapping[str | LandUse, Sequence[float]] | None = None,
        econ_metric: str = DEFAULT_ECON_METRIC,
        discount_rate: float | None = None,
    ) -> None:
        """Initialise the analyzer with reference profiles and weights.

        Parameters
        ----------
        benchmarks : Mapping[str | LandUse, Mapping[str, Any]]
            Mapping from land-use codes or ``LandUse`` enums to benchmark profiles describing
            profitability assumptions (densities, prices, etc.).
        weights_dict : Mapping[str | LandUse, Sequence[float]] or None, optional
            Optional override for spatial/economic weights per land-use key.
            Defaults to ``INVESTMENT_WEIGHTS`` when omitted.
        econ_metric : str, optional
            Economic metric to emphasise when normalising (``"EI"`` by default).
        discount_rate : float or None, optional
            Discount rate used when the benchmark profile does not specify one.
            If ``None``, ``DEFAULT_DISCOUNT_RATE`` is used.
        """
        self._benchmarks_enum = self._normalise_benchmarks(benchmarks)
        self.benchmarks = {
            land_use.value: dict(profile)
            for land_use, profile in self._benchmarks_enum.items()
        }
        enum_weights, plain_weights = self._normalise_weights(
            weights_dict or INVESTMENT_WEIGHTS
        )
        self._weights_enum = enum_weights
        self.weights = plain_weights
        self.metric = econ_metric.upper()
        self.discount_rate = discount_rate if discount_rate is not None else DEFAULT_DISCOUNT_RATE

    @staticmethod
    def _coerce_land_use(value: str | LandUse) -> LandUse:
        if isinstance(value, LandUse):
            return value
        try:
            return LandUse(str(value))
        except ValueError as exc:
            raise KeyError(f"Unknown land-use '{value}'") from exc

    def _normalise_benchmarks(
        self,
        benchmarks: Mapping[str | LandUse, Mapping[str, Any]] | None,
    ) -> dict[LandUse, dict[str, Any]]:
        """Convert benchmark keys to LandUse enum instances."""
        normalised: dict[LandUse, dict[str, Any]] = {}
        for key, profile in (benchmarks or {}).items():
            land_use = self._coerce_land_use(key)
            if not isinstance(profile, Mapping):
                raise ValueError(
                    f"Benchmark profile for '{land_use.value}' must be a mapping"
                )
            normalised[land_use] = dict(profile)
        return normalised

    def _normalise_weights(
        self,
        weights: Mapping[str | LandUse, Sequence[float]],
    ) -> tuple[dict[LandUse, tuple[float, float]], dict[str, tuple[float, float]]]:
        """Convert weights to LandUse-aware and string-keyed mappings."""
        enum_map: dict[LandUse, tuple[float, float]] = {}
        string_map: dict[str, tuple[float, float]] = {}
        for key, pair in weights.items():
            land_use = self._coerce_land_use(key)
            if not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError(
                    f"Weights for '{land_use.value}' must be a sequence of two values"
                )
            spatial, economic = float(pair[0]), float(pair[1])
            enum_map[land_use] = (spatial, economic)
            string_map[land_use.value] = (spatial, economic)
        for land_use in LAND_USE_CONFIGS:
            enum_map.setdefault(land_use, (0.5, 0.5))
            string_map.setdefault(land_use.value, (0.5, 0.5))
        return enum_map, string_map

    @staticmethod
    def _to_float(value: Any) -> float:
        """Convert arbitrary input to ``float`` returning ``nan`` on failure.

        Parameters
        ----------
        value : Any
            Value to convert.

        Returns
        -------
        float
            Converted float or ``nan`` if conversion fails.
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan

    @staticmethod
    def _round_clean(values: pd.Series | np.ndarray, decimals: int = 2) -> pd.Series:
        """Round numerical data and suppress near-zero artefacts.

        Parameters
        ----------
        values : pandas.Series or numpy.ndarray
            Values to round.
        decimals : int, optional
            Number of decimal places (default is ``2``).

        Returns
        -------
        pandas.Series
            Rounded values with tiny numbers coerced to ``0.0``.
        """
        serie = pd.Series(values, copy=True, dtype=float)
        serie = serie.round(decimals)
        tol = 10 ** (-decimals)
        serie[np.isclose(serie, 0.0, atol=tol, rtol=0.0)] = 0.0
        return serie

    @staticmethod
    def _coerce_numeric_columns(gdf: pd.DataFrame, columns: Sequence[str]) -> None:
        """Cast selected DataFrame columns to numeric values in-place.

        Parameters
        ----------
        gdf : pandas.DataFrame
            DataFrame whose columns are to be converted.
        columns : Sequence[str]
            Column names that should contain numeric data.
        """
        for col in columns:
            if col in gdf.columns:
                gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    def _prepare_profile(self, row: pd.Series, base_profile: dict[str, Any]) -> PreparedProfile:
        """Combine a row with a benchmark profile for cash-flow modelling.

        Parameters
        ----------
        row : pandas.Series
            Row with polygon attributes including geometry, areas and costs.
        base_profile : dict[str, Any]
            Benchmark parameters for the polygon's land-use type.

        Returns
        -------
        PreparedProfile
            Structured values ready for cash-flow generation.

        Raises
        ------
        ValueError
            If a valid land area cannot be derived from inputs.
        """
        params = dict(base_profile or {})

        land_area = self._to_float(row.get("site_area"))
        if not np.isfinite(land_area) or land_area <= 0:
            geom = row.get("geometry")
            land_area = float(getattr(geom, "area", math.nan))
        if not np.isfinite(land_area) or land_area <= 0:
            raise ValueError(f"Polygon {row.name} has no valid land area")

        land_cost_total = self._to_float(row.get("price_pred"))
        if np.isfinite(land_cost_total) and land_area > 0:
            params["land_cost"] = land_cost_total / land_area

        living = self._to_float(row.get("living_area"))
        non_living = self._to_float(row.get("non_living_area"))
        built_area = living + non_living
        if not np.isfinite(built_area) or built_area <= 0:
            built_area = self._to_float(row.get("build_floor_area"))

        if np.isfinite(built_area) and built_area > 0:
            params["built_area"] = built_area
        else:
            params.pop("built_area", None)
            built_area = math.nan

        if np.isfinite(built_area):
            gross_floor_area = built_area
        else:
            density = self._to_float(params.get("density"))
            gross_floor_area = (
                land_area * density if np.isfinite(density) and density > 0 else math.nan
            )

        share_val = self._to_float(row.get("share"))
        if np.isfinite(share_val) and share_val > 0:
            params["rent_share"] = max(0.0, min(share_val, 1.0))

        return PreparedProfile(
            params=params,
            land_area=land_area,
            built_area=built_area,
            gross_floor_area=gross_floor_area,
            land_cost_total=land_cost_total,
        )

    def _calculate_row_metrics(self, idx: Any, row: pd.Series) -> RowComputation:
        """Evaluate investment metrics for a single polygon row.

        Parameters
        ----------
        idx : Any
            Index label of the polygon.
        row : pandas.Series
            Polygon attributes enriched by prepared investment input.

        Returns
        -------
        RowComputation
            Computed metrics plus intermediate values for the polygon.

        Raises
        ------
        KeyError
            If the polygon's land-use lacks benchmark configuration.
        """
        land_use_raw = row.get("land_use")
        land_use_enum = self._coerce_land_use(land_use_raw)
        if land_use_enum not in self._benchmarks_enum:
            raise KeyError(f"No benchmark settings for land_use='{land_use_enum.value}'")

        profile = self._prepare_profile(row, self._benchmarks_enum[land_use_enum])
        cashflow = make_cashflow(land_use_enum.value, profile.land_area, profile.params)
        discount_rate = profile.params.get("discount_rate", self.discount_rate)

        metrics = InvestmentMetricsResult.from_cashflow(cashflow, discount_rate)

        cost_build_unit = self._to_float(profile.params.get("cost_build"))
        if np.isfinite(profile.gross_floor_area) and np.isfinite(cost_build_unit):
            construction_cost = profile.gross_floor_area * cost_build_unit
        else:
            construction_cost = math.nan

        finite_capex = [
            value
            for value in (profile.land_cost_total, construction_cost)
            if np.isfinite(value)
        ]
        investment_need = float(sum(finite_capex)) if finite_capex else math.nan

        return RowComputation(
            index=idx,
            land_use=land_use_enum.value,
            land_area=profile.land_area,
            built_area=profile.built_area,
            land_cost_total=profile.land_cost_total,
            construction_cost=construction_cost,
            investment_need=investment_need,
            cashflow=list(cashflow),
            metrics=metrics,
        )

    def _normalize_spatial_metric(self, price_series: pd.Series) -> NormalizedMetricRange:
        """Normalise land price proxies onto a 0-100 scale.

        Parameters
        ----------
        price_series : pandas.Series
            Series of land-price proxies for each polygon.

        Returns
        -------
        NormalizedMetricRange
            Original values, normalised scores, and their min/max bounds.
        """
        if price_series.notna().any():
            s_min, s_max = nanminmax(price_series[price_series.notna()])
            normalized = normalize_series(price_series.fillna(s_min), s_min, s_max).clip(0, 100)
        else:
            s_min = s_max = 0.0
            normalized = pd.Series(0.0, index=price_series.index)
        return NormalizedMetricRange(price_series, normalized, s_min, s_max)

    def _normalize_economic_metric(self, series: pd.Series) -> NormalizedMetricRange:
        """Normalise economic indicator values consistent with ``self.metric``.

        Parameters
        ----------
        series : pandas.Series
            Economic metric values for each polygon.

        Returns
        -------
        NormalizedMetricRange
            Original values, normalised scores, and computed bounds.
        """
        if self.metric == "EI":
            normalized = series.clip(lower=0, upper=100).fillna(0.0)
            return NormalizedMetricRange(series, normalized, math.nan, math.nan)

        valid = series.notna()
        if valid.any():
            e_min, e_max = nanminmax(series[valid])
            if self.metric in {"NPV", "ROI"}:
                bound = max(abs(e_min), abs(e_max))
                e_min, e_max = -bound, bound
            normalized = normalize_series(series.fillna(e_min), e_min, e_max).clip(0, 100)
        else:
            e_min = e_max = 0.0
            normalized = pd.Series(0.0, index=series.index)
        return NormalizedMetricRange(series, normalized, e_min, e_max)

    @staticmethod
    def _aggregate_cashflows(cashflows: Sequence[Sequence[float]]) -> list[float]:
        """Aggregate multiple cash-flow sequences period by period.

        Parameters
        ----------
        cashflows : sequence of sequence of float
            Cash-flow lists aligned per polygon.

        Returns
        -------
        list[float]
            Aggregate cash flow summing each period across sequences.
        """
        sequences = [list(cf) for cf in cashflows if cf]
        if not sequences:
            return []
        max_len = max(len(cf) for cf in sequences)
        return [
            sum(cf[i] if i < len(cf) else 0.0 for cf in sequences)
            for i in range(max_len)
        ]

    def _resolve_weights(
        self, land_use_series: pd.Series
    ) -> tuple[pd.Series, pd.Series, float, float]:
        """Map land-use categories to spatial/economic weights.

        Parameters
        ----------
        land_use_series : pandas.Series
            Series of land-use labels per polygon.

        Returns
        -------
        tuple[pandas.Series, pandas.Series, float, float]
            Spatial weights, economic weights, and their respective means.
        """
        spatial_lookup = {lu: weights[0] for lu, weights in self.weights.items()}
        economic_lookup = {lu: weights[1] for lu, weights in self.weights.items()}
        ws_mean = float(np.mean(list(spatial_lookup.values()))) if spatial_lookup else 0.5
        we_mean = float(np.mean(list(economic_lookup.values()))) if economic_lookup else 0.5
        spatial_weights = land_use_series.map(spatial_lookup).fillna(ws_mean)
        economic_weights = land_use_series.map(economic_lookup).fillna(we_mean)
        return spatial_weights, economic_weights, ws_mean, we_mean

    def calculate_investment_metrics(
        self,
        gdf: gpd.GeoDataFrame,
    ) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """Compute investment metrics and summary tables for polygons.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input dataset prepared with ``prepare_investment_input`` columns.

        Returns
        -------
        tuple[geopandas.GeoDataFrame, pandas.DataFrame]
            Tuple of per-polygon metrics (GeoDataFrame) and aggregated summary
            table (DataFrame). Returns empty structures if ``gdf`` is empty.

        Raises
        ------
        ValueError
            If the requested economic metric is missing from the prepared data.
        """
        if gdf.empty:
            return gdf.copy(), pd.DataFrame()

        working = prepare_investment_input(gdf)
        pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
        self._coerce_numeric_columns(working, INVESTMENT_NUMERIC_COLUMNS)

        row_results = [
            self._calculate_row_metrics(idx, row)
            for idx, row in working.iterrows()
        ]

        metrics_records = [
            {
                "_index": result.index,
                "ECON_NPV": result.metrics.npv,
                "ECON_IRR": result.metrics.irr,
                "ECON_ROI": result.metrics.roi,
                "ECON_PP_years": result.metrics.payback_years,
                "ECON_EI": result.metrics.economic_index,
                "land_area": result.land_area,
                "built_area": result.built_area,
                "land_cost": result.land_cost_total,
                "construction_cost": result.construction_cost,
                "investment_need": result.investment_need,
            }
            for result in row_results
        ]
        metrics_df = pd.DataFrame.from_records(metrics_records).set_index("_index")
        working = working.join(metrics_df)

        working["NPV"] = working["ECON_NPV"]
        working["IRR"] = working["ECON_IRR"]
        working["ROI"] = working["ECON_ROI"]
        working["PP_years"] = working["ECON_PP_years"]
        working["EI"] = working["ECON_EI"]

        price_series = (
            pd.to_numeric(working[DEFAULT_IP_VALUE], errors="coerce")
            if DEFAULT_IP_VALUE in working.columns
            else pd.Series(0.0, index=working.index)
        )
        spatial_range = self._normalize_spatial_metric(price_series)
        working["spatial_potential"] = spatial_range.values

        econ_metric_col = f"ECON_{self.metric}"
        if econ_metric_col not in working.columns:
            raise ValueError(f"Economic metric '{self.metric}' is not available")
        econ_values = pd.to_numeric(working[econ_metric_col], errors="coerce")
        economic_range = self._normalize_economic_metric(econ_values)

        spatial_weights, economic_weights, ws_mean, we_mean = self._resolve_weights(working["land_use"])
        working["INV"] = (
            spatial_weights * spatial_range.normalized
            + economic_weights * economic_range.normalized
        ).round(2)

        summary = pd.DataFrame(
            {
                "land_use": working["land_use"],
                "land_area": working["land_area"],
                "built_area": working["built_area"],
                "land_cost": working["land_cost"],
                "construction_cost": working["construction_cost"],
                "investment_need": working["investment_need"],
                "NPV": working["NPV"],
                "IRR": working["IRR"],
                "ROI": working["ROI"],
                "PP_years": working["PP_years"],
                "EI": working["EI"],
                "spatial_potential": working["spatial_potential"],
                "INV": working["INV"],
            },
            index=working.index,
        )

        project_cf = self._aggregate_cashflows(result.cashflow for result in row_results)
        project_metrics = (
            InvestmentMetricsResult.from_cashflow(project_cf, self.discount_rate)
            if project_cf
            else None
        )

        total_area = working["land_area"].sum(skipna=True)
        total_built = working["built_area"].sum(skipna=True)
        total_land_cost = working["land_cost"].sum(skipna=True)
        total_construction_cost = working["construction_cost"].sum(skipna=True)
        total_investment_need = working["investment_need"].sum(skipna=True)

        s_project = np.nan
        if total_area > 0 and price_series.notna().any():
            weighted = price_series * working["land_area"]
            s_project = float(weighted.sum(skipna=True) / total_area)

        e_project_val = np.nan
        e_norm_project = 0.0
        if project_metrics is not None:
            if self.metric == "EI":
                e_project_val = project_metrics.economic_index
                if np.isfinite(e_project_val):
                    e_norm_project = float(np.clip(e_project_val, 0, 100))
            else:
                metric_map = {
                    "NPV": project_metrics.npv,
                    "IRR": project_metrics.irr,
                    "ROI": project_metrics.roi,
                    "PP_years": project_metrics.payback_years,
                }
                e_project_val = metric_map.get(self.metric, project_metrics.npv)
                if (
                    np.isfinite(e_project_val)
                    and economic_range.has_valid
                    and economic_range.maximum > economic_range.minimum
                ):
                    e_norm_project = float(
                        np.clip(
                            100
                            * (e_project_val - economic_range.minimum)
                            / (economic_range.maximum - economic_range.minimum),
                            0,
                            100,
                        )
                    )

        if (
            np.isfinite(s_project)
            and spatial_range.has_valid
            and spatial_range.maximum > spatial_range.minimum
        ):
            s_norm_project = float(
                np.clip(
                    100
                    * (s_project - spatial_range.minimum)
                    / (spatial_range.maximum - spatial_range.minimum),
                    0,
                    100,
                )
            )
        else:
            s_norm_project = 0.0

        inv_project = ws_mean * s_norm_project + we_mean * e_norm_project

        summary.loc["project_total"] = {
            "land_use": "project",
            "land_area": total_area,
            "built_area": total_built,
            "land_cost": total_land_cost,
            "construction_cost": total_construction_cost,
            "investment_need": total_investment_need,
            "NPV": project_metrics.npv if project_metrics is not None else np.nan,
            "IRR": project_metrics.irr if project_metrics is not None else np.nan,
            "ROI": project_metrics.roi if project_metrics is not None else np.nan,
            "PP_years": project_metrics.payback_years if project_metrics is not None else np.nan,
            "EI": project_metrics.economic_index if project_metrics is not None else np.nan,
            "spatial_potential": s_project,
            "INV": round(inv_project, 2),
        }

        currency_columns = {"land_cost", "construction_cost", "investment_need", "NPV"}
        numeric_cols = summary.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            decimals = 2
            summary[col] = self._round_clean(summary[col], decimals=decimals)
            if col in currency_columns:
                summary[col] = summary[col].astype(float)

        round_specs: dict[str, int] = {
            "land_cost": 2,
            "construction_cost": 2,
            "investment_need": 2,
            "NPV": 2,
            "INV": 2,
            "spatial_potential": 2,
            "land_area": 2,
            "built_area": 2,
            "IRR": 2,
            "ROI": 2,
            "PP_years": 2,
            "EI": 2,
        }
        for col, decimals in round_specs.items():
            if col in working.columns:
                working[col] = self._round_clean(working[col], decimals=decimals)
                if col in currency_columns:
                    working[col] = working[col].astype(float)

        working = working.drop(
            columns=["ECON_NPV", "ECON_IRR", "ECON_ROI", "ECON_PP_years", "ECON_EI"],
            errors="ignore",
        )

        summary_columns = list(SUMMARY_COLUMNS)
        keep_cols = ["geometry"] + [col for col in summary_columns if col in working.columns]
        working = working[keep_cols]
        summary = summary[summary_columns]

        return working, summary
