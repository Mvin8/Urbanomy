import math
from decimal import Decimal, ROUND_HALF_UP
from typing import List
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import numpy as np
import pandas as pd


def npv(rate: float, cashflows: List[float]) -> float:
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))

def irr(cashflows: List[float], guess: float = 0.1, tol: float = 1e-6,
        max_iter: int = 100) -> float | None:
    rate = guess
    for _ in range(max_iter):
        npv_ = npv(rate, cashflows)
        deriv = sum(-t * cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cashflows))
        if deriv == 0: break
        new = rate - npv_ / deriv
        if abs(new - rate) < tol:
            return new
        rate = new
    return None

def payback_period(rate: float, cashflows: List[float]) -> float | None:
    cum = 0.0
    for t, cf in enumerate(cashflows):
        cum += cf / (1 + rate) ** t
        if cum >= 0:
            prev = cum - cf / (1 + rate) ** t
            return float(t) if cf == 0 else float(t)-(prev/(cf/(1+rate)**t))
    return None

def quantize(value: float | None, places: str = "0.01") -> Decimal | None:
    return (Decimal(value).quantize(Decimal(places), rounding=ROUND_HALF_UP)
            if value is not None else None)


def make_cashflow(lu: str, land_area: float, profile: Dict[str, Any]) -> List[float]:
    density = profile["density"]
    gfa = land_area * density

    land_cost = land_area * profile.get("land_cost", 0)
    build_cost = gfa * profile["cost_build"]
    years_build = profile.get("construction_years", 2)
    capex_per_year = build_cost / years_build

    cf: List[float] = [-land_cost - capex_per_year] + [-capex_per_year] * (years_build - 1)

    opex = profile.get("opex_rate", 0) * gfa
    if "price_sale" in profile:
        yrs = profile.get("sale_years", 3)
        rev_total = gfa * profile["price_sale"]
        rev_per_year = rev_total / yrs
        cf.extend(rev_per_year - opex for _ in range(yrs))
    elif "rent_annual" in profile:
        yrs = profile.get("rent_years", 10)
        occ = profile.get("occupancy", 0.9)
        rev_per_year = gfa * profile["rent_annual"] * occ
        cf.extend(rev_per_year - opex for _ in range(yrs))
    else:
        raise ValueError(f"Profile '{lu}' needs price_sale or rent_annual")
    return cf

def nanminmax(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.array(list(values), dtype=float)
    return float(np.nanmin(arr)), float(np.nanmax(arr))

def normalize_series(s: Sequence[float], vmin: float, vmax: float) -> pd.Series:
    if vmax > vmin:
        return 100 * (pd.Series(s, dtype=float) - vmin) / (vmax - vmin)
    # все в одной точке — возвращаем нули того же размера
    return pd.Series(0.0, index=pd.RangeIndex(len(s)))

def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^0-9\.\-]+", "", regex=True),
        errors="coerce",
    )

def economic_index(npv_val: float | None, irr_val: float | None, rate: float) -> float:
    ei = 0.0
    if npv_val is not None:
        arg = -npv_val / 1e8
        # защита от переполнения
        sig = 0.0 if arg > 700 else (1.0 if arg < -700 else 1.0 / (1 + math.exp(arg)))
        ei += 50 * sig
    if irr_val is not None:
        ei += 50 * max(0.0, irr_val - rate) / (0.3 - rate)
    return round(min(max(ei, 0.0), 100.0), 4)

def aggregate_project_cashflows(
    rows: Iterable[dict],
    benchmarks: Dict[str, Dict[str, Any]]
) -> List[float]:
    all_cfs: List[List[float]] = []
    for row in rows:
        lu = row.get("ip_type", "").replace("ИП_", "")
        if lu not in benchmarks:
            continue
        cf = make_cashflow(lu, float(row["area"]), benchmarks[lu])
        all_cfs.append(cf)
    if not all_cfs:
        return []
    max_len = max(len(cf) for cf in all_cfs)
    return [
        sum(cf[i] if i < len(cf) else 0.0 for cf in all_cfs)
        for i in range(max_len)
    ]