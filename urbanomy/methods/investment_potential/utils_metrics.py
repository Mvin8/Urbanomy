import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import numpy as np
import pandas as pd


def npv(rate: float, cashflows: List[float]) -> float:
    """
    Calculate the net present value of a series of cash flows.

    Parameters
    ----------
    rate : float
        Discount rate per period.
    cashflows : list of float
        Cash flows at each period, where `cashflows[0]` is time 0.

    Returns
    -------
    float
        Net present value.
    """
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))


def irr(
    cashflows: List[float],
    guess: float = 0.1,
    tol: float = 1e-6,
    max_iter: int = 100
) -> float | None:
    """
    Compute the internal rate of return for a series of cash flows.

    Uses the Newton–Raphson method to find the rate that zeroes NPV.

    Parameters
    ----------
    cashflows : list of float
        Cash flows at each period.
    guess : float, optional
        Initial guess for the IRR (default is 0.1).
    tol : float, optional
        Convergence tolerance (default is 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default is 100).

    Returns
    -------
    float or None
        Estimated IRR if converged within `max_iter`, otherwise `None`.
    """
    rate = guess
    for _ in range(max_iter):
        npv_ = npv(rate, cashflows)
        deriv = sum(
            -t * cf / (1 + rate) ** (t + 1)
            for t, cf in enumerate(cashflows)
        )
        if deriv == 0:
            break
        new = rate - npv_ / deriv
        if abs(new - rate) < tol:
            return new
        rate = new
    return None


def payback_period(rate: float, cashflows: List[float]) -> float | None:
    """
    Calculate the discounted payback period for cash flows.

    The payback period is the time when cumulative discounted cash flows
    become non-negative.

    Parameters
    ----------
    rate : float
        Discount rate per period.
    cashflows : list of float
        Cash flows at each period.

    Returns
    -------
    float or None
        Discounted payback period in periods (may be fractional), or `None`
        if the investment is never recovered.
    """
    cum = 0.0
    for t, cf in enumerate(cashflows):
        discounted = cf / (1 + rate) ** t
        prev = cum
        cum += discounted
        if cum >= 0:
            if discounted == 0:
                return float(t)
            return float(t) - (prev / discounted)
    return None


def quantize(value: float | None, places: str = "0.01") -> Decimal | None:
    """
    Quantize a float to a fixed number of decimal places.

    Parameters
    ----------
    value : float or None
        Value to quantize.
    places : str, optional
        Decimal quantization format, e.g. "0.01" for two decimal places.
        Default is "0.01".

    Returns
    -------
    Decimal or None
        Quantized `Decimal` value, or `None` if `value` is `None`.
    """
    if value is None:
        return None
    return Decimal(value).quantize(Decimal(places), rounding=ROUND_HALF_UP)


def make_cashflow(
    lu: str,
    land_area: float,
    profile: Dict[str, Any]
) -> List[float]:
    """
    Generate a series of cash flows for a land-use profile.

    Parameters
    ----------
    lu : str
        Land-use profile key.
    land_area : float
        Land area in the same units used by `profile["density"]`.
    profile : dict
        Profile parameters containing at least:
        - "density": float
        - "cost_build": float
        Optionally:
        - "land_cost": float
        - "construction_years": int
        - "opex_rate": float
        And either:
        - "price_sale": float and "sale_years": int
        or
        - "rent_annual": float, "rent_years": int, "occupancy": float

    Returns
    -------
    list of float
        Cash flow list: initial negative outflows for land and construction,
        followed by net operating revenues or sales.

    Raises
    ------
    ValueError
        If `profile` lacks both "price_sale" and "rent_annual".
    """
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
    """
    Compute the minimum and maximum of values, ignoring NaNs.

    Parameters
    ----------
    values : iterable of float
        Input values, may contain NaNs.

    Returns
    -------
    (float, float)
        Tuple `(min, max)` ignoring NaNs.
    """
    arr = np.array(list(values), dtype=float)
    return float(np.nanmin(arr)), float(np.nanmax(arr))


def normalize_series(
    s: Sequence[float],
    vmin: float,
    vmax: float
) -> pd.Series:
    """
    Normalize a sequence of values to a 0–100 scale.

    Parameters
    ----------
    s : sequence of float
        Input values.
    vmin : float
        Minimum value for normalization.
    vmax : float
        Maximum value for normalization.

    Returns
    -------
    pandas.Series
        Normalized values scaled to [0, 100], or zeros if `vmin == vmax`.
    """
    if vmax > vmin:
        return 100 * (pd.Series(s, dtype=float) - vmin) / (vmax - vmin)
    return pd.Series(0.0, index=pd.RangeIndex(len(s)))


def to_numeric(series: pd.Series) -> pd.Series:
    """
    Coerce a pandas Series to numeric, stripping non-numeric characters.

    Parameters
    ----------
    series : pandas.Series
        Input series, may contain strings with units or symbols.

    Returns
    -------
    pandas.Series
        Numeric series with non-convertible elements set to NaN.
    """
    cleaned = series.astype(str).str.replace(r"[^0-9\.\-]+", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def economic_index(
    npv_val: float | None,
    irr_val: float | None,
    rate: float
) -> float:
    """
    Compute a bounded economic index based on NPV and IRR.

    Combines normalized NPV and IRR contributions into a score between 0 and 100.

    Parameters
    ----------
    npv_val : float or None
        Net present value.
    irr_val : float or None
        Internal rate of return.
    rate : float
        Discount rate for IRR performance normalization.

    Returns
    -------
    float
        Economic index in [0, 100], rounded to four decimal places.
    """
    ei = 0.0
    if npv_val is not None:
        arg = -npv_val / 1e8
        sig = (
            0.0 if arg > 700 else
            1.0 if arg < -700 else
            1.0 / (1 + math.exp(arg))
        )
        ei += 50 * sig
    if irr_val is not None:
        ei += 50 * max(0.0, irr_val - rate) / (0.3 - rate)
    return round(min(max(ei, 0.0), 100.0), 4)


def aggregate_project_cashflows(
    rows: Iterable[dict],
    benchmarks: Dict[str, Dict[str, Any]]
) -> List[float]:
    """
    Aggregate cashflows from multiple rows into a single project cashflow series.

    Parameters
    ----------
    rows : iterable of dict
        Each dict must contain:
        - "ip_type": land-use profile key (possibly prefixed with "ИП_").
        - "area": numeric land area.
    benchmarks : dict of dict
        Profile-specific parameters for `make_cashflow`.

    Returns
    -------
    list of float
        Aggregated cash flow series summing individual cash flows by period.
        Returns an empty list if no valid profiles are found.
    """
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
