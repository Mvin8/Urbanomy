import math
from decimal import Decimal, ROUND_HALF_UP
from typing import List

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
