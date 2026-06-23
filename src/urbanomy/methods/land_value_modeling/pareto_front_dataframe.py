"""Lightweight Pareto-front table helpers.

This module intentionally avoids importing optional LLM dependencies so that
basic optimization notebooks can always build a Pareto summary table.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _json_ready(value: Any) -> Any:
    """Convert values to JSON-safe primitives."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if hasattr(value, "item"):
        try:
            return _json_ready(value.item())
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return str(value)


def build_pareto_front_dataframe(
    *,
    res: Any,
    problem: Any,
    baseline_land_value: float | None = None,
    use_history: bool = False,
    scenario_prefix: str = "scenario",
) -> pd.DataFrame:
    """Build a notebook-friendly Pareto-front DataFrame from optimizer outputs."""
    if baseline_land_value is None:
        baseline_land_value = float(
            problem.evaluate_catboost(
                geonome=problem.blocks,
                model=problem.model,
                orig_features=problem.estimator_kwargs["orig_features"],
                cat_features=problem.estimator_kwargs["categorical_features"],
                radius_list=None,
            )
        )

    x_all: list[Any] = []
    f_all: list[Any] = []
    if use_history and getattr(res, "history", None):
        for alg in res.history:
            pop = alg.pop
            xi, fi = pop.get("X"), pop.get("F")
            if xi is not None and fi is not None and len(xi):
                x_all.extend(list(xi))
                f_all.extend(list(fi))

    if x_all and f_all:
        x_values = x_all
        f_values = f_all
    else:
        x_values = list(res.X)
        f_values = list(res.F)

    rows: list[dict[str, Any]] = []
    for x_row, f_row in zip(x_values, f_values):
        params_optimal = {
            param: x_row[i]
            for i, param in enumerate(problem.constraints.keys())
        }
        params_repaired = problem._repair_genome(dict(params_optimal))
        land_use_value = params_repaired.get("land_use")
        land_use_name = getattr(land_use_value, "name", str(land_use_value).split(".")[-1])
        land_value_after = float(-f_row[0])
        land_value_gain = float(land_value_after - baseline_land_value)
        investor_npv = float(-f_row[1])
        llm_score = float(-f_row[2]) if len(f_row) > 2 else None
        row = {
            "target_id": _json_ready(getattr(problem, "target_id", None)),
            "land_use": land_use_name,
            "land_value_after": land_value_after,
            "land_value_gain": land_value_gain,
            "investor_npv": investor_npv,
            "params_repaired": _json_ready(params_repaired),
        }
        if llm_score is not None:
            row["llm score"] = llm_score
        rows.append(row)

    return pd.DataFrame(rows)


__all__ = ["build_pareto_front_dataframe"]
