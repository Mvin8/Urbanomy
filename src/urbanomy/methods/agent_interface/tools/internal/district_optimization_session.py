"""Session-level helpers for district optimization tools."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from urbanomy.methods.land_value_modeling import build_pareto_front_dataframe
from urbanomy.methods.land_value_modeling.ga_mc_optimizer import DistrictProblem

from ...models import DistrictOptimizationConfig, DistrictOptimizationSession
from .district_optimization_formatting import json_value

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
except ImportError as exc:
    raise ImportError(
        "District optimization tools require pymoo. Install the project with pymoo available."
    ) from exc


def build_default_constraints(site_area: float) -> dict[str, dict[str, Any]]:
    """Return the default optimization constraints used in the notebooks."""
    return {
        "footprint_area": {"type": "float", "min": 0.0, "max": 0.1 * float(site_area)},
        "l": {"type": "float", "min": 1.0, "max": 10.0},
        "mxi": {"type": "float", "min": 0.0, "max": 1.0},
        "residential": {"type": "float", "min": 0.0, "max": 1.0},
        "business": {"type": "float", "min": 0.0, "max": 1.0},
        "recreation": {"type": "float", "min": 0.0, "max": 1.0},
        "industrial": {"type": "float", "min": 0.0, "max": 1.0},
        "transport": {"type": "float", "min": 0.0, "max": 1.0},
        "special": {"type": "float", "min": 0.0, "max": 1.0},
        "agriculture": {"type": "float", "min": 0.0, "max": 1.0},
    }


def run_optimization_session(
    *,
    blocks,
    config: DistrictOptimizationConfig,
    target_id: int,
    pop_size: int | None = None,
    n_gen: int | None = None,
    seed: int | None = None,
) -> DistrictOptimizationSession:
    """Run NSGA2 district optimization and return a persisted session object."""
    site_area_series = pd.to_numeric(
        blocks.loc[blocks[config.target_id_column] == target_id, "site_area"],
        errors="coerce",
    )
    if site_area_series.empty or not np.isfinite(site_area_series.iloc[0]):
        raise ValueError(f"Не удалось определить site_area для target_id={target_id}.")
    site_area = float(site_area_series.iloc[0])

    constraints = (
        resolve_constraints(config.constraints_template, site_area=site_area)
        if config.constraints_template
        else build_default_constraints(site_area)
    )
    estimator_kwargs = {
        "orig_features": list(config.orig_features),
        "categorical_features": list(config.categorical_features),
    }
    if config.use_service_features is not None:
        estimator_kwargs["use_service_features"] = bool(config.use_service_features)
    if config.service_features is not None:
        estimator_kwargs["service_features"] = list(config.service_features)

    problem = DistrictProblem(
        blocks=blocks,
        target_id=target_id,
        model=config.model,
        constraints=constraints,
        estimator_kwargs=estimator_kwargs,
        target_id_column=config.target_id_column,
    )
    algorithm = NSGA2(
        pop_size=int(pop_size or config.pop_size),
        eliminate_duplicates=bool(config.eliminate_duplicates),
    )
    result = minimize(
        problem,
        algorithm,
        ("n_gen", int(n_gen or config.n_gen)),
        seed=int(seed if seed is not None else config.seed),
        verbose=bool(config.verbose),
        save_history=bool(config.save_history),
    )
    pareto_front_df = build_pareto_front_dataframe(
        res=result,
        problem=problem,
        use_history=bool(config.use_history),
        scenario_prefix=config.scenario_prefix,
    )
    return DistrictOptimizationSession(
        target_id=int(target_id),
        site_area=site_area,
        problem=problem,
        result=result,
        pareto_front_df=pareto_front_df,
    )


def latest_session_or_error(session_store: dict[str, Any]) -> DistrictOptimizationSession:
    """Return the latest optimization session or raise a clear error."""
    session = session_store.get("latest_district_optimization_session")
    if session is None:
        raise ValueError(
            "Нет активной сессии оптимизации. Сначала запустите оптимизацию, например: "
            "'Оптимизируй квартал с target_id=86'."
        )
    return session


def latest_session_summary(session: DistrictOptimizationSession) -> dict[str, Any]:
    """Return a compact summary of the latest optimization session."""
    pareto = session.pareto_front_df.copy()
    scenarios = []
    for idx, row in pareto.iterrows():
        scenarios.append(
            {
                "solution_number": int(idx) + 1,
                "scenario_id": row.get("scenario_id"),
                "title": row.get("title"),
                "land_use": row.get("land_use"),
                "land_value_after": json_value(row.get("land_value_after")),
                "land_value_gain": json_value(row.get("land_value_gain")),
                "investor_npv": json_value(row.get("investor_npv")),
            }
        )
    return {
        "target_id": session.target_id,
        "site_area": session.site_area,
        "n_solutions": int(len(pareto)),
        "solutions": scenarios,
    }


def resolve_constraints(
    constraints_template: Mapping[str, Mapping[str, Any]],
    *,
    site_area: float,
) -> dict[str, dict[str, Any]]:
    """Resolve dynamic constraint values based on site area."""
    resolved: dict[str, dict[str, Any]] = {}
    for key, spec in constraints_template.items():
        item = dict(spec)
        max_value = item.get("max")
        if isinstance(max_value, str) and max_value == "0.1*site_area":
            item["max"] = 0.1 * float(site_area)
        resolved[str(key)] = item
    return resolved
