"""Solution reconstruction helpers for district optimization tools."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Any

import numpy as np
import pandas as pd

from urbanomy.methods.investment_potential import calculate_investment_metrics
from urbanomy.methods.land_value_modeling import LandPriceEstimator, ScenarioTEPModifier

from ...models import DistrictOptimizationSession
from .district_optimization_formatting import (
    build_params_text,
    clean_text_block,
    dataframe_records,
    json_mapping,
)


def build_blocks_full_value_for_solution(
    session: DistrictOptimizationSession,
    *,
    solution_number: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.Series]:
    """Build before/after valuation tables for one Pareto solution."""
    row = resolve_solution_row(session, solution_number)
    params_repaired = dict(row["params_repaired"])
    problem = session.problem

    blocks_before = problem.blocks.copy()
    blocks_before["is_project"] = False
    blocks_before.loc[
        blocks_before[problem.target_id_column] == problem.target_id,
        "is_project",
    ] = True

    modifier = ScenarioTEPModifier(blocks_before)
    blocks_after = modifier.apply(
        problem.target_id,
        params_repaired,
        target_id_column=problem.target_id_column,
    )
    blocks_after["is_project"] = False
    blocks_after.loc[
        blocks_after[problem.target_id_column] == problem.target_id,
        "is_project",
    ] = True

    estimator_kwargs = {
        "model": problem.model,
        "orig_features": problem.estimator_kwargs["orig_features"],
        "categorical_features": problem.estimator_kwargs["categorical_features"],
        "use_service_features": bool(problem.estimator_kwargs.get("use_service_features", False)),
    }
    if "service_features" in problem.estimator_kwargs:
        estimator_kwargs["service_features"] = problem.estimator_kwargs["service_features"]

    baseline_estimator = LandPriceEstimator(blocks=blocks_before, **estimator_kwargs)
    blocks_before_pred = baseline_estimator.predict()

    scenario_estimator = LandPriceEstimator(blocks=blocks_after, **estimator_kwargs)
    blocks_after_pred = scenario_estimator.predict()

    for df in (blocks_before_pred, blocks_after_pred):
        df["land_value_per_100m2"] = np.where(
            pd.to_numeric(df["site_area"], errors="coerce") > 0,
            pd.to_numeric(df["land_value"], errors="coerce")
            / pd.to_numeric(df["site_area"], errors="coerce") * 100.0,
            np.nan,
        )

    baseline_cols = blocks_before_pred[
        [problem.target_id_column, "land_value", "land_value_per_100m2"]
    ].rename(
        columns={
            "land_value": "land_value_before",
            "land_value_per_100m2": "land_value_before_per_100m2",
        }
    )
    blocks_full_value = blocks_after_pred.merge(
        baseline_cols,
        on=problem.target_id_column,
        how="left",
        validate="one_to_one",
    )
    blocks_full_value["d_rub"] = (
        blocks_full_value["land_value"] - blocks_full_value["land_value_before"]
    )
    blocks_full_value["land_value_delta_pct"] = np.where(
        pd.to_numeric(blocks_full_value["land_value_before"], errors="coerce") > 0,
        (blocks_full_value["land_value"] / blocks_full_value["land_value_before"] - 1.0) * 100.0,
        np.nan,
    )
    blocks_full_value = blocks_full_value.replace([np.inf, -np.inf], np.nan)

    target_row = blocks_full_value.loc[
        blocks_full_value[problem.target_id_column] == problem.target_id
    ]
    if target_row.empty:
        raise ValueError("Не удалось восстановить целевой квартал для выбранного решения.")
    return blocks_full_value, params_repaired, target_row.iloc[0]


def calculate_solution_investment_summary(
    session: DistrictOptimizationSession,
    *,
    solution_number: int,
) -> dict[str, Any]:
    """Calculate investment metrics for one Pareto solution."""
    blocks_full_value, params_repaired, _ = build_blocks_full_value_for_solution(
        session,
        solution_number=solution_number,
    )
    stdout_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer):
        summary = calculate_investment_metrics(
            blocks_full_value,
            show_project_totals=True,
            show_warning=False,
        )
    return {
        "solution_number": int(solution_number),
        "target_id": session.target_id,
        "params_repaired": json_mapping(params_repaired),
        "n_rows": int(len(summary)),
        "summary_records": dataframe_records(summary),
        "project_totals_text": clean_text_block(stdout_buffer.getvalue()),
    }


def get_solution_parameters(
    session: DistrictOptimizationSession,
    *,
    solution_number: int,
) -> dict[str, Any]:
    """Return repaired parameters for one Pareto solution."""
    row = resolve_solution_row(session, solution_number)
    params_repaired = json_mapping(dict(row["params_repaired"]))
    return {
        "solution_number": int(solution_number),
        "target_id": session.target_id,
        "params_repaired": params_repaired,
        "params_text": build_params_text(solution_number=solution_number, params_repaired=params_repaired),
    }


def resolve_solution_row(session: DistrictOptimizationSession, solution_number: int) -> pd.Series:
    """Return one Pareto-solution row by 1-based solution number."""
    solution_index = int(solution_number) - 1
    if solution_index < 0 or solution_index >= len(session.pareto_front_df):
        raise ValueError(f"Номер решения вне диапазона: доступно от 1 до {len(session.pareto_front_df)}.")
    return session.pareto_front_df.iloc[solution_index]
