"""Plotting helpers for district optimization tools."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from urbanomy.methods.land_value_modeling import plot_scenario_impact

from ..models import DistrictOptimizationSession
from ._district_optimization_formatting import (
    build_pareto_front_text,
    build_plot_solution_summary_text,
    json_mapping,
    json_value,
    to_float,
)
from ._district_optimization_solution import build_blocks_full_value_for_solution


def plot_solution_impact(
    session: DistrictOptimizationSession,
    *,
    solution_number: int,
) -> dict[str, object]:
    """Plot scenario impact for one Pareto solution and return a compact payload."""
    blocks_full_value, params_repaired, target_row = build_blocks_full_value_for_solution(
        session,
        solution_number=solution_number,
    )
    problem = session.problem
    scenario_result = plot_scenario_impact(
        blocks=blocks_full_value,
        target_idx=problem.target_id,
        target_id_column=problem.target_id_column,
        print_summary=False,
        print_quarter_stats=False,
        figsize=(25, 35),
    )
    sum_before = float(pd.to_numeric(blocks_full_value["land_value_before"], errors="coerce").sum())
    sum_after = float(pd.to_numeric(blocks_full_value["land_value"], errors="coerce").sum())
    sum_delta = float(sum_after - sum_before)
    sum_delta_pct = float((sum_after / sum_before - 1.0) * 100.0) if sum_before > 0 else math.nan
    target_before = to_float(target_row.get("land_value_before"))
    target_after = to_float(target_row.get("land_value"))
    target_delta_rub = to_float(target_row.get("d_rub"))
    target_delta_pct = to_float(target_row.get("land_value_delta_pct"))
    return {
        "solution_number": int(solution_number),
        "target_id": problem.target_id,
        "params_repaired": json_mapping(params_repaired),
        "sum_before": json_value(sum_before),
        "sum_after": json_value(sum_after),
        "sum_delta_rub": json_value(sum_delta),
        "sum_delta_pct": json_value(sum_delta_pct),
        "target_before": json_value(target_before),
        "target_after": json_value(target_after),
        "target_delta_rub": json_value(target_delta_rub),
        "target_delta_pct": json_value(target_delta_pct),
        "summary_text": build_plot_solution_summary_text(
            sum_before=sum_before,
            sum_after=sum_after,
            sum_delta=sum_delta,
            sum_delta_pct=sum_delta_pct,
            target_before=target_before,
            target_after=target_after,
            target_delta_rub=target_delta_rub,
            target_delta_pct=target_delta_pct,
        ),
        "figure_created": scenario_result.get("figure") is not None,
    }


def plot_optimization_pareto_front(session: DistrictOptimizationSession) -> dict[str, object]:
    """Plot the Pareto front for the latest optimization session."""
    problem = session.problem
    result = session.result

    baseline_land_value = float(
        problem.evaluate_catboost(
            geonome=problem.blocks,
            model=problem.model,
            orig_features=problem.estimator_kwargs["orig_features"],
            cat_features=problem.estimator_kwargs["categorical_features"],
            radius_list=None,
        )
    )

    x_batches: list[np.ndarray] = []
    f_batches: list[np.ndarray] = []
    if getattr(result, "history", None):
        for algorithm in result.history:
            pop = getattr(algorithm, "pop", None)
            if pop is None:
                continue
            x_values = pop.get("X")
            f_values = pop.get("F")
            if x_values is None or f_values is None or len(x_values) == 0:
                continue
            x_batches.append(np.asarray(x_values))
            f_batches.append(np.asarray(f_values))

    if x_batches:
        x_all = np.vstack(x_batches)
        f_all = np.vstack(f_batches)
    else:
        x_all = np.asarray(result.X)
        f_all = np.asarray(result.F)

    if x_all.ndim == 1:
        x_all = x_all.reshape(1, -1)
    if f_all.ndim == 1:
        f_all = f_all.reshape(1, -1)
    if len(x_all) == 0 or len(f_all) == 0:
        raise ValueError("Не удалось построить график: решения оптимизации отсутствуют.")

    land_value_total = -f_all[:, 0]
    admin_gain = land_value_total - baseline_land_value
    investor_npv = -f_all[:, 1]

    landuse_labels: list[str] = []
    for genome_vec in x_all:
        changes = {name: genome_vec[j] for j, name in enumerate(problem.var_names)}
        repaired = problem._repair_genome(changes)
        land_use = repaired["land_use"]
        landuse_labels.append(getattr(land_use, "name", str(land_use).split(".")[-1]))

    landuse_labels_arr = np.asarray(landuse_labels)
    unique_lu = np.unique(landuse_labels_arr)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10", len(unique_lu))
    color_map = {lu: cmap(i) for i, lu in enumerate(unique_lu)}

    for lu in unique_lu:
        mask = landuse_labels_arr == lu
        ax.scatter(
            investor_npv[mask],
            admin_gain[mask],
            s=35,
            alpha=0.8,
            color=color_map[lu],
            label=f"LANDUSE: {lu}",
        )

    pf_x = -np.asarray(result.F)[:, 1]
    pf_y = -np.asarray(result.F)[:, 0] - baseline_land_value
    order = np.argsort(pf_x)
    ax.plot(
        pf_x[order],
        pf_y[order],
        color="black",
        lw=1.5,
        alpha=0.7,
        label="Pareto front",
    )

    ax.set_xlabel("NPV инвестора (X), руб.")
    ax.set_ylabel("Прирост общей стоимости земли (Y), руб.")
    ax.set_title("Парето фронт")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

    return {
        "target_id": session.target_id,
        "n_points": int(len(x_all)),
        "n_pareto_points": int(len(np.asarray(result.F))),
        "land_use_labels": [str(item) for item in unique_lu.tolist()],
        "baseline_land_value": json_value(baseline_land_value),
        "figure_created": fig is not None,
        "summary_text": build_pareto_front_text(
            n_points=int(len(x_all)),
            n_pareto_points=int(len(np.asarray(result.F))),
            land_use_labels=[str(item) for item in unique_lu.tolist()],
        ),
    }
