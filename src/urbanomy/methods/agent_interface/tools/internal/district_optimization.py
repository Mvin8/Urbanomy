"""Compatibility facade for district optimization helper functions."""

from .district_optimization_plots import (
    plot_optimization_pareto_front,
    plot_solution_impact,
)
from .district_optimization_session import (
    build_default_constraints,
    latest_session_or_error,
    latest_session_summary,
    resolve_constraints,
    run_optimization_session,
    run_optimization_session_nsga3,
)
from .district_optimization_solution import (
    build_blocks_full_value_for_solution,
    calculate_solution_investment_summary,
    get_solution_parameters,
    resolve_solution_row,
)

__all__ = [
    "build_blocks_full_value_for_solution",
    "build_default_constraints",
    "calculate_solution_investment_summary",
    "get_solution_parameters",
    "latest_session_or_error",
    "latest_session_summary",
    "plot_optimization_pareto_front",
    "plot_solution_impact",
    "resolve_constraints",
    "resolve_solution_row",
    "run_optimization_session",
    "run_optimization_session_nsga3",
]
