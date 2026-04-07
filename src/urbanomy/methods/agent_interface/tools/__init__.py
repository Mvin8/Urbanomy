"""Tools available to Urbanomy interface agents."""

from .calculate_pareto_solution_investment_metrics import (
    make_calculate_pareto_solution_investment_metrics_tool,
)
from .get_district_optimization_problem_statement import (
    make_get_district_optimization_problem_statement_tool,
)
from .get_active_decision_document import make_get_active_decision_document_tool
from .get_district_optimization_constraints import (
    make_get_district_optimization_constraints_tool,
)
from .get_block_parameter_definition import make_get_block_parameter_definition_tool
from .get_pareto_solution_parameters import make_get_pareto_solution_parameters_tool
from .plot_district_optimization_pareto_front import (
    make_plot_district_optimization_pareto_front_tool,
)
from .plot_land_value_per_100m2_map import make_plot_land_value_per_100m2_map_tool
from .plot_pareto_solution_impact import make_plot_pareto_solution_impact_tool
from .plot_target_block_map import make_plot_target_block_map_tool
from .plot_total_land_value_map import make_plot_total_land_value_map_tool
from .predict_land_value import make_predict_land_value_tool
from .register_decision_document import make_register_decision_document_tool
from .retrieve_decision_document_context import make_retrieve_decision_document_context_tool
from .run_district_optimization import make_run_district_optimization_tool
from .select_pareto_solution_by_document import make_select_pareto_solution_by_document_tool

__all__ = [
    "make_calculate_pareto_solution_investment_metrics_tool",
    "make_get_block_parameter_definition_tool",
    "make_get_active_decision_document_tool",
    "make_get_district_optimization_constraints_tool",
    "make_get_district_optimization_problem_statement_tool",
    "make_get_pareto_solution_parameters_tool",
    "make_plot_district_optimization_pareto_front_tool",
    "make_plot_land_value_per_100m2_map_tool",
    "make_plot_pareto_solution_impact_tool",
    "make_plot_target_block_map_tool",
    "make_plot_total_land_value_map_tool",
    "make_predict_land_value_tool",
    "make_register_decision_document_tool",
    "make_retrieve_decision_document_context_tool",
    "make_run_district_optimization_tool",
    "make_select_pareto_solution_by_document_tool",
]
