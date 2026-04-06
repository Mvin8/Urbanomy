"""Public exports for land value modelling utilities.

Optional LLM helpers are imported defensively so the core estimators remain
available in lightweight installs.
"""

from .constants import (
    CATEGORICAL_FEATURES,
    DEFAULT_ADJACENCY_RADIUS,
    DEFAULT_OUTPUT_COLUMNS,
    DEFAULT_SQM_PER_PERSON,
    ORIGINAL_FEATURES,
    RADIUS_LIST,
)
from .land_data_preparation import LandDataPreparator
from .land_price_estimation import LandPriceEstimator, transfer_baseline_prices
from .land_price_visualization import plot_land_price_maps
from .pareto_front_dataframe import build_pareto_front_dataframe
from .scenario_modification import ScenarioTEPModifier, plot_scenario_impact
from .ga_mc_optimizer import (
    DistrictProblem,
    StrategicAlignmentScorer,
    build_nsga3_reference_directions,
    run_nsga2_with_strategic_alignment,
    run_nsga3_with_strategic_alignment,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "DEFAULT_ADJACENCY_RADIUS",
    "DEFAULT_OUTPUT_COLUMNS",
    "DEFAULT_SQM_PER_PERSON",
    "ORIGINAL_FEATURES",
    "RADIUS_LIST",
    "LandDataPreparator",
    "LandPriceEstimator",
    "transfer_baseline_prices",
    "plot_land_price_maps",
    "build_pareto_front_dataframe",
    "ScenarioTEPModifier",
    "plot_scenario_impact",
    "DistrictProblem",
    "StrategicAlignmentScorer",
    "build_nsga3_reference_directions",
    "run_nsga2_with_strategic_alignment",
    "run_nsga3_with_strategic_alignment",
]

try:
    from .pareto_llm_selector import (
        ParetoMultiAgentOrchestrator,
        WinnerScenarioQAResult,
        ask_winner_scenario_question,
        collect_pareto_scenarios,
        run_pareto_vote,
        select_best_pareto_scenario,
        select_best_pareto_scenario_multiagent,
    )
except ImportError:
    ParetoMultiAgentOrchestrator = None
    WinnerScenarioQAResult = None
    ask_winner_scenario_question = None
    collect_pareto_scenarios = None
    run_pareto_vote = None
    select_best_pareto_scenario = None
    select_best_pareto_scenario_multiagent = None
else:
    __all__.extend(
        [
            "ParetoMultiAgentOrchestrator",
            "WinnerScenarioQAResult",
            "ask_winner_scenario_question",
            "collect_pareto_scenarios",
            "run_pareto_vote",
            "select_best_pareto_scenario",
            "select_best_pareto_scenario_multiagent",
        ]
    )
