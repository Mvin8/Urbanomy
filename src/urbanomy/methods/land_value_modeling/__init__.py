"""Public exports for land value modelling utilities.

The package used to expose LLM-based scenario helpers from sibling modules.
Those modules are optional and may be absent in lightweight installs or local
worktrees, so imports are guarded to keep core estimators available.
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
]

try:
    from .llm_agents import MultiAgentScenarioSelector
except ImportError:
    MultiAgentScenarioSelector = None
else:
    __all__.append("MultiAgentScenarioSelector")

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
