"""Agent-facing interface utilities for Urbanomy methods."""

from __future__ import annotations

import os
from importlib import import_module

# The agent interface eagerly imports plotting-heavy modules during package
# initialization. On macOS this must happen with a non-interactive backend,
# otherwise worker-thread rendering inside the chat server crashes.
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib

    matplotlib.use(os.environ["MPLBACKEND"])
except Exception:
    # Defer to Matplotlib defaults if it is unavailable at import time.
    pass

from .models import (
    DistrictOptimizationConfig,
    DistrictOptimizationSession,
    LandValueVisualizationArtifact,
    LandValueVisualizationRequest,
    LandValueVisualizationResult,
    TargetBlockVisualizationArtifact,
    TargetBlockVisualizationResult,
    UrbanomyOrchestratorRequest,
    UrbanomyOrchestratorResult,
    UrbanomyOrchestratorRouteDecision,
    VisualizationResult,
    VisualizationRouteDecision,
)

__all__ = [
    "DistrictOptimizationConfig",
    "DistrictOptimizationSession",
    "LandValueVisualizationArtifact",
    "LandValueVisualizationRequest",
    "LandValueVisualizationResult",
    "TargetBlockVisualizationArtifact",
    "TargetBlockVisualizationResult",
    "UrbanomyOrchestratorRequest",
    "UrbanomyOrchestratorResult",
    "UrbanomyOrchestratorRouteDecision",
    "VisualizationResult",
    "VisualizationRouteDecision",
    "create_urbanomy_orchestrator",
]


def _load_optional(module_name: str, names: list[str]) -> None:
    try:
        module = import_module(module_name, package=__name__)
    except ImportError as exc:
        missing = getattr(exc, "name", "") or ""
        if missing.startswith("urbanomy.methods.agent_interface"):
            raise
        for name in names:
            globals()[name] = None
        return
    for name in names:
        globals()[name] = getattr(module, name)
    __all__.extend(names)


def create_urbanomy_orchestrator(*args, **kwargs):
    """Lazily import and construct the top-level Urbanomy orchestrator."""
    module = import_module(".urbanomy_orchestrator", package=__name__)
    factory = getattr(module, "create_urbanomy_orchestrator")
    return factory(*args, **kwargs)


_load_optional(
    ".block_parameters_agent",
    [
        "BlockParametersAgent",
        "create_block_parameters_agent",
    ],
)
_load_optional(
    ".general_qa_agent",
    [
        "GeneralQaAgent",
        "create_general_qa_agent",
    ],
)
_load_optional(
    ".visualization_agent",
    [
        "VisualizationAgent",
        "create_visualization_agent",
        "visualize_from_request",
    ],
)
_load_optional(
    ".land_value_visualization_agent",
    [
        "LandValueVisualizationRoutingAgent",
        "create_land_value_visualization_agent",
        "visualize_land_value_from_request",
    ],
)
_load_optional(
    ".target_block_visualization_agent",
    [
        "TargetBlockVisualizationAgent",
        "create_target_block_visualization_agent",
    ],
)
_load_optional(
    ".district_optimization_agent",
    [
        "DistrictOptimizationAgent",
        "create_district_optimization_agent",
    ],
)
_load_optional(
    ".urbanomy_orchestrator",
    [
        "UrbanomyOrchestrator",
    ],
)
_load_optional(
    ".chat_runtime",
    [
        "create_chat_runtime",
    ],
)
