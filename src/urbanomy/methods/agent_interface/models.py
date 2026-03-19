"""Pydantic schemas for the Urbanomy agent interface."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LandValueVisualizationRequest(BaseModel):
    """Runtime configuration for a land-value visualization request."""

    model_config = ConfigDict(extra="forbid")

    user_request: str = Field(description="Natural-language request passed to the routing agent.")
    show_plot: bool = Field(default=True, description="Whether to display the map via matplotlib.")
    figsize: tuple[float, float] = Field(
        default=(20.0, 20.0),
        description="Figure size in inches as (width, height).",
    )
    cmap: str = Field(default="coolwarm", description="Matplotlib colormap name.")
    edgecolor: str = Field(default="black", description="Polygon edge color.")
    linewidth: float = Field(default=0.2, ge=0.0, description="Polygon edge width.")
    legend: bool = Field(default=True, description="Whether to display a color legend.")
    axis_off: bool = Field(default=True, description="Whether to hide axes.")
    default_total_title: str = Field(
        default="Карта стоимости земельных участков (руб.)",
        description="Default plot title for total land value.",
    )
    default_unit_title: str = Field(
        default="Карта стоимости земельных участков за сотку (руб.)",
        description="Default plot title for land value per 100 m2.",
    )

    @field_validator("user_request", mode="before")
    @classmethod
    def _validate_user_request(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("user_request cannot be empty")
        return text


class VisualizationRouteDecision(BaseModel):
    """Structured routing output for the unified visualization graph."""

    model_config = ConfigDict(extra="forbid")

    route: Literal[
        "plot_total_land_value_map",
        "plot_land_value_per_100m2_map",
        "plot_target_block_map",
    ]
    metric_kind: Literal["total_land_value", "land_value_per_100m2", "target_block"]
    price_column: str
    title: str = Field(description="Plot title that should be passed to the selected tool.")
    target_id: int | None = Field(
        default=None,
        description="Target block id for plot_target_block_map; otherwise null.",
    )
    reasoning: str = Field(description="Short explanation of why the route was chosen.")


class LandValueVisualizationArtifact(BaseModel):
    """Plot artifact created by a visualization tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    tool_name: str
    metric_kind: Literal["total_land_value", "land_value_per_100m2"]
    price_column: str
    title: str
    rows_plotted: int
    figure: Any | None = None
    axis: Any | None = None

    def tool_payload(self) -> dict[str, Any]:
        """Return a tool-safe payload without large matplotlib objects."""
        return {
            "tool_name": self.tool_name,
            "metric_kind": self.metric_kind,
            "price_column": self.price_column,
            "title": self.title,
            "rows_plotted": self.rows_plotted,
        }


class LandValueVisualizationResult(BaseModel):
    """Final routing-agent result returned to the caller."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    user_request: str
    route: Literal["plot_total_land_value_map", "plot_land_value_per_100m2_map"]
    metric_kind: Literal["total_land_value", "land_value_per_100m2"]
    price_column: str
    title: str
    reasoning: str
    agent_message: str = ""
    used_tool_fallback: bool = False
    tool_payload: dict[str, Any] = Field(default_factory=dict)
    artifact: LandValueVisualizationArtifact | None = None


class TargetBlockVisualizationArtifact(BaseModel):
    """Plot artifact created by the target-block visualization tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    tool_name: str
    target_id: int
    title: str
    rows_plotted: int
    figure: Any | None = None
    axis: Any | None = None

    def tool_payload(self) -> dict[str, Any]:
        """Return a tool-safe payload without large matplotlib objects."""
        return {
            "tool_name": self.tool_name,
            "target_id": self.target_id,
            "title": self.title,
            "rows_plotted": self.rows_plotted,
        }


class TargetBlockVisualizationResult(BaseModel):
    """Final result returned by the target-block visualization agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    user_request: str
    route: Literal["plot_target_block_map"]
    target_id: int
    title: str
    reasoning: str
    agent_message: str = ""
    used_tool_fallback: bool = False
    tool_payload: dict[str, Any] = Field(default_factory=dict)
    artifact: TargetBlockVisualizationArtifact | None = None


class VisualizationResult(BaseModel):
    """Final result returned by the unified visualization agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    user_request: str
    route: Literal[
        "plot_total_land_value_map",
        "plot_land_value_per_100m2_map",
        "plot_target_block_map",
    ]
    metric_kind: Literal["total_land_value", "land_value_per_100m2", "target_block"]
    price_column: str
    title: str
    reasoning: str
    target_id: int | None = None
    agent_message: str = ""
    used_tool_fallback: bool = False
    tool_payload: dict[str, Any] = Field(default_factory=dict)
    artifact: LandValueVisualizationArtifact | TargetBlockVisualizationArtifact | None = None


class DistrictOptimizationConfig(BaseModel):
    """Configuration required to run district optimization tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    model: Any
    orig_features: list[str]
    categorical_features: list[str]
    constraints_template: dict[str, dict[str, Any]] | None = None
    target_id_column: str = "id"
    pop_size: int = 10
    n_gen: int = 15
    seed: int = 42
    eliminate_duplicates: bool = True
    save_history: bool = True
    use_history: bool = False
    verbose: bool = True
    scenario_prefix: str = "scenario"
    use_service_features: bool | None = None
    service_features: list[str] | None = None


class DistrictOptimizationSession(BaseModel):
    """Persisted district optimization session used across tool calls."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    target_id: int
    site_area: float
    problem: Any
    result: Any
    pareto_front_df: Any


class UrbanomyOrchestratorRequest(BaseModel):
    """Top-level user request routed by the Urbanomy orchestrator."""

    model_config = ConfigDict(extra="forbid")

    user_request: str = Field(description="Natural-language request sent to the orchestrator.")

    @field_validator("user_request", mode="before")
    @classmethod
    def _validate_user_request(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("user_request cannot be empty")
        return text


class UrbanomyOrchestratorRouteDecision(BaseModel):
    """Structured routing decision for the top-level Urbanomy orchestrator."""

    model_config = ConfigDict(extra="forbid")

    route: Literal[
        "visualization",
        "land_value_visualization",
        "target_block_visualization",
        "district_optimization",
        "general_qa",
        "unsupported",
    ]
    reasoning: str = Field(description="Short explanation of why the route was chosen.")


class UrbanomyOrchestratorResult(BaseModel):
    """Final result returned by the top-level Urbanomy orchestrator."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    user_request: str
    route: Literal[
        "visualization",
        "land_value_visualization",
        "target_block_visualization",
        "district_optimization",
        "general_qa",
        "unsupported",
    ]
    reasoning: str
    response: str = ""
    visualization_result: VisualizationResult | None = None
    target_block_visualization_result: TargetBlockVisualizationResult | None = None
    district_optimization_result: Any | None = None

    def __repr__(self) -> str:
        if self.response:
            return self.response
        return super().__repr__()

    def __str__(self) -> str:
        return self.response or super().__repr__()
