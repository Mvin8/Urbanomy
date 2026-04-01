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
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Router confidence from 0.0 to 1.0.",
    )


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


class DecisionDocumentChunk(BaseModel):
    """One normalized chunk indexed in the document RAG store."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    char_start: int = Field(default=0, ge=0)
    char_end: int = Field(default=0, ge=0)
    keywords: list[str] = Field(default_factory=list)

    @field_validator("chunk_id", "text", mode="before")
    @classmethod
    def _coerce_chunk_text(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("chunk text fields cannot be empty")
        return text

    def preview_text(self, *, limit: int = 160) -> str:
        """Return a short preview for QA and notebook display."""
        value = " ".join(self.text.split()).strip()
        if len(value) <= limit:
            return value
        return value[: limit - 1].rstrip() + "…"


class RetrievedDocumentChunk(BaseModel):
    """One chunk returned by the document retriever for the current query."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    score: float = Field(default=0.0, ge=0.0)
    text: str
    rationale: str = ""

    @field_validator("chunk_id", "text", mode="before")
    @classmethod
    def _coerce_retrieved_text(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("retrieved chunk fields cannot be empty")
        return text

    def preview_text(self, *, limit: int = 160) -> str:
        """Return a short preview for compact RAG evidence display."""
        value = " ".join(self.text.split()).strip()
        if len(value) <= limit:
            return value
        return value[: limit - 1].rstrip() + "…"


class StrategyDecisionDocument(BaseModel):
    """One active planning or strategy document used for Pareto selection."""

    model_config = ConfigDict(extra="forbid")

    document_name: str
    document_type: Literal["master_plan", "development_strategy", "custom"] = "custom"
    source: Literal["inline_text", "file_path"] = "inline_text"
    source_ref: str | None = None
    text: str
    summary: str = ""
    extracted_priorities: list[str] = Field(default_factory=list)
    chunks: list[DecisionDocumentChunk] = Field(default_factory=list)
    retrieval_backend: str = "ollama_inmemory_vectorstore"

    @field_validator("document_name", "text", mode="before")
    @classmethod
    def _coerce_required_text(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("required text field cannot be empty")
        return text

    def preview_text(self, *, limit: int = 220) -> str:
        """Return a compact preview for context and notebook display."""
        value = " ".join(self.text.split()).strip()
        if len(value) <= limit:
            return value
        return value[: limit - 1].rstrip() + "…"

    @property
    def chunk_count(self) -> int:
        """Return the number of indexed RAG chunks for this document."""
        return len(self.chunks)


class ParetoDocumentCandidateReview(BaseModel):
    """One reviewed Pareto candidate scored against the active document."""

    model_config = ConfigDict(extra="forbid")

    solution_number: int
    scenario_id: str
    title: str = ""
    alignment_score: float = Field(ge=0.0, le=1.0)
    alignment_summary: str
    decisive_factors: list[str] = Field(default_factory=list)


class ParetoDocumentSelectionResult(BaseModel):
    """Best Pareto solution selected against a planning or strategy document."""

    model_config = ConfigDict(extra="forbid")

    document_name: str
    document_type: Literal["master_plan", "development_strategy", "custom"] = "custom"
    selection_goal: str
    document_summary: str
    selected_solution_number: int
    selected_scenario_id: str
    selected_title: str = ""
    rationale: str
    criteria_used: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedDocumentChunk] = Field(default_factory=list)
    candidate_reviews: list[ParetoDocumentCandidateReview] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)

    def to_text(self) -> str:
        """Render a concise human-readable decision summary."""
        lines = [
            "Выбрано решение по документу.",
            f"Документ: {self.document_name}",
            f"Решение: №{self.selected_solution_number} ({self.selected_scenario_id})",
        ]
        if self.selected_title:
            lines.append(f"Сценарий: {self.selected_title}")
        if self.document_summary:
            lines.append(f"Контекст документа: {self.document_summary}")
        if self.criteria_used:
            lines.append("Критерии:")
            lines.extend(f"- {item}" for item in self.criteria_used)
        if self.retrieved_chunks:
            lines.append("RAG-фрагменты:")
            lines.extend(
                f"- {item.chunk_id} (score={item.score:.2f}): {item.preview_text()}"
                for item in self.retrieved_chunks[:4]
            )
        lines.append(f"Обоснование: {self.rationale}")
        if self.candidate_reviews:
            lines.append("Кандидаты:")
            lines.extend(
                f"- №{item.solution_number}: {item.alignment_summary} "
                f"(score={item.alignment_score:.2f})"
                for item in self.candidate_reviews
            )
        if self.risks:
            lines.append("Риски:")
            lines.extend(f"- {item}" for item in self.risks)
        return "\n".join(lines).strip()


class VerificationIssue(BaseModel):
    """One issue found while verifying an agent result."""

    model_config = ConfigDict(extra="forbid")

    severity: Literal["warning", "error"]
    message: str


class VerificationReport(BaseModel):
    """Compact verification summary for one agent turn."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "warning", "error"]
    summary: str
    issues: list[VerificationIssue] = Field(default_factory=list)
    verified_facts: list[str] = Field(default_factory=list)

    def to_text(self) -> str:
        """Render a concise markdown-friendly summary."""
        lines = [f"Статус проверки: {self.status}", self.summary]
        if self.verified_facts:
            lines.append("Подтверждено:")
            lines.extend(f"- {item}" for item in self.verified_facts)
        if self.issues:
            lines.append("Замечания:")
            lines.extend(f"- [{item.severity}] {item.message}" for item in self.issues)
        return "\n".join(lines).strip()


class SessionTask(BaseModel):
    """One completed or failed action in a thread-local session ledger."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    route: str
    title: str
    status: Literal["completed", "failed"]
    user_request: str
    response_summary: str = ""
    verification_status: Literal["ok", "warning", "error"] = "ok"


class SessionArtifact(BaseModel):
    """Compact description of a runtime artifact produced in a session."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    label: str
    route: str
    target_id: int | None = None
    solution_number: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ResearchEntity(BaseModel):
    """One entity extracted from the user request during the research pass."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    value: str


class ResearchBrief(BaseModel):
    """Compact research pass over one user request before execution."""

    model_config = ConfigDict(extra="forbid")

    route: str
    complexity: Literal["simple", "multi_step"]
    summary: str
    active_target_id: int | None = None
    requested_solution_number: int | None = None
    requested_outputs: list[str] = Field(default_factory=list)
    entities: list[ResearchEntity] = Field(default_factory=list)

    def to_text(self) -> str:
        """Render a compact human-readable research brief."""
        lines = [
            f"маршрут: {self.route}",
            f"сложность: {self.complexity}",
            f"сводка: {self.summary}",
        ]
        if self.active_target_id is not None:
            lines.append(f"target_id: {self.active_target_id}")
        if self.requested_solution_number is not None:
            lines.append(f"solution_number: {self.requested_solution_number}")
        if self.requested_outputs:
            lines.append("ожидаемые результаты:")
            lines.extend(f"- {item}" for item in self.requested_outputs)
        if self.entities:
            lines.append("выделенные сущности:")
            lines.extend(f"- {item.kind}: {item.value}" for item in self.entities)
        return "\n".join(lines).strip()


class ExecutionPlanStep(BaseModel):
    """One high-level step in the execution plan for the current request."""

    model_config = ConfigDict(extra="forbid")

    step_id: str
    title: str
    rationale: str


class ExecutionPlan(BaseModel):
    """High-level plan for how the current request should be handled."""

    model_config = ConfigDict(extra="forbid")

    route: str
    summary: str
    complexity: Literal["simple", "multi_step"]
    steps: list[ExecutionPlanStep] = Field(default_factory=list)

    def to_text(self) -> str:
        """Render a compact human-readable plan."""
        lines = [
            f"маршрут: {self.route}",
            f"сложность: {self.complexity}",
            f"цель: {self.summary}",
        ]
        if self.steps:
            lines.append("шаги:")
            lines.extend(f"- {item.step_id}. {item.title} — {item.rationale}" for item in self.steps)
        return "\n".join(lines).strip()


class ConfirmationGate(BaseModel):
    """Expert confirmation checkpoint before executing a multi-step request."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["pending", "approved", "cancelled"]
    route: str
    original_user_request: str
    prompt: str
    involved_tools: list[str] = Field(default_factory=list)
    research_brief: ResearchBrief
    execution_plan: ExecutionPlan

    def to_text(self) -> str:
        """Render the gate as a user-facing confirmation message."""
        lines = [self.prompt]
        if self.involved_tools:
            lines.append("Планируемые инструменты:")
            lines.extend(f"- {item}" for item in self.involved_tools)
        lines.append("Research brief:")
        lines.append(self.research_brief.to_text())
        lines.append("План выполнения:")
        lines.append(self.execution_plan.to_text())
        lines.append("Если всё верно, ответьте: да")
        lines.append("Если нужно скорректировать план, ответьте: нет или напишите уточнение.")
        return "\n".join(lines).strip()


class SessionMemorySnapshot(BaseModel):
    """Durable session context distilled from recent turns and runtime outputs."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str
    latest_route: str | None = None
    active_target_id: int | None = None
    latest_solution_number: int | None = None
    latest_recommended_solution_number: int | None = None
    latest_visualization_route: str | None = None
    active_decision_document_name: str | None = None
    latest_response: str = ""
    algorithm_overrides: dict[str, int] | None = None
    recent_tasks: list[SessionTask] = Field(default_factory=list)
    recent_artifacts: list[SessionArtifact] = Field(default_factory=list)
    latest_verification_report: VerificationReport | None = None
    latest_research_brief: ResearchBrief | None = None
    active_plan: ExecutionPlan | None = None
    pending_confirmation_gate: ConfirmationGate | None = None
    durable_facts: list[str] = Field(default_factory=list)

    def to_text(self) -> str:
        """Render a compact human-readable memory block."""
        lines: list[str] = [f"thread_id: {self.thread_id}"]
        if self.latest_route:
            lines.append(f"последний маршрут: {self.latest_route}")
        if self.active_target_id is not None:
            lines.append(f"активный target_id: {self.active_target_id}")
        if self.latest_solution_number is not None:
            lines.append(f"последнее решение: {self.latest_solution_number}")
        if self.latest_recommended_solution_number is not None:
            lines.append(
                f"последнее рекомендованное решение: {self.latest_recommended_solution_number}"
            )
        if self.latest_visualization_route:
            lines.append(f"последняя визуализация: {self.latest_visualization_route}")
        if self.active_decision_document_name:
            lines.append(f"активный документ: {self.active_decision_document_name}")
        if self.algorithm_overrides:
            overrides = ", ".join(
                f"{key}={value}" for key, value in sorted(self.algorithm_overrides.items())
            )
            lines.append(f"параметры алгоритма: {overrides}")
        if self.durable_facts:
            lines.append("устойчивые факты:")
            lines.extend(f"- {item}" for item in self.durable_facts)
        if self.recent_tasks:
            lines.append("последние задачи:")
            lines.extend(
                f"- [{task.status}] {task.title} (route={task.route}, verify={task.verification_status})"
                for task in self.recent_tasks[-4:]
            )
        if self.recent_artifacts:
            lines.append("последние артефакты:")
            lines.extend(
                f"- {artifact.label} (kind={artifact.kind})"
                for artifact in self.recent_artifacts[-4:]
            )
        if self.latest_research_brief is not None:
            lines.append("research brief:")
            lines.append(self.latest_research_brief.to_text())
        if self.active_plan is not None:
            lines.append("active plan:")
            lines.append(self.active_plan.to_text())
        if self.pending_confirmation_gate is not None:
            lines.append("pending confirmation:")
            lines.append(self.pending_confirmation_gate.to_text())
        if self.latest_verification_report is not None:
            lines.append(self.latest_verification_report.to_text())
        if self.latest_response:
            lines.append(f"последний ответ: {self.latest_response}")
        return "\n".join(lines).strip()


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
        "block_parameters",
        "document_rag",
        "district_optimization",
        "general_qa",
        "unsupported",
    ]
    reasoning: str = Field(description="Short explanation of why the route was chosen.")
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Router confidence from 0.0 to 1.0.",
    )


class UrbanomyOrchestratorResult(BaseModel):
    """Final result returned by the top-level Urbanomy orchestrator."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    user_request: str
    route: Literal[
        "visualization",
        "land_value_visualization",
        "target_block_visualization",
        "block_parameters",
        "document_rag",
        "district_optimization",
        "general_qa",
        "unsupported",
    ]
    reasoning: str
    response: str = ""
    visualization_result: VisualizationResult | None = None
    target_block_visualization_result: TargetBlockVisualizationResult | None = None
    block_parameters_result: Any | None = None
    document_rag_result: Any | None = None
    district_optimization_result: Any | None = None
    verification_report: VerificationReport | None = None
    session_memory: SessionMemorySnapshot | None = None
    research_brief: ResearchBrief | None = None
    execution_plan: ExecutionPlan | None = None
    confirmation_gate: ConfirmationGate | None = None

    def __repr__(self) -> str:
        if self.response:
            return self.response
        return super().__repr__()

    def __str__(self) -> str:
        return self.response or super().__repr__()
