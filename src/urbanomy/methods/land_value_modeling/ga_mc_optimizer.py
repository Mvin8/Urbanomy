"""Scenario tool for Pareto-front solutions."""

from __future__ import annotations

import json
import math
import re
import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from blocksnet.analysis.indicators import calculate_density_indicators
from blocksnet.analysis.morphotypes import get_strelka_morphotypes
from blocksnet.enums import LandUse
from catboost import CatBoostRegressor
from geopandas import GeoDataFrame
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy
from pymoo.algorithms.moo.nsga2 import NSGA2  # Импорт алгоритма
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.optimize import minimize  # Функция для запуска оптимизации
from pymoo.problems import get_problem  # Для получения тестовых задач
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter  # Для визуализации

from urbanomy.methods.investment_potential import (
    DEFAULT_BENCHMARKS_RU,
    InvestmentAttractivenessAnalyzer,
)
from .constants import (
    BlockColumn,
    CATEGORICAL_FEATURES,
    DEFAULT_SQM_PER_PERSON,
    ORIGINAL_FEATURES,
    RADIUS_LIST,
    ScenarioResultKey,
)
from .land_price_estimation import LandPriceEstimator
from .scenario_modification import ScenarioTEPModifier, plot_scenario_impact


def _json_ready(value: Any) -> Any:
    """Convert nested values to JSON-safe primitives."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if hasattr(value, "item"):
        try:
            return _json_ready(value.item())
        except Exception:
            return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return str(value)


def _message_to_text(message: Any) -> str:
    """Extract plain text from raw strings or LangChain-style messages."""
    if isinstance(message, str):
        return message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if text:
                    chunks.append(str(text))
        return "\n".join(chunks).strip()
    return str(message)


def _strip_code_fences(text: str) -> str:
    """Remove Markdown fences around JSON content."""
    cleaned = str(text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


class StrategicAlignmentScorer:
    """Score scenarios with an LLM or baseline."""

    def __init__(
        self,
        *,
        llm: Any | None = None,
        baseline: Any | None = None,
        prompt: str,
        max_retries: int = 2,
    ) -> None:
        prompt_text = str(prompt).strip()
        if not prompt_text:
            raise ValueError("prompt must be a non-empty string.")
        if llm is None and baseline is None:
            raise ValueError("Either llm or baseline must be provided.")
        self.llm = llm
        self.baseline = baseline
        self.prompt = prompt_text
        self.max_retries = max(0, int(max_retries))
        self._cache: dict[str, dict[str, Any]] = {}

    def __deepcopy__(self, memo: dict[int, Any]):
        """Keep the underlying LLM by reference to avoid deepcopy/pickle failures in pymoo history."""
        copied = self.__class__.__new__(self.__class__)
        memo[id(self)] = copied
        copied.llm = self.llm
        copied.baseline = self.baseline
        copied.prompt = self.prompt
        copied.max_retries = self.max_retries
        copied._cache = copy.deepcopy(self._cache, memo)
        return copied

    def _invoke(self, prompt: str) -> str:
        if self.baseline is not None:
            if hasattr(self.baseline, "invoke_state"):
                return str(self.baseline.invoke_state(prompt).get("output", ""))
            response = self.baseline.invoke(prompt)
            return _message_to_text(response)
        response = self.llm.invoke(prompt)
        return _message_to_text(response)

    def _build_prompt(self, *, candidate_payload: Mapping[str, Any]) -> str:
        return (
            f"{self.prompt}\n\n"
            "Сценарий:\n"
            f"{json.dumps(_json_ready(candidate_payload), ensure_ascii=False, separators=(',', ':'))}"
        )

    def _repair_prompt(self, *, raw_text: str, error_text: str) -> str:
        return (
            "Исправь ответ так, чтобы он стал валидным JSON без markdown.\n"
            'Верни только JSON формата {"score": 0.0, "reasoning": "..."}. '
            "Поле score должно быть числом от 0 до 1.\n\n"
            f"Ошибка: {error_text}\n\n"
            f"Исходный ответ:\n{raw_text}"
        )

    def _parse_response(self, raw_text: str) -> tuple[float, str]:
        cleaned = _strip_code_fences(raw_text)
        if not cleaned:
            raise ValueError("Empty LLM response.")

        try:
            numeric = float(cleaned)
        except ValueError:
            numeric = None
        if numeric is not None:
            return max(0.0, min(1.0, numeric)), ""

        payload: dict[str, Any] | None = None
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, Mapping):
                payload = dict(parsed)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, Mapping):
                    payload = dict(parsed)

        if payload is None:
            raise ValueError(f"Unable to parse scorer JSON: {cleaned}")

        score_value = payload.get("score", payload.get("alignment_score", payload.get("ser_alignment_score")))
        if score_value is None:
            raise ValueError("Scorer JSON does not contain a score field.")

        score = max(0.0, min(1.0, float(score_value)))
        reasoning = str(
            payload.get("reasoning", payload.get("rationale", payload.get("summary", "")))
        ).strip()
        return score, reasoning

    def score_candidate(
        self,
        *,
        params_repaired: Mapping[str, Any],
        land_value_gain: float,
        investor_npv: float,
    ) -> dict[str, Any]:
        candidate_payload = {
            "params_repaired": _json_ready(params_repaired),
            "land_value_gain": float(land_value_gain),
            "investor_npv": float(investor_npv),
        }
        cache_key = json.dumps(candidate_payload, ensure_ascii=False, sort_keys=True)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = self._build_prompt(candidate_payload=candidate_payload)
        raw_text = self._invoke(prompt)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                score, reasoning = self._parse_response(raw_text)
                result = {
                    "score": float(score),
                    "reasoning": reasoning,
                    "candidate_payload": candidate_payload,
                }
                self._cache[cache_key] = result
                return result
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                raw_text = self._invoke(self._repair_prompt(raw_text=raw_text, error_text=str(exc)))

        raise ValueError(f"Unable to obtain a valid strategic alignment score from LLM: {last_error}") from last_error


def build_nsga3_reference_directions(
    *,
    n_obj: int,
    pop_size: int | None = None,
    n_partitions: int | None = None,
) -> np.ndarray:
    """Build Das-Dennis reference directions for NSGA-III."""
    if n_obj < 3:
        raise ValueError("NSGA-III reference directions are intended for 3 or more objectives.")

    if n_partitions is None:
        target_pop = max(int(pop_size or 0), 1)
        n_partitions = 1
        while math.comb(n_partitions + n_obj - 1, n_obj - 1) < target_pop:
            n_partitions += 1
    return get_reference_directions("das-dennis", n_obj, n_partitions=int(n_partitions))


class DistrictProblem(Problem):
    def __init__(
        self,
        blocks: GeoDataFrame,
        model: CatBoostRegressor,
        estimator_kwargs: Dict[str, Any],
        constraints: Dict[str, Dict[str, Any]],
        target_id: Any,
        benchmarks: Mapping[LandUse, Dict[str, Any]] | None = None,
        target_id_column: str = BlockColumn.ID.value,
        strategic_alignment_scorer: StrategicAlignmentScorer | None = None,
        log_optimization: bool = False,
    ):
        self.blocks = blocks
        self.model = model
        self.estimator_kwargs = estimator_kwargs
        self.constraints = constraints
        self.benchmarks = benchmarks or DEFAULT_BENCHMARKS_RU
        self.target_id = target_id
        self.target_id_column = target_id_column
        self.strategic_alignment_scorer = strategic_alignment_scorer
        self.log_optimization = bool(log_optimization)
        self.optimization_log: list[dict[str, Any]] = []
        self.objective_names = (
            ["land_value_total", "investor_npv", "llm score"]
            if strategic_alignment_scorer is not None
            else ["land_value_total", "investor_npv"]
        )
        self.var_names = list(constraints.keys())
        self._strategic_alignment_results: dict[str, dict[str, Any]] = {}
        self._baseline_land_value: float | None = None
        land_use_vars = [
            "residential",
            "business",
            "recreation",
            "industrial",
            "transport",
            "special",
            "agriculture",
        ]
        self.land_use_indices = [i for i, name in enumerate(self.var_names) if name in land_use_vars]

        super().__init__(
            n_var=len(constraints),
            n_obj=len(self.objective_names),
            n_constr=0,
            xl=np.array([c["min"] for c in constraints.values()]),
            xu=np.array([c["max"] for c in constraints.values()]),
            type_var=np.float64,
        )

    def _resolve_target_label(self):
        if self.target_id_column not in self.blocks.columns:
            raise KeyError(
                f"Target id column {self.target_id_column!r} is not present in the dataset."
            )

        matches = self.blocks.index[self.blocks[self.target_id_column] == self.target_id]
        if len(matches) == 0:
            raise KeyError(
                f"Block with {self.target_id_column}={self.target_id!r} is not present in the dataset."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple blocks found for {self.target_id_column}={self.target_id!r}; "
                "the identifier must be unique."
            )
        return matches[0]

    def _recompute_morphotype(self, genome: dict) -> Any:
        """Recompute morphotype for the target block using updated genome values."""
        target_label = self._resolve_target_label()
        updated = self.blocks.copy()
        for key, value in genome.items():
            if key in updated.columns:
                updated.at[target_label, key] = value

        try:
            morphotypes = get_strelka_morphotypes(updated)
        except Exception:
            return updated.at[target_label, "morphotype"] if "morphotype" in updated.columns else None

        if "morphotype" in morphotypes.columns and target_label in morphotypes.index:
            return morphotypes.at[target_label, "morphotype"]
        return updated.at[target_label, "morphotype"] if "morphotype" in updated.columns else None

    def _strategic_alignment_cache_key(
        self,
        *,
        params_repaired: Mapping[str, Any],
        land_value_after: float,
        investor_npv: float,
    ) -> str:
        payload = {
            "params_repaired": _json_ready(params_repaired),
            "land_value_gain": round(float(land_value_after - self.baseline_land_value()), 6),
            "investor_npv": round(float(investor_npv), 6),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _repair_genome(self, genome: dict) -> dict:
        target_label = self._resolve_target_label()
        row = self.blocks.loc[target_label]
        site_area = float(row["site_area"])
        footprint_max = 0.8 * site_area
        land_use_keys = [
            "residential",
            "business",
            "recreation",
            "industrial",
            "transport",
            "special",
            "agriculture",
        ]

        cleaned_shares = {k: max(float(genome.get(k, 0.0)), 0.0) for k in land_use_keys}
        total = sum(cleaned_shares.values())
        if total > 0:
            for k in land_use_keys:
                genome[k] = cleaned_shares[k] / total
        else:
            genome["residential"] = 1.0
            for k in land_use_keys[1:]:
                genome[k] = 0.0

        genome["share"] = float(genome["residential"])

        land_use_map = {
            "residential": LandUse.RESIDENTIAL,
            "business": LandUse.BUSINESS,
            "recreation": LandUse.RECREATION,
            "industrial": LandUse.INDUSTRIAL,
            "transport": LandUse.TRANSPORT,
            "special": LandUse.SPECIAL,
            "agriculture": LandUse.AGRICULTURE,
        }
        dominant_key = max(land_use_keys, key=lambda k: genome[k])
        genome["land_use"] = land_use_map[dominant_key]

        footprint = float(genome.get("footprint_area", row.get("footprint_area", 0.0)))
        footprint = float(np.clip(footprint, 0.0, footprint_max))
        genome["footprint_area"] = footprint

        if "l" in self.var_names:
            l_value = float(genome.get("l", row.get("l", 1.0)))
            l_value = max(l_value, 1.0)
            build = footprint * l_value
            genome["l"] = l_value
        else:
            build = max(float(genome.get("build_floor_area", row.get("build_floor_area", 0.0))), footprint)
            genome["l"] = (build / footprint) if footprint > 0 else 0.0

        residential_share = float(np.clip(genome.get("residential", 0.0), 0.0, 1.0))
        live = float(np.clip(build * residential_share * 0.8, 0.0, build))
        genome["mxi"] = (live / build) if build > 0 else 0.0

        non_live = max(build - live, 0.0)

        genome["build_floor_area"] = build
        genome["living_area"] = live
        genome["non_living_area"] = non_live
        genome["population"] = live / DEFAULT_SQM_PER_PERSON if DEFAULT_SQM_PER_PERSON > 0 else 0.0

        genome["fsi"] = build / site_area if site_area > 0 else 0.0
        genome["gsi"] = footprint / site_area if site_area > 0 else 0.0
        genome["morphotype"] = self._recompute_morphotype(genome)

        return genome

    def evaluate_catboost(
        self,
        geonome: GeoDataFrame,
        model: CatBoostRegressor,
        orig_features: Sequence[str] | None = None,
        cat_features: Sequence[str] | None = None,
        radius_list: Sequence[float] | None = None,
    ):
        features = tuple(orig_features) if orig_features is not None else ORIGINAL_FEATURES
        cats = tuple(cat_features) if cat_features is not None else CATEGORICAL_FEATURES
        radii = tuple(radius_list) if radius_list is not None else RADIUS_LIST

        estimator = LandPriceEstimator(
            model=model,
            blocks=geonome,
            radius_list=radii,
            orig_features=features,
            categorical_features=cats,
        )
        estimated = estimator.predict()
        return estimated["land_value"].sum()

    def baseline_land_value(self) -> float:
        if self._baseline_land_value is None:
            self._baseline_land_value = float(
                self.evaluate_catboost(
                    geonome=self.blocks,
                    model=self.model,
                    orig_features=self.estimator_kwargs["orig_features"],
                    cat_features=self.estimator_kwargs["categorical_features"],
                    radius_list=None,
                )
            )
        return self._baseline_land_value

    def _mark_project_block(self, geonome: GeoDataFrame) -> GeoDataFrame:
        """Mark the target block as project using the passed target index."""
        marked = geonome.copy()
        marked["is_project"] = False
        target_label = self._resolve_target_label()
        if target_label not in marked.index:
            raise KeyError(
                f"Cannot set is_project for {self.target_id_column}={self.target_id!r}: row not found."
            )
        marked.loc[target_label, "is_project"] = True
        return marked

    def evaluate_investment_potential(
        self,
        geonome: GeoDataFrame,
        benchmarks: Mapping[LandUse, Dict[str, Any]],
        discount_rate: float = 0.18,
    ):
        geonome_marked = self._mark_project_block(geonome)
        analyzer = InvestmentAttractivenessAnalyzer(benchmarks=benchmarks)
        summary = analyzer.calculate_investment_metrics(
            geonome_marked,
            discount_rate=discount_rate,
            show_project_totals=False,
            show_warning=False,
        )

        npv_series = pd.to_numeric(summary["NPV"], errors="coerce").dropna()
        if npv_series.empty:
            raise ValueError("NPV is not available for the selected project block.")
        return float(npv_series.iloc[0])

    def evaluate_strategic_alignment(
        self,
        *,
        params_repaired: Mapping[str, Any],
        land_value_after: float,
        investor_npv: float,
    ) -> dict[str, Any] | None:
        if self.strategic_alignment_scorer is None:
            return None

        key = self._strategic_alignment_cache_key(
            params_repaired=params_repaired,
            land_value_after=land_value_after,
            investor_npv=investor_npv,
        )
        cached = self._strategic_alignment_results.get(key)
        if cached is not None:
            return cached

        scored = self.strategic_alignment_scorer.score_candidate(
            params_repaired=params_repaired,
            land_value_gain=land_value_after - self.baseline_land_value(),
            investor_npv=investor_npv,
        )
        self._strategic_alignment_results[key] = scored
        return scored

    def lookup_strategic_alignment(
        self,
        *,
        params_repaired: Mapping[str, Any],
        land_value_after: float,
        investor_npv: float,
    ) -> dict[str, Any] | None:
        if self.strategic_alignment_scorer is None:
            return None
        key = self._strategic_alignment_cache_key(
            params_repaired=params_repaired,
            land_value_after=land_value_after,
            investor_npv=investor_npv,
        )
        return self._strategic_alignment_results.get(key)

    def _evaluate(self, X, out, *args, **kwargs):
        n = X.shape[0]
        F = np.zeros((n, len(self.objective_names)))

        for i in range(n):
            genome = X[i]
            changes = {name: genome[j] for j, name in enumerate(self.var_names)}
            changes = self._repair_genome(changes)
            modifier = ScenarioTEPModifier(self.blocks)
            blocks_after = modifier.apply(
                self.target_id,
                changes,
                target_id_column=self.target_id_column,
            )

            land_value = self.evaluate_catboost(
                geonome=blocks_after,
                model=self.model,
                orig_features=self.estimator_kwargs["orig_features"],
                cat_features=self.estimator_kwargs["categorical_features"],
                radius_list=RADIUS_LIST,
            )

            investment_potential = self.evaluate_investment_potential(
                geonome=blocks_after,
                benchmarks=self.benchmarks,
                discount_rate=0.18,
            )

            F[i, 0] = -land_value
            F[i, 1] = -investment_potential
            alignment = None
            if self.strategic_alignment_scorer is not None:
                alignment = self.evaluate_strategic_alignment(
                    params_repaired=changes,
                    land_value_after=land_value,
                    investor_npv=investment_potential,
                )
                F[i, 2] = -float(alignment["score"]) if alignment is not None else 0.0
            if self.log_optimization:
                self.optimization_log.append(
                    {
                        "eval_id": len(self.optimization_log),
                        "target_id": _json_ready(self.target_id),
                        "params_repaired": _json_ready(changes),
                        "land_value_total": float(land_value),
                        "land_value_gain": float(land_value - self.baseline_land_value()),
                        "investor_npv": float(investment_potential),
                        "llm score": (
                            float(alignment["score"]) if alignment is not None else None
                        ),
                        "ser_alignment_reasoning": (
                            str(alignment.get("reasoning", "")).strip()
                            if alignment is not None
                            else None
                        ),
                    }
                )

        out["F"] = F


class NSGA2GenerationStatsCallback(Callback):
    """Print compact per-generation objective stats instead of per-individual logs."""

    def __init__(self, every: int = 1) -> None:
        super().__init__()
        self.every = max(1, int(every))
        self.history: list[dict[str, float]] = []

    def notify(self, algorithm) -> None:
        gen = int(algorithm.n_gen)
        if gen % self.every != 0:
            return

        F = algorithm.pop.get("F")
        if F is None or len(F) == 0:
            return

        f_land = -np.asarray(F[:, 0], dtype=float)
        f_npv = -np.asarray(F[:, 1], dtype=float)
        stats = {
            "gen": gen,
            "land_min": float(np.min(f_land)),
            "land_median": float(np.median(f_land)),
            "land_max": float(np.max(f_land)),
            "npv_min": float(np.min(f_npv)),
            "npv_median": float(np.median(f_npv)),
            "npv_max": float(np.max(f_npv)),
        }
        ser_text = ""
        if F.shape[1] > 2:
            f_ser = -np.asarray(F[:, 2], dtype=float)
            stats.update(
                {
                    "ser_min": float(np.min(f_ser)),
                    "ser_median": float(np.median(f_ser)),
                    "ser_max": float(np.max(f_ser)),
                }
            )
            ser_text = " | SER min/med/max={ser_min:.3f}/{ser_median:.3f}/{ser_max:.3f}".format(**stats)
        self.history.append(stats)
        print(
            "[gen {gen:03d}] "
            "land_value min/med/max={land_min:,.0f}/{land_median:,.0f}/{land_max:,.0f} | "
            "NPV min/med/max={npv_min:,.0f}/{npv_median:,.0f}/{npv_max:,.0f}".format(**stats)
            + ser_text
        )


def _run_with_strategic_alignment(
    *,
    blocks: GeoDataFrame,
    model: CatBoostRegressor,
    estimator_kwargs: Dict[str, Any],
    constraints: Dict[str, Dict[str, Any]],
    target_id: Any,
    llm: Any | None,
    baseline: Any | None,
    prompt: str,
    benchmarks: Mapping[LandUse, Dict[str, Any]] | None,
    target_id_column: str,
    algorithm,
    n_gen: int,
    seed: int,
    save_history: bool,
    verbose: bool,
    scorer_max_retries: int,
    log_optimization: bool,
    optimization_log_path: str | Path | None,
):
    scorer = StrategicAlignmentScorer(
        llm=llm,
        baseline=baseline,
        prompt=prompt,
        max_retries=scorer_max_retries,
    )
    problem = DistrictProblem(
        blocks=blocks,
        model=model,
        estimator_kwargs=estimator_kwargs,
        constraints=constraints,
        target_id=target_id,
        benchmarks=benchmarks,
        target_id_column=target_id_column,
        strategic_alignment_scorer=scorer,
        log_optimization=bool(log_optimization or optimization_log_path),
    )
    result = minimize(
        problem,
        algorithm(problem),
        ("n_gen", int(n_gen)),
        seed=int(seed),
        verbose=bool(verbose),
        save_history=bool(save_history),
    )
    if optimization_log_path:
        path = Path(optimization_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(problem.optimization_log).to_json(
            path,
            orient="records",
            lines=True,
            force_ascii=False,
        )
    return result, problem


def run_nsga3_with_strategic_alignment(
    *,
    blocks: GeoDataFrame,
    model: CatBoostRegressor,
    estimator_kwargs: Dict[str, Any],
    constraints: Dict[str, Dict[str, Any]],
    target_id: Any,
    prompt: str,
    llm: Any | None = None,
    baseline: Any | None = None,
    benchmarks: Mapping[LandUse, Dict[str, Any]] | None = None,
    target_id_column: str = BlockColumn.ID.value,
    pop_size: int = 20,
    n_gen: int = 20,
    seed: int = 42,
    eliminate_duplicates: bool = True,
    save_history: bool = True,
    verbose: bool = True,
    ref_dirs: np.ndarray | None = None,
    ref_dir_partitions: int | None = None,
    scorer_max_retries: int = 2,
    log_optimization: bool = False,
    optimization_log_path: str | Path | None = None,
):
    """Run district optimization with a third LLM-based strategic objective via NSGA-III."""
    def algorithm(problem):
        directions = ref_dirs
        if directions is None:
            directions = build_nsga3_reference_directions(
                n_obj=problem.n_obj,
                pop_size=pop_size,
                n_partitions=ref_dir_partitions,
            )
        return NSGA3(
            ref_dirs=directions,
            pop_size=max(int(pop_size), len(directions)),
            eliminate_duplicates=bool(eliminate_duplicates),
        )

    return _run_with_strategic_alignment(
        blocks=blocks,
        model=model,
        estimator_kwargs=estimator_kwargs,
        constraints=constraints,
        target_id=target_id,
        llm=llm,
        baseline=baseline,
        prompt=prompt,
        benchmarks=benchmarks,
        target_id_column=target_id_column,
        algorithm=algorithm,
        n_gen=n_gen,
        seed=seed,
        save_history=save_history,
        verbose=verbose,
        scorer_max_retries=scorer_max_retries,
        log_optimization=log_optimization,
        optimization_log_path=optimization_log_path,
    )


def run_nsga2_with_strategic_alignment(
    *,
    blocks: GeoDataFrame,
    model: CatBoostRegressor,
    estimator_kwargs: Dict[str, Any],
    constraints: Dict[str, Dict[str, Any]],
    target_id: Any,
    prompt: str,
    llm: Any | None = None,
    baseline: Any | None = None,
    benchmarks: Mapping[LandUse, Dict[str, Any]] | None = None,
    target_id_column: str = BlockColumn.ID.value,
    pop_size: int = 20,
    n_gen: int = 20,
    seed: int = 42,
    eliminate_duplicates: bool = True,
    save_history: bool = True,
    verbose: bool = True,
    scorer_max_retries: int = 2,
    log_optimization: bool = False,
    optimization_log_path: str | Path | None = None,
):
    """Run district optimization with the same LLM-based third objective via NSGA-II."""
    return _run_with_strategic_alignment(
        blocks=blocks,
        model=model,
        estimator_kwargs=estimator_kwargs,
        constraints=constraints,
        target_id=target_id,
        llm=llm,
        baseline=baseline,
        prompt=prompt,
        benchmarks=benchmarks,
        target_id_column=target_id_column,
        algorithm=lambda _: NSGA2(
            pop_size=int(pop_size),
            eliminate_duplicates=bool(eliminate_duplicates),
        ),
        n_gen=n_gen,
        seed=seed,
        save_history=save_history,
        verbose=verbose,
        scorer_max_retries=scorer_max_retries,
        log_optimization=log_optimization,
        optimization_log_path=optimization_log_path,
    )
