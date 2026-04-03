from types import SimpleNamespace

import numpy as np
import pandas as pd

from urbanomy.methods.land_value_modeling.ga_mc_optimizer import (
    StrategicAlignmentScorer,
    build_nsga3_reference_directions,
)
from urbanomy.methods.land_value_modeling.pareto_front_dataframe import build_pareto_front_dataframe


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


def test_strategic_alignment_scorer_parses_json_and_caches():
    llm = FakeLLM('```json\n{"score": 0.82, "reasoning": "Сценарий поддерживает инновации и туризм."}\n```')
    scorer = StrategicAlignmentScorer(
        llm=llm,
        strategic_goals="Развитие инновационной экономики и международного туризма.",
    )
    params = {
        "land_use": "business",
        "business": 0.6,
        "recreation": 0.3,
        "residential": 0.1,
    }

    first = scorer.score_candidate(
        target_id=86,
        site_area=12_000.0,
        params_repaired=params,
        land_value_after=150_000_000.0,
        investor_npv=45_000_000.0,
    )
    second = scorer.score_candidate(
        target_id=86,
        site_area=12_000.0,
        params_repaired=params,
        land_value_after=150_000_000.0,
        investor_npv=45_000_000.0,
    )

    assert first["score"] == 0.82
    assert "туризм" in first["reasoning"].lower()
    assert second == first
    assert len(llm.calls) == 1


def test_build_nsga3_reference_directions_for_three_objectives():
    ref_dirs = build_nsga3_reference_directions(n_obj=3, pop_size=10)

    assert ref_dirs.shape[1] == 3
    assert len(ref_dirs) >= 10


def test_build_pareto_front_dataframe_includes_ser_alignment_columns():
    class FakeProblem:
        constraints = {
            "business": {"min": 0.0, "max": 1.0},
            "recreation": {"min": 0.0, "max": 1.0},
        }
        blocks = pd.DataFrame()
        model = object()
        estimator_kwargs = {"orig_features": [], "categorical_features": []}

        def evaluate_catboost(self, **kwargs):
            return 100.0

        def _repair_genome(self, genome):
            return {
                **genome,
                "land_use": "business",
                "business": float(genome["business"]),
                "recreation": float(genome["recreation"]),
            }

        def lookup_strategic_alignment(self, **kwargs):
            return {"reasoning": "Сценарий усиливает деловую активность и рекреацию."}

    result = SimpleNamespace(
        X=np.array([[0.7, 0.3]]),
        F=np.array([[-120.0, -80.0, -0.75]]),
        history=None,
    )

    df = build_pareto_front_dataframe(
        res=result,
        problem=FakeProblem(),
        baseline_land_value=100.0,
    )

    assert float(df.loc[0, "ser_alignment_score"]) == 0.75
    assert "ser_alignment_reasoning" in df.columns
    assert "SER score=0.75" in df.loc[0, "summary"]
