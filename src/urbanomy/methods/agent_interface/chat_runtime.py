"""Default runtime factory for the local Urbanomy browser chat."""

from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd

from urbanomy.methods.land_value_modeling.constants import (
    CATEGORICAL_FEATURES,
    ORIGINAL_FEATURES,
)

from .models import DistrictOptimizationConfig
from .urbanomy_orchestrator import create_urbanomy_orchestrator


def create_chat_runtime():
    """Build the default orchestrator runtime for the local browser chat."""
    _load_repo_env()

    baseline_blocks_path = _required_path("URBANOMY_BASELINE_BLOCKS")
    baseline_blocks = gpd.read_file(baseline_blocks_path)

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Install llm dependencies before running chat, for example: pip install -e '.[llm]'"
        ) from exc

    model_name = (
        os.environ.get("OPENAI_MODEL", "").strip()
        or os.environ.get("CHAT_MODEL", "").strip()
        or "gpt-4o-mini"
    )
    api_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("API_KEY", "").strip()
        or "dummy"
    )
    base_url = _resolve_base_url()

    llm_kwargs = {
        "model": model_name,
        "temperature": float(os.environ.get("TEMPERATURE", "0") or "0"),
        "api_key": api_key,
    }
    if base_url:
        llm_kwargs["base_url"] = base_url

    llm = ChatOpenAI(**llm_kwargs)

    district_optimization_config = None
    model_path = os.environ.get("URBANOMY_PRICE_MODEL", "").strip()
    if model_path:
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise RuntimeError("CatBoost is required when URBANOMY_PRICE_MODEL is set.") from exc
        price_model = CatBoostRegressor()
        price_model.load_model(model_path)
        district_optimization_config = DistrictOptimizationConfig(
            model=price_model,
            orig_features=list(ORIGINAL_FEATURES),
            categorical_features=list(CATEGORICAL_FEATURES),
        )

    orchestrator = create_urbanomy_orchestrator(
        llm=llm,
        baseline_blocks=baseline_blocks,
        district_optimization_config=district_optimization_config,
    )
    return {
        "orchestrator": orchestrator,
        "title": "Urbanomy Local Agent Chat",
        "welcome": (
            "Чат подключён к локальному Urbanomy orchestrator. "
            "Если задан URBANOMY_PRICE_MODEL, ветка district optimization тоже активна."
        ),
    }


def _required_path(env_name: str) -> str:
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise RuntimeError(f"Set {env_name} to a readable baseline blocks file path.")
    path = Path(value)
    if not path.exists():
        raise RuntimeError(f"Path from {env_name} does not exist: {path}")
    return str(path)


def _resolve_base_url() -> str | None:
    value = os.environ.get("OPENAI_BASE_URL", "").strip() or os.environ.get("BASE_URL", "").strip()
    if not value:
        return None
    return value if value.rstrip("/").endswith("/v1") else value.rstrip("/") + "/v1"


def _load_repo_env() -> None:
    env_path = Path(__file__).resolve().parents[4] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip("'").strip('"')
