"""Reusable helpers for land value training with compact logging output."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from libpysal.weights import DistanceBand, Queen, lag_spatial
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, ParameterSampler
from tqdm.auto import tqdm

LOGGER_NAME = "land_value_training"


@dataclass
class TrainingConfig:
    """Configuration for spatial CV + CatBoost training."""

    feature_cols: Sequence[str]
    cat_features: Sequence[str]
    radius_list: Sequence[int]
    target_col: str = "log_total_price"
    n_clusters: int = 10
    inner_splits: int = 5
    outer_splits: int = 5
    iterations: int = 15000
    od_wait: int = 300
    seed: int = 42
    hpo_iter: int = 24
    param_grid: Optional[Mapping[str, Sequence[Any]]] = None


def setup_logger(log_path: Path | str = "training.log", *, console_level: int = logging.INFO) -> logging.Logger:
    """Create a dual handler logger (console + file) that plays well with tqdm."""

    log_path = Path(log_path)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path) for h in logger.handlers):
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(isinstance(h, TqdmLoggingHandler) for h in logger.handlers):
        stream_handler = TqdmLoggingHandler()
        stream_handler.setLevel(console_level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


class TqdmLoggingHandler(logging.Handler):
    """Log records through tqdm.write to avoid breaking progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        from tqdm.auto import tqdm as _tqdm

        try:
            msg = self.format(record)
            _tqdm.write(msg)
        except Exception:  # pragma: no cover - defensive
            self.handleError(record)


def build_radii_weights(df: pd.DataFrame, radii: Iterable[int]) -> Dict[int, DistanceBand]:
    weights: Dict[int, DistanceBand] = {}
    for r in radii:
        w = DistanceBand.from_dataframe(df, threshold=r, binary=True, silence_warnings=True)
        w.transform = "r"
        weights[r] = w
    return weights


def build_lags(df: pd.DataFrame, radii: Iterable[int], numeric_cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    weights = build_radii_weights(df, radii)
    new_cols: Dict[str, Any] = {}
    for feat in numeric_cols:
        vec = df[feat].fillna(df[feat].mean())
        for r, w in weights.items():
            new_cols[f"lag{r}_{feat}"] = lag_spatial(w, vec)
    for r, w in weights.items():
        neigh_len = pd.Series({idx: len(w.neighbors[idx]) for idx in w.id_order})
        new_cols[f"n_neighbors_{r}"] = neigh_len.reindex(df.index).to_numpy()
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def prep_cat(df: pd.DataFrame, cat_cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cat_cols:
        df[c] = df[c].astype("string").fillna("missing")
    return df


def feature_names(df: pd.DataFrame, base_cols: Sequence[str], target_col: str) -> List[str]:
    extra = [c for c in df.columns if c.startswith("lag") or c.startswith("n_neighbors_")]
    final = list(dict.fromkeys(list(base_cols) + extra))
    return [c for c in final if c != target_col]


def pools_from_frames(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    y_te: pd.Series,
    cat_cols: Sequence[str],
    feats: Sequence[str],
) -> Tuple[Pool, Pool]:
    tr_pool = Pool(X_tr[feats], label=y_tr, cat_features=cat_cols, feature_names=feats)
    te_pool = Pool(X_te[feats], label=y_te, cat_features=cat_cols, feature_names=feats)
    return tr_pool, te_pool


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def spatial_groups(blocks: pd.DataFrame, n_clusters: int, *, random_state: int = 42) -> np.ndarray:
    if {"x", "y"}.issubset(blocks.columns):
        coords = blocks[["x", "y"]].to_numpy()
    else:
        cent = blocks.geometry.centroid
        coords = np.column_stack([cent.x.values, cent.y.values])
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit_predict(coords)


def run_hpo(
    df_train: pd.DataFrame,
    target_col: str,
    numeric_feats: Sequence[str],
    cat_features: Sequence[str],
    radii: Sequence[int],
    groups: np.ndarray,
    *,
    params_grid: Mapping[str, Sequence[Any]],
    n_iter: int,
    iterations: int,
    od_wait: int,
    seed: int,
    inner_splits: int,
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], float]:
    sampler = list(ParameterSampler(params_grid, n_iter=n_iter, random_state=seed))
    gkf_inner = GroupKFold(n_splits=inner_splits)
    best_params: Dict[str, Any] | None = None
    best_cv = float("inf")

    logger.info("Starting HPO across %d trials", len(sampler))
    for params in tqdm(sampler, desc="Trials", leave=False):
        fold_rmses: List[float] = []
        for k, (tr_idx, va_idx) in enumerate(gkf_inner.split(df_train, df_train[target_col], groups), 1):
            tr_df = prep_cat(build_lags(df_train.iloc[tr_idx], radii, numeric_feats), cat_features)
            va_df = prep_cat(build_lags(df_train.iloc[va_idx], radii, numeric_feats), cat_features)
            feats = feature_names(tr_df, numeric_feats + list(cat_features), target_col)
            tr_pool, va_pool = pools_from_frames(tr_df, tr_df[target_col], va_df, va_df[target_col], cat_features, feats)

            model = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                iterations=iterations,
                od_type="Iter",
                od_wait=od_wait,
                bootstrap_type="Bayesian",
                grow_policy="SymmetricTree",
                random_seed=seed,
                logging_level="Silent",
                **params,
            )
            model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=od_wait, use_best_model=True)
            fold_pred = model.predict(va_df[feats])
            fold_rmse = rmse(va_df[target_col], fold_pred)
            fold_rmses.append(fold_rmse)
            logger.debug("trial %s fold=%d rmse=%.4f iter=%d", params, k, fold_rmse, model.get_best_iteration())

        cv_rmse = float(np.mean(fold_rmses))
        logger.info("trial params=%s CV_RMSE=%.4f", params, cv_rmse)
        if cv_rmse < best_cv:
            best_cv = cv_rmse
            best_params = params
            logger.info("new best params=%s CV_RMSE=%.4f", best_params, best_cv)

    if best_params is None:
        raise RuntimeError("HPO did not evaluate any parameter sets.")
    return best_params, best_cv


def train_final_model(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    target_col: str,
    numeric_feats: Sequence[str],
    cat_features: Sequence[str],
    radii: Sequence[int],
    params: Mapping[str, Any],
    iterations: int,
    od_wait: int,
    seed: int,
    logger: logging.Logger,
) -> Tuple[CatBoostRegressor, Dict[str, float]]:
    df_train_lag = prep_cat(build_lags(df_train, radii, numeric_feats), cat_features)
    df_test_lag = prep_cat(build_lags(df_test, radii, numeric_feats), cat_features)
    feats_final = feature_names(df_train_lag, numeric_feats + list(cat_features), target_col)
    train_pool = Pool(df_train_lag[feats_final], label=df_train_lag[target_col], cat_features=cat_features, feature_names=feats_final)
    test_pool = Pool(df_test_lag[feats_final], label=df_test_lag[target_col], cat_features=cat_features, feature_names=feats_final)

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iterations,
        od_type="Iter",
        od_wait=od_wait,
        bootstrap_type="Bayesian",
        grow_policy="SymmetricTree",
        random_seed=seed,
        logging_level="Silent",
        **params,
    )
    logger.info("Fitting final model with params=%s", params)
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=od_wait, use_best_model=True)

    best_it = model.get_best_iteration() or model.tree_count_
    evals = model.get_evals_result()
    learn_rmse = evals["learn"]["RMSE"][best_it - 1]
    test_rmse = evals["validation"]["RMSE"][best_it - 1]
    logger.info("final best_iter=%d learn_RMSE=%.4f test_RMSE=%.4f", best_it, learn_rmse, test_rmse)

    y_true = df_test_lag[target_col].values
    y_pred = model.predict(df_test_lag[feats_final])
    metrics = {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }
    logger.info("Test metrics: R2=%.4f MAE=%.4f RMSE=%.4f", metrics["r2"], metrics["mae"], metrics["rmse"])

    resid = y_true - y_pred
    if "geometry" in df_test_lag.columns:
        wq = Queen.from_dataframe(df_test_lag)
        wq.transform = "r"
        from esda.moran import Moran

        mi = Moran(resid, wq)
        metrics["morans_i"] = mi.I
        metrics["morans_p"] = mi.p_sim
        logger.debug("Moran's I=%.4f p=%.4f", mi.I, mi.p_sim)

    return model, metrics


def default_param_grid() -> MutableMapping[str, Sequence[Any]]:
    return {
        "depth": [5, 6, 7, 8],
        "learning_rate": [0.02, 0.03, 0.05],
        "l2_leaf_reg": [3, 6, 9, 12],
        "random_strength": [0.5, 1.0, 1.5, 2.0],
        "bagging_temperature": [0.25, 0.5, 1.0, 2.0],
    }


def run_training(
    blocks: pd.DataFrame,
    config: TrainingConfig,
    *,
    log_path: Path | str = "training.log",
    console_level: int = logging.INFO,
) -> Tuple[CatBoostRegressor, Dict[str, float]]:
    """Full pipeline for notebook use; returns fitted model and metrics."""

    logger = setup_logger(log_path, console_level=console_level)
    logger.info("Starting training pipeline")

    groups = spatial_groups(blocks, config.n_clusters, random_state=config.seed)
    gkf_outer = GroupKFold(n_splits=config.outer_splits)
    train_idx, test_idx = list(gkf_outer.split(blocks, blocks[config.target_col], groups))[0]
    df_train = blocks.iloc[train_idx].copy()
    df_test = blocks.iloc[test_idx].copy()
    logger.info("Outer split: train=%d test=%d", len(df_train), len(df_test))

    params_grid = config.param_grid or default_param_grid()
    best_params, best_cv = run_hpo(
        df_train,
        config.target_col,
        [c for c in config.feature_cols if c not in config.cat_features],
        config.cat_features,
        config.radius_list,
        groups[train_idx],
        params_grid=params_grid,
        n_iter=config.hpo_iter,
        iterations=config.iterations,
        od_wait=config.od_wait,
        seed=config.seed,
        inner_splits=config.inner_splits,
        logger=logger,
    )
    logger.info("Best params=%s CV_RMSE=%.4f", best_params, best_cv)

    model, metrics = train_final_model(
        df_train,
        df_test,
        target_col=config.target_col,
        numeric_feats=[c for c in config.feature_cols if c not in config.cat_features],
        cat_features=config.cat_features,
        radii=config.radius_list,
        params=best_params,
        iterations=max(config.iterations * 2, 20000),
        od_wait=config.od_wait + 100,
        seed=config.seed,
        logger=logger,
    )
    logger.info("Finished training with metrics=%s", metrics)
    return model, metrics


__all__ = [
    "TrainingConfig",
    "setup_logger",
    "run_training",
    "run_hpo",
    "train_final_model",
    "default_param_grid",
    "build_lags",
    "prep_cat",
    "feature_names",
]
