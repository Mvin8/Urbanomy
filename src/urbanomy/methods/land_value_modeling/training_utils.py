from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from libpysal.weights import DistanceBand
from libpysal.weights.spatial_lag import lag_spatial
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, ParameterSampler
from tqdm.auto import tqdm

LOGGER_NAME = "land_value_training"


@dataclass
class TrainingConfig:
    feature_cols: Sequence[str]
    cat_features: Sequence[str]
    radius_list: Sequence[int]
    target_col: str = "log_total_price"

    n_clusters: int = 10
    inner_splits: int = 5
    outer_splits: int = 5

    iterations: int = 1500
    od_wait: int = 300
    seed: int = 42

    hpo_iter: int = 24
    param_grid: Optional[Mapping[str, Sequence[Any]]] = None

    # ускорение: HPO только на первом outer-fold
    hpo_on_first_outer_only: bool = True

    # как часто CatBoost печатает прогресс (итерации). 0 = тихо
    catboost_verbose: int = 200  # например 100/200


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        from tqdm.auto import tqdm as _tqdm
        try:
            _tqdm.write(self.format(record))
        except Exception:  # pragma: no cover
            self.handleError(record)


def setup_logger(log_path: Path | str = "training.log", *, console_level: int = logging.INFO) -> logging.Logger:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path) for h in logger.handlers):
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    if not any(isinstance(h, TqdmLoggingHandler) for h in logger.handlers):
        sh = TqdmLoggingHandler()
        sh.setLevel(console_level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def spatial_groups(blocks: pd.DataFrame, n_clusters: int, *, random_state: int = 42) -> np.ndarray:
    if {"x", "y"}.issubset(blocks.columns):
        coords = blocks[["x", "y"]].to_numpy()
    else:
        cent = blocks.geometry.centroid
        coords = np.column_stack([cent.x.values, cent.y.values])
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit_predict(coords)


def prep_cat_inplace(df: pd.DataFrame, cat_cols: Sequence[str]) -> None:
    for c in cat_cols:
        df[c] = df[c].astype("string").fillna("missing")


def build_radii_weights(df: pd.DataFrame, radii: Iterable[int]) -> Dict[int, DistanceBand]:
    weights: Dict[int, DistanceBand] = {}
    for r in radii:
        w = DistanceBand.from_dataframe(df, threshold=r, binary=True, silence_warnings=True)
        w.transform = "r"
        weights[r] = w
    return weights


def add_lags_full_df(
    df: pd.DataFrame,
    *,
    weights: Mapping[int, DistanceBand],
    numeric_cols: Sequence[str],
) -> pd.DataFrame:
    out = df.copy()
    new_cols: Dict[str, Any] = {}

    for feat in numeric_cols:
        vec = out[feat]
        if vec.isna().any():
            vec = vec.fillna(vec.mean())
        vec_np = vec.to_numpy()

        for r, w in weights.items():
            new_cols[f"lag{r}_{feat}"] = lag_spatial(w, vec_np)

    for r, w in weights.items():
        neigh_len = pd.Series({idx: len(w.neighbors[idx]) for idx in w.id_order})
        new_cols[f"n_neighbors_{r}"] = neigh_len.reindex(out.index).to_numpy()

    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

    return out


def feature_names(df: pd.DataFrame, base_cols: Sequence[str], target_col: str) -> List[str]:
    extra = [c for c in df.columns if c.startswith("lag") or c.startswith("n_neighbors_")]
    final = list(dict.fromkeys(list(base_cols) + extra))
    return [c for c in final if c != target_col]


def pools_from_frames(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    *,
    cat_cols: Sequence[str],
    feats: Sequence[str],
) -> Tuple[Pool, Pool]:
    tr_pool = Pool(X_tr[feats], label=y_tr, cat_features=cat_cols, feature_names=feats)
    va_pool = Pool(X_va[feats], label=y_va, cat_features=cat_cols, feature_names=feats)
    return tr_pool, va_pool


def default_param_grid() -> MutableMapping[str, Sequence[Any]]:
    return {
        "depth": [5, 6, 7, 8],
        "learning_rate": [0.02, 0.03, 0.05],
        "l2_leaf_reg": [3, 6, 9, 12],
        "random_strength": [0.5, 1.0, 1.5, 2.0],
        "bagging_temperature": [0.25, 0.5, 1.0, 2.0],
    }


def run_hpo_precomputed(
    df_train_lag: pd.DataFrame,
    *,
    target_col: str,
    feats: Sequence[str],
    cat_features: Sequence[str],
    groups_train: np.ndarray,
    params_grid: Mapping[str, Sequence[Any]],
    n_iter: int,
    iterations: int,
    od_wait: int,
    seed: int,
    inner_splits: int,
    catboost_verbose: int,
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], float]:
    sampler = list(ParameterSampler(params_grid, n_iter=n_iter, random_state=seed))
    gkf_inner = GroupKFold(n_splits=inner_splits)

    best_params: Dict[str, Any] | None = None
    best_cv = float("inf")

    logger.info("Starting HPO across %d trials", len(sampler))

    for params in tqdm(sampler, desc="Trials", leave=False):
        fold_rmses: List[float] = []

        inner_splits_iter = list(gkf_inner.split(df_train_lag, df_train_lag[target_col], groups_train))
        for k, (tr_idx, va_idx) in enumerate(
            tqdm(inner_splits_iter, desc="InnerCV", leave=False, total=len(inner_splits_iter)),
            1,
        ):
            tr_df = df_train_lag.iloc[tr_idx]
            va_df = df_train_lag.iloc[va_idx]

            tr_pool, va_pool = pools_from_frames(
                tr_df, tr_df[target_col], va_df, va_df[target_col],
                cat_cols=cat_features, feats=feats
            )

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
                verbose=0,  # inner лучше без спама
                **params,
            )
            model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=od_wait, use_best_model=True)

            pred = model.predict(va_df[feats])
            fold_rmses.append(rmse(va_df[target_col].values, pred))
            logger.debug("trial %s fold=%d rmse=%.4f best_iter=%s", params, k, fold_rmses[-1], model.get_best_iteration())

        cv_rmse = float(np.mean(fold_rmses))
        logger.info("trial params=%s CV_RMSE=%.4f", params, cv_rmse)

        if cv_rmse < best_cv:
            best_cv = cv_rmse
            best_params = dict(params)
            logger.info("new best params=%s CV_RMSE=%.4f", best_params, best_cv)

    if best_params is None:
        raise RuntimeError("HPO did not evaluate any parameter sets.")
    return best_params, best_cv


def fit_and_eval(
    df_train_lag: pd.DataFrame,
    df_test_lag: pd.DataFrame,
    *,
    target_col: str,
    feats: Sequence[str],
    cat_features: Sequence[str],
    params: Mapping[str, Any],
    iterations: int,
    od_wait: int,
    seed: int,
    catboost_verbose: int,
    logger: logging.Logger,
) -> Tuple[CatBoostRegressor, Dict[str, float], int]:
    train_pool = Pool(df_train_lag[feats], label=df_train_lag[target_col], cat_features=cat_features, feature_names=feats)
    test_pool = Pool(df_test_lag[feats], label=df_test_lag[target_col], cat_features=cat_features, feature_names=feats)

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
        verbose=0,  # outer folds тоже лучше без спама
        **params,
    )
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=od_wait, use_best_model=True)

    y_true = df_test_lag[target_col].to_numpy()
    y_pred = model.predict(df_test_lag[feats])

    m: Dict[str, float] = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
    }

    best_it = model.get_best_iteration()
    if best_it is None or best_it <= 0:
        best_it = model.tree_count_

    logger.info("Fold metrics: R2=%.4f MAE=%.4f RMSE=%.4f best_iter=%d", m["r2"], m["mae"], m["rmse"], int(best_it))
    return model, m, int(best_it)


def run_training(
    blocks: pd.DataFrame,
    config: TrainingConfig,
    *,
    log_path: Path | str = "training.log",
    console_level: int = logging.INFO,
) -> Tuple[CatBoostRegressor, Dict[str, float]]:
    logger = setup_logger(log_path, console_level=console_level)
    logger.info("Starting training pipeline (no Moran, full outer CV, precomputed lags)")

    target_col = config.target_col
    cat_features = list(config.cat_features)
    numeric_feats = [c for c in config.feature_cols if c not in cat_features and c != target_col]

    base_df = blocks.copy()
    prep_cat_inplace(base_df, cat_features)

    logger.info("Building weights & lags on full dataset")
    weights = build_radii_weights(base_df, config.radius_list)
    blocks_lag = add_lags_full_df(base_df, weights=weights, numeric_cols=numeric_feats)

    base_cols_for_model = list(dict.fromkeys(list(config.feature_cols) + cat_features))
    feats = feature_names(blocks_lag, base_cols_for_model, target_col)

    groups_all = spatial_groups(blocks_lag, config.n_clusters, random_state=config.seed)
    gkf_outer = GroupKFold(n_splits=config.outer_splits)

    params_grid = config.param_grid or default_param_grid()

    fold_metrics: List[Dict[str, float]] = []
    best_params: Dict[str, Any] | None = None
    best_cv_rmse: float | None = None
    best_iters: List[int] = []

    splits = list(gkf_outer.split(blocks_lag, blocks_lag[target_col], groups_all))
    logger.info("Outer CV folds: %d", len(splits))

    for fold_i, (train_idx, test_idx) in enumerate(tqdm(splits, desc="OuterCV", total=len(splits)), 1):
        df_train = blocks_lag.iloc[train_idx]
        df_test = blocks_lag.iloc[test_idx]
        groups_train = groups_all[train_idx]

        if best_params is None or not config.hpo_on_first_outer_only:
            logger.info("Running HPO for outer fold %d", fold_i)
            best_params, best_cv_rmse = run_hpo_precomputed(
                df_train,
                target_col=target_col,
                feats=feats,
                cat_features=cat_features,
                groups_train=groups_train,
                params_grid=params_grid,
                n_iter=config.hpo_iter,
                iterations=config.iterations,
                od_wait=config.od_wait,
                seed=config.seed,
                inner_splits=config.inner_splits,
                catboost_verbose=config.catboost_verbose,
                logger=logger,
            )
            logger.info("Best params (fold %d)=%s CV_RMSE=%.4f", fold_i, best_params, float(best_cv_rmse))

        model, m, best_it = fit_and_eval(
            df_train, df_test,
            target_col=target_col,
            feats=feats,
            cat_features=cat_features,
            params=best_params,
            iterations=max(config.iterations * 2, 4000),
            od_wait=config.od_wait,
            seed=config.seed,
            catboost_verbose=config.catboost_verbose,
            logger=logger,
        )

        fold_metrics.append(m)
        best_iters.append(best_it)

        if config.hpo_on_first_outer_only and fold_i == 1:
            logger.info("HPO fixed after first outer-fold: params=%s", best_params)

    metrics_df = pd.DataFrame(fold_metrics)
    mean_metrics = metrics_df.mean(numeric_only=True).to_dict()
    std_metrics = metrics_df.std(numeric_only=True).to_dict()

    summary: Dict[str, float] = {}
    for k, v in mean_metrics.items():
        summary[k] = float(v)
        summary[f"{k}_std"] = float(std_metrics.get(k, np.nan))

    logger.info("OuterCV mean metrics=%s", summary)

    if best_params is None:
        raise RuntimeError("best_params is None (unexpected)")

    # финальное число итераций = медиана best_iter по outer-fold
    final_iterations = int(np.median(best_iters)) if best_iters else max(config.iterations * 2, 4000)
    final_iterations = max(200, final_iterations)  # защита от слишком малого числа

    logger.info("Training final model on FULL dataset: iterations=%d params=%s", final_iterations, best_params)

    full_pool = Pool(blocks_lag[feats], label=blocks_lag[target_col], cat_features=cat_features, feature_names=feats)

    final_model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=final_iterations,
        bootstrap_type="Bayesian",
        grow_policy="SymmetricTree",
        random_seed=config.seed,
        logging_level="Silent",
        verbose=config.catboost_verbose,  # вот тут будет прогресс по итерациям
        **best_params,
    )
    final_model.fit(full_pool)

    return final_model, summary


__all__ = [
    "TrainingConfig",
    "setup_logger",
    "run_training",
    "default_param_grid",
    "build_radii_weights",
    "add_lags_full_df",
    "feature_names",
    "spatial_groups",
]
