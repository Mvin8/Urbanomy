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
    loss_function: str = "MAE"
    eval_metric: str = "MAE"
    group_col: Optional[str] = None

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


def to_original_scale(y: np.ndarray, *, target_col: str) -> np.ndarray:
    # Training uses log1p-based targets (log_*), metrics should be reported in rubles.
    if target_col.startswith("log_"):
        return np.expm1(y)
    return y


def wape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-12) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    denom = float(np.sum(np.abs(y_true[mask])))
    if denom <= eps:
        return float("nan")
    num = float(np.sum(np.abs(y_true[mask] - y_pred[mask])))
    return num / denom


def mpe(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-12) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > eps)
    if not np.any(mask):
        return float("nan")
    # Positive means overprediction, consistent with `mean_error`.
    return float(np.mean((y_pred[mask] - y_true[mask]) / y_true[mask]))


def mape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-12) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > eps)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def smape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-12) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    y_t = y_true[mask]
    y_p = y_pred[mask]
    denom = np.abs(y_t) + np.abs(y_p)
    valid = denom > eps
    if not np.any(valid):
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_t[valid] - y_p[valid]) / denom[valid]))


def mdape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-12) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > eps)
    if not np.any(mask):
        return float("nan")
    ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.median(ape))


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.clip(y_true, a_min=0.0, a_max=None)
    y_p = np.clip(y_pred, a_min=0.0, a_max=None)
    return float(np.sqrt(np.mean((np.log1p(y_t) - np.log1p(y_p)) ** 2)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, target_col: str) -> Dict[str, float]:
    y_true_eval = to_original_scale(y_true, target_col=target_col)
    y_pred_eval = to_original_scale(y_pred, target_col=target_col)

    metrics: Dict[str, float] = {
        "r2": float(r2_score(y_true_eval, y_pred_eval)),
        "mae": float(mean_absolute_error(y_true_eval, y_pred_eval)),
        "rmse": float(rmse(y_true_eval, y_pred_eval)),
        "mape_ratio": float(mape(y_true_eval, y_pred_eval)),
        "wape_ratio": float(wape(y_true_eval, y_pred_eval)),
    }
    return metrics


def format_metrics_report(train_metrics: Mapping[str, float], test_metrics: Mapping[str, float]) -> str:
    return (
        "Train metrics:\n"
        f"MAPE: {train_metrics['mape_ratio']:.6f} ({train_metrics['mape_ratio'] * 100.0:.2f}%)\n"
        f"WAPE: {train_metrics['wape_ratio']:.6f} ({train_metrics['wape_ratio'] * 100.0:.2f}%)\n"
        f"MAE:  {train_metrics['mae']:.6f}\n"
        f"RMSE: {train_metrics['rmse']:.6f}\n"
        f"R2:   {train_metrics['r2']:.6f}\n\n"
        "Test metrics:\n"
        f"MAPE: {test_metrics['mape_ratio']:.6f} ({test_metrics['mape_ratio'] * 100.0:.2f}%)\n"
        f"WAPE: {test_metrics['wape_ratio']:.6f} ({test_metrics['wape_ratio'] * 100.0:.2f}%)\n"
        f"MAE:  {test_metrics['mae']:.6f}\n"
        f"RMSE: {test_metrics['rmse']:.6f}\n"
        f"R2:   {test_metrics['r2']:.6f}"
    )


def mean_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Bias in target units: >0 means overprediction, <0 means underprediction.
    return float(np.mean(y_pred - y_true))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-12) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    y_t = y_true[mask]
    y_p = y_pred[mask]
    denom = float(np.max(y_t) - np.min(y_t))
    if denom <= eps:
        return float("nan")
    return float(rmse(y_t, y_p) / denom)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_train: np.ndarray,
    seasonality: int = 1,
    eps: float = 1e-12,
) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    y_t = y_true[mask]
    y_p = y_pred[mask]
    y_tr = y_train[np.isfinite(y_train)]
    if len(y_tr) <= seasonality:
        return float("nan")
    scale = float(np.mean(np.abs(y_tr[seasonality:] - y_tr[:-seasonality])))
    if scale <= eps:
        return float("nan")
    return float(np.mean(np.abs(y_t - y_p)) / scale)


def residuals_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_col: str,
) -> pd.DataFrame:
    y_true_eval = to_original_scale(y_true, target_col=target_col)
    y_pred_eval = to_original_scale(y_pred, target_col=target_col)
    residual = y_true_eval - y_pred_eval
    return pd.DataFrame({"y_true": y_true_eval, "y_pred": y_pred_eval, "residual": residual})


def plot_residual_diagnostics(
    residuals_df: pd.DataFrame,
    *,
    bins: int = 50,
    figsize: Tuple[int, int] = (14, 5),
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].hist(residuals_df["residual"].to_numpy(), bins=bins, alpha=0.9, edgecolor="white")
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Residuals Distribution")
    axes[0].set_xlabel("Residual (y_true - y_pred)")
    axes[0].set_ylabel("Count")

    axes[1].scatter(
        residuals_df["y_pred"].to_numpy(),
        residuals_df["residual"].to_numpy(),
        s=12,
        alpha=0.35,
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Residuals vs Predicted")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual (y_true - y_pred)")

    fig.tight_layout()
    return fig, axes


def spatial_groups(blocks: pd.DataFrame, n_clusters: int, *, random_state: int = 42) -> np.ndarray:
    if {"x", "y"}.issubset(blocks.columns):
        coords = blocks[["x", "y"]].to_numpy()
    else:
        cent = blocks.geometry.centroid
        coords = np.column_stack([cent.x.values, cent.y.values])
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit_predict(coords)


def groups_from_column(df: pd.DataFrame, group_col: str) -> np.ndarray:
    if group_col not in df.columns:
        raise KeyError(f"group_col '{group_col}' not found in DataFrame.")
    values = df[group_col].astype("string").fillna("missing_group")
    groups, _ = pd.factorize(values, sort=True)
    return groups


def resolve_groups(df: pd.DataFrame, config: TrainingConfig, logger: logging.Logger) -> np.ndarray:
    if config.group_col is not None:
        groups = groups_from_column(df, config.group_col)
        n_groups = int(pd.Series(groups).nunique())
        logger.info("Using GroupKFold groups from column '%s' (%d unique groups)", config.group_col, n_groups)
        return groups

    groups = spatial_groups(df, config.n_clusters, random_state=config.seed)
    logger.info("Using KMeans spatial groups (%d clusters)", config.n_clusters)
    return groups


def prep_cat_inplace(df: pd.DataFrame, cat_cols: Sequence[str]) -> None:
    for c in cat_cols:
        if c not in df.columns:
            continue
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
        if feat not in out.columns:
            continue
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
    return [c for c in final if c != target_col and c in df.columns]


def build_lagged_fold_frames(
    df_train_base: pd.DataFrame,
    df_test_base: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    cat_features: Sequence[str],
    numeric_feats: Sequence[str],
    radius_list: Sequence[int],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    # Leakage-safe setup: train/test lags are computed independently per fold.
    tr_df = df_train_base.copy()
    te_df = df_test_base.copy()

    prep_cat_inplace(tr_df, cat_features)
    prep_cat_inplace(te_df, cat_features)

    tr_weights = build_radii_weights(tr_df, radius_list)
    te_weights = build_radii_weights(te_df, radius_list)

    tr_lag = add_lags_full_df(tr_df, weights=tr_weights, numeric_cols=numeric_feats)
    te_lag = add_lags_full_df(te_df, weights=te_weights, numeric_cols=numeric_feats)

    base_cols_for_model = list(dict.fromkeys(list(feature_cols) + list(cat_features)))
    tr_feats = feature_names(tr_lag, base_cols_for_model, target_col)
    te_feats = feature_names(te_lag, base_cols_for_model, target_col)
    feats = [c for c in tr_feats if c in te_feats]
    if not feats:
        raise RuntimeError("No common feature columns between train and test fold after lag generation.")

    return tr_lag, te_lag, feats


def pools_from_frames(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    *,
    cat_cols: Sequence[str],
    feats: Sequence[str],
) -> Tuple[Pool, Pool]:
    cat_cols_used = [c for c in cat_cols if c in feats]
    tr_pool = Pool(X_tr[feats], label=y_tr, cat_features=cat_cols_used, feature_names=feats)
    va_pool = Pool(X_va[feats], label=y_va, cat_features=cat_cols_used, feature_names=feats)
    return tr_pool, va_pool


def default_param_grid() -> MutableMapping[str, Sequence[Any]]:
    return {
        "depth": [5, 6, 7, 8],
        "learning_rate": [0.02, 0.03, 0.05],
        "l2_leaf_reg": [3, 6, 9, 12],
        "random_strength": [0.5, 1.0, 1.5, 2.0],
        "bagging_temperature": [0.25, 0.5, 1.0, 2.0],
    }


def param_grid_size(params_grid: Mapping[str, Sequence[Any]]) -> int:
    size = 1
    for k, values in params_grid.items():
        n_vals = len(values)
        if n_vals <= 0:
            raise ValueError(f"param_grid['{k}'] must contain at least one value.")
        size *= n_vals
    return int(size)


def single_params_from_grid(params_grid: Mapping[str, Sequence[Any]]) -> Dict[str, Any] | None:
    if param_grid_size(params_grid) != 1:
        return None
    return {k: list(v)[0] for k, v in params_grid.items()}


def run_hpo_precomputed(
    df_train_base: pd.DataFrame,
    *,
    target_col: str,
    loss_function: str,
    eval_metric: str,
    cat_features: Sequence[str],
    feats: Sequence[str],
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
    best_cv_wape = float("inf")

    inner_splits_iter = list(gkf_inner.split(df_train_base, df_train_base[target_col], groups_train))
    precomputed_folds: List[Tuple[pd.DataFrame, pd.DataFrame, List[str]]] = []
    for tr_idx, va_idx in inner_splits_iter:
        tr_base = df_train_base.iloc[tr_idx]
        va_base = df_train_base.iloc[va_idx]
        precomputed_folds.append((tr_base, va_base, list(feats)))

    logger.info("Starting HPO across %d trials", len(sampler))

    for params in tqdm(sampler, desc="Trials", leave=False):
        fold_wapes: List[float] = []

        for k, (tr_df, va_df, fold_feats) in enumerate(
            tqdm(precomputed_folds, desc="InnerCV", leave=False, total=len(precomputed_folds)),
            1,
        ):
            tr_pool, va_pool = pools_from_frames(
                tr_df, tr_df[target_col], va_df, va_df[target_col],
                cat_cols=cat_features, feats=fold_feats
            )

            model = CatBoostRegressor(
                loss_function=loss_function,
                eval_metric=eval_metric,
                iterations=iterations,
                od_type="Iter",
                od_wait=od_wait,
                bootstrap_type="Bayesian",
                grow_policy="SymmetricTree",
                random_seed=seed,
                verbose=0,  # inner лучше без спама
                **params,
            )
            model.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=od_wait, use_best_model=True)

            pred = model.predict(va_df[fold_feats])
            y_true_eval = to_original_scale(va_df[target_col].to_numpy(), target_col=target_col)
            y_pred_eval = to_original_scale(np.asarray(pred), target_col=target_col)
            fold_wapes.append(wape(y_true_eval, y_pred_eval))
            logger.debug(
                "trial %s fold=%d wape=%.2f%% best_iter=%s",
                params,
                k,
                fold_wapes[-1] * 100.0,
                model.get_best_iteration(),
            )

        cv_wape = float(np.mean(fold_wapes))
        logger.info("trial params=%s CV_WAPE=%.2f%%", params, cv_wape * 100.0)

        if cv_wape < best_cv_wape:
            best_cv_wape = cv_wape
            best_params = dict(params)
            logger.info("new best params=%s CV_WAPE=%.2f%%", best_params, best_cv_wape * 100.0)

    if best_params is None:
        raise RuntimeError("HPO did not evaluate any parameter sets.")
    return best_params, best_cv_wape


def fit_and_eval(
    df_train_base: pd.DataFrame,
    df_test_base: pd.DataFrame,
    *,
    target_col: str,
    loss_function: str,
    eval_metric: str,
    cat_features: Sequence[str],
    feats: Sequence[str],
    params: Mapping[str, Any],
    iterations: int,
    od_wait: int,
    seed: int,
    catboost_verbose: int,
    logger: logging.Logger,
) -> Tuple[CatBoostRegressor, Dict[str, float], int]:
    cat_cols_used = [c for c in cat_features if c in feats]

    train_pool = Pool(df_train_base[feats], label=df_train_base[target_col], cat_features=cat_cols_used, feature_names=feats)
    test_pool = Pool(df_test_base[feats], label=df_test_base[target_col], cat_features=cat_cols_used, feature_names=feats)

    model = CatBoostRegressor(
        loss_function=loss_function,
        eval_metric=eval_metric,
        iterations=iterations,
        od_type="Iter",
        od_wait=od_wait,
        bootstrap_type="Bayesian",
        grow_policy="SymmetricTree",
        random_seed=seed,
        verbose=0,  # outer folds тоже лучше без спама
        **params,
    )
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=od_wait, use_best_model=True)

    y_train = df_train_base[target_col].to_numpy()
    y_pred_train = model.predict(df_train_base[feats])
    y_true = df_test_base[target_col].to_numpy()
    y_pred = model.predict(df_test_base[feats])
    y_train_eval = to_original_scale(y_train, target_col=target_col)

    train_metrics = regression_metrics(y_train, y_pred_train, target_col=target_col)
    test_metrics = regression_metrics(y_true, y_pred, target_col=target_col)

    y_true_eval = to_original_scale(y_true, target_col=target_col)
    y_pred_eval = to_original_scale(y_pred, target_col=target_col)
    mpe_ratio = float(mpe(y_true_eval, y_pred_eval))
    smape_ratio = float(smape(y_true_eval, y_pred_eval))
    mdape_ratio = float(mdape(y_true_eval, y_pred_eval))

    m: Dict[str, float] = {
        "r2": test_metrics["r2"],
        "mae": test_metrics["mae"],
        "rmse": test_metrics["rmse"],
        "nrmse": float(nrmse(y_true_eval, y_pred_eval)),
        "mean_error": float(mean_error(y_true_eval, y_pred_eval)),
        # Percentage metrics are reported in percent.
        "mpe": float(mpe_ratio * 100.0),
        "mape": float(test_metrics["mape_ratio"] * 100.0),
        "wape": float(test_metrics["wape_ratio"] * 100.0),
        "smape": float(smape_ratio * 100.0),
        "mdape": float(mdape_ratio * 100.0),
        "mase": float(mase(y_true_eval, y_pred_eval, y_train=y_train_eval)),
    }
    if target_col.startswith("log_"):
        m["rmsle"] = float(rmsle(y_true_eval, y_pred_eval))

    best_it = model.get_best_iteration()
    if best_it is None or best_it <= 0:
        best_it = model.tree_count_

    logger.info("%s\nbest_iter=%d", format_metrics_report(train_metrics, test_metrics), int(best_it))
    return model, m, int(best_it)


def run_training(
    blocks: pd.DataFrame,
    config: TrainingConfig,
    *,
    log_path: Path | str = "training.log",
    console_level: int = logging.INFO,
) -> Tuple[CatBoostRegressor, Dict[str, Any]]:
    logger = setup_logger(log_path, console_level=console_level)
    logger.info("Starting training pipeline (global lag precompute before outer CV)")
    print(
        "[run_training] "
        f"outer_splits={config.outer_splits}, inner_splits={config.inner_splits}, hpo_iter={config.hpo_iter}"
    )
    if config.outer_splits < 2:
        raise ValueError(f"outer_splits must be >= 2, got {config.outer_splits}.")
    if config.group_col is not None:
        logger.info("group_col='%s' is set; n_clusters=%d will be ignored.", config.group_col, config.n_clusters)
        print(f"[run_training] group_col='{config.group_col}' is set; n_clusters={config.n_clusters} is ignored.")

    target_col = config.target_col
    cat_features = list(config.cat_features)
    numeric_feats = [c for c in config.feature_cols if c not in cat_features and c != target_col and c in blocks.columns]

    base_df = blocks.copy()
    prep_cat_inplace(base_df, cat_features)
    weights = build_radii_weights(base_df, config.radius_list)
    blocks_lag = add_lags_full_df(base_df, weights=weights, numeric_cols=numeric_feats)
    base_cols_for_model = list(dict.fromkeys(list(config.feature_cols) + cat_features))
    feats = feature_names(blocks_lag, base_cols_for_model, target_col)

    groups_all = resolve_groups(blocks_lag, config, logger)
    n_groups_all = int(pd.Series(groups_all).nunique())
    if n_groups_all < config.outer_splits:
        raise ValueError(
            f"Not enough unique groups for outer CV: got {n_groups_all}, need at least {config.outer_splits}."
        )
    gkf_outer = GroupKFold(n_splits=config.outer_splits)

    params_grid = config.param_grid or default_param_grid()
    total_candidates = param_grid_size(params_grid)
    effective_hpo_iter = min(config.hpo_iter, total_candidates)
    fixed_params = single_params_from_grid(params_grid)
    skip_hpo = fixed_params is not None
    if skip_hpo:
        logger.info(
            "param_grid contains a single combination; HPO is skipped and fixed params are used: %s",
            fixed_params,
        )
        print(f"[run_training] HPO skipped: param_grid has one combination: {fixed_params}")
        if config.hpo_iter != 1:
            logger.info("hpo_iter=%d is ignored because only one params combination is available.", config.hpo_iter)
            print(
                f"[run_training] hpo_iter={config.hpo_iter} ignored because only one params combination is available."
            )
    else:
        if config.hpo_iter < 1:
            raise ValueError(f"hpo_iter must be >= 1 when HPO is enabled, got {config.hpo_iter}.")
        if config.inner_splits < 2:
            raise ValueError(f"inner_splits must be >= 2 when HPO is enabled, got {config.inner_splits}.")
        logger.info(
            "HPO enabled: %d/%d sampled trials, inner_splits=%d, outer_splits=%d",
            effective_hpo_iter,
            total_candidates,
            config.inner_splits,
            config.outer_splits,
        )
        print(
            "[run_training] "
            f"HPO enabled: sampled_trials={effective_hpo_iter}/{total_candidates}, "
            f"inner_splits={config.inner_splits}, outer_splits={config.outer_splits}"
        )

    fold_metrics: List[Dict[str, float]] = []
    best_params: Dict[str, Any] | None = dict(fixed_params) if fixed_params is not None else None
    best_cv_wape: float | None = None
    best_iters: List[int] = []

    splits = list(gkf_outer.split(blocks_lag, blocks_lag[target_col], groups_all))
    logger.info("Outer CV folds: %d", len(splits))

    for fold_i, (train_idx, test_idx) in enumerate(tqdm(splits, desc="OuterCV", total=len(splits)), 1):
        df_train = blocks_lag.iloc[train_idx]
        df_test = blocks_lag.iloc[test_idx]
        groups_train = groups_all[train_idx]

        run_hpo_on_fold = (best_params is None or not config.hpo_on_first_outer_only) and not skip_hpo
        if run_hpo_on_fold:
            n_groups_train = int(pd.Series(groups_train).nunique())
            if n_groups_train < config.inner_splits:
                raise ValueError(
                    f"Not enough unique groups for inner CV in fold {fold_i}: "
                    f"got {n_groups_train}, need at least {config.inner_splits}."
                )
            logger.info("Running HPO for outer fold %d", fold_i)
            print(f"[run_training] Running HPO on outer fold {fold_i}/{len(splits)}")
            best_params, best_cv_wape = run_hpo_precomputed(
                df_train,
                target_col=target_col,
                loss_function=config.loss_function,
                eval_metric=config.eval_metric,
                cat_features=cat_features,
                feats=feats,
                groups_train=groups_train,
                params_grid=params_grid,
                n_iter=effective_hpo_iter,
                iterations=config.iterations,
                od_wait=config.od_wait,
                seed=config.seed,
                inner_splits=config.inner_splits,
                catboost_verbose=config.catboost_verbose,
                logger=logger,
            )
            logger.info("Best params (fold %d)=%s CV_WAPE=%.2f%%", fold_i, best_params, float(best_cv_wape) * 100.0)
            print(
                f"[run_training] Best params after fold {fold_i}: {best_params}, "
                f"inner_cv_wape={float(best_cv_wape) * 100.0:.2f}%"
            )

        model, m, best_it = fit_and_eval(
            df_train, df_test,
            target_col=target_col,
            loss_function=config.loss_function,
            eval_metric=config.eval_metric,
            cat_features=cat_features,
            feats=feats,
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

    summary: Dict[str, Any] = {}
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

    cat_cols_used = [c for c in cat_features if c in feats]

    full_pool = Pool(blocks_lag[feats], label=blocks_lag[target_col], cat_features=cat_cols_used, feature_names=feats)

    final_model = CatBoostRegressor(
        loss_function=config.loss_function,
        eval_metric=config.eval_metric,
        iterations=final_iterations,
        bootstrap_type="Bayesian",
        grow_policy="SymmetricTree",
        random_seed=config.seed,
        verbose=config.catboost_verbose,  # вот тут будет прогресс по итерациям
        **best_params,
    )
    final_model.fit(full_pool)

    final_pred_raw = np.asarray(final_model.predict(blocks_lag[feats])).reshape(-1)
    final_train_metrics = regression_metrics(
        blocks_lag[target_col].to_numpy(),
        final_pred_raw,
        target_col=target_col,
    )
    outer_test_metrics = {
        "mape_ratio": float(summary["mape"] / 100.0),
        "wape_ratio": float(summary["wape"] / 100.0),
        "mae": float(summary["mae"]),
        "rmse": float(summary["rmse"]),
        "r2": float(summary["r2"]),
    }
    report = format_metrics_report(final_train_metrics, outer_test_metrics)
    summary["report"] = report
    logger.warning("%s", report)

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
    "mean_error",
    "residuals_dataframe",
    "plot_residual_diagnostics",
]
