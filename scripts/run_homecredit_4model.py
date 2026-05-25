#!/usr/bin/env python3
"""Run the 4-model pipeline on the Home Credit Default Risk preprocessed data.

Week 4 tasks covered:
  - Full 4-model baseline (LR, RF, XGBoost, LightGBM) on Home Credit.
  - Imbalance handling comparison: class_weight vs SMOTE vs ADASYN.
  - Metrics: ROC-AUC, PR-AUC, KS, F1@0.5, Accuracy, Specificity, Precision, Recall.
  - Pairwise statistical tests: DeLong test for ROC-AUC and exact McNemar test.
  - Produce ROC curves and PR curves (Figure 6 candidate).
  - Save results JSON (Table 8 data).

Usage
-----
  # Full run:
  python scripts/run_homecredit_4model.py \\
      --input outputs/homecredit_preprocessed.csv \\
      --output-json outputs/homecredit_4model_results.json \\
      --figures-dir outputs/figures \\
      --seed 42

  # Quick smoke-test:
  python scripts/run_homecredit_4model.py \\
      --input outputs/homecredit_preprocessed.csv \\
      --sample-n 10000 --seed 42
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import binomtest, norm
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COLUMN = "TARGET"
SORT_KEY_COLUMN = "sort_key"

IMBALANCE_STRATEGIES = ["class_weight", "smote", "adasyn"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitSummary:
    rows: int
    default_rate: float


@dataclass(frozen=True)
class Metrics:
    roc_auc: float
    pr_auc: float
    ks: float
    f1: float
    accuracy: float
    precision: float
    recall: float
    specificity: float
    positive_rate: float
    predicted_positive_rate: float


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def load_preprocessed(input_path: Path, sample_n: Optional[int], seed: int) -> pd.DataFrame:
    """Load the preprocessed Home Credit CSV; optionally stratified-sample."""
    df = pd.read_csv(input_path, low_memory=False)
    df = df.dropna(how="all").copy()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {input_path}.")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    if sample_n is not None and sample_n < len(df):
        # Stratified sample to preserve default rate
        rng = np.random.default_rng(seed)
        pos_idx = df.index[df[TARGET_COLUMN] == 1].tolist()
        neg_idx = df.index[df[TARGET_COLUMN] == 0].tolist()
        default_rate = len(pos_idx) / len(df)
        n_pos = max(1, int(sample_n * default_rate))
        n_neg = sample_n - n_pos
        sampled_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
        sampled_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
        df = df.loc[sorted(np.concatenate([sampled_pos, sampled_neg]))].copy()

    return df


def split_by_sort_key(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by SK_ID_CURR surrogate (ascending = older) then split 60/20/20."""
    if SORT_KEY_COLUMN in df.columns:
        df = df.sort_values(SORT_KEY_COLUMN).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def get_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    drop_cols = [TARGET_COLUMN, SORT_KEY_COLUMN]
    feat_cols = [c for c in df.columns if c not in drop_cols]
    return df[feat_cols].copy(), df[TARGET_COLUMN].copy()


def summarise(df: pd.DataFrame) -> SplitSummary:
    return SplitSummary(rows=len(df), default_rate=float(df[TARGET_COLUMN].mean()))


# ---------------------------------------------------------------------------
# Preprocessing transformer (all features already numeric after preprocess_homecredit.py)
# ---------------------------------------------------------------------------

def build_preprocessor(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    """Minimal transformer: impute remaining NaNs + optional scale."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers: list = []
    if num_cols:
        steps: list = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps), num_cols))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
            ]),
            cat_cols,
        ))

    return ColumnTransformer(transformers)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _make_model(name: str, seed: int, strategy: str) -> Any:
    use_cw = (strategy == "class_weight")
    cw = "balanced" if use_cw else None

    if name == "LR":
        return LogisticRegression(max_iter=1_000, random_state=seed, class_weight=cw, solver="lbfgs", C=1.0)
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            class_weight=cw, random_state=seed, n_jobs=-1,
        )
    if name == "XGBoost":
        # XGBoost uses scale_pos_weight for class imbalance
        return XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="aucpr",
            random_state=seed, verbosity=0,
        )
    if name == "LightGBM":
        return LGBMClassifier(
            n_estimators=200, num_leaves=63, learning_rate=0.05,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            class_weight=cw, random_state=seed, verbose=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(name: str, X_train: pd.DataFrame, seed: int, strategy: str) -> Pipeline:
    scale = (name == "LR")
    preprocessor = build_preprocessor(X_train, scale=scale)
    model = _make_model(name, seed, strategy)
    return Pipeline([("prep", preprocessor), ("clf", model)])


# ---------------------------------------------------------------------------
# Oversampling helpers
# ---------------------------------------------------------------------------

def apply_oversampling(
    strategy: str, X: np.ndarray, y: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE or ADASYN to training data; return resampled (X, y)."""
    if strategy == "smote":
        sampler = SMOTE(random_state=seed, k_neighbors=5)
    elif strategy == "adasyn":
        sampler = ADASYN(random_state=seed, n_neighbors=5)
    else:
        return X, y
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> Metrics:
    if len(np.unique(y_true)) < 2:
        nan = float("nan")
        return Metrics(nan, nan, nan, nan, nan, nan, nan, nan, float(y_true.mean()), nan)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks = float(np.max(tpr - fpr))
    preds = (y_score >= threshold).astype(int)

    tn = np.sum((preds == 0) & (y_true == 0))
    fp = np.sum((preds == 1) & (y_true == 0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    return Metrics(
        roc_auc=float(roc_auc_score(y_true, y_score)),
        pr_auc=float(average_precision_score(y_true, y_score)),
        ks=ks,
        f1=float(f1_score(y_true, preds, zero_division=0)),
        accuracy=float(accuracy_score(y_true, preds)),
        precision=float(precision_score(y_true, preds, zero_division=0)),
        recall=float(recall_score(y_true, preds, zero_division=0)),
        specificity=specificity,
        positive_rate=float(y_true.mean()),
        predicted_positive_rate=float(preds.mean()),
    )


# ---------------------------------------------------------------------------
# Curve data (for plotting)
# ---------------------------------------------------------------------------

def get_roc_data(y_true: np.ndarray, y_score: np.ndarray):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return fpr, tpr, auc


def get_pr_data(y_true: np.ndarray, y_score: np.ndarray):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return recall, precision, ap


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def _compute_midrank(values: np.ndarray) -> np.ndarray:
    """Return one-based midranks used by the fast DeLong implementation."""
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.zeros(len(values), dtype=float)

    i = 0
    while i < len(values):
        j = i
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1)
        i = j

    out = np.empty(len(values), dtype=float)
    out[order] = ranks + 1
    return out


def _fast_delong(
    predictions_sorted_transposed: np.ndarray,
    positive_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fast DeLong covariance for one or more correlated ROC-AUC estimates."""
    negative_count = predictions_sorted_transposed.shape[1] - positive_count
    positive_examples = predictions_sorted_transposed[:, :positive_count]
    negative_examples = predictions_sorted_transposed[:, positive_count:]
    classifier_count = predictions_sorted_transposed.shape[0]

    tx = np.empty((classifier_count, positive_count), dtype=float)
    ty = np.empty((classifier_count, negative_count), dtype=float)
    tz = np.empty((classifier_count, positive_count + negative_count), dtype=float)

    for r in range(classifier_count):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = (
        tz[:, :positive_count].sum(axis=1) / positive_count / negative_count
        - (positive_count + 1.0) / (2.0 * negative_count)
    )
    v01 = (tz[:, :positive_count] - tx) / negative_count
    v10 = 1.0 - (tz[:, positive_count:] - ty) / positive_count
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / positive_count + sy / negative_count
    return aucs, np.atleast_2d(delong_cov)


def delong_roc_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Dict[str, float]:
    """Two-sided DeLong test for two correlated ROC-AUC values."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return {"auc_a": float("nan"), "auc_b": float("nan"), "z": float("nan"), "p_value": float("nan")}

    order = np.argsort(-y_true)
    positive_count = int(y_true.sum())
    predictions = np.vstack([scores_a, scores_b])[:, order]
    aucs, covariance = _fast_delong(predictions, positive_count)
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast.T)
    if variance <= 0 or np.isnan(variance):
        z_score = float("nan")
        p_value = float("nan")
    else:
        z_score = float(abs(aucs[0] - aucs[1]) / np.sqrt(variance))
        p_value = float(2.0 * norm.sf(z_score))

    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "z": z_score,
        "p_value": p_value,
    }


def mcnemar_exact_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> Dict[str, float]:
    """Exact McNemar test on paired 0.5-threshold predictions."""
    correct_a = preds_a == y_true
    correct_b = preds_b == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    discordant = b + c
    p_value = float(binomtest(min(b, c), discordant, p=0.5).pvalue) if discordant > 0 else 1.0
    return {
        "b_a_correct_b_wrong": b,
        "c_a_wrong_b_correct": c,
        "discordant_pairs": discordant,
        "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

MODEL_NAMES = ["LR", "RF", "XGBoost", "LightGBM"]
COLORS = {"LR": "#2196F3", "RF": "#4CAF50", "XGBoost": "#FF9800", "LightGBM": "#9C27B0"}
LINE_STYLES = {"LR": "-", "RF": "--", "XGBoost": "-.", "LightGBM": ":"}


def run_strategy(
    strategy: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    models: List[str],
    seed: int,
) -> Dict[str, Any]:
    """Train all models with one imbalance strategy; return results dict."""
    X_train, y_train = get_features(train)
    X_val, y_val = get_features(val)
    X_test, y_test = get_features(test)

    results = {}
    curves = {}
    predictions = {}

    for name in models:
        print(f"    [{strategy}] Training {name} …", flush=True)
        pipe = build_pipeline(name, X_train, seed, strategy)

        # Fit preprocessor to get transformed training matrix for oversampling
        pipe["prep"].fit(X_train)
        X_tr_arr = pipe["prep"].transform(X_train)
        y_tr_arr = y_train.values

        if strategy in ("smote", "adasyn"):
            X_tr_arr, y_tr_arr = apply_oversampling(strategy, X_tr_arr, y_tr_arr, seed)

        # For XGBoost with class_weight-equivalent: use scale_pos_weight
        if name == "XGBoost" and strategy == "class_weight":
            neg = int((y_tr_arr == 0).sum())
            pos = int((y_tr_arr == 1).sum())
            pipe["clf"].set_params(scale_pos_weight=neg / max(pos, 1))

        pipe["clf"].fit(X_tr_arr, y_tr_arr)

        val_scores = pipe["clf"].predict_proba(pipe["prep"].transform(X_val))[:, 1]
        test_scores = pipe["clf"].predict_proba(pipe["prep"].transform(X_test))[:, 1]

        val_metrics = evaluate(y_val.values, val_scores)
        test_metrics = evaluate(y_test.values, test_scores)

        results[name] = {
            "validation": asdict(val_metrics),
            "test": asdict(test_metrics),
        }

        # Collect curve data from test set
        curves[name] = {
            "roc": get_roc_data(y_test.values, test_scores),
            "pr": get_pr_data(y_test.values, test_scores),
        }

        predictions[name] = {
            "validation_y_true": y_val.values,
            "validation_scores": val_scores,
            "y_true": y_test.values,
            "test_scores": test_scores,
            "test_preds": (test_scores >= 0.5).astype(int),
        }

    return {"metrics": results, "curves": curves, "predictions": predictions}


def compute_pairwise_statistical_tests(
    predictions_by_strategy: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    best_strategy: Dict[str, str],
    models: List[str],
) -> List[Dict[str, Any]]:
    """Run pairwise DeLong and McNemar tests using validation-selected strategies."""
    tests: List[Dict[str, Any]] = []
    for model_a, model_b in itertools.combinations(models, 2):
        out_a = predictions_by_strategy[best_strategy[model_a]][model_a]
        out_b = predictions_by_strategy[best_strategy[model_b]][model_b]
        y_true = out_a["y_true"]
        if not np.array_equal(y_true, out_b["y_true"]):
            raise ValueError("Pairwise statistical tests require identical test-set ordering.")

        delong = delong_roc_test(y_true, out_a["test_scores"], out_b["test_scores"])
        mcnemar = mcnemar_exact_test(y_true, out_a["test_preds"], out_b["test_preds"])
        tests.append({
            "model_a": model_a,
            "model_b": model_b,
            "strategy_a": best_strategy[model_a],
            "strategy_b": best_strategy[model_b],
            "delong": delong,
            "mcnemar": mcnemar,
        })
    return tests


def run_pipeline(
    input_path: Path,
    output_json: Optional[Path],
    figures_dir: Optional[Path],
    sample_n: Optional[int],
    models: List[str],
    strategies: List[str],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> Dict[str, Any]:

    print(f"\nLoading Home Credit data from {input_path} …")
    df = load_preprocessed(input_path, sample_n, seed)
    print(f"  Loaded {len(df):,} rows, default rate={df[TARGET_COLUMN].mean():.2%}")

    train, val, test = split_by_sort_key(df, train_frac, val_frac)
    print(f"  Split → train={len(train):,} ({train[TARGET_COLUMN].mean():.2%})"
          f"  val={len(val):,} ({val[TARGET_COLUMN].mean():.2%})"
          f"  test={len(test):,} ({test[TARGET_COLUMN].mean():.2%})")

    split_summary = {
        "train": asdict(summarise(train)),
        "validation": asdict(summarise(val)),
        "test": asdict(summarise(test)),
    }

    all_strategy_results: Dict[str, Any] = {}
    all_curves: Dict[str, Any] = {}
    predictions_by_strategy: Dict[str, Any] = {}

    for strategy in strategies:
        print(f"\n  Strategy: {strategy.upper()}")
        out = run_strategy(strategy, train, val, test, models, seed)
        all_strategy_results[strategy] = out["metrics"]
        all_curves[strategy] = out["curves"]
        predictions_by_strategy[strategy] = out["predictions"]

    # Determine best imbalance strategy per model using validation PR-AUC only.
    best_strategy: Dict[str, str] = {}
    for model in models:
        best_prauc = -1.0
        best_strat = strategies[0]
        for strat in strategies:
            prauc = all_strategy_results[strat][model]["validation"]["pr_auc"]
            if not np.isnan(prauc) and prauc > best_prauc:
                best_prauc = prauc
                best_strat = strat
        best_strategy[model] = best_strat

    print("\n  Best imbalance strategy per model (by validation PR-AUC):")
    for model, strat in best_strategy.items():
        prauc = all_strategy_results[strat][model]["validation"]["pr_auc"]
        print(f"    {model:<10s} → {strat:<14s}  PR-AUC={prauc:.4f}")

    # Build best-strategy metrics table (for Table 8)
    table8: Dict[str, Dict[str, float]] = {}
    for model in models:
        strat = best_strategy[model]
        table8[model] = all_strategy_results[strat][model]["test"]

    statistical_tests = compute_pairwise_statistical_tests(predictions_by_strategy, best_strategy, models)

    # --- Figures ---
    generated_figures: List[str] = []
    if figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)
        generated_figures.extend(_plot_roc_curves(all_curves, best_strategy, models, figures_dir, strategies))
        generated_figures.extend(_plot_pr_curves(all_curves, best_strategy, models, figures_dir, strategies))
        generated_figures.extend(_plot_strategy_comparison(all_strategy_results, models, strategies, figures_dir))
        print(f"\n  Figures saved → {figures_dir}/")

    result = {
        "dataset": str(input_path),
        "total_rows": len(df),
        "seed": seed,
        "splits": split_summary,
        "strategies_evaluated": strategies,
        "best_strategy_per_model": best_strategy,
        "all_strategy_metrics": all_strategy_results,
        "table8_best_metrics": table8,
        "statistical_tests": statistical_tests,
        "generated_figures": generated_figures,
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"  Results saved → {output_json}")

    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_roc_curves(
    all_curves: Dict,
    best_strategy: Dict[str, str],
    models: List[str],
    out_dir: Path,
    strategies: List[str],
) -> List[str]:
    """Figure 6a: ROC curves per model (best strategy each)."""
    fig, axes = plt.subplots(1, len(strategies), figsize=(6 * len(strategies), 5), sharey=True)
    if len(strategies) == 1:
        axes = [axes]

    for ax, strat in zip(axes, strategies):
        for name in models:
            fpr, tpr, auc = all_curves[strat][name]["roc"]
            best_marker = " *" if best_strategy.get(name) == strat else ""
            ax.plot(fpr, tpr, color=COLORS[name], ls=LINE_STYLES[name], lw=1.8,
                    label=f"{name}{best_marker} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.set_title(f"ROC Curves — {strat.upper()}", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)

    fig.suptitle("ROC Curves by Imbalance Strategy (Home Credit)", fontsize=12, y=1.01)
    plt.tight_layout()
    path = out_dir / "fig6a_roc_curves_homecredit.pdf"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved {path.name}")
    return [str(path), str(path.with_suffix(".png"))]


def _plot_pr_curves(
    all_curves: Dict,
    best_strategy: Dict[str, str],
    models: List[str],
    out_dir: Path,
    strategies: List[str],
) -> List[str]:
    """Figure 6b: PR curves per model (best strategy each)."""
    fig, axes = plt.subplots(
        1, len(strategies), figsize=(6.5 * len(strategies), 5.5), sharey=True
    )
    if len(strategies) == 1:
        axes = [axes]

    for ax, strat in zip(axes, strategies):
        for name in models:
            recall, precision, ap = all_curves[strat][name]["pr"]
            best_marker = " *" if best_strategy.get(name) == strat else ""
            ax.plot(recall, precision, color=COLORS[name], ls=LINE_STYLES[name], lw=1.8,
                    label=f"{name}{best_marker} (AP={ap:.3f})")
        ax.set_xlabel("Recall", fontsize=10)
        ax.set_ylabel("Precision", fontsize=10)
        ax.set_title(f"PR Curves — {strat.upper()}", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Precision–Recall Curves by Imbalance Strategy (Home Credit)", fontsize=12, y=1.01)
    plt.tight_layout()
    path = out_dir / "fig6b_pr_curves_homecredit.pdf"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved {path.name}")
    return [str(path), str(path.with_suffix(".png"))]


def _plot_strategy_comparison(
    all_strategy_metrics: Dict,
    models: List[str],
    strategies: List[str],
    out_dir: Path,
) -> List[str]:
    """Bar chart comparing validation PR-AUC across strategies per model."""
    metrics_key = "pr_auc"
    x = np.arange(len(models))
    width = 0.25
    strat_colors = {"class_weight": "#2196F3", "smote": "#4CAF50", "adasyn": "#FF9800"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, strat in enumerate(strategies):
        vals = [all_strategy_metrics[strat][m]["validation"][metrics_key] for m in models]
        ax.bar(x + i * width, vals, width, label=strat.upper(),
               color=strat_colors.get(strat, f"C{i}"), alpha=0.85, edgecolor="white")

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("PR-AUC (Validation)", fontsize=11)
    ax.set_title("Imbalance Strategy Comparison — Validation PR-AUC (Home Credit)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(
        all_strategy_metrics[s][m]["validation"][metrics_key]
        for s in strategies for m in models
        if not np.isnan(all_strategy_metrics[s][m]["validation"][metrics_key])
    ) * 1.15)
    ax.grid(axis="y", alpha=0.3)

    path = out_dir / "fig6c_strategy_comparison_homecredit.pdf"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved {path.name}")
    return [str(path), str(path.with_suffix(".png"))]


# ---------------------------------------------------------------------------
# CLI printing
# ---------------------------------------------------------------------------

def _print_results(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY — Home Credit 4-Model Pipeline (Week 4)")
    print("=" * 80)

    strategies = result["strategies_evaluated"]
    models = list(result["all_strategy_metrics"][strategies[0]].keys())

    for strat in strategies:
        print(f"\n  Strategy: {strat.upper()}")
        header = f"  {'Model':<10s}  {'Split':<12s}  {'ROC-AUC':>8s}  {'PR-AUC':>8s}  {'KS':>8s}  {'F1':>8s}  {'Acc':>8s}  {'Prec':>8s}  {'Recall':>8s}  {'Spec':>8s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for m in models:
            for split in ("validation", "test"):
                mtr = result["all_strategy_metrics"][strat][m][split]
                def fmt(v):
                    return f"{v:>8.4f}" if not np.isnan(v) else f"{'NaN':>8s}"
                print(
                    f"  {m:<10s}  {split:<12s}  {fmt(mtr['roc_auc'])}  {fmt(mtr['pr_auc'])}"
                    f"  {fmt(mtr['ks'])}  {fmt(mtr['f1'])}  {fmt(mtr['accuracy'])}  {fmt(mtr['precision'])}"
                    f"  {fmt(mtr['recall'])}  {fmt(mtr['specificity'])}"
                )
            print()

    print("\n  TABLE 8 CANDIDATE (validation-selected strategy per model, test set):")
    print(f"  {'Model':<10s}  {'Strategy':<14s}  {'ROC-AUC':>8s}  {'PR-AUC':>8s}  {'KS':>8s}  {'F1':>8s}  {'Acc':>8s}  {'Prec':>8s}  {'Recall':>8s}  {'Spec':>8s}")
    print("  " + "-" * 100)
    for m in models:
        strat = result["best_strategy_per_model"][m]
        mtr = result["table8_best_metrics"][m]
        def fmt(v):
            return f"{v:>8.4f}" if not np.isnan(v) else f"{'NaN':>8s}"
        print(
            f"  {m:<10s}  {strat:<14s}  {fmt(mtr['roc_auc'])}  {fmt(mtr['pr_auc'])}"
            f"  {fmt(mtr['ks'])}  {fmt(mtr['f1'])}  {fmt(mtr['accuracy'])}  {fmt(mtr['precision'])}"
            f"  {fmt(mtr['recall'])}  {fmt(mtr['specificity'])}"
        )

    print("\n  PAIRWISE STATISTICAL TESTS (test set, validation-selected strategies):")
    print(f"  {'Pair':<20s}  {'DeLong p':>12s}  {'McNemar p':>12s}  {'Discordant':>12s}")
    print("  " + "-" * 62)
    for row in result["statistical_tests"]:
        pair = f"{row['model_a']} vs {row['model_b']}"
        delong_p = row["delong"]["p_value"]
        mcnemar_p = row["mcnemar"]["p_value"]
        discordant = row["mcnemar"]["discordant_pairs"]
        print(
            f"  {pair:<20s}  {delong_p:>12.4g}  {mcnemar_p:>12.4g}  {discordant:>12d}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 4-model baseline + imbalance comparison on Home Credit data (Week 4)."
    )
    parser.add_argument(
        "--input", default="outputs/homecredit_preprocessed.csv",
        help="Path to preprocessed Home Credit CSV.",
    )
    parser.add_argument(
        "--output-json", default="outputs/homecredit_4model_results.json",
        help="JSON path for persisting full results.",
    )
    parser.add_argument(
        "--figures-dir", default="outputs/figures",
        help="Directory to save ROC/PR curve figures.",
    )
    parser.add_argument(
        "--sample-n", type=int, default=None,
        help="Stratified sample size for quick iterations (None = full dataset).",
    )
    parser.add_argument(
        "--models", nargs="+", default=MODEL_NAMES,
        choices=MODEL_NAMES, metavar="MODEL",
        help="Subset of models to run.",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=IMBALANCE_STRATEGIES,
        choices=IMBALANCE_STRATEGIES, metavar="STRATEGY",
        help="Imbalance handling strategies to compare.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    figures_dir = None if args.no_figures else Path(args.figures_dir)
    output_json = Path(args.output_json) if args.output_json else None

    result = run_pipeline(
        input_path=Path(args.input),
        output_json=output_json,
        figures_dir=figures_dir,
        sample_n=args.sample_n,
        models=args.models,
        strategies=args.strategies,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    _print_results(result)


if __name__ == "__main__":
    main()
