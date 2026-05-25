#!/usr/bin/env python3
"""Bootstrap B-sweep stability study.

For each ``B`` in ``--b-values`` (default ``30 50 80 100``) we run an INDEPENDENT
bootstrap loop on the Home Credit pipeline:

    For b in 1..B:
        - Resample training rows WITH replacement
          (seed = base_seed + B_offset(B) + b, where B_offset is unique per B)
        - For each model in {LR, RF, XGBoost, LightGBM}:
            * Refit the classifier on the resampled train (preprocessor fitted
              ONCE on the full train, then frozen — keeps the feature column
              ordering and dimensionality identical across iterations).
            * Compute SHAP on a FIXED test subsample (LinearExplainer for LR,
              TreeExplainer for the three tree models).
            * Persist mean(|SHAP|) per feature to disk; if the file already
              exists, skip refit/SHAP (resume-friendly).

After all iterations are done, for each (B, model) we compute pairwise
stability over C(B, 2) pairs of rankings:

    - Jaccard@5, Jaccard@10 (top-k set overlap)
    - Spearman rho on the full feature ranking (228 features after preprocess)

Aggregated as Mean ± Std, then written to:

    outputs/shap_bootstrap/B{B:03d}/{model}_iter{b:03d}_mean_abs.csv
    outputs/shap_bootstrap/B{B:03d}/stability.json
    outputs/shap_bootstrap/stability_b_sweep.csv
    outputs/shap_bootstrap/stability_b_sweep.json
    outputs/shap_bootstrap/optimal_B.json

Two convergence figures are saved to ``--figures-dir``:

    shap_b_sweep_convergence_mean.pdf  (Mean ± Std vs B, faceted by metric)
    shap_b_sweep_boxplots.pdf          (boxplot of pairwise values per B)

The "optimal B" is selected per (model, metric) as the smallest B such that
the absolute change in the Mean estimator between B and the next sampled B
is at most ``--epsilon`` (default 0.01). If no B satisfies the criterion,
the largest B is reported as the recommendation.

Usage
-----
  # Smoke test (~10–20 minutes on CPU, depending on hardware):
  ./.venv/bin/python scripts/run_shap_bootstrap_b_sweep.py \\
      --sample-n 8000 --shap-sample-n 500 --background-n 50 \\
      --b-values 6 10 \\
      --output-dir outputs/shap_bootstrap_smoke \\
      --figures-dir outputs/figures/supplementary_smoke

  # Full B-sweep (~6–10 hours on CPU; resumes if interrupted):
  ./.venv/bin/python scripts/run_shap_bootstrap_b_sweep.py
"""

from __future__ import annotations

# IMPORTANT: cap BLAS / OpenMP threads BEFORE numpy / xgboost / lightgbm
# imports. On this 16-core machine, leaving BLAS at the default of 16 threads
# while letting XGBoost also use 16 threads causes thread oversubscription:
# we measured XGBoost.fit() at ~36 s/iteration with the default and ~0.4 s with
# OMP=1. The user can still override these by exporting the variables before
# launching the script (setdefault never clobbers a pre-existing value).
import os

for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

try:
    from run_homecredit_4model import (  # noqa: E402
        MODEL_NAMES,
        TARGET_COLUMN,
        _make_model,
        build_preprocessor,
        get_features,
        load_preprocessed,
        split_by_sort_key,
        summarise,
    )
    from run_shap_analysis import _coerce_binary_shap  # noqa: E402
except ModuleNotFoundError:
    from scripts.run_homecredit_4model import (  # noqa: E402
        MODEL_NAMES,
        TARGET_COLUMN,
        _make_model,
        build_preprocessor,
        get_features,
        load_preprocessed,
        split_by_sort_key,
        summarise,
    )
    from scripts.run_shap_analysis import _coerce_binary_shap  # noqa: E402


warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STRATEGY = "class_weight"   # Standard strategy for all four models.
DEFAULT_B_VALUES: Tuple[int, ...] = (30, 50, 80, 100)
DEFAULT_SHAP_SAMPLE_N = 5_000
DEFAULT_BACKGROUND_N = 100
DEFAULT_TOP_KS: Tuple[int, ...] = (5, 10)
DEFAULT_EPSILON = 0.01

# B_OFFSET keeps the random-seed ranges of different B values disjoint, so the
# four bootstrap loops are statistically independent (per the user's choice).
B_OFFSETS = {30: 0, 50: 1_000, 80: 2_000, 100: 3_000}


@dataclass(frozen=True)
class StabilityValues:
    """Pairwise stability values (one entry per pair) for one (B, model)."""

    jaccard_at_k: Dict[int, List[float]]   # k -> list of pairwise Jaccard@k
    spearman: List[float]                  # pairwise Spearman rho


# ---------------------------------------------------------------------------
# Per-model preprocessing & explainer setup (done ONCE per model)
# ---------------------------------------------------------------------------

def _prepare_model_artifacts(
    name: str,
    X_train_full: pd.DataFrame,
    X_test_full: pd.DataFrame,
    shap_sample_n: int,
    seed: int,
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Fit the preprocessor on the full training set and pre-transform both
    splits. Then sample a fixed set of ``shap_sample_n`` test rows.

    Freezing the preprocessor keeps:
      - feature columns identical across all bootstrap iterations,
      - the comparison between rankings well-defined (same axis ordering),
      - runtime in check (no need to refit ColumnTransformer per iteration).

    Returns
    -------
    preprocessor       fitted ColumnTransformer
    X_train_full_arr   numpy (n_train, F) transformed full train
    X_test_full_arr    numpy (n_test, F)  transformed full test
    X_test_sub_arr     numpy (m, F)       fixed SHAP test subsample
    test_idx           numpy (m,)         indices used to build X_test_sub_arr
    feature_names      list of length F
    """
    scale = (name == "LR")
    preprocessor = build_preprocessor(X_train_full, scale=scale)
    preprocessor.fit(X_train_full)

    X_train_full_arr = np.asarray(preprocessor.transform(X_train_full))
    X_test_full_arr = np.asarray(preprocessor.transform(X_test_full))

    rng = np.random.default_rng(seed)
    m = min(shap_sample_n, len(X_test_full_arr))
    test_idx = rng.choice(len(X_test_full_arr), size=m, replace=False)
    X_test_sub_arr = X_test_full_arr[test_idx]

    # All Home Credit features are numeric after preprocess_homecredit.py, so
    # the ColumnTransformer's output column order matches X_train_full.columns.
    feature_names = list(X_train_full.columns)
    if X_train_full_arr.shape[1] != len(feature_names):
        # Fallback: derive names from the fitted preprocessor.
        feature_names = list(preprocessor.get_feature_names_out())
    return preprocessor, X_train_full_arr, X_test_full_arr, X_test_sub_arr, test_idx, feature_names


def _fit_classifier_on_bootstrap(
    name: str,
    strategy: str,
    X_boot_arr: np.ndarray,
    y_boot: np.ndarray,
    seed: int,
) -> Any:
    """Train one classifier on a bootstrap-resampled training matrix."""
    clf = _make_model(name, seed, strategy)
    if name == "XGBoost" and strategy == "class_weight":
        neg = int((y_boot == 0).sum())
        pos = int((y_boot == 1).sum())
        clf.set_params(scale_pos_weight=neg / max(pos, 1))
    clf.fit(X_boot_arr, y_boot)
    return clf


def _shap_mean_abs(
    name: str,
    classifier: Any,
    X_train_arr: np.ndarray,
    X_test_sub_arr: np.ndarray,
    background_n: int,
    seed: int,
) -> np.ndarray:
    """Run the right SHAP explainer and return mean(|SHAP|) per feature."""
    rng = np.random.default_rng(seed)
    if name == "LR":
        bg_size = min(background_n, len(X_train_arr))
        bg_idx = rng.choice(len(X_train_arr), size=bg_size, replace=False)
        X_bg = np.asarray(X_train_arr[bg_idx])
        explainer = shap.LinearExplainer(classifier, X_bg)
        sv = explainer.shap_values(X_test_sub_arr)
    elif name in {"RF", "XGBoost", "LightGBM"}:
        explainer = shap.TreeExplainer(classifier)
        sv = explainer.shap_values(X_test_sub_arr)
    else:
        raise ValueError(f"Unsupported model: {name}")

    sv_2d = _coerce_binary_shap(sv)
    return np.abs(sv_2d).mean(axis=0)


# ---------------------------------------------------------------------------
# Per-iteration driver (with on-disk caching for resume support)
# ---------------------------------------------------------------------------

def _iteration_path(out_dir: Path, B: int, model: str, b: int) -> Path:
    return out_dir / f"B{B:03d}" / f"{model}_iter{b:03d}_mean_abs.csv"


def _load_cached_ranking(path: Path, expected_features: List[str]) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if list(df["feature"]) != expected_features:
        # Schema drift — ignore the cached file, recompute.
        return None
    return df["mean_abs_shap"].to_numpy(dtype=float)


def _save_ranking(path: Path, feature_names: List[str], mean_abs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).to_csv(path, index=False)


def _run_one_iteration(
    name: str,
    strategy: str,
    X_train_full_arr: np.ndarray,
    y_train_full: np.ndarray,
    X_test_sub_arr: np.ndarray,
    feature_names: List[str],
    background_n: int,
    seed: int,
    out_path: Path,
) -> Tuple[np.ndarray, bool]:
    """Run ONE bootstrap iteration for ONE model. Returns (ranking, was_cached)."""
    cached = _load_cached_ranking(out_path, feature_names)
    if cached is not None:
        return cached, True

    n = len(X_train_full_arr)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=n)
    X_boot_arr = X_train_full_arr[boot_idx]
    y_boot = y_train_full[boot_idx]

    classifier = _fit_classifier_on_bootstrap(name, strategy, X_boot_arr, y_boot, seed)
    mean_abs = _shap_mean_abs(
        name=name,
        classifier=classifier,
        X_train_arr=X_boot_arr,
        X_test_sub_arr=X_test_sub_arr,
        background_n=background_n,
        seed=seed,
    )
    if mean_abs.shape[0] != len(feature_names):
        raise RuntimeError(
            f"SHAP feature count {mean_abs.shape[0]} != expected {len(feature_names)} for {name}."
        )
    _save_ranking(out_path, feature_names, mean_abs)
    return mean_abs, False


# ---------------------------------------------------------------------------
# Stability metrics
# ---------------------------------------------------------------------------

def _jaccard_at_k(rank_a: np.ndarray, rank_b: np.ndarray, k: int) -> float:
    top_a = set(np.argsort(-rank_a, kind="stable")[:k].tolist())
    top_b = set(np.argsort(-rank_b, kind="stable")[:k].tolist())
    union = top_a | top_b
    return len(top_a & top_b) / len(union) if union else 1.0


def _spearman_rho(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    rho, _ = spearmanr(rank_a, rank_b)
    if np.isnan(rho):
        return 0.0
    return float(rho)


def _pairwise_stability(
    rankings: np.ndarray, top_ks: Tuple[int, ...]
) -> StabilityValues:
    """Compute pairwise stability over C(B, 2) pairs for all metrics."""
    B = rankings.shape[0]
    jaccard: Dict[int, List[float]] = {k: [] for k in top_ks}
    spearman: List[float] = []
    for i, j in combinations(range(B), 2):
        ra, rb = rankings[i], rankings[j]
        for k in top_ks:
            jaccard[k].append(_jaccard_at_k(ra, rb, k))
        spearman.append(_spearman_rho(ra, rb))
    return StabilityValues(jaccard_at_k=jaccard, spearman=spearman)


def _aggregate(values: StabilityValues) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k, vals in values.jaccard_at_k.items():
        arr = np.asarray(vals, dtype=float)
        out[f"jaccard@{k}"] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "n_pairs": int(arr.size),
        }
    arr = np.asarray(values.spearman, dtype=float)
    out["spearman"] = {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "n_pairs": int(arr.size),
    }
    return out


# ---------------------------------------------------------------------------
# Optimal-B selection
# ---------------------------------------------------------------------------

def _optimal_B(
    metric_means: Dict[int, float],
    epsilon: float,
) -> Tuple[int, Dict[int, float]]:
    """Pick the smallest B beyond which the Mean estimator changes by <= epsilon.

    ``metric_means`` is ``{B: mean_value}`` sorted by ascending B.
    Returns (chosen_B, deltas) where deltas[B] is |mean(B) - mean(next_B)|
    (deltas[max_B] is set to 0 by convention).
    """
    sorted_Bs = sorted(metric_means.keys())
    deltas: Dict[int, float] = {}
    chosen: Optional[int] = None
    for idx, B in enumerate(sorted_Bs):
        if idx == len(sorted_Bs) - 1:
            deltas[B] = 0.0
            continue
        nxt = sorted_Bs[idx + 1]
        d = abs(metric_means[B] - metric_means[nxt])
        deltas[B] = d
        if chosen is None and d <= epsilon:
            chosen = B
    if chosen is None:
        chosen = sorted_Bs[-1]
    return chosen, deltas


# ---------------------------------------------------------------------------
# Figures (supplementary)
# ---------------------------------------------------------------------------

def _plot_convergence(
    summary_rows: List[Dict[str, Any]],
    figures_dir: Path,
    metrics: List[str],
) -> None:
    """Mean (with ±Std band) of each stability metric vs B, faceted by metric."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary_rows)
    if df.empty:
        return

    color_map = {"LR": "#2196F3", "RF": "#4CAF50", "XGBoost": "#FF9800", "LightGBM": "#9C27B0"}
    style_map = {"LR": "-", "RF": "--", "XGBoost": "-.", "LightGBM": ":"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4.5), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for model in df["model"].unique():
            sub = df[(df["model"] == model)].sort_values("B")
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            ax.plot(
                sub["B"], sub[mean_col],
                marker="o", lw=1.6,
                color=color_map.get(model, "k"),
                ls=style_map.get(model, "-"),
                label=model,
            )
            ax.fill_between(
                sub["B"],
                sub[mean_col] - sub[std_col],
                sub[mean_col] + sub[std_col],
                color=color_map.get(model, "k"), alpha=0.10,
            )
        ax.set_xlabel("Bootstrap iterations B")
        ax.set_ylabel(f"Mean pairwise {metric}")
        ax.set_title(f"Convergence of {metric} (mean ± std)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "SHAP stability vs B — Home Credit, 4 models, independent bootstrap loops",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    out_pdf = figures_dir / "shap_b_sweep_convergence_mean.pdf"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=150)
    plt.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved convergence figure → {out_pdf.name}")


def _plot_boxplots(
    pairwise_data: Dict[Tuple[int, str], StabilityValues],
    figures_dir: Path,
    top_ks: Tuple[int, ...],
) -> None:
    """Boxplot of pairwise stability values per B per model."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = [f"jaccard@{k}" for k in top_ks] + ["spearman"]
    Bs = sorted({B for (B, _) in pairwise_data.keys()})
    models = sorted({m for (_, m) in pairwise_data.keys()})

    fig, axes = plt.subplots(len(metrics), len(models), figsize=(4 * len(models), 3.0 * len(metrics)),
                             sharey="row", sharex="col")
    if len(metrics) == 1:
        axes = np.array([axes])
    if len(models) == 1:
        axes = axes[:, None]

    for r, metric in enumerate(metrics):
        for c, model in enumerate(models):
            ax = axes[r, c]
            data = []
            labels = []
            for B in Bs:
                key = (B, model)
                if key not in pairwise_data:
                    continue
                stab = pairwise_data[key]
                if metric == "spearman":
                    data.append(np.asarray(stab.spearman, dtype=float))
                else:
                    k = int(metric.split("@")[1])
                    data.append(np.asarray(stab.jaccard_at_k[k], dtype=float))
                labels.append(f"B={B}")
            if not data:
                ax.set_visible(False)
                continue
            ax.boxplot(data, tick_labels=labels, showfliers=False, widths=0.55)
            if r == 0:
                ax.set_title(model, fontsize=10)
            if c == 0:
                ax.set_ylabel(metric)
            ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "SHAP stability — pairwise distributions per B (Home Credit)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    out_pdf = figures_dir / "shap_b_sweep_boxplots.pdf"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=150)
    plt.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved boxplots figure   → {out_pdf.name}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    input_path: Path,
    output_dir: Path,
    figures_dir: Optional[Path],
    models: List[str],
    strategy: str,
    sample_n: Optional[int],
    shap_sample_n: int,
    background_n: int,
    b_values: List[int],
    top_ks: Tuple[int, ...],
    seed: int,
    train_frac: float,
    val_frac: float,
    epsilon: float,
) -> Dict[str, Any]:
    print(f"Loading preprocessed Home Credit data from {input_path} …")
    df = load_preprocessed(input_path, sample_n, seed)
    print(f"  Loaded {len(df):,} rows, default rate={df[TARGET_COLUMN].mean():.2%}")

    train, val, test = split_by_sort_key(df, train_frac, val_frac)
    print(
        f"  Split → train={len(train):,} ({train[TARGET_COLUMN].mean():.2%})"
        f"  val={len(val):,} ({val[TARGET_COLUMN].mean():.2%})"
        f"  test={len(test):,} ({test[TARGET_COLUMN].mean():.2%})"
    )

    X_train_full, y_train_full = get_features(train)
    X_test_full, _ = get_features(test)
    y_train_full_arr = y_train_full.to_numpy(dtype=int)

    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Per-model artifacts: preprocessor + transformed matrices + test sample
    # -----------------------------------------------------------------
    print("\nPreparing per-model preprocessing artifacts (1 fit per model) …")
    artifacts: Dict[str, Dict[str, Any]] = {}
    for name in models:
        t0 = time.perf_counter()
        prep, X_tr_arr, X_te_arr, X_te_sub_arr, te_idx, feat_names = _prepare_model_artifacts(
            name=name,
            X_train_full=X_train_full,
            X_test_full=X_test_full,
            shap_sample_n=shap_sample_n,
            seed=seed,
        )
        artifacts[name] = {
            "preprocessor": prep,
            "X_train_full_arr": X_tr_arr,
            "X_test_full_arr": X_te_arr,
            "X_test_sub_arr": X_te_sub_arr,
            "test_idx": te_idx,
            "feature_names": feat_names,
        }
        print(
            f"  [{name}] features={len(feat_names)}, "
            f"train_arr={X_tr_arr.shape}, test_sub={X_te_sub_arr.shape}, "
            f"prep_seconds={time.perf_counter() - t0:.1f}"
        )

    # -----------------------------------------------------------------
    # Bootstrap loops, one per B value (independent random seeds per B)
    # -----------------------------------------------------------------
    pairwise_data: Dict[Tuple[int, str], StabilityValues] = {}
    aggregate_per_B: Dict[int, Dict[str, Dict[str, Dict[str, float]]]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for B in sorted(b_values):
        if B not in B_OFFSETS:
            offset = B * 100  # fallback for unusual B values
        else:
            offset = B_OFFSETS[B]

        print("\n" + "=" * 80)
        print(f"BOOTSTRAP LOOP — B={B}  (seed offset={offset})")
        print("=" * 80)

        # Storage of rankings per model: shape (B, F)
        rankings_per_model: Dict[str, np.ndarray] = {}
        for name in models:
            rankings_per_model[name] = np.zeros(
                (B, len(artifacts[name]["feature_names"])), dtype=float
            )

        cached_count = 0
        new_count = 0
        loop_start = time.perf_counter()

        for b in range(1, B + 1):
            iter_seed = seed + offset + b
            for name in models:
                out_path = _iteration_path(output_dir, B, name, b)
                t0 = time.perf_counter()
                ranking, was_cached = _run_one_iteration(
                    name=name,
                    strategy=strategy,
                    X_train_full_arr=artifacts[name]["X_train_full_arr"],
                    y_train_full=y_train_full_arr,
                    X_test_sub_arr=artifacts[name]["X_test_sub_arr"],
                    feature_names=artifacts[name]["feature_names"],
                    background_n=background_n,
                    seed=iter_seed,
                    out_path=out_path,
                )
                rankings_per_model[name][b - 1] = ranking
                if was_cached:
                    cached_count += 1
                else:
                    new_count += 1
                tag = "cached " if was_cached else "fitted "
                print(
                    f"  B={B:>3d}  iter={b:>3d}/{B:<3d}  {name:<8s}  "
                    f"{tag}  elapsed={time.perf_counter() - t0:6.1f}s",
                    flush=True,
                )

        loop_elapsed = time.perf_counter() - loop_start
        print(
            f"\n  B={B} loop done — cached={cached_count}, fitted={new_count}, "
            f"total_seconds={loop_elapsed:.1f}"
        )

        # ----- Stability per model for this B -----
        per_model_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        for name in models:
            stab = _pairwise_stability(rankings_per_model[name], top_ks)
            agg = _aggregate(stab)
            per_model_summary[name] = agg
            pairwise_data[(B, name)] = stab
            for metric_key, vals in agg.items():
                summary_rows.append({
                    "B": B,
                    "model": name,
                    "metric": metric_key,
                    "mean": vals["mean"],
                    "std": vals["std"],
                    "n_pairs": vals["n_pairs"],
                    f"{metric_key}_mean": vals["mean"],
                    f"{metric_key}_std": vals["std"],
                })
        aggregate_per_B[B] = per_model_summary

        # Persist this B's stability JSON
        b_dir = output_dir / f"B{B:03d}"
        b_dir.mkdir(parents=True, exist_ok=True)
        (b_dir / "stability.json").write_text(
            json.dumps(per_model_summary, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved stability.json → {b_dir / 'stability.json'}")

    # -----------------------------------------------------------------
    # Build a wide summary CSV (one row per (B, model)) with all metrics
    # -----------------------------------------------------------------
    metric_keys = [f"jaccard@{k}" for k in top_ks] + ["spearman"]
    wide_rows: List[Dict[str, Any]] = []
    for B in sorted(aggregate_per_B.keys()):
        for name in models:
            row: Dict[str, Any] = {"B": B, "model": name}
            for mk in metric_keys:
                vals = aggregate_per_B[B][name][mk]
                row[f"{mk}_mean"] = vals["mean"]
                row[f"{mk}_std"] = vals["std"]
                row[f"{mk}_n_pairs"] = vals["n_pairs"]
            wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows)
    wide_csv = output_dir / "stability_b_sweep.csv"
    wide_df.to_csv(wide_csv, index=False)
    print(f"\nSaved wide summary CSV  → {wide_csv}")

    # -----------------------------------------------------------------
    # Optimal B per (model, metric)
    # -----------------------------------------------------------------
    optimal: Dict[str, Any] = {"epsilon": epsilon, "by_model": {}, "global_recommendation": {}}
    for name in models:
        optimal["by_model"][name] = {}
        for mk in metric_keys:
            means_by_B: Dict[int, float] = {
                B: aggregate_per_B[B][name][mk]["mean"]
                for B in aggregate_per_B
            }
            chosen, deltas = _optimal_B(means_by_B, epsilon)
            optimal["by_model"][name][mk] = {
                "chosen_B": int(chosen),
                "means_by_B": {int(B): float(v) for B, v in means_by_B.items()},
                "deltas_by_B": {int(B): float(v) for B, v in deltas.items()},
            }

    # Global recommendation: max over all (model, metric) chosen_Bs.
    # Rationale: pick the smallest B that is "good enough" for the WORST
    # (most variable) metric/model combination, so every entry in Table 3
    # is past convergence.
    all_chosen = [
        optimal["by_model"][m][mk]["chosen_B"]
        for m in models
        for mk in metric_keys
    ]
    optimal["global_recommendation"]["B"] = int(max(all_chosen))
    optimal["global_recommendation"]["criterion"] = (
        f"max over (model, metric) of the smallest B with |delta_to_next| <= {epsilon}"
    )

    optimal_path = output_dir / "optimal_B.json"
    optimal_path.write_text(json.dumps(optimal, indent=2), encoding="utf-8")
    print(f"Saved optimal-B summary → {optimal_path}")

    # -----------------------------------------------------------------
    # Full JSON dump
    # -----------------------------------------------------------------
    full_summary: Dict[str, Any] = {
        "dataset": str(input_path),
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "strategy": strategy,
        "shap_sample_n": shap_sample_n,
        "background_n": background_n,
        "top_ks": list(top_ks),
        "b_values": sorted(b_values),
        "epsilon": epsilon,
        "splits": {
            "train": summarise(train).__dict__,
            "validation": summarise(val).__dict__,
            "test": summarise(test).__dict__,
        },
        "stability": {
            int(B): aggregate_per_B[B] for B in sorted(aggregate_per_B.keys())
        },
        "optimal_B": optimal,
    }
    full_json_path = output_dir / "stability_b_sweep.json"
    full_json_path.write_text(json.dumps(full_summary, indent=2, default=str), encoding="utf-8")
    print(f"Saved full JSON summary → {full_json_path}")

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    if figures_dir is not None:
        # Convergence (Mean ± Std vs B), one panel per metric.
        _plot_convergence(
            summary_rows=wide_rows,
            figures_dir=figures_dir,
            metrics=metric_keys,
        )
        _plot_boxplots(pairwise_data=pairwise_data, figures_dir=figures_dir, top_ks=top_ks)

    # -----------------------------------------------------------------
    # Console overview
    # -----------------------------------------------------------------
    _print_overview(aggregate_per_B, optimal, models, metric_keys)
    return full_summary


def _print_overview(
    aggregate_per_B: Dict[int, Dict[str, Dict[str, Dict[str, float]]]],
    optimal: Dict[str, Any],
    models: List[str],
    metric_keys: List[str],
) -> None:
    print("\n" + "=" * 84)
    print("BOOTSTRAP B-SWEEP — Mean(±Std) pairwise stability per (B, model)")
    print("=" * 84)
    Bs = sorted(aggregate_per_B.keys())

    for mk in metric_keys:
        print(f"\n  Metric: {mk}")
        header = f"  {'Model':<10s}" + "".join(f"  B={B:<7d}" for B in Bs)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name in models:
            cells = []
            for B in Bs:
                vals = aggregate_per_B[B][name][mk]
                cells.append(f"  {vals['mean']:.3f}±{vals['std']:.3f}")
            print(f"  {name:<10s}" + "".join(cells))

    print("\n" + "-" * 84)
    print(f"OPTIMAL B PER (MODEL, METRIC)  (epsilon={optimal['epsilon']})")
    print("-" * 84)
    for name in models:
        for mk in metric_keys:
            entry = optimal["by_model"][name][mk]
            deltas_str = ", ".join(
                f"B={B}->Δ{entry['deltas_by_B'][B]:.4f}" for B in sorted(entry["deltas_by_B"].keys())
            )
            print(
                f"  {name:<10s}  {mk:<11s}  chosen_B={entry['chosen_B']:<3d}   {deltas_str}"
            )
    rec_B = optimal["global_recommendation"]["B"]
    print("-" * 84)
    print(f"GLOBAL RECOMMENDATION  →  B = {rec_B}")
    print(f"  ({optimal['global_recommendation']['criterion']})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep B in {30, 50, 80, 100} to find the "
            "optimal number of bootstrap iterations for SHAP stability."
        ),
    )
    parser.add_argument(
        "--input",
        default="outputs/homecredit_preprocessed.csv",
        help="Path to preprocessed Home Credit CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/shap_bootstrap",
        help=(
            "Directory for per-iteration ranking CSVs, per-B stability JSONs, "
            "and the consolidated stability_b_sweep.csv/.json."
        ),
    )
    parser.add_argument(
        "--figures-dir",
        default="outputs/figures/supplementary",
        help="Directory for convergence and boxplot figures.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_NAMES,
        choices=MODEL_NAMES,
        metavar="MODEL",
        help="Subset of models to include (default: LR RF XGBoost LightGBM).",
    )
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        choices=["class_weight", "smote", "adasyn"],
        help="Imbalance strategy used at fit time (default: class_weight).",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Optional stratified sample of the dataset before splitting (smoke runs).",
    )
    parser.add_argument(
        "--shap-sample-n",
        type=int,
        default=DEFAULT_SHAP_SAMPLE_N,
        help="Number of fixed test rows used to compute SHAP every iteration.",
    )
    parser.add_argument(
        "--background-n",
        type=int,
        default=DEFAULT_BACKGROUND_N,
        help="Background sample size for LinearExplainer (LR).",
    )
    parser.add_argument(
        "--b-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_B_VALUES),
        help="Bootstrap iteration counts to sweep (default: 30 50 80 100).",
    )
    parser.add_argument(
        "--top-ks",
        nargs="+",
        type=int,
        default=list(DEFAULT_TOP_KS),
        help="Top-k values for the Jaccard@k metric (default: 5 10).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="Convergence threshold for selecting optimal B (default: 0.01).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    print(
        "[threading] OMP_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s MKL_NUM_THREADS=%s"
        % (
            os.environ.get("OMP_NUM_THREADS", "(unset)"),
            os.environ.get("OPENBLAS_NUM_THREADS", "(unset)"),
            os.environ.get("MKL_NUM_THREADS", "(unset)"),
        )
    )

    figures_dir = None if args.no_figures else Path(args.figures_dir)
    run(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        figures_dir=figures_dir,
        models=args.models,
        strategy=args.strategy,
        sample_n=args.sample_n,
        shap_sample_n=args.shap_sample_n,
        background_n=args.background_n,
        b_values=args.b_values,
        top_ks=tuple(args.top_ks),
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        epsilon=args.epsilon,
    )


if __name__ == "__main__":
    main()
