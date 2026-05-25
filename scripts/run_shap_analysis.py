#!/usr/bin/env python3
"""Global SHAP analysis for the 4-model Home Credit pipeline.

For each of LR, RF, XGBoost, LightGBM we:
  1. Reload the preprocessed Home Credit data and rebuild the same time-ordered
     60/20/20 train/val/test split used in Week 4 / Week 5.
  2. Retrain the model with the best imbalance strategy selected in Week 4
     (``class_weight`` for all four models).
  3. Compute SHAP values on a random subsample of the test set:
       - LR        -> ``shap.LinearExplainer``
       - RF        -> ``shap.TreeExplainer``
       - XGBoost   -> ``shap.TreeExplainer``
       - LightGBM  -> ``shap.TreeExplainer``
  4. Aggregate mean(|SHAP|) per feature and write a per-model CSV ranking to
     ``outputs/shap/``.
  5. Save beeswarm + top-k bar plots to ``outputs/figures/supplementary/``
     (these stay outside the 4-figure main budget; see notes/outline.md).

The script is the first half of the SHAP track. Day 2 (``run_shap_bootstrap.py``)
reuses the same training scaffolding inside a 50-iteration resampling loop.

Usage
-----
  # Full run (uses outputs/homecredit_preprocessed.csv by default):
  ./.venv/bin/python scripts/run_shap_analysis.py

  # Quick smoke (3k rows, 500 SHAP samples):
  ./.venv/bin/python scripts/run_shap_analysis.py \\
      --sample-n 3000 --shap-sample-n 500 --background-n 50 --no-figures

  # Limit which models run:
  ./.venv/bin/python scripts/run_shap_analysis.py --models LR LightGBM
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

try:
    from run_homecredit_4model import (
        MODEL_NAMES,
        TARGET_COLUMN,
        apply_oversampling,
        build_pipeline,
        get_features,
        load_preprocessed,
        split_by_sort_key,
        summarise,
    )
except ModuleNotFoundError:
    from scripts.run_homecredit_4model import (
        MODEL_NAMES,
        TARGET_COLUMN,
        apply_oversampling,
        build_pipeline,
        get_features,
        load_preprocessed,
        split_by_sort_key,
        summarise,
    )


warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STRATEGY = "class_weight"  # Standard strategy for all 4 Home Credit models.
DEFAULT_SHAP_SAMPLE_N = 5_000
DEFAULT_BACKGROUND_N = 100
DEFAULT_TOP_K = 20


@dataclass(frozen=True)
class ModelShapResult:
    model: str
    strategy: str
    n_test_samples: int
    n_features: int
    n_background: int
    elapsed_seconds: float
    mean_abs_shap: List[Tuple[str, float]]  # (feature, mean_abs_shap)


# ---------------------------------------------------------------------------
# Training (slim version specialised for SHAP — keeps preprocessor + clf)
# ---------------------------------------------------------------------------

def _fit_model_for_shap(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    strategy: str,
) -> Tuple[Any, Any, np.ndarray]:
    """Fit ``(preprocessor, classifier)`` and return the transformed train matrix.

    Returns
    -------
    preprocessor : fitted ``ColumnTransformer``.
    classifier   : fitted estimator that takes already-transformed input.
    X_train_arr  : transformed training data (numpy), useful for SHAP backgrounds.
    """
    pipe = build_pipeline(name, X_train, seed, strategy)
    pipe["prep"].fit(X_train)
    X_train_arr = pipe["prep"].transform(X_train)
    y_train_arr = y_train.values

    if strategy in ("smote", "adasyn"):
        X_train_arr, y_train_arr = apply_oversampling(strategy, X_train_arr, y_train_arr, seed)

    if name == "XGBoost" and strategy == "class_weight":
        neg = int((y_train_arr == 0).sum())
        pos = int((y_train_arr == 1).sum())
        pipe["clf"].set_params(scale_pos_weight=neg / max(pos, 1))

    pipe["clf"].fit(X_train_arr, y_train_arr)
    return pipe["prep"], pipe["clf"], X_train_arr


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def _coerce_binary_shap(values: Any) -> np.ndarray:
    """Reduce SHAP output to a 2-D array of shape ``(n_samples, n_features)``.

    SHAP versions / model types disagree on the layout for binary classifiers:
      * XGBoost / LightGBM (binary:logistic, sklearn API) -> ``(n, F)``
        on the log-odds scale.
      * sklearn ``RandomForestClassifier`` -> either ``(n, F, 2)`` or
        ``[shap_class0, shap_class1]``; we always take the positive class.
      * ``LinearExplainer`` -> ``(n, F)`` for class 1.
    """
    if isinstance(values, list):
        # Older API: list of arrays, one per class. Use the positive class.
        values = values[1] if len(values) >= 2 else values[0]
    arr = np.asarray(values)
    if arr.ndim == 3:
        # Shape (n, F, n_classes). Take class 1.
        arr = arr[..., 1]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")
    return arr


def _compute_shap_values(
    name: str,
    classifier: Any,
    X_train_arr: np.ndarray,
    X_test_arr: np.ndarray,
    background_n: int,
    seed: int,
) -> np.ndarray:
    """Run the appropriate SHAP explainer and return ``(n, F)`` SHAP values."""
    rng = np.random.default_rng(seed)

    if name == "LR":
        bg_size = min(background_n, len(X_train_arr))
        bg_idx = rng.choice(len(X_train_arr), size=bg_size, replace=False)
        X_bg = np.asarray(X_train_arr[bg_idx])
        explainer = shap.LinearExplainer(classifier, X_bg)
        sv = explainer.shap_values(X_test_arr)
        return _coerce_binary_shap(sv)

    if name in {"RF", "XGBoost", "LightGBM"}:
        explainer = shap.TreeExplainer(classifier)
        sv = explainer.shap_values(X_test_arr)
        return _coerce_binary_shap(sv)

    raise ValueError(f"Unsupported model for SHAP: {name}")


# ---------------------------------------------------------------------------
# Per-model driver
# ---------------------------------------------------------------------------

def _run_one_model(
    name: str,
    strategy: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    shap_sample_n: int,
    background_n: int,
    output_dir: Path,
    figures_dir: Optional[Path],
    top_k: int,
) -> ModelShapResult:
    print(f"\n[{name}] strategy={strategy}")
    start = time.perf_counter()

    X_train, y_train = get_features(train)
    X_test, y_test = get_features(test)
    feature_names = list(X_train.columns)

    print(f"  Fitting model on {len(X_train):,} train rows × {len(feature_names)} features …")
    preprocessor, classifier, X_train_arr = _fit_model_for_shap(
        name=name, X_train=X_train, y_train=y_train, seed=seed, strategy=strategy
    )

    X_test_arr = preprocessor.transform(X_test)

    rng = np.random.default_rng(seed)
    test_sample_size = min(shap_sample_n, len(X_test_arr))
    test_idx = rng.choice(len(X_test_arr), size=test_sample_size, replace=False)
    X_test_sub = np.asarray(X_test_arr[test_idx])

    print(f"  Computing SHAP on {test_sample_size:,} test rows "
          f"(background={min(background_n, len(X_train_arr)):,}) …")
    shap_values = _compute_shap_values(
        name=name,
        classifier=classifier,
        X_train_arr=X_train_arr,
        X_test_arr=X_test_sub,
        background_n=background_n,
        seed=seed,
    )

    if shap_values.shape[1] != len(feature_names):
        raise RuntimeError(
            f"SHAP value count ({shap_values.shape[1]}) does not match feature count "
            f"({len(feature_names)}) for {name}. Check the preprocessing pipeline."
        )

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = sorted(
        zip(feature_names, mean_abs.tolist()),
        key=lambda kv: kv[1],
        reverse=True,
    )

    # Persist ranking CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_df = pd.DataFrame(
        {
            "rank": np.arange(1, len(ranking) + 1, dtype=int),
            "feature": [f for f, _ in ranking],
            "mean_abs_shap": [v for _, v in ranking],
        }
    )
    ranking_path = output_dir / f"{name}_shap_mean_abs.csv"
    ranking_df.to_csv(ranking_path, index=False)
    print(f"  Saved ranking → {ranking_path}")

    # Figures (supplementary)
    if figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)
        _save_supplementary_figures(
            name=name,
            shap_values=shap_values,
            X_test_sub=X_test_sub,
            feature_names=feature_names,
            top_k=top_k,
            out_dir=figures_dir,
        )

    elapsed = time.perf_counter() - start

    print(f"  Top-5 features for {name}:")
    for i, (feat, val) in enumerate(ranking[:5], start=1):
        print(f"    {i:>2d}. {feat:<40s}  mean|SHAP|={val:.6f}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return ModelShapResult(
        model=name,
        strategy=strategy,
        n_test_samples=int(test_sample_size),
        n_features=int(len(feature_names)),
        n_background=int(min(background_n, len(X_train_arr))) if name == "LR" else 0,
        elapsed_seconds=float(elapsed),
        mean_abs_shap=ranking,
    )


# ---------------------------------------------------------------------------
# Supplementary figures (beeswarm + bar)
# ---------------------------------------------------------------------------

def _save_supplementary_figures(
    name: str,
    shap_values: np.ndarray,
    X_test_sub: np.ndarray,
    feature_names: List[str],
    top_k: int,
    out_dir: Path,
) -> None:
    explanation = shap.Explanation(
        values=shap_values,
        data=X_test_sub,
        feature_names=feature_names,
    )

    # Beeswarm
    plt.figure(figsize=(8, 6))
    try:
        shap.plots.beeswarm(explanation, max_display=top_k, show=False)
        ax = plt.gca()
        ax.set_title(f"SHAP Beeswarm — {name} (Home Credit test set)", fontsize=11)
        beeswarm_pdf = out_dir / f"shap_beeswarm_{name}.pdf"
        plt.savefig(beeswarm_pdf, bbox_inches="tight", dpi=150)
        plt.savefig(beeswarm_pdf.with_suffix(".png"), bbox_inches="tight", dpi=150)
        print(f"  Saved beeswarm → {beeswarm_pdf.name}")
    finally:
        plt.close("all")

    # Bar plot of top-k mean |SHAP|
    plt.figure(figsize=(8, 6))
    try:
        shap.plots.bar(explanation, max_display=top_k, show=False)
        ax = plt.gca()
        ax.set_title(f"Top {top_k} Features by mean(|SHAP|) — {name}", fontsize=11)
        bar_pdf = out_dir / f"shap_bar_{name}.pdf"
        plt.savefig(bar_pdf, bbox_inches="tight", dpi=150)
        plt.savefig(bar_pdf.with_suffix(".png"), bbox_inches="tight", dpi=150)
        print(f"  Saved bar     → {bar_pdf.name}")
    finally:
        plt.close("all")


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
    seed: int,
    train_frac: float,
    val_frac: float,
    top_k: int,
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

    results: Dict[str, ModelShapResult] = {}
    for name in models:
        results[name] = _run_one_model(
            name=name,
            strategy=strategy,
            train=train,
            test=test,
            seed=seed,
            shap_sample_n=shap_sample_n,
            background_n=background_n,
            output_dir=output_dir,
            figures_dir=figures_dir,
            top_k=top_k,
        )

    summary: Dict[str, Any] = {
        "dataset": str(input_path),
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "strategy": strategy,
        "shap_sample_n": shap_sample_n,
        "background_n": background_n,
        "top_k": top_k,
        "splits": {
            "train": summarise(train).__dict__,
            "validation": summarise(val).__dict__,
            "test": summarise(test).__dict__,
        },
        "models": {},
    }
    for name, res in results.items():
        summary["models"][name] = {
            "strategy": res.strategy,
            "n_test_samples": res.n_test_samples,
            "n_features": res.n_features,
            "n_background": res.n_background,
            "elapsed_seconds": res.elapsed_seconds,
            "top_features": [
                {"rank": i + 1, "feature": feat, "mean_abs_shap": val}
                for i, (feat, val) in enumerate(res.mean_abs_shap[:top_k])
            ],
        }

    summary_path = output_dir / "shap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nSummary saved → {summary_path}")

    _print_overview(results, top_k)
    return summary


def _print_overview(results: Dict[str, ModelShapResult], top_k: int) -> None:
    print("\n" + "=" * 80)
    print(f"Global SHAP rankings (top-{min(top_k, 10)} per model)")
    print("=" * 80)
    for name, res in results.items():
        print(f"\n  {name}  (strategy={res.strategy}, n_test={res.n_test_samples})")
        for i, (feat, val) in enumerate(res.mean_abs_shap[: min(top_k, 10)], start=1):
            print(f"    {i:>2d}. {feat:<40s}  mean|SHAP|={val:.6f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate global SHAP rankings for the 4-model Home Credit pipeline.",
    )
    parser.add_argument(
        "--input",
        default="outputs/homecredit_preprocessed.csv",
        help="Path to preprocessed Home Credit CSV (output of preprocess_homecredit.py).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/shap",
        help="Directory to write per-model mean(|SHAP|) CSVs and the summary JSON.",
    )
    parser.add_argument(
        "--figures-dir",
        default="outputs/figures/supplementary",
        help="Directory for supplementary SHAP beeswarm/bar plots.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_NAMES,
        choices=MODEL_NAMES,
        metavar="MODEL",
        help="Subset of models to explain (default: LR RF XGBoost LightGBM).",
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
        help="Optional stratified sample of the full dataset before splitting (for fast smoke runs).",
    )
    parser.add_argument(
        "--shap-sample-n",
        type=int,
        default=DEFAULT_SHAP_SAMPLE_N,
        help="Number of test-set rows on which to compute SHAP values per model.",
    )
    parser.add_argument(
        "--background-n",
        type=int,
        default=DEFAULT_BACKGROUND_N,
        help="Background sample size for LinearExplainer (LR).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k features to render in supplementary figures and JSON summary.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip beeswarm/bar figure generation (only CSVs + JSON).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

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
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
