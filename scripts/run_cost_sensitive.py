#!/usr/bin/env python3
"""Run cost-sensitive threshold optimization on Home Credit.

This script treats Home Credit as the baseline dataset for threshold optimization.
Thresholds are selected on the validation split and then reported on the test split
to avoid leakage.
"""


from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

try:
    from run_homecredit_4model import (
        COLORS,
        LINE_STYLES,
        MODEL_NAMES,
        TARGET_COLUMN,
        load_preprocessed,
        run_strategy,
        split_by_sort_key,
        summarise,
    )
except ModuleNotFoundError:
    from scripts.run_homecredit_4model import (
        COLORS,
        LINE_STYLES,
        MODEL_NAMES,
        TARGET_COLUMN,
        load_preprocessed,
        run_strategy,
        split_by_sort_key,
        summarise,
    )


@dataclass(frozen=True)
class ThresholdEvaluation:
    threshold: float
    tpr: float
    fpr: float
    expected_cost: float
    precision: float
    recall: float
    specificity: float
    f1: float
    accuracy: float
    predicted_positive_rate: float


def make_threshold_grid(step: float) -> np.ndarray:
    """Return a [0, 1] grid that always includes 0.0, 0.5, and 1.0."""
    if step <= 0 or step > 1:
        raise ValueError("--threshold-step must be in the interval (0, 1].")
    grid = np.arange(0.0, 1.0 + step / 2.0, step, dtype=float)
    grid = np.clip(grid, 0.0, 1.0)
    grid = np.unique(np.concatenate([grid, np.array([0.0, 0.5, 1.0])]))
    return np.round(grid, 10)


def expected_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    cost_fn: float,
    cost_fp: float,
) -> float:
    """Compute expected cost for PD scores where y=1 means default.

    A score above the threshold predicts default/reject. Therefore approving a
    borrower who defaults is a false negative (cost_fn), while rejecting a good
    borrower is a false positive (cost_fp).
    """
    if cost_fn < 0 or cost_fp < 0:
        raise ValueError("Misclassification costs must be non-negative.")

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")
    if y_true.size == 0:
        raise ValueError("At least one observation is required.")

    preds = (y_score >= threshold).astype(int)
    positives = y_true == 1
    negatives = y_true == 0

    pi1 = float(np.mean(positives))
    pi0 = float(np.mean(negatives))
    tpr = float(np.mean(preds[positives] == 1)) if positives.any() else 0.0
    fpr = float(np.mean(preds[negatives] == 1)) if negatives.any() else 0.0
    return pi1 * (1.0 - tpr) * cost_fn + pi0 * fpr * cost_fp


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    cost_fn: float,
    cost_fp: float,
) -> ThresholdEvaluation:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    preds = (y_score >= threshold).astype(int)

    positives = y_true == 1
    negatives = y_true == 0
    tpr = float(np.mean(preds[positives] == 1)) if positives.any() else float("nan")
    fpr = float(np.mean(preds[negatives] == 1)) if negatives.any() else float("nan")
    specificity = 1.0 - fpr if not math.isnan(fpr) else float("nan")

    return ThresholdEvaluation(
        threshold=float(threshold),
        tpr=tpr,
        fpr=fpr,
        expected_cost=expected_cost(y_true, y_score, threshold, cost_fn, cost_fp),
        precision=float(precision_score(y_true, preds, zero_division=0)),
        recall=float(recall_score(y_true, preds, zero_division=0)),
        specificity=specificity,
        f1=float(f1_score(y_true, preds, zero_division=0)),
        accuracy=float(accuracy_score(y_true, preds)),
        predicted_positive_rate=float(np.mean(preds)),
    )


def sweep_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cost_fn: float,
    cost_fp: float,
    threshold_step: float,
) -> pd.DataFrame:
    rows = [
        asdict(evaluate_at_threshold(y_true, y_score, threshold, cost_fn, cost_fp))
        for threshold in make_threshold_grid(threshold_step)
    ]
    return pd.DataFrame(rows)


def select_cost_optimal_threshold(sweep: pd.DataFrame) -> float:
    """Pick the lowest threshold among ties for deterministic behavior."""
    if sweep.empty:
        raise ValueError("Threshold sweep is empty.")
    best_ec = sweep["expected_cost"].min()
    best = sweep.loc[sweep["expected_cost"] == best_ec].sort_values("threshold").iloc[0]
    return float(best["threshold"])


def select_youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return the ROC threshold maximizing TPR - FPR."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[idx])
    if not np.isfinite(threshold):
        return 1.0
    return float(np.clip(threshold, 0.0, 1.0))


def cost_savings_ratio(baseline_ec: float, optimized_ec: float) -> float:
    if baseline_ec == 0:
        return 0.0 if optimized_ec == 0 else float("nan")
    return float((baseline_ec - optimized_ec) / baseline_ec * 100.0)


def run_sensitivity_analysis(
    predictions: Dict[str, Dict[str, np.ndarray]],
    models: Iterable[str],
    ratios: Iterable[float],
    threshold_step: float,
) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for ratio in ratios:
        for model in models:
            pred = predictions[model]
            val_sweep = sweep_thresholds(
                pred["validation_y_true"],
                pred["validation_scores"],
                cost_fn=float(ratio),
                cost_fp=1.0,
                threshold_step=threshold_step,
            )
            threshold = select_cost_optimal_threshold(val_sweep)
            test_eval = evaluate_at_threshold(
                pred["y_true"],
                pred["test_scores"],
                threshold,
                cost_fn=float(ratio),
                cost_fp=1.0,
            )
            rows.append(
                {
                    "dataset": "Home Credit",
                    "model": model,
                    "cost_fn": float(ratio),
                    "cost_fp": 1.0,
                    "cost_ratio": float(ratio),
                    "threshold_ec": threshold,
                    "test_expected_cost": test_eval.expected_cost,
                    "test_tpr": test_eval.tpr,
                    "test_fpr": test_eval.fpr,
                }
            )
    return pd.DataFrame(rows)


def _rows_for_model(
    model: str,
    strategy: str,
    pred: Dict[str, np.ndarray],
    cost_fn: float,
    cost_fp: float,
    threshold_step: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    validation_sweep = sweep_thresholds(
        pred["validation_y_true"],
        pred["validation_scores"],
        cost_fn,
        cost_fp,
        threshold_step,
    )
    fixed_threshold = 0.5
    youden_threshold = select_youden_threshold(pred["validation_y_true"], pred["validation_scores"])
    ec_threshold = select_cost_optimal_threshold(validation_sweep)

    thresholds = {
        "fixed_0_5": fixed_threshold,
        "youden": youden_threshold,
        "cost_optimal": ec_threshold,
    }

    rows: list[dict[str, Any]] = []
    evals: dict[str, ThresholdEvaluation] = {}
    baseline_eval: Optional[ThresholdEvaluation] = None
    for label, threshold in thresholds.items():
        test_eval = evaluate_at_threshold(pred["y_true"], pred["test_scores"], threshold, cost_fn, cost_fp)
        evals[label] = test_eval
        if label == "fixed_0_5":
            baseline_eval = test_eval

    assert baseline_eval is not None
    for label, test_eval in evals.items():
        row = {
            "dataset": "Home Credit",
            "model": model,
            "strategy": strategy,
            "threshold_type": label,
            "threshold": test_eval.threshold,
            "cost_fn": cost_fn,
            "cost_fp": cost_fp,
            "cost_ratio": cost_fn / cost_fp if cost_fp else float("inf"),
            "cost_savings_vs_0_5_pct": cost_savings_ratio(
                baseline_eval.expected_cost,
                test_eval.expected_cost,
            ),
            **asdict(test_eval),
        }
        rows.append(row)

    opt = evals["cost_optimal"]
    table10_row = {
        "dataset": "Home Credit",
        "model": model,
        "strategy": strategy,
        "threshold_ec": opt.threshold,
        "ec_at_threshold": opt.expected_cost,
        "ec_at_0_5": baseline_eval.expected_cost,
        "cost_savings_vs_0_5_pct": cost_savings_ratio(baseline_eval.expected_cost, opt.expected_cost),
        "tpr_at_threshold": opt.tpr,
        "fpr_at_threshold": opt.fpr,
        "precision_at_threshold": opt.precision,
        "recall_at_threshold": opt.recall,
        "specificity_at_threshold": opt.specificity,
        "f1_at_threshold": opt.f1,
        "accuracy_at_threshold": opt.accuracy,
    }
    return rows, table10_row, validation_sweep


def _plot_ec_curves(
    sweeps_by_model: Dict[str, pd.DataFrame],
    table10: pd.DataFrame,
    figures_dir: Path,
) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, sweep in sweeps_by_model.items():
        ax.plot(
            sweep["threshold"],
            sweep["expected_cost"],
            color=COLORS.get(model),
            linestyle=LINE_STYLES.get(model, "-"),
            linewidth=1.8,
            label=model,
        )
        threshold = float(table10.loc[table10["model"] == model, "threshold_ec"].iloc[0])
        ec_value = float(sweep.loc[sweep["threshold"] == threshold, "expected_cost"].iloc[0])
        ax.axvline(threshold, color=COLORS.get(model), alpha=0.25, linewidth=1.0)
        ax.scatter([threshold], [ec_value], color=COLORS.get(model), s=24)

    ax.set_title("Expected Cost Curves on Home Credit Validation Set", fontsize=12)
    ax.set_xlabel("Decision threshold", fontsize=11)
    ax.set_ylabel("Expected Cost", fontsize=11)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = figures_dir / "fig7_ec_curves_homecredit.pdf"
    png_path = path.with_suffix(".png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return [str(path), str(png_path)]


def _plot_sensitivity_heatmap(sensitivity: pd.DataFrame, figures_dir: Path) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    heatmap = sensitivity.pivot(index="model", columns="cost_ratio", values="threshold_ec")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.heatmap(
        heatmap,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Validation-selected EC threshold"},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Cost Sensitivity: EC-Optimal Thresholds on Home Credit", fontsize=12)
    ax.set_xlabel("C_FN / C_FP ratio", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    fig.tight_layout()

    path = figures_dir / "fig8_cost_sensitivity_homecredit.pdf"
    png_path = path.with_suffix(".png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return [str(path), str(png_path)]


def _copy_to_paper(figures: Iterable[str], paper_figures_dir: Optional[Path]) -> list[str]:
    if paper_figures_dir is None:
        return []
    paper_figures_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for figure in figures:
        src = Path(figure)
        dst = paper_figures_dir / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def run_cost_sensitive(
    input_path: Path,
    output_json: Path,
    tables_dir: Path,
    figures_dir: Path,
    paper_figures_dir: Optional[Path],
    sample_n: Optional[int],
    models: List[str],
    seed: int,
    train_frac: float,
    val_frac: float,
    cost_fn: float,
    cost_fp: float,
    threshold_step: float,
    sensitivity_ratios: List[float],
) -> Dict[str, Any]:
    print(f"\nLoading Home Credit data from {input_path} ...")
    df = load_preprocessed(input_path, sample_n, seed)
    train, val, test = split_by_sort_key(df, train_frac, val_frac)
    print(
        f"  Split -> train={len(train):,} ({train[TARGET_COLUMN].mean():.2%})"
        f"  val={len(val):,} ({val[TARGET_COLUMN].mean():.2%})"
        f"  test={len(test):,} ({test[TARGET_COLUMN].mean():.2%})"
    )

    print("\n  Training Home Credit models with CLASS_WEIGHT strategy ...")
    strategy = "class_weight"
    strategy_out = run_strategy(strategy, train, val, test, models, seed)
    predictions = strategy_out["predictions"]

    table9_rows: list[dict[str, Any]] = []
    table10_rows: list[dict[str, Any]] = []
    sweeps_by_model: dict[str, pd.DataFrame] = {}

    for model in models:
        rows, table10_row, validation_sweep = _rows_for_model(
            model,
            strategy,
            predictions[model],
            cost_fn,
            cost_fp,
            threshold_step,
        )
        table9_rows.extend(rows)
        table10_rows.append(table10_row)
        sweeps_by_model[model] = validation_sweep

    table9 = pd.DataFrame(table9_rows)
    table10 = pd.DataFrame(table10_rows)
    lr_baseline_ec = float(
        table9.loc[
            (table9["model"] == "LR") & (table9["threshold_type"] == "fixed_0_5"),
            "expected_cost",
        ].iloc[0]
    )
    table10["cost_savings_vs_lr_0_5_pct"] = table10["ec_at_threshold"].map(
        lambda ec: cost_savings_ratio(lr_baseline_ec, float(ec))
    )
    table10 = table10.sort_values("ec_at_threshold").reset_index(drop=True)
    table10["ec_rank"] = np.arange(1, len(table10) + 1)

    sensitivity = run_sensitivity_analysis(predictions, models, sensitivity_ratios, threshold_step)

    tables_dir.mkdir(parents=True, exist_ok=True)
    table9_path = tables_dir / "table9_threshold_comparison_homecredit.csv"
    table10_path = tables_dir / "table10_expected_cost_homecredit.csv"
    sensitivity_path = tables_dir / "cost_sensitivity_homecredit.csv"
    table9.to_csv(table9_path, index=False)
    table10.to_csv(table10_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)

    generated_figures = []
    generated_figures.extend(_plot_ec_curves(sweeps_by_model, table10, figures_dir))
    generated_figures.extend(_plot_sensitivity_heatmap(sensitivity, figures_dir))
    copied_figures = _copy_to_paper(generated_figures, paper_figures_dir)

    result = {
        "dataset": "Home Credit",
        "dataset_status": {
            "home_credit": "completed",
            "internal_bank_data": "pending_real_data_from_advisor",
        },
        "input": str(input_path),
        "total_rows": len(df),
        "seed": seed,
        "cost_matrix": {
            "cost_fn": cost_fn,
            "cost_fp": cost_fp,
            "cost_ratio": cost_fn / cost_fp if cost_fp else float("inf"),
            "note": "Normalized C_FN:C_FP = 5:1 primary assumption; absolute EAD cancels out under constant costs.",
        },
        "threshold_selection": "Validation split only; test split used for final reporting.",
        "splits": {
            "train": asdict(summarise(train)),
            "validation": asdict(summarise(val)),
            "test": asdict(summarise(test)),
        },
        "models": models,
        "strategy": strategy,
        "table9_csv": str(table9_path),
        "table10_csv": str(table10_path),
        "sensitivity_csv": str(sensitivity_path),
        "table9_threshold_comparison": table9.to_dict(orient="records"),
        "table10_expected_cost": table10.to_dict(orient="records"),
        "sensitivity_analysis": sensitivity.to_dict(orient="records"),
        "generated_figures": generated_figures,
        "copied_paper_figures": copied_figures,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\n  Table 9 CSV -> {table9_path}")
    print(f"  Table 10 CSV -> {table10_path}")
    print(f"  Sensitivity CSV -> {sensitivity_path}")
    print(f"  Results JSON -> {output_json}")
    print(f"  Figures -> {figures_dir}")
    if copied_figures:
        print(f"  Paper figure copies -> {paper_figures_dir}")

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run cost-sensitive threshold optimization on Home Credit."
    )
    parser.add_argument("--input", default="outputs/homecredit_preprocessed.csv")
    parser.add_argument("--output-json", default="outputs/cost_sensitive_homecredit.json")
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--figures-dir", default="outputs/figures")
    parser.add_argument("--paper-figures-dir", default="paper/llncs/figures")
    parser.add_argument("--no-paper-copy", action="store_true", help="Do not copy generated figures into paper/llncs/figures.")
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES, metavar="MODEL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--cost-fn", type=float, default=5.0)
    parser.add_argument("--cost-fp", type=float, default=1.0)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--sensitivity-ratios",
        type=float,
        nargs="+",
        default=[3, 4, 5, 6, 7, 8, 9, 10],
        help="C_FN/C_FP ratios to scan with C_FP fixed at 1.",
    )
    return parser


def _print_summary(result: Dict[str, Any]) -> None:
    print("\nTABLE 10 CANDIDATE - Home Credit expected cost at validation-selected EC threshold")
    print(f"{'Rank':>4s}  {'Model':<10s}  {'t*_EC':>7s}  {'EC(t*)':>10s}  {'EC(0.5)':>10s}  {'CSR%':>8s}  {'vs LR@0.5%':>12s}")
    print("-" * 74)
    for row in result["table10_expected_cost"]:
        print(
            f"{int(row['ec_rank']):>4d}  {row['model']:<10s}  {row['threshold_ec']:>7.2f}"
            f"  {row['ec_at_threshold']:>10.6f}  {row['ec_at_0_5']:>10.6f}"
            f"  {row['cost_savings_vs_0_5_pct']:>8.2f}  {row['cost_savings_vs_lr_0_5_pct']:>12.2f}"
        )


def main() -> None:
    args = build_arg_parser().parse_args()
    paper_figures_dir = None if args.no_paper_copy else Path(args.paper_figures_dir)
    result = run_cost_sensitive(
        input_path=Path(args.input),
        output_json=Path(args.output_json),
        tables_dir=Path(args.tables_dir),
        figures_dir=Path(args.figures_dir),
        paper_figures_dir=paper_figures_dir,
        sample_n=args.sample_n,
        models=args.models,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        cost_fn=args.cost_fn,
        cost_fp=args.cost_fp,
        threshold_step=args.threshold_step,
        sensitivity_ratios=args.sensitivity_ratios,
    )
    _print_summary(result)


if __name__ == "__main__":
    main()

