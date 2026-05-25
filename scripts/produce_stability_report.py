#!/usr/bin/env python3
"""Day 2 report script — produce Table 3 CSV and Figure 3 from B=50 bootstrap results.

Reads per-iteration mean(|SHAP|) CSVs from ``outputs/shap_bootstrap/B050/``,
recomputes all C(B,2) pairwise stability values (Jaccard@{5,10}, Spearman ρ),
then writes:

    outputs/tables/table3_stability_homecredit.csv
    outputs/figures/fig3_stability_boxplots.{pdf,png}
    outputs/figures/supplementary/fig3_stability_boxplots_full.{pdf,png}

All computation is CPU-light (no model fitting, no SHAP).  Typical runtime
on the full 228-feature, 50-iteration, 4-model dataset: < 30 seconds.

Usage
-----
    ./.venv/bin/python scripts/produce_stability_report.py

    # Custom B or paths:
    ./.venv/bin/python scripts/produce_stability_report.py \\
        --bootstrap-dir outputs/shap_bootstrap \\
        --b 50 --top-ks 5 10 \\
        --table-out outputs/tables/table3_stability_homecredit.csv \\
        --figures-dir outputs/figures
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Stability helpers (mirrors run_shap_bootstrap_b_sweep.py logic exactly)
# ---------------------------------------------------------------------------

def _jaccard_at_k(rank_a: np.ndarray, rank_b: np.ndarray, k: int) -> float:
    top_a = set(np.argsort(-rank_a, kind="stable")[:k].tolist())
    top_b = set(np.argsort(-rank_b, kind="stable")[:k].tolist())
    union = top_a | top_b
    return len(top_a & top_b) / len(union) if union else 1.0


def _spearman(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    rho, _ = spearmanr(rank_a, rank_b)
    return 0.0 if np.isnan(rho) else float(rho)


# ---------------------------------------------------------------------------
# Load per-iteration rankings
# ---------------------------------------------------------------------------

def _load_rankings(b_dir: Path, model: str, B: int) -> Tuple[np.ndarray, List[str]]:
    """Return (rankings_array of shape (B, F), feature_names)."""
    rankings: List[np.ndarray] = []
    feature_names: Optional[List[str]] = None
    for b in range(1, B + 1):
        path = b_dir / f"{model}_iter{b:03d}_mean_abs.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing iteration file: {path}\n"
                f"Run the bootstrap sweep first for B={B}."
            )
        df = pd.read_csv(path)
        if feature_names is None:
            feature_names = df["feature"].tolist()
        rankings.append(df["mean_abs_shap"].to_numpy(dtype=float))
    return np.stack(rankings, axis=0), feature_names  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Pairwise stability computation
# ---------------------------------------------------------------------------

def compute_pairwise(
    rankings: np.ndarray,
    top_ks: Tuple[int, ...],
) -> Dict[str, List[float]]:
    """Return dict of raw pairwise stability values over C(B,2) pairs."""
    result: Dict[str, List[float]] = {f"jaccard@{k}": [] for k in top_ks}
    result["spearman"] = []
    for i, j in combinations(range(len(rankings)), 2):
        for k in top_ks:
            result[f"jaccard@{k}"].append(_jaccard_at_k(rankings[i], rankings[j], k))
        result["spearman"].append(_spearman(rankings[i], rankings[j]))
    return result


# ---------------------------------------------------------------------------
# Table 3 CSV
# ---------------------------------------------------------------------------

def build_table3(
    pairwise_per_model: Dict[str, Dict[str, List[float]]],
    B: int,
    dataset_name: str,
    model_order: List[str],
    top_ks: Tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    for model in model_order:
        pw = pairwise_per_model[model]
        n_pairs = len(pw["spearman"])
        row: Dict = {
            "model": model,
            "dataset": dataset_name,
            "B": B,
            "n_pairs": n_pairs,
        }
        for k in top_ks:
            arr = np.asarray(pw[f"jaccard@{k}"])
            row[f"jaccard{k}_mean"] = round(float(arr.mean()), 4)
            row[f"jaccard{k}_std"]  = round(float(arr.std(ddof=0)), 4)
        arr = np.asarray(pw["spearman"])
        row["spearman_mean"] = round(float(arr.mean()), 4)
        row["spearman_std"]  = round(float(arr.std(ddof=0)), 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 3 — boxplots
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "LR":       "#2196F3",
    "RF":       "#4CAF50",
    "XGBoost":  "#FF9800",
    "LightGBM": "#9C27B0",
}


def _fig3_boxplots(
    pairwise_per_model: Dict[str, Dict[str, List[float]]],
    model_order: List[str],
    out_path: Path,
    dataset_name: str,
    B: int,
    top_ks: Tuple[int, ...],
    compact: bool = True,
) -> None:
    """Figure 3: boxplots of selected stability metrics — 4 models side-by-side."""
    metrics = (
        [f"jaccard@{k}" for k in top_ks] + ["spearman"]
        if not compact
        else [f"jaccard@{max(top_ks)}", "spearman"]
    )

    n_metrics = len(metrics)
    n_rows = 1
    fig, axes = plt.subplots(n_rows, n_metrics, figsize=(4.5 * n_metrics, 3.6 * n_rows))
    axes_arr = np.asarray(axes).reshape(n_rows, n_metrics)
    width = 0.55

    label_map = {
        "spearman": "Spearman ρ",
        "jaccard@5":  "Jaccard@5",
        "jaccard@10": "Jaccard@10",
    }

    for ax, metric in zip(axes_arr[0], metrics):
        data  = [np.asarray(pairwise_per_model[m][metric]) for m in model_order]
        colors = [MODEL_COLORS.get(m, "grey") for m in model_order]

        bps = ax.boxplot(
            data,
            tick_labels=model_order,
            patch_artist=True,
            showfliers=False,
            widths=width,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, color in zip(bps["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_ylabel(dataset_name, fontsize=11)
        ax.set_title(label_map.get(metric, metric), fontsize=12)
        ax.set_ylim(max(0, min(d.min() for d in data) - 0.05), 1.05)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(axis="x", labelsize=10)

    fig.suptitle(
        f"SHAP Bootstrap Stability (B={B}, C(B,2) pairs)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved figure → {out_path}")


# ---------------------------------------------------------------------------
# Console summary (for quick review)
# ---------------------------------------------------------------------------

def _print_summary(
    table3: pd.DataFrame,
    top_ks: Tuple[int, ...],
) -> None:
    print("\n" + "=" * 72)
    print("TABLE 3 — SHAP Bootstrap Stability  (Home Credit, B=50)")
    print("=" * 72)
    cols_k = []
    for k in top_ks:
        cols_k += [f"jaccard{k}_mean", f"jaccard{k}_std"]
    header = f"  {'Model':<10s}"
    for k in top_ks:
        header += f"  {'Jaccard@'+str(k)+' mean':>14s}  {'±std':>7s}"
    header += f"  {'Spearman mean':>14s}  {'±std':>7s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, row in table3.iterrows():
        line = f"  {row['model']:<10s}"
        for k in top_ks:
            line += f"  {row[f'jaccard{k}_mean']:>14.4f}  {row[f'jaccard{k}_std']:>7.4f}"
        line += f"  {row['spearman_mean']:>14.4f}  {row['spearman_std']:>7.4f}"
        print(line)
    print()

    # LaTeX snippet for copy-paste
    print("LaTeX rows (ready to paste into Table 3):")
    print("-" * 72)
    for _, row in table3.iterrows():
        cells = [row["model"]]
        for k in top_ks:
            cells.append(f"${row[f'jaccard{k}_mean']:.3f} \\pm {row[f'jaccard{k}_std']:.3f}$")
        cells.append(f"${row['spearman_mean']:.3f} \\pm {row['spearman_std']:.3f}$")
        print("  " + " & ".join(cells) + r" \\")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    bootstrap_dir: Path,
    B: int,
    dataset_name: str,
    top_ks: Tuple[int, ...],
    table_out: Path,
    figures_dir: Path,
    model_order: List[str],
) -> None:
    b_dir = bootstrap_dir / f"B{B:03d}"
    if not b_dir.exists():
        raise FileNotFoundError(
            f"Bootstrap directory not found: {b_dir}\n"
            f"Run: ./.venv/bin/python scripts/run_shap_bootstrap_b_sweep.py --b-values {B}"
        )

    print(f"Loading B={B} per-iteration rankings from {b_dir} …")
    pairwise_per_model: Dict[str, Dict[str, List[float]]] = {}
    for model in model_order:
        rankings, feature_names = _load_rankings(b_dir, model, B)
        print(f"  [{model}] loaded {rankings.shape[0]} iterations × {rankings.shape[1]} features")
        pairwise_per_model[model] = compute_pairwise(rankings, top_ks)

    # Table 3
    table3 = build_table3(pairwise_per_model, B, dataset_name, model_order, top_ks)
    table_out.parent.mkdir(parents=True, exist_ok=True)
    table3.to_csv(table_out, index=False)
    print(f"\nTable 3 CSV  → {table_out}")

    # Figure 3 (main): include all Table 3 stability metrics for consistency.
    _fig3_boxplots(
        pairwise_per_model,
        model_order,
        figures_dir / "fig3_stability_boxplots.pdf",
        dataset_name,
        B,
        top_ks,
        compact=False,
    )

    # Supplementary figure (all metrics)
    _fig3_boxplots(
        pairwise_per_model,
        model_order,
        figures_dir / "supplementary" / "fig3_stability_boxplots_full.pdf",
        dataset_name,
        B,
        top_ks,
        compact=False,
    )

    _print_summary(table3, top_ks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day 2 report: produce Table 3 CSV and Figure 3 from bootstrap results."
    )
    parser.add_argument(
        "--bootstrap-dir", default="outputs/shap_bootstrap",
        help="Root directory written by run_shap_bootstrap_b_sweep.py.",
    )
    parser.add_argument("--b", type=int, default=50, help="Bootstrap B to use (default: 50).")
    parser.add_argument(
        "--dataset-name", default="Home Credit",
        help="Dataset label for figure titles.",
    )
    parser.add_argument(
        "--top-ks", nargs="+", type=int, default=[5, 10],
        help="Top-k values for Jaccard (default: 5 10).",
    )
    parser.add_argument(
        "--table-out",
        default="outputs/tables/table3_stability_homecredit.csv",
    )
    parser.add_argument(
        "--figures-dir", default="outputs/figures",
        help="Root figures directory.",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["LR", "RF", "XGBoost", "LightGBM"],
    )
    args = parser.parse_args()

    run(
        bootstrap_dir=Path(args.bootstrap_dir),
        B=args.b,
        dataset_name=args.dataset_name,
        top_ks=tuple(args.top_ks),
        table_out=Path(args.table_out),
        figures_dir=Path(args.figures_dir),
        model_order=args.models,
    )


if __name__ == "__main__":
    main()
