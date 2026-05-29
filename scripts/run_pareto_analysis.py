"""
Day 3 — Multi-Objective Pareto Analysis
----------------------------------------
Computes normalized W scores and Pareto frontier for all 4 models on Home Credit.

Inputs (pre-computed):
  outputs/homecredit_4model_results.json    → ROC-AUC
  outputs/cost_sensitive_homecredit.json    → EC(t*)
  outputs/tables/table3_stability_homecredit.csv → S̄_SHAP (Spearman ρ mean)

Outputs:
  outputs/tables/table4_pareto_homecredit.csv   — Table 4 (paper)
  outputs/figures/fig4_pareto_homecredit.pdf    — Figure 4 (paper)
  outputs/figures/fig4_pareto_homecredit.png
  outputs/pareto_summary.json                   — machine-readable summary

W score definition:
  W = ⅓ (norm_AUC  −  norm_EC  +  norm_Stability)

  where each metric is min-max normalised across the 4 models:
    norm_AUC   in [0,1]  (higher is better, so +)
    norm_EC    in [0,1]  (higher means worse cost, so −)
    norm_Stab  in [0,1]  (higher is better, so +)
  → W ∈ [−1/3, +2/3]; a model with max-AUC, min-EC, max-Stab scores W = 2/3.

Pareto screening:
  Main Pareto status is computed in the 3-D objective space:
    maximise ROC-AUC, minimise EC(t*), maximise S̄_SHAP.
  The plotted frontier is a 2-D EC-stability projection with ROC-AUC shown by
  marker intensity.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_4MODEL = ROOT / "outputs" / "homecredit_4model_results.json"
RESULTS_COST_SENSITIVE = ROOT / "outputs" / "cost_sensitive_homecredit.json"
TABLE3_CSV = ROOT / "outputs" / "tables" / "table3_stability_homecredit.csv"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"
OUT_JSON = ROOT / "outputs" / "pareto_summary.json"
TABLE4_CSV = TABLES_DIR / "table4_pareto_homecredit.csv"
FIG4_PDF = FIGURES_DIR / "fig4_pareto_homecredit.pdf"
FIG4_PNG = FIGURES_DIR / "fig4_pareto_homecredit.png"

MODELS = ["LR", "RF", "XGBoost", "LightGBM"]
COLORS = {"LR": "#4C72B0", "RF": "#55A868",
          "XGBoost": "#C44E52", "LightGBM": "#DD8452"}
MARKERS = {"LR": "o", "RF": "s", "XGBoost": "^", "LightGBM": "D"}

# Institutional weight profiles: (w1_AUC, w2_EC, w3_Stability)
# W(w) = w1*norm_AUC - w2*norm_EC + w3*norm_Stability
WEIGHT_PROFILES: dict[str, tuple[float, float, float]] = {
    "neutral_equal":  (1/3, 1/3, 1/3),      # neutral baseline
    # conservative lender, prioritise EC reduction
    "cost_averse":    (0.20, 0.60, 0.20),
    # audit/governance focus, prioritise stability
    "audit_centric":  (0.10, 0.10, 0.80),
}


def display_path(path: pathlib.Path) -> str:
    """Return a stable repo-relative path for logs when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


# ── helpers ────────────────────────────────────────────────────────────────────

def minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def pareto_mask(ec: np.ndarray, stab: np.ndarray) -> np.ndarray:
    """Return boolean array — True when model i is Pareto-optimal.
    Objectives: minimise EC, maximise Stability.
    Model i is dominated iff ∃ j such that ec[j] ≤ ec[i] AND stab[j] ≥ stab[i]
    with at least one strict inequality.
    """
    n = len(ec)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if ec[j] <= ec[i] and stab[j] >= stab[i]:
                if ec[j] < ec[i] or stab[j] > stab[i]:
                    dominated[i] = True
                    break
    return ~dominated


# ── NSGA-II helpers ────────────────────────────────────────────────────────────

def _dominates(obj_i: np.ndarray, obj_j: np.ndarray, maximize: np.ndarray) -> bool:
    """Return True if solution i dominates solution j.

    Parameters
    ----------
    obj_i, obj_j : 1-D arrays of raw objective values.
    maximize     : boolean mask — True means the objective should be maximised.
    """
    # Convert everything to a "higher is better" space.
    i_better = np.where(maximize, obj_i, -obj_i)
    j_better = np.where(maximize, obj_j, -obj_j)
    return bool(np.all(i_better >= j_better) and np.any(i_better > j_better))


def nsga2_nondominated_sort(objectives: np.ndarray, maximize: np.ndarray) -> list[list[int]]:
    """Fast non-dominated sort (NSGA-II, Deb et al. 2002).

    Parameters
    ----------
    objectives : shape (n, m) — n solutions, m objectives.
    maximize   : boolean array of length m.

    Returns
    -------
    fronts : list of lists of indices, fronts[0] = Pareto-optimal set (rank 1).
    """
    n = objectives.shape[0]
    domination_count = np.zeros(n, dtype=int)   # how many solutions dominate i
    dominated_set: list[list[int]] = [[]
                                      # solutions i dominates
                                      for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(objectives[i], objectives[j], maximize):
                dominated_set[i].append(j)
            elif _dominates(objectives[j], objectives[i], maximize):
                domination_count[i] += 1

    fronts: list[list[int]] = []
    current_front = [i for i in range(n) if domination_count[i] == 0]
    while current_front:
        fronts.append(current_front)
        next_front: list[int] = []
        for i in current_front:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front = next_front
    return fronts


def nsga2_crowding_distance(front_objectives: np.ndarray) -> np.ndarray:
    """Compute crowding distance for solutions in a single front.

    Boundary points receive distance = inf; interior points accumulate
    normalised spans across all objectives.

    Parameters
    ----------
    front_objectives : shape (k, m) — k solutions in the front, m objectives.

    Returns
    -------
    distances : 1-D array of length k.
    """
    k, m = front_objectives.shape
    distances = np.zeros(k)
    if k <= 2:
        distances[:] = np.inf
        return distances

    for obj_idx in range(m):
        order = np.argsort(front_objectives[:, obj_idx])
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        f_min = front_objectives[order[0],  obj_idx]
        f_max = front_objectives[order[-1], obj_idx]
        span = f_max - f_min
        if span == 0:
            continue
        for pos in range(1, k - 1):
            distances[order[pos]] += (
                front_objectives[order[pos + 1], obj_idx]
                - front_objectives[order[pos - 1], obj_idx]
            ) / span
    return distances


# ── load data ──────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    # 1. ROC-AUC
    with open(RESULTS_4MODEL) as f:
        r4m = json.load(f)
    auc = {m: r4m["table8_best_metrics"][m]["roc_auc"] for m in MODELS}

    # 2. EC(t*)  — "cost_optimal" threshold row
    with open(RESULTS_COST_SENSITIVE) as f:
        rw5 = json.load(f)
    ec_opt = {}
    for row in rw5["table9_threshold_comparison"]:
        if row["threshold_type"] == "cost_optimal":
            ec_opt[row["model"]] = {
                "ec": row["expected_cost"],
                "threshold": row["threshold"],
                "savings_pct": row["cost_savings_vs_0_5_pct"],
            }

    # 3. Stability — Spearman ρ mean from Table 3
    t3 = pd.read_csv(TABLE3_CSV)
    stab = dict(zip(t3["model"], t3["spearman_mean"]))

    rows = []
    for m in MODELS:
        rows.append({
            "model": m,
            "roc_auc": auc[m],
            "ec_optimal": ec_opt[m]["ec"],
            "threshold_star": ec_opt[m]["threshold"],
            "savings_pct": ec_opt[m]["savings_pct"],
            "spearman_mean": stab[m],
        })
    return pd.DataFrame(rows)


# ── scoring ────────────────────────────────────────────────────────────────────

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["norm_auc"] = minmax(df["roc_auc"].values)
    df["norm_ec"] = minmax(df["ec_optimal"].values)      # higher = worse
    df["norm_stab"] = minmax(df["spearman_mean"].values)

    df["W"] = (df["norm_auc"] - df["norm_ec"] + df["norm_stab"]) / 3.0

    # Pareto frontier in the 2D EC-stability projection, used only to draw the
    # projected frontier line. The paper's main Pareto status is the 3-objective
    # NSGA-II front below.
    is_projection_pareto = pareto_mask(
        df["ec_optimal"].values,
        df["spearman_mean"].values,
    )
    df["pareto_projection_ec_stability"] = is_projection_pareto

    df["W_rank"] = df["W"].rank(ascending=False).astype(int)

    # ── NSGA-II: non-dominated sorting + crowding distance ────────────────────
    # Three objectives: AUC (↑), EC (↓), Stability (↑)
    objectives = df[["roc_auc", "ec_optimal", "spearman_mean"]].values
    maximize = np.array([True, False, True])

    fronts = nsga2_nondominated_sort(objectives, maximize)

    nsga2_rank = np.zeros(len(df), dtype=int)
    crowding = np.zeros(len(df))
    for rank_idx, front in enumerate(fronts, start=1):
        front_arr = np.array(front)
        front_objs = objectives[front_arr]
        cd = nsga2_crowding_distance(front_objs)
        for local_pos, global_idx in enumerate(front_arr):
            nsga2_rank[global_idx] = rank_idx
            crowding[global_idx] = cd[local_pos]

    df["nsga2_rank"] = nsga2_rank
    df["crowding_distance"] = crowding
    df["pareto_optimal"] = df["nsga2_rank"] == 1

    return df


# ── weight sensitivity ─────────────────────────────────────────────────────────

def w_score(norm_auc: float, norm_ec: float, norm_stab: float,
            w1: float, w2: float, w3: float) -> float:
    """W(w) = w1·norm_AUC − w2·norm_EC + w3·norm_Stability."""
    return w1 * norm_auc - w2 * norm_ec + w3 * norm_stab


def weight_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute W scores under each institutional weight profile.

    Returns a long-form DataFrame with columns:
        profile, weights, model, W, selected
    """
    rows = []
    for profile, (w1, w2, w3) in WEIGHT_PROFILES.items():
        scores = [
            w_score(r["norm_auc"], r["norm_ec"], r["norm_stab"], w1, w2, w3)
            for _, r in df.iterrows()
        ]
        winner_idx = int(np.argmax(scores))
        winner = df.iloc[winner_idx]["model"]
        for idx, (_, row) in enumerate(df.iterrows()):
            rows.append({
                "profile":  profile,
                "weights":  f"({w1:.2f},{w2:.2f},{w3:.2f})",
                "model":    row["model"],
                "W":        round(scores[idx], 4),
                "selected": row["model"] == winner,
            })
    return pd.DataFrame(rows)


# ── figure 4 ───────────────────────────────────────────────────────────────────

def plot_pareto(df: pd.DataFrame, out_pdf: pathlib.Path, out_png: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.5), constrained_layout=True)

    ec = df["ec_optimal"].values
    stab = df["spearman_mean"].values
    auc = df["roc_auc"].values

    # color scale mapped to AUC
    norm_auc_vals = (auc - auc.min()) / (auc.max() - auc.min() + 1e-12)
    cmap = plt.cm.RdYlGn
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
        vmin=auc.min() - 0.005, vmax=auc.max() + 0.005))
    sm.set_array([])

    # ── scatter points ─────────────────────────────────────────────────────────
    for _, row in df.iterrows():
        m = row["model"]
        color = cmap(norm_auc_vals[df.index[df["model"] == m][0]])
        ax.scatter(
            row["ec_optimal"],
            row["spearman_mean"],
            s=220,
            color=color,
            marker=MARKERS[m],
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )

    # ── Pareto frontier line (2D projection: EC vs Stability) ─────────────────
    # XGBoost is still on NSGA-II F1 in 3D due to its AUC advantage.
    pareto_df = df[df["pareto_projection_ec_stability"]].sort_values("ec_optimal")
    if len(pareto_df) >= 2:
        # staircase / step-function frontier
        px = pareto_df["ec_optimal"].values
        py = pareto_df["spearman_mean"].values
        # draw step lines connecting Pareto points
        step_x, step_y = [px[0]], [py[0]]
        for k in range(1, len(px)):
            step_x += [px[k], px[k]]
            step_y += [py[k - 1], py[k]]
        step_x.append(px[-1])
        step_y.append(py[-1])
        ax.plot(step_x, step_y, "k--", lw=1.6, alpha=0.6,
                zorder=4, label="Pareto frontier")
    elif len(pareto_df) == 1:
        ax.plot(
            pareto_df["ec_optimal"].values,
            pareto_df["spearman_mean"].values,
            "k*", ms=18, zorder=6, label="Pareto optimal",
        )

    x_span = float(ec.max() - ec.min())
    y_span = float(stab.max() - stab.min())
    x_pad = max(x_span * 0.14, 0.0015)
    y_pad = max(y_span * 0.14, 0.0070)
    ax.set_xlim(float(ec.min() - x_pad), float(ec.max() + x_pad))
    ax.set_ylim(float(stab.min() - y_pad), float(stab.max() + y_pad))

    # ── model labels ──────────────────────────────────────────────────────────
    label_styles = {
        "LR":       {"offset": (-36,  14), "ha": "right",  "va": "bottom"},
        "RF":       {"offset": (0, -24), "ha": "center", "va": "top"},
        "XGBoost":  {"offset": (12,  -2), "ha": "left",   "va": "center"},
        "LightGBM": {"offset": (14, -18), "ha": "left",   "va": "top"},
    }
    for _, row in df.iterrows():
        m = row["model"]
        style = label_styles.get(
            m, {"offset": (10, 10), "ha": "left", "va": "bottom"})
        # Use nsga2_rank for ★: 3D NSGA-II F1 includes XGBoost (AUC advantage)
        star = " ★" if row["nsga2_rank"] == 1 else ""
        label = ax.annotate(
            f"{m}{star}\nW={row['W']:.3f}",
            xy=(row["ec_optimal"], row["spearman_mean"]),
            xytext=style["offset"],
            textcoords="offset points",
            fontsize=9,
            ha=style["ha"],
            va=style["va"],
            clip_on=False,
            zorder=7,
        )
        label.set_path_effects([
            path_effects.withStroke(linewidth=3, foreground="white"),
            path_effects.Normal(),
        ])

    # ── arrows indicating "better" direction ──────────────────────────────────
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    guide_x = x_min + (x_max - x_min) * 0.14
    guide_y = y_min + (y_max - y_min) * 0.07
    ax.annotate(
        "", xy=(guide_x, guide_y),
        xytext=(guide_x + (x_max - x_min) * 0.42, guide_y),
        arrowprops=dict(arrowstyle="<-", color="grey", lw=1.4),
    )
    ax.text(
        guide_x + (x_max - x_min) * 0.008,
        guide_y + (y_max - y_min) * 0.012,
        "lower EC",
        fontsize=7.5,
        color="grey",
        ha="left",
    )

    ax.annotate(
        "", xy=(guide_x, guide_y + (y_max - y_min) * 0.09),
        xytext=(guide_x, guide_y),
        arrowprops=dict(arrowstyle="->", color="grey", lw=1.4),
    )
    ax.text(
        guide_x + (x_max - x_min) * 0.075,
        guide_y + (y_max - y_min) * 0.035,
        "higher\nstability",
        fontsize=7.5,
        color="grey",
        va="bottom",
    )

    # ── colorbar for ROC-AUC ──────────────────────────────────────────────────
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("ROC-AUC", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # ── legend for Pareto frontier ─────────────────────────────────────────────
    legend_elements = [
        plt.Line2D([0], [0], linestyle="--", color="black",
                   lw=1.6, alpha=0.7, label="Pareto frontier"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # ── custom marker legend ───────────────────────────────────────────────────
    for m, mk in MARKERS.items():
        ax.scatter([], [], marker=mk, color="grey", s=70, label=m)
    ax.legend(
        handles=[
            mpatches.Patch(color="white", label="Models:"),
            *[
                plt.Line2D(
                    [0], [0], marker=MARKERS[m], color="w",
                    markerfacecolor="grey", markersize=8, label=m,
                )
                for m in MODELS
            ],
            plt.Line2D([0], [0], linestyle="--", color="black",
                       lw=1.5, alpha=0.7, label="Pareto frontier"),
        ],
        fontsize=8.5,
        loc="lower right",
        framealpha=0.9,
    )

    ax.set_xlabel(
        r"Expected Cost at $t^*$  $\left[EC(t^*)\right]$", fontsize=11)
    ax.set_ylabel(
        r"SHAP Stability  $\left(\bar{S}_{\mathrm{SHAP}},\;\mathrm{Spearman}\;\rho\right)$", fontsize=11)
    ax.tick_params(labelsize=9)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figure 4 saved → {out_pdf.name}  |  {out_png.name}")


# ── main ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Day 3 — Pareto analysis: W scores + Figure 4")
    parser.add_argument("--output-dir",   default=str(TABLES_DIR),
                        help="Directory for CSV outputs")
    parser.add_argument("--figures-dir",  default=str(FIGURES_DIR),
                        help="Directory for figure outputs")
    parser.add_argument("--no-figures",   action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args(argv)

    tables_dir = pathlib.Path(args.output_dir)
    figures_dir = pathlib.Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    df = load_data()
    print("\n── Raw metrics ───────────────────────────────────────────")
    print(df[["model", "roc_auc", "ec_optimal", "threshold_star",
          "spearman_mean"]].to_string(index=False))

    print("\nComputing normalised W scores and Pareto frontier …")
    df = compute_scores(df)

    print("\n── W Scores, NSGA-II Pareto Status & Ranks ───────────────")
    display_cols = ["model", "roc_auc", "ec_optimal", "spearman_mean",
                    "norm_auc", "norm_ec", "norm_stab", "W", "W_rank",
                    "pareto_optimal", "pareto_projection_ec_stability",
                    "nsga2_rank", "crowding_distance"]
    print(df[display_cols].to_string(index=False))

    # ── Table 4 CSV ───────────────────────────────────────────────────────────
    table4 = df[[
        "model", "roc_auc", "ec_optimal", "threshold_star",
        "spearman_mean", "W", "W_rank", "pareto_optimal",
        "nsga2_rank", "crowding_distance",
    ]].rename(columns={
        "model":              "Model",
        "roc_auc":            "ROC-AUC",
        "ec_optimal":         "EC(t*)",
        "threshold_star":     "t*",
        "spearman_mean":      "S\u0304_SHAP (Spearman \u03c1)",
        "W":                  "W",
        "W_rank":             "W rank",
        "pareto_optimal":     "3-objective Pareto-optimal",
        "nsga2_rank":         "NSGA-II Front",
        "crowding_distance":  "Crowding Distance",
    })
    t4_path = tables_dir / "table4_pareto_homecredit.csv"
    table4.to_csv(t4_path, index=False, float_format="%.4f")
    print(f"\n[OK] Table 4 saved → {display_path(t4_path)}")

    # ── machine-readable summary ──────────────────────────────────────────────
    summary = {
        "dataset": "Home Credit",
        "n_models": len(MODELS),
        "w_score_formula": "W = (1/3) * (norm_AUC - norm_EC + norm_Stability)",
        "stability_metric": "Spearman rho mean (B=50, 1225 pairs)",
        "models": {},
        "pareto_optimal_models": df.loc[df["pareto_optimal"], "model"].tolist(),
        "best_W_model": df.loc[df["W"].idxmax(), "model"],
    }
    for _, row in df.iterrows():
        m = row["model"]
        cd_val = row["crowding_distance"]
        summary["models"][m] = {
            "roc_auc": round(row["roc_auc"], 6),
            "ec_optimal": round(row["ec_optimal"], 6),
            "threshold_star": round(row["threshold_star"], 4),
            "spearman_mean": round(row["spearman_mean"], 6),
            "norm_auc": round(row["norm_auc"], 4),
            "norm_ec": round(row["norm_ec"], 4),
            "norm_stab": round(row["norm_stab"], 4),
            "W": round(row["W"], 4),
            "W_rank": int(row["W_rank"]),
            "pareto_optimal": bool(row["pareto_optimal"]),
            "nsga2_front": int(row["nsga2_rank"]),
            "crowding_distance": "inf" if np.isinf(cd_val) else round(cd_val, 4),
        }
    out_json = ROOT / "outputs" / "pareto_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Summary JSON saved → {display_path(out_json)}")

    # ── Figure 4 ──────────────────────────────────────────────────────────────
    if not args.no_figures:
        fig4_pdf = figures_dir / "fig4_pareto_homecredit.pdf"
        fig4_png = figures_dir / "fig4_pareto_homecredit.png"
        print("\nGenerating Figure 4 …")
        plot_pareto(df, fig4_pdf, fig4_png)

    # ── print concise summary ─────────────────────────────────────────────────
    print("\n── Summary ───────────────────────────────────────────────")
    print(f"  Best W score      : {summary['best_W_model']} "
          f"(W = {summary['models'][summary['best_W_model']]['W']:.4f})")
    print(
        f"  Pareto-optimal    : {', '.join(summary['pareto_optimal_models'])}")
    print("\n── NSGA-II Fronts ────────────────────────────────────────")
    for _, row in df.sort_values("nsga2_rank").iterrows():
        cd = row["crowding_distance"]
        cd_str = "\u221e" if np.isinf(cd) else f"{cd:.4f}"
        print(f"  F{int(row['nsga2_rank'])}  {row['model']:<10}  CD={cd_str}")

    # ── Weight sensitivity: institutional selection rule ─────────────────────
    print("\n── W(w) Institutional Selection Rule ─────────────────────")
    sens = weight_sensitivity(df)
    for profile, group in sens.groupby("profile", sort=False):
        w_str = group["weights"].iloc[0]
        winner = group.loc[group["selected"], "model"].values[0]
        scores_str = "  ".join(
            f"{r['model']}: {r['W']:+.4f}{'*' if r['selected'] else ''}"
            for _, r in group.iterrows()
        )
        print(f"  [{profile}] w={w_str}  \u2192 {winner}")
        print(f"    {scores_str}")
    print("\nDay 3 — DONE ✓")


if __name__ == "__main__":
    main()
