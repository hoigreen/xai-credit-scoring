# Reproducibility Artifacts

This directory contains lightweight, paper-level outputs from the public Home
Credit experiment. It is intentionally separate from `outputs/`, which is
ignored because it may contain large regenerated files and local caches.

Primary artifacts:

- `results/homecredit_4model_results.json` — predictive benchmark summary.
- `results/cost_sensitive_homecredit.json` — expected-cost threshold analysis.
- `results/pareto_summary.json` — three-objective NSGA-II-style Pareto summary.
- `tables/table3_stability_homecredit.csv` — B=50 SHAP stability table.
- `tables/table4_pareto_homecredit.csv` — three-objective model-selection table.
- `figures/*.pdf` — paper-level generated figures.
- `shap_bootstrap/stability_b_sweep.*` and `shap_bootstrap/optimal_B.json` —
  aggregate bootstrap-length sensitivity summaries.

Large local artifacts are intentionally excluded from this tracked snapshot:

- raw Kaggle data
- preprocessed row-level data
- model-level SHAP ranking CSVs
- B=50 per-iteration bootstrap ranking caches

The full pipeline can regenerate these files from the public Kaggle dataset:

```bash
bash scripts/run_pipeline.sh
```
