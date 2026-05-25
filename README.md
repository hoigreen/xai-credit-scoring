# Reproducibility Guide for XAI-Credit Scoring Project

This document describes how to regenerate the results from the public Home Credit Default Risk dataset.

## Scope

The reproducibility package covers:

- Home Credit preprocessing.
- Four-model benchmark: Logistic Regression, Random Forest, XGBoost, and LightGBM.
- Cost-sensitive threshold optimization with `C_FN:C_FP = 5:1`.
- Global SHAP rankings.
- Bootstrap SHAP stability with `B in {30, 50, 80, 100}` and the primary paper setting `B = 50`.
- Pareto and weighted-score analysis.

## Environment

Use Python 3 and the pinned package list:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python --version
```

The original runs used seed `42`. The scripts store seed and split metadata in their JSON outputs.

## Data

The dataset is the public Kaggle Home Credit Default Risk competition.

Before downloading:

1. Install the dependencies from `requirements.txt`.
2. Put the Kaggle token at `~/.kaggle/kaggle.json`.
3. Accept the competition rules on Kaggle.

Download and validate:

```bash
./.venv/bin/python scripts/setup_homecredit_data.py --data-dir data/homecredit
```

## Adapting to Internal Bank Data

This experiment is intentionally scoped to public Home Credit dataset. To run the same protocol on private bank data, add a separate preprocessing adapter rather than editing the benchmark scripts in place. The adapter should:

1. Construct a binary `TARGET` label from the institution's approved default definition and observation window.
2. Exclude post-outcome, collection, recovery, manually overridden, and other leakage-prone fields.
3. Preserve a deterministic temporal ordering column named `sort_key`; use the application or scoring timestamp when available, otherwise document any surrogate ordering.
4. Apply institution-approved imputation, categorical encoding, and optional feature filters, then write a modeling-ready CSV with one row per scored application.
5. Run the same downstream commands by setting `XAI_PREPROCESSED_DATA` or each script's `--input` argument to the internal preprocessed CSV.

Before comparing internal results with the paper, re-check class prevalence, train/validation/test default rates, cost-ratio assumptions, feature availability at scoring time, and any regulatory restrictions on storing model outputs or SHAP values. Do not commit proprietary raw data, derived customer-level files, or confidential result tables to the public repository.

## Full Pipeline

The recommended one-command reproduction path is:

```bash
bash scripts/run_pipeline.sh
```

This command can take several hours on CPU because the bootstrap stability sweep refits four models across multiple `B` values.

If the bootstrap rankings already exist under `outputs/shap_bootstrap/`, reuse them and regenerate only the downstream reports:

```bash
XAI_SKIP_BOOTSTRAP=1 bash scripts/run_pipeline.sh
```

## Manual Commands

Run these from the repo root:

```bash
./.venv/bin/python scripts/preprocess_homecredit.py \
  --input data/homecredit/application_train.csv \
  --output outputs/homecredit_preprocessed.csv

./.venv/bin/python scripts/run_homecredit_4model.py \
  --input outputs/homecredit_preprocessed.csv \
  --output-json outputs/homecredit_4model_results.json \
  --figures-dir outputs/figures \
  --seed 42

./.venv/bin/python scripts/run_cost_sensitive.py \
  --input outputs/homecredit_preprocessed.csv \
  --output-json outputs/cost_sensitive_homecredit.json \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --seed 42

./.venv/bin/python scripts/run_shap_analysis.py \
  --input outputs/homecredit_preprocessed.csv \
  --output-dir outputs/shap \
  --figures-dir outputs/figures \
  --seed 42

./.venv/bin/python scripts/run_shap_bootstrap_b_sweep.py \
  --input outputs/homecredit_preprocessed.csv \
  --output-dir outputs/shap_bootstrap \
  --figures-dir outputs/figures \
  --seed 42

./.venv/bin/python scripts/produce_stability_report.py \
  --bootstrap-dir outputs/shap_bootstrap \
  --b 50 \
  --table-out outputs/tables/table3_stability_homecredit.csv \
  --figures-dir outputs/figures

./.venv/bin/python scripts/run_pareto_analysis.py \
  --output-dir outputs/tables \
  --figures-dir outputs/figures
```

## Expected Primary Outputs

The values are expected to match these machine-readable outputs:

- `outputs/homecredit_4model_results.json`
- `outputs/cost_sensitive_homecredit.json`
- `outputs/tables/table3_stability_homecredit.csv`
- `outputs/tables/table4_pareto_homecredit.csv`
- `outputs/pareto_summary.json`

## Known Reproducibility Notes

- The Home Credit dataset has no calendar timestamp in `application_train.csv`; the source uses `SK_ID_CURR` as a deterministic surrogate ordering for the 60/20/20 split.
- Generated experiment artifacts are excluded from git via `outputs/`; regenerate them with `scripts/run_pipeline.sh` when auditing the manuscript values.
- Small numerical differences may occur across CPU, BLAS/OpenMP, XGBoost, and LightGBM runtime backends, but seed, split, and package versions are fixed.
