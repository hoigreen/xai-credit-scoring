#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
SEED="${XAI_SEED:-${SIMC_SEED:-42}}"
RAW_DATA="${XAI_RAW_DATA:-${SIMC_RAW_DATA:-data/homecredit/application_train.csv}}"
PREPROCESSED_DATA="${XAI_PREPROCESSED_DATA:-${SIMC_PREPROCESSED_DATA:-outputs/homecredit_preprocessed.csv}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN"
  echo "Create the environment first: python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt"
  exit 1
fi

if [[ ! -f "$RAW_DATA" ]]; then
  if [[ "${XAI_SKIP_DOWNLOAD:-${SIMC_SKIP_DOWNLOAD:-0}}" == "1" ]]; then
    echo "Missing raw data: $RAW_DATA"
    echo "Unset XAI_SKIP_DOWNLOAD or place application_train.csv at the expected path."
    exit 1
  fi
  "$PYTHON_BIN" scripts/setup_homecredit_data.py --data-dir data/homecredit
fi

echo "Preprocessing Home Credit data..."
"$PYTHON_BIN" scripts/preprocess_homecredit.py \
  --input "$RAW_DATA" \
  --output "$PREPROCESSED_DATA"

echo "Running four-model benchmark..."
"$PYTHON_BIN" scripts/run_homecredit_4model.py \
  --input "$PREPROCESSED_DATA" \
  --output-json outputs/homecredit_4model_results.json \
  --figures-dir outputs/figures \
  --seed "$SEED"

echo "Running cost-sensitive threshold analysis..."
"$PYTHON_BIN" scripts/run_cost_sensitive.py \
  --input "$PREPROCESSED_DATA" \
  --output-json outputs/cost_sensitive_homecredit.json \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --paper-figures-dir paper/simc/figures \
  --seed "$SEED"

echo "Running global SHAP analysis..."
"$PYTHON_BIN" scripts/run_shap_analysis.py \
  --input "$PREPROCESSED_DATA" \
  --output-dir outputs/shap \
  --figures-dir outputs/figures/supplementary \
  --seed "$SEED"

if [[ "${XAI_SKIP_BOOTSTRAP:-${SIMC_SKIP_BOOTSTRAP:-0}}" == "1" ]]; then
  echo "Skipping bootstrap sweep because XAI_SKIP_BOOTSTRAP=1."
else
  echo "Running SHAP bootstrap B-sweep..."
  "$PYTHON_BIN" scripts/run_shap_bootstrap_b_sweep.py \
    --input "$PREPROCESSED_DATA" \
    --output-dir outputs/shap_bootstrap \
    --figures-dir outputs/figures/supplementary \
    --seed "$SEED"
fi

echo "Producing stability report..."
"$PYTHON_BIN" scripts/produce_stability_report.py \
  --bootstrap-dir outputs/shap_bootstrap \
  --b 50 \
  --table-out outputs/tables/table3_stability_homecredit.csv \
  --figures-dir outputs/figures

echo "Running Pareto analysis..."
"$PYTHON_BIN" scripts/run_pareto_analysis.py \
  --output-dir outputs/tables \
  --figures-dir outputs/figures

echo "Synchronizing paper figures..."
mkdir -p paper/simc/figures
cp outputs/figures/fig6a_roc_curves_homecredit.pdf paper/simc/figures/fig1_roc_curves_homecredit.pdf || true
cp outputs/figures/fig6b_pr_curves_homecredit.pdf paper/simc/figures/fig2_pr_curves_homecredit.pdf || true
cp outputs/figures/fig7_ec_curves_homecredit.pdf paper/simc/figures/fig3_ec_curves_homecredit.pdf || true
cp outputs/figures/fig8_cost_sensitivity_homecredit.pdf paper/simc/figures/fig4_cost_sensitivity_homecredit.pdf || true
cp outputs/figures/fig3_stability_boxplots.pdf paper/simc/figures/fig5_stability_boxplots.pdf || true
cp outputs/figures/fig4_pareto_homecredit.pdf paper/simc/figures/fig6_pareto_homecredit.pdf || true

echo "Reproducibility pipeline completed."

