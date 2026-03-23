#!/usr/bin/env bash
# eval_all.sh — Run privacy + diversity + meaning evaluation for all
# model outputs across all datasets.
#
# Each model's output must be a pickle file at:
#   results/<model>_<dataset>_syn.p
#
# Usage:
#   bash eval_all.sh                          # all models × all datasets
#   DATASETS="yelp" bash eval_all.sh          # single dataset
#   MODELS="ours dpmlm" bash eval_all.sh      # subset of models

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
DATASETS="${DATASETS:-yelp tweet imdb}"
MODELS="${MODELS:-ours dpmlm presidio dipper stylemix tarot}"
RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_DIR="${LOG_DIR:-eval_logs}"

mkdir -p "$LOG_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────
run_eval() {
    local model="$1"
    local dataset="$2"
    local pkl="${RESULTS_DIR}/${model}_${dataset}_syn.p"

    if [ ! -f "$pkl" ]; then
        echo "  [SKIP] ${pkl} not found"
        return
    fi

    local log="${LOG_DIR}/${model}_${dataset}.log"
    echo "  Running evals → ${log}"

    {
        echo "=== ${model} / ${dataset} ==="
        echo "Pickle: ${pkl}"
        echo ""

        echo "--- privacy_eval.py ---"
        python3 privacy_eval.py "$pkl"

        echo ""
        echo "--- diversity_eval.py ---"
        python3 diversity_eval.py "$pkl"

        echo ""
        echo "--- meaning_eval.py ---"
        python3 meaning_eval.py "$pkl"

    } 2>&1 | tee "$log"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo " Evaluation sweep"
echo " Models:   ${MODELS}"
echo " Datasets: ${DATASETS}"
echo " Results:  ${RESULTS_DIR}/"
echo " Logs:     ${LOG_DIR}/"
echo "============================================================"

for dataset in $DATASETS; do
    echo ""
    echo "────────────────────────────────────────"
    echo " Dataset: ${dataset}"
    echo "────────────────────────────────────────"
    for model in $MODELS; do
        echo " Model: ${model}"
        run_eval "$model" "$dataset"
    done
done

echo ""
echo "============================================================"
echo " All done. Per-run logs saved in ${LOG_DIR}/"
echo "============================================================"
