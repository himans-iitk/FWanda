#!/bin/bash
# Ablation: which Fisher matters?  (LLaMA-2-7B, 50% unstructured)
#   empirical    — default; E[g^2] from CE loss against true labels
#   mean_only    — (E[g])^2  (decouples scale from per-token variation)
#   true         — labels sampled from the model's predictive distribution

set -euo pipefail
cd "$(dirname "$0")/../.."

DRY="${DRY:-0}"
submit() { if [ "$DRY" = "1" ]; then echo "[dry] $*"; else "$@"; fi; }

for variant in empirical mean_only true; do
    tag="abl_fisher_${variant}"
    submit sbatch \
        --job-name="$tag" \
        --output="results/logs/${tag}_%j.out" \
        --error="results/logs/${tag}_%j.err" \
        --time=6:00:00 \
        --export=ALL,FW_MODEL=meta-llama/Llama-2-7b-hf,FW_METHOD=fwanda,FW_SPARSITY=0.5,FW_PATTERN=unstructured,FW_FISHER_VARIANT=$variant,FW_TAG=$tag,FW_RESULTS_CSV=results/tables/ablation_fisher_variants.csv,FW_EVAL=full \
        scripts/run_one.sh
done
