#!/bin/bash
# Ablation: F-Wanda calibration size sweep (LLaMA-2-7B, 50% unstructured).
# n_samples in {32, 64, 128, 256, 512}. Submits 5 SLURM jobs.

set -euo pipefail
cd "$(dirname "$0")/../.."

DRY="${DRY:-0}"
submit() { if [ "$DRY" = "1" ]; then echo "[dry] $*"; else "$@"; fi; }

for n in 32 64 128 256 512; do
    tag="abl_calib_n${n}"
    submit sbatch \
        --job-name="$tag" \
        --output="results/logs/${tag}_%j.out" \
        --error="results/logs/${tag}_%j.err" \
        --time=6:00:00 \
        --export=ALL,FW_MODEL=meta-llama/Llama-2-7b-hf,FW_METHOD=fwanda,FW_SPARSITY=0.5,FW_PATTERN=unstructured,FW_NSAMPLES=$n,FW_TAG=$tag,FW_RESULTS_CSV=results/tables/ablation_calib_size.csv,FW_EVAL=full \
        scripts/run_one.sh
done
