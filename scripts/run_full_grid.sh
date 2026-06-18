#!/bin/bash
# Submit the full experimental grid (Section 7.2) as N independent SLURM
# jobs — one per cell — so they can preempt/restart independently.
#
# Grid: 2 models x 4 methods x 2 sparsity configs = 16 jobs.
# Each job runs PPL + 7 zero-shot + MMLU and appends to main_results.csv.
#
# Run from the repo root:
#   bash scripts/run_full_grid.sh
#
# Pass DRY=1 to print the sbatch commands without submitting them.

set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=(
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-13b-hf"
)
METHODS=(magnitude wanda sparsegpt fwanda)
SPARSITIES=(
    "0.5 unstructured"
    "0.5 2:4"
)

DRY="${DRY:-0}"
submit() {
    if [ "$DRY" = "1" ]; then
        echo "[dry] $*"
    else
        "$@"
    fi
}

for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
        for sp in "${SPARSITIES[@]}"; do
            read -r sparsity pattern <<<"$sp"
            tag="$(echo "${model##*/}_${method}_${sparsity}_${pattern}" \
                | tr ':' '-' | tr '/' '_')"
            time_hours=4
            [[ "$model" == *13b* ]] && time_hours=12
            [[ "$method" == "sparsegpt" || "$method" == "fsparsegpt" ]] \
                && time_hours=$((time_hours + 4))
            submit sbatch \
                --time=${time_hours}:00:00 \
                --job-name="fw_${tag}" \
                --output="results/logs/${tag}_%j.out" \
                --error="results/logs/${tag}_%j.err" \
                --export=ALL,FW_MODEL="$model",FW_METHOD="$method",FW_SPARSITY="$sparsity",FW_PATTERN="$pattern",FW_TAG="$tag",FW_EVAL=full \
                scripts/run_one.sh
        done
    done
done

echo
echo "Submitted (or dry-printed) the full 16-cell grid."
echo "Monitor with: squeue -u \$USER  |  tail -f results/logs/*.out"
