#!/bin/bash
#SBATCH --job-name=fw_smoke
#SBATCH --account=def-mageed
#SBATCH --partition=gpupreempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=results/logs/smoke_%j.out
#SBATCH --error=results/logs/smoke_%j.err
#
# Tiny end-to-end smoke: JackFram/llama-68m (a real LLaMA-arch 68M model)
# with 4 calibration sequences. Verifies that the full pipeline (load -> hooks
# -> fwd+bwd -> Fisher budget -> apply mask -> PPL eval) runs without errors
# before you commit a multi-hour real job.

source /home/himishra/FWanda/scripts/_env.sh

for method in magnitude wanda fwanda sparsegpt; do
    echo "================ smoke: $method ================"
    python -m fwanda.prune \
        --model JackFram/llama-68m \
        --method "$method" \
        --sparsity 0.5 \
        --pattern unstructured \
        --nsamples 4 \
        --seqlen 256 \
        --eval_seqlen 256 \
        --torch_dtype float32 \
        --eval_ppl \
        --results_csv results/tables/smoke.csv \
        || { echo "smoke FAILED for $method"; exit 1; }
done

echo "All smoke configurations passed."
