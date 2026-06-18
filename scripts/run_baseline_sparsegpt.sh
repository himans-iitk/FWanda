#!/bin/bash
#SBATCH --job-name=fw_sparsegpt_baseline
#SBATCH --account=def-mageed
#SBATCH --partition=gpupreempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=results/logs/sparsegpt_baseline_%j.out
#SBATCH --error=results/logs/sparsegpt_baseline_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=himanshum4444@gmail.com
#
# SparseGPT baseline reproduction (LLaMA-2-7B, 50% unstructured).
# Expected WikiText-2 PPL ~= 7.0 (Frantar & Alistarh, ICML 2023).

source /home/himishra/FWanda/scripts/_env.sh

python -m fwanda.prune \
    --model meta-llama/Llama-2-7b-hf \
    --method sparsegpt \
    --sparsity 0.5 \
    --pattern unstructured \
    --use_flash \
    --eval_ppl \
    --results_csv results/tables/baseline_sparsegpt.csv
