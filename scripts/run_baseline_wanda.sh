#!/bin/bash
#SBATCH --job-name=fw_wanda_baseline
#SBATCH --account=def-mageed
#SBATCH --partition=gpupreempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=results/logs/wanda_baseline_%j.out
#SBATCH --error=results/logs/wanda_baseline_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=himanshum4444@gmail.com
#
# Section 8.2 gate: reproduce the published Wanda baseline on LLaMA-2-7B at
# 50% unstructured. Expected WikiText-2 PPL ~= 6.92. If this differs by more
# than 0.05, DEBUG BEFORE running anything else.

source /home/himishra/FWanda/scripts/_env.sh

python -m fwanda.prune \
    --model meta-llama/Llama-2-7b-hf \
    --method wanda \
    --sparsity 0.5 \
    --pattern unstructured \
    --use_flash \
    --eval_ppl \
    --results_csv results/tables/baseline_wanda.csv
