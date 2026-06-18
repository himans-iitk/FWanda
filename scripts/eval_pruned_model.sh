#!/bin/bash
#SBATCH --job-name=fw_eval
#SBATCH --account=def-mageed
#SBATCH --partition=gpupreempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=results/logs/eval_%j.out
#SBATCH --error=results/logs/eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=himanshum4444@gmail.com
#
# Re-evaluate a previously saved pruned model directory on PPL + zero-shot +
# MMLU. The 'method' is recorded as ``saved`` so the row joins cleanly to the
# main results table.
#
#   sbatch --export=ALL,FW_MODEL_DIR=/path/to/pruned scripts/eval_pruned_model.sh

source /home/himishra/FWanda/scripts/_env.sh

: "${FW_MODEL_DIR:?FW_MODEL_DIR must be set}"

python -m fwanda.prune \
    --model "$FW_MODEL_DIR" \
    --method magnitude \
    --sparsity 0.0 \
    --pattern unstructured \
    --use_flash \
    --eval_ppl --eval_zero_shot --eval_mmlu \
    --results_csv results/tables/eval_only.csv
# NOTE: --method magnitude with --sparsity 0 is a no-op pruner; we just want
# the eval pipeline on the already-pruned checkpoint.
