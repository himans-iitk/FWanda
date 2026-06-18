#!/bin/bash
#SBATCH --job-name=fw_fwanda
#SBATCH --account=def-mageed
#SBATCH --partition=gpupreempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=results/logs/fwanda_%j.out
#SBATCH --error=results/logs/fwanda_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=himanshum4444@gmail.com
#
# F-Wanda on LLaMA-2-7B at 50% unstructured (the headline configuration).

source /home/himishra/FWanda/scripts/_env.sh

python -m fwanda.prune \
    --model meta-llama/Llama-2-7b-hf \
    --method fwanda \
    --sparsity 0.5 \
    --pattern unstructured \
    --fisher_variant empirical \
    --use_flash \
    --eval_ppl --eval_zero_shot --eval_mmlu \
    --results_csv results/tables/main_results.csv
