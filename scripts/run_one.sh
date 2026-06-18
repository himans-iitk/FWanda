#!/bin/bash
#SBATCH --job-name=fw_one
#SBATCH --account=def-mageed
#SBATCH --partition=gpupreempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --output=results/logs/one_%j.out
#SBATCH --error=results/logs/one_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=himanshum4444@gmail.com
#
# Generic single-config job. Used by run_full_grid.sh — every cell of the
# experimental grid is one of these. Submit with:
#
#   sbatch --export=ALL,\
#       FW_MODEL=meta-llama/Llama-2-7b-hf,\
#       FW_METHOD=fwanda,\
#       FW_SPARSITY=0.5,\
#       FW_PATTERN=unstructured,\
#       FW_TAG=llama2-7b_fwanda_0.5_unstructured \
#       scripts/run_one.sh

source /home/himishra/FWanda/scripts/_env.sh

: "${FW_MODEL:?FW_MODEL must be set}"
: "${FW_METHOD:?FW_METHOD must be set}"
: "${FW_SPARSITY:?FW_SPARSITY must be set}"
: "${FW_PATTERN:?FW_PATTERN must be set}"
: "${FW_TAG:?FW_TAG must be set}"
: "${FW_FISHER_VARIANT:=empirical}"
: "${FW_NSAMPLES:=128}"
: "${FW_RESULTS_CSV:=results/tables/main_results.csv}"
: "${FW_EVAL:=full}"        # 'full' = ppl+zero_shot+mmlu, 'ppl' = ppl only
: "${FW_SELECTION:=auto}"   # auto|per_row|budget|global (auto: global for fwanda)

EVAL_FLAGS="--eval_ppl"
if [ "$FW_EVAL" = "full" ]; then
    EVAL_FLAGS="$EVAL_FLAGS --eval_zero_shot --eval_mmlu"
fi

python -m fwanda.prune \
    --model "$FW_MODEL" \
    --method "$FW_METHOD" \
    --sparsity "$FW_SPARSITY" \
    --pattern "$FW_PATTERN" \
    --nsamples "$FW_NSAMPLES" \
    --fisher_variant "$FW_FISHER_VARIANT" \
    --selection "$FW_SELECTION" \
    --use_flash \
    $EVAL_FLAGS \
    --results_csv "$FW_RESULTS_CSV"
