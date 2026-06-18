#!/bin/bash
#SBATCH --job-name=fw_layer_breakdown
#SBATCH --account=def-mageed
#SBATCH --partition=gpupreempt
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=results/logs/abl_layer_breakdown_%j.out
#SBATCH --error=results/logs/abl_layer_breakdown_%j.err
#
# Per-layer breakdown: dump omega_bar distributions and per-layer F-Wanda vs
# Wanda mask-disagreement counts to results/figures/layer_breakdown.npz so
# paper/generate_figures.py can build Figure 2.

source /home/himishra/FWanda/scripts/_env.sh

python paper/layer_breakdown.py \
    --model meta-llama/Llama-2-7b-hf \
    --sparsity 0.5 \
    --out results/figures/layer_breakdown.npz
