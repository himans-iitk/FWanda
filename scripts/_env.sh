#!/bin/bash
# Compute Canada environment setup — sourced by every job script in this repo.
# Matches the conventions used by long_context_discourse: module-pure StdEnv,
# project venv, HF cache pinned to project storage.

set -euo pipefail

module purge
module load StdEnv/2023 python/3.11 arrow/21.0.0

source /project/def-mageed/himishra/venv/bin/activate

export HF_HOME=/project/def-mageed/himishra/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
# Avoid HF tokenizers parallelism warning under SLURM.
export TOKENIZERS_PARALLELISM=false
# Reproducibility — Wanda/SparseGPT match-checks depend on this.
export PYTHONHASHSEED=0
# Reduce CUDA fragmentation on the F-Wanda backward pass (13B is memory-tight).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Repo root (job scripts cd here so relative paths resolve).
export FWANDA_ROOT=/home/himishra/FWanda
cd "$FWANDA_ROOT"

# Make `python -m fwanda.prune` importable without `pip install -e .`.
export PYTHONPATH="$FWANDA_ROOT:${PYTHONPATH:-}"
