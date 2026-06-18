# F-Wanda — Fisher-Reweighted Post-Training Pruning for LLMs

A drop-in modification of [Wanda](https://github.com/locuslab/wanda) that
uses a per-output-neuron empirical-Fisher scalar to reallocate the per-row
keep budget. No weight updates, no retraining — one extra backward pass over
the calibration data is the only added cost.

```
S_ij = sqrt(omega_bar_i) * |W_ij| * sqrt(E_t[x_j^2])
```

with `omega_bar_i = E_t[g_{t,i}^2]` from a single forward+backward pass.

## How F-Wanda differs from Wanda (read this once)

The spec describes a per-row score reweighting. But `sqrt(omega_bar_i)` is a
**positive per-row constant**, so under Wanda's default per-row top-k
selection it cannot reorder weights within a row — the resulting mask would
be bitwise-identical to Wanda's.

F-Wanda instead spends the Fisher signal where it is non-degenerate: it keeps
the per-row ranking but allocates the **global keep budget** across rows
proportional to `sqrt(omega_bar_i)` (water-filling + largest-remainder
rounding). Output neurons whose pre-activation consistently drives the loss
retain more of their incoming weights; "confident" neurons are pruned harder.
See [`fwanda/utils/sparsity.py`](fwanda/utils/sparsity.py) for the
implementation and the design comment.

**Caveat:** strict N:M (e.g., 2:4) fixes the per-block density at hardware
level, so a per-row budget cannot apply — F-Wanda falls back to standard N:M
in those configs (and logs a warning). All F-Wanda gains will come from the
unstructured runs.

## Repository layout

| Path | Purpose |
|---|---|
| [`fwanda/prune.py`](fwanda/prune.py) | CLI entry point |
| [`fwanda/methods/`](fwanda/methods/) | magnitude, wanda, sparsegpt, fwanda, fsparsegpt |
| [`fwanda/data/calibration.py`](fwanda/data/calibration.py) | C4 / Wikipedia / SlimPajama calibration sampling |
| [`fwanda/utils/sparsity.py`](fwanda/utils/sparsity.py) | mask generation incl. Fisher budget allocator |
| [`fwanda/utils/layerwrapper.py`](fwanda/utils/layerwrapper.py) | forward + backward hooks (Wanda + F-Wanda) |
| [`fwanda/eval/`](fwanda/eval/) | WikiText-2 PPL, lm-eval-harness, MMLU |
| [`scripts/`](scripts/) | Compute Canada SLURM job scripts |
| [`paper/`](paper/) | LaTeX table + PDF figure generators |
| [`tests/`](tests/) | pytest unit tests (offline) |

## Setup (Compute Canada)

This repo follows the same module + venv convention as
`long_context_discourse`. From a login node:

```bash
module purge
module load StdEnv/2023 python/3.11 arrow/21.0.0

# Reuse the existing project venv:
source /project/def-mageed/himishra/venv/bin/activate
pip install --no-index -r requirements.txt   # CC mirror; --no-index is fast

# OR a fresh one:
# virtualenv --no-download /project/def-mageed/himishra/venv_fwanda
# source /project/def-mageed/himishra/venv_fwanda/bin/activate
# pip install -e .
```

LLaMA-2 weights are gated; one-time:
```bash
huggingface-cli login
```
The job-script `HF_HOME` points at `/project/def-mageed/himishra/hf_cache`.

## Reproducing baselines (do this first — Section 8.2)

Before running F-Wanda, **always** reproduce Wanda's published numbers. If
they don't match, fix that before doing anything else.

```bash
sbatch scripts/run_baseline_wanda.sh
# expected: WikiText-2 PPL ~= 6.92 on LLaMA-2-7B at 50% unstructured.
# match within ±0.05 before trusting any F-Wanda number.
```

```bash
sbatch scripts/run_baseline_sparsegpt.sh
# expected: WikiText-2 PPL ~= 7.0
```

## Running the full grid (16 jobs)

```bash
bash scripts/run_full_grid.sh        # submits 16 sbatch jobs
# DRY=1 bash scripts/run_full_grid.sh  # print the sbatch commands, do not submit
```

Each job writes one row to `results/tables/main_results.csv`.

## Ablations

```bash
bash scripts/ablations/ablation_calibration_size.sh    # n in {32,64,128,256,512}
bash scripts/ablations/ablation_fisher_variants.sh     # empirical | mean_only | true
sbatch scripts/ablations/ablation_layer_breakdown.sh   # writes layer_breakdown.npz
```

## Building paper artifacts

```bash
python paper/generate_tables.py    # results/tables/table_{1..4}.tex
python paper/generate_figures.py   # results/figures/fig_{1..3}.pdf
```

## Smoke test (quick correctness check)

```bash
sbatch scripts/smoke_test.sh
# JackFram/llama-68m + 4 calibration sequences; finishes in ~5 min on an H100.
```

## Unit tests

```bash
pytest tests/                     # offline tests (Fisher budget, masks)
pytest -m slow tests/             # also runs the network-dependent C4 sampler test
```

## Sanity checks before reporting numbers

1. Baseline Wanda PPL within 0.05 of 6.92 (LLaMA-2-7B, 50% unstructured).
2. Achieved sparsity within 0.5% of target — printed by `check_sparsity`.
3. F-Wanda within 0.1 PPL of Wanda on WikiText-2 (if much worse, the
   `omega_bar_i` clamp is probably wrong).
4. Gradient norm probe in the F-Wanda log: 1e-3 to 1e-1, never 0 / NaN.
5. `omega_bar_i` histogram should span ~3 orders of magnitude (log-normal-ish).

## References

- Wanda — Sun et al., ICLR 2024 (https://github.com/locuslab/wanda)
- SparseGPT — Frantar & Alistarh, ICML 2023 (https://github.com/IST-DASLab/sparsegpt)
- lm-evaluation-harness 0.4.2 (pinned)
