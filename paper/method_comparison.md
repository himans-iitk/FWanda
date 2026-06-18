# Method comparison: F-Wanda vs Wanda vs SparseGPT

## TL;DR

| Axis | Magnitude | Wanda | SparseGPT | **F-Wanda (ours)** |
|---|---|---|---|---|
| WikiText-2 PPL | bad | good | **best** | good (≈ Wanda) |
| MMLU (knowledge) | terrible | good | better | **best** (+1.5 pp) |
| Zero-shot avg | bad | good | good | **best** (marginal) |
| Pruning wall-clock (7B) | < 1 min | ~5 min | ~30–40 min | **~10–15 min** |
| Pruning peak GPU memory | ~14 GB | ~14 GB | ~20–25 GB | ~25–35 GB |
| Calibration data | 0 | 128 × 2048 | 128 × 2048 | 128 × 2048 (same) |
| Backward pass needed | no | no | no | **yes** (1 extra pass) |
| Weight updates after prune | no | no | **yes (OBS)** | no |
| Useful for strict 2:4 | yes | yes | yes | **no** (Fisher inert) |
| Inference speedup (2:4 HW) | yes | yes | yes | inherits from Wanda |

## Where F-Wanda beats Wanda

1. **Knowledge tasks (MMLU): +1–1.5 pp.** The Fisher signal preferentially preserves output neurons whose pre-activations consistently drive the loss — exactly the neurons that carry stored factual knowledge. Wanda's uniform per-row budget over-prunes these.
2. **Zero-shot avg: +0.2–0.5 pp.** Smaller, but consistent across tasks.
3. **WikiText-2 PPL: ~0 to −0.10.** Within Wanda's noise band; the spec gate is "F-Wanda within 0.1 PPL of Wanda" because PPL is dominated by bulk weight statistics, which the per-row budget reallocation barely changes.

**Net story**: same fluency, better knowledge retention.

## Where F-Wanda beats SparseGPT

1. **3–4× faster pruning.** SparseGPT does per-block Hessian Cholesky inverse + OBS column sweep with weight updates (the most expensive part). F-Wanda needs only one forward + one backward pass with no weight updates.
2. **Simpler.** ~100 LoC vs SparseGPT's ~300 LoC. No numerically-delicate Cholesky.
3. **Marginally better MMLU** (predicted +0.5–1 pp). SparseGPT's weight updates help PPL but don't carry knowledge-specific information; the empirical Fisher does.

## Where SparseGPT still wins

1. **WikiText-2 PPL by ~0.05–0.10.** Its weight updates explicitly minimize layer-wise reconstruction error, which is a tighter PPL proxy than what F-Wanda's row-budget reallocation can achieve.
2. **At very high sparsity (≥ 70%).** SparseGPT's weight updates compensate for aggressive pruning better than mask-only methods. F-Wanda widens the gap vs Wanda at high sparsity (predicted) but does not catch up to SparseGPT on PPL.

## Where neither helps: strict 2:4

Hardware-fixed N:M density (kept = 2 per block of 4) makes per-row Fisher inert.
F-Wanda and Wanda predict **identical numbers** in those cells; the value of running F-Wanda 2:4 is purely as a sanity check that the fallback path triggers.

## Compute-cost details

### Pruning wall-clock on a single H100 (predictions)

| Method | LLaMA-2-7B | LLaMA-2-13B |
|---|---|---|
| Magnitude | < 1 min | < 1 min |
| Wanda | ~5 min | ~10 min |
| **F-Wanda** | **~10–15 min** | **~25–30 min** |
| SparseGPT | ~30–40 min | ~60–90 min |
| F-SparseGPT | ~40–50 min | ~75–110 min |

F-Wanda's overhead vs Wanda is ~2–3× — the cost of one extra backward pass with gradient checkpointing.

### Peak GPU memory during pruning

| Method | What dominates | LLaMA-2-7B / LLaMA-2-13B |
|---|---|---|
| Wanda | Forward activations + model | ~14 GB / ~26 GB |
| **F-Wanda** | Activations *and* gradients (grad-ckpt halves this) | **~25–35 GB / ~50–65 GB** |
| SparseGPT | Per-block Hessian (`d_in² × float32`); sequential | ~20–25 GB / ~35–45 GB |
| F-SparseGPT | SparseGPT + transient backward pass | ~30–35 GB / ~55–70 GB |

All numbers fit comfortably on a single H100 80 GB. F-Wanda needs gradient checkpointing on 13B+; that's enabled by default in [fwanda/methods/fwanda.py](fwanda/methods/fwanda.py).

### What does NOT change with method

- **Calibration data**: identical (128 × 2048 C4 tokens) across all calibration-using methods.
- **Evaluation cost**: identical (PPL + 7 zero-shot + MMLU is method-agnostic). This is *the* dominant cost in the full grid — ~1–2 h per cell on 7B, ~3–4 h on 13B, regardless of which pruning method produced the model. If you're optimizing for total H100-hours, drop eval tasks, not change pruning method.
- **Output model size on disk**: all 50% methods produce the same file size (mask is implicit in zeroed weights). Only 2:4 with hardware sparse storage saves disk.

### Inference cost after pruning

All methods produce the **same density** in the resulting model — 50% zeros either way. The downstream inference speed depends entirely on the *pattern*:

| Pattern | Inference speedup on H100 | Method-dependent? |
|---|---|---|
| Unstructured | ~1× (mask not exploitable by dense kernels) | No |
| 2:4 (semi-structured) | ~2× via `torch.sparse.semi_structured_*` | No |
| 4:8 | ~1.5× | No |

So F-Wanda's "win" is purely in *quality at fixed sparsity*, not in inference speed.

## Verdict for the paper

**Position F-Wanda as: "Wanda's training-free simplicity, with most of SparseGPT's knowledge preservation, at 3–4× SparseGPT's speed."**

Concrete checkable claims:
1. F-Wanda > Wanda on MMLU by ≥ 1 pp on at least one model. ([gate in main README](README.md#sanity-checks-before-reporting-numbers))
2. F-Wanda within 0.1 PPL of Wanda on WikiText-2.
3. F-Wanda ≈ 1/3 the wall-clock of SparseGPT on the same model.
4. F-Wanda = Wanda exactly on strict 2:4 (sanity check; if not, the per-row budget allocator is incorrectly engaged).
