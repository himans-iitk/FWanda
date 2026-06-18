# Expected (exp) values — predictions, not measurements

Naming convention used throughout this repo: **`exp`** = expected/predicted,
**`ac`** = actual/measured.

The CSVs in `results/tables/exp/` are **predicted** values for every cell of
the experimental grid and the ablations. They exist for two reasons:

1. **Bug-catching.** If an ac run lands wildly off the exp prediction,
   something is broken (wrong dtype, wrong calibration shard, wrong sparsity
   pattern, etc.).
2. **Comparison plots.** `paper/compare_exp_vs_ac.py` overlays the ac numbers
   on these so you can spot deviations at a glance.

These are not paper claims. F-Wanda numbers in particular are **principled
extrapolations**, not measurements.

## Source of each prediction

### Baseline methods (Wanda, SparseGPT, Magnitude)

Taken from the **Wanda paper** (Sun et al., ICLR 2024, arXiv:2306.11695),
Tables 2 and 3. Where the paper does not report a number (e.g. MMLU under
pruning), I used the closest published replication value, typically within
±0.5 pp of the literature consensus.

### F-Wanda predictions

- **WikiText-2 PPL:** ≈ Wanda −0.05 to −0.10. The Fisher reweighting changes
  which weights survive but not by much; PPL is dominated by the bulk-weight
  distribution and is fairly insensitive. F-Wanda within 0.1 of Wanda is the
  spec's own gate, so I predict at the lower edge of that range.
- **MMLU 5-shot:** ≈ Wanda +1.0 to +1.5 pp. The headline claim. Fisher
  reweighting preferentially preserves neurons whose pre-activation drives
  the loss; knowledge tasks are exactly the regime where this matters.
- **Zero-shot avg:** ≈ Wanda +0.2 to +0.4 pp. Small uniform gain across
  tasks, dominated by hellaswag and arc improvements; rte may move either
  way due to small test set.
- **Strict 2:4 patterns:** F-Wanda ≡ Wanda. Hardware-fixed per-block density
  makes the Fisher term mathematically inert. **Predicted identical numbers.**

### Calibration size ablation

F-Wanda's marginal gain over Wanda is expected to plateau around n=128
calibration samples (which is why the Wanda paper picks 128 too):
diminishing returns above ~256.

### Fisher variants ablation

- `empirical` (default): full E[g²] reweighting; best.
- `mean_only`: (E[g])² ≈ 0 for mean-zero gradient sums across many tokens,
  so the budget allocation degenerates toward uniform — predicted to be
  **≈ Wanda baseline** (i.e., F-Wanda's effect washes out).
- `true`: labels sampled from the model's own distribution. Marginally
  different from `empirical` because the gradient distribution is somewhat
  shifted; predicted to be ~= empirical.

## How to compare ac against exp

Once an ac CSV exists at `results/tables/main_results.csv`:

```bash
python paper/compare_exp_vs_ac.py
```

This produces `results/figures/compare_*.pdf` with side-by-side bars
(exp vs ac) per configuration and a tabular delta report at
`results/tables/delta_report.csv`.
