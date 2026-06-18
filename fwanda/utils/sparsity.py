"""Mask generation: uniform per-row (Wanda/baselines), N:M structured, and
the Fisher-allocated per-row budget used by F-Wanda.

Design note (important — read before changing anything here):

    The F-Wanda score is  S_ij = sqrt(omega_bar_i) * |W_ij| * ||X_j||.
    `sqrt(omega_bar_i)` is *constant within row i*, so it cannot change the
    *order* of weights inside a row. Under Wanda's default per-row top-k
    selection it would therefore produce a mask bitwise-identical to Wanda.

    F-Wanda instead spends the Fisher signal where it is not degenerate: it
    keeps Wanda's per-row ranking but makes the per-row *keep budget*
    proportional to sqrt(omega_bar_i). Output neurons whose pre-activation
    consistently drives the loss (large omega_bar_i) retain more of their
    incoming weights; "confident" neurons (small omega_bar_i) are pruned
    harder. The layer-wide sparsity target is preserved exactly (up to
    integer rounding and the per-row floor).

    For strict N:M patterns the density is fixed by hardware (n kept per
    block of m), so a per-row budget cannot apply and the Fisher term is
    provably inert; F-Wanda falls back to standard N:M there (callers log it).
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------------- #
# Uniform baselines (Wanda / magnitude / SparseGPT score selection)
# --------------------------------------------------------------------------- #
def _unstructured_mask(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Per-row top-(1-sparsity) selection (uniform budget across rows)."""
    d_out, d_in = score.shape
    keep = int(round(d_in * (1.0 - sparsity)))
    keep = max(0, min(keep, d_in))
    mask = torch.zeros_like(score, dtype=torch.bool)
    if keep > 0:
        _, topk_idx = torch.topk(score, k=keep, dim=1)
        mask.scatter_(1, topk_idx, True)
    return mask


def _nm_mask(score: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """N:M sparsity — keep the top ``n`` in every block of ``m`` columns."""
    d_out, d_in = score.shape
    assert d_in % m == 0, f"d_in={d_in} must be divisible by m={m}"
    blocks = score.view(d_out, d_in // m, m)
    _, topk = torch.topk(blocks, k=n, dim=2)
    mask_blocks = torch.zeros_like(blocks, dtype=torch.bool)
    mask_blocks.scatter_(2, topk, True)
    return mask_blocks.view(d_out, d_in)


def generate_mask(score: torch.Tensor, sparsity: float,
                  pattern: str) -> torch.Tensor:
    """Boolean keep-mask (True = keep) with a *uniform* per-row budget.

    Used by magnitude / Wanda / SparseGPT-score selection.
    """
    if pattern == "unstructured":
        return _unstructured_mask(score, sparsity)
    if pattern == "2:4":
        return _nm_mask(score, n=2, m=4)
    if pattern == "4:8":
        return _nm_mask(score, n=4, m=8)
    raise ValueError(f"Unknown pattern {pattern!r}")


# --------------------------------------------------------------------------- #
# F-Wanda: Fisher-allocated per-row budget
# --------------------------------------------------------------------------- #
def allocate_row_budget(weights: torch.Tensor, sparsity: float, d_in: int,
                        min_keep: int = 1) -> torch.Tensor:
    """Distribute the layer keep-budget across rows proportional to ``weights``.

    Args:
        weights:  per-row positive importance, here ``sqrt(omega_bar_i)``.
        sparsity: target fraction of weights to zero in the layer.
        d_in:     number of columns (per-row maximum keep).
        min_keep: per-row floor so no output neuron is fully disconnected.

    Returns:
        LongTensor ``(d_out,)`` of integer keep counts, summing to the global
        budget ``round((1-sparsity) * d_out * d_in)`` (clamped for feasibility),
        with every entry in ``[min_keep, d_in]``.

    Uses iterative water-filling (cap rows hitting ``d_in`` or the ``min_keep``
    floor, redistribute the remainder by weight) followed by largest-remainder
    rounding so the integer totals match the budget exactly.
    """
    # Run the allocator on CPU regardless of where ``weights`` came from:
    # this is integer bookkeeping over d_out (<= ~16k) values; moving it off
    # GPU avoids device-mismatch errors and is faster for this workload.
    w = weights.detach().double().cpu().clamp(min=1e-12)
    d_out = w.numel()
    total = int(round((1.0 - sparsity) * d_out * d_in))
    total = max(d_out * min_keep, min(total, d_out * d_in))

    keep = torch.zeros(d_out, dtype=torch.long)
    capped = torch.zeros(d_out, dtype=torch.bool)

    while True:
        free = ~capped
        remaining = total - int(keep[capped].sum().item())
        wsum = w[free].sum()
        if wsum <= 0:
            break
        alloc = torch.zeros(d_out, dtype=torch.double)
        alloc[free] = remaining * w[free] / wsum

        over = free & (alloc > d_in)
        if over.any():
            keep[over] = d_in
            capped |= over
            continue
        under = free & (alloc < min_keep)
        if under.any():
            keep[under] = min_keep
            capped |= under
            continue
        break

    free = ~capped
    remaining = total - int(keep[capped].sum().item())
    if free.any() and remaining > 0:
        wsum = w[free].sum()
        cont = torch.zeros(d_out, dtype=torch.double)
        cont[free] = remaining * w[free] / wsum
        cont = cont.clamp(max=float(d_in))
        keep[free] = cont[free].floor().long()
        deficit = remaining - int(keep[free].sum().item())
        # Largest-remainder: hand the leftover units to the rows with the
        # biggest fractional parts (skip rows already at the d_in cap).
        frac = cont - cont.floor()
        frac[~free] = -1.0
        order = torch.argsort(frac, descending=True).tolist()
        i = 0
        while deficit > 0 and order:
            r = order[i % len(order)]
            if keep[r] < d_in:
                keep[r] += 1
                deficit -= 1
            i += 1
            if i > 16 * len(order):  # safety: every free row already at d_in
                break

    return keep.clamp_(0, d_in)


def mask_from_row_budget(score: torch.Tensor,
                         keep_per_row: torch.Tensor) -> torch.Tensor:
    """Keep the top ``keep_per_row[i]`` entries of row ``i`` by ``score``.

    Vectorised: rank columns within each row once, then threshold each row by
    its own budget.
    """
    d_out, d_in = score.shape
    # rank[i, j] = position of column j when row i is sorted descending
    order = torch.argsort(score, dim=1, descending=True)
    rank = torch.empty_like(order)
    ar = torch.arange(d_in, device=score.device).expand(d_out, d_in)
    rank.scatter_(1, order, ar)
    keep = keep_per_row.to(score.device).view(d_out, 1)
    return rank < keep


def global_mask(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Single per-layer threshold: keep the top ``(1-sparsity)`` fraction of
    *all* weights in the matrix by ``score`` (rows compete for one budget).

    Unlike the per-row variants, a row-wise positive multiplier on ``score``
    (e.g. F-Wanda's ``sqrt(omega_i)``) changes which weights survive here,
    because rows are ranked against each other.
    """
    d_out, d_in = score.shape
    numel = d_out * d_in
    k = int(round((1.0 - sparsity) * numel))
    mask = torch.zeros(numel, dtype=torch.bool, device=score.device)
    if k >= numel:
        return torch.ones((d_out, d_in), dtype=torch.bool, device=score.device)
    if k > 0:
        _, idx = torch.topk(score.flatten(), k)
        mask[idx] = True
    return mask.view(d_out, d_in)


def generate_fisher_mask(score: torch.Tensor, omega_bar: torch.Tensor,
                         sparsity: float, pattern: str,
                         min_keep: int = 1, selection: str = "global"):
    """F-Wanda keep-mask.

    ``selection``:
        ``global`` (default) — single per-layer threshold on the Fisher-scaled
            score, so rows compete; ``sqrt(omega_i)`` genuinely reweights which
            weights survive across the whole matrix.
        ``budget`` — keep Wanda's per-row ranking but allocate the per-row keep
            count proportional to ``sqrt(omega_i)`` (the conservative variant;
            empirically near-identical to Wanda).

    Returns ``(mask, info)``; ``info['fisher_active']`` is False under strict
    N:M, where neither variant can apply.
    """
    if pattern != "unstructured":
        mask = generate_mask(score, sparsity, pattern)
        return mask, {"fisher_active": False, "reason": f"strict {pattern}"}

    if selection == "global":
        # ``score`` already carries the sqrt(omega_i) row factor (set in
        # WrappedFWanda.get_score_matrix), so a global threshold lets the
        # Fisher term move weights across rows.
        mask = global_mask(score, sparsity)
        per_row = mask.int().sum(dim=1)
        info = {"fisher_active": True, "selection": "global",
                "keep_per_row_min": int(per_row.min()),
                "keep_per_row_max": int(per_row.max()),
                "keep_per_row_mean": float(per_row.float().mean())}
        return mask, info

    if selection == "budget":
        budget = allocate_row_budget(
            omega_bar.clamp(min=1e-8).sqrt(), sparsity,
            d_in=score.shape[1], min_keep=min_keep)
        mask = mask_from_row_budget(score, budget)
        info = {"fisher_active": True, "selection": "budget",
                "keep_per_row_min": int(budget.min()),
                "keep_per_row_max": int(budget.max()),
                "keep_per_row_mean": float(budget.float().mean())}
        return mask, info

    raise ValueError(f"Unknown selection {selection!r}; "
                     "expected 'global' or 'budget'")


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def check_sparsity(model, verbose: bool = True) -> float:
    """Compute (and optionally print) achieved sparsity over 2-D weights."""
    total, zeros = 0, 0
    per_layer = []
    for name, p in model.named_parameters():
        if name.endswith("weight") and p.dim() == 2:
            n = p.numel()
            z = (p == 0).sum().item()
            total += n
            zeros += z
            per_layer.append((name, z / max(n, 1)))
    overall = zeros / max(total, 1)
    if verbose:
        print(f"[sparsity] overall = {overall:.4f} "
              f"({zeros:,}/{total:,} zeros)")
    return overall
