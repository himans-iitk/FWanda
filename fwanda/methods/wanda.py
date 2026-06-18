"""Wanda baseline (Sun et al., ICLR 2024) — faithful reimplementation.

Score  S_ij = |W_ij| * sqrt(E_t[x_j^2])  with Wanda's default *per-output*
(per-row) comparison group and uniform per-row sparsity. No weight update.

Reproduce this first (Section 8.2) — LLaMA-2-7B @ 50% unstructured should give
WikiText-2 PPL ~= 6.92 before trusting any F-Wanda number.
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from fwanda.utils.layerwrapper import WrappedWanda
from fwanda.utils.logging_utils import get_logger
from fwanda.utils.model_loader import find_layers, get_decoder_layers
from fwanda.utils.sparsity import generate_mask, global_mask

_log = get_logger(__name__)


def _all_linears(model):
    out = {}
    for li, layer in enumerate(get_decoder_layers(model)):
        for name, m in find_layers(layer).items():
            out[f"layers.{li}.{name}"] = m
    return out


@torch.no_grad()
def prune(model, calib_data, sparsity: float = 0.5,
          pattern: str = "unstructured", batch_size: int = 1,
          device=None, selection: str = "per_row", **kwargs):
    """Wanda pruning.

    ``selection``: ``per_row`` (default; standard Wanda) | ``global`` (single
        per-layer threshold). The ``global`` variant is the fair baseline for
        F-Wanda's global-selection mode — same selection rule, no Fisher term —
        so their difference isolates the Fisher contribution.
    """
    model.eval()
    device = device or next(model.parameters()).device
    linears = _all_linears(model)
    wrapped = {n: WrappedWanda(m, device) for n, m in linears.items()}

    def make_fwd(w):
        def hook(_module, inp, out):
            w.add_batch_forward(inp[0].detach(), out.detach())
        return hook

    handles = [m.register_forward_hook(make_fwd(wrapped[n]))
               for n, m in linears.items()]

    # One forward pass over the calibration set (no gradients for Wanda).
    for i in tqdm(range(0, calib_data.shape[0], batch_size),
                  desc="Wanda calibration (fwd)"):
        batch = calib_data[i: i + batch_size].to(device)
        model(batch)

    for h in handles:
        h.remove()

    use_global = (selection == "global" and pattern == "unstructured")
    for name, wm in tqdm(wrapped.items(), desc="Wanda apply masks"):
        score = wm.get_score_matrix()
        if use_global:
            mask = global_mask(score, sparsity)
        else:
            mask = generate_mask(score, sparsity, pattern)
        lin = linears[name]
        lin.weight.data *= mask.to(lin.weight.dtype).to(lin.weight.device)
        wm.free()

    _log.info("Wanda pruning done (sparsity=%s pattern=%s selection=%s)",
              sparsity, pattern, selection)
    return model
