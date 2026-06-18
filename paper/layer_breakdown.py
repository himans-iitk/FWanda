"""Compute per-layer Fisher statistics and F-Wanda vs Wanda mask
disagreement, dumped as an ``.npz`` for the paper's Figure 2.

For every Linear inside every decoder block we record:
  - omega_bar_i over the calibration set (the per-row Fisher scalar)
  - Wanda's uniform per-row keep count
  - F-Wanda's Fisher-allocated per-row keep count
  - the count of weights that disagree between the two masks
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from fwanda.data.calibration import get_calibration
from fwanda.utils.layerwrapper import WrappedFWanda
from fwanda.utils.model_loader import (find_layers, get_decoder_layers,
                                        load_model)
from fwanda.utils.sparsity import (allocate_row_budget, mask_from_row_budget,
                                    _unstructured_mask)


def main(args):
    model, tokenizer = load_model(args.model, torch_dtype=args.torch_dtype,
                                  use_flash=args.use_flash)
    device = next(model.parameters()).device
    calib = get_calibration("c4", tokenizer, n_samples=args.nsamples,
                            seq_len=args.seqlen, seed=args.seed)

    layers = get_decoder_layers(model)
    linears = {}
    for li, block in enumerate(layers):
        for name, m in find_layers(block).items():
            linears[(li, name)] = m
    wrapped = {k: WrappedFWanda(m, device) for k, m in linears.items()}

    def make_fwd(w):
        def hook(_m, inp, out): w.add_batch_forward(inp[0].detach(),
                                                    out.detach())
        return hook

    def make_bwd(w):
        def hook(_m, _gin, gout): w.add_batch_backward(gout[0].detach())
        return hook

    handles = []
    for k, m in linears.items():
        handles.append(m.register_forward_hook(make_fwd(wrapped[k])))
        handles.append(m.register_full_backward_hook(make_bwd(wrapped[k])))

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    for i in tqdm(range(calib.shape[0]), desc="calibration fwd+bwd"):
        batch = calib[i: i + 1].to(device)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            model(batch, labels=batch).loss.backward()
    model.zero_grad(set_to_none=True)
    for h in handles:
        h.remove()

    rows = []
    for (li, name), wm in wrapped.items():
        omega = wm.get_omega_bar().detach().cpu()
        score = wm.get_score_matrix().detach().cpu()
        d_out, d_in = score.shape
        wanda_mask = _unstructured_mask(score, args.sparsity)
        budget = allocate_row_budget(omega.clamp(min=1e-8).sqrt(),
                                     args.sparsity, d_in)
        fw_mask = mask_from_row_budget(score, budget)
        disagree = int((wanda_mask ^ fw_mask).sum().item())
        rows.append(dict(
            layer=li, sub=name,
            d_out=d_out, d_in=d_in,
            omega_mean=float(omega.mean()),
            omega_std=float(omega.std()),
            omega_p10=float(omega.quantile(0.1)),
            omega_p90=float(omega.quantile(0.9)),
            wanda_keep_per_row=int(round(d_in * (1 - args.sparsity))),
            fw_budget_min=int(budget.min()),
            fw_budget_max=int(budget.max()),
            fw_budget_std=float(budget.float().std()),
            mask_disagree=disagree,
            mask_disagree_frac=disagree / (d_out * d_in),
        ))
        wm.free()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, rows=np.array(rows, dtype=object),
             sparsity=args.sparsity, model=args.model)
    print(f"Wrote {args.out} ({len(rows)} linears).")


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument("--nsamples", type=int, default=32)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--torch_dtype", default="bfloat16")
    p.add_argument("--use_flash", action="store_true", default=True)
    p.add_argument("--out", default="results/figures/layer_breakdown.npz")
    args = p.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
