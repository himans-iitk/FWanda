"""SparseGPT baseline (Frantar & Alistarh, ICML 2023).

Faithful port of ``IST-DASLab/sparsegpt`` ``sparsegpt.py`` + the standard
per-decoder-block harness used by both SparseGPT and Wanda. The fasterprune
inner loop is mathematically delicate; do not reorder or simplify without
diff-checking against the reference implementation.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
from tqdm import tqdm

from fwanda.utils.logging_utils import get_logger
from fwanda.utils.model_loader import find_layers, get_decoder_layers

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Per-Linear SparseGPT helper
# --------------------------------------------------------------------------- #
class SparseGPT:
    """Accumulate the activation Hessian H = sum_t x_t x_t^T for one Linear,
    then prune + update its weight in place via OBS column sweep."""

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows, self.cols = layer.weight.shape
        self.H = torch.zeros((self.cols, self.cols), device=self.dev,
                             dtype=torch.float32)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.float()
        n_new = inp.shape[0]
        # Running average of x x^T.
        self.H *= self.nsamples / (self.nsamples + n_new)
        self.nsamples += n_new
        inp = inp * math.sqrt(2.0 / self.nsamples)
        self.H += inp.t() @ inp

    @torch.no_grad()
    def fasterprune(self, sparsity: float, prune_n: int = 0, prune_m: int = 0,
                    blocksize: int = 128, percdamp: float = 0.01,
                    extra_row_weight: torch.Tensor = None):
        """Run the SparseGPT column sweep.

        ``extra_row_weight`` (used by F-SparseGPT) is an optional per-row
        positive multiplier applied to the OBS score *before* the per-block
        sparsity threshold is chosen — i.e., the F-Wanda Fisher trick lifted to
        SparseGPT. When ``None`` this is pure SparseGPT.
        """
        W = self.layer.weight.data.clone().float()  # (rows, cols)
        H = self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.cols, device=self.dev)
        H[diag, diag] += damp

        # Hinv as the upper-triangular Cholesky factor of H^-1.
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None
        if extra_row_weight is not None:
            erw = extra_row_weight.to(W.device).float().view(-1, 1)
        else:
            erw = None

        for i1 in range(0, self.cols, blocksize):
            i2 = min(i1 + blocksize, self.cols)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0:
                # Unstructured: per-block global threshold on OBS score.
                tmp = W1.pow(2) / torch.diag(Hinv1).reshape(1, -1).pow(2)
                if erw is not None:
                    tmp = tmp * erw  # F-SparseGPT row reweight
                thresh = torch.sort(tmp.flatten())[0][
                    int(tmp.numel() * sparsity)]
                mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1, dtype=torch.bool)

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i: i + prune_m].pow(2) / torch.diag(
                        Hinv1)[i: i + prune_m].reshape(1, -1).pow(2)
                    if erw is not None:
                        tmp = tmp * erw
                    _, idx = torch.topk(tmp, prune_n, dim=1, largest=False)
                    mask1.scatter_(1, i + idx, True)

                q = w.clone()
                q[mask1[:, i]] = 0
                Q1[:, i] = q
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(
                    Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        self.layer.weight.data.copy_(
            W.reshape(self.layer.weight.shape).to(self.layer.weight.dtype))

    def free(self):
        self.H = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Per-block calibration harness — shared with F-SparseGPT
# --------------------------------------------------------------------------- #
class _CatcherDone(Exception):
    """Used to short-circuit forward once the first block has been invoked."""
    pass


# Keys we never want to roundtrip — they carry per-call cache state that
# would otherwise leak between captured samples or trigger version mismatches
# on newer transformers releases.
_DROP_KWARGS = frozenset({
    "past_key_value", "past_key_values", "use_cache",
    "output_attentions", "output_hidden_states",
})


class _Catcher(nn.Module):
    """Captures the input + kwargs of the first decoder layer, then aborts.

    Accepts both positional and keyword forms — newer HF releases pass
    ``hidden_states`` positionally and may add new kwargs (``cache_position``,
    ``position_embeddings``); we forward them all through unchanged.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.inps = []
        self.kwargs = []

    def forward(self, *args, **kwargs):
        hidden_states = args[0] if args else kwargs.pop("hidden_states")
        self.inps.append(hidden_states.detach().cpu())
        # Drop cache-related kwargs (they would leak state between samples).
        kw = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
              for k, v in kwargs.items()
              if k not in _DROP_KWARGS}
        self.kwargs.append(kw)
        raise _CatcherDone


@torch.no_grad()
def capture_block_inputs(model, calib_data, device):
    """Stash the residual stream entering decoder block 0 for every sample."""
    layers = get_decoder_layers(model)
    layers[0] = _Catcher(layers[0])
    for i in range(calib_data.shape[0]):
        try:
            model(calib_data[i: i + 1].to(device))
        except _CatcherDone:
            pass
    catcher = layers[0]
    layers[0] = catcher.module
    return catcher.inps, catcher.kwargs


def _run_block_collect(block, inps, kwargs_list, gpts: Dict[str, SparseGPT],
                       device):
    handles = []

    def make_hook(g):
        def hook(_m, inp, _out):
            g.add_batch(inp[0].detach())
        return hook

    for name, lin in find_layers(block).items():
        handles.append(lin.register_forward_hook(make_hook(gpts[name])))
    for x, kw in zip(inps, kwargs_list):
        kw_dev = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in kw.items()}
        _ = block(x.to(device), **kw_dev)
    for h in handles:
        h.remove()


@torch.no_grad()
def _run_block_forward(block, inps, kwargs_list, device):
    """Run the (now-pruned) block to produce inputs for the next block."""
    outs = []
    for x, kw in zip(inps, kwargs_list):
        kw_dev = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in kw.items()}
        y = block(x.to(device), **kw_dev)
        outs.append((y[0] if isinstance(y, tuple) else y).detach().cpu())
    return outs


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def prune(model, calib_data, sparsity: float = 0.5,
          pattern: str = "unstructured", device=None,
          percdamp: float = 0.01, blocksize: int = 128,
          extra_row_weight_factory=None, **kwargs):
    """Apply SparseGPT in place.

    ``extra_row_weight_factory(block_idx, sub_name) -> Tensor|None`` is the
    extension point used by F-SparseGPT to inject per-row Fisher weights.
    """
    model.eval()
    device = device or next(model.parameters()).device
    if hasattr(model, "config"):
        model.config.use_cache = False

    prune_n, prune_m = (0, 0)
    if pattern == "2:4":
        prune_n, prune_m = 2, 4
    elif pattern == "4:8":
        prune_n, prune_m = 4, 8
    elif pattern != "unstructured":
        raise ValueError(f"Unknown pattern {pattern!r}")

    inps, kwargs_list = capture_block_inputs(model, calib_data, device)
    layers = get_decoder_layers(model)

    for li, block in enumerate(tqdm(layers, desc="SparseGPT layers")):
        sub = find_layers(block)
        gpts = {name: SparseGPT(lin) for name, lin in sub.items()}

        _run_block_collect(block, inps, kwargs_list, gpts, device)

        for name, g in gpts.items():
            erw = (extra_row_weight_factory(li, name)
                   if extra_row_weight_factory is not None else None)
            g.fasterprune(sparsity=sparsity, prune_n=prune_n, prune_m=prune_m,
                          blocksize=blocksize, percdamp=percdamp,
                          extra_row_weight=erw)
            g.free()

        inps = _run_block_forward(block, inps, kwargs_list, device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if hasattr(model, "config"):
        model.config.use_cache = True
    _log.info("SparseGPT done (sparsity=%s pattern=%s)", sparsity, pattern)
    return model
