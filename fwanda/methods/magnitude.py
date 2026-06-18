"""Magnitude pruning baseline — score = |W|, uniform per-row budget.

No calibration data is used. Kept in the same shape as the other methods so
the entry point can dispatch uniformly.
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from fwanda.utils.logging_utils import get_logger
from fwanda.utils.model_loader import find_layers, get_decoder_layers
from fwanda.utils.sparsity import generate_mask

_log = get_logger(__name__)


@torch.no_grad()
def prune(model, calib_data=None, sparsity: float = 0.5,
          pattern: str = "unstructured", **kwargs):
    layers = get_decoder_layers(model)
    for layer in tqdm(layers, desc="magnitude pruning"):
        for _, lin in find_layers(layer).items():
            W = lin.weight.data
            mask = generate_mask(W.abs().float(), sparsity, pattern)
            W *= mask.to(W.dtype).to(W.device)
    _log.info("magnitude pruning done (sparsity=%s pattern=%s)",
              sparsity, pattern)
    return model
