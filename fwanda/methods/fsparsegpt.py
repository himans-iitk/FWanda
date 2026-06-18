"""F-SparseGPT (appendix / stretch goal).

Same SparseGPT machinery, plus the F-Wanda Fisher row reweighting injected
into the OBS score before each per-block sparsity threshold is chosen. Weight
updates are unchanged. Computes ``omega_bar_i`` in one extra fwd+bwd pass
identical to F-Wanda's, then hands it to SparseGPT via
``extra_row_weight_factory``.
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from fwanda.methods import sparsegpt as _sg
from fwanda.utils.layerwrapper import WrappedFWanda
from fwanda.utils.logging_utils import get_logger
from fwanda.utils.model_loader import find_layers, get_decoder_layers

_log = get_logger(__name__)


def _collect_omega(model, calib_data, device,
                   fisher_variant: str = "empirical"):
    """One fwd+bwd pass to collect per-row Fisher scalars for every Linear."""
    layers = get_decoder_layers(model)
    linears = {}
    for li, layer in enumerate(layers):
        for name, m in find_layers(layer).items():
            linears[(li, name)] = m
    wrapped = {k: WrappedFWanda(m, device, fisher_variant=fisher_variant)
               for k, m in linears.items()}

    def make_bwd(w):
        def hook(_m, _gin, gout):
            w.add_batch_backward(gout[0].detach())
        return hook

    handles = [m.register_full_backward_hook(make_bwd(wrapped[k]))
               for k, m in linears.items()]

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    # Only Linear-output grads are needed (hooks); freeze params to skip ~26GB
    # of param-grad storage on 13B. Keep input embedding trainable so the graph
    # still flows. Restored afterwards.
    orig_requires_grad = {id(p): p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    model.get_input_embeddings().weight.requires_grad_(True)
    for i in tqdm(range(calib_data.shape[0]),
                  desc="F-SparseGPT omega pass (fwd+bwd)"):
        batch = calib_data[i: i + 1].to(device)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = model(batch, labels=batch).loss
            loss.backward()
    model.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.requires_grad_(orig_requires_grad.get(id(p), True))
    model.gradient_checkpointing_disable()

    for h in handles:
        h.remove()

    omega = {k: w.get_omega_bar().detach().cpu() for k, w in wrapped.items()}
    for w in wrapped.values():
        w.free()
    return omega


def prune(model, calib_data, sparsity: float = 0.5,
          pattern: str = "unstructured", device=None,
          fisher_variant: str = "empirical", **kwargs):
    model.eval()
    device = device or next(model.parameters()).device

    omega = _collect_omega(model, calib_data, device, fisher_variant)

    def factory(li, sub_name):
        # sqrt(omega) keeps the OBS score positive-multiplied; falls back to
        # ones if the entry is missing (defensive — shouldn't happen).
        key = (li, sub_name)
        if key not in omega:
            return None
        return omega[key].clamp(min=1e-8).sqrt()

    _sg.prune(model, calib_data, sparsity=sparsity, pattern=pattern,
              device=device, extra_row_weight_factory=factory, **kwargs)
    _log.info("F-SparseGPT done (variant=%s)", fisher_variant)
    return model
