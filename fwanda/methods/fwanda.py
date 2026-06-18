"""F-Wanda — Fisher-reweighted post-training pruning (this paper's method).

vs. Wanda: one extra backward pass over the calibration data collects the
per-token output-gradients g_{t,i}. The per-row Fisher scalar

    omega_bar_i = (1/N) * sum_t g_{t,i}^2

reweights the *per-row keep budget* (see ``utils/sparsity.py`` for why a
score-only reweighting would be a no-op under per-row selection). Weights are
never updated — like Wanda, only masked.
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from fwanda.utils.layerwrapper import WrappedFWanda
from fwanda.utils.logging_utils import get_logger
from fwanda.utils.model_loader import find_layers, get_decoder_layers
from fwanda.utils.sparsity import generate_fisher_mask

_log = get_logger(__name__)


def _all_linears(model):
    out = {}
    for li, layer in enumerate(get_decoder_layers(model)):
        for name, m in find_layers(layer).items():
            out[f"layers.{li}.{name}"] = m
    return out


def _sampled_label_loss(model, batch):
    """Loss against labels sampled from the model's own predictive
    distribution — yields the *true* Fisher rather than the empirical one."""
    logits = model(batch).logits[:, :-1, :].float()
    probs = torch.softmax(logits, dim=-1)
    sampled = torch.multinomial(
        probs.reshape(-1, probs.shape[-1]), 1).reshape(logits.shape[:-1])
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), sampled.reshape(-1))


def prune(model, calib_data, sparsity: float = 0.5,
          pattern: str = "unstructured", batch_size: int = 1,
          device=None, fisher_variant: str = "empirical",
          selection: str = "global",
          grad_checkpointing: bool = True, **kwargs):
    """Apply F-Wanda in place.

    ``fisher_variant``: ``empirical`` (default) | ``mean_only`` | ``true``.
    ``selection``: ``global`` (default; per-layer threshold so the Fisher term
        reweights across rows) | ``budget`` (per-row keep-count allocation).
    """
    model.eval()
    device = device or next(model.parameters()).device
    if grad_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    linears = _all_linears(model)
    wrapped = {n: WrappedFWanda(m, device, fisher_variant=fisher_variant)
               for n, m in linears.items()}

    def make_fwd(w):
        def hook(_m, inp, out):
            w.add_batch_forward(inp[0].detach(), out.detach())
        return hook

    def make_bwd(w):
        def hook(_m, _gin, gout):
            w.add_batch_backward(gout[0].detach())
        return hook

    handles = []
    for n, m in linears.items():
        handles.append(m.register_forward_hook(make_fwd(wrapped[n])))
        handles.append(m.register_full_backward_hook(make_bwd(wrapped[n])))

    # Memory: we only need gradients at the Linear *outputs* (captured by the
    # backward hooks), never the parameter gradients. Freeze every parameter
    # so autograd skips ~|theta| floats of .grad storage (~26GB on 13B), and
    # keep only the input embedding trainable so the graph still flows back
    # through every activation. Saved state is restored afterwards.
    orig_requires_grad = {id(p): p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    model.get_input_embeddings().weight.requires_grad_(True)

    # ONE forward + backward pass over the whole calibration set.
    for i in tqdm(range(0, calib_data.shape[0], batch_size),
                  desc=f"F-Wanda calibration (fwd+bwd, {fisher_variant})"):
        batch = calib_data[i: i + batch_size].to(device)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            if fisher_variant == "true":
                loss = _sampled_label_loss(model, batch)
            else:
                loss = model(batch, labels=batch).loss
            loss.backward()
    model.zero_grad(set_to_none=True)

    for p in model.parameters():
        p.requires_grad_(orig_requires_grad.get(id(p), True))

    for h in handles:
        h.remove()
    if grad_checkpointing:
        model.gradient_checkpointing_disable()
        model.config.use_cache = True

    # Sanity log: gradient magnitude should be ~1e-3..1e-1, never 0/NaN.
    g_probe = next(iter(wrapped.values())).grad_sq_norm_running
    _log.info("backward grad-norm probe (last batch, first layer): %.3e",
              g_probe)

    n_active = 0
    for name, wm in tqdm(wrapped.items(), desc="F-Wanda apply masks"):
        score = wm.get_score_matrix()
        omega = wm.get_omega_bar()
        mask, info = generate_fisher_mask(score, omega, sparsity, pattern,
                                          selection=selection)
        n_active += int(info.get("fisher_active", False))
        lin = linears[name]
        with torch.no_grad():
            lin.weight.data *= mask.to(lin.weight.dtype).to(lin.weight.device)
        wm.free()

    if pattern != "unstructured":
        _log.warning(
            "pattern=%s is strict N:M — Fisher term is inert (hardware-fixed "
            "density); F-Wanda reduces to Wanda for these configs.", pattern)
    _log.info("F-Wanda done (sparsity=%s pattern=%s variant=%s selection=%s; "
              "Fisher active on %d/%d layers)", sparsity, pattern,
              fisher_variant, selection, n_active, len(wrapped))
    return model
