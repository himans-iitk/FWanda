"""Hooks that collect per-layer statistics during calibration.

``WrappedWanda``  — forward only: ``||X_j||`` per input column (baseline).
``WrappedFWanda`` — adds a backward hook collecting per-token output-gradients
                    to form the per-row Fisher scalar ``omega_bar_i``.

Both keep Wanda's running-mean accumulation shape so the F-Wanda code diffs
cleanly against the original repo.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WrappedWanda:
    """Wrap an ``nn.Linear`` to accumulate ``E_t[x_j^2]`` per input column."""

    def __init__(self, layer: nn.Linear, device: torch.device):
        self.layer = layer
        self.device = device
        d_in = layer.weight.shape[1]
        self.scaler_row = torch.zeros(d_in, device=device, dtype=torch.float32)
        self.nsamples = 0

    def add_batch_forward(self, inp: torch.Tensor, out: torch.Tensor = None):
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.float()
        n_new = inp.shape[0]
        denom = self.nsamples + n_new
        self.scaler_row *= self.nsamples / denom
        self.scaler_row += inp.pow(2).sum(dim=0) / denom
        self.nsamples += n_new

    def get_input_norms(self) -> torch.Tensor:
        """``sqrt(E_t[x_j^2])`` — a positive per-column scale (Wanda form)."""
        return self.scaler_row.sqrt()

    def get_score_matrix(self) -> torch.Tensor:
        W = self.layer.weight.data.float()
        return W.abs() * self.get_input_norms().unsqueeze(0)

    def free(self):
        self.scaler_row = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class WrappedFWanda(WrappedWanda):
    """Adds per-row Fisher accumulation via a backward hook on the output.

    ``fisher_variant``:
        ``empirical``  — omega_bar_i = mean_t g_{t,i}^2  (default; the
                         empirical Fisher of the LM loss).
        ``mean_only``  — omega_bar_i = (mean_t g_{t,i})^2  (ablation).
        ``true``       — same accumulation as ``empirical``; the *labels* are
                         sampled from the model's own predictive distribution
                         by the calibration loop (true Fisher). The wrapper is
                         identical; only the loss target differs.
    """

    def __init__(self, layer: nn.Linear, device: torch.device,
                 fisher_variant: str = "empirical"):
        super().__init__(layer, device)
        d_out = layer.weight.shape[0]
        self.fisher_variant = fisher_variant
        self.omega_accum = torch.zeros(d_out, device=device,
                                       dtype=torch.float32)
        self.grad_sum = torch.zeros(d_out, device=device, dtype=torch.float32)
        self.grad_tokens = 0
        self.grad_sq_norm_running = 0.0  # for the sanity-check log

    def add_batch_backward(self, grad_out: torch.Tensor):
        """``grad_out`` is dL/dy for y = Wx + b — exactly the g_t we need."""
        if grad_out.dim() == 3:
            grad_out = grad_out.reshape(-1, grad_out.shape[-1])
        grad_out = grad_out.float()
        self.omega_accum += grad_out.pow(2).sum(dim=0)
        self.grad_sum += grad_out.sum(dim=0)
        self.grad_tokens += grad_out.shape[0]
        self.grad_sq_norm_running = float(grad_out.pow(2).sum().sqrt())

    def get_omega_bar(self) -> torch.Tensor:
        """Per-row Fisher scalar, clamped away from zero for stability."""
        n = max(self.grad_tokens, 1)
        if self.fisher_variant == "mean_only":
            omega = (self.grad_sum / n).pow(2)
        else:  # empirical / true
            omega = self.omega_accum / n
        return omega.clamp(min=1e-8)

    def get_score_matrix(self) -> torch.Tensor:
        """S_ij = sqrt(omega_bar_i) * |W_ij| * sqrt(E[x_j^2]).

        The Fisher factor does not reorder within a row (it is a per-row
        constant); it is kept for fidelity and is consumed for real by the
        per-row budget allocator in ``sparsity.generate_fisher_mask``.
        """
        W = self.layer.weight.data.float()
        omega = self.get_omega_bar()
        x_norm = self.get_input_norms()
        return omega.sqrt().unsqueeze(1) * W.abs() * x_norm.unsqueeze(0)

    def free(self):
        self.omega_accum = None
        self.grad_sum = None
        super().free()
