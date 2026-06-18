"""Pruning methods: magnitude, wanda (baseline), sparsegpt (baseline),
fwanda (ours), fsparsegpt (ours, appendix)."""

from fwanda.methods import fsparsegpt, fwanda, magnitude, sparsegpt, wanda

__all__ = ["magnitude", "wanda", "sparsegpt", "fwanda", "fsparsegpt"]
