"""Model loading and architecture-agnostic layer discovery."""

from __future__ import annotations

import importlib.util

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from fwanda.utils.logging_utils import get_logger

_log = get_logger("fwanda.model_loader")

_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def load_model(model_name: str, torch_dtype: str = "bfloat16",
                use_flash: bool = False, device_map: str = "cuda"):
    """Load a HF causal LM + tokenizer with the project's standard settings."""
    dtype = _DTYPES[torch_dtype.lower()]
    kwargs = dict(torch_dtype=dtype, device_map=device_map)
    if use_flash and importlib.util.find_spec("flash_attn") is not None:
        kwargs["attn_implementation"] = "flash_attention_2"
    else:
        if use_flash:
            _log.warning(
                "use_flash=True but flash_attn is not installed; "
                "falling back to SDPA attention (memory-efficient, exact).")
        # SDPA (torch>=2.1) uses the fused memory-efficient/flash kernel and
        # does NOT materialise the full T x T attention matrix — essential for
        # the F-Wanda backward pass on 13B, where eager attention OOMs an 80GB
        # H100. SDPA is exact, so results are unaffected.
        kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_decoder_layers(model) -> nn.ModuleList:
    """Return the decoder block list, robust across LLaMA/Mistral/Llama-3.

    All currently targeted architectures expose ``model.model.layers``;
    fall back to a search for the first ``nn.ModuleList`` of blocks.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    for _, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            return module
    raise AttributeError(
        "Could not locate the decoder layer list on this model.")


def find_layers(module, layers=(nn.Linear,), name: str = ""):
    """Recursively collect ``{qualified_name: module}`` for target layer types.

    Adapted in spirit from the Wanda repo's ``find_layers`` so F-Wanda has the
    same code shape for easy diffing.
    """
    if isinstance(module, tuple(layers)):
        return {name: module}
    res = {}
    for child_name, child in module.named_children():
        child_full = f"{name}.{child_name}" if name else child_name
        res.update(find_layers(child, layers=layers, name=child_full))
    return res
