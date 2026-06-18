"""Perplexity on WikiText-2 and C4 — matches the Wanda/SparseGPT protocol.

Non-overlapping ``seq_len`` windows; chunk losses aggregated as
``exp(sum(nll) / total_tokens)``.
"""

from __future__ import annotations

import torch
from datasets import load_dataset
from tqdm import tqdm


@torch.no_grad()
def _ppl_from_chunks(model, enc, seq_len: int, desc: str):
    n_chunks = enc.shape[1] // seq_len
    if n_chunks == 0:
        raise ValueError(
            f"Tokenized eval text is shorter than seq_len={seq_len}")
    nlls = []
    for i in tqdm(range(n_chunks), desc=desc):
        chunk = enc[:, i * seq_len: (i + 1) * seq_len].to(model.device)
        loss = model(chunk, labels=chunk).loss
        nlls.append(loss.float() * seq_len)
    return torch.exp(torch.stack(nlls).sum() / (n_chunks * seq_len)).item()


@torch.no_grad()
def evaluate_wikitext2(model, tokenizer, seq_len: int = 2048) -> float:
    model.eval()
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(testdata["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids
    return _ppl_from_chunks(model, enc, seq_len, "WikiText-2 PPL")


@torch.no_grad()
def evaluate_c4(model, tokenizer, seq_len: int = 2048,
                n_val_samples: int = 256) -> float:
    """C4 validation PPL — same shard pin as the calibration loader."""
    model.eval()
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation")
    parts = []
    total = 0
    for row in valdata:
        ids = tokenizer(row["text"], return_tensors="pt").input_ids
        parts.append(ids)
        total += ids.shape[1]
        if total >= n_val_samples * seq_len:
            break
    enc = torch.cat(parts, dim=1)[:, : n_val_samples * seq_len]
    return _ppl_from_chunks(model, enc, seq_len, "C4 PPL")
