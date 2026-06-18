"""Calibration data loading.

Replicates the SparseGPT / Wanda C4 sampling protocol *exactly* — this is
essential for matching their published baseline numbers. The robustness
ablation (Section 7.4) additionally supports Wikipedia and SlimPajama
sources behind the same interface.
"""

from __future__ import annotations

import random

import torch
from datasets import load_dataset


def get_c4_calibration(tokenizer, n_samples: int = 128, seq_len: int = 2048,
                       seed: int = 0) -> torch.Tensor:
    """Sample ``n_samples`` sequences of ``seq_len`` tokens from the C4 train set.

    Matches the SparseGPT/Wanda protocol: a uniformly random document is drawn,
    rejected if shorter than ``seq_len`` tokens, and a random contiguous window
    of exactly ``seq_len`` tokens is taken.

    Returns a LongTensor of shape ``(n_samples, seq_len)``.
    """
    random.seed(seed)

    # The original allenai/c4 'en' config is large/gated; pin to the exact
    # single shard that Wanda/SparseGPT use so token statistics match.
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )

    samples = []
    n_docs = len(traindata)
    while len(samples) < n_samples:
        i = random.randint(0, n_docs - 1)
        text = traindata[i]["text"]
        enc = tokenizer(text, return_tensors="pt")
        if enc.input_ids.shape[1] <= seq_len:
            continue
        start = random.randint(0, enc.input_ids.shape[1] - seq_len - 1)
        sample = enc.input_ids[:, start: start + seq_len]
        samples.append(sample)

    return torch.cat(samples, dim=0)  # (n_samples, seq_len)


def _get_generic_calibration(hf_path, hf_config, text_key, tokenizer,
                             n_samples, seq_len, seed):
    """Same rejection-sampling protocol over an arbitrary text corpus.

    Used by the calibration-source robustness ablation (Wikipedia / SlimPajama).
    """
    random.seed(seed)
    if hf_config is not None:
        data = load_dataset(hf_path, hf_config, split="train")
    else:
        data = load_dataset(hf_path, split="train")

    samples = []
    n_docs = len(data)
    tries = 0
    while len(samples) < n_samples:
        tries += 1
        if tries > 200 * n_samples:
            raise RuntimeError(
                f"Could not find {n_samples} docs >= {seq_len} tokens in "
                f"{hf_path}; corpus may be too short.")
        i = random.randint(0, n_docs - 1)
        text = data[i][text_key]
        enc = tokenizer(text, return_tensors="pt")
        if enc.input_ids.shape[1] <= seq_len:
            continue
        start = random.randint(0, enc.input_ids.shape[1] - seq_len - 1)
        samples.append(enc.input_ids[:, start: start + seq_len])

    return torch.cat(samples, dim=0)


_SOURCES = {
    "wikipedia": dict(hf_path="wikipedia", hf_config="20220301.en",
                      text_key="text"),
    "slimpajama": dict(hf_path="DKYoon/SlimPajama-6B", hf_config=None,
                       text_key="text"),
}


def get_calibration(source: str, tokenizer, n_samples: int = 128,
                    seq_len: int = 2048, seed: int = 0) -> torch.Tensor:
    """Dispatch to the requested calibration corpus.

    ``source`` is one of ``c4`` (default / paper setting), ``wikipedia``,
    ``slimpajama``.
    """
    source = source.lower()
    if source == "c4":
        return get_c4_calibration(tokenizer, n_samples, seq_len, seed)
    if source in _SOURCES:
        return _get_generic_calibration(
            tokenizer=tokenizer, n_samples=n_samples, seq_len=seq_len,
            seed=seed, **_SOURCES[source])
    raise ValueError(
        f"Unknown calibration source {source!r}; "
        f"expected one of: c4, {', '.join(_SOURCES)}")
