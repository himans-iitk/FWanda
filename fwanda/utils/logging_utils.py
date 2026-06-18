"""Lightweight logging + a CSV results appender shared by all experiments."""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
from datetime import datetime, timezone

_LOCK = threading.Lock()


def get_logger(name: str = "fwanda") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


# Stable column order so every run appends compatibly.
RESULT_FIELDS = [
    "timestamp", "model", "method", "sparsity", "pattern",
    "calib_source", "nsamples", "seqlen", "seed",
    "achieved_sparsity", "fisher_variant", "selection", "fisher_active",
    "wikitext2_ppl", "c4_ppl",
    "boolq", "rte", "hellaswag", "winogrande",
    "arc_easy", "arc_challenge", "openbookqa", "zeroshot_avg",
    "mmlu", "extra_json",
]


def append_result(row: dict, csv_path: str) -> None:
    """Append one result row to ``csv_path`` (creating it with a header).

    Unknown keys are folded into an ``extra_json`` column so nothing is lost.
    """
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    row = dict(row)
    row.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    extra = {k: row.pop(k) for k in list(row) if k not in RESULT_FIELDS}
    if extra:
        merged = {}
        if row.get("extra_json"):
            try:
                merged.update(json.loads(row["extra_json"]))
            except (ValueError, TypeError):
                pass
        merged.update(extra)
        row["extra_json"] = json.dumps(merged, sort_keys=True)
    with _LOCK:
        exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_FIELDS,
                               extrasaction="ignore")
            if not exists:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in RESULT_FIELDS})
