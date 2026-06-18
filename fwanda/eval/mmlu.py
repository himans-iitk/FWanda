"""MMLU 5-shot accuracy via lm-evaluation-harness 0.4.2."""

from __future__ import annotations


def evaluate_mmlu(model, tokenizer, num_fewshot: int = 5,
                  batch_size: int = 4) -> float:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    out = simple_evaluate(model=lm, tasks=["mmlu"], num_fewshot=num_fewshot)
    res = out["results"]["mmlu"]
    return float(res.get("acc,none", res.get("acc")))
