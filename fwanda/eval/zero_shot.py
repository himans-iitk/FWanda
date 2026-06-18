"""Zero-shot accuracy on the Wanda task suite via ``lm-evaluation-harness``.

Pin lm-eval to ``0.4.2`` — newer versions changed the simple_evaluate API and
the metric key from ``acc`` to ``acc,none``.
"""

from __future__ import annotations

DEFAULT_TASKS = [
    "boolq",
    "rte",
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
]


def evaluate_zero_shot(model, tokenizer, tasks=None, batch_size: int = 4,
                       num_fewshot: int = 0):
    tasks = list(tasks) if tasks is not None else DEFAULT_TASKS
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    out = simple_evaluate(model=lm, tasks=tasks, num_fewshot=num_fewshot)
    return out["results"]
