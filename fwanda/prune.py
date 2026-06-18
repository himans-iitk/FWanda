"""F-Wanda pruning entry point.

Example:
    python -m fwanda.prune \\
        --model meta-llama/Llama-2-7b-hf \\
        --method fwanda --sparsity 0.5 --pattern unstructured \\
        --eval_ppl --eval_zero_shot --eval_mmlu \\
        --results_csv results/tables/main_results.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from fwanda.data.calibration import get_calibration
from fwanda.methods import fsparsegpt, fwanda, magnitude, sparsegpt, wanda
from fwanda.utils.logging_utils import append_result, get_logger
from fwanda.utils.model_loader import load_model
from fwanda.utils.sparsity import check_sparsity

_log = get_logger("fwanda.prune")

_METHODS = {
    "magnitude": magnitude.prune,
    "wanda": wanda.prune,
    "sparsegpt": sparsegpt.prune,
    "fwanda": fwanda.prune,
    "fsparsegpt": fsparsegpt.prune,
}


def main(args):
    t0 = time.time()
    _log.info("Loading model %s (dtype=%s, flash=%s)",
              args.model, args.torch_dtype, args.use_flash)
    model, tokenizer = load_model(args.model,
                                  torch_dtype=args.torch_dtype,
                                  use_flash=args.use_flash)

    if args.method != "magnitude":
        _log.info("Sampling calibration: source=%s n=%d seq=%d seed=%d",
                  args.calib_source, args.nsamples, args.seqlen, args.seed)
        calib = get_calibration(args.calib_source, tokenizer,
                                n_samples=args.nsamples,
                                seq_len=args.seqlen, seed=args.seed)
    else:
        calib = None

    method_kwargs = {}
    if args.method in ("fwanda", "fsparsegpt"):
        method_kwargs["fisher_variant"] = args.fisher_variant

    # Selection rule. Default depends on method: F-Wanda uses the global
    # per-layer threshold (so the Fisher term reweights across rows); Wanda
    # uses standard per-row selection. Override explicitly with --selection.
    if args.selection != "auto":
        method_kwargs["selection"] = args.selection
    elif args.method == "fwanda":
        method_kwargs["selection"] = "global"
    elif args.method == "wanda":
        method_kwargs["selection"] = "per_row"

    _log.info("Method=%s sparsity=%s pattern=%s selection=%s", args.method,
              args.sparsity, args.pattern, method_kwargs.get("selection", "-"))
    _METHODS[args.method](model, calib, sparsity=args.sparsity,
                          pattern=args.pattern, **method_kwargs)

    achieved = check_sparsity(model, verbose=True)
    target = args.sparsity
    if abs(achieved - target) > 0.005 and args.pattern == "unstructured":
        _log.warning("Achieved sparsity %.4f differs from target %.4f by >0.5%%",
                     achieved, target)

    result = dict(model=args.model, method=args.method,
                  sparsity=args.sparsity, pattern=args.pattern,
                  calib_source=args.calib_source, nsamples=args.nsamples,
                  seqlen=args.seqlen, seed=args.seed,
                  achieved_sparsity=round(achieved, 5),
                  fisher_variant=args.fisher_variant if args.method in
                  ("fwanda", "fsparsegpt") else "",
                  selection=method_kwargs.get("selection", ""))

    if args.eval_ppl:
        from fwanda.eval.perplexity import evaluate_wikitext2
        ppl = evaluate_wikitext2(model, tokenizer, seq_len=args.eval_seqlen)
        _log.info("WikiText-2 PPL: %.4f", ppl)
        result["wikitext2_ppl"] = round(ppl, 4)
        if args.eval_c4:
            from fwanda.eval.perplexity import evaluate_c4
            c4ppl = evaluate_c4(model, tokenizer, seq_len=args.eval_seqlen)
            _log.info("C4 PPL: %.4f", c4ppl)
            result["c4_ppl"] = round(c4ppl, 4)

    if args.eval_zero_shot:
        from fwanda.eval.zero_shot import evaluate_zero_shot, DEFAULT_TASKS
        zs = evaluate_zero_shot(model, tokenizer, tasks=DEFAULT_TASKS,
                                batch_size=args.eval_batch_size)
        accs = []
        for t in DEFAULT_TASKS:
            v = zs.get(t, {})
            acc = v.get("acc,none", v.get("acc"))
            if acc is not None:
                result[t] = round(float(acc), 4)
                accs.append(float(acc))
        if accs:
            result["zeroshot_avg"] = round(sum(accs) / len(accs), 4)
        _log.info("zero-shot: %s", {k: result.get(k) for k in DEFAULT_TASKS})

    if args.eval_mmlu:
        from fwanda.eval.mmlu import evaluate_mmlu
        mmlu_acc = evaluate_mmlu(model, tokenizer,
                                 num_fewshot=args.mmlu_fewshot,
                                 batch_size=args.eval_batch_size)
        result["mmlu"] = round(float(mmlu_acc), 4)
        _log.info("MMLU (%d-shot): %.4f", args.mmlu_fewshot, mmlu_acc)

    if args.save_dir:
        _log.info("Saving pruned model to %s", args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)

    if args.results_csv:
        append_result(result, args.results_csv)
        _log.info("Appended results -> %s", args.results_csv)
    else:
        print(json.dumps(result, indent=2))

    _log.info("Total wall-clock: %.1fs", time.time() - t0)


def cli():
    p = argparse.ArgumentParser(prog="fwanda.prune")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--method", choices=list(_METHODS), required=True)
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument("--pattern",
                   choices=["unstructured", "2:4", "4:8"],
                   default="unstructured")
    p.add_argument("--calib_source", default="c4",
                   choices=["c4", "wikipedia", "slimpajama"])
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fisher_variant",
                   choices=["empirical", "mean_only", "true"],
                   default="empirical")
    p.add_argument("--selection",
                   choices=["auto", "per_row", "budget", "global"],
                   default="auto",
                   help="mask selection rule. 'auto' => global for fwanda, "
                        "per_row for wanda. 'budget' = F-Wanda per-row Fisher "
                        "budget; 'global' = single per-layer threshold.")
    p.add_argument("--torch_dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--use_flash", action="store_true")
    p.add_argument("--eval_ppl", action="store_true")
    p.add_argument("--eval_c4", action="store_true")
    p.add_argument("--eval_zero_shot", action="store_true")
    p.add_argument("--eval_mmlu", action="store_true")
    p.add_argument("--eval_seqlen", type=int, default=2048)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--mmlu_fewshot", type=int, default=5)
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--results_csv", type=str,
                   default="results/tables/main_results.csv")
    args = p.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
