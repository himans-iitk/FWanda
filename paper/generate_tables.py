"""Build the paper's LaTeX tables from ``results/tables/main_results.csv``.

Tables produced:
    Table 1  — main results (PPL, zero-shot avg, MMLU) per (model, sparsity).
    Table 2  — per-task zero-shot breakdown.
    Table 3  — calibration-size ablation.
    Table 4  — Fisher-variant ablation.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

_ZSTASKS = ["boolq", "rte", "hellaswag", "winogrande",
            "arc_easy", "arc_challenge", "openbookqa"]
_METHOD_PRETTY = {
    "magnitude": "Magnitude",
    "wanda": "Wanda",
    "sparsegpt": "SparseGPT",
    "fwanda": r"\textbf{F-Wanda (ours)}",
    "fsparsegpt": "F-SparseGPT",
}
_METHOD_ORDER = ["magnitude", "wanda", "sparsegpt", "fwanda", "fsparsegpt"]


def _fmt(x, prec=2):
    if pd.isna(x) or x == "":
        return "--"
    return f"{float(x):.{prec}f}"


def _latex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def _table1(df, out_dir):
    rows = []
    for model in sorted(df["model"].unique()):
        for pat in ["unstructured", "2:4"]:
            sub = df[(df["model"] == model) & (df["pattern"] == pat) &
                    (df["sparsity"].astype(float) == 0.5)]
            if sub.empty:
                continue
            for m in _METHOD_ORDER:
                r = sub[sub["method"] == m]
                if r.empty:
                    continue
                r = r.iloc[-1]
                rows.append({
                    "Model": _latex_escape(model.split("/")[-1]),
                    "Pattern": pat,
                    "Method": _METHOD_PRETTY[m],
                    "WikiText-2 PPL": _fmt(r["wikitext2_ppl"]),
                    "Zero-shot avg": _fmt(r["zeroshot_avg"], 3),
                    "MMLU": _fmt(r["mmlu"], 3),
                })
    if not rows:
        return
    tbl = pd.DataFrame(rows)
    tex = tbl.to_latex(index=False, escape=False, column_format="llcrrr",
                       caption="Main results.", label="tab:main")
    path = os.path.join(out_dir, "table_1.tex")
    open(path, "w").write(tex)
    print(f"  wrote {path}")


def _table2(df, out_dir):
    rows = []
    for model in sorted(df["model"].unique()):
        for pat in ["unstructured", "2:4"]:
            sub = df[(df["model"] == model) & (df["pattern"] == pat) &
                    (df["sparsity"].astype(float) == 0.5)]
            if sub.empty:
                continue
            for m in _METHOD_ORDER:
                r = sub[sub["method"] == m]
                if r.empty:
                    continue
                r = r.iloc[-1]
                d = {"Model": _latex_escape(model.split("/")[-1]),
                     "Pattern": pat, "Method": _METHOD_PRETTY[m]}
                for t in _ZSTASKS:
                    d[t] = _fmt(r.get(t), 3)
                rows.append(d)
    if not rows:
        return
    tbl = pd.DataFrame(rows)
    tex = tbl.to_latex(index=False, escape=False,
                       caption="Per-task zero-shot.", label="tab:zeroshot")
    path = os.path.join(out_dir, "table_2.tex")
    open(path, "w").write(tex)
    print(f"  wrote {path}")


def _ablation_table(df, by, sort_key, out_path, caption, label):
    if df.empty:
        return
    df = df.sort_values(sort_key)
    cols = [by, "wikitext2_ppl", "zeroshot_avg", "mmlu"]
    tex = df[cols].to_latex(index=False, escape=False,
                            caption=caption, label=label)
    open(out_path, "w").write(tex)
    print(f"  wrote {out_path}")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    main_path = os.path.join(args.results_dir, "main_results.csv")
    if os.path.isfile(main_path):
        df = pd.read_csv(main_path)
        _table1(df, args.out_dir)
        _table2(df, args.out_dir)
    else:
        print(f"[warn] {main_path} not found — skipping Tables 1/2")

    cs = os.path.join(args.results_dir, "ablation_calib_size.csv")
    if os.path.isfile(cs):
        _ablation_table(pd.read_csv(cs), by="nsamples", sort_key="nsamples",
                        out_path=os.path.join(args.out_dir, "table_3.tex"),
                        caption="Calibration size ablation.",
                        label="tab:calib")

    fv = os.path.join(args.results_dir, "ablation_fisher_variants.csv")
    if os.path.isfile(fv):
        _ablation_table(pd.read_csv(fv), by="fisher_variant",
                        sort_key="fisher_variant",
                        out_path=os.path.join(args.out_dir, "table_4.tex"),
                        caption="Fisher variant ablation.",
                        label="tab:fisher")


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/tables")
    p.add_argument("--out_dir", default="results/tables")
    main(p.parse_args())


if __name__ == "__main__":
    cli()
