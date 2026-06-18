"""Compare predicted (exp) vs measured (ac) results, plot the deltas, and
flag anomalies.

Naming convention: ``exp`` = expected/predicted, ``ac`` = actual/measured.

Loads each CSV in ``results/tables/exp/`` and looks for a same-named CSV in
``results/tables/`` (the ac/measured results from real SLURM runs). For every
matched pair it:

  1. Joins on (model, method, sparsity, pattern[, nsamples/fisher_variant]).
  2. Plots paired bars (exp vs ac) for each metric: WikiText-2 PPL,
     zero-shot avg, MMLU.
  3. Writes a ``delta_report.csv`` with per-cell deltas + a pass/fail flag
     against the spec's gates:
        - |WikiText-2 PPL ac - exp| <= 0.10
        - |MMLU ac - exp|           <= 0.02 (i.e., 2 pp absolute)
        - |zeroshot_avg ac - exp|   <= 0.015

If no ac file exists yet the script still runs — it produces an exp-only
chart so you can stare at the predictions before the real run lands.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)

_EXP_DIR = os.path.join(_ROOT, "results", "tables", "exp")
_AC_DIR = os.path.join(_ROOT, "results", "tables")
_FIG_DIR = os.path.join(_ROOT, "results", "figures")

_METRICS = [
    ("wikitext2_ppl", "WikiText-2 PPL", False, 0.10),
    ("zeroshot_avg", "Zero-shot avg", True,  0.015),
    ("mmlu",         "MMLU (5-shot)", True,  0.02),
]

_METHOD_ORDER = ["magnitude", "wanda", "sparsegpt", "fwanda", "fsparsegpt"]
_METHOD_COLOR = {
    "magnitude": "#999999",
    "wanda": "#1f77b4",
    "sparsegpt": "#2ca02c",
    "fwanda": "#d62728",
    "fsparsegpt": "#9467bd",
}


def _key_cols(df):
    cols = ["model", "method", "sparsity", "pattern"]
    if "nsamples" in df.columns and df["nsamples"].nunique() > 1:
        cols.append("nsamples")
    if "fisher_variant" in df.columns and df["fisher_variant"].nunique() > 1:
        cols.append("fisher_variant")
    return cols


def _load(path):
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    # numeric coercion (csv read sometimes leaves strings)
    for c in ("wikitext2_ppl", "zeroshot_avg", "mmlu", "sparsity"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _plot_paired(exp, ac, metric, label, higher_is_better, out_path,
                 csv_name):
    rows = []
    keys = _key_cols(exp)
    for _, e in exp.iterrows():
        row = {k: e[k] for k in keys}
        row["exp"] = e[metric]
        if ac is not None:
            mask = np.logical_and.reduce(
                [ac[k] == e[k] for k in keys]) if keys else None
            sub = ac[mask] if mask is not None else ac
            row["ac"] = sub[metric].iloc[-1] if len(sub) else np.nan
        else:
            row["ac"] = np.nan
        rows.append(row)
    plot_df = pd.DataFrame(rows)
    if plot_df["exp"].isna().all():
        return None

    # Build a compact label per row
    def lbl(r):
        bits = [r["model"].split("/")[-1].replace("Llama-2-", "L2-")]
        bits.append(r["method"])
        if "pattern" in r and r["pattern"] != "unstructured":
            bits.append(str(r["pattern"]))
        for k in ("nsamples", "fisher_variant"):
            if k in r and pd.notna(r[k]) and str(r[k]) not in ("", "nan"):
                bits.append(f"{k}={r[k]}")
        return " | ".join(bits)
    plot_df["label"] = plot_df.apply(lbl, axis=1)

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(plot_df)), 4))
    x = np.arange(len(plot_df))
    w = 0.4
    ax.bar(x - w/2, plot_df["exp"], w, label="exp (predicted)",
           color="lightgray", edgecolor="black")
    has_ac = plot_df["ac"].notna().any()
    if has_ac:
        colors = [_METHOD_COLOR.get(m, "#444") for m in plot_df["method"]]
        ax.bar(x + w/2, plot_df["ac"], w, label="ac (measured)",
               color=colors, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(label)
    arrow = "↑" if higher_is_better else "↓"
    ax.set_title(f"{label}  ({arrow} better)   [{csv_name}]", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return plot_df


def _process_csv(name):
    exp_path = os.path.join(_EXP_DIR, name)
    ac_path = os.path.join(_AC_DIR, name)
    exp = _load(exp_path)
    ac = _load(ac_path)
    if exp is None:
        return None
    print(f"[{name}] exp: {len(exp)} rows; "
          f"ac: {'missing' if ac is None else f'{len(ac)} rows'}")

    stem = os.path.splitext(name)[0]
    deltas = []
    for metric, label, higher, tol in _METRICS:
        if metric not in exp.columns or exp[metric].notna().sum() == 0:
            continue
        out = os.path.join(_FIG_DIR, f"compare_{stem}_{metric}.pdf")
        plot_df = _plot_paired(exp, ac, metric, label, higher, out, name)
        if plot_df is None:
            continue
        print(f"  wrote {out}")
        if plot_df["ac"].notna().any():
            for _, r in plot_df.iterrows():
                if pd.notna(r["ac"]):
                    d = r["ac"] - r["exp"]
                    flag = "OK" if abs(d) <= tol else "WARN"
                    deltas.append(dict(csv=name, metric=metric,
                                       label=r["label"],
                                       exp=r["exp"],
                                       ac=r["ac"], delta=d,
                                       tolerance=tol, flag=flag))
    return deltas


def main(out_csv: str):
    os.makedirs(_FIG_DIR, exist_ok=True)
    all_deltas = []
    for name in sorted(os.listdir(_EXP_DIR)):
        if not name.endswith(".csv"):
            continue
        d = _process_csv(name)
        if d:
            all_deltas.extend(d)
    if all_deltas:
        df = pd.DataFrame(all_deltas)
        df.to_csv(out_csv, index=False)
        print(f"\nWrote delta report: {out_csv}")
        warns = df[df["flag"] == "WARN"]
        if len(warns):
            print(f"\n{len(warns)} cell(s) outside tolerance:")
            print(warns.to_string(index=False))
        else:
            print("\nAll ac (measured) cells within tolerance.")
    else:
        print("\nNo ac results matched exp files yet — "
              "exp-only plots were produced.")


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(_AC_DIR,
                                                 "delta_report.csv"))
    main(p.parse_args().out)


if __name__ == "__main__":
    cli()
