"""Compute-cost bar charts and a quality-vs-cost Pareto scatter.

Naming convention: ``exp`` = expected/predicted, ``ac`` = actual/measured.

Three figures:
    cost_pruning_time.pdf  — pruning wall-clock per method per model
    cost_gpu_memory.pdf    — peak GPU memory per method per model
    pareto_mmlu_vs_cost.pdf — MMLU (higher = better) vs pruning-time (lower = better)

Reads exp costs from ``results/tables/exp/compute_cost.csv`` and exp MMLU
from ``results/tables/exp/main_results.csv``. When a real (ac) results CSV
``results/tables/main_results.csv`` exists it is overlaid on the Pareto plot.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
_EXP = os.path.join(_ROOT, "results", "tables", "exp")
_FIG = os.path.join(_ROOT, "results", "figures")
os.makedirs(_FIG, exist_ok=True)

_METHOD_ORDER = ["magnitude", "wanda", "sparsegpt", "fwanda", "fsparsegpt"]
_COLOR = {
    "magnitude": "#999999",
    "wanda": "#1f77b4",
    "sparsegpt": "#2ca02c",
    "fwanda": "#d62728",
    "fsparsegpt": "#9467bd",
}


def _bar_by_method(df, value_col, ylabel, title, out):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    models = sorted(df["model"].unique())
    methods = [m for m in _METHOD_ORDER if m in df["method"].unique()]
    x = np.arange(len(methods))
    w = 0.4
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("method").reindex(methods)
        offset = (i - (len(models) - 1) / 2) * w
        ax.bar(x + offset, sub[value_col], w,
               label=model.split("/")[-1].replace("Llama-2-", "L2-"),
               color=[_COLOR[m] for m in methods],
               edgecolor="black",
               hatch="" if i == 0 else "//")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def _pareto(cost_df, qual_df, out_path, ac_df=None):
    """Scatter pruning-time (x) vs MMLU (y) per method, both models."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    # restrict to unstructured 50% — that's where F-Wanda is supposed to win
    q = qual_df[(qual_df["pattern"] == "unstructured") &
                (qual_df["sparsity"].astype(float) == 0.5)]
    merged = q.merge(cost_df, on=["model", "method"])
    for model, g in merged.groupby("model"):
        marker = "o" if "7b" in model.lower() else "s"
        for _, r in g.iterrows():
            ax.scatter(r["pruning_minutes_h100"], r["mmlu"] * 100,
                       s=160, marker=marker, color=_COLOR[r["method"]],
                       edgecolor="black", zorder=3, alpha=0.85,
                       label=f"{model.split('/')[-1].replace('Llama-2-','L2-')} · {r['method']}")
            ax.annotate(r["method"], (r["pruning_minutes_h100"], r["mmlu"] * 100),
                        textcoords="offset points", xytext=(7, 5), fontsize=8)
    if ac_df is not None:
        a = ac_df[(ac_df["pattern"] == "unstructured") &
                      (ac_df["sparsity"].astype(float) == 0.5)]
        if len(a):
            am = a.merge(cost_df, on=["model", "method"])
            for _, r in am.iterrows():
                ax.scatter(r["pruning_minutes_h100"], r["mmlu"] * 100,
                           s=160, marker="x", color="black", linewidths=2,
                           zorder=4)
    ax.set_xlabel("Pruning wall-clock (min, H100)  ↓ better")
    ax.set_ylabel("MMLU 5-shot (%)  ↑ better")
    ax.set_title("Quality vs. cost  (50% unstructured)", fontsize=10)
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    cost = pd.read_csv(os.path.join(_EXP, "compute_cost.csv"))
    main_exp = pd.read_csv(os.path.join(_EXP, "main_results.csv"))

    _bar_by_method(
        cost, "pruning_minutes_h100",
        "Pruning wall-clock (min, H100)  ↓ better",
        "Predicted pruning cost per method",
        os.path.join(_FIG, "cost_pruning_time.pdf"))

    _bar_by_method(
        cost, "peak_gpu_gb",
        "Peak GPU memory during pruning (GB)  ↓ better",
        "Predicted peak GPU memory per method",
        os.path.join(_FIG, "cost_gpu_memory.pdf"))

    ac = None
    ac_path = os.path.join(_ROOT, "results", "tables", "main_results.csv")
    if os.path.isfile(ac_path):
        ac = pd.read_csv(ac_path)
        ac["mmlu"] = pd.to_numeric(ac["mmlu"], errors="coerce")
        ac["sparsity"] = pd.to_numeric(ac["sparsity"], errors="coerce")
        print("  overlaying ac (measured) points (black 'x')")

    _pareto(cost, main_exp,
            os.path.join(_FIG, "pareto_mmlu_vs_cost.pdf"),
            ac_df=ac)


if __name__ == "__main__":
    main()
