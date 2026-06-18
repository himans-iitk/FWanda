"""Build the paper figures.

Figure 1  — F-Wanda vs Wanda MMLU gain across sparsity levels.
Figure 2  — Layer-wise heatmap of F-Wanda vs Wanda mask disagreement.
Figure 3  — Per-row omega_bar scatter (kept by F-Wanda vs kept by Wanda).

Inputs:
    results/tables/main_results.csv             (Figure 1)
    results/figures/layer_breakdown.npz         (Figures 2, 3)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # SLURM-safe: no display
import matplotlib.pyplot as plt


def _figure1(results_csv, out_path):
    if not os.path.isfile(results_csv):
        print(f"[warn] {results_csv} missing — skip Figure 1")
        return
    df = pd.read_csv(results_csv)
    df = df[df["pattern"] == "unstructured"]
    if df.empty:
        return
    pivot = df.pivot_table(index=["model", "sparsity"], columns="method",
                           values="mmlu", aggfunc="last").reset_index()
    if "wanda" not in pivot.columns or "fwanda" not in pivot.columns:
        return
    pivot["gain_pp"] = (pivot["fwanda"] - pivot["wanda"]) * 100
    fig, ax = plt.subplots(figsize=(5, 3))
    for model, g in pivot.groupby("model"):
        ax.bar([f"{s:.0%}" for s in g["sparsity"]], g["gain_pp"],
               label=model.split("/")[-1])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("MMLU gain (pp)  F-Wanda − Wanda")
    ax.set_xlabel("Sparsity")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _figure2_and_3(npz_path, out_dir):
    if not os.path.isfile(npz_path):
        print(f"[warn] {npz_path} missing — skip Figures 2/3")
        return
    z = np.load(npz_path, allow_pickle=True)
    rows = list(z["rows"])
    if not rows:
        return
    df = pd.DataFrame(rows)

    # Figure 2: heatmap layer x sublinear of disagreement fraction
    sublins = sorted(df["sub"].unique())
    layers = sorted(df["layer"].unique())
    mat = np.zeros((len(layers), len(sublins)))
    for _, r in df.iterrows():
        mat[layers.index(r["layer"]), sublins.index(r["sub"])] = (
            r["mask_disagree_frac"])
    fig, ax = plt.subplots(figsize=(6, max(3, 0.15 * len(layers))))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(sublins)))
    ax.set_xticklabels([s.split(".")[-1] for s in sublins], rotation=45,
                       ha="right", fontsize=7)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=6)
    ax.set_ylabel("Decoder layer")
    fig.colorbar(im, ax=ax, label="Mask disagreement (frac)")
    fig.tight_layout()
    p2 = os.path.join(out_dir, "fig_2.pdf")
    fig.savefig(p2)
    plt.close(fig)
    print(f"  wrote {p2}")

    # Figure 3: omega_bar p10/p90 spread vs Fisher-budget spread
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(np.log10(df["omega_p90"].astype(float)) -
               np.log10(df["omega_p10"].astype(float).clip(lower=1e-12)),
               df["fw_budget_std"], s=12, alpha=0.6)
    ax.set_xlabel(r"$\log_{10}(\omega_{p90}/\omega_{p10})$")
    ax.set_ylabel("F-Wanda per-row keep-count std")
    fig.tight_layout()
    p3 = os.path.join(out_dir, "fig_3.pdf")
    fig.savefig(p3)
    plt.close(fig)
    print(f"  wrote {p3}")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    _figure1(os.path.join(args.results_dir, "main_results.csv"),
             os.path.join(args.out_dir, "fig_1.pdf"))
    _figure2_and_3(os.path.join(args.out_dir, "layer_breakdown.npz"),
                   args.out_dir)


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/tables")
    p.add_argument("--out_dir", default="results/figures")
    main(p.parse_args())


if __name__ == "__main__":
    cli()
