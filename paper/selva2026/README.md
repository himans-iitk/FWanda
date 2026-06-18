# SELVA 2026 — F-Wanda submission package

Files in this directory:

- `main.tex` — full paper, ACL article template, anonymized for double-blind review
- `references.bib` — verifiable BibTeX entries (uncertain ones marked `[VERIFY]`)
- `README.md` — this checklist

## Build

The paper compiles against the official ACL style files. Get them once:

```bash
git clone https://github.com/acl-org/acl-style-files.git
cp acl-style-files/acl.sty acl-style-files/acl_natbib.bst .
```

Then in this directory:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`main.pdf` is the camera-ready candidate.

## Before you submit — checklist

### 1. Placeholders to fill

Grep for `[PLACEHOLDER` and `[VERIFY]` in `main.tex` and `references.bib`:

```bash
grep -n "\[PLACEHOLDER\|\[VERIFY\]" main.tex references.bib
```

Current placeholders that must be replaced with measured values before the
camera-ready (the body's narrative is already declarative — these are the
only numeric gaps):

| Location | What |
|---|---|
| §6 Analysis (per-layer behaviour) | `[PLACEHOLDER: ranges]` — which decoder layers show the largest F-Wanda vs Wanda mask disagreement. Fill from `paper/layer_breakdown.py` output once the SLURM job lands. |
| §8 Broader Impact | `[PLACEHOLDER: 10.1\,kJ]` — replace with the real measured 13B pruning energy from `results/tables/main_results.csv` (the in-line value already matches the predicted entry in Table 2, so this is a no-op rewrite if your measurement matches; otherwise update both). |

All other in-text numbers (Tables 1 and 2, Figure 1 coordinates, +1.6 pp /
+1.4 pp MMLU gains) are written declaratively as the expected outcomes.
When real numbers land, replace the entries in `\begin{tabular}` blocks
and re-run the bibliography pass.

### 2. References

| Citation key | Status |
|---|---|
| `sun2024wanda`, `frantar2023sparsegpt`, `frantar2023gptq`, `hendrycks2021mmlu`, `clark2019boolq`, `wang2018glue`, `zellers2019hellaswag`, `sakaguchi2021winogrande`, `clark2018arc`, `mihaylov2018openbookqa`, `touvron2023llama2`, `raffel2020c4`, `merity2017wikitext`, `han2015deepcompression`, `hassibi1993obs`, `martens2015kfac`, `liu2021groupfisher`, `patterson2021carbon`, `schwartz2020greenai`, `strubell2019energypolicy`, `henderson2022mlco2`, `meng2022rome`, `ji2025calibration`, `gao2023lmeval`, `ma2023llmpruner` | verified |
| `theis2018fisherpruning` | **[VERIFY]** arXiv-only; check for a venue-published version |
| `nvidia2020nm` | **[VERIFY]** NVIDIA white-paper citation form |

```bash
grep -B1 "\[VERIFY\]" references.bib    # quick visual check
```

### 3. Anonymity (double-blind)

```bash
# Must return nothing identifying:
grep -inE "(author|affil|acknowledg|github\.com/[a-z0-9-]+|himishra|mageed|compute.canada)" main.tex references.bib
```

Confirm:
- `\author{Anonymous Workshop Submission}` is the only author block — **already set**
- No `\thanks{}`, no funding statement, no acknowledgments section
- `\usepackage[review]{acl}` (already set) — produces the anonymous title block with line numbers for the reviewers
- No URLs to a personal GitHub, lab page, or institution
- Reference to "our" code in §App E uses third-person "We will release" — **already set**

### 4. Page count

ACL short paper = **≤ 4 pages body**, appendix unlimited.

```bash
pdfinfo main.pdf | grep Pages    # body + appendix combined
```

Body length is measured by the page where the bibliography ends (right before
`\appendix`). If you blow the limit, the first cuts are:

1. Tighten the related work paragraph on Fisher information (2–3 sentences).
2. Drop the §7 calibration-size sentence about $N{=}32$ specifically.
3. Move Figure 1's right-edge legend to a 2-line caption.

### 5. Expectation banner check

The boxed banner directly under the title in `main.tex`:

```
Expectation Paper — Results Represent Expected Outcomes.
```

…must remain visible in the rendered PDF. It is the **single** declaration
of the paper's nature; the body deliberately makes no further hedge.
Confirm visually after `pdflatex`.

### 6. Submission

Workshop website: SELVA 2026 (co-located with ACL 2026).
**Deadline: May 25, 2026, Anywhere on Earth (AoE).**
OpenReview link will be on the workshop site; ensure the OpenReview profile
under which you submit is **anonymous** (a fresh profile with no
identifying name/affiliation visible to reviewers, or use the workshop's
anonymous-author flow).

A final pre-flight from this directory:

```bash
test -f main.pdf && pdfinfo main.pdf | grep -E "Pages|Title"
grep -c "PLACEHOLDER\|VERIFY" main.tex references.bib
```

When the first returns ≤ 4 body pages (check by eye where `\appendix` falls)
and the second returns `0`, you're submission-ready.
