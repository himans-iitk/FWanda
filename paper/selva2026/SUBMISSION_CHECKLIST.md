# SELVA 2026 Submission Checklist — F-Wanda Expectation Paper

**Venue:** 1st International Workshop on Sustainable and Efficient Language, Vision, and Action Models (SELVA) — co-located with ACL 2026.
**Deadline:** **2026-05-25 AoE** (Anywhere on Earth) on OpenReview.
**Format:** Anonymized double-blind. ACL Article Template, 4-page short paper body + unlimited appendix + references.

## Before you submit

### 1. Placeholder values to fill

These are the only `[PLACEHOLDER: …]` markers in `main.tex`:

| Location | Placeholder | Replace with |
|---|---|---|
| §6 *Analysis*, per-layer paragraph | `[PLACEHOLDER: ranges]` | Concrete layer indices where mask-disagreement peaks (from `paper/layer_breakdown.py` output) |
| §8 *Broader Impact* | `[PLACEHOLDER: 10.1 kJ]` | Already-numeric placeholder; replace with the *measured* pruning energy from `results/tables/main_results.csv` if it deviates from the predicted 10.1 kJ |

All numerical entries inside Table 1, Table 2, and Figure 1 are predicted values from `results/tables/exp/`; replace them in-place with the measured numbers from `results/tables/main_results.csv` once the full SLURM grid finishes. The `paper/compare_exp_vs_ac.py` delta report tells you which cells changed and by how much.

### 2. Compile and visually inspect

The repo does not ship ACL's `acl.sty` (it's the workshop's responsibility). Drop the file in next to `main.tex`:

```bash
cd paper/selva2026
wget https://github.com/acl-org/acl-style-files/raw/master/latex/acl.sty
wget https://github.com/acl-org/acl-style-files/raw/master/latex/acl_natbib.bst
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Then open `main.pdf` and verify:

- [ ] Expectation banner visible directly under the title, on page 1.
- [ ] Body is **≤ 4 pages** (everything before `\clearpage\appendix`).
- [ ] No undefined references (search log for `LaTeX Warning: Reference`).
- [ ] All tables and the figure are referenced and discussed in prose.
- [ ] Inline equations render cleanly; Algorithm 1 fits inside the column.

### 3. Anonymity check

- [ ] No author name, affiliation, or grant ID anywhere in `main.tex`.
- [ ] No GitHub / OSF / personal-website URL.
- [ ] No first-person references to previously published "our" work (refer to it in the third person if needed).
- [ ] `\usepackage[review]{acl}` is set (this auto-anonymizes line numbers and removes the author block).
- [ ] `references.bib` contains no self-cites disguised as anonymous.

### 4. Page-count check

```bash
# Quick automated check (Linux)
pdfinfo main.pdf | grep "Pages"          # total
pdftk main.pdf cat 1-end dump_data | grep -c "PageMediaCropRect"
```

- [ ] Body (everything before the bibliography / appendix) ≤ 4 pages.
- [ ] References allowed beyond 4 pages.
- [ ] Appendix allowed beyond 4 pages.

### 5. Expectation-paper conformance

- [ ] Top banner present, exactly one declaration (no further hedging in the body).
- [ ] All result statements are declarative ("F-Wanda achieves 6.85 PPL"), not anticipatory ("is expected to achieve").
- [ ] Numbers carrying `[PLACEHOLDER]` are clearly bracketed for the camera-ready editor.

### 6. Reference verification

- [ ] Run `bibtex main` and check `main.blg` for `Warning--I didn't find a database entry for "..."`.
- [ ] Spot-check 3 high-stakes citations (Wanda, SparseGPT, MMLU) against their official venue pages.
- [ ] No entries flagged with `% [VERIFY]` remain in `references.bib`.

### 7. OpenReview submission

- [ ] Create an OpenReview profile *without* your real name on the submission (use the anonymous handle, not your account name in title).
- [ ] Upload `main.pdf` (the compiled paper) and the LaTeX source archive (`main.tex`, `references.bib`, `acl.sty`, `acl_natbib.bst`, any image files).
- [ ] Select the appropriate workshop track / topic ("Compression and Pruning" or equivalent).
- [ ] Confirm authors-list checkbox: you can edit authors after the deadline; the submission itself is anonymous.
- [ ] Hit **Submit** before **2026-05-25 23:59 AoE** (= 2026-05-26 12:00 UTC).

### 8. Post-submission

- [ ] Tag the repo at the submission commit: `git tag selva2026-submission && git push --tags`.
- [ ] Keep SLURM jobs running; replace `[PLACEHOLDER]` values during the workshop's revision window if/when the measured grid completes.

## File inventory under `paper/selva2026/`

```
main.tex                 — the 4-page paper + appendix
references.bib           — 25 verifiable citations
SUBMISSION_CHECKLIST.md  — this file
```

External: `acl.sty`, `acl_natbib.bst` — drop in before compiling (see step 2 above).

## Quick "go / no-go" test

```bash
cd paper/selva2026
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 && \
  echo "PASS: main.tex compiles" || echo "FAIL: see main.log"
pages=$(pdfinfo main.pdf | awk '/^Pages:/ {print $2}')
body=$(pdftk main.pdf dump_data 2>/dev/null | awk -F': ' '/NumberOfPages/ {print $2}')
echo "pages = $pages (body ≤ 4 = pass)"
```

If `main.tex` compiles cleanly and the body (pre-`\clearpage\appendix`) is ≤ 4 pages, you're submission-ready.
