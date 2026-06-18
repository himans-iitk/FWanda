"""Fisher-allocated per-row budget — the heart of F-Wanda.

These tests pin the invariants the prune loop depends on:
  - the global keep-count is preserved exactly (modulo feasibility floor),
  - rows with larger Fisher weight get >= keep than smaller ones,
  - per-row counts are in [min_keep, d_in],
  - and when all rows have equal weight, allocation degenerates to Wanda's
    uniform per-row keep count (the no-Fisher limit).
"""

import pytest

torch = pytest.importorskip("torch")

from fwanda.utils.sparsity import (allocate_row_budget,
                                    generate_fisher_mask,
                                    global_mask,
                                    mask_from_row_budget)


def test_budget_sums_to_target():
    d_out, d_in = 32, 256
    for sparsity in [0.2, 0.5, 0.7]:
        w = torch.rand(d_out).abs() + 0.01
        keep = allocate_row_budget(w, sparsity, d_in)
        expected = int(round((1 - sparsity) * d_out * d_in))
        # min_keep floor may bump the total up slightly when infeasible;
        # otherwise it should be exact.
        assert int(keep.sum()) == max(expected, d_out)


def test_budget_bounds():
    d_in = 128
    w = torch.rand(16).abs() + 0.01
    keep = allocate_row_budget(w, 0.5, d_in, min_keep=1)
    assert (keep >= 1).all()
    assert (keep <= d_in).all()


def test_budget_monotone_in_weight():
    """Bigger Fisher → kept >= smaller Fisher (under the same target).

    The per-row floor and integer rounding can tie adjacent rows, so we
    assert weak monotonicity (kept[order_by_w]) is non-decreasing in w.
    """
    d_in = 256
    w = torch.linspace(0.01, 1.0, 32)
    keep = allocate_row_budget(w, 0.5, d_in)
    order = torch.argsort(w)
    sorted_keep = keep[order]
    diffs = sorted_keep[1:] - sorted_keep[:-1]
    assert (diffs >= 0).all()


def test_budget_uniform_when_weights_equal():
    """If all rows are equally important, F-Wanda budget == Wanda uniform."""
    d_out, d_in, sparsity = 16, 128, 0.5
    w = torch.ones(d_out)
    keep = allocate_row_budget(w, sparsity, d_in)
    assert int(keep.max() - keep.min()) <= 1  # rounding leeway


def test_mask_from_row_budget_keeps_top_per_row():
    g = torch.Generator().manual_seed(0)
    score = torch.rand(8, 64, generator=g)
    keep = torch.tensor([10, 20, 30, 40, 8, 50, 16, 32])
    mask = mask_from_row_budget(score, keep)
    for r in range(8):
        assert int(mask[r].sum()) == int(keep[r])
        kept = score[r][mask[r]]
        dropped = score[r][~mask[r]]
        assert kept.min() >= dropped.max()


def test_fisher_mask_inert_for_2_4():
    """Strict N:M is hardware-fixed → Fisher cannot influence the mask."""
    score = torch.rand(4, 32)
    omega = torch.rand(4).abs() + 0.01
    mask, info = generate_fisher_mask(score, omega, 0.5, "2:4")
    assert info["fisher_active"] is False
    # Every block of 4 keeps exactly 2.
    blocks = mask.view(4, -1, 4).int().sum(dim=2)
    assert (blocks == 2).all()


def test_fisher_mask_changes_total_per_row_distribution():
    """budget selection: large omega spread yields wide per-row keep spread."""
    score = torch.rand(8, 256)
    omega = torch.tensor([100., 100., 100., 100., 0.01, 0.01, 0.01, 0.01])
    mask, info = generate_fisher_mask(score, omega, 0.5, "unstructured",
                                      selection="budget")
    assert info["fisher_active"]
    per_row = mask.int().sum(dim=1)
    assert per_row[:4].min() > per_row[4:].max()  # high-w rows keep more


def test_budget_accepts_any_device():
    """Regression: ``allocate_row_budget`` must not assume input device.

    Hit by the first F-Wanda smoke run: ``omega_bar`` arrived on CUDA while
    the allocator's working tensors were on CPU; ``alloc[free] = ... w[free]``
    raised a device-mismatch RuntimeError. The fix moves ``w`` to CPU at
    entry. Verify here with a CPU tensor (and with CUDA if available).
    """
    w_cpu = torch.linspace(0.01, 1.0, 32)
    keep_cpu = allocate_row_budget(w_cpu, 0.5, 256)
    assert keep_cpu.device.type == "cpu"
    assert int(keep_cpu.sum()) == int(round((1 - 0.5) * 32 * 256))

    if torch.cuda.is_available():
        w_cuda = w_cpu.cuda()
        keep_cuda = allocate_row_budget(w_cuda, 0.5, 256)
        # Function returns CPU tensor regardless; allow downstream callers
        # (e.g. mask_from_row_budget) to do their own .to(device).
        assert keep_cuda.device.type == "cpu"
        assert torch.equal(keep_cpu, keep_cuda)


def test_global_mask_exact_count():
    """global_mask keeps exactly round((1-s)*numel) weights."""
    g = torch.Generator().manual_seed(3)
    score = torch.rand(16, 64, generator=g)
    for s in [0.25, 0.5, 0.75]:
        m = global_mask(score, s)
        assert int(m.sum()) == int(round((1 - s) * 16 * 64))


def test_global_mask_rows_compete():
    """Under a global threshold, a per-row positive scale on the score DOES
    change the mask — the property that makes F-Wanda-global non-degenerate
    (and that per-row selection lacks)."""
    g = torch.Generator().manual_seed(4)
    base = torch.rand(4, 64, generator=g)
    # Boost two rows 50x; under a global threshold they should keep far more.
    scale = torch.tensor([50., 50., 1., 1.]).view(4, 1)
    m_plain = global_mask(base, 0.5)
    m_scaled = global_mask(base * scale, 0.5)
    assert not torch.equal(m_plain, m_scaled)
    kept = m_scaled.int().sum(dim=1)
    assert kept[:2].min() > kept[2:].max()  # boosted rows win the budget


def test_fisher_global_vs_budget_differ():
    """global and budget selection produce different masks for spread omega.

    Real callers pass the Fisher-scaled score (omega baked in, exactly as
    WrappedFWanda.get_score_matrix returns); mirror that here.
    """
    raw_score = torch.rand(8, 128)
    omega = torch.tensor([100., 100., 50., 50., 1., 1., 0.1, 0.1])
    score = omega.clamp(min=1e-8).sqrt().unsqueeze(1) * raw_score
    m_global, info_g = generate_fisher_mask(score, omega, 0.5, "unstructured",
                                            selection="global")
    m_budget, info_b = generate_fisher_mask(score, omega, 0.5, "unstructured",
                                            selection="budget")
    assert info_g["fisher_active"] and info_b["fisher_active"]
    assert info_g["selection"] == "global"
    assert not torch.equal(m_global, m_budget)
    # Under global selection, high-omega rows should keep more than low-omega.
    kept = m_global.int().sum(dim=1)
    assert kept[:2].float().mean() > kept[6:].float().mean()
