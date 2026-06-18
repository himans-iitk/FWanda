"""Mask correctness — uniform per-row, N:M, and achieved-sparsity tolerance."""

import pytest

torch = pytest.importorskip("torch")

from fwanda.utils.sparsity import (_nm_mask, _unstructured_mask,
                                    generate_mask)


def test_unstructured_keeps_correct_count():
    g = torch.Generator().manual_seed(0)
    score = torch.rand(64, 256, generator=g)
    for sparsity in [0.1, 0.3, 0.5, 0.7]:
        m = _unstructured_mask(score, sparsity)
        assert m.shape == score.shape
        expected_keep_per_row = int(round(256 * (1 - sparsity)))
        for r in range(64):
            assert int(m[r].sum()) == expected_keep_per_row


def test_unstructured_keeps_largest():
    """Kept entries are exactly the top-k by score in each row."""
    g = torch.Generator().manual_seed(1)
    score = torch.rand(8, 32, generator=g)
    m = _unstructured_mask(score, 0.5)
    for r in range(8):
        kept_vals = score[r][m[r]]
        dropped_vals = score[r][~m[r]]
        assert kept_vals.min() >= dropped_vals.max()


@pytest.mark.parametrize("n,m", [(2, 4), (4, 8)])
def test_nm_pattern_density(n, m):
    g = torch.Generator().manual_seed(2)
    score = torch.rand(16, 4 * m, generator=g)
    mask = _nm_mask(score, n, m)
    # Every block of m columns keeps exactly n entries, in every row.
    blocks = mask.view(16, -1, m).int().sum(dim=2)
    assert (blocks == n).all()


def test_generate_mask_dispatch():
    score = torch.rand(4, 16)
    assert generate_mask(score, 0.5, "unstructured").shape == (4, 16)
    assert generate_mask(score, 0.5, "2:4").shape == (4, 16)
    assert generate_mask(score, 0.5, "4:8").shape == (4, 16)
    with pytest.raises(ValueError):
        generate_mask(score, 0.5, "bogus")
