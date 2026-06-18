"""Calibration sampler — shape, dtype, determinism (the easy invariants).

Network-dependent tests (actual HF download) are marked ``slow`` and skipped
by default; run them on the SLURM env once before trusting any baseline:

    pytest -m slow tests/test_calibration.py
"""

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.slow
def test_c4_calibration_shape_and_seed():
    transformers = pytest.importorskip("transformers")
    from fwanda.data.calibration import get_c4_calibration
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    a = get_c4_calibration(tok, n_samples=4, seq_len=128, seed=0)
    b = get_c4_calibration(tok, n_samples=4, seq_len=128, seed=0)
    assert a.shape == (4, 128)
    assert torch.equal(a, b), "Same seed must yield identical samples"


def test_dispatch_rejects_unknown_source():
    from fwanda.data.calibration import get_calibration
    with pytest.raises(ValueError):
        get_calibration("not-a-corpus", tokenizer=None, n_samples=1,
                        seq_len=4, seed=0)
