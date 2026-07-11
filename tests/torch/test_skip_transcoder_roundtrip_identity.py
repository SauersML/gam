from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
torch = pytest.importorskip("torch")
from gamfit.torch.skip_transcoder import SkipAffineSmooth


def test_skip_transcoder_roundtrip_identity() -> None:
    torch.manual_seed(5)
    smooth = SkipAffineSmooth(
        in_dim=8, out_dim=8, n_atoms=8, rank_skip=0, activation_threshold=1e-6
    )
    x = torch.randn(20, 8, dtype=torch.float32)
    z = smooth.code(x)
    y, _ = smooth(x)
    assert y.shape == x.shape, "Expected skip-transcoder decode output to have the same shape as the encoded input."
    assert torch.isfinite(z).all() and torch.isfinite(y).all(), "Expected skip-transcoder encode and decode tensors to remain finite through a round-trip."
