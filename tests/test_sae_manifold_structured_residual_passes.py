"""Regression tests for the ``structured_residual_passes`` /
``promote_from_residual`` wrapper plumbing.

The native ``sae_manifold_fit_minimal`` pyfunction has long accepted a
``structured_residual_passes`` (int, clamped to ``STRUCTURED_RESIDUAL_PASSES_MAX``)
and ``promote_from_residual`` (bool) opt-in, but the public
``sae_manifold_fit`` wrapper never forwarded them. These tests pin:

* the eager pure-Python validator (``structured_residual_passes < 0`` raises
  ``ValueError`` before any Rust call), and
* that the two kwargs actually reach the native solver and change the fit
  (default ``0`` vs. ``2`` passes + promotion produce different reconstructions).
"""
from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")

from gamfit._sae_manifold import ManifoldSAE, sae_manifold_fit  # noqa: E402


def _planted_circle(n: int = 200, seed: int = 0) -> np.ndarray:
    """A small planted-circle response embedded in a few output channels."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    base = np.column_stack([np.cos(theta), np.sin(theta), 0.5 * np.cos(2.0 * theta)])
    # A fixed random linear lift into 6 output channels plus a little noise so
    # there is genuine structured residual for the extra passes to sculpt.
    lift = rng.standard_normal((3, 6))
    x = base @ lift + 0.05 * rng.standard_normal((n, 6))
    return np.ascontiguousarray(x, dtype=np.float64)


def test_structured_residual_passes_negative_raises() -> None:
    """A negative pass count is rejected eagerly (pure-Python, no Rust call)."""
    x = _planted_circle(n=64)
    with pytest.raises(ValueError, match="structured_residual_passes"):
        sae_manifold_fit(x, K=4, structured_residual_passes=-1)


def test_structured_residual_passes_forwarded_changes_fit() -> None:
    """The default (0 passes) and an opt-in (2 passes + promotion) both fit and
    return a ``ManifoldSAE``, and the residual passes change the result."""
    x = _planted_circle(n=200, seed=1)

    baseline = sae_manifold_fit(
        x, K=4, atom_topology="circle", n_iter=25, random_state=0,
        structured_residual_passes=0,
    )
    sculpted = sae_manifold_fit(
        x, K=4, atom_topology="circle", n_iter=25, random_state=0,
        structured_residual_passes=2, promote_from_residual=True,
    )

    assert isinstance(baseline, ManifoldSAE)
    assert isinstance(sculpted, ManifoldSAE)

    # The residual passes must actually change the learned reconstruction; if the
    # kwargs were dropped on the floor these two fits would be bit-identical.
    recon_base = np.asarray(baseline.fitted)
    recon_sculpt = np.asarray(sculpted.fitted)
    assert recon_base.shape == recon_sculpt.shape
    assert not np.allclose(recon_base, recon_sculpt), (
        "structured_residual_passes=2/promote_from_residual=True produced a "
        "bit-identical fit to the default — the kwargs are not reaching the "
        "native solver."
    )
