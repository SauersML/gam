"""#1784 — the ibp_map assignment must default its concentration to the K-aware
``default_ibp_concentration_for_k_atoms(K)`` when the caller leaves ``alpha``
unset, instead of the historical fixed ``alpha=1.0`` that collapsed the ordered
stick-breaking prior to a near-hard mask past the first ~3 atoms (which made the
manifold SAE underfit an equal-K linear dictionary and left the K=128 joint
Hessian rank-deficient → ``RemlConvergenceError``).

These tests pin the Python-side *wiring* (which alpha the facade forwards to the
Rust ``sae_manifold_fit_minimal`` kernel) without a real fit, using the same
fake-rust-module pattern as ``test_sae_manifold_ibp_refresh``. The reconstruction
*quality* invariant (manifold EV ≥ linear EV at equal K) is pinned in the Rust
``manifold::tests_ibp_capacity_1784`` unit test against the real solver.
"""
import math

import numpy as np
import pytest

import gamfit
import gamfit._sae_manifold as sae
from test_sae_manifold_ibp_refresh import _FakeRustModule


class _AlphaCapturingRust(_FakeRustModule):
    """Records the ``alpha`` (and assignment kind) the facade forwards."""

    def __init__(self):
        super().__init__()
        self.captured_alpha = None
        self.captured_kind = None
        self.captured_learnable = None

    def sae_manifold_fit_minimal(self, z, atom_basis, atom_dim, alpha, tau,
                                 learnable_alpha, assignment_kind, **kw):
        self.captured_alpha = float(alpha)
        self.captured_kind = str(assignment_kind)
        self.captured_learnable = bool(learnable_alpha)
        return super().sae_manifold_fit_minimal(
            z, atom_basis, atom_dim, alpha, tau, learnable_alpha,
            assignment_kind, **kw)


def _x(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, 2))


def _fit(monkeypatch, *, K, **kwargs):
    fake = _AlphaCapturingRust()
    monkeypatch.setattr(sae, "rust_module", lambda: fake)
    model = sae.sae_manifold_fit(
        _x(3 * K + 1), K=K, atom_topology="circle", d_atom=1,
        assignment="ibp_map", n_iter=1, **kwargs)
    return fake, model


def test_default_ibp_concentration_formula_matches_rust():
    # α = 1/(exp(1/K) − 1), floored at 1.0. Spot-check the boundary + span.
    assert sae._default_ibp_concentration_for_k_atoms(1) == pytest.approx(1.0)
    for K in (2, 8, 64, 128, 512):
        alpha = sae._default_ibp_concentration_for_k_atoms(K)
        # last atom retains prior mass π_{K-1} = (α/(α+1))^K ≈ e^{-1}.
        pi_last = (alpha / (alpha + 1.0)) ** K
        assert pi_last == pytest.approx(math.exp(-1.0), rel=1e-9)
        assert alpha >= 1.0


@pytest.mark.parametrize("K", [64, 128])
def test_ibp_unset_alpha_defaults_to_k_aware(monkeypatch, K):
    fake, model = _fit(monkeypatch, K=K)  # alpha left unset
    expected = sae._default_ibp_concentration_for_k_atoms(K)
    assert fake.captured_kind == "ibp_map"
    assert fake.captured_alpha == pytest.approx(expected)
    assert float(model.alpha) == pytest.approx(expected)
    # The old fixed default would have been 1.0; the K-aware value must be larger.
    assert fake.captured_alpha > 1.0


def test_ibp_explicit_alpha_is_respected(monkeypatch):
    fake, model = _fit(monkeypatch, K=64, alpha=2.5)
    assert fake.captured_alpha == pytest.approx(2.5)
    assert float(model.alpha) == pytest.approx(2.5)


def test_ibp_alpha_override_leaves_base_at_one(monkeypatch):
    # A per-fit ibp_alpha override drives the concentration in Rust, so the base
    # alpha the facade forwards stays the historical 1.0 (no K-aware bump).
    fake, _ = _fit(monkeypatch, K=64, ibp_alpha=3.0)
    assert fake.captured_alpha == pytest.approx(1.0)


def test_ibp_auto_alpha_starts_at_one(monkeypatch):
    # Learnable alpha ("auto") seeds at 1.0 and is refined by the solver; the
    # K-aware default only moves the fixed seed.
    fake, model = _fit(monkeypatch, K=128, alpha="auto")
    assert fake.captured_learnable is True
    assert fake.captured_alpha == pytest.approx(1.0)
    assert bool(model.learnable_alpha) is True
