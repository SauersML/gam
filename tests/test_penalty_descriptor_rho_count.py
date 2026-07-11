"""RED tests for issue #226: penalty descriptors must not pass empty rho.

The descriptors in ``gamfit/_penalty_descriptors.py`` call the Rust FFI
``analytic_penalty_value_grad`` / ``analytic_penalty_hvp`` with
``rho = np.zeros(0)``. The FFI rejects any rho whose length disagrees with
``registry.total_rho_count()`` (``crates/gam-pyffi/src/lib.rs:25444``).

ARD declares ``rho_count == latent_dim`` (``analytic_penalties.rs:2417``),
so ``ARDPenalty(...).value_grad(t)`` currently raises:

    rho length 0 does not match analytic penalty rho_count <d>

The fix is to pass ``None`` (or a correctly sized vector) so the FFI's
default-rho branch (``lib.rs:25440-25443``) fills zeros of the right length.

These tests are RED until that fix lands. The IBP / BlockOrthogonality /
MechanismSparsity descriptors happen to declare ``rho_count == 0`` for the
non-learnable mode the wrapper sets, so they pass today; pinning the
contract here guards against future descriptors that expose rho_count > 0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import gamfit
from gamfit._penalty_descriptors import (
    ARDPenalty,
    BlockOrthogonalityDescriptor,
    OrderedBetaBernoulliPenalty,
    MechanismSparsityDescriptor,
)


# ---------------------------------------------------------------------------
# ARD — rho_count == latent_dim, currently fails
# ---------------------------------------------------------------------------


def test_ard_value_grad_numpy_does_not_raise_rho_length() -> None:
    """ARDPenalty.value_grad on numpy must not be rejected for rho length."""
    t = np.ones((4, 3), dtype=np.float64)
    # No assertion on the value yet — just that no rho-length ValueError.
    ARDPenalty(weight=1.0).value_grad(t)


def test_ard_value_grad_numpy_matches_closed_form() -> None:
    """ARD value at rho=0 is ridge + Occam log-det:

        value = Σ_j [ ½·λ_j·‖t_j‖² − ½·N_eff·ln(λ_j) ]

    where λ_j = weight·exp(rho_j) (analytic_penalties.rs:2353-2368). With the
    wrapper's default rho=0 and the Rust default ``N_eff = n_obs``
    (target.len()/latent_dim), this collapses to
    ``½·w·‖t‖² − ½·(d·n_obs)·ln(w)``. The log-det piece has no t-derivative,
    so the gradient is still ``w·t``.
    """
    weight = 0.7
    rng = np.random.default_rng(0)
    t = rng.standard_normal((5, 3))
    n_obs, d = t.shape
    v, g = ARDPenalty(weight=weight).value_grad(t)
    expected = 0.5 * weight * float(np.sum(t * t)) - 0.5 * (d * n_obs) * math.log(weight)
    assert float(v) == pytest.approx(expected, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(g, weight * t, atol=1e-12, rtol=1e-12)


def test_ard_value_grad_torch_does_not_raise_rho_length() -> None:
    torch = pytest.importorskip("torch")
    t = torch.ones(4, 3, dtype=torch.float64)
    ARDPenalty(weight=1.0).value_grad(t)


def test_ard_hvp_numpy_matches_closed_form() -> None:
    """For default rho=0, ARD Hessian is diagonal weight·I."""
    weight = 1.3
    rng = np.random.default_rng(1)
    t = rng.standard_normal((6, 2))
    v = rng.standard_normal((6, 2))
    hv = ARDPenalty(weight=weight).hvp(t, v)
    np.testing.assert_allclose(hv, weight * v, atol=1e-12, rtol=1e-12)


def test_ard_value_grad_vector_target() -> None:
    """1-D target → d=1 → rho_count=1 (still > 0, still failed by old wrapper).

    Same Rust formula as the 2-D case: ½·w·‖t‖² − ½·(d·n_obs)·ln(w).
    """
    weight = 0.5
    t = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    n_obs, d = t.shape[0], 1
    v, g = ARDPenalty(weight=weight).value_grad(t)
    expected = 0.5 * weight * float(np.sum(t * t)) - 0.5 * (d * n_obs) * math.log(weight)
    assert float(v) == pytest.approx(expected, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(g, weight * t, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# IBP / BlockOrthogonality / MechanismSparsity — rho_count == 0 in non-
# learnable mode (which is what the wrapper sets). These should keep
# working after the fix; they pin the contract so it doesn't regress.
# ---------------------------------------------------------------------------


def test_ibp_value_grad_numpy_runs() -> None:
    rng = np.random.default_rng(2)
    t = rng.standard_normal((8, 4))
    v, g = OrderedBetaBernoulliPenalty(alpha=1.0, tau=1.0).value_grad(t)
    assert np.isfinite(float(v))
    assert g.shape == t.shape and np.all(np.isfinite(g))


def test_block_orthogonality_value_grad_numpy_runs() -> None:
    rng = np.random.default_rng(3)
    t = rng.standard_normal((8, 4))
    pen = BlockOrthogonalityDescriptor(groups=[[0, 1], [2, 3]], weight=0.5, n_eff=8)
    v, g = pen.value_grad(t)
    assert np.isfinite(float(v))
    assert g.shape == t.shape and np.all(np.isfinite(g))


def test_mechanism_sparsity_value_grad_numpy_runs() -> None:
    rng = np.random.default_rng(4)
    t = rng.standard_normal((6, 4))
    pen = MechanismSparsityDescriptor(
        feature_groups=[[0, 1], [2, 3]], weight=0.4, n_eff=6
    )
    v, g = pen.value_grad(t)
    assert np.isfinite(float(v))
    assert g.shape == t.shape and np.all(np.isfinite(g))


# ---------------------------------------------------------------------------
# rho_count contract: the FFI default-rho branch must be exercised by the
# wrapper. The most direct probe is a passing call on a descriptor whose
# rho_count > 0 (= ARD). If this passes, ``None``/correctly-sized rho is
# reaching the FFI and the wrapper is no longer hard-coding ``zeros(0)``.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [1, 2, 5])
def test_ard_rho_count_respected_at_various_widths(d: int) -> None:
    """Each latent width is its own rho_count — wrapper must handle all."""
    t = np.ones((3, d), dtype=np.float64)
    v, g = ARDPenalty(weight=1.0).value_grad(t)
    assert float(v) == pytest.approx(0.5 * float(np.sum(t * t)))
    np.testing.assert_allclose(g, t, atol=1e-12, rtol=1e-12)
