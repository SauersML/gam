"""End-to-end test for examples/compose_tiers.py (tiered SAE composition).

The driver composes a linear base dictionary (T1, ``sparse_dictionary_fit``)
with a small curved SAE manifold (T2, ``sae_manifold_fit``) fit on the T1
residual over a stratified subsample, then adds the two reconstructions. We
check that:

1. The driver runs end-to-end on small planted data and the curved tier adds
   reconstruction (combined EV strictly exceeds T1-only EV).
2. The combined reconstruction has the shape of the corpus.
3. The stratified subsample reproduces the ``rho_cascade`` contract: it is
   deterministic, realizes the target fraction in expectation, and carries the
   ``1/fraction`` importance weight.
4. An over-large curved ``K`` is rejected before any expensive Rust call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# The driver is a top-level script in examples/, not a gamfit submodule.
_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import compose_tiers as ct  # noqa: E402

# The curved joint solve on planted data runs past the fast Python-API CI
# budget, matching the sibling SAE example tests.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def planted() -> np.ndarray:
    args = ct.build_parser().parse_args(
        ["--synthetic", "--n-tokens", "5000", "--p", "64",
         "--k1", "16", "--k2", "8", "--random-state", "0"]
    )
    return ct._planted_activations(args)


def test_stratified_subsample_matches_rho_cascade_contract() -> None:
    n, target, seed = 200_000, 20_000, 7
    idx, weight = ct.stratified_subsample(n, target, seed)
    # Deterministic (same seed -> same draw).
    idx2, weight2 = ct.stratified_subsample(n, target, seed)
    assert np.array_equal(idx, idx2)
    assert weight == weight2
    # Fraction realized in expectation and weight is 1/fraction.
    fraction = target / n
    assert idx.size == pytest.approx(target, rel=0.05)
    assert weight == pytest.approx(1.0 / fraction, rel=1e-9)
    # Full corpus when target >= n: everything, weight 1.
    full_idx, full_w = ct.stratified_subsample(n, n, seed)
    assert full_idx.size == n
    assert full_w == 1.0


def test_compose_adds_reconstruction_over_linear_tier(planted: np.ndarray) -> None:
    result = ct.compose_tiers(
        planted, k1=16, k2=8, atom_topology="circle",
        assignment="threshold_gate", subsample_tokens=2500,
        alternation=True, random_state=0,
    )
    assert result.combined_recon.shape == planted.shape
    assert result.alternated is True
    assert result.subsample_rows < planted.shape[0]  # actually subsampled
    # The curved tier must explain residual structure the linear tier missed.
    assert result.combined_ev > result.t1_ev
    assert result.ev_gain > 0.0
    # Additive composition is exactly T1_recon + T2_recon.
    np.testing.assert_allclose(
        result.combined_recon, result.t1_recon + result.t2_recon, rtol=1e-5, atol=1e-5
    )


def test_no_alternation_flag_is_respected(planted: np.ndarray) -> None:
    result = ct.compose_tiers(
        planted, k1=16, k2=8, subsample_tokens=2500,
        alternation=False, random_state=0,
    )
    assert result.alternated is False
    assert result.combined_ev > result.t1_ev


def test_oversized_curved_k_rejected() -> None:
    x = np.zeros((128, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="K<=64"):
        ct.compose_tiers(x, k1=4, k2=128)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
