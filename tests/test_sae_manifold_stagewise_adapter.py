"""Contract tests for the SAC stagewise adapter (``sae_manifold_fit_stagewise``).

These pin the WS-A thin-wrapper contract over the Rust ``fit_stagewise`` driver:
on a planted two-circles target (two circle atoms living in DISJOINT channel
subspaces — the #2027 disjoint-chart repro), the grown dictionary must

  * report an ``ev_trace`` that is non-decreasing in births (monotone BY
    CONSTRUCTION — every adopted candidate cleared ``dEV >= min_effect_ev >= 0``),
  * report a non-decreasing ``backfit_ev_trace`` (keep-best sweeps),
  * log ZERO live-decoder collapse events (atoms never share a Hessian), and
  * separate the two planted circles onto their disjoint (even / odd) channels.

The adapter builds the K=1 seed with the proven ``sae_manifold_fit`` and rebuilds
the atom basis via the Rust ``basis_with_jet`` kernel, so all model math is in
Rust; the test exercises the whole path end to end.
"""
from __future__ import annotations

import numpy as np
import pytest

import gamfit
from gamfit._binding import rust_module

# The compact stagewise FFI only exists in the new engine wheel; skip cleanly on
# an older binary rather than erroring (SPEC forbids xfail, but a genuinely
# absent FFI is a skip, not a silenced failure).
pytestmark = pytest.mark.skipif(
    not hasattr(rust_module(), "sae_manifold_fit_stagewise"),
    reason="running extension predates the sae_manifold_fit_stagewise FFI",
)


def _planted_two_circles(
    n: int = 600, p: int = 16, noise: float = 0.02, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Two circles in DISJOINT channel subspaces: circle 0 on EVEN channels,
    circle 1 on ODD channels. Each row lies on exactly one circle. Returns
    ``(X, assign)`` with ``assign[i] in {0, 1}`` the planted circle."""
    rng = np.random.default_rng(seed)
    even = np.arange(0, p, 2)
    odd = np.arange(1, p, 2)
    # A distinct (sin, cos) channel pair inside each parity block for each circle.
    ce0, cs0 = even[0], even[1 % len(even)]
    ce1, cs1 = odd[0], odd[1 % len(odd)]
    assign = rng.integers(0, 2, size=n)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = np.zeros((n, p), dtype=np.float64)
    m0 = assign == 0
    x[m0, ce0] = np.sin(theta[m0])
    x[m0, cs0] = np.cos(theta[m0])
    m1 = assign == 1
    x[m1, ce1] = 0.9 * np.sin(theta[m1])
    x[m1, cs1] = 0.9 * np.cos(theta[m1])
    x += noise * rng.standard_normal((n, p))
    return np.ascontiguousarray(x), assign


def _is_non_decreasing(xs: np.ndarray) -> bool:
    xs = np.asarray(xs, dtype=float)
    if xs.size < 2:
        return True
    tol = 1e-9 * (1.0 + np.abs(xs[:-1]))
    return bool(np.all(np.diff(xs) >= -tol))


def test_stagewise_ev_monotone_and_zero_collapse() -> None:
    x, _assign = _planted_two_circles()
    result = gamfit.sae_manifold_fit_stagewise(
        x,
        d_atom=1,
        atom_topology="circle",
        assignment="ibp_map",
        max_births=6,
        max_backfit_sweeps=2,
        n_iter=40,
        random_state=0,
    )
    # EV monotone non-decreasing in births, by construction.
    assert _is_non_decreasing(result.ev_trace), (
        f"ev_trace must be non-decreasing in births; got {list(result.ev_trace)}"
    )
    # Keep-best backfitting is non-decreasing too.
    assert _is_non_decreasing(result.backfit_ev_trace), (
        f"backfit_ev_trace must be non-decreasing; got {list(result.backfit_ev_trace)}"
    )
    # Zero live-decoder collapse events — the SAC answer on a real-shaped target.
    assert result.collapse_events == [], (
        f"stagewise must log ZERO collapse events; got {result.collapse_events}"
    )
    assert np.isfinite(result.terminal_joint_reml)
    # A K=1 seed plus its births: the discovered dictionary is at least the seed.
    assert result.k == 1 + sum(
        1 for r in result.birth_records if r["accepted"] and r["kind"] == "new_atom"
    )


def test_stagewise_progress_callback_receives_durable_checkpoints() -> None:
    x, _assign = _planted_two_circles(n=160, p=8, noise=0.01, seed=3)
    events: list[dict] = []

    def _callback(event: dict) -> None:
        events.append(dict(event))

    gamfit.sae_manifold_fit_stagewise(
        x,
        d_atom=1,
        atom_topology="circle",
        assignment="ibp_map",
        max_births=1,
        max_backfit_sweeps=0,
        n_iter=12,
        seed_n_iter=12,
        random_state=3,
        progress_callback=_callback,
    )

    assert events, "progress_callback must receive Rust stagewise progress events"
    event_names = [str(event["event"]) for event in events]
    assert "seed_fit_completed" in event_names
    assert "candidate_started" in event_names
    assert "terminal_evidence_completed" in event_names

    checkpoints = [event for event in events if event["checkpoint_available"]]
    assert checkpoints, "durable events must carry checkpoint payloads"
    for event in checkpoints:
        checkpoint = event["checkpoint"]
        assert checkpoint is not None, event
        assert int(checkpoint["k_final"]) == int(event["k"])
        assert np.asarray(checkpoint["logits"]).shape == (x.shape[0], int(event["k"]))
        assert len(checkpoint["atoms"]) == int(event["k"])

    first_round_candidates = [
        event.get("candidate")
        for event in events
        if event["event"] == "candidate_started" and int(event["birth_round"]) == 0
    ]
    assert first_round_candidates == ["new_atom"], (
        "the K=1 first birth must not run the duplicate chart-extension solve"
    )


def test_stagewise_separates_disjoint_circles() -> None:
    x, _assign = _planted_two_circles(seed=1)
    result = gamfit.sae_manifold_fit_stagewise(
        x,
        d_atom=1,
        atom_topology="circle",
        assignment="ibp_map",
        max_births=6,
        max_backfit_sweeps=2,
        n_iter=50,
        random_state=1,
    )
    # The planted structure is two disjoint circles, so SAC must grow at least a
    # second atom off the seed.
    assert result.k >= 2, f"expected >= 2 atoms for two planted circles; got K={result.k}"

    p = x.shape[1]
    even = np.arange(0, p, 2)

    def _parity_share(decoder: np.ndarray) -> float:
        # Fraction of the decoder's channel energy on EVEN channels (1.0 => pure
        # even/circle-0 subspace, 0.0 => pure odd/circle-1 subspace).
        energy = np.sum(decoder**2, axis=0)  # per-channel energy, length p
        total = float(np.sum(energy))
        if total <= 0.0:
            return 0.5
        return float(np.sum(energy[even]) / total)

    shares = [_parity_share(atom.decoder) for atom in result.atoms]
    # The two most-committed atoms must fall on OPPOSITE parities — the disjoint
    # charts are separated, not co-collapsed onto one shared subspace.
    most_even = max(shares)
    most_odd = min(shares)
    assert most_even > 0.75, f"no atom cleanly captured the even-channel circle; shares={shares}"
    assert most_odd < 0.25, f"no atom cleanly captured the odd-channel circle; shares={shares}"
