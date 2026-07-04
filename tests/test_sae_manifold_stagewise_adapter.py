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
    assert "seed_ready" in event_names
    assert "birth_round_started" in event_names
    assert "residual_model_started" in event_names
    assert "current_evidence_started" in event_names
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


def _behavioral_fisher_shard(n: int, p: int, s: int = 4, seed: int = 0) -> dict:
    """A minimal ``behavioral_fisher`` shard dict (the layout
    ``_normalize_fisher_factors`` accepts): ``U (n, p, s)`` random probe factors
    with ``G_n = U_n U_nᵀ`` a rank-s output-Fisher sketch, tagged so the fit
    installs ``RowMetric::behavioral_fisher`` (the Rung-1 GLS likelihood weight)."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, p, s)).astype(np.float64) / np.sqrt(s)
    return {"U": np.ascontiguousarray(u), "provenance": "behavioral_fisher"}


def test_stagewise_behavioral_fisher_runs_and_auto_disables_structured_whitening() -> None:
    # The Rung-1 (B4) GLS lane: a behavioral_fisher shard installs the output-Fisher
    # metric AS the reconstruction likelihood weight (nats) on the seed and every
    # born atom. structured_whitening defaults to None → resolves to False here
    # (the fixed harvest metric and the per-birth Σ-refit are rival metric sources),
    # so the fit runs under the fixed GLS metric alone.
    x, _assign = _planted_two_circles(n=300, p=8, noise=0.02, seed=5)
    shard = _behavioral_fisher_shard(x.shape[0], x.shape[1], s=4, seed=5)
    result = gamfit.sae_manifold_fit_stagewise(
        x,
        d_atom=1,
        atom_topology="circle",
        assignment="ibp_map",
        fisher_factors=shard,  # structured_whitening left at default None
        max_births=3,
        max_backfit_sweeps=1,
        n_iter=30,
        random_state=5,
    )
    assert result.k >= 1
    assert np.isfinite(result.terminal_joint_reml)
    assert result.fitted.shape == x.shape
    # A monotone-by-construction EV trace still holds under the GLS metric.
    assert _is_non_decreasing(result.ev_trace), list(result.ev_trace)


def _centered_ev(x: np.ndarray, recon: np.ndarray) -> float:
    """Centered explained variance of ``recon`` against target ``x``."""
    x = np.asarray(x, dtype=float)
    recon = np.asarray(recon, dtype=float)
    rss = float(np.sum((x - recon) ** 2))
    tss = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - rss / tss if tss > 0.0 else 0.0


def test_stagewise_out_of_sample_transform_honors_x() -> None:
    # #2118 — a SAC-composed dictionary must be scorable on HELD-OUT data. Before
    # the fix ``StagewiseSAE.reconstruct(X)`` ignored ``X`` and returned the
    # training reconstruction; now ``reconstruct``/``transform`` route the frozen
    # decoders through the existing Rust fixed-decoder OOS solve (via the lifted
    # ``to_manifold_sae()``), so passing fresh rows tracks THOSE rows.
    x_train, _assign = _planted_two_circles(n=300, p=16, noise=0.02, seed=11)
    fit = gamfit.sae_manifold_fit_stagewise(
        x_train,
        d_atom=1,
        atom_topology="circle",
        assignment="ibp_map",
        max_births=4,
        max_backfit_sweeps=1,
        n_iter=30,
        random_state=11,
    )
    assert fit.k >= 2, f"expected >= 2 atoms for two planted circles; got K={fit.k}"

    # The lift exposes the joint model's OOS surface.
    lifted = fit.to_manifold_sae()
    assert isinstance(lifted, gamfit.ManifoldSAE)
    assert len(lifted.atoms) == fit.k

    # --- Held-out rows with a DIFFERENT row count: the OLD code could not even
    # return an object of this shape (it returned the (n_train, p) training
    # reconstruction), so a matching-shape, high-EV result proves X is honored.
    x_hold, _ = _planted_two_circles(n=150, p=16, noise=0.02, seed=99)
    recon_oos = fit.transform(x_hold)
    assert recon_oos.shape == x_hold.shape, (
        f"OOS reconstruction must track X_new shape; got {recon_oos.shape} "
        f"for X_new {x_hold.shape}"
    )
    ev_hold = _centered_ev(x_hold, recon_oos)
    # Near the #2118 reference (top-1 = 0.9948 / greedy = 0.9952) with a modest
    # tolerance band; the noise floor (0.02) caps the achievable held-out EV.
    assert ev_hold > 0.95, f"held-out EV must be high; got {ev_hold:.4f}"

    # reconstruct(X_new) is the same OOS path (transform is the explicit alias).
    recon_via_reconstruct = fit.reconstruct(x_hold)
    assert np.allclose(recon_via_reconstruct, recon_oos)

    # --- Same-row-count held-out set: a VALUE-based proof the training
    # reconstruction is no longer returned. The OOS recon must (a) differ from the
    # cached training reconstruction and (b) explain the held-out rows far better
    # than blindly returning the training reconstruction ever could.
    x_hold2, _ = _planted_two_circles(n=300, p=16, noise=0.02, seed=99)
    recon2 = fit.transform(x_hold2)
    train_recon = fit.fitted  # in-sample composed reconstruction (the OLD output)
    assert not np.allclose(recon2, train_recon), (
        "reconstruct(X_new) still returns the training reconstruction (bug #2118)"
    )
    ev_oos = _centered_ev(x_hold2, recon2)
    ev_stale = _centered_ev(x_hold2, train_recon)
    assert ev_oos > 0.95, f"held-out EV must be high; got {ev_oos:.4f}"
    assert ev_oos > ev_stale + 0.5, (
        f"OOS recon must track X_new far better than the stale training "
        f"reconstruction; ev_oos={ev_oos:.4f} vs ev_stale={ev_stale:.4f}"
    )

    # In-sample surface is unchanged: reconstruct(None) is still the SAC sum.
    in_sample = fit.reconstruct()
    assert in_sample.shape == x_train.shape
    assert np.allclose(in_sample, fit.fitted)

    # encode(X_new) routes through the same OOS solve and is (N, K) shaped.
    codes = fit.encode(x_hold)
    assert codes.shape == (x_hold.shape[0], fit.k)


def test_stagewise_behavioral_fisher_conflicts_with_structured_whitening() -> None:
    # The fixed likelihood-whitening metric and the per-birth Σ-refit are two rival
    # sources for the SAME per-row inner product; the wrapper refuses the ambiguous
    # combination rather than let the Σ-refit silently clobber the harvest metric.
    x, _assign = _planted_two_circles(n=200, p=8, noise=0.02, seed=6)
    shard = _behavioral_fisher_shard(x.shape[0], x.shape[1], s=4, seed=6)
    with pytest.raises(ValueError, match="structured_whitening"):
        gamfit.sae_manifold_fit_stagewise(
            x,
            d_atom=1,
            atom_topology="circle",
            assignment="ibp_map",
            fisher_factors=shard,
            structured_whitening=True,  # explicit conflict
            max_births=1,
            n_iter=10,
            random_state=6,
        )
