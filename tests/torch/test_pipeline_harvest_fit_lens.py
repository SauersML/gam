"""End-to-end #980 pipeline: harvest → fit → lens → (steer).

This is the *first* full-stack test of the output-Fisher pullback metric (#980).
A tiny fixed torch model is harvested for per-token output-Fisher factors
(:func:`gamfit.torch.harvest.harvest_output_fisher_factors`), the resulting
shard is installed into an SAE-manifold fit
(:func:`gamfit.sae_manifold_fit(..., fisher_factors=...)`), and the fitted
:class:`gamfit.ManifoldSAE` is interrogated through the post-fit lens / gauge.

The load-bearing contracts asserted here are the *amended-contract* invariants
of #980:

* **Provenance is reported.** ``metric_provenance == "OutputFisher"`` exactly
  when a shard is supplied — the gauge/lens metric was installed.
* **The data fit is bit-identical to the Euclidean (no-shard) fit.** Installing
  ``RowMetric::OutputFisher`` changes *only* the inner product the gauge / lens
  / isometry are read through; it does **not** whiten the reconstruction
  likelihood. With the isometry gauge off (the default), the shard run and the
  no-shard run must agree on ``reml_score`` / ``reconstruction_r2`` / ``fitted``
  exactly. This is the single most important assertion in the file: a divergence
  means the metric leaked into the likelihood.
* **The two-score lens is present and sane.** Under OutputFisher provenance the
  per-atom lens reports finite ``presence`` (> 0 for an active atom) *and* a
  finite ``coupling`` (the behavioral output-Fisher axis is available); the
  Euclidean run reports ``coupling`` as ``NaN`` (not available, not zero).
* **Steering dosimetry (when reachable).** If a steer entry point is exposed
  from the fitted payload, the path-integrated KL dose is checked against the
  synthetic model's analytic KL. The steering primitive
  (``gam::inference::steering::steer_delta``) is currently **Rust-only with no
  FFI/Python surface**, so that arm is skeletoned and marked
  ``xfail``-pending-FFI; see :func:`test_steer_dosimetry_against_analytic_kl`.

Run status (authored 2026-06-09, Actor D): torch is not installed in this
environment and the installed gam extension predates the ``fisher_factors``
parameter of ``sae_manifold_fit_minimal`` (it raises ``TypeError: ... got an
unexpected keyword argument 'fisher_factors'``). These tests were therefore
**authored by inspection** against the documented post-rebuild API; they are
gated behind ``pytest.importorskip`` for torch + gamfit and will execute once
Actor A's rebuilt extension lands. The harvest portion is exercised standalone
by the sibling ``test_harvest.py`` (which passes).

Fixed seeds throughout; no clock entropy.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# gamfit must import (the Rust extension binds lazily, so this succeeds even
# before the fisher_factors-aware binary is rebuilt; the *fit calls* are what
# require the rebuild).
gamfit = pytest.importorskip("gamfit")

from gamfit import ManifoldSAE, sae_manifold_fit  # noqa: E402
from gamfit._binding import rust_module  # noqa: E402
from gamfit.torch.harvest import (  # noqa: E402
    HarvestShard,
    harvest_output_fisher_factors,
)


# ---------------------------------------------------------------------------
# Tiny synthetic model + planted-manifold data
# ---------------------------------------------------------------------------


class _LinearHead(torch.nn.Module):
    """``logits = x @ Wᵀ`` with the hook site = the identity-passed input ``x``.

    Mirrors ``test_harvest._LinearHead``: ``feature`` is an ``nn.Identity``
    whose output is the hook-site activation ``x_n``, and ``head`` is a fixed
    linear map, so ``∂logits/∂x_n = W`` for every token. This makes the
    per-token output-Fisher pullback ``G_n = Wᵀ F_n W`` exactly computable in
    closed form, which is what lets the steering dose be checked against an
    analytic KL.
    """

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.feature = torch.nn.Identity()
        self.head = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        with torch.no_grad():
            self.head.weight.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.feature(x))


def _planted_circle_activations(n: int, p: int, *, noise: float, seed: int) -> np.ndarray:
    """``(n, p)`` activations sampled along a single planted circle in ℝ^p.

    Atom-recoverable structure so the fit has something to type and the lens
    reports a genuinely *present* atom. The first two channels carry the circle;
    the remaining channels are a fixed low-amplitude linear image of the phase so
    every output dimension is touched (keeps the output-Fisher pullback full-rank
    enough for a rank-r harvest to be meaningful).
    """
    rng = np.random.default_rng(seed)
    phase = np.sort(rng.uniform(0.0, 1.0, size=n))
    ang = 2.0 * np.pi * phase
    cols = [np.cos(ang), np.sin(ang)]
    for j in range(2, p):
        cols.append(0.4 * np.cos(ang + 0.5 * j))
    X = np.stack(cols, axis=1).astype(np.float64)
    X += noise * rng.standard_normal(X.shape)
    return np.ascontiguousarray(X)


# ---------------------------------------------------------------------------
# Shared fixtures: the harvested shard + the two fits (shard vs no-shard)
# ---------------------------------------------------------------------------

_C, _P, _N = 6, 5, 32  # classes, activation dim, tokens
_RANK = 3
_NOISE = 0.01
_FIT_KW = dict(
    K=1,
    d_atom=1,
    atom_topology="circle",
    assignment="softmax",
    n_iter=30,
    random_state=0,
    # Isometry gauge OFF (the default): the metric must not enter the data fit,
    # so the shard run reproduces the Euclidean run bit-for-bit.
    isometry_weight=0.0,
)


@pytest.fixture(scope="module")
def planted_X() -> np.ndarray:
    return _planted_circle_activations(_N, _P, noise=_NOISE, seed=0)


@pytest.fixture(scope="module")
def harvest_shard(planted_X: np.ndarray) -> HarvestShard:
    """Harvest output-Fisher factors at the model's hook site for ``planted_X``.

    The model's linear head fixes ``J_n = W`` for every token, so the harvested
    factors are the exact rank-r truncation of ``G_n = Wᵀ F_n W`` (the harvest
    contract is independently checked in ``test_harvest.py``).
    """
    rng = np.random.default_rng(7)
    W = torch.from_numpy(rng.standard_normal((_C, _P))).to(torch.float64)
    model = _LinearHead(W).to(torch.float64)
    X = torch.from_numpy(planted_X).to(torch.float64)
    shard = harvest_output_fisher_factors(
        model,
        model.feature,
        X,
        rank=_RANK,
        oversample=3,
        n_iter=4,
        trace_probes=_P,
        seed=0,
    )
    assert isinstance(shard, HarvestShard)
    assert shard.X.shape == (_N, _P)
    assert shard.U.shape == (_N, _P, _RANK)
    assert shard.mass_residual.shape == (_N,)
    return shard


@pytest.fixture(scope="module")
def fit_no_shard(planted_X: np.ndarray) -> ManifoldSAE:
    """Baseline Euclidean fit — no output-Fisher shard installed."""
    return sae_manifold_fit(planted_X, fisher_factors=None, **_FIT_KW)


@pytest.fixture(scope="module")
def fit_with_shard(planted_X: np.ndarray, harvest_shard: HarvestShard) -> ManifoldSAE:
    """OutputFisher fit — the harvested shard is installed as the gauge/lens metric."""
    return sae_manifold_fit(planted_X, fisher_factors=harvest_shard, **_FIT_KW)


# ---------------------------------------------------------------------------
# 1. Provenance is reported
# ---------------------------------------------------------------------------


def test_metric_provenance_reported(
    fit_with_shard: ManifoldSAE, fit_no_shard: ManifoldSAE
) -> None:
    """The shard run installs OutputFisher; the no-shard run stays Euclidean."""
    assert fit_with_shard.metric_provenance == "OutputFisher"
    assert fit_with_shard.fisher_factor_kind == "uncertified_approximation"
    assert fit_no_shard.metric_provenance == "Euclidean"
    # The per-row truncation diagnostic rides into the model under a shard, and
    # is absent under the Euclidean path.
    assert fit_with_shard.fisher_mass_residual is not None
    assert np.asarray(fit_with_shard.fisher_mass_residual).shape == (_N,)
    assert np.all(np.isfinite(fit_with_shard.fisher_mass_residual))
    # PSD ⇒ the off-subspace mass is non-negative.
    assert np.all(np.asarray(fit_with_shard.fisher_mass_residual) >= -1e-9)
    assert fit_no_shard.fisher_mass_residual is None


# ---------------------------------------------------------------------------
# 2. The data fit is IDENTICAL to the no-shard fit (the amended contract)
# ---------------------------------------------------------------------------


def test_data_fit_identical_to_euclidean(
    fit_with_shard: ManifoldSAE, fit_no_shard: ManifoldSAE
) -> None:
    """Installing the metric must not touch the reconstruction likelihood.

    With the isometry gauge off, the OutputFisher metric only changes the inner
    product the gauge / lens are read through, never the data fit. So the two
    runs must agree *exactly* on the criterion (``reml_score``), the predictive
    summary (``reconstruction_r2``), and the fitted reconstruction (``fitted``).
    A nonzero difference here is the #980 leak we are guarding against.
    """
    # Exact agreement (same solver, same data, same seed; only the gauge metric
    # differs and it is off-likelihood). atol is a hair above 0 for f64 FFI
    # round-trip noise, not a tolerance on a genuine numerical difference.
    assert fit_with_shard.reml_score == pytest.approx(fit_no_shard.reml_score, abs=1e-12)
    assert fit_with_shard.reconstruction_r2 == pytest.approx(
        fit_no_shard.reconstruction_r2, abs=1e-12
    )
    np.testing.assert_allclose(
        fit_with_shard.fitted, fit_no_shard.fitted, rtol=0.0, atol=1e-12
    )
    # The recovered latent geometry (decoder, coords, assignments) is the same
    # object too — the metric did not move the optimum.
    assert len(fit_with_shard.decoder_blocks) == len(fit_no_shard.decoder_blocks)
    for b_shard, b_eucl in zip(
        fit_with_shard.decoder_blocks, fit_no_shard.decoder_blocks
    ):
        np.testing.assert_allclose(b_shard, b_eucl, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        fit_with_shard.assignments, fit_no_shard.assignments, rtol=0.0, atol=1e-10
    )


# ---------------------------------------------------------------------------
# 3. The two-score lens is present and sane
# ---------------------------------------------------------------------------


def test_atom_two_lens_present_and_coupled(
    fit_with_shard: ManifoldSAE, fit_no_shard: ManifoldSAE
) -> None:
    """Under OutputFisher the lens reports finite presence AND coupling.

    The lens (``atom_two_lens``) is the #980 read of the fitted model: per-atom
    ``presence`` (Fisher-free, activation-side) and ``coupling`` (behavioral,
    the only place Fisher enters). For the single planted-circle atom the
    presence must be finite and strictly positive (the atom is genuinely
    expressed), and under OutputFisher provenance the coupling must be *present*
    (finite) — there is a behavioral axis to compare against. The Euclidean run
    has no behavioral axis: its coupling is ``NaN`` (unavailable, not zero).
    """
    lens = fit_with_shard.atom_two_lens
    assert lens is not None, "OutputFisher fit must surface atom_two_lens"
    # Keys mirror the Rust `sae_atom_two_lens_dict` serialization.
    for key in (
        "names",
        "presence",
        "presence_normalized",
        "coupling",
        "coupling_normalized",
        "discrepancy",
        "coupling_available",
    ):
        assert key in lens, f"atom_two_lens missing key {key!r}"

    presence = np.asarray(lens["presence"], dtype=float)
    coupling = np.asarray(lens["coupling"], dtype=float)
    assert presence.shape == (len(fit_with_shard.atoms),)
    assert np.all(np.isfinite(presence))
    # The single planted atom is active ⇒ strictly positive presence.
    assert float(presence[0]) > 0.0

    # OutputFisher ⇒ the behavioral coupling axis is available and finite.
    assert bool(lens["coupling_available"]) is True
    assert np.all(np.isfinite(coupling))
    assert float(coupling[0]) > 0.0
    # Provenance of the coupling axis is echoed (OutputFisher).
    assert "OutputFisher" in str(lens.get("coupling_provenance"))

    # The Euclidean run has NO behavioral axis: coupling is NaN, not zero.
    eucl_lens = fit_no_shard.atom_two_lens
    assert eucl_lens is not None
    assert bool(eucl_lens["coupling_available"]) is False
    eucl_coupling = np.asarray(eucl_lens["coupling"], dtype=float)
    assert np.all(np.isnan(eucl_coupling))
    # Presence is Fisher-free, so it is finite and positive in BOTH runs.
    eucl_presence = np.asarray(eucl_lens["presence"], dtype=float)
    assert np.all(np.isfinite(eucl_presence))
    assert float(eucl_presence[0]) > 0.0


def test_residual_gauge_present(fit_with_shard: ManifoldSAE) -> None:
    """The residual-gauge certificate rides into the fitted model under #980."""
    gauge = fit_with_shard.residual_gauge
    assert gauge is not None, "OutputFisher fit must surface residual_gauge"
    for key in (
        "group_signature",
        "metric_provenance",
        "pinning_rank",
        "residual_gauge_dim",
        "generators",
    ):
        assert key in gauge, f"residual_gauge missing key {key!r}"
    # The certificate records the inner product it was computed in.
    assert "OutputFisher" in str(gauge["metric_provenance"])
    assert isinstance(gauge["group_signature"], str)
    assert int(gauge["residual_gauge_dim"]) >= 0


# ---------------------------------------------------------------------------
# 4. Steering dosimetry — skeletoned, pending an FFI surface for steer_delta
# ---------------------------------------------------------------------------


def _analytic_softmax_kl(W: np.ndarray, x_from: np.ndarray, x_to: np.ndarray) -> float:
    """Exact KL(softmax(W x_from) ‖ softmax(W x_to)) for the linear head.

    The synthetic ``_LinearHead`` makes the output distribution at an activation
    ``x`` exactly ``softmax(W x)``. For a steering move that drives the hidden
    state from ``x_from`` to ``x_to``, the *true* behavioral effect is this KL —
    the analytic ground truth the endpoint output-Fisher dose
    (``SteerPlan.predicted_nats``) should match to second order for a small move.
    """
    def _softmax(z: np.ndarray) -> np.ndarray:
        e = np.exp(z - z.max())
        return e / e.sum()

    p = _softmax(W @ x_from)
    q = _softmax(W @ x_to)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def test_steer_dosimetry_against_analytic_kl(
    fit_with_shard: ManifoldSAE, harvest_shard: HarvestShard
) -> None:
    """Endpoint output-Fisher dose ≈ analytic KL for a small steer move.

    The steering FFI (``ManifoldSAE.steer`` → ``gam-pyffi::sae_steer_delta`` →
    ``gam::inference::steering::steer_delta``) drives the planted atom a small
    latent step along the recovered circle and returns the SteerPlan dosimetry:

    1. Pick the planted atom (k=0) and two nearby on-manifold coordinates
       ``t_from`` / ``t_to`` (a small latent step along the recovered circle).
    2. The decoder maps them to activation-space points ``x_from = g_k(t_from)``
       and ``x_to = g_k(t_to)``; the SteerPlan's ``delta`` is the move
       ``δ = a·(x_to − x_from)`` (amplitude ``a = 1`` for this single-atom
       softmax fit, where the only atom carries unit assignment mass on every
       row).
    3. The SteerPlan's ``predicted_nats`` (endpoint Fisher quadratic) must
       match the analytic KL of the synthetic head between those two activations
       to second order: small move ⇒ ``predicted_nats ≈ KL`` within a relative
       tolerance; ``off_manifold_norm ≈ 0``; ``validity_radius`` ≥ the move
       length.

    The synthetic ``_LinearHead`` makes the output distribution at an activation
    ``x`` exactly ``softmax(W x)``, so the *true* behavioral effect of the move is
    the analytic softmax-KL between the decoded endpoints under the model weight
    ``W``. ``W`` is reconstructed here from the same fixed seed the
    ``harvest_shard`` fixture used, so the harvested output-Fisher factors
    (``G_n = Wᵀ F_n W``) and this analytic ground truth are the SAME geometry.
    """
    # The steer entry point is now FFI-exposed and surfaced on the fitted model.
    steer = getattr(fit_with_shard, "steer", None)
    assert steer is not None, "ManifoldSAE must expose a `steer` entry point"

    # Drive the planted atom a small latent step along the recovered circle.
    atom_k = 0
    coords = np.asarray(fit_with_shard.coords[atom_k], dtype=float)
    # Two nearby coordinates: a genuinely small latent step keeps the move in the
    # second-order regime where the path-integrated dose matches the analytic KL.
    order = np.argsort(coords[:, 0])
    t_from = coords[order[0]]
    t_to = coords[order[1]]
    metric_row = int(order[0])
    amplitude = float(fit_with_shard.assignments[metric_row, atom_k])
    plan = steer(
        atom_k,
        metric_row=metric_row,
        amplitude=amplitude,
        t_from=t_from,
        t_to=t_to,
    )
    # Geometry self-checks: the move stays on the learned surface, the dose is
    # measured through OutputFisher, and the linearization is trusted past the
    # move length.
    assert plan["metric_provenance"] == "OutputFisher"
    assert plan["predicted_nats"] is not None
    assert plan["off_manifold_norm"] == pytest.approx(0.0, abs=1e-6)
    move_len = float(np.linalg.norm(np.asarray(t_to) - np.asarray(t_from)))
    assert float(plan["validity_radius"]) >= move_len - 1e-9

    # Dosimetry ground truth: reconstruct the synthetic head weight W from the
    # same fixed seed the harvest fixture used (rng default_rng(7)), so the
    # harvested output-Fisher factors and this analytic KL share one geometry.
    rng = np.random.default_rng(7)
    W = rng.standard_normal((_C, _P))
    # The decoder-decoded activations the move δ traverses. amplitude == 1 here
    # (single-atom softmax ⇒ unit mass), so δ == x_to − x_from and the SteerPlan
    # delta equals that chord.
    assert float(plan["amplitude"]) == pytest.approx(1.0, abs=1e-9)
    decoder = np.asarray(fit_with_shard.decoder_blocks[atom_k], dtype=float)
    resolution = fit_with_shard.geometry_plans[atom_k]["resolution"]
    assert resolution["kind"] == "periodic_harmonics"
    phi = np.asarray(
        rust_module().basis_with_jet(
            "periodic",
            np.ascontiguousarray(np.asarray([t_from, t_to], dtype=float).reshape(2, 1)),
            {"n_harmonics": int(resolution["order"])},
        )[0],
        dtype=float,
    )
    decoded = phi @ decoder  # (2, p): rows are g_k(t_from), g_k(t_to)
    x_from, x_to = decoded[0], decoded[1]
    # The SteerPlan delta is exactly the decoded chord (amplitude 1).
    np.testing.assert_allclose(
        np.asarray(plan["delta"], dtype=float), x_to - x_from, rtol=0.0, atol=1e-8
    )
    kl = _analytic_softmax_kl(W, x_from, x_to)
    # Second-order agreement for the small step. The endpoint output-Fisher
    # dose is the quadratic (Fisher) model of the KL; for a genuinely small move it
    # matches the exact KL to a loose relative tolerance. Two gaps keep this from
    # being exact: (i) the softmax KL has third-order curvature beyond the Fisher
    # quadratic, and (ii) the dose reads the harvested per-row Fisher at the atom's
    # explicitly selected source row (`metric_row`), whose base activation differs
    # slightly from the decoded `x_from`; both are O(move) on this well-recovered circle.
    assert plan["predicted_nats"] == pytest.approx(kl, rel=0.3, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. #981 replicate-agreement skeleton — gated on Actor A's K=2 fixture
# ---------------------------------------------------------------------------
#
# Two replicate fits of the SAME data under two different seeds must agree
# *exactly up to the reported residual-gauge group*, and no further:
#
#   * their ``residual_gauge["group_signature"]`` strings must be identical
#     (the order-independent signature of the unpinned generator families —
#     `group_signature_of` in src/identifiability/sae.rs guarantees two
#     replicates agree on their residual gauge iff these strings are equal);
#   * the recovered parameters (decoder / coords) must agree only up to that
#     named symmetry — e.g. under a diffeomorphism-unpinned gauge the per-token
#     reconstruction ``fitted`` (a gauge invariant) must match, while the raw
#     latent coordinates may differ by the reported reparametrization.
#
# This is gated OFF until Actor A's K=2 fixture actually recovers structure: a
# replicate-agreement assertion is only meaningful once both fits converge to
# the same manifold (otherwise "agreement up to symmetry" is vacuous noise).
# Flip ``_REPLICATE_FIXTURE_READY`` to True (and drop the skip) once that lands.

_REPLICATE_FIXTURE_READY = False


@pytest.mark.skipif(
    not _REPLICATE_FIXTURE_READY,
    reason=(
        "#981 replicate-agreement: enable once Actor A's K=2 fixture recovers "
        "structure. Two seeds must agree EXACTLY up to the reported residual "
        "gauge group_signature and only up to that named symmetry; this is "
        "vacuous until the fit actually recovers the planted manifold."
    ),
)
def test_replicate_agreement_up_to_residual_gauge(planted_X: np.ndarray) -> None:
    """#981: two seeds agree exactly up to the reported residual-gauge group.

    SKELETON. Fit the same data twice under two seeds and assert:

    1. ``group_signature`` equality — the two replicates report the *same*
       residual-gauge group (identical order-independent signature string);
    2. parameters agree only up to that named symmetry — gauge-invariant
       quantities (the per-token reconstruction ``fitted``) must match across
       replicates, while gauge-variant raw coordinates need not.
    """
    fit_a = sae_manifold_fit(planted_X, random_state=0, **{
        k: v for k, v in _FIT_KW.items() if k != "random_state"
    })
    fit_b = sae_manifold_fit(planted_X, random_state=1, **{
        k: v for k, v in _FIT_KW.items() if k != "random_state"
    })

    gauge_a = fit_a.residual_gauge
    gauge_b = fit_b.residual_gauge
    assert gauge_a is not None and gauge_b is not None

    # (1) EXACT residual-gauge group agreement.
    assert gauge_a["group_signature"] == gauge_b["group_signature"], (
        "replicate fits disagree on their residual gauge group: "
        f"{gauge_a['group_signature']!r} vs {gauge_b['group_signature']!r}"
    )

    # (2) Gauge-invariant reconstruction agrees across replicates; the raw
    # coordinates are allowed to differ by exactly the reported symmetry. Once
    # the K=2 fixture recovers structure this becomes a tight tolerance.
    np.testing.assert_allclose(fit_a.fitted, fit_b.fitted, rtol=0.0, atol=1e-6)
