"""#2091 — the direct native fit builder reproduces the public fit artifact.

Design (A) moves the raw-payload -> flat-`to_dict`-schema coercion into Rust so
`sae_manifold_fit` returns the Rust-owned `ManifoldSAE` with no Python adapter.
This test captures the real raw `sae_manifold_fit_minimal` payload from a live
fit, rebuilds it independently, and asserts exact artifact equality. Any field
the builder mis-assembles (kind/topology derivation, n_harmonics, per-atom
gate-column slicing, channel-cov factor, periodic shape-band reorder, report
passthroughs) surfaces as a dict mismatch. Three atom kinds are exercised
(circle=periodic shape-band path, Euclidean kind derivation, and Duchon-center
retention), plus the declared-linear-block relabel.

MAINLINE only: no Fisher shard, no linear_block relabel (follow-up arms). The
mainline fixture is asserted free of non-finite values so a future data change cannot
silently exercise the Rust mapping-coercion rejection branch. These fixtures disable
the independent outer smoothing-parameter and structure searches: this contract
starts at the raw live-fit payload boundary and must not inherit optimizer
certification failures that occur before the builder is called."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")
from gamfit._sae_manifold import rust_module  # noqa: E402


def _builder():
    return rust_module().sae_manifold_from_fit_payload


def _no_nonfinite(v) -> bool:
    if isinstance(v, bool):
        return True
    if isinstance(v, float):
        return math.isfinite(v)
    if isinstance(v, dict):
        return all(_no_nonfinite(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return all(_no_nonfinite(x) for x in v)
    return True


def _data_for(topology: str, n: int, p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if topology == "circle":
        theta = rng.uniform(0.0, 2.0 * np.pi, n)
        harm = np.column_stack([np.cos(theta), np.sin(theta)])
        x = harm @ rng.standard_normal((2, p))
    else:
        t = rng.standard_normal((n, 2))
        x = t @ rng.standard_normal((2, p)) + 0.3 * (t[:, :1] ** 2) @ rng.standard_normal((1, p))
    x = x + 0.02 * rng.standard_normal((n, p))
    return (x - x.mean(axis=0, keepdims=True)).astype(np.float64)


def test_builder_full_fit_equiv_and_kind_derivation(monkeypatch):
    builder = _builder()
    rm = rust_module()
    captured: dict = {}
    orig = rm.sae_manifold_fit_minimal

    def capture(*args, **kwargs):
        payload = orig(*args, **kwargs)
        captured["raw"] = dict(payload)
        return payload

    monkeypatch.setattr(rm, "sae_manifold_fit_minimal", capture)

    x = _data_for("circle", n=60, p=5, seed=0)
    fit = gamfit.sae_manifold_fit(
        X=x, K=2, d_atom=1, atom_topology="circle", assignment="softmax",
        n_iter=8, random_state=0, _run_structure_search=False,
        _run_outer_rho_search=False,
    )
    assert "raw" in captured, "raw sae_manifold_fit_minimal payload was not captured"

    penalties = list(fit.primitive_names[1:])
    core = builder(
        captured["raw"],
        x,
        str(fit.atom_topology),
        fit.assignment,
        fit.assignment_label,
        penalties,
        float(fit.alpha),
        bool(fit.learnable_alpha),
        float(fit.tau),
        float(fit.sparsity_strength),
        float(fit.smoothness),
        float(fit.learning_rate),
        int(fit.max_iter),
        int(fit.random_state),
        fit.top_k,
        float(fit.jumprelu_threshold),
    )

    old = fit.to_dict()
    new = core.to_dict()
    # Precondition: the mainline fixture must be non-finite-free so this test can
    # never silently exercise the NaN branch (whose handling is pinned separately).
    assert _no_nonfinite(old), "mainline equiv fixture unexpectedly carries a non-finite value"
    assert new == old

    # The builder's kind-specific derivations do not depend on re-running the
    # nonlinear SAE solver. Retag the successfully captured live payload at the
    # raw plan boundary, keeping every numeric array and per-atom gate column
    # unchanged, and exercise the Euclidean, Duchon-center, and declared
    # linear-block branches deterministically.
    k_atoms = len(fit.atom_topologies)
    variants = [
        ("euclidean", "euclidean", "euclidean"),
        ("duchon", "duchon", "euclidean"),
        ("linear", "linear_block", "linear_block"),
    ]
    for fitted_kind, declared_kind, expected_topology in variants:
        raw = copy.deepcopy(captured["raw"])
        for plan in raw["atom_plans"]:
            plan["kind"] = fitted_kind
            plan["n_harmonics"] = 0
            if fitted_kind == "duchon":
                basis_size = int(plan["basis_size"])
                plan["duchon_centers"] = np.linspace(
                    -1.0, 1.0, basis_size, dtype=np.float64
                ).reshape(-1, 1)
            else:
                plan["duchon_centers"] = None
        for atom in raw["atoms"]:
            atom["basis_kind"] = fitted_kind
            atom["on_atom_coords_u_arc"] = None
            atom["shape_band_coords"] = None
            atom["shape_band_mean"] = None
            atom["shape_band_sd"] = None

        derived = builder(
            raw,
            x,
            expected_topology,
            fit.assignment,
            fit.assignment_label,
            penalties,
            float(fit.alpha),
            bool(fit.learnable_alpha),
            float(fit.tau),
            float(fit.sparsity_strength),
            float(fit.smoothness),
            float(fit.learning_rate),
            int(fit.max_iter),
            int(fit.random_state),
            fit.top_k,
            float(fit.jumprelu_threshold),
            None,
            None,
            [declared_kind] * k_atoms,
        )
        payload = derived.to_dict()
        assert payload["basis_specs"] == [fitted_kind] * k_atoms
        assert payload["basis_kinds"] == [declared_kind] * k_atoms
        assert payload["atom_topologies"] == [expected_topology] * k_atoms
        assert payload["atom_topology"] == expected_topology
        if fitted_kind == "duchon":
            assert all(center is not None for center in derived.duchon_centers)


def test_builder_full_fit_equiv_with_fisher_shard(monkeypatch):
    """Fisher-shard arm: a fit with a raw (n,p,r) output-Fisher shard installs
    metric_provenance='OutputFisher' + retains fisher_factors/fisher_provenance
    POST-from_payload. The builder threads the same shard and must reproduce
    to_dict (fisher_factors serialized, provenance present, dose state intact)."""
    builder = _builder()
    rm = rust_module()
    captured: dict = {}
    orig = rm.sae_manifold_fit_minimal

    def capture(*args, **kwargs):
        payload = orig(*args, **kwargs)
        captured["raw"] = dict(payload)
        return payload

    monkeypatch.setattr(rm, "sae_manifold_fit_minimal", capture)

    n, p, r = 48, 6, 2
    x = _data_for("circle", n=n, p=p, seed=1)
    u = np.random.default_rng(2).standard_normal((n, p, r)).astype(np.float64)
    fit = gamfit.sae_manifold_fit(
        X=x, K=1, d_atom=1, atom_topology="circle", assignment="softmax",
        n_iter=8, random_state=0, fisher_factors=u,
        _run_structure_search=False, _run_outer_rho_search=False,
    )
    assert fit.metric_provenance == "OutputFisher"
    assert "raw" in captured

    core = builder(
        captured["raw"], x, str(fit.atom_topology), fit.assignment, fit.assignment_label,
        list(fit.primitive_names[1:]),
        float(fit.alpha), bool(fit.learnable_alpha), float(fit.tau),
        float(fit.sparsity_strength), float(fit.smoothness), float(fit.learning_rate),
        int(fit.max_iter), int(fit.random_state), fit.top_k, float(fit.jumprelu_threshold),
        np.ascontiguousarray(fit.fisher_factors), fit.fisher_provenance,
    )
    old = fit.to_dict()
    new = core.to_dict()
    assert _no_nonfinite(old)
    assert new == old
    # The shard must actually be present (this is the OutputFisher path).
    assert new["fisher_factors"] is not None
    assert new["metric_provenance"] == "OutputFisher"


def test_builder_rejects_nonfinite_required_numeric_field():
    """The Rust mapping coercion nulls NaN and required numeric fields reject it."""
    builder = _builder()
    bad_payload = {"atom_plans": [], "atoms": [], "chosen_k": 0, "dispersion": np.nan}
    x = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        builder(
            bad_payload, x, "euclidean", "softmax", "softmax", [],
            1.0, False, 0.5, 1.0, 1.0, 0.04, 50, 0, None, 0.0,
        )
