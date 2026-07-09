"""Bitwise steer() equivalence: ManifoldSaeCore vs the Python dataclass (#2091).

Cutover increment 3 routes `ManifoldSAE.steer` and `ManifoldSaeCore.steer`
through ONE shared Rust rebuild (`steer_delta_from_arrays`): the dataclass path
marshals the model arrays across the FFI per call, while the pyclass path reads
them from Rust-owned state (so an attached Fisher shard is not re-marshalled —
acceptance bullet 2). Because both execute the identical helper on identical
inputs, their `SteerPlan`s must be **bitwise** identical; any difference is an
arg-threading divergence (e.g. the per-kind `n_harmonics` gate, `duchon_centers`,
the Fisher shard), which is exactly what this test exists to catch. A tolerance
would hide that class, so equality here is exact.

The fit is a REAL `sae_manifold_fit` run (not a stored payload) so the rebuild
sees actual trained decoder widths. Two atom kinds are exercised: `circle`
(periodic — the `n_harmonics` gate + a Fisher shard) and `euclidean` (the
duchon-family `duchon_centers` threading). The Fisher shard is a raw `(n, p, r)`
array (no torch dependency), so `metric_provenance` is `OutputFisher` and the
`predicted_nats` / `validity_radius` dose is exercised, not just the geometry.
"""
from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")
from gamfit._sae_manifold import rust_module  # noqa: E402


def _core_cls():
    cls = getattr(rust_module(), "ManifoldSaeCore", None)
    if cls is None:
        pytest.skip("wheel predates ManifoldSaeCore.steer (#2091 cutover)")
    if not hasattr(cls, "steer"):
        pytest.skip("wheel predates ManifoldSaeCore.steer (#2091 cutover)")
    return cls


def _assert_plans_bitwise_equal(a: dict, b: dict) -> None:
    assert set(a) == set(b), f"steer plan key sets differ: {set(a) ^ set(b)}"
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, (list, np.ndarray)) or isinstance(vb, (list, np.ndarray)):
            np.testing.assert_array_equal(
                np.asarray(va), np.asarray(vb), err_msg=f"field {key!r} differs"
            )
        elif isinstance(va, float) and (va != va):  # NaN
            assert vb != vb, f"field {key!r}: {va} vs {vb}"
        else:
            assert va == vb, f"field {key!r}: {va!r} vs {vb!r}"


def _planted_circle(n: int, p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    harm = np.column_stack([np.cos(theta), np.sin(theta)])
    mixing = rng.standard_normal((2, p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    x = harm @ mixing + 0.02 * rng.standard_normal((n, p))
    return (x - x.mean(axis=0, keepdims=True)).astype(np.float64)


def test_steer_bitwise_equivalence_circle_with_fisher() -> None:
    """Periodic (circle) atom + a raw output-Fisher shard: exercises the
    n_harmonics gate and the Fisher-metric install / dose path."""
    core_cls = _core_cls()
    n, p, r = 48, 6, 2
    x = _planted_circle(n, p, seed=0)
    u = np.random.default_rng(1).standard_normal((n, p, r)).astype(np.float64)

    fit = gamfit.sae_manifold_fit(
        X=x, K=1, d_atom=1, atom_topology="circle", assignment="softmax",
        n_iter=8, random_state=0, fisher_factors=u,
    )
    assert fit.metric_provenance == "OutputFisher"
    core = core_cls(fit.to_dict())

    t_from = np.array([0.0], dtype=np.float64)
    t_to = np.array([0.75], dtype=np.float64)
    plan_dc = fit.steer(0, t_from, t_to)
    plan_core = core.steer(0, t_from, t_to)
    # The dose must actually be present (this is the fisher path, not geometry).
    assert plan_dc["predicted_nats"] is not None
    _assert_plans_bitwise_equal(plan_dc, plan_core)


def test_steer_bitwise_equivalence_euclidean_duchon_centers() -> None:
    """Euclidean (degree-2 patch) atom: exercises the duchon_centers threading
    that the circle atom (no centers) does not."""
    core_cls = _core_cls()
    n, p = 50, 5
    rng = np.random.default_rng(3)
    # A 2-D latent blob so a d=2 euclidean patch has real structure to fit.
    t = rng.standard_normal((n, 2))
    mixing = rng.standard_normal((2, p))
    x = t @ mixing + 0.5 * (t[:, :1] ** 2) @ rng.standard_normal((1, p))
    x = (x + 0.02 * rng.standard_normal((n, p))).astype(np.float64)
    x -= x.mean(axis=0, keepdims=True)

    fit = gamfit.sae_manifold_fit(
        X=x, K=1, d_atom=2, atom_topology="euclidean", assignment="softmax",
        n_iter=8, random_state=0,
    )
    core = core_cls(fit.to_dict())
    assert fit.atoms[0].basis in {"euclidean", "linear"}

    t_from = np.array([0.0, 0.0], dtype=np.float64)
    t_to = np.array([0.5, -0.25], dtype=np.float64)
    plan_dc = fit.steer(0, t_from, t_to)
    plan_core = core.steer(0, t_from, t_to)
    _assert_plans_bitwise_equal(plan_dc, plan_core)
