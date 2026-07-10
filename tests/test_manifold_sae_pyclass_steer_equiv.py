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

ManifoldSaeCore = rust_module().ManifoldSaeCore


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
    n, p, r = 48, 6, 2
    x = _planted_circle(n, p, seed=0)
    u = np.random.default_rng(1).standard_normal((n, p, r)).astype(np.float64)

    fit = gamfit.sae_manifold_fit(
        X=x, K=1, d_atom=1, atom_topology="circle", assignment="softmax",
        n_iter=8, random_state=0, fisher_factors=u,
    )
    assert fit.metric_provenance == "OutputFisher"
    core = ManifoldSaeCore(fit.to_dict())

    t_from = np.array([0.0], dtype=np.float64)
    t_to = np.array([0.75], dtype=np.float64)
    plan_dc = fit.steer(0, 0, 1.0, t_from, t_to)
    plan_core = core.steer(0, 0, 1.0, t_from, t_to)
    # The dose must actually be present (this is the fisher path, not geometry).
    assert plan_dc["predicted_nats"] is not None
    _assert_plans_bitwise_equal(plan_dc, plan_core)


def test_reconstruct_training_bitwise_equivalence() -> None:
    """The in-sample reconstruction rebuilt from stored codes is bitwise
    identical between the dataclass and the pyclass (both call the same
    reconstruct_persisted_atom_set core over the same stored state)."""
    x = _planted_circle(40, 6, seed=5)
    fit = gamfit.sae_manifold_fit(
        X=x, K=1, d_atom=1, atom_topology="circle", assignment="softmax",
        n_iter=8, random_state=0,
    )
    core = ManifoldSaeCore(fit.to_dict())
    np.testing.assert_array_equal(
        fit.reconstruct_training(), core.reconstruct_training()
    )


def test_reconstruct_encode_oos_bitwise_equivalence_circle() -> None:
    """Held-out `reconstruct`/`encode` are bitwise identical between the
    dataclass and the pyclass. Both run the frozen-decoder OOS Newton solve
    through the same `sae_manifold_predict_oos` core; the pyclass builds the
    argument bundle (trained geometry, terminal rho*, hybrid-collapsed straight
    sub-models parsed from `hybrid_split`) from Rust-owned state, the dataclass
    from its attributes. Any divergence is an arg-threading bug in that rebuild,
    which is exactly what exact equality catches. Held-out X (a distinct seed)
    forces the OOS branch on both — no training-data shortcut."""
    x = _planted_circle(44, 6, seed=7)
    fit = gamfit.sae_manifold_fit(
        X=x, K=1, d_atom=1, atom_topology="circle", assignment="softmax",
        n_iter=8, random_state=0,
    )
    core = ManifoldSaeCore(fit.to_dict())
    x_oos = _planted_circle(20, 6, seed=101)
    np.testing.assert_array_equal(fit.reconstruct(x_oos), core.reconstruct(x_oos))
    np.testing.assert_array_equal(fit.encode(x_oos), core.encode(x_oos))


def test_reconstruct_encode_oos_hybrid_split_parse_exercised() -> None:
    """Exercise the Rust `_hybrid_linear_images_for_oos` port at the boundary.

    A natural hybrid collapse is gate/data-dependent and not reliably forced
    from a small Python fit, so — following the established #1204/#1026 pattern —
    a schema-valid `hybrid_split` (exactly the shape the FFI emits) is injected
    onto a real euclidean d=1 fit and carried through `to_dict()`. Both the
    dataclass (`ManifoldSAE._hybrid_linear_images_for_oos`) and the pyclass
    (`manifold_sae_hybrid_linear_images`) parse that same block into the OOS
    linear-image map and feed identical images to the identical OOS solve, so
    the reconstruction/encoding are bitwise equal IFF the two parsers agree.
    Three parser branches are covered: `v` absent (ordinary straight image),
    `v` present (collapse-rescue), and a `linear_image`-less entry (skipped).
    A plain-vs-injected sanity assertion guarantees the images are actually
    consumed, so this cannot pass vacuously with a silently-dropped parse."""
    n, p = 40, 5
    rng = np.random.default_rng(11)
    # Two 1-D linear latents -> two euclidean d=1 atoms, both collapse-eligible.
    t = rng.standard_normal((n, 2))
    mixing = rng.standard_normal((2, p))
    x = (t @ mixing + 0.02 * rng.standard_normal((n, p))).astype(np.float64)
    x -= x.mean(axis=0, keepdims=True)
    fit = gamfit.sae_manifold_fit(
        X=x, K=2, d_atom=1, atom_topology="euclidean", assignment="softmax",
        n_iter=8, random_state=0,
    )

    core_plain = ManifoldSaeCore(fit.to_dict())  # hybrid_split is None here

    # A hybrid_split shaped exactly as the FFI emits it (plain JSON scalars/lists
    # so it survives to_dict -> json.dumps). b0/b1/v are length p, as production.
    b0 = [0.10 * (i + 1) for i in range(p)]
    b1 = [0.05 * (p - i) for i in range(p)]
    v_unit = [1.0] + [0.0] * (p - 1)
    fit.hybrid_split = {
        "curved_atom_count": 0,
        "linear_atom_count": 2,
        "atoms": [
            {  # v absent -> ordinary straight image
                "atom": "atom_0", "kept_curved": False,
                "linear_image": {"atom_idx": 0, "t_bar": 0.3, "b0": b0, "b1": b1},
            },
            {  # v present -> collapse-rescue (target-aware) branch
                "atom": "atom_1", "kept_curved": False,
                "linear_image": {"atom_idx": 1, "t_bar": -0.2, "b0": b0, "b1": b1, "v": v_unit},
            },
            {"atom": "atom_2_dummy", "kept_curved": True},  # no linear_image -> skipped
        ],
    }
    assert fit._hybrid_linear_images_for_oos() is not None  # Python parse populated
    core_hyb = ManifoldSaeCore(fit.to_dict())

    x_oos = (rng.standard_normal((16, p))).astype(np.float64)
    x_oos -= x_oos.mean(axis=0, keepdims=True)

    recon_dc = fit.reconstruct(x_oos)
    recon_core = core_hyb.reconstruct(x_oos)
    np.testing.assert_array_equal(recon_dc, recon_core)
    np.testing.assert_array_equal(fit.encode(x_oos), core_hyb.encode(x_oos))
    # Not vacuous: the injected images must actually move the reconstruction, so
    # the parse is observable in the output (a dropped parse would match plain).
    assert not np.array_equal(core_plain.reconstruct(x_oos), recon_core), (
        "injected hybrid linear images did not change the OOS reconstruction — "
        "the parse is not being exercised"
    )


def test_steer_bitwise_equivalence_euclidean_duchon_centers() -> None:
    """Euclidean (degree-2 patch) atom: exercises the duchon_centers threading
    that the circle atom (no centers) does not."""
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
    core = ManifoldSaeCore(fit.to_dict())
    assert fit.atoms[0].basis in {"euclidean", "linear"}

    t_from = np.array([0.0, 0.0], dtype=np.float64)
    t_to = np.array([0.5, -0.25], dtype=np.float64)
    plan_dc = fit.steer(0, 0, 1.0, t_from, t_to)
    plan_core = core.steer(0, 0, 1.0, t_from, t_to)
    _assert_plans_bitwise_equal(plan_dc, plan_core)
