"""Hard failing accuracy tests for SAE-manifold — out-of-sample, multi-seed.

These complement `test_sae_manifold_curved_beats_linear.py` which fixes
in-sample R^2 on a single seed. Here we exercise the engine harder:

  - Held-out OOS R^2 (train on half, score on the other half).
  - Multi-seed mean R^2 (a single bad draw shouldn't masquerade as
    a correct fix).
  - Sphere topology on actual S^2 data (the periodic-1D bug doesn't
    cover the sphere chart; this is an orthogonal check).

All assertions are non-skippable. Failing here means the SAE-manifold
engine isn't actually generalising — it might be overfitting in-sample
or it might be broken on a different topology than the canonical circle.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")

# #1512 triage: these SAE-manifold OOS-accuracy fits run well past the standard
# Python-API CI runner budget (>240s for the file in triage), so they are tagged
# slow and excluded from the directory-level `-m "not slow"` CI step while still
# being collected (and run by a bare `pytest tests/` locally).
pytestmark = pytest.mark.slow


def _circle_data(
    n: int, p: int, noise: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    harm = np.column_stack([np.cos(theta), np.sin(theta)])
    mixing = rng.normal(size=(harm.shape[1], p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    z = harm @ mixing + noise * rng.normal(size=(n, p))
    z -= z.mean(axis=0, keepdims=True)
    return z, theta


def _sphere_data(
    n: int, p: int, noise: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Spherical harmonic Y_1^0 / Y_1^1 (lat-lon parametrised) mixed into p
    output dims. Intrinsic latent is (lat, lon) ∈ [-π/2, π/2] × [0, 2π)."""
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-math.pi / 2.0, math.pi / 2.0, n)
    lon = rng.uniform(0.0, 2.0 * math.pi, n)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z_axis = np.sin(lat)
    harm = np.column_stack([x, y, z_axis])
    mixing = rng.normal(size=(harm.shape[1], p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    out = harm @ mixing + noise * rng.normal(size=(n, p))
    out -= out.mean(axis=0, keepdims=True)
    angles = np.column_stack([lat, lon])
    return out, angles


def _r2(x: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 13])
def test_curved_circle_atom_in_sample_r2_per_seed(seed: int):
    """Curved K=1 atom on a clean 1-harmonic circle should hit R^2 >= 0.9
    on every seed. If any seed drops below this it usually means the
    optimization got stuck in a bad local minimum (initialisation bug)."""
    z, _ = _circle_data(n=400, p=64, noise=0.04, seed=seed)
    fit = gamfit.sae_manifold_fit(
        X=z,
        K=1,
        atom_basis="periodic",
        d_atom=2,
        assignment="ibp_map",
        n_iter=50,
        learning_rate=0.04,
        random_state=seed,
    )
    r2 = _r2(z, fit.fitted)
    assert r2 >= 0.9, (
        f"seed={seed}: curved K=1 atom failed to recover 1-harmonic circle; "
        f"R^2 = {r2:.4f}"
    )


def test_curved_circle_atom_mean_r2_across_seeds():
    """Across 5 seeds, mean R^2 should be well above 0.9. A single seed
    landing at 0.92 while the others sit at 0.6 isn't acceptable."""
    scores = []
    for seed in range(5):
        z, _ = _circle_data(n=400, p=64, noise=0.04, seed=seed)
        fit = gamfit.sae_manifold_fit(
            X=z,
            K=1,
            atom_basis="periodic",
            d_atom=2,
            assignment="ibp_map",
            n_iter=50,
            learning_rate=0.04,
            random_state=seed,
        )
        scores.append(_r2(z, fit.fitted))
    mean = float(np.mean(scores))
    worst = float(np.min(scores))
    assert mean >= 0.92, f"mean R^2 across seeds = {mean:.4f}; per-seed {scores}"
    assert worst >= 0.85, f"worst-seed R^2 = {worst:.4f}; per-seed {scores}"


def test_curved_circle_atom_oos_r2():
    """Train the curved atom on the first 200 rows, then score on a
    held-out 200 rows drawn from the same underlying circle. The atom
    parameters are fixed by training; OOS R^2 should not collapse
    relative to in-sample R^2 (the model is supposed to recover the
    intrinsic structure, not memorise the training set).
    """
    z_full, theta = _circle_data(n=400, p=64, noise=0.04, seed=42)
    z_train = z_full[:200]
    z_test = z_full[200:]

    fit = gamfit.sae_manifold_fit(
        X=z_train,
        K=1,
        atom_basis="periodic",
        d_atom=2,
        assignment="ibp_map",
        n_iter=50,
        learning_rate=0.04,
        random_state=0,
    )
    in_sample = _r2(z_train, fit.fitted)

    # Reuse the trained atom to score held-out rows. The fit result must
    # expose enough state for this — a working engine offers a `predict`
    # or `reconstruct` method on the fit object.
    assert hasattr(fit, "reconstruct") or hasattr(fit, "predict"), (
        "SaeManifoldFitResult must expose an OOS prediction method "
        "(`reconstruct` or `predict`); neither is implemented. "
        "In-sample R^2 was {:.4f} so training works, but held-out scoring "
        "is impossible without a predict surface — this is a regression.".format(
            in_sample
        )
    )
    if hasattr(fit, "reconstruct"):
        oos_fitted = fit.reconstruct(z_test)
    else:
        oos_fitted = fit.predict(z_test)

    oos = _r2(z_test, oos_fitted)
    assert oos >= 0.85, (
        f"OOS R^2 collapsed: in-sample {in_sample:.4f}, OOS {oos:.4f}. "
        f"The curved atom should generalise on data drawn from the same "
        f"distribution; if OOS << in-sample the engine is memorising "
        f"training-row latents instead of recovering the manifold."
    )
    assert (in_sample - oos) <= 0.1, (
        f"OOS R^2 ({oos:.4f}) lags in-sample ({in_sample:.4f}) by too much; "
        f"gap = {in_sample - oos:.4f}, expected <= 0.10."
    )


def test_oos_uses_fit_time_hyperparameters():
    """The OOS predict path must solve the *same* regularized problem as
    the fit: ``reconstruct``/``predict`` must thread the fit-time alpha,
    tau, sparsity_strength, smoothness, and the effective learning_rate
    into the Rust OOS solver — not hardcoded alpha=1.0/tau=0.5/
    sparsity=1.0/smoothness=1.0/lr=0.04. We fit with deliberately
    non-default knobs and assert the fit object persists exactly those
    values, then confirm a serialize/deserialize round-trip reproduces
    OOS predictions bit-exactly (which is only possible if every knob is
    carried, not silently reset to a default).
    """
    z_full, _ = _circle_data(n=400, p=64, noise=0.04, seed=3)
    z_train = z_full[:200]
    z_test = z_full[200:]

    fit = gamfit.sae_manifold_fit(
        X=z_train,
        K=1,
        atom_basis="periodic",
        d_atom=2,
        assignment="ibp_map",
        n_iter=37,
        learning_rate=0.07,
        tau=0.3,
        sparsity_strength=0.5,
        smoothness=2.0,
        random_state=5,
    )

    # The fit-time knobs must be persisted on the fit object verbatim so
    # OOS re-estimation reproduces the training-time regularized problem.
    assert fit.tau == pytest.approx(0.3)
    assert fit.sparsity_strength == pytest.approx(0.5)
    assert fit.smoothness == pytest.approx(2.0)
    assert fit.learning_rate == pytest.approx(0.07)
    assert fit.max_iter == 37
    assert fit.random_state == 5
    # None of these may collapse to the old hardcoded OOS defaults.
    assert not (
        fit.tau == pytest.approx(0.5)
        and fit.sparsity_strength == pytest.approx(1.0)
        and fit.smoothness == pytest.approx(1.0)
        and fit.learning_rate == pytest.approx(0.04)
    ), "OOS hyperparameters collapsed to the old hardcoded defaults."

    oos_direct = fit.reconstruct(z_test)

    # Round-trip through serialization: the deserialized fit must carry
    # the same hyperparameters and therefore reproduce OOS predictions
    # bit-exactly. If any knob were dropped on the way through, the OOS
    # solve would diverge here.
    restored = gamfit.ManifoldSAE.from_dict(fit.to_dict())
    assert restored.tau == pytest.approx(0.3)
    assert restored.sparsity_strength == pytest.approx(0.5)
    assert restored.smoothness == pytest.approx(2.0)
    assert restored.learning_rate == pytest.approx(0.07)
    assert restored.max_iter == 37
    assert restored.random_state == 5
    oos_restored = restored.reconstruct(z_test)
    np.testing.assert_allclose(oos_restored, oos_direct, rtol=0.0, atol=0.0)


def test_curved_sphere_atom_on_sphere_data():
    """Sphere topology check: a Sphere atom on S^2 ground-truth data
    should reach R^2 >= 0.85. This catches bugs that the periodic-1D
    case doesn't — for example, if `_sphere_chart_basis` is mis-using
    lat/lon ranges, this test fails while the periodic test passes.
    """
    z, _ = _sphere_data(n=500, p=48, noise=0.03, seed=0)
    fit = gamfit.sae_manifold_fit(
        X=z,
        K=1,
        atom_basis="sphere",
        d_atom=2,
        assignment="ibp_map",
        n_iter=50,
        learning_rate=0.04,
        random_state=0,
    )
    r2 = _r2(z, fit.fitted)
    assert r2 >= 0.85, (
        f"sphere atom on S^2 data: R^2 = {r2:.4f}. A single Sphere atom "
        f"can represent any first-order spherical harmonic mixture; "
        f"if this is below threshold the sphere-chart basis or its "
        f"jacobian is wrong."
    )


def test_sae_manifold_oos_reconstruction_idempotence():
    """OOS prediction is a pure function of (input data, trained atoms,
    frozen hyperparameters): the Rust OOS solver uses fixed initialization,
    a deterministic Newton optimizer (Arrow-Schur + Armijo, no RNG), and
    frozen decoder weights. Calling ``reconstruct`` repeatedly on the same
    held-out matrix must therefore return bit-exactly identical arrays.

    ``test_oos_uses_fit_time_hyperparameters`` validates this indirectly via
    a serialize/deserialize round-trip; this test asserts the invariant
    directly — three identical OOS calls in a row — and also confirms the
    input matrix is not mutated in place by the FFI.
    """
    z_full, _ = _circle_data(n=400, p=64, noise=0.04, seed=11)
    z_train = z_full[:200]
    z_test = z_full[200:]

    fit = gamfit.sae_manifold_fit(
        X=z_train,
        K=1,
        atom_basis="periodic",
        d_atom=2,
        assignment="ibp_map",
        n_iter=40,
        learning_rate=0.04,
        random_state=0,
    )

    z_test_guard = z_test.copy()
    outputs = [fit.reconstruct(z_test) for _ in range(3)]

    for r in outputs:
        assert np.all(np.isfinite(r)), "OOS reconstruction produced NaN/Inf"

    # All three calls must agree bit-exactly (no RNG, no mutable state).
    np.testing.assert_array_equal(
        outputs[0],
        outputs[1],
        err_msg="OOS reconstruct is not idempotent: call 1 != call 2",
    )
    np.testing.assert_array_equal(
        outputs[0],
        outputs[2],
        err_msg="OOS reconstruct is not idempotent: call 1 != call 3",
    )

    # The OOS input must not be mutated in place by the FFI round-trip.
    np.testing.assert_array_equal(
        z_test,
        z_test_guard,
        err_msg="reconstruct mutated its input array in place",
    )


def test_near_duplicate_training_input_takes_oos_path_not_cache():
    """Regression for S17 (#1235): the cached-training shortcut must require
    BIT-EXACT input. A tiny perturbation of the training matrix is NOT the
    training data, so ``reconstruct`` / ``encode`` must run the OOS solve and
    produce input-specific output — never silently return the cached training
    fit (which the old ``np.allclose`` shortcut did for any near-duplicate
    within tolerance).
    """
    z_train, _ = _circle_data(n=200, p=48, noise=0.04, seed=11)
    fit = gamfit.sae_manifold_fit(
        X=z_train,
        K=1,
        atom_basis="periodic",
        d_atom=2,
        assignment="ibp_map",
        n_iter=50,
        learning_rate=0.04,
        random_state=0,
    )

    # Exact training input still hits the cache (bit-identical to fit.fitted).
    cached = fit.reconstruct(z_train)
    np.testing.assert_array_equal(
        cached,
        fit.fitted,
        err_msg="exact training input must return the cached training fit",
    )

    # A perturbation well inside np.allclose's default tolerance (rtol=1e-5,
    # atol=1e-8) but NOT bit-exact: the old shortcut would have returned the
    # cache for this; the exact shortcut must route it through the OOS solve.
    z_near = z_train.copy()
    z_near[0, 0] += 1e-7
    assert np.allclose(z_near, z_train), (
        "test perturbation must be within np.allclose tolerance to exercise "
        "the regression (otherwise the old code would also miss the cache)"
    )
    assert not np.array_equal(z_near, z_train)

    near = fit.reconstruct(z_near)
    # The OOS path must produce a result that reflects the perturbed input,
    # not a verbatim copy of the cached training fit.
    assert not np.array_equal(near, fit.fitted), (
        "near-duplicate input silently returned the cached training "
        "reconstruction — the tolerance-based shortcut is back"
    )
    assert np.all(np.isfinite(near))

    # encode shares the same exact-match guard.
    enc_exact = fit.encode(z_train)
    enc_near = fit.encode(z_near)
    np.testing.assert_array_equal(
        enc_exact,
        fit.assignments,
        err_msg="exact training input must return cached assignments",
    )
    assert not np.array_equal(enc_near, fit.assignments), (
        "near-duplicate input silently returned cached encode assignments"
    )
