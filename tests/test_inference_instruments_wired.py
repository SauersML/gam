"""End-to-end Python reachability of the landed inference instruments.

Covers the three previously-unwired instruments now exposed through the
``gamfit`` Python API:

* #1013 functorial layer transport (``layer_transport_fit`` /
  ``layer_transport_ladder``),
* #984 anytime-valid structure discovery (``atom_birth_gate`` /
  ``e_bh_dictionary_certificate`` / ``split_likelihood_log_e`` /
  ``log_e_from_p_value``),
* #1109 KL-optimal steering-probe design (``plan_probe_for_contested_claim`` /
  ``select_probe_by_expected_evidence`` / ``expected_resolution_budget``),
* #939 Lawley likelihood-ratio Bartlett correction (``lawley_bartlett_factor`` /
  ``lawley_bartlett_factor_estimated_lambda``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")
pytest.importorskip("gamfit._rust")


def test_shape_controlled_census_replays_the_exact_full_pipeline() -> None:
    rows = np.arange(64, dtype=np.float64)
    activations = np.column_stack(
        [
            1.0e6 + rows,
            -2.0e6 + np.sin(rows),
            (rows % 7.0) - 3.0,
            np.cos(0.25 * rows),
        ]
    )
    pristine = activations.copy()
    calls: list[tuple[np.ndarray, int]] = []

    def pipeline(matrix: np.ndarray, seed: int) -> dict[str, float | int]:
        snapshot = matrix.copy()
        calls.append((snapshot, seed))
        # A real training pipeline is allowed to mutate its working matrix. The
        # runner must isolate that mutation from both later controls.
        matrix.fill(np.nan)
        return {"seed": seed, "checksum": float(snapshot.sum())}

    result = gamfit.run_shape_controlled_census(
        activations,
        pipeline,
        control_seed=2262,
        pipeline_seed=17,
    )
    assert isinstance(result, gamfit.ShapeControlledCensus)
    assert len(calls) == 3
    assert [seed for _, seed in calls] == [17, 17, 17]
    np.testing.assert_array_equal(calls[0][0], pristine)
    np.testing.assert_array_equal(activations, pristine)

    expected_shuffle = gamfit.shape_matched_control(
        pristine,
        "per_dimension_shuffle",
        seed=result.per_dimension_shuffle_seed,
    )
    expected_hadamard = gamfit.shape_matched_control(
        pristine,
        "covariance_exact_hadamard",
        seed=result.covariance_exact_hadamard_seed,
    )
    np.testing.assert_array_equal(calls[1][0], expected_shuffle)
    np.testing.assert_array_equal(calls[2][0], expected_hadamard)
    assert result.observed["checksum"] == pytest.approx(float(pristine.sum()))
    assert result.per_dimension_shuffle["checksum"] == pytest.approx(
        float(expected_shuffle.sum())
    )
    assert result.covariance_exact_hadamard["checksum"] == pytest.approx(
        float(expected_hadamard.sum())
    )


def test_shape_controlled_census_rejects_ambiguous_seeds_and_bad_data() -> None:
    def pipeline(matrix: np.ndarray, seed: int) -> tuple[tuple[int, ...], int]:
        return matrix.shape, seed

    with pytest.raises(TypeError, match="control_seed"):
        gamfit.run_shape_controlled_census(np.ones((4, 2)), pipeline, control_seed=True)
    with pytest.raises(ValueError, match="pipeline_seed"):
        gamfit.run_shape_controlled_census(np.ones((4, 2)), pipeline, pipeline_seed=-1)
    with pytest.raises(ValueError, match="finite"):
        gamfit.run_shape_controlled_census(
            np.array([[1.0, np.nan], [2.0, 3.0]]), pipeline
        )
    with pytest.raises(TypeError, match="complex dtype"):
        gamfit.run_shape_controlled_census(
            np.ones((4, 2), dtype=np.complex64) * (1.0 + 2.0j), pipeline
        )


def test_float32_shape_controls_preserve_dtype_seed_marginals_and_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gamfit._shape_census as shape_census

    rows = np.arange(16_384, dtype=np.float32)
    phase = np.float32(2.0 * np.pi) * rows / np.float32(rows.size)
    activations = np.column_stack(
        [
            np.float32(10_000.0) + np.float32(2.0) * np.cos(phase),
            np.float32(-20_000.0)
            + np.float32(0.8) * np.cos(phase)
            + np.float32(1.2) * np.sin(phase),
            np.float32(5_000.0)
            - np.float32(0.6) * np.cos(phase)
            + np.float32(0.3) * np.sin(phase),
        ]
    ).astype(np.float32, copy=False)

    def forbid_float64_control(*args, **kwargs):
        raise AssertionError("float32 census dispatched through the float64 native control")

    monkeypatch.setattr(shape_census, "shape_matched_control", forbid_float64_control)
    calls: list[tuple[np.ndarray, int]] = []

    def pipeline(matrix: np.ndarray, seed: int) -> tuple[np.dtype, tuple[int, ...], int]:
        assert matrix.dtype == np.float32
        assert matrix.nbytes == matrix.size * np.dtype(np.float32).itemsize
        calls.append((matrix.copy(), seed))
        return matrix.dtype, matrix.shape, seed

    result = gamfit.run_shape_controlled_census(
        activations,
        pipeline,
        control_seed=2262,
        pipeline_seed=17,
    )
    assert len(calls) == 3
    assert [seed for _, seed in calls] == [17, 17, 17]
    assert all(matrix.dtype == np.float32 for matrix, _ in calls)

    shuffle = gamfit.shape_matched_control_f32(
        activations,
        "per_dimension_shuffle",
        seed=result.per_dimension_shuffle_seed,
    )
    repeated_shuffle = gamfit.shape_matched_control_f32(
        activations,
        "per_dimension_shuffle",
        seed=result.per_dimension_shuffle_seed,
    )
    assert shuffle.dtype == np.float32
    np.testing.assert_array_equal(shuffle, repeated_shuffle)
    np.testing.assert_array_equal(calls[1][0], shuffle)
    for col in range(activations.shape[1]):
        np.testing.assert_array_equal(np.sort(shuffle[:, col]), np.sort(activations[:, col]))

    hadamard = gamfit.shape_matched_control_f32(
        activations,
        "covariance_exact_hadamard",
        seed=result.covariance_exact_hadamard_seed,
    )
    repeated_hadamard = gamfit.shape_matched_control_f32(
        activations,
        "covariance_exact_hadamard",
        seed=result.covariance_exact_hadamard_seed,
    )
    assert hadamard.dtype == np.float32
    np.testing.assert_array_equal(hadamard, repeated_hadamard)
    np.testing.assert_array_equal(calls[2][0], hadamard)
    np.testing.assert_allclose(
        hadamard.astype(np.float64).mean(axis=0),
        activations.astype(np.float64).mean(axis=0),
        rtol=2.0e-6,
        atol=2.0e-3,
    )
    target_covariance = np.cov(activations.astype(np.float64), rowvar=False, bias=True)
    realized_covariance = np.cov(hadamard.astype(np.float64), rowvar=False, bias=True)
    scales = np.sqrt(np.diag(target_covariance))
    tolerance = 5.0e-3 * np.maximum(np.outer(scales, scales), 1.0e-8)
    np.testing.assert_array_less(
        np.abs(realized_covariance - target_covariance), tolerance
    )


def test_native_contiguous_source_is_shared_but_callbacks_are_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gamfit._shape_census as shape_census

    original = np.arange(96, dtype=np.float32).reshape(32, 3)
    native_sources: list[np.ndarray] = []
    callback_matrices: list[np.ndarray] = []

    def recording_control(
        source: np.ndarray, kind: str, *, seed: int
    ) -> np.ndarray:
        del kind, seed
        native_sources.append(source)
        assert source.flags.c_contiguous
        assert not source.flags.writeable
        return source.copy(order="C")

    monkeypatch.setattr(shape_census, "shape_matched_control_f32", recording_control)

    def pipeline(matrix: np.ndarray, seed: int) -> int:
        assert seed == 11
        assert matrix.flags.c_contiguous
        assert matrix.flags.writeable
        assert not np.shares_memory(matrix, original)
        callback_matrices.append(matrix)
        return matrix.shape[0]

    gamfit.run_shape_controlled_census(original, pipeline)
    assert original.flags.writeable
    assert len(native_sources) == 2
    assert all(np.shares_memory(source, original) for source in native_sources)
    assert len(callback_matrices) == 3
    assert all(
        not np.shares_memory(left, right)
        for index, left in enumerate(callback_matrices)
        for right in callback_matrices[index + 1 :]
    )


def test_shape_controlled_census_converts_non_native_dtype_once_to_float64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gamfit._shape_census as shape_census

    def forbid_float32_control(*args, **kwargs):
        raise AssertionError("non-float32 census dispatched through the float32 native control")

    monkeypatch.setattr(shape_census, "shape_matched_control_f32", forbid_float32_control)
    source = np.arange(96, dtype=np.int16).reshape(32, 3)

    class CountingArrayLike:
        dtype = source.dtype

        def __init__(self) -> None:
            self.conversions = 0

        def __array__(self, dtype=None, copy=None):
            self.conversions += 1
            assert np.dtype(dtype) == np.dtype(np.float64)
            return source.astype(dtype, copy=True)

    counted_source = CountingArrayLike()
    seen_dtypes: list[np.dtype] = []

    def pipeline(matrix: np.ndarray, seed: int) -> int:
        seen_dtypes.append(matrix.dtype)
        return seed

    gamfit.run_shape_controlled_census(counted_source, pipeline)
    assert counted_source.conversions == 1
    assert seen_dtypes == [np.dtype(np.float64)] * 3


def test_layer_transport_fit_reaches_python():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.0, 2.0 * math.pi, size=200))
    # Identity-ish circle->circle map plus small wiggle: degree 1, low defect.
    s = (t + 0.05 * np.sin(t)) % (2.0 * math.pi)
    report = gamfit.layer_transport_fit(t, s, "circle", "circle")
    assert report["degree"] == 1
    assert report["topology_preserved"] is True
    assert report["isometry_defect"] >= 0.0
    assert report["isometry_defect_se"] >= 0.0


def test_fit_transport_object_inverts_and_composes():
    # The live, invertible transport object (vs the summary dict). Its invert
    # is what lets a caller form g_B o g_A^-1 from two fitted transports.
    #
    # The warps are derivative-bounded strictly-monotone maps so the span-exact
    # monotonicity certificate accepts them everywhere on [0, 1] (a t^1.5 warp
    # has h'(0)=0 at the left endpoint and the whole-domain certificate would,
    # correctly, refuse to invert it).
    frm = np.linspace(0.0, 1.0, 64)
    # g_A(t) = t + 0.25*sin(2*pi*t)/(2*pi): h' = 1 + 0.25*cos(2*pi*t) in
    # [0.75, 1.25], strictly increasing.
    a_warp = frm + 0.25 * np.sin(2.0 * math.pi * frm) / (2.0 * math.pi)
    g_a = gamfit.fit_transport(frm, a_warp, "interval", "interval")
    assert g_a.topology_preserved is True
    assert g_a.isometry_defect >= 0.0
    assert set(g_a.report().keys()) >= {"degree", "topology_preserved", "isometry_defect"}

    # eval / invert round-trip.
    probe = np.array([0.2, 0.5, 0.8])
    assert np.allclose(g_a.invert(g_a.eval(probe)), probe, atol=1e-6)

    # phi = g_B o g_A^-1 from two fitted transports. With no closed form for the
    # inverse of g_A, assert the composition is internally consistent: applying
    # g_A^-1 then evaluating g_B reproduces g_B at the true pre-image, and the
    # composed map is strictly monotone (an honest g_B o g_A^-1).
    # g_B(t) = t - 0.25*sin(2*pi*t)/(2*pi): h' = 1 - 0.25*cos(2*pi*t) in
    # [0.75, 1.25], strictly increasing.
    b_warp = frm - 0.25 * np.sin(2.0 * math.pi * frm) / (2.0 * math.pi)
    g_b = gamfit.fit_transport(frm, b_warp, "interval", "interval")
    # Targets in g_A's image; recover the pre-image and compose.
    t_true = np.array([0.2, 0.5, 0.8])
    y = g_a.eval(t_true)
    t_pre = g_a.invert(y)
    assert np.allclose(t_pre, t_true, atol=1e-6)
    phi = g_b.eval(g_a.invert(y))
    # phi(y) == g_B(g_A^-1(y)) == g_B(t_true), the composition law.
    assert np.allclose(phi, g_b.eval(t_true), atol=1e-6)
    # And the composed map is monotone increasing across the probe.
    assert np.all(np.diff(phi) > 0.0)


def test_fit_transport_invert_rejects_non_finite_targets():
    # The #1361 finiteness guard, exercised through the Python API: a non-finite
    # inverse target must raise ValueError rather than silently returning a
    # boundary coordinate.
    frm = np.linspace(0.0, 1.0, 64)
    g = gamfit.fit_transport(frm, 0.5 * frm, "interval", "interval")
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            g.invert(np.array([bad]))


def test_fit_transport_invert_rejects_folded_map():
    # A non-topology-preserving (folded) fit must refuse to invert. The target
    # is non-monotone in the source coordinate, so the fitted map folds and the
    # span-exact monotonicity certificate rejects the inverse.
    frm = np.linspace(0.0, 1.0, 200)
    # A clear fold: rises then falls (a non-monotone, fold-bearing map).
    target = np.sin(2.0 * math.pi * frm)
    lo, hi = float(target.min()), float(target.max())
    g = gamfit.fit_transport(frm, target, "interval", "interval")
    assert g.topology_preserved is False
    with pytest.raises(ValueError):
        g.invert(np.array([0.5 * (lo + hi)]))


def test_fit_transport_degree_minus_one_circle_round_trips():
    # A degree -1 (orientation-reversing) circle cover must invert correctly
    # through the Python API.
    t = np.linspace(0.0, 2.0 * math.pi, 256, endpoint=False)
    s = (-t + 0.4 + 0.15 * np.sin(t)) % (2.0 * math.pi)
    g = gamfit.fit_transport(t, s, "circle", "circle")
    assert g.degree == -1
    assert g.topology_preserved is True
    probe = (np.arange(7) + 0.5) * (2.0 * math.pi / 7.0)
    back = g.invert(g.eval(probe))
    # Compare modulo 2π.
    d = np.abs((back - probe + math.pi) % (2.0 * math.pi) - math.pi)
    assert np.all(d < 1e-4)


def test_structure_discovery_gate_and_certificate():
    # split-LR log e-value is just the likelihood gap.
    assert gamfit.split_likelihood_log_e(-8.0, -10.0) == pytest.approx(2.0)

    # Gate certifies once the running e-process supremum crosses 1/alpha.
    gate = gamfit.atom_birth_gate(0.05)
    assert gate.certified() is False
    gate.absorb_shard(-8.0, -10.0)  # log e = 2.0
    gate.absorb_shard(-8.0, -10.0)  # cumulative 4.0 > log(1/0.05)
    assert gate.certified() is True
    verdict = gate.verdict()
    assert verdict["verdict"] == "certified"
    assert verdict["log_e"] == pytest.approx(4.0)

    # e-BH dictionary certificate confirms the overwhelming claim only.
    confirmed = gamfit.e_bh_dictionary_certificate([25.0, 0.01, -0.2, 0.0], 0.05)
    assert confirmed == [0]

    # p->e calibration is the conservative 1/p lower bound family.
    assert gamfit.log_e_from_p_value(0.04) == pytest.approx(math.log(2.5))


def test_kl_optimal_probe_design_reaches_python():
    # Row-aligned candidate probes. Candidate 0 has a large raw delta, but both
    # hypotheses predict the same response, so it teaches the e-process nothing.
    # Candidate 1 has a smaller raw delta but separates the hypotheses along the
    # high-Fisher axis, so it is the design-optimal probe.
    delta = np.array([[10.0, 0.0], [0.0, 1.0]])
    predicted_null = np.array([[4.0, 4.0], [0.0, 0.0]])
    predicted_alt = np.array([[4.0, 4.0], [1.0, 0.2]])
    fisher = np.array([[2.0, 0.0], [0.0, 0.5]])

    selected = gamfit.select_probe_by_expected_evidence(
        delta, predicted_null, predicted_alt, fisher
    )
    assert selected is not None
    assert selected["probe"] == 1
    assert selected["expected_log_growth"] == pytest.approx(1.01)
    assert selected["delta"] == [0.0, 1.0]

    assert gamfit.expected_resolution_budget(0.05, 1.01) == pytest.approx(
        -math.log(0.05) / 1.01
    )

    from_zero = gamfit.plan_probe_for_contested_claim(
        delta, predicted_null, predicted_alt, fisher, 0.05
    )
    assert from_zero is not None
    assert from_zero["probe"] == 1
    assert from_zero["budget_from_scratch"] == pytest.approx(-math.log(0.05) / 1.01)
    assert from_zero["budget_remaining"] == pytest.approx(
        from_zero["budget_from_scratch"]
    )

    halfway = gamfit.plan_probe_for_contested_claim(
        delta, predicted_null, predicted_alt, fisher, 0.05, current_log_e=1.5
    )
    assert halfway is not None
    assert halfway["budget_remaining"] == pytest.approx((-math.log(0.05) - 1.5) / 1.01)
    assert halfway["budget_remaining"] < from_zero["budget_remaining"]

    blind = gamfit.plan_probe_for_contested_claim(
        delta[:1], predicted_null[:1], predicted_alt[:1], fisher, 0.05
    )
    assert blind is None


def test_lawley_bartlett_factor_exponential_fixture():
    # Exponential (Gamma-log, phi=1), intercept-only: ε_0 = 0 so the factor is
    # the certified textbook c = 1 + 1/(6n).
    n = 32
    design = np.ones((n, 1))
    eta = np.full(n, 0.4)
    out = gamfit.lawley_bartlett_factor(
        design, "gamma", eta, 0, 1, 1.0, dispersion=1.0, lr_statistic=5.0
    )
    assert out["bartlett_factor"] == pytest.approx(1.0 + 1.0 / (6.0 * n), rel=1e-8)
    # Corrected statistic divides by the factor; corrected p-value is larger.
    assert out["corrected_statistic"] == pytest.approx(5.0 / out["bartlett_factor"])
    assert out["p_value_corrected"] >= out["p_value_uncorrected"]


def test_lawley_bartlett_factor_estimated_lambda_reaches_python():
    n = 50
    z = np.linspace(-0.5, 0.5, n)
    design = np.column_stack([np.ones(n), z])
    eta = 0.3 + 0.6 * z
    penalty = np.diag([0.0, 3.0])
    rho_cov = np.array([[0.8]])

    conditional = gamfit.lawley_bartlett_factor(
        design,
        "poisson",
        eta,
        1,
        2,
        1.0,
        penalty=penalty,
        lr_statistic=4.0,
    )
    estimated = gamfit.lawley_bartlett_factor_estimated_lambda(
        design,
        "poisson",
        eta,
        1,
        2,
        1.0,
        penalty=penalty,
        components=[penalty],
        rho_cov=rho_cov,
        lr_statistic=4.0,
    )

    assert estimated["bartlett_factor_conditional"] == pytest.approx(
        conditional["bartlett_factor"]
    )
    assert estimated["rho_variation_shift"] != pytest.approx(0.0, abs=1e-10)
    assert estimated["mean_shift"] == pytest.approx(
        estimated["mean_shift_conditional"] + estimated["rho_variation_shift"]
    )
    assert estimated["bartlett_factor"] == pytest.approx(
        1.0 + estimated["mean_shift"]
    )
    assert estimated["corrected_statistic"] == pytest.approx(
        4.0 / estimated["bartlett_factor"]
    )
    assert estimated["p_value_corrected"] >= 0.0

    with pytest.raises(ValueError, match="rho_cov must be symmetric"):
        gamfit.lawley_bartlett_factor_estimated_lambda(
            design,
            "poisson",
            eta,
            1,
            2,
            1.0,
            penalty=penalty,
            components=[penalty, penalty],
            rho_cov=np.array([[1.0, 0.25], [0.20, 1.0]]),
        )


def test_smooth_significance_auto_applies_lawley_and_surfaces_material_flag():
    # #939/#1063 deliverable (4): the magic per-term LR Bartlett correction is
    # auto-applied on the smooth-term significance test and its >10% materiality
    # diagnostic is surfaced alongside the first-order p-value.
    rng = np.random.default_rng(11)
    n = 80
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    # A genuine non-flat smooth (so the term is significant) with Poisson noise.
    mu = np.exp(0.5 + 1.2 * np.sin(2.0 * math.pi * x))
    y = rng.poisson(mu).astype(float)
    frame = {"y": y, "x": x}
    model = gamfit.fit(frame, "y ~ s(x)", family="poisson")

    rows = model.smooth_significance(frame)
    assert rows, "expected at least one penalized smooth term"
    row = rows[0]
    # Every documented field is present.
    for key in (
        "name",
        "statistic_lr",
        "ref_df",
        "bartlett_factor",
        "bartlett_factor_conditional",
        "rho_variation_shift",
        "statistic_corrected",
        "p_value_uncorrected",
        "p_value_corrected",
        "material",
        "correction_provenance",
    ):
        assert key in row, f"smooth_significance row missing '{key}'"
    # Poisson carries closed-form Lawley jets, so the correction auto-applies.
    assert row["correction_provenance"] == "lawley_lr_estimated_lambda"
    # The corrected statistic is the raw LR divided by the Bartlett factor.
    assert row["statistic_corrected"] == pytest.approx(
        row["statistic_lr"] / row["bartlett_factor"], rel=1e-9
    )
    # `material` is a bool and is consistent with the 10% rule it documents.
    assert isinstance(row["material"], bool)
    factor_move = abs(row["bartlett_factor"] - 1.0)
    p_lo = min(row["p_value_uncorrected"], row["p_value_corrected"])
    p_hi = max(row["p_value_uncorrected"], row["p_value_corrected"])
    p_move = (p_hi - p_lo) / max(p_hi, np.finfo(float).tiny)
    assert row["material"] == bool(factor_move > 0.10 or p_move > 0.10)


def test_glm_full_conformal_bernoulli_reaches_python_and_covers():
    # #942: the exact full-conformal engine for a canonical-link GLM. The
    # Bernoulli support {0,1} is exhaustive, so the returned set is the exact
    # full-conformal set with finite-sample coverage >= 1 - alpha.
    rng = np.random.default_rng(7)
    p = 2
    s_lambda = np.eye(p)  # ridge-penalized logistic; frozen smoothing.
    x_star = np.array([1.0, 0.6])
    alpha = 0.2

    # Structural reachability + the conservative tie convention on a small fit.
    n = 24
    x = np.column_stack([np.ones(n), rng.normal(size=n)])
    eta = x @ np.array([0.3, 1.1])
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    out = gamfit.glm_full_conformal(x, y, s_lambda, x_star, "bernoulli", alpha)
    assert out["n_augmented"] == n + 1
    assert set(out["candidates"]) == {0.0, 1.0}
    # p-values are honest conformal p-values in (0, 1]; membership = p > alpha.
    for z, pv in zip(out["candidates"], out["p_values"]):
        assert 0.0 < pv <= 1.0
        assert (z in out["members"]) == (pv > alpha)
    # The exactness witness is finite (homotopy certified or cold-refit).
    assert math.isfinite(out["max_beta_error_bound"])
    assert out["max_beta_error_bound"] >= 0.0

    # OBJECTIVE finite-sample coverage: over many independent draws of a
    # (training set, fresh test outcome) pair, the exact set covers the held-out
    # outcome at >= 1 - alpha. Full conformal's guarantee is finite-sample, so
    # this must hold at the modest n below where split conformal would not.
    trials = 300
    n_small = 12
    beta_true = np.array([0.2, 0.9])
    covered = 0
    for _ in range(trials):
        xt = np.column_stack([np.ones(n_small), rng.normal(size=n_small)])
        et = xt @ beta_true
        yt = (rng.uniform(size=n_small) < 1.0 / (1.0 + np.exp(-et))).astype(float)
        p_star = 1.0 / (1.0 + np.exp(-(x_star @ beta_true)))
        y_star = float(rng.uniform() < p_star)
        res = gamfit.glm_full_conformal(
            xt, yt, s_lambda, x_star, "bernoulli", alpha
        )
        if y_star in res["members"]:
            covered += 1
    # Allow a small Monte-Carlo slack below the nominal 1 - alpha = 0.8.
    assert covered / trials >= 0.8 - 3.0 * math.sqrt(0.8 * 0.2 / trials)
