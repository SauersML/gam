"""Regression contract for #2299: fitted predictors have one typed affine API.

The old public design-matrix endpoint returned a bare matrix for ordinary GAMs
and rejected link-wiggle fits. That shape encoded neither a model offset nor
the coefficient or covariance coordinate system. A link-wiggle predictor is
affine at its fitted state as ``offset + X @ beta_mean + B(index) @ beta_w``;
critically, #2141 requires ``B`` to be evaluated at the saved frozen index, not
at the de-aliased base. The joint frame is necessary for mean uncertainty and
mean--wiggle cross-covariance to remain available to external contrasts.

The DESIGN-MATRIX CONTRACT (frame name, offset/matrix/coefficient shapes, the
``offset + matrix @ coefficients == linear_predictor`` identity, the frozen
#2141 index, and all three covariance definitions) is a property of the exact
affine representation, independent of which convergence lane produced the fit.
The two contract tests below therefore run on well-conditioned fixtures that
converge cleanly (mirroring the green
``bug_hunt_1602_design_matrix_coef_reproduces_linear_predictor_test`` and
``bug_hunt_flexible_link_engages_and_predicts_test`` fixtures). Two OTHER lanes
that exercise the same contract but currently blow up before the fit exists are
preserved verbatim as honest red gates at the bottom of this file:

  * ``test_ordinary_affine_design_reml_offset_smoothing_boundary_red_gate`` --
    a smooth-of-x model offset drives a REML-with-offset smoothing-boundary
    non-stationarity for ``s(x)`` + offset (gaussian).
  * ``test_link_wiggle_affine_design_flex_link_joint_newton_blowup_red_gate`` --
    the ``s(x)`` + ``flexible_link=True`` binomial joint-Newton blow-up
    (#979 / #1596; min-eig ~ -1e200, degenerate fit, covariance unavailable).

Those gates keep the contract assertions unweakened; they simply cannot pass
until their convergence lanes are fixed, and they will flip green when they are.
"""

import numpy as np
from scipy.stats import norm

import gamfit


def _linear_predictor(model, data) -> np.ndarray:
    prediction = model.predict(data, return_type="dict")
    return np.asarray(prediction["linear_predictor"], dtype=float)


def _assert_affine_identity(model, data, expected_frame: str) -> gamfit.AffineDesign:
    affine = model.design_matrix(data)
    assert isinstance(affine, gamfit.AffineDesign)
    assert affine.coefficient_frame == expected_frame
    assert affine.coefficient_start == 0
    assert affine.coefficient_stop == affine.coefficients.shape[0]
    assert affine.coefficient_slice == slice(0, affine.coefficients.shape[0])
    assert affine.offset.ndim == 1
    assert affine.matrix.ndim == 2
    assert affine.coefficients.ndim == 1
    assert affine.matrix.shape == (
        affine.offset.shape[0],
        affine.coefficients.shape[0],
    )
    conditional = affine.covariance_conditional
    assert conditional is not None
    for covariance in (
        affine.covariance_conditional,
        affine.covariance_smoothing_corrected,
        affine.covariance_frequentist,
    ):
        if covariance is None:
            continue
        assert covariance.shape == (
            affine.coefficients.shape[0],
            affine.coefficients.shape[0],
        )
        assert np.all(np.isfinite(covariance))

    eta_variance = np.einsum(
        "ij,jk,ik->i",
        affine.matrix,
        conditional,
        affine.matrix,
    )
    assert np.all(np.isfinite(eta_variance))
    assert float(np.min(eta_variance)) >= -1e-12

    reconstructed = affine.offset + affine.matrix @ affine.coefficients
    expected = _linear_predictor(model, data)
    np.testing.assert_allclose(reconstructed, expected, rtol=2e-12, atol=2e-12)
    return affine


def test_ordinary_affine_design_exposes_model_offset_and_full_frame() -> None:
    # Well-conditioned ordinary GAM: a genuinely smooth mean signal with a known
    # per-row model offset that is NOT collinear with s(x), so REML has a clean
    # interior optimum. (The smooth-of-x offset that drives the REML boundary
    # non-stationarity is preserved in the red gate below.)
    rng = np.random.default_rng(2299)
    n = 800
    x = rng.uniform(0.0, 1.0, n)
    offset = rng.uniform(-0.5, 0.5, n)
    y = 0.6 + 1.2 * np.sin(2.0 * np.pi * x) + offset + rng.normal(0.0, 0.2, n)
    data = {"y": y, "x": x, "offset": offset}

    model = gamfit.fit(data, "y ~ s(x)", family="gaussian", offset="offset")
    affine = _assert_affine_identity(model, data, "full")

    # The ordinary affine row offset is the supplied model offset itself; it is
    # not silently dropped into a bare X matrix.
    np.testing.assert_allclose(affine.offset, offset, rtol=0.0, atol=0.0)


def test_link_wiggle_affine_design_uses_joint_frame_and_exact_2141_index() -> None:
    # Converging link-wiggle fit, taken verbatim from the GREEN
    # ``bug_hunt_flexible_link_engages_and_predicts`` fixture that the module
    # docstring names as the model to mirror: a probit-truth binomial with a
    # ``link(type=flexible(logit))`` warp on a parametric mean. The warp engages
    # (the logit base is misspecified for probit data) and the joint Newton
    # solve certifies a stationary optimum, so the fit is minted and the exact
    # affine design exists.
    #
    # The prior committed fixture asked for ``linkwiggle(degree=2, ...)``, which
    # the parser rejects outright ("Spline degree 2 is too low for derivative
    # order 3; need degree >= 3"), so it could never reach a single assertion.
    # This fixture actually exercises the contract -- including the covariance
    # blocks, which is the whole point of #2299: before the covariance wiring
    # fix a converged link-wiggle fit returned ``covariance_conditional=None``
    # (the ``_assert_affine_identity`` helper's ``conditional is not None``
    # assertion is exactly the regression guard), so this test is red before the
    # fix and green after.
    #
    # The model-offset-separation half of the contract cannot ride on a
    # link-wiggle fit yet -- adding a model offset drives the joint solve
    # non-stationary (#2358) -- so it is exercised on the ORDINARY frame in
    # ``test_ordinary_affine_design_exposes_model_offset_and_full_frame`` and
    # kept as an honest link-wiggle red gate below, rather than silently dropped.
    #
    # WHY the DEFAULT flexible(logit) warp and not a leaner/heavier-penalty one:
    # do not "simplify" this to an explicit ``linkwiggle(internal_knots=2)`` or a
    # ``double_penalty=true`` warp to make the covariance "more PD" -- both were
    # measured to FAIL to converge (the binomial mean link-wiggle joint Newton
    # does not certify a stationary optimum), as does a stronger signal slope.
    # The convergent window of this family is narrow and mildly conditioned
    # (#2358); the default flexible(logit) warp is the fixture that both
    # converges AND exposes a finite covariance. Its joint covariance is reliably
    # non-None here (measured non-None across every clean sequential run); it can
    # only flip to a typed absence under pathological concurrent machine load,
    # where rayon-fold summation order at the PD tolerance decides the smallest
    # eigenvalue of ``H + S_lambda`` -- that is the #2358 solver-stability
    # marginality, not a design-matrix defect. The deterministic coverage of the
    # best-effort downgrade itself lives in the Rust unit test
    # ``best_effort_covariance_tests`` (gam-custom-family/src/covariance.rs),
    # which feeds a genuinely singular joint precision rather than relying on
    # this knife-edge.
    rng = np.random.default_rng(11)
    n = 2500
    x = rng.uniform(-2.5, 2.5, n)
    eta = -0.3 + 1.4 * x
    # TRUE link = probit; the requested flexible(logit) base is misspecified, so
    # the warp genuinely engages rather than sitting at zero.
    probability = np.clip(norm.cdf(eta), 1e-4, 1.0 - 1e-4)
    y = (rng.uniform(size=n) < probability).astype(float)
    data = {"y": y, "x": x}

    model = gamfit.fit(
        data,
        "y ~ x + link(type=flexible(logit))",
        family="binomial",
    )
    affine = _assert_affine_identity(model, data, "link_wiggle_joint")

    # The complete joint frame keeps the fitted Mean block IN the matrix (rather
    # than folding the fitted base into the row offset), so the returned
    # same-frame covariances carry mean variance and every Mean--wiggle cross
    # term for external variance calculations. There is no model offset here, so
    # the affine row offset is the zero vector; the offset-SEPARATION assertion
    # lives on the ordinary frame and the link-wiggle red gate below.
    assert affine.offset.shape == (n,)
    np.testing.assert_allclose(affine.offset, 0.0, rtol=0.0, atol=0.0)

    # The exact affine identity that ``_assert_affine_identity`` already checked
    # (offset + [X, B(index)] @ [beta_mean, beta_w] == linear_predictor) can only
    # hold if B is evaluated at the saved frozen #2141 index, so it pins the
    # frozen-index behavior end to end. The covariance blocks the helper
    # validated are the #2299 deliverable: a converged custom-family link-wiggle
    # fit now carries its joint [Mean, LinkWiggle] covariance.
    assert affine.covariance_conditional is not None
    assert affine.covariance_conditional.shape == (
        affine.coefficients.shape[0],
        affine.coefficients.shape[0],
    )


def test_link_wiggle_affine_design_offset_separation_red_gate() -> None:
    """RED GATE (#2358 link-wiggle + offset joint-Newton non-convergence).

    The #2299 offset-SEPARATION contract for the link-wiggle joint frame: a
    fitted link-wiggle predictor with a known per-row model offset must expose
    that offset as ``affine.offset`` (never folded into the design), so external
    variance/contrast math sees ``offset + [X, B] @ beta``. This is the
    link-wiggle analogue of the green ordinary-frame offset assertion in
    ``test_ordinary_affine_design_exposes_model_offset_and_full_frame``.

    It fails today for a convergence-lane reason ORTHOGONAL to the design-matrix
    contract: adding a model offset to the converging flexible-link fit above
    drives the binomial mean link-wiggle joint solve non-stationary (outer
    smoothing does not certify; |Pg| ~ 2.8e-2 vs bound ~ 6.3e-3), so no fit is
    minted and the affine design cannot be built. See #2358 (the offset is a
    manifestation; the same non-convergence reproduces with no offset for an
    explicit ``linkwiggle(...)`` spec). When that lane converges this gate passes
    unchanged -- the assertion is NOT weakened to match the broken path.
    """
    rng = np.random.default_rng(11)
    n = 2500
    x = rng.uniform(-2.5, 2.5, n)
    offset = rng.uniform(-0.15, 0.15, n)
    eta = -0.3 + 1.4 * x + offset
    probability = np.clip(norm.cdf(eta), 1e-4, 1.0 - 1e-4)
    y = (rng.uniform(size=n) < probability).astype(float)
    data = {"y": y, "x": x, "offset": offset}

    model = gamfit.fit(
        data,
        "y ~ x + link(type=flexible(logit))",
        family="binomial",
        offset="offset",
    )
    affine = _assert_affine_identity(model, data, "link_wiggle_joint")
    np.testing.assert_allclose(affine.offset, offset, rtol=0.0, atol=0.0)


def test_design_matrix_array_returns_the_same_typed_affine_contract() -> None:
    rng = np.random.default_rng(23)
    x = rng.normal(size=(96, 2))
    y = 0.4 + 0.8 * x[:, 0] - 0.3 * x[:, 1] + rng.normal(0.0, 0.05, 96)
    model = gamfit.fit_array(x, y, formula="y ~ x0 + x1", family="gaussian")

    affine = model.design_matrix_array(x)
    assert isinstance(affine, gamfit.AffineDesign)
    assert affine.coefficient_frame == "full"
    reconstructed = affine.offset + affine.matrix @ affine.coefficients
    expected = _linear_predictor(
        model,
        {"x0": x[:, 0], "x1": x[:, 1], "y": y},
    )
    np.testing.assert_allclose(reconstructed, expected, rtol=2e-12, atol=2e-12)


# ---------------------------------------------------------------------------
# Honest red gates: the SAME #2299 affine-design contract, exercised on the two
# convergence lanes that currently blow up before a fit exists. These assert the
# unweakened contract (no xfail/skip); they fail today because the fit does not
# converge, and they will flip green when their lanes are fixed. Do NOT weaken
# or delete -- they are the coverage the contract tests above deliberately move
# off of hostile fixtures.
# ---------------------------------------------------------------------------


def test_ordinary_affine_design_reml_offset_smoothing_boundary_red_gate() -> None:
    """RED GATE (REML-with-offset boundary non-stationarity).

    A model offset that is itself a smooth function of ``x`` is collinear with
    ``s(x)``; REML then drives the smoothing parameter to a boundary where the
    outer objective is non-stationary and the fit does not settle. This is a
    convergence-lane defect ORTHOGONAL to the #2299 design-matrix contract,
    which is exercised on a well-conditioned fixture in
    ``test_ordinary_affine_design_exposes_model_offset_and_full_frame``. When
    the REML-with-offset lane is stationary this gate passes unchanged.
    """
    rng = np.random.default_rng(2299)
    n = 160
    x = rng.uniform(-1.5, 1.5, n)
    offset = 0.35 * np.sin(1.7 * x) - 0.1
    y = 0.6 + 1.2 * x + offset + rng.normal(0.0, 0.08, n)
    data = {"y": y, "x": x, "offset": offset}

    model = gamfit.fit(data, "y ~ s(x)", family="gaussian", offset="offset")
    affine = _assert_affine_identity(model, data, "full")
    np.testing.assert_allclose(affine.offset, offset, rtol=0.0, atol=0.0)


def test_link_wiggle_affine_design_flex_link_joint_newton_blowup_red_gate() -> None:
    """RED GATE (#979 / #1596 flexible-link joint-Newton blow-up).

    Deterministic flexible-link repro inherited from #2141: a smooth mean
    ``s(x)`` aliased against a ``flexible_link=True`` warp collapses the joint
    Newton solve (min-eig ~ -1e200, degenerate fit, no conditional covariance),
    so the affine design cannot be built. This is a convergence-lane defect
    ORTHOGONAL to the #2299 contract, which is exercised on an identifiable
    parametric-mean flexible-link fit in
    ``test_link_wiggle_affine_design_uses_joint_frame_and_exact_2141_index``.
    On this geometry the de-alias shift is material: evaluating B at the base
    predictor instead of the saved frozen index produced a dramatically
    different fitted link. When the #979 / #1596 lane converges this gate passes
    unchanged.
    """
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(-2.0, 2.0, n)
    offset = 0.2 * np.sin(1.3 * x)
    eta = 0.5 + 1.5 * x + offset
    probability = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < probability).astype(float)
    data = {"y": y, "x": x, "offset": offset}

    model = gamfit.fit(
        data,
        "y ~ s(x)",
        family="binomial",
        offset="offset",
        flexible_link=True,
    )
    affine = _assert_affine_identity(model, data, "link_wiggle_joint")
    np.testing.assert_allclose(affine.offset, offset, rtol=0.0, atol=0.0)
