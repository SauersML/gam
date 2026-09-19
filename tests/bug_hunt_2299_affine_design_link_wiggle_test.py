"""Regression contract for #2299: fitted predictors have one typed affine API.

The old public design-matrix endpoint returned a bare matrix for ordinary GAMs
and rejected link-wiggle fits. That shape encoded neither a model offset nor
the coefficient or covariance coordinate system. A link-wiggle predictor is
affine at its fitted state as ``offset + X @ beta_mean + B(index) @ beta_w``;
critically, #2141 requires ``B`` to be evaluated at the saved frozen index, not
at the de-aliased base. The joint frame is necessary for mean uncertainty and
mean--wiggle cross-covariance to remain available to external contrasts.

Value and derivative are two operators, not one. ``matrix`` reproduces the
fitted ``eta``; ``eta_gradient`` is ``d eta / d beta`` and is the only operator
that may be paired with the shipped covariances. They are the same array for an
ordinary GAM (``eta`` is linear in ``beta``) and differ for a link-wiggle fit,
whose warp index moves with the mean coefficients, so ``d eta / d beta_mean`` is
``diag(dq/dq0) @ X`` rather than ``X``.

The DESIGN-MATRIX CONTRACT (frame name, offset/matrix/coefficient shapes, the
``offset + matrix @ coefficients == linear_predictor`` identity, the frozen
#2141 index, the value/derivative split, and all three covariance definitions)
is a property of the exact affine representation, independent of which
convergence lane produced the fit.
The GREEN contract tests -- ``test_ordinary_affine_design_exposes_model_offset``
and ``test_design_matrix_array_returns_the_same_typed_affine_contract`` -- run
on ordinary GAM fixtures that converge cleanly and are deterministic (their
covariance is always finite), so they assert the full contract including all
covariance definitions.

The LINK-WIGGLE frame's contract has a narrow convergent window: an explicit
``linkwiggle`` warp, a ``double_penalty`` warp, larger ``n``, and a stronger
slope were each measured to fail to converge, and only the mild default
``flexible(logit)`` warp converged, with a joint precision close to the PD
tolerance (#2358). The deterministic covariance-function coverage (exact
``M^-1`` and singular-posterior fit refusal) lives in Rust
``required_covariance_tests`` (gam-custom-family/src/covariance.rs), which feeds
a controlled joint precision with no marginal fit involved.

The link-wiggle and hostile-geometry tests in this file assert the same
unweakened contract:

  * ``test_link_wiggle_affine_design_covariance_and_identity`` -- the #2299
    joint-frame covariance + affine identity on the default flexible(logit) warp.
  * ``test_link_wiggle_affine_design_offset_separation`` -- the link-wiggle
    model-offset separation (#2358 was its convergence lane).
  * ``test_ordinary_affine_design_reml_offset_smoothing_boundary`` -- ``s(x)`` +
    a smooth-of-x model offset (gaussian), a REML-with-offset smoothing-boundary
    geometry.
  * ``test_link_wiggle_affine_design_flex_link_joint_newton_blowup`` -- the
    ``s(x)`` + ``flexible_link=True`` binomial geometry that once blew up the
    joint Newton solve (#979 / #1596).

None of them is expected to be red: a failure in any of them is a defect.
"""

import numpy as np
from scipy.stats import norm

import gamfit


def _linear_predictor(model, data) -> np.ndarray:
    prediction = model.predict(data, return_type="dict")
    return np.asarray(prediction["linear_predictor_plugin"], dtype=float)


def _assert_affine_identity(model, data, expected_frame: str) -> gamfit.results.AffineDesign:
    affine = model.design_matrix(data)
    assert isinstance(affine, gamfit.results.AffineDesign)
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

    # The covariances ship in the coefficient frame, so external variance math
    # must use the DERIVATIVE operator, not the value operator. They are the
    # same array only when the fitted predictor is linear in its coefficients.
    assert affine.eta_gradient.shape == affine.matrix.shape
    if expected_frame == "full":
        assert affine.eta_gradient is affine.matrix
    eta_variance = np.einsum(
        "ij,jk,ik->i",
        affine.eta_gradient,
        conditional,
        affine.eta_gradient,
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
    # non-stationarity is preserved in the test below.)
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


def test_link_wiggle_affine_design_covariance_and_identity() -> None:
    """#2358 fit-marginality + the #2299 predict-mu path.

    The #2299 joint-frame covariance + affine-identity contract for a converged
    link-wiggle fit, asserted UNWEAKENED: frame == ``link_wiggle_joint``, the
    joint ``[Mean, LinkWiggle]`` covariance is non-None, and
    ``offset + matrix @ coefficients == linear_predictor`` to the fp floor.

    Two things upstream of the design-matrix contract decide this test:
      * the fit was #2358-marginal: the mild logit=~probit warp leaves a
        weakly-identified warp direction, so the smallest eigenvalue of
        ``H + S_lambda`` sat near the PD tolerance and rayon-fold summation order
        (load-dependent) decided finite covariance versus a typed fit refusal. A
        PD verdict that depends on iteration order is a defect, so a red result
        under load is a real failure, not noise.
      * ``_assert_affine_identity`` obtains the engine linear predictor via
        ``model.predict``, which for a curved flexible link routes through the
        posterior-mean path and REQUIRES the joint covariance to integrate
        ``E[g^-1(eta)]``. A posterior-incomplete fit is refused before a model
        can reach this assertion.

    The covariance FUNCTION is pinned deterministically, with exact ``M^-1``
    values, in the Rust ``required_covariance_tests``
    (gam-custom-family/src/covariance.rs); this test is the end-to-end wiring
    half.

    Do NOT "simplify" the fixture to a leaner or heavier-penalty warp to make the
    covariance "more PD": an explicit ``linkwiggle(internal_knots=2)``, a
    ``double_penalty=true`` warp, a larger ``n``, and a stronger signal slope
    were each measured to FAIL to converge (the binomial mean link-wiggle joint
    Newton does not certify a stationary optimum). The default flexible(logit)
    warp is the only fixture in this family's narrow convergent window (#2358),
    and it is the model the module docstring names as the one to mirror.
    """
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
    # lives on the ordinary frame and the link-wiggle test below.
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

    # The warp index moves with the mean coefficients, so the value operator is
    # NOT d(eta)/d(beta): its mean block is missing the warp slope dq/dq0.
    # Exporting one matrix for both jobs would hand external variance math an
    # operator that silently disagrees with predict's own standard errors.
    assert affine.eta_gradient is not affine.matrix
    identical_columns = np.all(affine.eta_gradient == affine.matrix, axis=0)
    # The wiggle block B(index) is shared verbatim by both operators...
    assert bool(identical_columns.any())
    # ...and the mean block is not: it carries the warp slope.
    assert not bool(identical_columns.all())


def test_link_wiggle_affine_design_offset_separation() -> None:
    """Gate (#2358 link-wiggle + offset joint-Newton convergence).

    The #2299 offset-SEPARATION contract for the link-wiggle joint frame: a
    fitted link-wiggle predictor with a known per-row model offset must expose
    that offset as ``affine.offset`` (never folded into the design), so external
    variance/contrast math sees ``offset + [X, B] @ beta``. This is the
    link-wiggle analogue of the green ordinary-frame offset assertion in
    ``test_ordinary_affine_design_exposes_model_offset_and_full_frame``.

    It failed for a convergence-lane reason ORTHOGONAL to the design-matrix
    contract: adding a model offset to the converging flexible-link fit above
    drove the binomial mean link-wiggle joint solve non-stationary (outer
    smoothing did not certify; |Pg| ~ 2.8e-2 vs bound ~ 6.3e-3), so no fit was
    minted and the affine design could not be built (#2358). The assertion was
    never weakened to match the broken path.
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
    assert isinstance(affine, gamfit.results.AffineDesign)
    assert affine.coefficient_frame == "full"
    reconstructed = affine.offset + affine.matrix @ affine.coefficients
    expected = _linear_predictor(
        model,
        {"x0": x[:, 0], "x1": x[:, 1], "y": y},
    )
    np.testing.assert_allclose(reconstructed, expected, rtol=2e-12, atol=2e-12)


# ---------------------------------------------------------------------------
# The SAME #2299 affine-design contract, exercised on the two hostile geometries
# whose convergence lanes once blew up before a fit existed. They assert the
# unweakened contract. Do NOT weaken or delete them: they are the coverage the
# contract tests above deliberately move off of hostile fixtures.
# ---------------------------------------------------------------------------


def test_ordinary_affine_design_reml_offset_smoothing_boundary() -> None:
    """REML-with-offset smoothing-boundary geometry.

    A model offset that is itself a smooth function of ``x`` is collinear with
    ``s(x)``; REML drives the smoothing parameter toward a boundary where the
    outer objective was once non-stationary and the fit did not settle. That
    convergence lane is ORTHOGONAL to the #2299 design-matrix contract, which
    is exercised on a well-conditioned fixture in
    ``test_ordinary_affine_design_exposes_model_offset_and_full_frame``.
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


def test_link_wiggle_affine_design_flex_link_joint_newton_blowup() -> None:
    """#979 / #1596 flexible-link joint-Newton geometry.

    Deterministic flexible-link repro inherited from #2141: a smooth mean
    ``s(x)`` aliased against a ``flexible_link=True`` warp once collapsed the
    joint Newton solve (min-eig ~ -1e200, degenerate fit, no conditional
    covariance), so the affine design could not be built. That convergence lane
    is ORTHOGONAL to the #2299 contract, which is exercised on an identifiable
    parametric-mean flexible-link fit in
    ``test_link_wiggle_affine_design_covariance_and_identity``.
    On this geometry the de-alias shift is material: evaluating B at the base
    predictor instead of the saved frozen index produced a dramatically
    different fitted link.
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
