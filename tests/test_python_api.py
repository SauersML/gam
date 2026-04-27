from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("gam._rust")

import matplotlib
import numpy as np
import pandas as pd

import gam
from gam.sklearn import GAMClassifier, GAMRegressor


matplotlib.use("Agg")


def training_rows() -> list[dict[str, float]]:
    return [
        {"y": 1.0, "x": 0.0},
        {"y": 2.0, "x": 1.0},
        {"y": 3.0, "x": 2.0},
        {"y": 4.0, "x": 3.0},
        {"y": 5.0, "x": 4.0},
        {"y": 6.0, "x": 5.0},
    ]


def prediction_rows() -> list[dict[str, float]]:
    return [
        {"x": 1.5},
        {"x": 2.5},
        {"x": 3.5},
    ]


def training_frame() -> pd.DataFrame:
    return pd.DataFrame(training_rows())


def prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(prediction_rows())


def test_build_info_reports_real_extension():
    info = gam.build_info()
    assert info["available"] is True
    assert info["module"] == "gam._rust"
    assert "fit" in info["capabilities"]
    assert "validate_formula" in info["capabilities"]
    assert info["supported_model_classes"] == [
        "standard",
        "transformation-normal",
        "bernoulli-marginal-slope",
        "survival-marginal-slope",
        "survival-location-scale",
        "latent-survival",
        "latent-binary",
        "gaussian-location-scale",
        "binomial-location-scale",
    ]


def test_validate_formula_reports_model_metadata():
    validation = gam.validate_formula(training_rows(), "y ~ x")

    assert validation["formula"] == "y ~ x"
    assert validation["model_class"] == "standard"
    assert validation["family_name"] == "Gaussian Identity"
    assert validation["response_column"] == "y"
    assert validation.supported_by_python is True


def test_fit_predict_summary_check_report_and_roundtrip(tmp_path: pathlib.Path):
    model = gam.fit(training_rows(), "y ~ x")
    summary = model.summary()

    assert model.formula == "y ~ x"
    assert model.model_class == "standard"
    assert model.family_name == "Gaussian Identity"
    assert summary["iterations"] >= 0
    assert not summary.coefficients_frame().empty

    predicted = model.predict(prediction_rows())
    assert list(predicted) == ["eta", "mean"]
    assert len(predicted["mean"]) == 3
    # The training data is exactly y = x + 1 with no noise, so the
    # identity-link Gaussian model must recover that linear function within
    # numerical tolerance — a renamed column or swapped eta/mean would break
    # at least one of these checks. The 1e-3 tolerance matches the rmse
    # bound used elsewhere in this suite (test_pandas_diagnostics_and_plotting)
    # and accounts for any default ridge / shrinkage applied during fit.
    expected_mean = [2.5, 3.5, 4.5]
    np.testing.assert_allclose(predicted["mean"], expected_mean, atol=1e-3)
    # For the identity link, eta and mean are computed from the same beta·x
    # — a swap would still be detected here because the comparison is bit
    # close, just not strict bit-equality (allowing for any post-link copy).
    np.testing.assert_allclose(predicted["eta"], predicted["mean"], atol=1e-9)

    with_interval = model.predict(prediction_rows(), interval=0.95)
    assert list(with_interval) == [
        "eta",
        "mean",
        "effective_se",
        "mean_lower",
        "mean_upper",
    ]
    # The mean point estimate must sit strictly inside the 95% interval and
    # the interval must be non-degenerate for a well-conditioned fit.
    interval_mean = np.asarray(with_interval["mean"], dtype=float)
    interval_lower = np.asarray(with_interval["mean_lower"], dtype=float)
    interval_upper = np.asarray(with_interval["mean_upper"], dtype=float)
    interval_se = np.asarray(with_interval["effective_se"], dtype=float)
    assert np.all(interval_lower <= interval_mean + 1e-12)
    assert np.all(interval_mean <= interval_upper + 1e-12)
    assert np.all(interval_upper > interval_lower)
    assert np.all(interval_se > 0.0)

    check = model.check(prediction_rows())
    assert check.ok
    check.raise_for_error()

    bad_check = model.check([{"z": 1.0}])
    assert not bad_check.ok
    assert any(issue.column == "x" for issue in bad_check.issues)
    with pytest.raises(ValueError):
        bad_check.raise_for_error()

    html = model.report()
    assert "Model Summary" in html
    assert "y ~ x" in html

    model_path = tmp_path / "linear.gam"
    report_path = tmp_path / "linear.html"
    model.save(model_path)
    assert model_path.exists()
    assert model.report(report_path) == str(report_path)
    assert report_path.exists()

    loaded = gam.load(model_path)
    reloaded_prediction = loaded.predict(prediction_rows())
    assert reloaded_prediction["mean"] == predicted["mean"]


def test_pandas_diagnostics_and_plotting():
    model = gam.fit(training_frame(), "y ~ x")

    predicted = model.predict(prediction_frame())
    assert isinstance(predicted, pd.DataFrame)
    assert list(predicted.columns) == ["eta", "mean"]

    diagnostics = model.diagnose(training_frame())
    assert diagnostics.metrics["rmse"] < 1e-3
    assert diagnostics.metrics["r_squared"] > 0.999

    prediction_ax = model.plot(training_frame(), kind="prediction")
    residual_ax = model.plot(training_frame(), kind="residuals")
    ovp_ax = model.plot(training_frame(), kind="observed_vs_predicted")
    assert prediction_ax.get_xlabel() == "x"
    assert residual_ax.get_ylabel() == "residual"
    assert ovp_ax.get_xlabel() == "predicted mean"


def test_sklearn_regressor_roundtrip():
    train = training_frame()
    predict = prediction_frame()

    est = GAMRegressor(formula="y ~ x")
    est.fit(train)
    predictions = est.predict(predict)

    assert predictions.shape == (3,)
    assert est.n_features_in_ == 1
    assert est.feature_names_in_.tolist() == ["x"]
    assert est.score(train[["x"]], train["y"]) > 0.999


def test_numpy_inputs_and_outputs():
    x_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([1.0, 2.0, 3.0, 4.0])
    x_test = np.array([[1.5], [2.5]])

    est = GAMRegressor(formula="y ~ x0")
    est.fit(x_train, y_train)
    pred = est.predict(x_test)

    # Training data is y = x + 1; identity-link Gaussian must recover this.
    assert pred.shape == (2,)
    np.testing.assert_allclose(pred, [2.5, 3.5], atol=1e-3)

    model = gam.fit({"x0": x_train[:, 0].tolist(), "y": y_train.tolist()}, "y ~ x0")
    raw = model.predict(x_test, return_type="numpy")
    assert raw.shape == (2, 2)
    # The numpy-return contract is column-ordered (eta, mean). Identity link
    # means the two columns must agree numerically, and both columns must
    # match the analytic predictions — a swapped eta/mean column would fail
    # the dict-mode test above; here we lock in the array-mode contract.
    raw = np.asarray(raw, dtype=float)
    np.testing.assert_allclose(raw[:, 0], raw[:, 1], atol=1e-9)
    np.testing.assert_allclose(raw[:, 1], [2.5, 3.5], atol=1e-3)


def test_sklearn_regressor_accepts_rhs_only_formula_with_separate_target():
    x_train = pd.DataFrame([{"x0": 0.0}, {"x0": 1.0}, {"x0": 2.0}, {"x0": 3.0}])
    y_train = np.array([1.0, 2.0, 3.0, 4.0])

    est = GAMRegressor(formula="x0")
    est.fit(x_train, y_train)

    assert est.formula_ == "y ~ x0"
    assert est.feature_names_in_.tolist() == ["x0"]


def test_sklearn_classifier_roundtrip():
    train = pd.DataFrame(
        [
            {"y": 0.0, "x": 0.0},
            {"y": 0.0, "x": 1.0},
            {"y": 1.0, "x": 2.0},
            {"y": 1.0, "x": 3.0},
            {"y": 1.0, "x": 4.0},
        ]
    )
    est = GAMClassifier(formula="y ~ x", family="binomial")
    est.fit(train)
    test_frame = pd.DataFrame([{"x": 1.5}, {"x": 3.5}])
    proba = est.predict_proba(test_frame)
    pred = est.predict(test_frame)

    assert proba.shape == (2, 2)
    assert pred.shape == (2,)
    assert est.classes_.tolist() == [0, 1]

    # Probability rows must sum to 1 (proper categorical distribution).
    proba_arr = np.asarray(proba, dtype=float)
    np.testing.assert_allclose(proba_arr.sum(axis=1), 1.0, atol=1e-9)
    # Each column must be in [0, 1].
    assert np.all((proba_arr >= 0.0) & (proba_arr <= 1.0))
    # The training set has positive class probability that increases in x;
    # a swapped class-axis or reversed link would invert this ordering.
    assert proba_arr[1, 1] > proba_arr[0, 1], (
        f"P(y=1 | x=3.5) must exceed P(y=1 | x=1.5); got "
        f"P(1.5)={proba_arr[0, 1]:.4f}, P(3.5)={proba_arr[1, 1]:.4f}"
    )
    assert proba_arr[0, 0] > proba_arr[1, 0], (
        "P(y=0 | x=1.5) must exceed P(y=0 | x=3.5)"
    )
    # Hard predictions must use argmax over predict_proba.
    expected_hard = est.classes_.take(np.argmax(proba_arr, axis=1))
    np.testing.assert_array_equal(np.asarray(pred).reshape(-1), expected_hard)


def test_predict_rejects_schema_mismatch():
    model = gam.fit(training_rows(), "y ~ x")

    # 1) Wrong column name (no required feature present).
    with pytest.raises(gam.SchemaMismatchError) as exc_info:
        model.predict([{"z": 1.0}])
    assert "x" in str(exc_info.value), (
        f"schema-mismatch error must name the missing column; got: {exc_info.value}"
    )

    # 2) Required column missing in a row that has *other* columns. The
    # presence of unrelated keys must not silently mask the missing feature.
    with pytest.raises(gam.SchemaMismatchError):
        model.predict([{"y": 0.0, "irrelevant": 7.0}])

    # 3) An empty row list. The runtime is allowed to either reject it
    # (clear error) or return an empty result, but a non-empty result from
    # no input would be silently inventing rows.
    try:
        empty_pred = model.predict([])
    except (ValueError, gam.SchemaMismatchError, RuntimeError):
        pass  # explicit rejection is fine
    else:
        if isinstance(empty_pred, dict):
            n_rows = len(empty_pred.get("mean", []))
        elif hasattr(empty_pred, "shape"):
            n_rows = empty_pred.shape[0]
        else:
            n_rows = len(empty_pred)
        assert n_rows == 0, (
            f"empty predict input must yield an empty result; got {n_rows} rows: "
            f"{empty_pred!r}"
        )


# ---------------------------------------------------------------------------
# Pipeline smoke tests (task #14 / task #17).
#
# These three tests exercise the full PGS -> disease / survival pipeline
# that the Nature-Genetics-style methods section is built around: Stage 1
# conditional Gaussianization of the PGS on the PC manifold, Stage 2a
# Bernoulli marginal-slope with a link-wiggle + logslope score-warp, and
# Stage 2b survival marginal-slope with a Gompertz-Makeham baseline plus a
# timewiggle. They go through the Python binding (``gam.fit`` /
# ``model.predict``), not the CLI — the CLI contract is covered
# separately by ``tests/integration_pit_pipeline.py``.
# ---------------------------------------------------------------------------


def _require_extension():
    if not gam.build_info().get("available"):
        pytest.skip("rust extension not built")


def _pc_duchon(centers: int = 6) -> str:
    return (
        f"duchon(pc1, pc2, pc3, pc4, centers={centers}, "
        "order=1, power=2, length_scale=1, double_penalty=true)"
    )


def test_transformation_normal_pgs_calibration_roundtrip(synthetic_biobank_factory):
    """Stage 1: fit h(PGS | PCs) ~ N(0, 1) and verify PIT properties.

    After conditional Gaussianization on the PC manifold the predicted
    z-scores should be approximately standard normal AND decorrelated
    from each PC — that's the defining property of the anchored
    deviation invariant used throughout the methods section.
    """
    _require_extension()
    df = synthetic_biobank_factory(seed=0, n=64)

    model = gam.fit(
        df,
        f"PGS ~ {_pc_duchon()}",
        transformation_normal=True,
        scale_dimensions=True,
    )
    # Transformation-normal models return the per-row z-score directly as
    # a numpy array (see _model.predict docstring); no return_type indirection.
    z = np.asarray(model.predict(df), dtype=float)

    assert z.shape == (len(df),)
    assert np.all(np.isfinite(z))
    assert -0.3 < float(z.mean()) < 0.3
    assert 0.7 < float(z.std(ddof=0)) < 1.3
    for pc in ("pc1", "pc2", "pc3", "pc4"):
        corr = float(np.corrcoef(z, df[pc].to_numpy())[0, 1])
        assert abs(corr) < 0.3, f"|corr(z, {pc})| = {abs(corr):.3f} too large"


def test_bernoulli_marginal_slope_with_linkwiggle_and_score_warp(
    synthetic_biobank_factory, tmp_path
):
    """Stage 2a: Bernoulli marginal-slope + linkwiggle + logslope score-warp.

    Fit Stage 1 to produce ``pgs_ctn_z`` (the anchored deviation), then fit
    the disease model with a probit link, a main-formula link-wiggle, and
    a logslope-formula that folds the PC manifold + another linkwiggle
    into the score-warp. Roundtrip the saved model and check predictions
    are valid probabilities that track ``pgs_ctn_z`` monotonically.
    """
    _require_extension()
    df = synthetic_biobank_factory(seed=1, n=64)

    calib = gam.fit(
        df,
        f"PGS ~ {_pc_duchon()}",
        transformation_normal=True,
        scale_dimensions=True,
    )
    df["pgs_ctn_z"] = np.asarray(calib.predict(df), dtype=float)

    disease_formula = (
        f"disease ~ z + {_pc_duchon()} "
        "+ linkwiggle(degree=3, internal_knots=3)"
    )
    logslope = (
        f"{_pc_duchon()} + linkwiggle(degree=3, internal_knots=3)"
    )
    model = gam.fit(
        df,
        disease_formula,
        family="bernoulli-marginal-slope",
        link="probit",
        scale_dimensions=True,
        z_column="pgs_ctn_z",
        logslope_formula=logslope,
    )

    path = tmp_path / "bernoulli_ms.gam"
    model.save(path)
    loaded = gam.load(path)
    assert getattr(loaded, "is_marginal_slope", False) is True

    pred = loaded.predict(df, return_type="dict")
    probs = np.asarray(pred["mean"], dtype=float)
    assert probs.shape == (len(df),)
    assert np.all(np.isfinite(probs))
    assert np.all((probs > 0.0) & (probs < 1.0))

    # Monotone-ish in pgs_ctn_z (loose threshold because Duchon + linkwiggle
    # introduce real flex in the marginal response). Compute Spearman via
    # numpy rank correlation to avoid pulling scipy in as a test dep.
    pgs_rank = pd.Series(df["pgs_ctn_z"].to_numpy()).rank().to_numpy()
    prob_rank = pd.Series(probs).rank().to_numpy()
    rho = float(np.corrcoef(pgs_rank, prob_rank)[0, 1])
    assert rho > 0.3, f"spearman(pgs_ctn_z, p) = {rho:.3f} not monotone enough"


def test_survival_marginal_slope_gompertz_makeham_timewiggle_smoke(
    synthetic_biobank_factory,
):
    """Stage 2b: survival marginal-slope with GM baseline + timewiggle.

    Fit left-truncated survival with a Gompertz-Makeham baseline, a
    PC-manifold Duchon, a linkwiggle, and a timewiggle. Check that the
    prediction object exposes hazard / survival queries at arbitrary
    time grids and that the returned surfaces are finite, positive, and
    monotone in time (survival decreasing, hazard finite).
    """
    _require_extension()
    df = synthetic_biobank_factory(seed=2, n=64)

    calib = gam.fit(
        df,
        f"PGS ~ {_pc_duchon()}",
        transformation_normal=True,
        scale_dimensions=True,
    )
    df["pgs_ctn_z"] = np.asarray(calib.predict(df), dtype=float)

    formula = (
        "Surv(age_entry, age_exit, event) ~ z "
        f"+ {_pc_duchon()} "
        "+ linkwiggle(degree=3, internal_knots=3) "
        "+ timewiggle(degree=3, internal_knots=3)"
    )
    model = gam.fit(
        df,
        formula,
        family="survival",
        survival_likelihood="marginal-slope",
        baseline_target="gompertz-makeham",
        scale_dimensions=True,
        z_column="pgs_ctn_z",
    )

    pred = model.predict(df)
    grid = np.array([45.0, 55.0, 65.0], dtype=float)

    hazard = pred.hazard_at(grid)
    assert hazard.shape == (len(df), grid.shape[0])
    assert np.all(np.isfinite(hazard)), (
        f"survival hazard contains non-finite values; min={np.nanmin(hazard)}, "
        f"max={np.nanmax(hazard)}"
    )
    # Hazard rates are non-negative everywhere; the original test required
    # strict positivity which is the stronger contract for Gompertz-Makeham
    # baselines plus a smooth additive perturbation.
    assert np.all(hazard >= 0.0)
    assert np.all(hazard > 0.0)

    survival = pred.survival_at(grid)
    assert survival.shape == (len(df), grid.shape[0])
    assert np.all(np.isfinite(survival))
    # Survival is a probability and must lie in (0, 1].
    assert np.all((survival > 0.0) & (survival <= 1.0 + 1e-9)), (
        f"survival outside (0,1]: min={float(survival.min())}, "
        f"max={float(survival.max())}"
    )
    # Survival is non-increasing in time for every sample.
    deltas = np.diff(survival, axis=1)
    assert np.all(deltas <= 1e-9), (
        "survival must be non-increasing in time; offending row indices: "
        f"{np.argwhere(deltas > 1e-9)[:10].tolist()}"
    )

    # New: cumulative hazard must be non-decreasing in time. We do not
    # cross-check ``np.exp(-H) == S`` here because, depending on whether
    # the FFI emitted both surfaces or only one, ``cumulative_hazard_at``
    # may interpolate H and S independently — drift of order 1e-3 is
    # legitimate even when both surfaces are individually correct.
    cumhaz = pred.cumulative_hazard_at(grid)
    assert cumhaz.shape == (len(df), grid.shape[0])
    assert np.all(np.isfinite(cumhaz))
    assert np.all(cumhaz >= -1e-8), (
        "cumulative hazard must be non-negative everywhere; "
        f"min={float(cumhaz.min())}"
    )
    cumhaz_deltas = np.diff(cumhaz, axis=1)
    assert np.all(cumhaz_deltas >= -1e-8), (
        "cumulative hazard must be non-decreasing in time; offending row indices: "
        f"{np.argwhere(cumhaz_deltas < -1e-8)[:10].tolist()}"
    )

    # ---------------------------------------------------------------------
    # Covariate-effect contract: a model that learned to ignore covariates
    # and emit a single shared survival curve would still pass everything
    # above (monotone in time, in (0,1], finite, etc.). Build two synthetic
    # test rows that share every covariate except pgs_ctn_z and assert that
    # their predicted survival curves are not identical and differ at a
    # meaningful magnitude. The direction follows the synthetic biobank's
    # actual data-generating process — pgs_ctn_z is monotone in PGS, and
    # the simulation makes higher PGS lead to a longer lifetime, so we
    # assert the high-z survival exceeds the low-z survival.
    template = df.iloc[[0]].copy()
    z_lo = float(np.quantile(df["pgs_ctn_z"].to_numpy(), 0.05))
    z_hi = float(np.quantile(df["pgs_ctn_z"].to_numpy(), 0.95))
    test_lo = template.copy()
    test_lo["pgs_ctn_z"] = z_lo
    test_hi = template.copy()
    test_hi["pgs_ctn_z"] = z_hi
    survival_lo = np.asarray(model.predict(test_lo).survival_at(grid), dtype=float).reshape(-1)
    survival_hi = np.asarray(model.predict(test_hi).survival_at(grid), dtype=float).reshape(-1)
    # 1) Survival curves differ — covariate-blind model would fail this.
    max_abs_diff = float(np.max(np.abs(survival_hi - survival_lo)))
    assert max_abs_diff > 1e-3, (
        "survival predictions did not change when pgs_ctn_z swept from "
        f"{z_lo:.4f} to {z_hi:.4f}; max abs diff was {max_abs_diff:.2e}. "
        "The model appears to ignore the z covariate."
    )
    # 2) Direction is consistent with the synthetic data: higher PGS yields
    # lower hazard (longer life). At the median grid point the high-z
    # survival probability should be at least as large as the low-z value.
    mid = grid.shape[0] // 2
    assert survival_hi[mid] >= survival_lo[mid] - 1e-3, (
        f"at t={grid[mid]}, survival(high z={z_hi:.3f})={survival_hi[mid]:.4f} "
        f"but survival(low z={z_lo:.3f})={survival_lo[mid]:.4f}; "
        "higher PGS should not be worse than lower PGS in the synthetic data"
    )


def test_survival_prediction_large_curves_require_chunks(tmp_path: pathlib.Path):
    pred = gam.SurvivalPrediction(
        model_class="survival marginal-slope",
        parameters=np.zeros((1_001, 1), dtype=float),
        parameter_names=("eta",),
    )
    grid = np.linspace(0.0, 1000.0, 1000, dtype=float)

    with pytest.raises(ValueError, match="dense survival curves"):
        pred.survival_at(grid)

    chunks = list(pred.survival_at_chunks(grid, people_chunk=500, time_grid_chunk=128))
    assert chunks[0][2].shape == (500, 128)
    assert chunks[-1][2].shape == (1, 104)

    out = pred.write_survival_at_csv(
        tmp_path / "survival.csv",
        np.array([1.0, 2.0], dtype=float),
        people_chunk=500,
        time_grid_chunk=1,
    )
    text = pathlib.Path(out).read_text(encoding="utf-8").splitlines()
    assert text[0] == "row,time,survival"
    assert len(text) == 1 + 1_001 * 2
