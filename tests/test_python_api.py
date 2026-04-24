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
    assert info["supported_model_classes"] == ["standard"]


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

    with_interval = model.predict(prediction_rows(), interval=0.95)
    assert list(with_interval) == [
        "eta",
        "mean",
        "effective_se",
        "mean_lower",
        "mean_upper",
    ]

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

    assert pred.shape == (2,)

    model = gam.fit({"x0": x_train[:, 0].tolist(), "y": y_train.tolist()}, "y ~ x0")
    raw = model.predict(x_test, return_type="numpy")
    assert raw.shape == (2, 2)


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
    proba = est.predict_proba(pd.DataFrame([{"x": 1.5}, {"x": 3.5}]))
    pred = est.predict(pd.DataFrame([{"x": 1.5}, {"x": 3.5}]))

    assert proba.shape == (2, 2)
    assert pred.shape == (2,)
    assert est.classes_.tolist() == [0, 1]


def test_predict_rejects_schema_mismatch():
    model = gam.fit(training_rows(), "y ~ x")
    with pytest.raises(gam.SchemaMismatchError):
        model.predict([{"z": 1.0}])


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


def _pc_duchon(centers: int = 24) -> str:
    return (
        f"duchon(pc1, pc2, pc3, pc4, centers={centers}, "
        "order=1, power=1, double_penalty=true)"
    )


def test_transformation_normal_pgs_calibration_roundtrip(synthetic_biobank):
    """Stage 1: fit h(PGS | PCs) ~ N(0, 1) and verify PIT properties.

    After conditional Gaussianization on the PC manifold the predicted
    z-scores should be approximately standard normal AND decorrelated
    from each PC — that's the defining property of the anchored
    deviation invariant used throughout the methods section.
    """
    _require_extension()
    df = synthetic_biobank

    model = gam.fit(
        df,
        f"PGS ~ {_pc_duchon(centers=24)}",
        transformation_normal=True,
    )
    pred = model.predict(df, return_type="dict")
    z = np.asarray(pred["eta"], dtype=float)

    assert z.shape == (len(df),)
    assert np.all(np.isfinite(z))
    assert -0.3 < float(z.mean()) < 0.3
    assert 0.7 < float(z.std(ddof=0)) < 1.3
    for pc in ("pc1", "pc2", "pc3", "pc4"):
        corr = float(np.corrcoef(z, df[pc].to_numpy())[0, 1])
        assert abs(corr) < 0.3, f"|corr(z, {pc})| = {abs(corr):.3f} too large"


def test_bernoulli_marginal_slope_with_linkwiggle_and_score_warp(
    synthetic_biobank, tmp_path
):
    """Stage 2a: Bernoulli marginal-slope + linkwiggle + logslope score-warp.

    Fit Stage 1 to produce ``PGS_cal`` (the anchored deviation), then fit
    the disease model with a probit link, a main-formula link-wiggle, and
    a logslope-formula that folds the PC manifold + another linkwiggle
    into the score-warp. Roundtrip the saved model and check predictions
    are valid probabilities that track ``PGS_cal`` monotonically.
    """
    _require_extension()
    df = synthetic_biobank.copy()

    calib = gam.fit(
        df,
        f"PGS ~ {_pc_duchon(centers=24)}",
        transformation_normal=True,
    )
    df["PGS_cal"] = np.asarray(
        calib.predict(df, return_type="dict")["eta"], dtype=float
    )

    disease_formula = (
        f"disease ~ z + {_pc_duchon(centers=24)} "
        "+ linkwiggle(degree=3, internal_knots=8)"
    )
    logslope = (
        f"{_pc_duchon(centers=24)} + linkwiggle(degree=3, internal_knots=8)"
    )
    model = gam.fit(
        df,
        disease_formula,
        family="bernoulli-marginal-slope",
        link="probit",
        z_column="PGS_cal",
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

    # Monotone-ish in PGS_cal (loose threshold because Duchon + linkwiggle
    # introduce real flex in the marginal response). Compute Spearman via
    # numpy rank correlation to avoid pulling scipy in as a test dep.
    pgs_rank = pd.Series(df["PGS_cal"].to_numpy()).rank().to_numpy()
    prob_rank = pd.Series(probs).rank().to_numpy()
    rho = float(np.corrcoef(pgs_rank, prob_rank)[0, 1])
    assert rho > 0.3, f"spearman(PGS_cal, p) = {rho:.3f} not monotone enough"


def test_survival_marginal_slope_gompertz_makeham_timewiggle_smoke(
    synthetic_biobank,
):
    """Stage 2b: survival marginal-slope with GM baseline + timewiggle.

    Fit left-truncated survival with a Gompertz-Makeham baseline, a
    PC-manifold Duchon, a linkwiggle, and a timewiggle. Check that the
    prediction object exposes hazard / survival queries at arbitrary
    time grids and that the returned surfaces are finite, positive, and
    monotone in time (survival decreasing, hazard finite).
    """
    _require_extension()
    df = synthetic_biobank.copy()

    calib = gam.fit(
        df,
        f"PGS ~ {_pc_duchon(centers=24)}",
        transformation_normal=True,
    )
    df["PGS_cal"] = np.asarray(
        calib.predict(df, return_type="dict")["eta"], dtype=float
    )

    formula = (
        "Surv(age_entry, age_exit, event) ~ z "
        f"+ {_pc_duchon(centers=24)} "
        "+ linkwiggle(degree=3, internal_knots=8) "
        "+ timewiggle(degree=3, internal_knots=6)"
    )
    model = gam.fit(
        df,
        formula,
        family="survival",
        survival_likelihood="marginal-slope",
        baseline_target="gompertz-makeham",
        z_column="PGS_cal",
    )

    pred = model.predict(df)
    grid = np.array([45.0, 55.0, 65.0], dtype=float)

    hazard = pred.hazard_at(grid)
    assert hazard.shape == (len(df), grid.shape[0])
    assert np.all(np.isfinite(hazard))
    assert np.all(hazard > 0.0)

    survival = pred.survival_at(grid)
    assert survival.shape == (len(df), grid.shape[0])
    assert np.all(np.isfinite(survival))
    assert np.all((survival > 0.0) & (survival <= 1.0 + 1e-9))
    # Survival is non-increasing in time for every sample.
    deltas = np.diff(survival, axis=1)
    assert np.all(deltas <= 1e-9)


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
