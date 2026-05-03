from __future__ import annotations
import typing

import pathlib

import pytest

pytest.importorskip("gam._rust")

import matplotlib
import numpy as np
import pandas as pd

import gam
from gam.pgs import PgsCalibration
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


def test_build_info_reports_real_extension() -> None:
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


def test_validate_formula_reports_model_metadata() -> None:
    validation = gam.validate_formula(training_rows(), "y ~ x")

    assert validation["formula"] == "y ~ x"
    assert validation["model_class"] == "standard"
    assert validation["family_name"] == "Gaussian Identity"
    assert validation["response_column"] == "y"
    assert validation.supported_by_python is True


def test_fit_predict_summary_check_report_and_roundtrip(tmp_path: pathlib.Path) -> None:
    model = gam.fit(training_rows(), "y ~ x")
    summary = model.summary()

    assert model.formula == "y ~ x"
    assert model.model_class == "standard"
    assert model.family_name == "Gaussian Identity"
    assert model.training_table_kind == "records"
    assert not model.is_survival
    assert not model.is_transformation_normal
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


def test_pandas_diagnostics_and_plotting() -> None:
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


def test_sklearn_regressor_roundtrip() -> None:
    train = training_frame()
    predict = prediction_frame()

    est = GAMRegressor(formula="y ~ x")
    est.fit(train)
    predictions = est.predict(predict)

    assert predictions.shape == (3,)
    assert est.n_features_in_ == 1
    assert est.feature_names_in_.tolist() == ["x"]
    assert est.score(train[["x"]], train["y"]) > 0.999

    fitted_training_response = pd.Series(est.predict(train[["x"]]))
    y_with_masked_outlier = fitted_training_response.copy()
    y_with_masked_outlier.iloc[-1] += 100.0
    weights = np.ones(len(y_with_masked_outlier), dtype=float)
    weights[-1] = 0.0
    assert (
        est.score(
            train[["x"]],
            y_with_masked_outlier,
            sample_weight=weights,
        )
        > 0.999
    )


def test_numpy_inputs_and_outputs() -> None:
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


def test_sklearn_regressor_accepts_rhs_only_formula_with_separate_target() -> None:
    x_train = pd.DataFrame([{"x0": 0.0}, {"x0": 1.0}, {"x0": 2.0}, {"x0": 3.0}])
    y_train = np.array([1.0, 2.0, 3.0, 4.0])

    est = GAMRegressor(formula="x0")
    est.fit(x_train, y_train)

    assert est.formula_ == "y ~ x0"
    assert est.feature_names_in_.tolist() == ["x0"]


def test_sklearn_classifier_roundtrip() -> None:
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

    fitted_training_labels = pd.Series(est.predict(train[["x"]]))
    y_with_masked_outlier = fitted_training_labels.copy()
    y_with_masked_outlier.iloc[-1] = 1 - int(y_with_masked_outlier.iloc[-1])
    weights = np.ones(len(y_with_masked_outlier), dtype=float)
    weights[-1] = 0.0
    assert (
        est.score(
            train[["x"]],
            y_with_masked_outlier,
            sample_weight=weights,
        )
        == 1.0
    )


def test_predict_rejects_schema_mismatch() -> None:
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


def test_predict_can_passthrough_id_column() -> None:
    model = gam.fit(training_rows(), "y ~ x")

    pred = model.predict(
        [
            {"person_id": "a", "x": 1.5},
            {"person_id": "b", "x": 2.5},
        ],
        id_column="person_id",
        return_type="dict",
    )

    assert pred["person_id"] == ["a", "b"]
    assert set(pred) == {"person_id", "eta", "mean"}


# ---------------------------------------------------------------------------
# Pipeline smoke tests (task #14 / task #17).
#
# These tests exercise the PGS-facing Python surface: Stage 1 conditional
# Gaussianization of the PGS on the PC manifold, Stage 2a Bernoulli
# marginal-slope on the calibrated score, and survival prediction surfaces.
# They go through the Python binding (``gam.fit`` / ``model.predict``), not
# the CLI — the CLI contract is covered separately by
# ``tests/integration_pit_pipeline.py``.
# ---------------------------------------------------------------------------


def _require_extension() -> None:
    if not gam.build_info().get("available"):
        pytest.skip("rust extension not built")


def _pc_duchon(centers: int = 6) -> str:
    return (
        f"duchon(pc1, pc2, pc3, pc4, centers={centers}, "
        "order=1, power=2, length_scale=1)"
    )


def test_transformation_normal_pgs_calibration_roundtrip(synthetic_biobank_factory: typing.Any) -> None:
    """Stage 1: fit h(PGS | PCs) ~ N(0, 1) and verify PIT properties.

    After conditional Gaussianization on the PC manifold the predicted
    z-scores should be approximately standard normal AND decorrelated
    from each PC — that's the defining property of the anchored
    deviation invariant used throughout the methods section.
    """
    _require_extension()
    df = synthetic_biobank_factory(seed=0, n=128)

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


def test_pgs_calibration_predicts_minimal_new_samples_after_fit_on_full_df(
    synthetic_biobank_factory: typing.Any,
) -> None:
    _require_extension()
    df = synthetic_biobank_factory(seed=10, n=128)
    df["person_id"] = [f"p{i}" for i in range(len(df))]
    df["batch"] = ["a" if i % 2 == 0 else "b" for i in range(len(df))]

    calibration = PgsCalibration(
        pc_columns=["pc1", "pc2", "pc3", "pc4"],
        pgs_column="PGS",
    ).fit(df)

    minimal = df[["PGS", "pc1", "pc2", "pc3", "pc4"]].copy()
    z = np.asarray(calibration.predict(minimal), dtype=float)

    assert z.shape == (len(minimal),)
    assert np.all(np.isfinite(z))


def test_transformation_normal_check_requires_raw_pgs(synthetic_biobank_factory: typing.Any) -> None:
    _require_extension()
    df = synthetic_biobank_factory(seed=11, n=128)

    model = gam.fit(
        df,
        f"PGS ~ {_pc_duchon()}",
        transformation_normal=True,
        scale_dimensions=True,
    )

    missing_pgs = df[["pc1", "pc2", "pc3", "pc4"]].copy()
    check = model.check(missing_pgs)

    assert not check.ok
    assert any(issue.column == "PGS" for issue in check.issues)
    with pytest.raises(gam.SchemaMismatchError):
        model.predict(missing_pgs)


def test_pgs_calibration_formula_uses_duchon_operator_penalties_without_double_penalty() -> None:
    calibration = PgsCalibration(pc_columns=["pc1", "pc2"], pgs_column="PGS")

    assert "double_penalty" not in calibration.formula
    assert "duchon(pc1, pc2" in calibration.formula


def test_pgs_calibration_save_load_restores_wrapper_metadata(
    synthetic_biobank_factory: typing.Any,
    tmp_path: typing.Any,
) -> None:
    _require_extension()
    df = synthetic_biobank_factory(seed=12, n=128)
    calibration = PgsCalibration(
        pc_columns=["pc1", "pc2", "pc3", "pc4"],
        pgs_column="PGS",
        out_column="pgs_z",
    ).fit(df)

    path = tmp_path / "stage1.gam"
    calibration.save(path)
    loaded = PgsCalibration.load(path)
    minimal = df[["PGS", "pc1", "pc2", "pc3", "pc4"]]

    assert loaded.pc_columns == ["pc1", "pc2", "pc3", "pc4"]
    assert loaded.pgs_column == "PGS"
    assert loaded.out_column == "pgs_z"
    np.testing.assert_allclose(
        np.asarray(loaded.predict(minimal), dtype=float),
        np.asarray(calibration.predict(minimal), dtype=float),
    )


def test_bernoulli_marginal_slope_roundtrip_tracks_calibrated_score(
    synthetic_biobank_factory: typing.Any, tmp_path: typing.Any
) -> None:
    """Stage 2a: Bernoulli marginal-slope on the calibrated score.

    Fit Stage 1 to produce ``pgs_ctn_z`` (the anchored deviation), then fit
    the disease model with a probit link and constant marginal slope.
    Roundtrip the saved model and check predictions are valid probabilities
    that track ``pgs_ctn_z`` monotonically.
    """
    _require_extension()
    df = synthetic_biobank_factory(seed=1, n=128)

    calib = gam.fit(
        df,
        f"PGS ~ {_pc_duchon()}",
        transformation_normal=True,
        scale_dimensions=True,
    )
    df["pgs_ctn_z"] = np.asarray(calib.predict(df), dtype=float)

    model = gam.fit(
        df,
        "disease ~ z",
        family="bernoulli-marginal-slope",
        link="probit",
        scale_dimensions=True,
        z_column="pgs_ctn_z",
        logslope_formula="1",
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

    # Compute Spearman via numpy rank correlation to avoid pulling scipy in
    # as a test dependency.
    pgs_rank = pd.Series(df["pgs_ctn_z"].to_numpy()).rank().to_numpy()
    prob_rank = pd.Series(probs).rank().to_numpy()
    rho = float(np.corrcoef(pgs_rank, prob_rank)[0, 1])
    assert rho > 0.3, f"spearman(pgs_ctn_z, p) = {rho:.3f} not monotone enough"


def test_survival_prediction_dense_surfaces_smoke() -> None:
    """Survival predictions expose finite, monotone interpolated surfaces."""
    times = np.array([45.0, 55.0, 65.0], dtype=float)
    cumulative = np.array(
        [
            [0.05, 0.15, 0.32],
            [0.02, 0.08, 0.18],
        ],
        dtype=float,
    )
    survival_surface = np.exp(-cumulative)
    hazard_surface = np.array(
        [
            [0.004, 0.010, 0.017],
            [0.002, 0.006, 0.010],
        ],
        dtype=float,
    )
    pred = gam.SurvivalPrediction(
        model_class="survival marginal-slope",
        parameters=np.zeros((2, 1), dtype=float),
        parameter_names=("eta",),
        times=times,
        hazard=hazard_surface,
        survival=survival_surface,
        cumulative_hazard=cumulative,
    )
    np.testing.assert_allclose(np.asarray(pred.cumulative_hazard, dtype=float), cumulative)
    grid = np.array([45.0, 55.0, 65.0], dtype=float)

    hazard = pred.hazard_at(grid)
    assert hazard.shape == (2, grid.shape[0])
    assert np.all(np.isfinite(hazard)), (
        f"survival hazard contains non-finite values; min={np.nanmin(hazard)}, "
        f"max={np.nanmax(hazard)}"
    )
    assert np.all(hazard > 0.0)

    survival = pred.survival_at(grid)
    assert survival.shape == (2, grid.shape[0])
    assert np.all(np.isfinite(survival))
    assert np.all((survival > 0.0) & (survival <= 1.0 + 1e-9)), (
        f"survival outside (0,1]: min={float(survival.min())}, "
        f"max={float(survival.max())}"
    )
    deltas = np.diff(survival, axis=1)
    assert np.all(deltas <= 1e-9), (
        "survival must be non-increasing in time; offending row indices: "
        f"{np.argwhere(deltas > 1e-9)[:10].tolist()}"
    )

    cumhaz = pred.cumulative_hazard_at(grid)
    assert cumhaz.shape == (2, grid.shape[0])
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

    query = np.array([50.0, 60.0], dtype=float)
    query_survival = np.asarray(pred.survival_at(query), dtype=float)
    max_abs_diff = float(np.max(np.abs(query_survival[1, :] - query_survival[0, :])))
    assert max_abs_diff > 1e-3, (
        "survival prediction rows should retain row-specific curves; "
        f"max abs diff was {max_abs_diff:.2e}"
    )
    assert np.all(query_survival[1, :] > query_survival[0, :])


def test_survival_prediction_write_csv_preserves_ids(tmp_path: pathlib.Path) -> None:
    pred = gam.SurvivalPrediction(
        model_class="survival",
        parameters=np.array([[np.log(0.10)], [np.log(0.20)]], dtype=float),
        parameter_names=("log_hazard",),
        id_column="person_id",
        row_ids=("p0", "p1"),
    )

    out = pred.write_survival_at_csv(
        tmp_path / "survival_ids.csv",
        np.array([1.0, 2.0], dtype=float),
        people_chunk=1,
        time_grid_chunk=1,
    )
    text = pathlib.Path(out).read_text(encoding="utf-8").splitlines()

    assert text[0] == "row,person_id,time,survival"
    assert [line.split(",")[:3] for line in text[1:]] == [
        ["0", "p0", "1.0"],
        ["0", "p0", "2.0"],
        ["1", "p1", "1.0"],
        ["1", "p1", "2.0"],
    ]
    values = [float(line.split(",")[3]) for line in text[1:]]
    assert values[1] < values[0]
    assert values[3] < values[2]
    assert values[2] < values[0]


def test_survival_prediction_large_curves_require_chunks(tmp_path: pathlib.Path) -> None:
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

    chunk_grid = np.array([1.0, 2.0], dtype=float)
    cumhaz_chunks = list(
        pred.cumulative_hazard_at_chunks(
            chunk_grid,
            people_chunk=500,
            time_grid_chunk=1,
        )
    )
    hazard_chunks = list(
        pred.hazard_at_chunks(
            chunk_grid,
            people_chunk=500,
            time_grid_chunk=1,
        )
    )
    assert cumhaz_chunks[0][2].shape == (500, 1)
    assert hazard_chunks[-1][2].shape == (1, 1)

    out = pred.write_survival_at_csv(
        tmp_path / "survival.csv",
        np.array([1.0, 2.0], dtype=float),
        people_chunk=500,
        time_grid_chunk=1,
    )
    text = pathlib.Path(out).read_text(encoding="utf-8").splitlines()
    assert text[0] == "row,time,survival"
    assert len(text) == 1 + 1_001 * 2
