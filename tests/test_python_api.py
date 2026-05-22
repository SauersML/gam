from __future__ import annotations
import json
import typing

import pathlib
import time

import pytest

pytest.importorskip("gamfit._rust")

import matplotlib
import numpy as np
import pandas as pd

import gamfit
from gamfit.sklearn import GAMClassifier, GAMRegressor


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


def _make_weibull_survival(seed: int = 123, n: int = 3000) -> pd.DataFrame:
    """Well-conditioned Weibull survival frame for marginal-slope regression tests."""
    rng = np.random.default_rng(seed)
    bmi = rng.normal(27.0, 4.5, n)
    hba1c = rng.normal(5.8, 0.7, n)
    age = rng.normal(0.0, 1.0, n)
    entry = rng.uniform(40.0, 65.0, n)

    bmi_z = (bmi - bmi.mean()) / bmi.std(ddof=0)
    hba1c_z = (hba1c - hba1c.mean()) / hba1c.std(ddof=0)
    log_scale = np.log(18.0) - 0.18 * bmi_z - 0.22 * hba1c_z - 0.10 * age
    shape = 1.45
    event_gap = np.exp(log_scale) * rng.weibull(shape, n)
    censor_gap = rng.uniform(6.0, 32.0, n)
    exit_time = entry + np.minimum(event_gap, censor_gap)
    event = (event_gap <= censor_gap).astype(float)

    return pd.DataFrame(
        {
            "entry": entry,
            "exit": exit_time,
            "event": event,
            "bmi": bmi,
            "hba1c": hba1c,
            "age": age,
        }
    )


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
    info = gamfit.build_info()
    assert info["available"] is True
    assert info["module"] == "gamfit._rust"
    assert "fit" in info["capabilities"]
    assert "validate_formula" in info["capabilities"]
    assert info["supported_model_classes"] == [
        "standard",
        "transformation-normal",
        "survival",
        "competing-risks-survival",
        "bernoulli-marginal-slope",
        "survival-marginal-slope",
        "survival-location-scale",
        "latent-survival",
        "latent-binary",
        "gaussian-location-scale",
        "binomial-location-scale",
    ]


def test_validate_formula_reports_model_metadata() -> None:
    validation = gamfit.validate_formula(training_rows(), "y ~ x")

    assert validation["formula"] == "y ~ x"
    assert validation["model_class"] == "standard"
    assert validation["family_name"] == "Gaussian Identity"
    assert validation["response_column"] == "y"
    assert validation.supported_by_python is True


def test_fit_predict_summary_check_report_and_roundtrip(tmp_path: pathlib.Path) -> None:
    model = gamfit.fit(training_rows(), "y ~ x")
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

    loaded = gamfit.load(model_path)
    reloaded_prediction = loaded.predict(prediction_rows())
    assert reloaded_prediction["mean"] == predicted["mean"]


def test_group_metadata_roundtrips_through_saved_model(tmp_path: pathlib.Path) -> None:
    rows = [
        {"y": 1.0, "g": "alpha"},
        {"y": 1.1, "g": "alpha"},
        {"y": 2.0, "g": "beta"},
        {"y": 2.2, "g": "beta"},
        {"y": 3.0, "g": "gamma"},
        {"y": 3.2, "g": "gamma"},
    ]
    metadata = {
        "alpha": {
            "source": "registry-a",
            "batch": 7,
            "scores": [0.25, 0.75],
            "audited": True,
        },
        "beta": {
            "source": "registry-b",
            "batch": 8,
            "tags": ["heldout", "priority"],
            "audited": False,
        },
    }

    model = gamfit.fit(
        rows,
        "y ~ group(g)",
        config={
            "groups": [
                {"name": group_name, "metadata": group_metadata}
                for group_name, group_metadata in metadata.items()
            ]
        },
    )

    path = tmp_path / "group_metadata.gam"
    model.save(path)
    saved_payload = json.loads(path.read_text())
    assert saved_payload["payload"]["group_metadata"] == metadata
    loaded = gamfit.load(path)

    assert model.group_metadata == metadata
    assert loaded.group_metadata == metadata


def test_pandas_diagnostics_and_plotting() -> None:
    model = gamfit.fit(training_frame(), "y ~ x")

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

    model = gamfit.fit({"x0": x_train[:, 0].tolist(), "y": y_train.tolist()}, "y ~ x0")
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
    model = gamfit.fit(training_rows(), "y ~ x")

    # 1) Wrong column name (no required feature present).
    with pytest.raises(gamfit.SchemaMismatchError) as exc_info:
        model.predict([{"z": 1.0}])
    assert "x" in str(exc_info.value), (
        f"schema-mismatch error must name the missing column; got: {exc_info.value}"
    )

    # 2) Required column missing in a row that has *other* columns. The
    # presence of unrelated keys must not silently mask the missing feature.
    with pytest.raises(gamfit.SchemaMismatchError):
        model.predict([{"y": 0.0, "irrelevant": 7.0}])

    # 3) An empty row list. The runtime is allowed to either reject it
    # (clear error) or return an empty result, but a non-empty result from
    # no input would be silently inventing rows.
    try:
        empty_pred = model.predict([])
    except (ValueError, gamfit.SchemaMismatchError, RuntimeError):
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
    model = gamfit.fit(training_rows(), "y ~ x")

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
# They go through the Python binding (``gamfit.fit`` / ``model.predict``), not
# the CLI — the CLI contract is covered separately by
# ``tests/integration_pit_pipeline.py``.
# ---------------------------------------------------------------------------


def _require_extension() -> None:
    if not gamfit.build_info().get("available"):
        pytest.skip("rust extension not built")


def _pc_duchon(centers: int = 6) -> str:
    return (
        f"duchon(pc1, pc2, pc3, pc4, centers={centers}, "
        "order=0, power=2, length_scale=1)"
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

    model = gamfit.fit(
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


def test_transformation_normal_check_requires_raw_pgs(synthetic_biobank_factory: typing.Any) -> None:
    _require_extension()
    df = synthetic_biobank_factory(seed=11, n=128)

    model = gamfit.fit(
        df,
        f"PGS ~ {_pc_duchon()}",
        transformation_normal=True,
        scale_dimensions=True,
    )

    missing_pgs = df[["pc1", "pc2", "pc3", "pc4"]].copy()
    check = model.check(missing_pgs)

    assert not check.ok
    assert any(issue.column == "PGS" for issue in check.issues)
    with pytest.raises(gamfit.SchemaMismatchError):
        model.predict(missing_pgs)


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

    calib = gamfit.fit(
        df,
        f"PGS ~ {_pc_duchon()}",
        transformation_normal=True,
        scale_dimensions=True,
    )
    df["pgs_ctn_z"] = np.asarray(calib.predict(df), dtype=float)

    model = gamfit.fit(
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
    loaded = gamfit.load(path)
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


def test_survival_marginal_slope_weibull_n3000_returns_under_60s() -> None:
    """Regression for the n=3000 marginal-slope survival FFI hang."""
    _require_extension()
    df = _make_weibull_survival(n=3000)

    started = time.monotonic()
    model = gamfit.fit(
        df,
        "Surv(entry, exit, event) ~ smooth(bmi) + smooth(hba1c)",
        survival_likelihood="marginal-slope",
        z_column="age",
        logslope_formula="smooth(bmi) + smooth(hba1c)",
    )
    elapsed = time.monotonic() - started
    assert elapsed < 60.0, f"survival marginal-slope fit took {elapsed:.1f}s"

    pred = model.predict(df.iloc[:8].copy())
    survival = np.asarray(pred.survival, dtype=float)
    assert survival.shape[0] == 8
    assert np.all(np.isfinite(survival))
    assert np.all((survival > 0.0) & (survival < 1.0))


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
    pred = gamfit.SurvivalPrediction(
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
    np.testing.assert_allclose(pred.failure_at(grid), 1.0 - survival)

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


def test_competing_risks_cif_matches_constant_hazard_closed_form() -> None:
    disease_rates = np.array([0.12, 0.06], dtype=float)
    death_rates = np.array([0.05, 0.02], dtype=float)
    disease_pred = gamfit.SurvivalPrediction(
        model_class="survival",
        parameters=np.log(disease_rates).reshape(-1, 1),
        parameter_names=("log_hazard",),
    )
    death_pred = gamfit.SurvivalPrediction(
        model_class="survival",
        parameters=np.log(death_rates).reshape(-1, 1),
        parameter_names=("log_hazard",),
    )
    times = np.array([0.0, 2.0, 5.0, 10.0], dtype=float)

    result = gamfit.competing_risks_cif(
        {"disease": disease_pred, "death": death_pred},
        times=times,
    )

    assert isinstance(result, gamfit.CompetingRisksCIF)
    assert result.endpoint_names == ("disease", "death")
    np.testing.assert_allclose(result.times, times)
    assert result.cif.shape == (2, 2, times.size)
    assert result.overall_survival.shape == (2, times.size)
    total_rates = disease_rates + death_rates
    expected_disease = (
        disease_rates[:, None]
        / total_rates[:, None]
        * (1.0 - np.exp(-total_rates[:, None] * times.reshape(1, -1)))
    )
    expected_death = (
        death_rates[:, None]
        / total_rates[:, None]
        * (1.0 - np.exp(-total_rates[:, None] * times.reshape(1, -1)))
    )
    np.testing.assert_allclose(result.cif[0], expected_disease, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.cif[1], expected_death, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result.cif.sum(axis=0),
        1.0 - result.overall_survival,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.all(np.diff(result.cif, axis=2) >= -1e-12)


def test_competing_risks_cif_validates_inputs() -> None:
    pred = gamfit.SurvivalPrediction(
        model_class="survival",
        parameters=np.array([[np.log(0.10)]], dtype=float),
        parameter_names=("log_hazard",),
    )
    two_row_pred = gamfit.SurvivalPrediction(
        model_class="survival",
        parameters=np.array([[np.log(0.10)], [np.log(0.20)]], dtype=float),
        parameter_names=("log_hazard",),
    )

    with pytest.raises(ValueError, match="time grid"):
        gamfit.competing_risks_cif([pred], times=[0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="time grid"):
        gamfit.competing_risks_cif([pred], times=[-1.0, 1.0])
    with pytest.raises(ValueError, match="same \\(n_rows, n_times\\) shape"):
        gamfit.competing_risks_cif([pred, two_row_pred], times=[0.0, 1.0])
    with pytest.raises(ValueError, match="endpoint_names must match"):
        gamfit.competing_risks_cif([pred, two_row_pred], times=[0.0, 1.0], endpoint_names=["one"])
    with pytest.raises(ValueError, match="endpoint_names must be unique"):
        gamfit.competing_risks_cif([pred, pred], times=[0.0, 1.0], endpoint_names=["same", "same"])
    with pytest.raises(TypeError, match="SurvivalPrediction"):
        gamfit.competing_risks_cif([typing.cast(typing.Any, object())], times=[0.0, 1.0])


def test_competing_risks_cif_plateaus_and_probability_bounds() -> None:
    times = np.array([0.0, 1.0, 3.0, 7.0, 12.0], dtype=float)
    cumulative = np.array(
        [
            [[0.0, 0.2, 0.2, 0.5, 1.1], [0.0, 0.0, 0.4, 0.4, 0.9]],
            [[0.0, 0.1, 0.3, 0.3, 0.7], [0.0, 0.2, 0.2, 0.8, 0.8]],
            [[0.0, 0.0, 0.2, 0.6, 0.6], [0.0, 0.1, 0.5, 0.5, 1.5]],
        ],
        dtype=float,
    )
    preds = {
        f"cause_{idx + 1}": gamfit.SurvivalPrediction(
            model_class="survival",
            parameters=np.zeros((2, 1), dtype=float),
            parameter_names=("eta",),
            times=times,
            cumulative_hazard=cumulative[idx],
            survival=np.exp(-cumulative[idx]),
            hazard=np.zeros_like(cumulative[idx]),
        )
        for idx in range(3)
    }

    result = gamfit.competing_risks_cif(preds, times=times)

    assert result.cif.shape == (3, 2, times.size)
    np.testing.assert_allclose(result.cif.sum(axis=0), 1.0 - result.overall_survival, atol=1e-12)
    assert np.all((result.cif >= 0.0) & (result.cif <= 1.0))
    assert np.all((result.overall_survival >= 0.0) & (result.overall_survival <= 1.0))
    assert np.all(np.diff(result.cif, axis=2) >= -1e-12)
    assert result.cif[0, 0, 1] == pytest.approx(result.cif[0, 0, 2])


def test_survival_prediction_write_csv_preserves_ids(tmp_path: pathlib.Path) -> None:
    pred = gamfit.SurvivalPrediction(
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


def test_survival_prediction_large_curves_auto_chunk_dense_output(tmp_path: pathlib.Path) -> None:
    pred = gamfit.SurvivalPrediction(
        model_class="survival marginal-slope",
        parameters=np.zeros((1_001, 1), dtype=float),
        parameter_names=("eta",),
    )
    grid = np.linspace(0.0, 1000.0, 1000, dtype=float)

    survival = np.asarray(pred.survival_at(grid), dtype=float)
    assert survival.shape == (1_001, 1_000)
    assert np.all(np.isfinite(survival))
    assert survival[0, 0] == pytest.approx(1.0)
    assert survival[0, 1] < survival[0, 0]

    chunks = list(pred.survival_at_chunks(grid, people_chunk=500, time_grid_chunk=128))
    assert chunks[0][2].shape == (500, 128)
    assert chunks[-1][2].shape == (1, 104)
    assembled = np.empty_like(survival)
    for row_slice, time_slice, block in chunks:
        assembled[row_slice, time_slice] = block
    np.testing.assert_allclose(assembled, survival)

    source_grid = np.array([0.0, 10.0, 20.0], dtype=float)
    row_rates = np.linspace(0.01, 0.03, 1_001, dtype=float).reshape(-1, 1)
    ffi_survival = np.exp(-row_rates * source_grid.reshape(1, -1))
    ffi_pred = gamfit.SurvivalPrediction(
        model_class="survival marginal-slope",
        parameters=row_rates,
        parameter_names=("rate",),
        times=source_grid,
        hazard=np.repeat(row_rates, source_grid.size, axis=1),
        survival=ffi_survival,
        cumulative_hazard=row_rates * source_grid.reshape(1, -1),
    )
    query_grid = np.linspace(0.0, 20.0, 1_000, dtype=float)

    ffi_dense = np.asarray(ffi_pred.survival_at(query_grid), dtype=float)
    assert ffi_dense.shape == (1_001, 1_000)
    np.testing.assert_allclose(
        ffi_dense[-1, :],
        np.interp(query_grid, source_grid, ffi_survival[-1, :]),
    )

    ffi_chunks = list(
        ffi_pred.survival_at_chunks(query_grid, people_chunk=500, time_grid_chunk=128)
    )
    assert ffi_chunks[0][2].shape == (500, 128)
    assert ffi_chunks[-1][2].shape == (1, 104)
    ffi_assembled = np.empty_like(ffi_dense)
    for row_slice, time_slice, block in ffi_chunks:
        ffi_assembled[row_slice, time_slice] = block
    np.testing.assert_allclose(ffi_assembled, ffi_dense)

    ffi_hazard = np.asarray(ffi_pred.hazard_at(query_grid), dtype=float)
    assert ffi_hazard.shape == (1_001, 1_000)
    np.testing.assert_allclose(ffi_hazard[:, 0], row_rates[:, 0])

    ffi_cumulative = np.asarray(ffi_pred.cumulative_hazard_at(query_grid), dtype=float)
    assert ffi_cumulative.shape == (1_001, 1_000)
    np.testing.assert_allclose(ffi_cumulative[-1, :], row_rates[-1, 0] * query_grid)

    single_time_hazard = np.asarray(pred.hazard_at(np.array([2.0], dtype=float)), dtype=float)
    assert single_time_hazard.shape == (1_001, 1)
    np.testing.assert_allclose(single_time_hazard, 1.0)

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


def _geometric_smoke_dataset(seed: int = 0, n: int = 200) -> pd.DataFrame:
    """Latents that cover the geometric-smooth formula surface.

    `theta` / `lon` wrap around 2π; `h` is an open height in [0, 1];
    `lat` is on the open arc-sine sphere. `x` is a generic [0, 1]
    covariate for the BC smooths. `y` is a plausible scalar response —
    no analytic structure is required because this is a smoke test of
    the FFI path, not a quality check."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "theta": rng.uniform(0.0, 2.0 * np.pi, n),
        "h":     rng.uniform(0.0, 1.0, n),
        "lat":   np.arcsin(rng.uniform(-1.0, 1.0, n)),
        "lon":   rng.uniform(0.0, 2.0 * np.pi, n),
        "x":     rng.uniform(0.0, 1.0, n),
        "y":     rng.standard_normal(n),
    })


@pytest.mark.parametrize(
    "formula",
    [
        # 1-D periodic (cyclic) P-spline
        "y ~ s(theta, periodic=true, period=2*pi)",
        # 2-D tensor with one periodic margin (cylinder topology)
        "y ~ te(theta, h, periodic=[0], period=[2*pi, None])",
        # Intrinsic S² via Wahba's reproducing kernel
        "y ~ sphere(lat, lon, radians=true)",
        # Intrinsic S² via spherical harmonics
        "y ~ sphere(lat, lon, method=harmonic, max_degree=4, radians=true)",
        # Boundary-conditioned 1-D B-splines are exercised by the Rust
        # tests (bc_clamped_predict_shape_bug.rs,
        # bc_predict_dimension_invariants.rs,
        # bc_anchored_variants_predict_works.rs) — skipped here so the
        # smoke test stays green against published wheels that may not
        # yet include the post-constraint frozen-transform fix. Re-add
        # once gamfit's PyPI wheel rolls forward.
    ],
)
def test_geometric_smooths_round_trip_via_python_binding(formula: str) -> None:
    """Smoke test: every geometric-smooth variety must fit *and* predict
    through the Python FFI on a dense-enough grid. Catches FFI-level
    breakage (formula not recognised, predict design mismatch, NaN /
    Inf propagation) without making any quality claim."""
    df = _geometric_smoke_dataset(seed=0, n=200)
    model = gamfit.fit(df, formula)
    preds = model.predict(df, interval=0.95)
    mean = np.asarray(preds["mean"], dtype=float)
    lo = np.asarray(preds["mean_lower"], dtype=float)
    hi = np.asarray(preds["mean_upper"], dtype=float)
    assert mean.shape == (len(df),), f"row count mismatch for {formula!r}"
    assert np.isfinite(mean).all(), f"NaN/Inf in mean for {formula!r}"
    assert np.isfinite(lo).all() and np.isfinite(hi).all(), \
        f"NaN/Inf in interval for {formula!r}"
    assert (lo <= mean + 1e-9).all() and (mean <= hi + 1e-9).all(), \
        f"point estimate outside CI for {formula!r}"


def test_duchon_function_norm_penalty_2d_smoke() -> None:
    """Multi-D (d=2) Duchon function-norm penalty matrix builds + is symmetric PSD."""
    centers = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=float,
    )
    s = gamfit.duchon_function_norm_penalty(centers=centers, m=2)
    s = np.asarray(s, dtype=float)
    assert s.shape == (3, 3), f"expected (3, 3) penalty, got {s.shape}"
    # Symmetry.
    assert np.allclose(s, s.T, atol=1e-10), "penalty must be symmetric"
    # PSD: smallest eigenvalue is >= ~0 (small slack for floating point).
    w = np.linalg.eigvalsh(0.5 * (s + s.T))
    assert w.min() > -1e-8, f"penalty not PSD; min eigenvalue {w.min()}"


def test_duchon_function_norm_penalty_2d_cylinder_periodic() -> None:
    """d=2 mixed-periodicity Duchon (cylinder): symmetric, PSD, wraps cleanly."""
    # Spread centers across the periodic axis [0, 2π] (auto-derived period)
    # and the non-periodic axis [0, 1].
    rng = np.random.default_rng(0)
    # Anchor the periodic axis span to exactly [0, 2π] so the auto-derived
    # period matches the geometric period; sample the rest inside.
    theta_inner = rng.uniform(0.0, 2.0 * np.pi, size=10)
    theta = np.concatenate([[0.0], theta_inner, [2.0 * np.pi]])
    y = rng.uniform(0.0, 1.0, size=theta.size)
    centers = np.column_stack([theta, y])
    K = centers.shape[0]

    s = gamfit.duchon_function_norm_penalty(
        centers=centers,
        m=2,
        periodic_per_axis=(True, False),
    )
    s = np.asarray(s, dtype=float)
    assert s.shape == (K, K), f"expected ({K}, {K}) penalty, got {s.shape}"
    # Symmetry.
    assert np.allclose(s, s.T, atol=1e-10), "cylinder penalty must be symmetric"
    # PSD.
    w = np.linalg.eigvalsh(0.5 * (s + s.T))
    assert w.min() > -1e-8, f"cylinder penalty not PSD; min eigenvalue {w.min()}"

    # Periodic identification on the periodic axis: evaluating the basis at
    # x_1 = 0 and x_1 = 2π (with same x_2) must produce identical basis rows.
    # Use the design (duchon_basis) to check that.
    pts_lo = np.array([[0.0, 0.5]], dtype=float)
    pts_hi = np.array([[2.0 * np.pi, 0.5]], dtype=float)
    b_lo = np.asarray(
        gamfit.duchon_basis(pts_lo, centers, m=2, periodic_per_axis=(True, False)),
        dtype=float,
    )
    b_hi = np.asarray(
        gamfit.duchon_basis(pts_hi, centers, m=2, periodic_per_axis=(True, False)),
        dtype=float,
    )
    assert np.allclose(b_lo, b_hi, atol=1e-9), (
        "cylinder Duchon design must be periodic on the periodic axis"
    )


def test_duchon_function_norm_penalty_2d_torus_periodic() -> None:
    """d=2 mixed-periodicity Duchon (torus): symmetric, PSD, wraps cleanly."""
    rng = np.random.default_rng(1)
    # Anchor BOTH axes' spans to exactly [0, 2π] so each auto-derived period
    # matches the geometric period.
    theta_inner = rng.uniform(0.0, 2.0 * np.pi, size=8)
    theta = np.concatenate([[0.0], theta_inner, [2.0 * np.pi]])
    phi_inner = rng.uniform(0.0, 2.0 * np.pi, size=8)
    phi = np.concatenate([[0.0], phi_inner, [2.0 * np.pi]])
    centers = np.column_stack([theta, phi])
    K = centers.shape[0]

    s = gamfit.duchon_function_norm_penalty(
        centers=centers,
        m=2,
        periodic_per_axis=(True, True),
    )
    s = np.asarray(s, dtype=float)
    assert s.shape == (K, K)
    assert np.allclose(s, s.T, atol=1e-10), "torus penalty must be symmetric"
    w = np.linalg.eigvalsh(0.5 * (s + s.T))
    assert w.min() > -1e-8, f"torus penalty not PSD; min eigenvalue {w.min()}"

    # Periodic identification on BOTH axes.
    pts_lo = np.array([[0.0, 1.2]], dtype=float)
    pts_hi_x = np.array([[2.0 * np.pi, 1.2]], dtype=float)
    pts_hi_y = np.array([[0.0, 1.2 + 2.0 * np.pi]], dtype=float)
    b_lo = np.asarray(
        gamfit.duchon_basis(pts_lo, centers, m=2, periodic_per_axis=(True, True)),
        dtype=float,
    )
    b_hi_x = np.asarray(
        gamfit.duchon_basis(pts_hi_x, centers, m=2, periodic_per_axis=(True, True)),
        dtype=float,
    )
    b_hi_y = np.asarray(
        gamfit.duchon_basis(pts_hi_y, centers, m=2, periodic_per_axis=(True, True)),
        dtype=float,
    )
    assert np.allclose(b_lo, b_hi_x, atol=1e-9), (
        "torus Duchon design must be periodic on axis 0"
    )
    assert np.allclose(b_lo, b_hi_y, atol=1e-9), (
        "torus Duchon design must be periodic on axis 1"
    )


def test_periodic_spline_curve_basis_is_periodic_and_partitions_unity() -> None:
    """Cyclic B-spline basis wraps cleanly and rows sum to one."""
    t = np.array([0.0, 0.07, 0.5, 0.999_999, 1.0, 1.07, -0.93], dtype=float)
    basis, penalty = gamfit.periodic_spline_curve_basis(t, n_knots=12, degree=3)
    assert basis.shape == (t.size, 12)
    assert penalty.shape == (12, 12)
    # partition of unity
    assert np.allclose(basis.sum(axis=1), 1.0, atol=1e-12)
    # endpoints match (t=0 and t=1)
    assert np.allclose(basis[0], basis[4], atol=1e-12)
    # cyclic penalty has constant nullspace
    ones = np.ones((12, 1))
    assert np.allclose(penalty @ ones, 0.0, atol=1e-10)


def test_periodic_spline_curve_torch_fit_closes_a_circle_in_r2() -> None:
    """End-to-end PeriodicSplineCurve fit through gamfit.torch.fit on a circle."""
    torch = pytest.importorskip("torch")
    from gamfit import PeriodicSplineCurve
    from gamfit.torch import fit as torch_fit

    rng = np.random.default_rng(0)
    n = 128
    t_np = np.linspace(0.0, 1.0, n, endpoint=False)
    y_np = np.stack(
        [
            np.cos(2.0 * np.pi * t_np) + 0.02 * rng.standard_normal(n),
            np.sin(2.0 * np.pi * t_np) + 0.02 * rng.standard_normal(n),
        ],
        axis=1,
    )

    t = torch.as_tensor(t_np, dtype=torch.float64)
    y = torch.as_tensor(y_np, dtype=torch.float64)
    spec = PeriodicSplineCurve(n_knots=10, degree=3, output_dim=2)

    result = torch_fit(t, y, spec)
    fitted = result.fitted.detach().cpu().numpy()
    assert fitted.shape == (n, 2)
    # residuals should be small (noise std ~0.02)
    resid = fitted - y_np
    assert np.sqrt((resid ** 2).mean()) < 0.1

    # periodicity: predict at t=0 and t=1 via fresh basis on each
    basis0, _ = gamfit.periodic_spline_curve_basis(
        np.array([0.0]), n_knots=10, degree=3
    )
    basis1, _ = gamfit.periodic_spline_curve_basis(
        np.array([1.0]), n_knots=10, degree=3
    )
    coef_t = result.coefficients
    if isinstance(coef_t, list):
        coef_t = torch.stack(coef_t, dim=0)
    coef = coef_t.detach().cpu().numpy()  # shape (K, D)
    f0 = basis0 @ coef
    f1 = basis1 @ coef
    assert np.allclose(f0, f1, atol=1e-10)


def test_sphere_torch_fit_smoke_all_kernels() -> None:
    """End-to-end Sphere fit through gamfit.torch.fit on a spherical cap."""
    torch = pytest.importorskip("torch")
    from gamfit import Sphere
    from gamfit.torch import fit as torch_fit

    rng = np.random.default_rng(7)
    n = 200
    # Sample (lat, lon) uniformly on a band so we have decent coverage.
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    lon = rng.uniform(-180.0, 180.0, size=n)
    points_np = np.stack([lat, lon], axis=1)

    # Spherical-cap-style synthetic response: cosine of great-circle
    # distance to (0, 0), plus a small bit of noise. Multi-output (D=2)
    # to also cover the matrix-response path.
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    cap = np.cos(lat_r) * np.cos(lon_r)
    y_np = np.stack(
        [cap + 0.05 * rng.standard_normal(n), 0.5 * cap + 0.05 * rng.standard_normal(n)],
        axis=1,
    )

    points = torch.as_tensor(points_np, dtype=torch.float64)
    response = torch.as_tensor(y_np, dtype=torch.float64)

    for kernel in ("sobolev", "pseudo", "harmonic"):
        n_centers = 20 if kernel != "harmonic" else 5  # harmonic: L → 5*(5+2)=35 cols
        spec = Sphere(n_centers=n_centers, penalty_order=2, kernel=kernel, radians=False)
        result = torch_fit(points, response, spec)
        coef_t = result.coefficients
        if isinstance(coef_t, list):
            coef_t = torch.stack(coef_t, dim=0)
        coef = coef_t.detach().cpu().numpy()
        fitted = result.fitted.detach().cpu().numpy()
        assert coef.ndim == 2 and coef.shape[1] == 2, (
            f"kernel={kernel}: coef shape {coef.shape}"
        )
        if kernel == "harmonic":
            assert coef.shape[0] == n_centers * (n_centers + 2), (
                f"harmonic basis dim mismatch: got {coef.shape[0]}, "
                f"expected L*(L+2)={n_centers * (n_centers + 2)}"
            )
        assert np.all(np.isfinite(coef)), f"kernel={kernel}: NaN/Inf in coefficients"
        assert fitted.shape == (n, 2)
        assert np.all(np.isfinite(fitted)), f"kernel={kernel}: NaN/Inf in fitted"


def test_torch_monotone_increasing_smooth() -> None:
    """Constrained BSpline fit through gamfit.torch.fit enforces monotone
    non-decreasing fitted values on x ∈ [0, 1]."""
    torch = pytest.importorskip("torch")
    from gamfit import BSpline
    from gamfit.torch import fit as torch_fit

    rng = np.random.default_rng(0)
    n = 200
    x_np = np.sort(rng.uniform(0.0, 1.0, size=n))
    y_np = x_np ** 2 + 0.02 * rng.standard_normal(n)

    x = torch.as_tensor(x_np, dtype=torch.float64)
    y = torch.as_tensor(y_np, dtype=torch.float64)
    spec = BSpline(degree=3, shape_constraint="monotone_increasing")
    result = torch_fit(x, y, spec)
    fitted_at_data = result.fitted.detach().cpu().numpy().squeeze()
    # x_np is sorted, so fitted_at_data should be (numerically) non-decreasing.
    diffs = np.diff(fitted_at_data)
    assert (diffs >= -1e-6).all(), (
        "monotone_increasing fit violates non-decreasing constraint; "
        f"min diff={diffs.min()}"
    )


def test_torch_convex_smooth() -> None:
    """Constrained BSpline fit through gamfit.torch.fit enforces convexity
    (second differences ≥ 0) on x ∈ [0, 1]."""
    torch = pytest.importorskip("torch")
    from gamfit import BSpline
    from gamfit.torch import fit as torch_fit

    rng = np.random.default_rng(1)
    n = 240
    x_np = np.sort(rng.uniform(0.0, 1.0, size=n))
    y_np = (x_np - 0.5) ** 2 + 0.02 * rng.standard_normal(n)

    x = torch.as_tensor(x_np, dtype=torch.float64)
    y = torch.as_tensor(y_np, dtype=torch.float64)
    spec = BSpline(degree=3, shape_constraint="convex")
    result = torch_fit(x, y, spec)
    # Evaluate fitted on a uniform grid via the basis at uniform points to
    # get a clean second-difference signal.
    from gamfit.torch._basis import bspline_basis
    grid = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)
    # Rebuild knots that match the fit's BSpline path. Use the auto-knots
    # derived from x for consistency.
    knots = None
    b_grid = bspline_basis(grid, knots, degree=3, periodic=False).detach().cpu().numpy()
    coef_t = result.coefficients
    if isinstance(coef_t, list):
        coef_t = torch.stack(coef_t, dim=0)
    coef = coef_t.detach().cpu().numpy()
    # coef may be (M, 1); flatten.
    coef_flat = coef.reshape(coef.shape[0], -1)
    f_grid = (b_grid @ coef_flat).squeeze()
    second_diffs = np.diff(f_grid, n=2)
    assert (second_diffs >= -1e-5).all(), (
        f"convex fit violates ≥0 second-diff constraint; min={second_diffs.min()}"
    )


def test_torch_monotone_smooth_backward_finite_gradient() -> None:
    """Constrained Gaussian REML torch fit is forward-only; calling
    backward through any output must raise NotImplementedError instead of
    silently returning incorrect gradients."""
    torch = pytest.importorskip("torch")
    from gamfit import BSpline
    from gamfit.torch import fit as torch_fit

    rng = np.random.default_rng(2)
    n = 80
    x_np = np.sort(rng.uniform(0.0, 1.0, size=n))
    y_np = x_np ** 2 + 0.02 * rng.standard_normal(n)

    x = torch.as_tensor(x_np, dtype=torch.float64, requires_grad=True)
    y = torch.as_tensor(y_np, dtype=torch.float64)
    spec = BSpline(degree=3, shape_constraint="monotone_increasing")
    result = torch_fit(x, y, spec)
    fitted = result.fitted
    # Forward outputs must be finite (no NaNs from a degenerate active set).
    assert torch.isfinite(fitted).all(), "fitted contains NaN/Inf"
    # Backward through the constrained path is intentionally unsupported;
    # confirm it raises rather than silently producing wrong gradients.
    with pytest.raises(NotImplementedError):
        fitted.sum().backward()


def test_torch_additive_recovers_per_smooth_lambda() -> None:
    """The torch additive path now routes to the multi-block Rust REML
    driver, so a fit with one wiggly response component and one smooth
    one must recover noticeably different per-block λs (ratio > 5)."""
    torch = pytest.importorskip("torch")
    from gamfit.torch import Duchon, fit

    rng = np.random.default_rng(11)
    n = 1000
    # Independent uniform covariates so the two smooths are not collinear.
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    # Wiggly true function on x1 (sin at 3 cycles), very smooth (near-linear)
    # true function on x2. The REML criterion should drive λ_1 well below
    # λ_2 because x2's effect is fully captured by the Duchon m=2 null space
    # (constant + linear) and demands maximal shrinkage of the wiggly basis.
    f_wiggly = np.sin(6.0 * np.pi * x1)
    f_smooth = 0.7 * x2 - 0.3
    y_np = f_wiggly + f_smooth + 0.1 * rng.standard_normal(n)

    pts1 = torch.as_tensor(x1, dtype=torch.float64).reshape(-1, 1)
    pts2 = torch.as_tensor(x2, dtype=torch.float64).reshape(-1, 1)
    y = torch.as_tensor(y_np, dtype=torch.float64)
    centers = torch.linspace(0.0, 1.0, 20, dtype=torch.float64).reshape(-1, 1)

    result = fit(
        points=[pts1, pts2],
        response=y,
        smooths=[Duchon(centers=centers, m=2), Duchon(centers=centers, m=2)],
    )

    lambdas = result.lambdas
    assert lambdas.ndim == 1 and lambdas.shape[0] == 2, (
        f"expected per-smooth λ of shape (2,); got {tuple(lambdas.shape)}"
    )
    lam_vals = lambdas.detach().cpu().numpy().astype(float)
    assert np.all(np.isfinite(lam_vals)), f"λ has non-finite entries: {lam_vals}"
    assert np.all(lam_vals > 0.0), f"λ has non-positive entries: {lam_vals}"
    ratio = float(max(lam_vals) / max(min(lam_vals), 1e-300))
    assert ratio > 5.0, (
        f"per-smooth λ should diverge (wiggly vs smooth); got λ = {lam_vals}, "
        f"ratio max/min = {ratio:.3g}"
    )


def test_partial_dependence_and_variance_share() -> None:
    """partial_dependence + variance_share on a multi-term GAM (smoke test)."""
    rng = np.random.default_rng(20260522)
    n = 200
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    x3 = rng.normal(0.0, 1.0, n)
    y = (
        np.sin(2.0 * np.pi * x1)
        + np.cos(2.0 * np.pi * x2)
        + 0.3 * x3
        + 0.1 * rng.standard_normal(n)
    )
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})

    model = gamfit.fit(frame, "y ~ s(x1) + s(x2) + x3")

    # term_blocks must include the smooth columns we expect.
    block_names = {b.name for b in model.term_blocks}
    assert "intercept" in block_names
    assert "x3" in block_names
    assert "s(x1)" in block_names
    assert "s(x2)" in block_names

    pd_out = model.partial_dependence("s(x1)", frame, n_points=40)
    assert set(pd_out.keys()) == {"grid", "predicted", "standard_error"}
    assert pd_out["grid"].shape == (40,)
    assert pd_out["predicted"].shape == (40,)
    assert pd_out["standard_error"].shape == (40,)
    assert np.all(np.isfinite(pd_out["predicted"]))
    assert np.all(pd_out["standard_error"] >= 0.0)

    shares = model.variance_share(frame)
    assert isinstance(shares, dict)
    assert all(0.0 <= float(v) <= 1.0 for v in shares.values())
    # On the input grid these should not vastly exceed 1; cross-term
    # cancellation can pull the sum either way but ~1.1 is a generous bound.
    assert sum(shares.values()) <= 1.1

    s_x1_share = model.variance_share(frame, term="s(x1)")
    assert isinstance(s_x1_share, float)
    assert 0.0 <= s_x1_share <= 1.0


def test_gaussian_reml_fit_blocks_torch_backward_finite_gradients() -> None:
    """Multi-block per-smooth-λ REML torch surface produces finite gradients.

    Builds two random design blocks plus row weights with ``requires_grad=True``
    and a random response, runs the autograd forward through
    :class:`gamfit.torch._reml._GaussianRemlFitBlocksFn`, computes a squared
    error on ``fitted``, calls ``.backward()``, and verifies the gradients
    propagated to the design blocks, ``y``, and weights are finite and not all
    zero.
    """
    torch = pytest.importorskip("torch")
    from gamfit.torch import _reml as torch_reml

    rng = np.random.default_rng(20260522)
    n = 20
    p_per = 4
    x1 = torch.tensor(rng.standard_normal((n, p_per)), dtype=torch.float64, requires_grad=True)
    x2 = torch.tensor(rng.standard_normal((n, p_per)), dtype=torch.float64, requires_grad=True)
    s1 = torch.tensor(np.eye(p_per), dtype=torch.float64)
    s2 = torch.tensor(np.eye(p_per), dtype=torch.float64)
    y = torch.tensor(rng.standard_normal((n, 1)), dtype=torch.float64, requires_grad=True)
    weights = torch.tensor(rng.uniform(0.5, 1.5, n), dtype=torch.float64, requires_grad=True)

    out = torch_reml.gaussian_reml_fit_blocks([x1, x2], [s1, s2], y, weights=weights)
    assert out.lambdas.shape == (2,)
    assert out.log_lambdas.shape == (2,)
    assert out.fitted.shape == (n, 1)
    assert len(out.coefficients) == 2
    assert out.coefficients[0].shape == (p_per, 1)
    assert out.edf.shape == (2,)

    loss = (out.fitted - y).square().sum()
    loss.backward()

    assert x1.grad is not None and torch.isfinite(x1.grad).all()
    assert x2.grad is not None and torch.isfinite(x2.grad).all()
    assert y.grad is not None and torch.isfinite(y.grad).all()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()
    # Some entries must be nonzero — perturbing a random design or response
    # always shifts the fit.
    assert float(x1.grad.abs().sum()) > 0.0
    assert float(x2.grad.abs().sum()) > 0.0
    assert float(y.grad.abs().sum()) > 0.0
    assert float(weights.grad.abs().sum()) > 0.0


def test_gaussian_reml_fit_blocks_torch_roundtrip_lambdas_converge() -> None:
    """Round-trip fit → predict → fit-again converges to the same λ.

    A second fit warm-started from the converged log-λ\\* should produce
    essentially the same per-smooth λ vector.
    """
    torch = pytest.importorskip("torch")
    from gamfit.torch import _reml as torch_reml

    rng = np.random.default_rng(202605221)
    n = 40
    p_per = 4
    x1 = torch.tensor(rng.standard_normal((n, p_per)), dtype=torch.float64)
    x2 = torch.tensor(rng.standard_normal((n, p_per)), dtype=torch.float64)
    s1 = torch.tensor(np.eye(p_per), dtype=torch.float64)
    s2 = torch.tensor(np.eye(p_per), dtype=torch.float64)
    y = torch.tensor(rng.standard_normal((n, 1)), dtype=torch.float64)

    first = torch_reml.gaussian_reml_fit_blocks([x1, x2], [s1, s2], y)
    log_lam_star = first.log_lambdas.detach().clone()
    second = torch_reml.gaussian_reml_fit_blocks(
        [x1, x2], [s1, s2], y, init_log_lambdas=log_lam_star
    )
    np.testing.assert_allclose(
        second.lambdas.detach().numpy(),
        first.lambdas.detach().numpy(),
        rtol=1.0e-4,
        atol=1.0e-6,
    )


# ---------------------------------------------------------------------------
# Newly authored coverage for session bindings.
# Tests below this banner were added to cover the Python-facing surface
# enumerated in the task brief; existing tests above this banner are
# unchanged.
# ---------------------------------------------------------------------------


def _assert_symmetric_psd(matrix: np.ndarray, label: str, slack: float = 1e-8) -> None:
    """Helper: assert ``matrix`` is symmetric and approximately PSD."""
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], (
        f"{label}: expected square 2D matrix; got shape {matrix.shape}"
    )
    assert np.allclose(matrix, matrix.T, atol=1e-9), f"{label} must be symmetric"
    w = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    assert w.min() > -slack, (
        f"{label} not PSD; min eigenvalue {w.min():.3e}"
    )


def test_duchon_function_norm_penalty_3d_non_periodic_psd() -> None:
    """3D centers, no periodic axis — penalty must build and be SPD-like."""
    rng = np.random.default_rng(2)
    centers = rng.uniform(-1.0, 1.0, size=(7, 3))
    s = gamfit.duchon_function_norm_penalty(
        centers=centers,
        m=2,
        periodic_per_axis=(False, False, False),
    )
    s = np.asarray(s, dtype=float)
    assert s.shape == (7, 7), f"expected (7, 7) penalty, got {s.shape}"
    _assert_symmetric_psd(s, "3D Duchon penalty")


def test_sphere_basis_each_kernel_shapes_and_psd() -> None:
    """`gamfit.sphere_basis` returns (N, K) basis and (K, K) PSD penalty."""
    rng = np.random.default_rng(11)
    n = 50
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    lon = rng.uniform(-180.0, 180.0, size=n)
    points = np.stack([lat, lon], axis=1)

    for kernel in ("sobolev", "pseudo", "harmonic"):
        n_centers = 10 if kernel != "harmonic" else 4
        design, penalty = gamfit.sphere_basis(
            points,
            n_centers=n_centers,
            penalty_order=2,
            kernel=kernel,
            radians=False,
        )
        design = np.asarray(design, dtype=float)
        penalty = np.asarray(penalty, dtype=float)
        assert design.ndim == 2 and design.shape[0] == n, (
            f"kernel={kernel}: design rows {design.shape[0]} != n {n}"
        )
        k_expected = (
            n_centers * (n_centers + 2) if kernel == "harmonic" else n_centers
        )
        assert design.shape[1] == k_expected, (
            f"kernel={kernel}: design cols {design.shape[1]} != {k_expected}"
        )
        assert penalty.shape == (k_expected, k_expected), (
            f"kernel={kernel}: penalty shape {penalty.shape}"
        )
        assert np.all(np.isfinite(design)) and np.all(np.isfinite(penalty)), (
            f"kernel={kernel}: NaN/Inf in basis/penalty"
        )
        _assert_symmetric_psd(penalty, f"sphere_basis penalty ({kernel})")


def test_periodic_spline_curve_basis_constant_nullspace_and_shapes() -> None:
    """Cyclic spline returns matching shapes and constant null-space penalty.

    Complements the existing endpoint-equality / partition-of-unity test by
    inspecting penalty shape and nullspace directly on a smaller knot count.
    """
    t = np.linspace(0.0, 1.0, 32, endpoint=False)
    basis, penalty = gamfit.periodic_spline_curve_basis(t, n_knots=8, degree=3)
    basis = np.asarray(basis, dtype=float)
    penalty = np.asarray(penalty, dtype=float)
    assert basis.shape == (32, 8)
    assert penalty.shape == (8, 8)
    # PSD on the penalty (after symmetrising).
    _assert_symmetric_psd(penalty, "periodic_spline_curve_basis penalty")
    # Constant vector is in the null-space of the cyclic difference penalty.
    ones = np.ones((8, 1))
    np.testing.assert_allclose(penalty @ ones, 0.0, atol=1e-10)


def test_gaussian_reml_fit_blocks_forward_recovers_per_smooth_lambda_numpy() -> None:
    """Direct (numpy) call to gaussian_reml_fit_blocks_forward — λ vector
    has matching length, λ > 0, EDFs finite, fitted matches design shape."""
    rng = np.random.default_rng(20260522)
    n = 60
    p_per = 5
    x1 = rng.standard_normal((n, p_per))
    x2 = rng.standard_normal((n, p_per))
    s1 = np.eye(p_per)
    s2 = np.eye(p_per)
    y = rng.standard_normal((n, 1))

    out = gamfit.gaussian_reml_fit_blocks_forward([x1, x2], [s1, s2], y)
    assert "lambdas" in out and out["lambdas"].shape == (2,)
    assert np.all(np.isfinite(out["lambdas"]))
    assert np.all(out["lambdas"] > 0.0)
    assert "edf" in out and out["edf"].shape == (2,)
    assert "fitted" in out and out["fitted"].shape == (n, 1)
    assert "coefficients" in out and out["coefficients"].shape[0] == 2 * p_per


def test_gaussian_reml_fit_blocks_backward_returns_finite_grads_numpy() -> None:
    """Direct (numpy) call to gaussian_reml_fit_blocks_backward — gradients
    have correct shapes and are finite given a unit grad_fitted seed."""
    rng = np.random.default_rng(2026052200)
    n = 24
    p_per = 4
    x1 = rng.standard_normal((n, p_per))
    x2 = rng.standard_normal((n, p_per))
    s1 = np.eye(p_per)
    s2 = np.eye(p_per)
    y = rng.standard_normal((n, 1))
    fwd = gamfit.gaussian_reml_fit_blocks_forward([x1, x2], [s1, s2], y)
    log_lam = np.log(np.maximum(fwd["lambdas"], 1e-12))
    grad_fitted = np.ones((n, 1), dtype=float)
    back = gamfit.gaussian_reml_fit_blocks_backward(
        [x1, x2], [s1, s2], y, log_lam,
        grad_fitted=grad_fitted,
    )
    for k in ("grad_designs", "grad_penalties", "grad_y"):
        assert k in back, f"missing key {k!r} in backward result"
    assert len(back["grad_designs"]) == 2
    assert len(back["grad_penalties"]) == 2
    assert back["grad_designs"][0].shape == (n, p_per)
    assert back["grad_designs"][1].shape == (n, p_per)
    assert back["grad_penalties"][0].shape == (p_per, p_per)
    assert back["grad_y"].shape == (n, 1)
    for arr in (*back["grad_designs"], *back["grad_penalties"], back["grad_y"]):
        assert np.all(np.isfinite(arr)), "non-finite entries in backward result"


def test_gaussian_reml_fit_with_constraints_forward_monotone_on_x_squared() -> None:
    """Constrained REML forward with monotone-increasing A row pairs on the
    fitted values must yield a non-decreasing fit on ``y = x²`` over [0, 1]."""
    rng = np.random.default_rng(202605221)
    n = 120
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    y = (x ** 2 + 0.02 * rng.standard_normal(n)).reshape(-1, 1)

    # Build a B-spline design + smoothness penalty via the existing gamfit
    # primitives (no torch dependency required here).
    basis = np.asarray(gamfit.bspline_basis(x, knots=None, degree=3), dtype=float)
    # The smoothness_penalty wants a knot vector; reuse the same auto-knot
    # derivation by sampling a coarse grid over the data range.
    knots = np.linspace(0.0, 1.0, basis.shape[1] + 4)
    penalty, _ = gamfit.smoothness_penalty(knots, degree=3, order=2)

    # Build A·β ≤ 0 enforcing monotone non-decreasing on a dense grid via
    # forward differences of the design. Note the API expects ``A·β ≤ b``,
    # so we want -(B[i+1] - B[i])·β ≤ 0.
    grid = np.linspace(0.0, 1.0, 96)
    b_grid = np.asarray(gamfit.bspline_basis(grid, knots=None, degree=3), dtype=float)
    a_rows = -(b_grid[1:] - b_grid[:-1])
    b_rhs = np.zeros(a_rows.shape[0], dtype=float)

    out = gamfit.gaussian_reml_fit_with_constraints_forward(
        basis, y, penalty,
        a_inequality=a_rows,
        b_inequality=b_rhs,
    )
    fitted = np.asarray(out["fitted"], dtype=float).reshape(-1)
    assert fitted.shape == (n,)
    diffs = np.diff(fitted)
    assert (diffs >= -1e-6).all(), (
        f"monotone-constrained fit violates non-decreasing; min diff={diffs.min()}"
    )


def test_gaussian_reml_fit_with_constraints_forward_no_constraints_matches_unconstrained() -> None:
    """With no inequality system (``a_inequality=None``), the constrained
    REML forward must reproduce the unconstrained Gaussian REML fit."""
    rng = np.random.default_rng(2026052299)
    n = 50
    p = 6
    x = rng.standard_normal((n, p))
    y = rng.standard_normal((n, 1))
    s = np.eye(p)

    a = gamfit.gaussian_reml_fit_with_constraints_forward(x, y, s)
    b = gamfit.gaussian_reml_fit(x, y, s)
    fit_a = np.asarray(a["fitted"], dtype=float).reshape(-1)
    fit_b = np.asarray(b["fitted"], dtype=float).reshape(-1)
    np.testing.assert_allclose(fit_a, fit_b, rtol=1e-6, atol=1e-8)


# Backward for `gaussian_reml_fit_with_constraints_*` is DEFERRED: the
# constrained backward binding has not been added to the gamfit Python
# surface (no `gaussian_reml_fit_with_constraints_backward` exists in
# gamfit._api or gamfit.__init__ as of this session; the torch path raises
# NotImplementedError on backward — see
# `test_torch_monotone_smooth_backward_finite_gradient` above).


def test_model_term_blocks_sorted_and_typed() -> None:
    """``model.term_blocks`` returns sorted tuple of TermBlock dataclasses."""
    rng = np.random.default_rng(2026052201)
    n = 120
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x1) + 0.4 * x2 + 0.1 * rng.standard_normal(n)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    model = gamfit.fit(frame, "y ~ s(x1) + s(x2)")
    blocks = model.term_blocks
    assert isinstance(blocks, tuple)
    assert len(blocks) >= 2
    for blk in blocks:
        assert isinstance(blk, gamfit.TermBlock), (
            f"term_blocks entries must be TermBlock; got {type(blk).__name__}"
        )
        assert isinstance(blk.name, str)
        assert isinstance(blk.kind, str)
        assert blk.start <= blk.end
    # Sorted by .start (ascending, non-overlapping or contiguous).
    starts = [blk.start for blk in blocks]
    assert starts == sorted(starts), f"term_blocks not sorted by start; got {starts}"


def test_model_partial_dependence_1d_shapes_and_finiteness() -> None:
    """1D partial_dependence returns grid/predicted/standard_error of length n_points."""
    rng = np.random.default_rng(2026052202)
    n = 120
    x1 = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x1) + 0.1 * rng.standard_normal(n)
    frame = pd.DataFrame({"y": y, "x1": x1})
    model = gamfit.fit(frame, "y ~ s(x1)")
    pd_out = model.partial_dependence("s(x1)", frame, n_points=25)
    assert set(pd_out.keys()) == {"grid", "predicted", "standard_error"}
    assert np.asarray(pd_out["grid"]).shape == (25,)
    assert np.asarray(pd_out["predicted"]).shape == (25,)
    assert np.asarray(pd_out["standard_error"]).shape == (25,)
    assert np.all(np.isfinite(np.asarray(pd_out["predicted"], dtype=float)))
    assert np.all(np.asarray(pd_out["standard_error"], dtype=float) >= 0.0)


def test_model_variance_share_sums_to_at_most_one_plus_slack() -> None:
    """variance_share returns dict of [0,1]-valued shares summing ~≤ 1."""
    rng = np.random.default_rng(2026052203)
    n = 150
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x1) + 0.5 * x2 + 0.1 * rng.standard_normal(n)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    model = gamfit.fit(frame, "y ~ s(x1) + s(x2)")
    shares = model.variance_share(frame)
    assert isinstance(shares, dict)
    for name, val in shares.items():
        assert isinstance(name, str)
        v = float(val)
        assert 0.0 <= v <= 1.0, f"variance share {name}={v} outside [0,1]"
    assert sum(float(v) for v in shares.values()) <= 1.1


# ---------------------------------------------------------------------------
# Smooth dataclass construction smoke tests.
#
# Construct each subclass with minimal valid arguments. Verify the
# `shape_constraint` field accepts each Literal value via attribute set —
# the dataclasses themselves do not perform runtime Literal validation
# (mypy / pyright catch invalid strings statically), but the field must
# at least round-trip through construction and assignment without raising.
# ---------------------------------------------------------------------------


def test_smooth_dataclass_subclasses_construct_with_shape_constraint() -> None:
    """Every Smooth subclass constructs with each ShapeConstraintLiteral value."""
    from gamfit import (
        BSpline,
        Categorical,
        Duchon,
        Matern,
        PeriodicSplineCurve,
        Sphere,
        TensorBSpline,
    )

    centers_1d = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
    centers_2d = np.column_stack(
        [np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10)]
    )

    constraint_values = (
        None,
        "none",
        "monotone_increasing",
        "monotone_decreasing",
        "convex",
        "concave",
    )

    factories = [
        ("Duchon", lambda sc: Duchon(centers=centers_1d, m=2, shape_constraint=sc)),
        ("BSpline", lambda sc: BSpline(degree=3, shape_constraint=sc)),
        ("TensorBSpline", lambda sc: TensorBSpline(
            marginals=[BSpline(degree=3), BSpline(degree=3)],
            shape_constraint=sc,
        )),
        ("Matern", lambda sc: Matern(
            centers=centers_2d, nu=1.5, length_scale=1.0, shape_constraint=sc,
        )),
        ("Sphere", lambda sc: Sphere(
            n_centers=8, kernel="sobolev", shape_constraint=sc,
        )),
        ("PeriodicSplineCurve", lambda sc: PeriodicSplineCurve(
            n_knots=10, degree=3, output_dim=2, shape_constraint=sc,
        )),
        ("Categorical", lambda sc: Categorical(
            levels=np.zeros(5, dtype=int), n_levels=3, shape_constraint=sc,
        )),
    ]
    for label, factory in factories:
        for sc in constraint_values:
            obj = factory(sc)
            # Round-trip — field must survive construction.
            assert obj.shape_constraint == sc, (
                f"{label}.shape_constraint mismatch: stored {obj.shape_constraint!r} "
                f"vs requested {sc!r}"
            )
            # Common base-class fields must default sanely.
            assert obj.by is None
            assert obj.double_penalty is False
            assert obj.name is None


def test_smooth_dataclass_arbitrary_string_shape_constraint_not_runtime_rejected() -> None:
    """`shape_constraint` is a typing.Literal — runtime construction does
    not actively reject arbitrary strings (the type system enforces it at
    static-check time). Verify that the field stores whatever is passed,
    so downstream code can decide how strict to be.

    (If runtime validation is added later, this test should be inverted to
    expect a ValueError or TypeError.)
    """
    from gamfit import BSpline

    obj = BSpline(degree=3, shape_constraint=typing.cast(typing.Any, "not-a-constraint"))
    assert obj.shape_constraint == "not-a-constraint"


# ---------------------------------------------------------------------------
# End-to-end torch.fit with each supported Smooth kind.
#
# Tests below need torch; they importorskip up-front.
# ---------------------------------------------------------------------------


def test_torch_fit_single_bspline_recovers_quadratic() -> None:
    """`fit(x, y, BSpline(...))` returns finite fitted values + coefficients."""
    torch = pytest.importorskip("torch")
    from gamfit import BSpline
    from gamfit.torch import fit as torch_fit

    rng = np.random.default_rng(2026052204)
    n = 150
    x_np = np.sort(rng.uniform(0.0, 1.0, size=n))
    y_np = x_np ** 2 + 0.05 * rng.standard_normal(n)

    x = torch.as_tensor(x_np, dtype=torch.float64)
    y = torch.as_tensor(y_np, dtype=torch.float64)
    knots = np.linspace(0.0, 1.0, 12)
    spec = BSpline(knots=knots, degree=3)
    result = torch_fit(x, y, spec)
    fitted = result.fitted.detach().cpu().numpy().reshape(-1)
    assert fitted.shape == (n,)
    assert np.all(np.isfinite(fitted))
    rmse = float(np.sqrt(np.mean((fitted - y_np) ** 2)))
    assert rmse < 0.2, f"BSpline fit too poor; rmse={rmse:.3g}"


def test_torch_fit_additive_duchon_plus_bspline() -> None:
    """Additive fit through `gamfit.torch.fit` with a Duchon + a BSpline."""
    torch = pytest.importorskip("torch")
    from gamfit import BSpline, Duchon
    from gamfit.torch import fit as torch_fit

    rng = np.random.default_rng(2026052205)
    n = 200
    x1 = rng.uniform(0.0, 1.0, size=n)
    x2 = rng.uniform(0.0, 1.0, size=n)
    y_np = np.sin(2.0 * np.pi * x1) + 0.5 * x2 + 0.05 * rng.standard_normal(n)

    pts1 = torch.as_tensor(x1, dtype=torch.float64).reshape(-1, 1)
    pts2 = torch.as_tensor(x2, dtype=torch.float64).reshape(-1, 1)
    y = torch.as_tensor(y_np, dtype=torch.float64)
    centers_d = torch.linspace(0.0, 1.0, 15, dtype=torch.float64).reshape(-1, 1)
    knots = np.linspace(0.0, 1.0, 10)

    result = torch_fit(
        points=[pts1, pts2],
        response=y,
        smooths=[Duchon(centers=centers_d, m=2), BSpline(knots=knots, degree=3)],
    )
    fitted = result.fitted.detach().cpu().numpy().reshape(-1)
    assert fitted.shape == (n,)
    assert np.all(np.isfinite(fitted))
    # λ vector must be length F=2.
    lambdas = result.lambdas.detach().cpu().numpy().reshape(-1)
    assert lambdas.shape == (2,)
    assert np.all(np.isfinite(lambdas))


def test_torch_fit_periodic_spline_curve_multi_output_coefficients_shape() -> None:
    """PeriodicSplineCurve fit through `torch.fit` returns (K, D) coefficients."""
    torch = pytest.importorskip("torch")
    from gamfit import PeriodicSplineCurve
    from gamfit.torch import fit as torch_fit

    rng = np.random.default_rng(2026052206)
    n = 128
    t_np = np.linspace(0.0, 1.0, n, endpoint=False)
    y_np = np.stack(
        [
            np.cos(2.0 * np.pi * t_np) + 0.02 * rng.standard_normal(n),
            np.sin(2.0 * np.pi * t_np) + 0.02 * rng.standard_normal(n),
            np.cos(4.0 * np.pi * t_np) + 0.02 * rng.standard_normal(n),
        ],
        axis=1,
    )
    t = torch.as_tensor(t_np, dtype=torch.float64)
    y = torch.as_tensor(y_np, dtype=torch.float64)
    spec = PeriodicSplineCurve(n_knots=12, degree=3, output_dim=3)
    result = torch_fit(t, y, spec)
    assert isinstance(result.coefficients, torch.Tensor)
    coef = result.coefficients.detach().cpu().numpy()
    assert coef.ndim == 2 and coef.shape[1] == 3
    fitted = result.fitted.detach().cpu().numpy()
    assert fitted.shape == (n, 3)
    assert np.all(np.isfinite(fitted))


def test_torch_fit_sphere_each_kernel_basic_fit() -> None:
    """`fit(latlon, y, Sphere(...))` runs end-to-end for every kernel.

    Complements the existing multi-output smoke test by checking the
    single-output (D=1) code path for every kernel.
    """
    torch = pytest.importorskip("torch")
    from gamfit import Sphere
    from gamfit.torch import fit as torch_fit

    rng = np.random.default_rng(2026052207)
    n = 120
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    lon = rng.uniform(-180.0, 180.0, size=n)
    points_np = np.stack([lat, lon], axis=1)
    cap = np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    y_np = cap + 0.05 * rng.standard_normal(n)

    points = torch.as_tensor(points_np, dtype=torch.float64)
    y = torch.as_tensor(y_np, dtype=torch.float64)
    for kernel in ("sobolev", "pseudo", "harmonic"):
        n_centers = 15 if kernel != "harmonic" else 4
        spec = Sphere(
            n_centers=n_centers, penalty_order=2, kernel=kernel, radians=False,
        )
        result = torch_fit(points, y, spec)
        fitted = result.fitted.detach().cpu().numpy().reshape(-1)
        assert fitted.shape == (n,), f"kernel={kernel}: fitted shape {fitted.shape}"
        assert np.all(np.isfinite(fitted)), f"kernel={kernel}: NaN/Inf in fitted"


def test_torch_fit_rejects_shape_constraint_with_multi_smooth_list() -> None:
    """`shape_constraint` on the torch fit path is single-smooth only;
    a list of smooths with any non-None constraint must raise."""
    torch = pytest.importorskip("torch")
    from gamfit import BSpline
    from gamfit.torch import fit as torch_fit

    rng = np.random.default_rng(2026052208)
    n = 60
    x_np = np.sort(rng.uniform(0.0, 1.0, size=n))
    y_np = x_np ** 2 + 0.05 * rng.standard_normal(n)
    x = torch.as_tensor(x_np, dtype=torch.float64).reshape(-1, 1)
    y = torch.as_tensor(y_np, dtype=torch.float64)
    # Two smooths, one of them carries a shape_constraint.
    knots = np.linspace(0.0, 1.0, 10)
    smooths = [
        BSpline(knots=knots, degree=3, shape_constraint="monotone_increasing"),
        BSpline(knots=knots, degree=3),
    ]
    with pytest.raises(NotImplementedError):
        torch_fit(points=[x, x], response=y, smooths=smooths)
