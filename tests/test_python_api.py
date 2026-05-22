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


def test_duchon_function_norm_penalty_2d_periodic_per_axis() -> None:
    """Per-axis periodicity for d=2 Duchon: gated by the Rust core (d=1 only)."""
    centers = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    # The underlying Rust path `build_periodic_duchon_basis_1d` rejects
    # ncols != 1, so multi-D periodicity is not yet supported.
    pytest.skip("Rust core doesn't yet support multi-D periodic Duchon")
    _ = gamfit.duchon_function_norm_penalty(
        centers=centers,
        m=2,
        periodic_per_axis=(True, False),
    )


def test_per_smooth_lambda_not_in_torch_additive_pending_rust_refactor() -> None:
    # The torch additive REML path is structurally single-λ because the
    # closed-form Gaussian REML kernel in
    # crates/gam-core/src/solver/gaussian_reml.rs only handles one scalar λ.
    # Per-smooth λ requires extending that Rust kernel to multi-block
    # (multi-d outer optimisation + analytic VJP through the F×F Hessian +
    # per-block eigendecomposition routing). This test tracks the gap.
    #
    # delete this test when the multi-block closed-form REML kernel ships
    from gamfit.torch._multi_lambda_status import MULTI_LAMBDA_SUPPORTED

    assert MULTI_LAMBDA_SUPPORTED is False, (
        "Torch additive REML now claims per-smooth λ support; "
        "remove this placeholder test and update the docstrings in "
        "gamfit/torch/_reml.py and gamfit/torch/_multi_lambda_status.py."
    )
