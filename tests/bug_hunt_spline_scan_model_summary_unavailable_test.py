"""Bug hunt: a spline-scan-routed 1-D Gaussian smooth loses its entire
summary / introspection API (#1046).

Since #1030 the standard workflow auto-detects the single-1-D-smooth,
Gaussian-identity shape and routes it through the exact O(n) state-space spline
scan (`crate::solver::spline_scan::fit_spline_scan`); #1044 extended this to the
order-3 quintic smoother. A scan-bearing `SavedModel` carries a `SplineScanFit`
state and **no dense `fit_result`** — by design, the scan keeps no dense
design/Gram.

The predict path is scan-aware, but every Python-FFI summary/introspection entry
point funnelled through one chokepoint (`fit_result_from_saved_model_for_prediction`)
that demanded a dense `fit_result`. For any scan-routed model
(`double_penalty=false`, degree == 2*order-1) they therefore all aborted with::

    ValueError: model is missing canonical fit_result payload; refit

even though the model fits and predicts perfectly and the selected smoothing
parameter, EDF and REML score are all present in the saved `SplineScanFit`.

This test fits both the cubic (`degree=3, penalty_order=2`) and quintic
(`degree=5, penalty_order=3`) scan-routed forms, asserts the model actually
predicts and tracks the signal, then asserts every summary/introspection entry
point returns a finite, principled value reconstructed from the scan state.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _dataset(seed: int = 7, n: int = 140) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y = np.sin(2.5 * np.pi * x) + 0.4 * x + rng.normal(0.0, 0.1, n)
    return pd.DataFrame({"x": x, "y": y})


# (degree, penalty_order, null_space_dim==order) for every scan-eligible order.
_SCAN_FORMS = [
    pytest.param(1, 1, 1, id="linear-order1"),
    pytest.param(3, 2, 2, id="cubic-order2"),
    pytest.param(5, 3, 3, id="quintic-order3"),
]


def _fit_scan(df: pd.DataFrame, degree: int, penalty_order: int) -> gamfit.Model:
    formula = (
        f'y ~ s(x, bs="ps", degree={degree}, '
        f"penalty_order={penalty_order}, double_penalty=False)"
    )
    return gamfit.fit(df, formula)


@pytest.mark.parametrize("degree, penalty_order, order", _SCAN_FORMS)
def test_scan_model_predicts_and_summarizes(degree, penalty_order, order):
    df = _dataset()
    n = len(df)
    model = _fit_scan(df, degree, penalty_order)

    # The model fits and predicts, tracking the signal.
    pred = np.asarray(model.predict(df))
    assert np.all(np.isfinite(pred))
    assert np.corrcoef(pred, df["y"].to_numpy())[0, 1] > 0.8

    # smoothing_parameters(): a positive, finite selected lambda.
    lambdas = model.smoothing_parameters()
    assert isinstance(lambdas, dict)
    assert len(lambdas) >= 1
    for value in lambdas.values():
        assert np.isfinite(value)
        assert value > 0.0

    # summary(): finite REML score and an EDF strictly between the polynomial
    # null-space dimension (== order) and n.
    summary = model.summary()
    reml = float(summary.reml_score)
    assert np.isfinite(reml)
    edf = float(summary.edf_total)
    assert order < edf < n, f"edf {edf} must lie in ({order}, {n})"

    # evidence(): a finite REML/LAML cost on the comparison scale.
    assert np.isfinite(float(model.evidence))

    # term_blocks: exactly one contiguous coefficient block for the smooth.
    blocks = model.term_blocks
    assert len(blocks) >= 1
    total = 0
    for blk in blocks:
        assert 0 <= blk.start <= blk.end
        total += blk.end - blk.start
    assert total >= 1
    assert any(blk.kind == "smooth" for blk in blocks)

    # diagnose(): scores the fit on held-out data (routes through summary()).
    diag = model.diagnose(df)
    assert diag is not None


def test_scan_summary_matches_dense_double_penalty_reference():
    """The scan and the dense (double_penalty=true) fit of the *same* smooth
    must agree on the headline fitted quantities: comparable EDF and a fit that
    tracks the signal equally well. This guards against the scan summary path
    fabricating numbers that diverge from the canonical dense introspection."""
    df = _dataset()
    scan = _fit_scan(df, degree=3, penalty_order=2)
    dense = gamfit.fit(
        df, 'y ~ s(x, bs="ps", degree=3, penalty_order=2, double_penalty=True)'
    )

    edf_scan = float(scan.summary().edf_total)
    edf_dense = float(dense.summary().edf_total)
    # Different penalty structure (single vs double penalty), so not identical,
    # but both must land in a sane smooth band and within a factor of ~2.
    assert 2.0 < edf_scan < len(df)
    assert 2.0 < edf_dense < len(df)
    assert 0.4 < edf_scan / edf_dense < 2.5

    yhat_scan = np.asarray(scan.predict(df))
    yhat_dense = np.asarray(dense.predict(df))
    y = df["y"].to_numpy()
    rmse_scan = float(np.sqrt(np.mean((yhat_scan - y) ** 2)))
    rmse_dense = float(np.sqrt(np.mean((yhat_dense - y) ** 2)))
    # The scan must not be meaningfully worse than the dense reference.
    assert rmse_scan <= 1.25 * rmse_dense


def test_scan_design_matrix_gives_actionable_error():
    """A scan model retains no dense design, so design_matrix() must fail with a
    precise structural message (not a model-swapping fallback recommendation)."""
    df = _dataset()
    model = _fit_scan(df, degree=3, penalty_order=2)
    with pytest.raises(Exception) as exc:
        model.design_matrix(df)
    msg = str(exc.value).lower()
    assert "spline scan" in msg
    assert "finite coefficient-frame design" in msg
    assert "double_penalty" not in msg


def test_scan_predict_conformal_gives_actionable_error():
    """Split-conformal needs the dense predictor a scan model does not carry, so
    predict_conformal() must fail with a precise message pointing to the
    scan-aware posterior-interval path, not the cryptic resolved_termspec one."""
    df = _dataset(n=200)
    model = _fit_scan(df, degree=3, penalty_order=2)
    tr, cal, te = df.iloc[:120], df.iloc[120:160], df.iloc[160:]
    del tr
    with pytest.raises(Exception) as exc:
        model.predict_conformal(te, calibration=cal, conformal_level=0.9, return_type="dict")
    msg = str(exc.value).lower()
    assert "spline scan" in msg
    assert "interval" in msg or "double_penalty" in msg


def test_scan_predict_point_and_interval_still_work():
    """The scan-aware predict path is unaffected: point predictions and
    posterior intervals both work and bracket the mean."""
    df = _dataset()
    model = _fit_scan(df, degree=5, penalty_order=3)
    # #2296: the scan posterior variance is conditional on the profiled
    # smoothing parameter, so the interval must be requested as conditional —
    # the (default) smoothing-corrected request is an honest typed refusal.
    out = model.predict(df, interval=0.9, covariance_mode="conditional")
    mean = np.asarray(out["mean"])
    lower = np.asarray(out["mean_lower"])
    upper = np.asarray(out["mean_upper"])
    assert np.all(np.isfinite(mean))
    assert np.all(lower <= mean + 1e-9)
    assert np.all(mean <= upper + 1e-9)
    assert np.all(upper >= lower)


def test_scan_predictions_intervals_and_summary_replay_exactly_after_save_load(tmp_path):
    """Persistence must replay the scan posterior, never reconstruct a dense fit.

    The query deliberately mixes both training-domain and extrapolation rows.
    Exact array equality is load-bearing: a dense refit can be numerically close
    while still being a different posterior, whereas JSON round-tripping the
    lossless ``SplineScanState`` must preserve every emitted float exactly.
    """
    df = _dataset()
    model = _fit_scan(df, degree=5, penalty_order=3)
    train_lo = float(df["x"].min())
    train_hi = float(df["x"].max())
    query = pd.DataFrame(
        {
            "x": [
                train_lo - 0.25,
                train_lo,
                0.5 * (train_lo + train_hi),
                train_hi,
                train_hi + 0.25,
            ]
        }
    )
    assert query["x"].iloc[0] < train_lo
    assert query["x"].iloc[-1] > train_hi

    point_before = model.predict(query, return_type="dict")
    bands_before = model.predict(
        query,
        interval=0.9,
        covariance_mode="conditional",
        observation_interval=True,
        return_type="dict",
    )
    for key in ("linear_predictor", "mean"):
        np.testing.assert_array_equal(point_before[key], bands_before[key])

    path = tmp_path / "scan_model.gam"
    model.save(str(path))
    wire = json.loads(path.read_text())
    payload = wire["payload"]
    assert payload["spline_scan"]["feature_column"] == "x"
    assert payload["fit_result"] is None
    assert payload["unified"] is None
    assert payload["resolved_termspec"] is None

    reloaded = gamfit.load(str(path))
    point_after = reloaded.predict(query, return_type="dict")
    bands_after = reloaded.predict(
        query,
        interval=0.9,
        covariance_mode="conditional",
        observation_interval=True,
        return_type="dict",
    )

    assert point_after.keys() == point_before.keys()
    for key in point_before:
        np.testing.assert_array_equal(point_after[key], point_before[key])

    expected_band_columns = {
        "linear_predictor",
        "mean",
        "std_error",
        "mean_lower",
        "mean_upper",
        "observation_lower",
        "observation_upper",
        # Result-owned provenance tag (#2296): interval dicts name the exact
        # covariance definition the band used.
        "covariance_source",
    }
    assert bands_after.keys() == bands_before.keys()
    assert set(bands_before) == expected_band_columns
    assert bands_before["covariance_source"] == "conditional"
    for key in bands_before:
        if key == "covariance_source":
            assert bands_after[key] == bands_before[key]
            continue
        np.testing.assert_array_equal(bands_after[key], bands_before[key])

    s0 = model.summary()
    s1 = reloaded.summary()
    assert s1.to_dict() == s0.to_dict()
    l0 = model.smoothing_parameters()
    l1 = reloaded.smoothing_parameters()
    assert l1 == l0
    assert reloaded.evidence == model.evidence
