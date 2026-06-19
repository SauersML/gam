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


def test_scan_summary_matches_default_double_penalty_route():
    """The explicit scan formula and default double-penalty formula for the
    same eligible smooth must agree on headline fitted quantities. This guards
    against the scan summary path fabricating numbers and against #1266
    regressing back to the inflated dense/sparse two-rho route."""
    df = _dataset()
    scan = _fit_scan(df, degree=3, penalty_order=2)
    default_fit = gamfit.fit(
        df, 'y ~ s(x, bs="ps", degree=3, penalty_order=2, double_penalty=True)'
    )

    edf_scan = float(scan.summary().edf_total)
    edf_default = float(default_fit.summary().edf_total)
    # Both formulas should land on the exact scan route for this eligible
    # single-smooth Gaussian problem.
    assert 2.0 < edf_scan < len(df)
    assert 2.0 < edf_default < len(df)
    assert abs(edf_scan - edf_default) < 1e-8

    yhat_scan = np.asarray(scan.predict(df))
    yhat_default = np.asarray(default_fit.predict(df))
    y = df["y"].to_numpy()
    rmse_scan = float(np.sqrt(np.mean((yhat_scan - y) ** 2)))
    rmse_default = float(np.sqrt(np.mean((yhat_default - y) ** 2)))
    assert abs(rmse_scan - rmse_default) < 1e-8


def test_scan_design_matrix_gives_actionable_error():
    """A scan model retains no dense design, so design_matrix() must fail with a
    precise, actionable message (not the cryptic missing-resolved_termspec one)."""
    df = _dataset()
    model = _fit_scan(df, degree=3, penalty_order=2)
    with pytest.raises(Exception) as exc:
        model.design_matrix(df)
    msg = str(exc.value).lower()
    assert "spline scan" in msg
    assert "double_penalty" in msg


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
    out = model.predict(df, interval=0.9)
    mean = np.asarray(out["mean"])
    lower = np.asarray(out["mean_lower"])
    upper = np.asarray(out["mean_upper"])
    assert np.all(np.isfinite(mean))
    assert np.all(lower <= mean + 1e-9)
    assert np.all(mean <= upper + 1e-9)
    assert np.all(upper >= lower)


def test_scan_summary_survives_save_load_roundtrip(tmp_path):
    """A persisted-then-reloaded scan model must summarize identically — the
    summary path reconstructs from the saved `SplineScanFit`, so it must work
    off a round-tripped payload, not just the freshly-fitted in-memory one."""
    df = _dataset()
    model = _fit_scan(df, degree=5, penalty_order=3)
    path = tmp_path / "scan_model.gam"
    model.save(str(path))
    reloaded = gamfit.load(str(path))

    s0 = model.summary()
    s1 = reloaded.summary()
    assert float(s1.reml_score) == pytest.approx(float(s0.reml_score), rel=1e-9, abs=1e-9)
    assert float(s1.edf_total) == pytest.approx(float(s0.edf_total), rel=1e-9, abs=1e-9)
    l0 = model.smoothing_parameters()
    l1 = reloaded.smoothing_parameters()
    assert l1.keys() == l0.keys()
    for key in l0:
        assert l1[key] == pytest.approx(l0[key], rel=1e-9, abs=1e-9)
