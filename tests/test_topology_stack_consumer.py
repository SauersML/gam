"""Unit tests for the topology stacking consumer (#768).

These exercise the Python-side held-out density-table assembly, the binding
contract, and the stacked-mean combination with stubbed candidate fits and a
stubbed Rust binding, so they run without the compiled extension.
"""

import json
import math

from statistics import NormalDist

import gamfit._select_topology as st


class _StubFit:
    """Candidate fit whose predict() returns deterministic per-point moments.

    `mean_fn(x)` gives the response-scale mean; the observation band is the
    symmetric Gaussian band mean ± z·sd used by the real Rust predictor, so the
    consumer recovers exactly `sd` back out.
    """

    def __init__(self, mean_fn, sd):
        self._mean_fn = mean_fn
        self._sd = sd

    def predict(self, data, **kwargs):
        xs = data["x"]
        means = [self._mean_fn(x) for x in xs]
        if kwargs.get("observation_interval"):
            z = NormalDist().inv_cdf(0.5 + 0.5 * kwargs["interval"])
            lower = [m - z * self._sd for m in means]
            upper = [m + z * self._sd for m in means]
            return {
                "mean": means,
                "observation_lower": lower,
                "observation_upper": upper,
            }
        return {"mean": means}


class _CapturingRust:
    """Stub binding that records the log-density table and returns fixed weights."""

    def __init__(self, weights):
        self._weights = weights
        self.captured_names = None
        self.captured_rows = None

    def stacking_weights_from_log_density(self, names, log_density_rows):
        self.captured_names = list(names)
        self.captured_rows = [list(row) for row in log_density_rows]
        return json.dumps(
            {
                "weights": dict(self._weights),
                "mean_log_score": -1.234,
                "iterations": 7,
            }
        )


def _gaussian_logpdf(y, mean, sd):
    z = (y - mean) / sd
    return -0.5 * math.log(2.0 * math.pi) - math.log(sd) - 0.5 * z * z


def test_holdout_log_density_table_matches_gaussian(monkeypatch):
    holdout = {"x": [0.0, 1.0, 2.0], "y": [0.1, 0.9, 2.2]}
    fits = {
        "flat": _StubFit(lambda x: 0.0, sd=1.0),
        "linear": _StubFit(lambda x: x, sd=0.5),
    }
    rust = _CapturingRust({"flat": 0.3, "linear": 0.7})
    monkeypatch.setattr(st, "_topology_rust", lambda: rust)

    stack = st.stack_topologies(fits, holdout, "y")

    # The table the binding received is the per-point Gaussian held-out
    # log-density of the true y under each candidate's recovered (mean, sd).
    assert rust.captured_names == ["flat", "linear"]
    y = holdout["y"]
    for i, yi in enumerate(y):
        expected_flat = _gaussian_logpdf(yi, 0.0, 1.0)
        expected_linear = _gaussian_logpdf(yi, float(i), 0.5)
        assert math.isclose(rust.captured_rows[i][0], expected_flat, rel_tol=1e-9)
        assert math.isclose(rust.captured_rows[i][1], expected_linear, rel_tol=1e-9)

    assert stack.weights == {"flat": 0.3, "linear": 0.7}
    assert stack.mean_log_score == -1.234


def test_stacked_predict_is_weighted_mixture(monkeypatch):
    holdout = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
    fits = {
        "flat": _StubFit(lambda x: 2.0, sd=1.0),
        "linear": _StubFit(lambda x: 10.0 * x, sd=1.0),
    }
    rust = _CapturingRust({"flat": 0.25, "linear": 0.75})
    monkeypatch.setattr(st, "_topology_rust", lambda: rust)

    stack = st.stack_topologies(fits, holdout, "y")
    out = stack.predict({"x": [0.0, 1.0, 2.0]})

    # flat predicts 2 everywhere; linear predicts 10*x. Mixture = 0.25*2 + 0.75*10*x.
    assert math.isclose(out[0], 0.25 * 2.0 + 0.75 * 0.0, rel_tol=1e-9)
    assert math.isclose(out[1], 0.25 * 2.0 + 0.75 * 10.0, rel_tol=1e-9)
    assert math.isclose(out[2], 0.25 * 2.0 + 0.75 * 20.0, rel_tol=1e-9)


def test_zero_weighted_candidate_is_not_predicted(monkeypatch):
    holdout = {"x": [0.0, 1.0], "y": [0.0, 1.0]}

    class _Exploding(_StubFit):
        def predict(self, data, **kwargs):
            if not kwargs.get("observation_interval"):
                raise AssertionError("zero-weighted candidate must not be predicted")
            return super().predict(data, **kwargs)

    fits = {
        "keep": _StubFit(lambda x: x, sd=1.0),
        "drop": _Exploding(lambda x: 99.0, sd=1.0),
    }
    rust = _CapturingRust({"keep": 1.0, "drop": 0.0})
    monkeypatch.setattr(st, "_topology_rust", lambda: rust)

    stack = st.stack_topologies(fits, holdout, "y")
    out = stack.predict({"x": [3.0, 4.0]})
    assert math.isclose(out[0], 3.0, rel_tol=1e-9)
    assert math.isclose(out[1], 4.0, rel_tol=1e-9)


def test_non_positive_sd_rows_are_dropped_from_a_candidate(monkeypatch):
    # A candidate whose observation band collapses (sd == 0, e.g. fully clamped
    # at the support) yields -inf log-density for those rows, not a crash.
    holdout = {"x": [0.0, 1.0], "y": [0.0, 1.0]}

    class _Degenerate(_StubFit):
        def predict(self, data, **kwargs):
            out = super().predict(data, **kwargs)
            if kwargs.get("observation_interval"):
                # Collapse the band on the first row.
                out["observation_lower"][0] = out["mean"][0]
                out["observation_upper"][0] = out["mean"][0]
            return out

    fits = {
        "good": _StubFit(lambda x: x, sd=1.0),
        "degen": _Degenerate(lambda x: x, sd=1.0),
    }
    rust = _CapturingRust({"good": 0.5, "degen": 0.5})
    monkeypatch.setattr(st, "_topology_rust", lambda: rust)

    st.stack_topologies(fits, holdout, "y")
    # First row's degenerate column is -inf; the good column stays finite.
    assert rust.captured_rows[0][1] == float("-inf")
    assert math.isfinite(rust.captured_rows[0][0])


def test_missing_response_column_is_rejected():
    fits = {"a": _StubFit(lambda x: x, sd=1.0)}
    try:
        st.stack_topologies(fits, {"x": [0.0]}, "y")
    except ValueError as exc:
        assert "response" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing response column")
