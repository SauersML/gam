"""Bug hunt: ``Model.predict_array(X)`` (no ``interval``) returns a 2-D
``[linear_predictor, mean]`` matrix instead of the 1-D response-scale
prediction vector that ``Model.predict(...)`` returns and that its own
docstring promises.

``predict_array`` documents itself as the positional-array twin of ``predict``
— "``interval`` is the single uncertainty knob (issue #342); see
:meth:`predict` for its semantics."  And ``predict`` documents that, with no
``interval`` / ``id_column`` / ``return_type``, it returns "a 1-D ``ndarray`` of
point predictions on the response scale".  So ``predict_array(X)`` with no
``interval`` should likewise return a 1-D response-scale prediction vector.

Observed before this fix: ``predict_array(X)`` returned a 2-D array.  With no
interval its shape is ``(n, 2)`` whose **alphabetically-ordered** columns are
``[linear_predictor, mean]`` (with an interval it is ``(n, 5)`` —
``[linear_predictor, mean, mean_lower, mean_upper, std_error]``).  The root
cause is ``predict_array_impl`` returning ``columns_to_array(columns)`` over the
full prediction ``BTreeMap`` (``crates/gam-pyffi/src/geometry_ffi.rs:5733-5747``),
rather than the single response-scale ``mean`` column for the no-interval case.

Two concrete failures:

1. **Shape / API parity.** ``predict()`` returns ``(n,)``; ``predict_array()``
   returns ``(n, 2)``.  A drop-in array caller (and any ``np.asarray(pred) - y``
   style downstream) breaks on the unexpected second axis.

2. **Silent correctness trap on non-identity links.** Column 0 is
   ``linear_predictor`` — the **link-scale** ``η̂``, NOT the response-scale
   prediction.  A user who reaches for ``predict_array(X)`` as "the prediction"
   (e.g. ``predict_array(X).ravel()`` or ``[:, 0]``) silently gets log-rate /
   log-odds values for a Poisson / Bernoulli model.  For identity-link Gaussian
   the two columns are byte-identical, so the defect is invisible there and only
   bites once a link is involved.

This test asserts the documented contract: ``predict_array(X)`` with no interval
is 1-D and equals the response-scale ``mean`` from the named ``predict`` path.
It uses a Poisson (log-link) model so the response mean is unambiguously
distinct from the link-scale linear predictor.  When ``predict_array`` returns
the 1-D response prediction, the test passes without edits.
"""

import os

os.environ.setdefault("GAM_LOG", "off")
os.environ.setdefault("RUST_LOG", "off")

import numpy as np
import pytest

import gamfit


def _fit_poisson_array():
    rng = np.random.default_rng(0)
    n = 300
    X = rng.uniform(-2.0, 2.0, (n, 2))
    y = rng.poisson(np.exp(0.3 + 0.6 * np.sin(X[:, 0]))).astype(np.float64)
    model = gamfit.fit_array(X, y, "y ~ s(x0) + s(x1)", family="poisson")
    # Reference: the named predict path returns the 1-D response-scale mean.
    named_mean = np.asarray(
        model.predict({"x0": X[:, 0], "x1": X[:, 1], "y": y}), dtype=float
    )
    return model, X, named_mean


def test_predict_array_no_interval_is_one_dimensional_response_prediction():
    model, X, named_mean = _fit_poisson_array()
    out = np.asarray(model.predict_array(X), dtype=float)

    assert out.ndim == 1, (
        "predict_array(X) with no interval must return a 1-D response-scale "
        f"prediction vector (parity with predict()); got shape {out.shape}. "
        "It is returning the full [linear_predictor, mean] column matrix, whose "
        "first column is the link-scale linear predictor — a silent correctness "
        "trap for a naive array caller on non-identity links."
    )
    np.testing.assert_allclose(out, named_mean, rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
