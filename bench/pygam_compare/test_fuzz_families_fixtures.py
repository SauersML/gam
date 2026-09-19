"""Fixtures from the family/link convergence fuzzer.

Each cell below failed before its root-cause fix and must now be clean under
:func:`failure_cause`:

* gamma(inverse): the dispersion profile evaluated the shape and phi at
  ``exp(eta)`` whatever the link, so on the inverse link the fitted scale sat
  hundreds of standard errors off the truth.
* binomial-trials: the fully-normalized log-likelihood required ``fl(w * y)``
  to be an exact integer, which most stored proportions ``k / w`` miss by an
  ulp (``439 / 1000 * 1000 = 439.00000000000006``), so every
  almost-deterministic binomial refused. main now evaluates the continuous
  normalizer ``ln C(w, w y)`` for any finite positive weight; the cells stay
  as its regression fixtures.
* inverse-gaussian (canonical link): the smoothing-parameter seed read the
  prior weights, not the Fisher working weights ``mu^3 / 4``, so it was not
  equivariant under a change of response units and, in small units, landed on
  the ``lambda -> infinity`` plateau, where the REML gradient vanishes and the
  intercept-only fit certified.

A failing cell reruns in isolation with
``python worker.py gamfit FAMILY N DESIGN SEED``.
"""

from __future__ import annotations

import gamfit
import numpy as np
import pytest

from .fuzz_families import CASE_BY_LABEL, FORMULA, draw, failure_cause, fuzz_design, run

FIXTURES: tuple[tuple[str, int, str, int], ...] = (
    ("gamma(inverse)", 500, "base", 0),
    ("binomial-trials(logit)", 50, "lowdisp", 0),
    ("binomial-trials(cloglog)", 500, "lowdisp", 0),
    ("inverse-gaussian", 500, "range", 0),
    ("inverse-gaussian", 50, "edge", 0),
)


@pytest.mark.parametrize(("label", "n", "regime", "seed"), FIXTURES)
def test_fuzz_fixture_is_clean(label: str, n: int, regime: str, seed: int) -> None:
    design = fuzz_design(regime)
    record = run(label, n, design, seed) | {
        "family": label,
        "n": n,
        "design": design,
        "seed": seed,
    }
    assert failure_cause(record) is None, record.get("error_head")


def test_inverse_gaussian_canonical_fit_is_unit_equivariant() -> None:
    """``y -> c y`` rescales the canonical inverse-Gaussian fit exactly: the
    same smooths (same edf), predictions times ``c`` and scale over ``c``."""
    case = CASE_BY_LABEL["inverse-gaussian"]
    data = draw(case.label, 500, "range", 0)
    fits = {}
    for c in (1.0, 1000.0):
        train = dict(data.train) | {"y": c * data.train["y"]}
        model = gamfit.fit(train, FORMULA, family=case.family)
        summary = model.summary()
        assert summary.convergence["certified"]
        pred = np.asarray(model.predict(data.test), dtype=float).reshape(-1)
        fits[c] = (summary.edf_total, pred / c, summary.scale * c)
    (edf1, pred1, scale1), (edf2, pred2, scale2) = fits[1.0], fits[1000.0]
    np.testing.assert_allclose(edf1, edf2, rtol=1e-4)
    np.testing.assert_allclose(pred1, pred2, rtol=1e-4)
    np.testing.assert_allclose(scale1, scale2, rtol=1e-4)
    assert edf1 > 2.0
