"""Response-scale invariance of the smooth-term Wald test (issue #675).

For estimated-scale families (Gaussian/Gamma) the per-smooth Wald p-value
reported by ``Model.summary()`` must not depend on the *units* of the
response.  Multiplying every ``y`` by a positive constant is a pure
relabelling that changes nothing about the fit's geometry, so the
chi-square statistic, the F-statistic it implies, and the resulting
p-value must all be invariant.

The defect (now fixed): the estimated-scale branch of ``wood_smooth_test``
divided the Wald statistic by the dispersion ``phi`` a *second* time, even
though the posterior covariance that formed the statistic already had
``phi`` baked in.  That made the F-statistic scale as ``1 / phi`` (i.e.
``1 / c**2`` under ``y -> c * y``), pushing a strongly significant smooth
toward ``p ~ 0.63`` at ``c = 100``.

``chi_sq`` was already empirically scale-invariant, which is what isolates
the defect to the p-value transform rather than the statistic itself.
"""

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

import gamfit


def _fit_smooth(scale_y):
    """Fit ``y ~ s(x)`` on a fixed sinusoid scaled by ``scale_y`` and return
    the single smooth-term summary row."""
    rng = np.random.default_rng(1)
    n = 400
    x = np.sort(rng.uniform(-3.0, 3.0, n))
    y = scale_y * (np.sin(1.5 * x) + 0.4 * rng.standard_normal(n))
    fr = pd.DataFrame({"x": x, "y": y})
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        m = gamfit.fit(fr, "y ~ s(x)", family="gaussian")
        s = m.summary()
    return s.smooth_terms[0]


SCALES = (1.0, 10.0, 100.0, 1000.0)


def test_smooth_chi_square_is_scale_invariant():
    """Anchor: the Wald chi-square statistic is already scale-invariant.

    This passed before the fix too; it isolates the defect to the p-value
    transform.  If this ever regresses, the covariance scaling itself broke.
    """
    chi = [float(_fit_smooth(c)["chi_sq"]) for c in SCALES]
    ref = chi[0]
    assert ref > 1.0, f"degenerate anchor chi_sq={ref}"
    for c, value in zip(SCALES, chi):
        rel = abs(value - ref) / ref
        assert rel < 1e-6, f"chi_sq not scale-invariant at c={c}: {value} vs {ref}"


def test_smooth_pvalue_is_invariant_to_response_units():
    """The p-value must agree across every response scaling."""
    pvals = [float(_fit_smooth(c)["p_value"]) for c in SCALES]
    ref = pvals[0]
    # All p-values share the same fit geometry, so they must be equal to
    # within solver tolerance.  Compare on the log scale where the values
    # are astronomically small; an additive epsilon guards log(0).
    eps = 1e-300
    log_ref = np.log10(ref + eps)
    for c, p in zip(SCALES, pvals):
        log_p = np.log10(p + eps)
        assert abs(log_p - log_ref) < 1.0, (
            f"p-value not scale-invariant at c={c}: {p:.4g} vs {ref:.4g}"
        )


def test_strong_smooth_stays_significant_at_every_scale():
    """A smooth that is overwhelmingly significant at unit scale must stay
    overwhelmingly significant at every other scale."""
    for c in SCALES:
        p = float(_fit_smooth(c)["p_value"])
        assert p < 1e-3, f"strong smooth lost significance at c={c}: p={p:.4g}"


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    for c in SCALES:
        t = _fit_smooth(c)
        print(f"c={c:8}: chi_sq={t['chi_sq']:.3f}  p_value={t['p_value']:.4g}")
    raise SystemExit(pytest.main([__file__, "-v"]))
