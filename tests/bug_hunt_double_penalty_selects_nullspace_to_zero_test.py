"""Independent selection audit for the #1266 two-lambda B-spline double penalty.

The companion regression
``tests/bug_hunt_double_penalty_inflates_edf_instead_of_shrinking_test.py``
only asserts that the double penalty does **not inflate** EDF (``edf_on <= edf_off``).
That bound is *necessary* but not *sufficient*: it is satisfied even if
``double_penalty=True`` is a no-op, or a pure monotone wiggliness shrinkage that
never selects the constant/linear null space out of an unsupported term.

This module asserts the stronger property the fix actually claims (mgcv
``select = TRUE``): with a second, *identified* smoothing parameter on the
``Z Zᵀ`` null-space ridge, an unsupported smooth's effective degrees of freedom
collapse toward **zero**, *independently of wiggliness*, rather than resting on
the single-penalty floor (~1 for an order-2 difference penalty, whose null space
is the un-penalizable linear slope).

If any of these fail, the EDF-inflation test can be green while term selection
is silently broken — do not close #1266 on the inflation test alone.
"""

import numpy as np
import pandas as pd

import gamfit


# A single-penalty (``double_penalty=False``) order-2 B-spline leaves its linear
# null space un-penalized, so an *irrelevant* smooth keeps an EDF floor of about
# one slope. A working double penalty must shrink BELOW that floor — and a truly
# selected-out term lands near zero. These thresholds bracket "genuine selection"
# vs "merely not inflating".
_SELECTED_OUT_CEILING = 0.5  # double-penalty EDF for an unsupported term -> ~0
_SINGLE_PENALTY_FLOOR = 0.75  # single-penalty unsupported EDF stays >= ~1 slope
_MARGIN = 0.25  # double penalty must beat single by a real gap, not noise


def _smooth_edf(model, var):
    term = next(t for t in model.summary().smooth_terms if var in t["name"])
    return float(term["edf"])


def test_unsupported_smooth_is_selected_to_near_zero_edf():
    """A purely-irrelevant covariate's smooth collapses to EDF ~ 0 under the
    double penalty, and well below its single-penalty floor.

    This is the direct ``select = TRUE`` claim: the null space (constant + linear)
    of an unsupported term is driven out, not just prevented from inflating."""
    n = 1000
    dp_edf, sp_edf = [], []
    for seed in range(7):
        rng = np.random.default_rng(1000 + seed)
        x = rng.uniform(0.0, 1.0, n)
        z = rng.uniform(0.0, 1.0, n)  # z is genuinely irrelevant
        # Signal lives entirely in x; z must be selected out.
        y = np.sin(5.0 * x) + rng.normal(0.0, 0.3, n)
        df = pd.DataFrame({"x": x, "z": z, "y": y})

        m_dp = gamfit.fit(
            df,
            "y ~ s(x, k=15, bs=ps, double_penalty=True)"
            " + s(z, k=15, bs=ps, double_penalty=True)",
        )
        m_sp = gamfit.fit(
            df,
            "y ~ s(x, k=15, bs=ps, double_penalty=False)"
            " + s(z, k=15, bs=ps, double_penalty=False)",
        )
        dp_edf.append(_smooth_edf(m_dp, "z"))
        sp_edf.append(_smooth_edf(m_sp, "z"))

    dp_mean = float(np.mean(dp_edf))
    sp_mean = float(np.mean(sp_edf))

    # 1. The double penalty actually SELECTS: unsupported EDF -> ~0.
    assert dp_mean <= _SELECTED_OUT_CEILING, (
        f"double-penalty unsupported smooth EDF={dp_mean:.4f} did not collapse "
        f"to ~0; selection (select=TRUE) is broken, not just non-inflating. "
        f"per-seed dp={dp_edf}"
    )
    # 2. The single penalty CANNOT select (linear null space un-penalized): it
    #    rests on its floor, proving the gap is from the new null-space ridge and
    #    not from some unrelated over-shrinkage of both fits.
    assert sp_mean >= _SINGLE_PENALTY_FLOOR, (
        f"single-penalty unsupported smooth EDF={sp_mean:.4f} unexpectedly low; "
        f"thresholds need recalibration (per-seed sp={sp_edf})"
    )
    # 3. The improvement is a real gap, not numerical noise.
    assert sp_mean - dp_mean >= _MARGIN, (
        f"double penalty did not select the null space below the single-penalty "
        f"floor by a real margin: dp={dp_mean:.4f}, sp={sp_mean:.4f}"
    )


def test_irrelevant_smooth_selected_even_beside_a_strong_linear_signal():
    """The unsupported smooth is selected out (EDF ~ 0) even when a STRONG linear
    response on another covariate dominates the fit — i.e. the null-space ridge's
    lambda is identified per-term, not coupled to the model-wide signal scale.

    Here ``y`` is driven by a large linear trend in ``x`` (a separate parametric
    term, so ``x`` carries no smooth confound) plus a wiggle in ``x``; ``z`` is
    pure noise. A working second lambda must drive ``s(z)`` to EDF ~ 0 below its
    single-penalty linear floor, regardless of the strong ``x`` signal."""
    n = 1000
    dp_edf, sp_edf = [], []
    for seed in range(7):
        rng = np.random.default_rng(2000 + seed)
        x = rng.uniform(0.0, 1.0, n)
        z = rng.uniform(0.0, 1.0, n)  # genuinely irrelevant
        y = 10.0 * x + np.sin(6.0 * x) + rng.normal(0.0, 0.25, n)
        df = pd.DataFrame({"x": x, "z": z, "y": y})

        m_dp = gamfit.fit(
            df,
            "y ~ x + s(x, k=15, bs=ps, double_penalty=True)"
            " + s(z, k=15, bs=ps, double_penalty=True)",
        )
        m_sp = gamfit.fit(
            df,
            "y ~ x + s(x, k=15, bs=ps, double_penalty=False)"
            " + s(z, k=15, bs=ps, double_penalty=False)",
        )
        dp_edf.append(_smooth_edf(m_dp, "z"))
        sp_edf.append(_smooth_edf(m_sp, "z"))

    dp_mean = float(np.mean(dp_edf))
    sp_mean = float(np.mean(sp_edf))

    assert dp_mean <= _SELECTED_OUT_CEILING, (
        f"double-penalty irrelevant smooth EDF={dp_mean:.4f} did not collapse to "
        f"~0 beside a strong linear signal; per-term null-space lambda not "
        f"identified. per-seed dp={dp_edf}"
    )
    assert sp_mean - dp_mean >= _MARGIN, (
        f"double penalty did not select below the single-penalty floor: "
        f"dp={dp_mean:.4f}, sp={sp_mean:.4f}"
    )


def test_supported_wiggliness_is_retained_not_over_shrunk():
    """Guard against the cheap way to pass the two tests above: blanket
    over-shrinkage. A genuinely wiggly signal must KEEP meaningful EDF under the
    double penalty (selection retains supported structure, it does not kill it)."""
    n = 1000
    dp_edf = []
    for seed in range(7):
        rng = np.random.default_rng(3000 + seed)
        x = rng.uniform(0.0, 1.0, n)
        y = np.sin(8.0 * x) + rng.normal(0.0, 0.2, n)  # clearly wiggly support
        df = pd.DataFrame({"x": x, "y": y})
        m_dp = gamfit.fit(df, "y ~ s(x, k=20, bs=ps, double_penalty=True)")
        dp_edf.append(_smooth_edf(m_dp, "x"))

    dp_mean = float(np.mean(dp_edf))
    # sin(8x) over [0,1] has ~1.3 cycles: a working fit must spend several EDF.
    assert dp_mean >= 3.0, (
        f"double penalty over-shrank a genuinely wiggly term to EDF={dp_mean:.4f}; "
        f"this would let the selection tests pass via blanket shrinkage rather "
        f"than real null-space selection. per-seed={dp_edf}"
    )
