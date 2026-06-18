"""Bug hunt: `link(type=flexible(...))` is a silent no-op on non-binomial families.

`docs/families-and-links.md` documents `flexible(base)` as a link that "adds a
jointly fit anchored spline offset to a base link" and lists `identity`, `log`,
`logit`, `probit`, `cloglog` as accepted base links. The formula parser accepts
`flexible(identity)` / `flexible(log)` without complaint.

But the fit pipeline gates the link wiggle to binomial families only
(`src/solver/fit_orchestration/materialize/standard.rs:191` returns `None` for
`!family.is_binomial()`; the location-scale path does the same at
`materialize/location_scale.rs:102`). So on Gaussian / Poisson / Gamma the
"flexible" offset is silently discarded and the fit is bit-identical to the
plain base link. The two documented bases that are *inherently* non-binomial —
`identity` (Gaussian) and `log` (Poisson/Gamma) — can therefore never do
anything at all: they parse, fit, and produce exactly the base-link model.

A jointly-fit anchored spline offset nests the base link (offset = 0 recovers
it), so on data whose true mean structure is misspecified for the base link the
flexible fit must achieve a strictly lower penalized deviance and visibly
different predictions. This test pins that: a flexible(log) Poisson fit on
log-misspecified data must differ from the plain log fit.
"""

import os

os.environ.setdefault("GAM_LOG", "off")

import numpy as np
import pandas as pd

import gamfit


def test_flexible_log_link_is_not_a_silent_noop_on_poisson():
    # A rigid parametric linear predictor (y ~ x) means the link offset is the
    # ONLY available source of curvature: log(mu) is deliberately non-linear in
    # x (the tanh bump), so a working flexible(log) offset must bend the fit and
    # cut the deviance. With the wiggle silently dropped the two fits coincide.
    n = 2000
    rng = np.random.default_rng(7)
    x = np.linspace(-1.5, 1.5, n)
    mu_true = np.exp(0.5 + 0.8 * x + 0.6 * np.tanh(3.0 * x))
    y = rng.poisson(mu_true).astype(float)
    df = pd.DataFrame({"y": y, "x": x})

    m_log = gamfit.fit(df, "y ~ x", family="poisson", link="log")
    m_flex = gamfit.fit(df, "y ~ x + link(type=flexible(log))", family="poisson")

    p_log = np.asarray(m_log.predict(df), dtype=float).ravel()
    p_flex = np.asarray(m_flex.predict(df), dtype=float).ravel()

    max_pred_diff = float(np.max(np.abs(p_log - p_flex)))
    dev_log = float(m_log.summary().deviance)
    dev_flex = float(m_flex.summary().deviance)

    # The documented flexible offset must actually flex the fit.
    assert max_pred_diff > 1e-6, (
        "flexible(log) Poisson fit is bit-identical to the plain log fit "
        f"(max|pred diff|={max_pred_diff:.3e}): the documented anchored-spline "
        "link offset is being silently discarded for non-binomial families"
    )
    # Nesting => the flexible fit cannot be worse, and on log-misspecified data
    # it must be strictly better.
    assert dev_flex <= dev_log - 0.5, (
        "flexible(log) did not improve the deviance over plain log "
        f"(dev_log={dev_log:.4f}, dev_flex={dev_flex:.4f}); the jointly-fit "
        "spline offset is inert on this non-binomial family"
    )


def test_flexible_identity_link_is_not_a_silent_noop_on_gaussian():
    # Same story for the Gaussian/identity base: a curvature-rich mean fit with a
    # rigid linear predictor should be bent by the flexible(identity) offset.
    n = 2000
    rng = np.random.default_rng(13)
    x = np.linspace(-1.5, 1.5, n)
    mu_true = 1.0 + 0.8 * x + 0.7 * np.tanh(3.0 * x)
    y = mu_true + rng.normal(0.0, 0.3, n)
    df = pd.DataFrame({"y": y, "x": x})

    m_id = gamfit.fit(df, "y ~ x", family="gaussian", link="identity")
    m_flex = gamfit.fit(df, "y ~ x + link(type=flexible(identity))", family="gaussian")

    p_id = np.asarray(m_id.predict(df), dtype=float).ravel()
    p_flex = np.asarray(m_flex.predict(df), dtype=float).ravel()
    max_pred_diff = float(np.max(np.abs(p_id - p_flex)))

    assert max_pred_diff > 1e-6, (
        "flexible(identity) Gaussian fit is bit-identical to the plain identity "
        f"fit (max|pred diff|={max_pred_diff:.3e}): the documented flexible link "
        "offset is silently discarded for non-binomial families"
    )
