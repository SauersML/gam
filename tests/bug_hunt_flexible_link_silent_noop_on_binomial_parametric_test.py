"""Regression for #1596: binomial flexible(logit) must not be a silent no-op.

On a parametric, identifiable binomial predictor with a *misspecified* base link
(true link = cloglog, requested = flexible(logit)), the learnable link warp is
identifiable and should help. The defect: the coupled link-wiggle joint solve
failed KKT certification and a suppressed ``log::warn!`` fallback silently
returned the *no-wiggle baseline*, bit-identical to plain logit (same deviance,
no wiggle block). The user asked for a flexible link and silently got plain
logit.

The corrected contract: a flexible-link request must NEVER silently return the
fixed base link. It must either engage the warp (strictly lower deviance on this
grossly link-misspecified data) OR fail loudly with a clear error. This test
pins that contract; it fails on main (silent, deviance identical to logit).
"""

import os

os.environ.setdefault("GAM_LOG", "off")

import numpy as np
import pandas as pd

import gamfit


def _deviance(fit):
    return float(fit.summary().deviance)


def test_flexible_logit_not_silent_logit_noop_on_cloglog_data():
    rng = np.random.default_rng(3)
    n = 2000
    x = rng.uniform(-2.5, 2.5, n)
    eta = -0.5 + 1.8 * x
    # TRUE link = cloglog; logit is grossly misspecified.
    p = np.clip(1.0 - np.exp(-np.exp(eta)), 1e-4, 1.0 - 1e-4)
    y = (rng.uniform(size=n) < p).astype(float)
    df = pd.DataFrame({"x": x, "y": y})

    plain = gamfit.fit(df, "y ~ x", family="binomial", link="logit")
    dev_plain = _deviance(plain)

    try:
        flex = gamfit.fit(df, "y ~ x + link(type=flexible(logit))", family="binomial")
    except Exception as exc:  # noqa: BLE001 - a loud failure is acceptable
        msg = str(exc).lower()
        assert ("wiggle" in msg) or ("flexible" in msg) or ("converge" in msg), (
            "flexible(logit) failed (allowed), but the error must clearly name the "
            f"link-wiggle non-convergence; got: {exc!r}"
        )
        return

    # A *successful* flexible fit must genuinely improve on this misspecified
    # data — not be the silent plain-logit baseline returned under the hood.
    dev_flex = _deviance(flex)
    assert dev_flex < dev_plain - 1.0, (
        "flexible(logit) returned a successful fit that did not improve on plain logit "
        f"(dev_flex={dev_flex}, dev_plain={dev_plain}): the warp never engaged — the "
        "silent no-wiggle logit masquerade (#1596)."
    )
