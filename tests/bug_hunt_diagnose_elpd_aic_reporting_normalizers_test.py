"""Bug hunt (#1581, #1582, #1583): ``gam diagnose`` must report AIC / PSIS-LOO
``elpd`` as fully-normalized, scale-aware, cross-family-comparable quantities.

These are three symptoms of one root cause in
``crates/gam-solve/src/pirls/deviance.rs``: the per-row log-likelihood kernel
that backs the user-facing ``log_likelihood`` (→ conditional AIC) and the
PSIS-LOO pointwise predictive densities (→ ``elpd``) dropped family-specific
normalizing constants and, for the profiled Gaussian, ignored the estimated
residual variance (using ``phi = 1``). That kernel is correct for its OTHER
consumer — the REML/LAML outer objective, where the dropped constants cancel and
are needed to avoid Gamma overflow — but wrong when the value is reported as an
absolute number.

Symptoms asserted here, exactly as in the issue repros:

* #1581 Poisson: a discrete model's ``elpd`` is a sum of log probability MASSES,
  so ``elpd <= 0`` and ``AIC = -2*logLik + 2*edf > 0``. Pre-fix the dropped
  ``-ln(y!)`` flipped both signs (elpd ``+291``, AIC ``-582``).
* #1582 Poisson vs NB(theta=1e5): the same model in the ``theta -> inf`` limit,
  so their AIC and elpd must agree to well under one nat. Pre-fix the Poisson
  arm dropped ``-ln(y!)`` while NB kept its normalizer, so they differed by
  ``~1750`` nats.
* #1583 Gaussian: a continuous predictive density obeys the change-of-variables
  law, so under ``y -> c*y`` the elpd must SHIFT by ``-n*ln(c)`` (it is scale
  equivariant), not scale by ``c^2`` as the old ``-0.5*RSS`` did.

Driven through the ``gam`` CLI, which is the buildable front-end whose
``model_comparison_from_unified`` assembles exactly these reported channels.
"""

from __future__ import annotations

import csv
import math
import os
import re
import subprocess
import tempfile

import numpy as np
import pytest

GAM_BIN = os.path.join(os.path.dirname(__file__), "..", "target", "release", "gam")

pytestmark = pytest.mark.skipif(
    not os.path.exists(GAM_BIN),
    reason="release `gam` CLI binary not built (target/release/gam absent)",
)


def _write_csv(path: str, cols: dict[str, np.ndarray]) -> None:
    keys = list(cols)
    n = len(cols[keys[0]])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            w.writerow([cols[k][i] for k in keys])


def _run(*args: str) -> str:
    r = subprocess.run([GAM_BIN, *args], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"gam {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout


def _diagnose_metrics(model_json: str, data_csv: str) -> dict[str, float]:
    """Parse AIC (conditional) and PSIS-LOO elpd from `gam diagnose` output."""
    out = _run("diagnose", model_json, data_csv)
    metrics: dict[str, float] = {}
    for line in out.splitlines():
        if "AIC (conditional)" in line:
            m = re.search(r"(-?\d+\.\d+)", line)
            if m:
                metrics["aic"] = float(m.group(1))
        elif "elpd" in line.lower() and "PSIS" in line:
            m = re.search(r"(-?\d+\.\d+)", line)
            if m:
                metrics["elpd"] = float(m.group(1))
    return metrics


def test_poisson_elpd_is_nonpositive_and_aic_positive(tmp_path):
    # #1581
    rng = np.random.default_rng(7)
    n = 300
    x = rng.uniform(0, 1, n)
    mu = np.exp(0.6 + 1.2 * x)
    y = rng.poisson(mu)
    data = str(tmp_path / "c.csv")
    _write_csv(data, {"x": x, "y": y.astype(int)})
    model = str(tmp_path / "cp.json")
    _run("fit", data, "y ~ s(x)", "--family", "poisson-log", "--out", model)
    met = _diagnose_metrics(model, data)
    assert "elpd" in met and "aic" in met, met
    # A discrete model's elpd is a sum of log-masses ≤ 0; AIC = -2logLik+2edf > 0.
    assert met["elpd"] <= 0.0, f"Poisson PSIS-LOO elpd must be ≤ 0, got {met['elpd']}"
    assert met["aic"] > 0.0, f"Poisson conditional AIC must be > 0, got {met['aic']}"


def test_poisson_and_negbin_large_theta_are_comparable(tmp_path):
    # #1582
    rng = np.random.default_rng(7)
    n = 300
    x = rng.uniform(0, 1, n)
    mu = np.exp(0.6 + 1.2 * x)
    y = rng.poisson(mu)
    data = str(tmp_path / "c.csv")
    _write_csv(data, {"x": x, "y": y.astype(int)})

    mp = str(tmp_path / "cp.json")
    mnb = str(tmp_path / "cnb.json")
    _run("fit", data, "y ~ s(x)", "--family", "poisson-log", "--out", mp)
    _run("fit", data, "y ~ s(x)", "--family", "negative-binomial",
         "--negative-binomial-theta", "100000", "--out", mnb)
    p = _diagnose_metrics(mp, data)
    nb = _diagnose_metrics(mnb, data)
    # NB(theta=1e5) is the Poisson limit; AIC and elpd must agree to < ~1 nat
    # (the residual is O(n/theta)). Pre-fix they differed by ~1750 / ~876.
    assert abs(p["aic"] - nb["aic"]) < 2.0, f"|ΔAIC|={abs(p['aic']-nb['aic'])}"
    assert abs(p["elpd"] - nb["elpd"]) < 2.0, f"|Δelpd|={abs(p['elpd']-nb['elpd'])}"


def test_gaussian_elpd_obeys_change_of_variables(tmp_path):
    # #1583
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0, 1, n)
    y = 2 + 3 * x + rng.normal(0, 0.2, n)

    def elpd_for_scale(c: float) -> float:
        data = str(tmp_path / f"g{c}.csv")
        _write_csv(data, {"x": x, "y": y * c})
        model = str(tmp_path / f"mg{c}.json")
        _run("fit", data, "y ~ x", "--out", model)
        return _diagnose_metrics(model, data)["elpd"]

    base = elpd_for_scale(1.0)
    for c in (0.5, 2.0, 10.0):
        shifted = elpd_for_scale(c)
        # Continuous density ⇒ f_{cY}(cy) = f_Y(y)/c ⇒ each log-density shifts by
        # −ln c, so the summed elpd shifts by −n·ln c — NOT c²·base.
        want = base - n * math.log(c)
        assert abs(shifted - want) < 1.0, (
            f"c={c}: elpd shift must be −n·ln c (={want:.2f}), got {shifted:.2f}; "
            f"c²·base would be {c * c * base:.2f}"
        )
