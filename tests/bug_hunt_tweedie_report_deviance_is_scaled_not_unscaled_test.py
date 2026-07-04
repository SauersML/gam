"""Bug hunt: the reported Tweedie deviance is the SCALED deviance ``D / φ̂``,
not the conventional unscaled deviance ``D = 2·Σ wᵢ·d(yᵢ, μ̂ᵢ)``.

Every GLM family reports the *unscaled* deviance ``D = 2·Σ wᵢ·d(yᵢ, μ̂ᵢ)``
(Poisson, Binomial, NB, Beta, and — since #2126 — Gamma), matching
``R`` (``glm``/``tweedie``), ``mgcv``, and ``statsmodels``. The dispersion is a
separate reported quantity; the deviance itself must be scale-free so that
``deviance_explained = 1 − D_resid / D_null`` is a pure ratio of like-scaled
deviances.

For a Tweedie fit it is not. ``gam report`` prints the deviance divided by the
fitted dispersion ``φ̂`` — i.e. the *scaled* deviance ``D / φ̂`` — which for a
sub-unit ``φ̂`` inflates the reported number by ``1/φ̂``. On a
Tweedie(``p = 1.5``, ``φ = 0.8``) compound Poisson-Gamma sample the reported
deviance is ``≈ 578.8`` where the true unscaled deviance (recomputed here from
the fitted means with the exact Tweedie unit deviance) is ``≈ 447.2`` — a factor
of ``1/φ̂ ≈ 1/0.77`` too large.

Root cause (read, not patched): ``calculate_deviance`` in
``crates/gam-solve/src/pirls/deviance.rs`` accumulates, for the Tweedie arm
(lines ~184-202), ``priorweights[i] * tweedie_unit_deviance(yᵢ, μᵢ, p) / phi``
— it divides each unit deviance by ``φ``. The adjacent Gamma arm (lines
~229-248) was explicitly de-scaled for #2126 and now accumulates the *bare*
``priorweights · unit_deviance`` with ``φ ≡ 1``; its own comment states that
"Multiplying the unit deviance by the fitted shape (≈ 1/φ̂) would report the
scaled deviance D/φ̂ — the #2126 defect." Tweedie is the lone family still
dividing by ``φ``, so it reports the exact defect #2126 fixed for Gamma. The
fix is to drop the ``/ phi`` at that line, matching every other family.

The test fits a Tweedie smooth, reads the reported deviance from ``gam report``,
independently recomputes the unscaled Tweedie deviance from the fitted means, and
asserts the report matches the *unscaled* value (and is NOT the ``φ̂``-scaled
one). It currently FAILS (report ≈ D/φ̂) and will PASS once the ``/ phi`` is
removed.

Related: #2126 (the identical defect, fixed for Gamma), #2105 (Tweedie
dispersion φ̂ magnitude). This is the Tweedie sibling of #2126.

The test drives the ``gam`` CLI (on $PATH); the Python ``gamfit`` wheel is not
required.
"""

import csv
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

GAM = shutil.which("gam")

TWEEDIE_P = 1.5  # the engine's Tweedie variance power (see #2026)
TRUE_PHI = 0.8


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(list(r))


def _tweedie_unit_deviance(y, mu, p):
    """Exact Tweedie unit deviance d(y, mu), matching
    crates/gam-solve/src/pirls/deviance.rs::tweedie_unit_deviance."""
    y = np.asarray(y, float)
    mu = np.maximum(np.asarray(mu, float), 1e-10)
    zero = y == 0.0
    d = np.empty_like(y)
    d[zero] = mu[zero] ** (2 - p) / (2 - p)
    yp = np.maximum(y[~zero], 0.0)
    mp = mu[~zero]
    d[~zero] = (
        yp ** (2 - p) / ((1 - p) * (2 - p))
        - yp * mp ** (1 - p) / (1 - p)
        + mp ** (2 - p) / (2 - p)
    )
    return d


def _report_deviance(model_path):
    # `gam report` writes `<model_stem>.report.html` relative to the working
    # directory, so run it in the model's own directory to keep the artifact
    # inside the temp tree.
    r = subprocess.run(
        [GAM, "report", model_path.name],
        cwd=str(model_path.parent),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"report failed: {r.stderr}"
    html = (model_path.parent / f"{model_path.stem}.report.html").read_text()
    m = re.search(
        r"Deviance</span><span class=\"stat-value\">(-?\d+\.?\d*(?:e-?\d+)?)</span>",
        html,
    )
    assert m is not None, "could not find Deviance in report HTML"
    return float(m.group(1))


@pytest.mark.skipif(GAM is None, reason="gam CLI not on PATH")
def test_tweedie_report_deviance_is_unscaled():
    rng = np.random.default_rng(2)
    n = 500
    p, phi = TWEEDIE_P, TRUE_PHI
    x = rng.uniform(0.0, 1.0, n)
    mu = np.exp(0.5 + 1.0 * np.sin(2 * np.pi * x))
    # Tweedie compound Poisson-Gamma sampler.
    lam = mu ** (2 - p) / (phi * (2 - p))
    a = (2 - p) / (p - 1)
    scale = phi * (p - 1) * mu ** (p - 1)
    counts = rng.poisson(lam)
    y = np.array(
        [rng.gamma(a * c, scale[i]) if c > 0 else 0.0 for i, c in enumerate(counts)]
    )

    with tempfile.TemporaryDirectory() as d:
        wd = Path(d)
        train = wd / "tw.csv"
        _write(train, ["x", "y"], [(x[i], y[i]) for i in range(n)])
        model = wd / "m_tw.gam"
        fit = subprocess.run(
            [GAM, "fit", str(train), "y ~ smooth(x)", "--family", "tweedie",
             "--out", str(model)],
            capture_output=True, text=True,
        )
        assert fit.returncode == 0, f"fit failed: {fit.stderr}"

        reported = _report_deviance(model)

        pred = wd / "fit.csv"
        pr = subprocess.run(
            [GAM, "predict", str(model), str(train), "--out", str(pred)],
            capture_output=True, text=True,
        )
        assert pr.returncode == 0, f"predict failed: {pr.stderr}"
        mu_hat = np.array([float(r["mean"]) for r in csv.DictReader(open(pred))])

        # Fitted dispersion φ̂ recorded in the serialized model.
        phi_match = re.search(r'"phi":(\d+\.\d+(?:e-?\d+)?)', model.read_bytes().decode("latin-1"))
        assert phi_match is not None, "could not read fitted Tweedie phi from model"
        phi_hat = float(phi_match.group(1))

    # Conventional unscaled deviance D = 2·Σ wᵢ·d(yᵢ, μ̂ᵢ), w ≡ 1.
    unscaled = 2.0 * float(np.sum(_tweedie_unit_deviance(y, mu_hat, p)))
    scaled = unscaled / phi_hat

    assert phi_hat < 0.9, (
        f"control: fitted Tweedie φ̂ = {phi_hat:.4f} must be clearly < 1 so the "
        f"scaled and unscaled deviances are distinguishable."
    )

    # The reported deviance must be the UNSCALED D (like every other family and
    # R/mgcv), not the φ̂-scaled D/φ̂. (The ~1% slack absorbs the difference
    # between the report's internal fitted μ and the bias-corrected predict μ
    # used to recompute D here; it is far below the 1/φ̂ ≈ 1.3 scale gap.)
    rel_to_unscaled = abs(reported - unscaled) / unscaled
    rel_to_scaled = abs(reported - scaled) / scaled
    assert rel_to_unscaled < 0.08, (
        f"Tweedie reported deviance {reported:.4f} is the SCALED deviance D/φ̂, "
        f"not the unscaled D = 2·Σ wᵢ·d(yᵢ, μ̂ᵢ) = {unscaled:.4f} that every other "
        f"family reports. It matches D/φ̂ = {scaled:.4f} (φ̂ = {phi_hat:.4f}) to "
        f"{rel_to_scaled:.1%}, i.e. it is inflated by ~1/φ̂. `calculate_deviance` "
        f"divides the Tweedie unit deviance by φ (deviance.rs ~line 199); the "
        f"Gamma arm was de-scaled for #2126 but Tweedie was left scaled."
    )
