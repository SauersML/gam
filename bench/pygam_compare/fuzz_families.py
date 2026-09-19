"""Convergence fuzz over response families and links, with data at the edges.

A fuzz cell is ``(FAMILY, n, ff-<regime>)``. ``FAMILY`` is one family/link
label from :data:`CASES`: every response family in gamfit's family registry
(gamma, inverse Gaussian, negative binomial, Tweedie, beta, scaled t, Poisson,
binomial with trials, Gaussian with its non-identity link), each with every
link its legality table admits. gamfit has no quasi families. The regime
picks where the data sit relative to the edges of that family's support:

  base      moderate signal, moderate dispersion
  edge      responses at or near the boundary of the support (tiny positive
            responses, proportions within 1e-4 of 0 or 1, counts that are
            mostly zero, Cauchy-tailed t noise)
  range     a mean that spans many orders of magnitude across the covariates
  zeros     a region of the covariate space with no events at all (counts,
            Tweedie, binomial trials); for continuous families the region
            instead carries the smallest representable-scale responses
  lowdisp   near-degenerate dispersion: an almost deterministic response
  highdisp  the opposite edge: gamma shape 0.3, NB theta 0.1, U-shaped beta,
            Cauchy t, Bernoulli trials, Tweedie with mostly zeros
  null      no signal at all, so every smooth should shrink to its null space
  scale     the base law in extreme units (responses near 1e4..1e6)

Every fit is ``y ~ s(x0) + s(x1)`` with gamfit only. The worker runs
:func:`run` in a fresh subprocess and reports the phases, the convergence
certificate, the estimated scale, and whether predictions and intervals came
back finite. :func:`failure_cause` turns one record into its failure cause
(``None`` for a clean fit), and running this module on run directories prints
the triage table:

    python -m pygam_compare.fuzz_families RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import importlib
import json
import re
import resource
import sys
import time
import traceback
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy import special

FloatArray = NDArray[np.float64]

REGIMES: tuple[str, ...] = (
    "base",
    "edge",
    "range",
    "zeros",
    "lowdisp",
    "highdisp",
    "null",
    "scale",
)
DESIGN_PREFIX = "ff-"
FORMULA = "y ~ s(x0) + s(x1)"
INTERVAL_LEVEL = 0.95
TEST_SEED_OFFSET = 1000
# Interval finiteness is checked on this many held-out rows; interval speed is
# the core plans' business, not the fuzzer's.
INTERVAL_ROWS = 200


@dataclass(frozen=True)
class Case:
    """One family/link cell: how to fit it and how to draw its response.

    ``kind`` names the data law; ``link`` the inverse link the truth is built
    on (the same link the fit uses, so every cell is well specified).
    """

    label: str
    family: str
    kind: str
    link: str
    trials: bool = False


CASES: tuple[Case, ...] = (
    Case("gamma", "gamma", "gamma", "log"),
    Case("gamma(inverse)", "gamma(inverse)", "gamma", "inverse"),
    Case("inverse-gaussian", "inverse-gaussian", "invgauss", "inverse_squared"),
    Case("inverse-gaussian(log)", "inverse-gaussian(log)", "invgauss", "log"),
    Case("negative-binomial", "negative-binomial", "negbin", "log"),
    Case("tweedie(1.2)", "tweedie(1.2)", "tweedie:1.2", "log"),
    Case("tweedie(1.5)", "tweedie(1.5)", "tweedie:1.5", "log"),
    Case("tweedie(1.8)", "tweedie(1.8)", "tweedie:1.8", "log"),
    Case("beta", "beta", "beta", "logit"),
    Case("student-t", "student-t", "student_t", "identity"),
    Case("poisson", "poisson", "poisson", "log"),
    Case("gaussian(inverse)", "gaussian(inverse)", "gaussian", "inverse"),
    Case("binomial-trials(logit)", "binomial(logit)", "binomial", "logit", True),
    Case("binomial-trials(probit)", "binomial(probit)", "binomial", "probit", True),
    Case("binomial-trials(cloglog)", "binomial(cloglog)", "binomial", "cloglog", True),
    Case("binomial-trials(loglog)", "binomial(loglog)", "binomial", "loglog", True),
    Case("binomial-trials(cauchit)", "binomial(cauchit)", "binomial", "cauchit", True),
    Case("binomial-trials(sas)", "binomial(sas)", "binomial", "logit", True),
    Case(
        "binomial-trials(beta-logistic)",
        "binomial(beta-logistic)",
        "binomial",
        "logit",
        True,
    ),
)
CASE_BY_LABEL: dict[str, Case] = {c.label: c for c in CASES}
FAMILY_LABELS: tuple[str, ...] = tuple(c.label for c in CASES)


def is_fuzz_design(design: str) -> bool:
    return design.startswith(DESIGN_PREFIX) and design[len(DESIGN_PREFIX) :] in REGIMES


def fuzz_design(regime: str) -> str:
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
    return DESIGN_PREFIX + regime


# --------------------------------------------------------------------------
# Data laws
# --------------------------------------------------------------------------


def _inverse_link(link: str, eta: FloatArray) -> FloatArray:
    if link == "identity":
        return eta
    if link == "log":
        return np.exp(eta)
    if link == "inverse":
        return 1.0 / eta
    if link == "inverse_squared":
        return 1.0 / np.sqrt(eta)
    if link == "logit":
        return special.expit(eta)
    if link == "probit":
        return special.ndtr(eta)
    if link == "cloglog":
        return -np.expm1(-np.exp(eta))
    if link == "loglog":
        return np.exp(-np.exp(-eta))
    if link == "cauchit":
        return 0.5 + np.arctan(eta) / np.pi
    raise ValueError(f"unknown link {link!r}")


def _link(link: str, mu: FloatArray) -> FloatArray:
    if link == "identity":
        return mu
    if link == "log":
        return np.log(mu)
    if link == "inverse":
        return 1.0 / mu
    if link == "inverse_squared":
        return mu**-2.0
    if link == "logit":
        return special.logit(mu)
    if link == "probit":
        return special.ndtri(mu)
    if link == "cloglog":
        return np.log(-np.log1p(-mu))
    if link == "loglog":
        return -np.log(-np.log(mu))
    if link == "cauchit":
        return np.tan(np.pi * (mu - 0.5))
    raise ValueError(f"unknown link {link!r}")


def _shape(x0: FloatArray, x1: FloatArray) -> FloatArray:
    """Centred truth shape in [-1, 1]: a sine in x0 plus a bowl in x1."""
    f = 0.6 * np.sin(2 * np.pi * x0) + 0.4 * (2.0 * (2 * x1 - 1) ** 2 - 1.0)
    return f


# The ``scale`` regime is the base problem with the response in units this
# many times smaller (positive families) or shifted this far (real line).
SCALE_FACTOR = 1e6

# Response-scale mean band per regime for each mean domain. ``pos`` is (0, inf),
# ``unit`` is (0, 1), ``real`` is the real line. A band (lo, hi) is where the
# truth's mean runs as the shape goes from -1 to 1 (geometric for ``pos``).
_POS_BAND = {
    "base": (0.5, 4.0),
    "edge": (1e-4, 1e-2),
    "range": (1e-4, 1e4),
    "zeros": (0.5, 4.0),
    "lowdisp": (0.5, 4.0),
    "highdisp": (0.5, 4.0),
    "null": (2.0, 2.0),
    "scale": (0.5 * SCALE_FACTOR, 4.0 * SCALE_FACTOR),
}
_COUNT_BAND = {
    "base": (0.5, 8.0),
    "edge": (0.01, 0.2),
    "range": (1e-3, 1e4),
    "zeros": (0.5, 8.0),
    "lowdisp": (0.5, 8.0),
    "highdisp": (0.5, 8.0),
    "null": (3.0, 3.0),
    "scale": (1e4, 1e5),
}
_UNIT_BAND = {
    "base": (0.2, 0.8),
    "edge": (1e-4, 1e-2),
    "range": (1e-4, 1 - 1e-4),
    "zeros": (0.2, 0.8),
    "lowdisp": (0.2, 0.8),
    "highdisp": (0.2, 0.8),
    "null": (0.35, 0.35),
    "scale": (0.2, 0.8),
}
_REAL_BAND = {
    "base": (-1.0, 1.0),
    "edge": (-1.0, 1.0),
    "range": (-1e3, 1e3),
    "zeros": (-1.0, 1.0),
    "lowdisp": (-1.0, 1.0),
    "highdisp": (-1.0, 1.0),
    "null": (0.3, 0.3),
    "scale": (SCALE_FACTOR - 1.0, SCALE_FACTOR + 1.0),
}

# Dispersion knob per regime: gamma shape k, inverse-Gaussian phi, NB theta,
# Tweedie phi, beta precision, Student-t (sigma, nu), binomial trials cap.
_DISP = {
    "gamma": {"lowdisp": 1e4, "highdisp": 0.3, "edge": 2.0, "_": 3.0},
    "invgauss": {"lowdisp": 1e-4, "highdisp": 5.0, "edge": 0.5, "_": 0.3},
    "negbin": {"lowdisp": 1e4, "highdisp": 0.1, "edge": 1.0, "_": 2.0},
    "tweedie": {"lowdisp": 1e-3, "highdisp": 5.0, "edge": 1.0, "_": 1.0},
    "beta": {"lowdisp": 1e4, "highdisp": 0.5, "edge": 20.0, "_": 10.0},
    "student_t_sigma": {"lowdisp": 1e-4, "highdisp": 0.3, "edge": 0.3, "_": 0.3},
    "student_t_nu": {"lowdisp": 30.0, "highdisp": 1.0, "edge": 1.0, "_": 4.0},
    "gaussian": {"lowdisp": 1e-4, "highdisp": 0.3, "edge": 0.05, "_": 0.05},
    "trials": {"lowdisp": 2000, "highdisp": 1, "edge": 20, "_": 20},
}


def _disp(key: str, regime: str) -> float:
    table = _DISP[key]
    return table.get(regime, table["_"])


def _band_mean(band: tuple[float, float], f: FloatArray, geometric: bool) -> FloatArray:
    lo, hi = band
    t = 0.5 * (f + 1.0)
    if geometric:
        return np.exp(np.log(lo) + t * (np.log(hi) - np.log(lo)))
    return lo + t * (hi - lo)


def _mean(case: Case, regime: str, x0: FloatArray, x1: FloatArray) -> FloatArray:
    """True response-scale mean. Built on the case's link so the fit is well
    specified: the shape is mapped into the regime's mean band on the link
    scale, then pushed through the inverse link."""
    f = _shape(x0, x1)
    if regime == "null":
        f = np.zeros_like(f)
    kind = case.kind.split(":")[0]
    if kind in ("poisson", "negbin", "tweedie"):
        band, domain = _COUNT_BAND[regime], "pos"
    elif kind in ("gamma", "invgauss"):
        band, domain = _POS_BAND[regime], "pos"
    elif kind in ("beta", "binomial"):
        band, domain = _UNIT_BAND[regime], "unit"
    else:
        band, domain = _REAL_BAND[regime], "real"
    if kind == "gaussian":
        # The inverse link needs a positive mean; keep the Gaussian-inverse
        # truth in (0.5, 2) (scale regime: a unit shift of 1e6 is still
        # positive, and range: 1e-2..1e2).
        band = {"range": (1e-2, 1e2), "scale": _REAL_BAND["scale"]}.get(
            regime, (0.5, 2.0) if regime != "null" else (1.0, 1.0)
        )
        domain = "pos"
    if band[0] == band[1]:
        return np.full_like(f, band[0])
    # Map the shape to the band on the link scale so the truth is a smooth
    # function of the covariates in the fitted model's own linear predictor.
    ends = _link(case.link, np.asarray(band, dtype=float))
    t = 0.5 * (f + 1.0)
    eta = ends[0] + t * (ends[1] - ends[0])
    mu = _inverse_link(case.link, eta)
    if domain == "unit":
        mu = np.clip(mu, band[0], band[1])
    if regime == "zeros" and kind in ("poisson", "negbin", "tweedie", "binomial"):
        # No events at all for x0 < 0.25: the truth's mean there is so small
        # (a probability of 1e-12) that the response is exactly zero.
        eta_zero = _link(case.link, np.asarray([1e-12]))[0]
        mu = np.where(x0 < 0.25, _inverse_link(case.link, np.full_like(mu, eta_zero)), mu)
    return mu


def _draw_response(
    case: Case, regime: str, mu: FloatArray, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray | None]:
    """Return ``(y, weights)``; ``weights`` is the binomial trial count."""
    n = mu.shape[0]
    kind, _, param = case.kind.partition(":")
    if kind == "gamma":
        k = _disp("gamma", regime)
        y = rng.gamma(k, mu / k)
        if regime == "zeros":
            # Continuous stand-in for "no events": a region at the smallest
            # scale the rest of the data leaves representable.
            y = np.where(mu < np.median(mu), y * 1e-8, y)
        return y, None
    if kind == "invgauss":
        phi = _disp("invgauss", regime)
        # numpy's Wald(mean, scale): Var = mean^3 / scale, so scale = 1/phi
        # in the gauge the fit reports (Var = phi mu^3). phi is not unit free:
        # rescaling y by c maps IG(mu, lam) to IG(c mu, c lam), so the
        # ``scale`` regime's law is the base law in the rescaled units.
        lam = (SCALE_FACTOR if regime == "scale" else 1.0) / phi
        y = rng.wald(mu, lam)
        if regime == "zeros":
            y = np.where(mu < np.median(mu), y * 1e-8, y)
        return y, None
    if kind == "poisson":
        return rng.poisson(mu).astype(float), None
    if kind == "negbin":
        theta = _disp("negbin", regime)
        return rng.negative_binomial(theta, theta / (theta + mu)).astype(float), None
    if kind == "tweedie":
        p = float(param)
        phi = _disp("tweedie", regime)
        lam = mu ** (2 - p) / (phi * (2 - p))
        alpha = (2 - p) / (p - 1)
        gam = phi * (p - 1) * mu ** (p - 1)
        counts = rng.poisson(lam)
        y = np.where(counts > 0, rng.gamma(np.maximum(counts, 1) * alpha, gam), 0.0)
        return y, None
    if kind == "beta":
        prec = _disp("beta", regime)
        y = rng.beta(mu * prec, (1 - mu) * prec)
        # A beta draw can round to exactly 0 or 1 in float64 when a shape is
        # tiny; the law's support is open, so move those onto the nearest
        # representable interior point (a response *at* the edge).
        y = np.clip(y, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
        return y, None
    if kind == "student_t":
        sigma = _disp("student_t_sigma", regime)
        nu = _disp("student_t_nu", regime)
        return mu + sigma * rng.standard_t(nu, n), None
    if kind == "gaussian":
        sd = _disp("gaussian", regime)
        return mu + sd * rng.standard_normal(n), None
    if kind == "binomial":
        cap = int(_disp("trials", regime))
        m = rng.integers(1, cap + 1, n) if cap > 1 else np.ones(n, dtype=np.int64)
        s = rng.binomial(m, mu)
        return s / m, m.astype(float)
    raise ValueError(f"unknown kind {case.kind!r}")


@dataclass(frozen=True)
class Draw:
    train: dict[str, FloatArray]
    test: dict[str, FloatArray]
    mu_train: FloatArray
    mu_test: FloatArray
    weights: bool


def draw(label: str, n: int, regime: str, seed: int) -> Draw:
    """Draw the training table and a held-out covariate table for one cell."""
    case = CASE_BY_LABEL[label]
    # Seed by cell as well as rep, so two cells never share a draw.
    key = zlib.crc32(f"{label}|{regime}|{n}".encode())
    rng = np.random.default_rng([seed, key])
    x0 = rng.uniform(0.0, 1.0, n)
    x1 = rng.uniform(0.0, 1.0, n)
    mu = _mean(case, regime, x0, x1)
    y, w = _draw_response(case, regime, mu, rng)
    train = {"x0": x0, "x1": x1, "y": y}
    if w is not None:
        train["w"] = w
    trng = np.random.default_rng([seed + TEST_SEED_OFFSET, key])
    t0 = trng.uniform(0.0, 1.0, n)
    t1 = trng.uniform(0.0, 1.0, n)
    return Draw(
        train=train,
        test={"x0": t0, "x1": t1},
        mu_train=mu,
        mu_test=_mean(case, regime, t0, t1),
        weights=w is not None,
    )


# --------------------------------------------------------------------------
# One rep (run by worker.py in a fresh subprocess)
# --------------------------------------------------------------------------


def _rss_peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run(label: str, n: int, design: str, seed: int) -> dict[str, Any]:
    """One fuzz rep: fit, predict, interval, summary; record what broke."""
    regime = design[len(DESIGN_PREFIX) :]
    case = CASE_BY_LABEL[label]
    data = draw(label, n, regime, seed)
    out: dict[str, Any] = {"base_rss_mb": _rss_peak_mb(), "formula": FORMULA}
    errors: dict[str, str] = {}
    error_types: dict[str, str] = {}

    def phase(name: str, fn: Callable[[], Any]) -> Any:
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            value = fn()
        except Exception as exc:
            errors[name] = traceback.format_exc(limit=4)[-2000:]
            error_types[name] = type(exc).__name__
            out.setdefault("error_head", f"{name}: {type(exc).__name__}: {exc}"[:400])
            return None
        out[f"{name}_s"] = time.perf_counter() - wall
        out[f"{name}_cpu_s"] = time.process_time() - cpu
        return value

    gamfit: Any = phase("import", lambda: importlib.import_module("gamfit"))
    model: Any = None
    if gamfit is not None:
        out["lib_version"] = str(gamfit.__version__)
        kwargs: dict[str, Any] = {"family": case.family}
        if data.weights:
            kwargs["weights"] = "w"
        model = phase("fit", lambda: gamfit.fit(data.train, FORMULA, **kwargs))
    if model is not None:
        out["rss_after_fit_mb"] = _rss_peak_mb()
        head = {k: v[:INTERVAL_ROWS] for k, v in data.test.items()}
        pred = phase(
            "pred",
            lambda: np.asarray(model.predict(data.test), dtype=float).reshape(-1),
        )
        iv = phase(
            "interval",
            lambda: model.predict(head, interval=INTERVAL_LEVEL, return_type="dict"),
        )
        summ = phase("info", model.summary)
        if summ is not None:
            conv = getattr(summ, "convergence", None)
            out["convergence"] = json.loads(json.dumps(conv, default=str))
            out["certified"] = None if conv is None else bool(conv.get("certified"))
            out["edf"] = None if summ.edf_total is None else float(summ.edf_total)
            out["scale"] = None if summ.scale is None else float(summ.scale)
            out["family_name"] = summ.family_name
        if pred is not None:
            out["pred_finite"] = bool(np.all(np.isfinite(pred)))
            if out["pred_finite"]:
                rel = (pred - data.mu_test) / np.maximum(np.abs(data.mu_test), 1e-300)
                out["rmse_mu"] = float(np.sqrt(np.mean((pred - data.mu_test) ** 2)))
                out["median_abs_rel_err"] = float(np.median(np.abs(rel)))
        if iv is not None:
            lo = np.asarray(iv["posterior_mean_lower"], dtype=float)
            hi = np.asarray(iv["posterior_mean_upper"], dtype=float)
            finite = np.isfinite(lo) & np.isfinite(hi)
            out["interval_finite"] = bool(np.all(finite))
            if out["interval_finite"]:
                mu = data.mu_test[: lo.shape[0]]
                out["coverage"] = float(np.mean((mu >= lo) & (mu <= hi)))
    out["peak_rss_mb"] = _rss_peak_mb()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    out["cpu_user_s"] = usage.ru_utime
    out["cpu_sys_s"] = usage.ru_stime
    out["status"] = "error" if errors else "ok"
    if errors:
        out["errors"] = errors
        out["error_types"] = error_types
    return out


# --------------------------------------------------------------------------
# Triage
# --------------------------------------------------------------------------

_NUM = re.compile(r"[-+]?\d[\d_.eE+-]*")


def _message_head(record: dict[str, Any]) -> str:
    head = str(record.get("error_head") or "")
    # Keep the typed error and its leading clause; numbers vary per rep.
    head = head.split("\n")[0]
    head = _NUM.sub("#", head)
    return head[:160]


# A fitted scale further than this many oracle standard errors from the truth
# is a broken dispersion estimate, not sampling noise. The oracle standard
# error (see :func:`scale_z`) prices only the draw's own noise, not the fit's
# smoothing bias, so clean fits reach a few standard errors at n=50; every
# broken estimate the fuzzer has found sits above ten.
SCALE_Z_MAX = 10.0


def _true_scale(
    case: Case, regime: str
) -> tuple[float, Callable[[FloatArray], FloatArray]] | None:
    """``(phi, V)`` with ``Var(y) = phi V(mu)`` in the gauge gamfit reports
    as ``scale``, or ``None`` where the scale is not a free dispersion, the
    regime's draw is not the family's law (the continuous ``zeros`` rows), or
    the truth is not resolvable at the noise level (``lowdisp``: there the
    basis's approximation error of the shape is as large as the noise, so a
    correct fit's scale carries it, and its distance from the noise-only
    truth grows as sqrt(n) — gaussian(inverse) at n=5000 sits 25 standard
    errors out on a saturated basis)."""
    if regime == "lowdisp":
        return None
    kind, _, param = case.kind.partition(":")
    if kind == "gamma" and regime != "zeros":
        return 1.0 / _disp("gamma", regime), lambda m: m**2
    if kind == "invgauss" and regime != "zeros":
        units = SCALE_FACTOR if regime == "scale" else 1.0
        return _disp("invgauss", regime) / units, lambda m: m**3
    if kind == "tweedie":
        p = float(param)
        return _disp("tweedie", regime), lambda m: m**p
    if kind == "gaussian":
        return _disp("gaussian", regime) ** 2, np.ones_like
    if kind == "beta":
        return 1.0 / (1.0 + _disp("beta", regime)), lambda m: m * (1.0 - m)
    return None


def scale_z(record: dict[str, Any]) -> float | None:
    """How many standard errors the fitted scale sits from the truth, on the
    log scale, or ``None`` when the family has no free dispersion.

    The standard error is the oracle Pearson estimate's: the relative spread
    of ``(y - mu)^2 / V(mu)`` at the true mean over sqrt(n), recomputed from
    the rep's seeded draw, so it grows with the law's tails and shrinks with
    n the way a correct dispersion estimate's error does."""
    scale = record.get("scale")
    case = CASE_BY_LABEL.get(str(record.get("family")))
    design = str(record.get("design", ""))
    if scale is None or case is None or not is_fuzz_design(design):
        return None
    regime = design[len(DESIGN_PREFIX) :]
    truth = _true_scale(case, regime)
    if truth is None or not (np.isfinite(scale) and scale > 0):
        return None
    phi, variance = truth
    data = draw(case.label, int(record["n"]), regime, int(record["seed"]))
    q = (data.train["y"] - data.mu_train) ** 2 / variance(data.mu_train)
    se = float(np.std(q) / np.mean(q) / np.sqrt(q.shape[0]))
    return float(np.log(scale / phi) / se)


def failure_cause(record: dict[str, Any]) -> str | None:
    """The failure cause of one fuzz record, or ``None`` for a clean fit.

    A rep fails when it hung (hit the harness safety net), crashed, raised in
    any phase, did not certify its optimum, predicted a non-finite value,
    reported a non-finite or non-positive scale, or reported a scale more
    than :data:`SCALE_Z_MAX` standard errors from the truth.
    """
    status = record.get("status")
    if status in ("timeout", "memcap", "crash") or str(status).startswith("not_run"):
        return str(status)
    if status == "error":
        return "raise " + _message_head(record)
    if record.get("certified") is not True:
        return "uncertified"
    if record.get("pred_finite") is not True:
        return "nonfinite prediction"
    if record.get("interval_finite") is not True:
        return "nonfinite interval"
    scale = record.get("scale")
    if scale is not None and not (np.isfinite(scale) and scale > 0):
        return "nonfinite or zero scale"
    if record.get("nonfinite"):
        return "nonfinite output " + ",".join(sorted(record["nonfinite"]))
    z = scale_z(record)
    if z is not None and abs(z) > SCALE_Z_MAX:
        return "scale off truth"
    return None


def load(dirs: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for d in dirs:
        for line in (d / "records.jsonl").read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return [r for r in records if is_fuzz_design(str(r.get("design", "")))]


def triage(records: list[dict[str, Any]]) -> str:
    """Markdown tables: failure rate by cause, by family and by regime."""
    lines: list[str] = []
    total = len(records)
    fails = [(r, failure_cause(r)) for r in records]
    bad = [(r, c) for r, c in fails if c is not None]
    lines.append(f"fits: {total}, failures: {len(bad)} ({100.0 * len(bad) / max(total, 1):.1f}%)")
    lines.append("")
    lines.append("| cause | count | example (FAMILY N DESIGN SEED) |")
    lines.append("|---|---|---|")
    by_cause: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r, c in bad:
        by_cause[str(c)].append(r)
    for cause, rs in sorted(by_cause.items(), key=lambda kv: -len(kv[1])):
        ex = rs[0]
        lines.append(
            f"| {cause.replace('|', '/')} | {len(rs)} | "
            f"`'{ex['family']}' {ex['n']} {ex['design']} {ex['seed']}` |"
        )
    for axis in ("family", "design", "n"):
        lines.append("")
        lines.append(f"| {axis} | fits | failures |")
        lines.append("|---|---|---|")
        tot: Counter[str] = Counter(str(r[axis]) for r in records)
        nb: Counter[str] = Counter(str(r[axis]) for r, _ in bad)
        for key in sorted(tot, key=lambda k: (len(k), k)):
            lines.append(f"| {key} | {tot[key]} | {nb[key]} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    print(triage(load([Path(a) for a in argv])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
