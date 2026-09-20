"""One benchmark rep: one library, one (family, n, design, seed) cell, fresh process.

Usage: worker.py LIB FAMILY N DESIGN SEED [N_PREDICT] [--postfit]

  LIB        gamfit | pygam | pygam_gs
  FAMILY     gaussian | binomial | poisson, a binomial variant of the
             ``binomial_*`` plans (see ``BINOMIAL_FAMILIES``), or a
             positive-continuous family of the ``positive_*`` plans (see
             ``POSITIVE_FAMILIES``)
  DESIGN     p<k> (additive in k covariates, e.g. p1, p3, p5, p20) | te | te+s
             | by (see ``make_data``) | fz<case> | ff-<regime>
  N_PREDICT  held-out rows to predict on (default: N)
  --postfit  also time the post-fit operations listed below

A ``fz<case>`` design is a convergence-fuzz case (``fuzz_terms.py``): a seeded
term structure — tensor, ``ti``, ``by=``, factor, random-effect, cyclic, 2-D
isotropic, shape-constrained and concurvity terms — fitted with gamfit only
on the gaussian, binomial and poisson families; see :func:`run_fuzz`.

An ``ff-<regime>`` design is a family-convergence-fuzz cell
(``fuzz_families.py``): FAMILY is then a fuzz family label (every family and
link gamfit supports, e.g. ``gamma(inverse)``, ``binomial-trials(cauchit)``)
and the data sit at an edge of the family's support; gamfit only.

Prints exactly one ``RESULT {json}`` line on stdout. The driver (``run.py``)
launches this script with a pinned thread environment and a scratch working
directory, so the installed gamfit wheel is imported rather than the source
tree's ``./gamfit`` (which has no compiled ``_rust``).

Phases are timed with both ``perf_counter`` (wall) and ``process_time`` (CPU of
this process, all threads). CPU is the primary speed metric: on a loaded or
shared host wall time measures the neighbours as much as the library.

  import    import of the library (numpy/scipy already imported)
  fit       one cold fit (the first fit in the process)
  fit_warm  a second fit of the same data in the same process: the per-fit cost
            once imports, lazy initialisation and caches are paid
  pred      point prediction on ``n_predict`` fresh rows
  interval  95% interval prediction on the same rows (plus the observation
            interval for Gaussian, which the Gaussian log score needs)

With ``--postfit`` the fitted model's other post-fit operations are timed too:

  pd        a term's partial dependence with its standard error on a
            200-point grid
  summary   the model summary
  save      writing the model to disk (``save_bytes`` is the file size)
  load      reading it back
  sample    100 coefficient draws from the posterior
  sig       gamfit's per-term smooth significance (pyGAM has no separate
            operation: its p-values are part of the fit)

Held-out accuracy is computed on ``n_predict`` fresh rows drawn with ``seed + 1000``,
against both the true mean (``rmse_mu``, ``coverage``) and the drawn response
(``deviance``, ``logscore``). A phase that raises is recorded under ``errors``
and makes the rep's status ``error``; the metrics of the phases that did run
are still reported.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import pickle
import re
import resource
import sys
import time
import traceback
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats

if __package__:
    from . import fuzz_families, fuzz_terms
else:  # run as a script by run.py: this directory is sys.path[0]
    import fuzz_families  # type: ignore[no-redef]
    import fuzz_terms  # type: ignore[no-redef]

LIBS = ("gamfit", "pygam", "pygam_gs")
FAMILIES = ("gaussian", "binomial", "poisson")
# Binomial variants of the ``binomial_*`` plans (audit lane sweep-binomial).
# ``binomial`` itself is the balanced case (prevalence about 0.5).
# ``binomial_p10`` / ``binomial_p01`` shift the logit by ``logit(0.1)`` /
# ``logit(0.01)`` for rare outcomes; ``binomial_trials`` is a grouped binomial
# with ``m_i`` trials per row drawn uniformly from ``1..TRIALS_MAX``, fitted as
# the observed proportion ``y_i / m_i`` with prior weight ``m_i`` by both
# libraries (the binomial likelihood with ``m_i`` trials up to a constant).
BINOMIAL_FAMILIES = ("binomial_p10", "binomial_p01", "binomial_trials")
BINOMIAL_PREVALENCE = {"binomial_p10": 0.1, "binomial_p01": 0.01}
BINOMIAL_SLOPE = 1.5
TRIALS_MAX = 20
# Positive-continuous families of the ``positive_*`` plans (audit lane
# sweep-positive). The mean is ``exp(0.5 + 0.7 eta)`` unless stated.
#   gamma_log           Gamma, shape 3, log link.
#   gamma_skew          Gamma, shape 1/2, mean ``exp(0.7 eta - 2)``: heavy right
#                       skew with most responses near zero.
#   gamma_inverse       Gamma, shape 3, fitted on the canonical inverse link;
#                       the truth is ``1 / mu = 1 + 0.2 eta``, which stays
#                       positive for every design (``|eta| <= sqrt(p)``).
#   inverse_gaussian    inverse Gaussian, ``V = phi mu^3`` with phi = 0.3; the
#                       truth is on the log scale, the fit on the canonical
#                       ``1 / mu^2`` link (gamfit's default for the family).
#   lognormal_gaussian  ``log y = 0.5 + 0.7 eta + N(0, 0.5^2)`` fitted as a
#                       Gaussian on the log scale (response and mean are logs).
#   lognormal_gamma     the same draw of ``y`` fitted by Gamma(log) on the raw
#                       scale; the mean is ``E[y] = exp(0.5 + 0.7 eta + 0.125)``.
#   student_t           ``y = eta + 0.5 t_3``, fitted with scale and degrees
#                       of freedom estimated.
POSITIVE_FAMILIES = (
    "gamma_log",
    "gamma_skew",
    "gamma_inverse",
    "inverse_gaussian",
    "lognormal_gaussian",
    "lognormal_gamma",
    "student_t",
)
ALL_FAMILIES = FAMILIES + BINOMIAL_FAMILIES + POSITIVE_FAMILIES
GAMMA_SHAPE, GAMMA_SKEW_SHAPE, GAMMA_SKEW_LEVEL = 3.0, 0.5, -2.0
GAMMA_INVERSE_SLOPE = 0.2
INVERSE_GAUSSIAN_PHI = 0.3
LOGNORMAL_SD = 0.5
STUDENT_T_DF, STUDENT_T_SCALE = 3.0, 0.5
# pyGAM fits every binomial variant, the Gamma families (log and inverse
# links) and both log-normal fits. It has no scaled-t family, and its InvGaussGAM stores sqrt(phi) as its
# scale (see inverse_gaussian_scale.py), so it is not a like-for-like
# comparator: those two run gamfit alone and report absolute times and the
# certification rate.
PYGAM_FAMILIES = frozenset(
    {
        *FAMILIES,
        *BINOMIAL_FAMILIES,
        "gamma_log",
        "gamma_skew",
        "gamma_inverse",
        "lognormal_gaussian",
        "lognormal_gamma",
    }
)
# gamfit (family, link) of each family whose name is not a gamfit family.
GAMFIT_FAMILY: dict[str, tuple[str, str | None]] = {
    **{family: ("binomial", None) for family in BINOMIAL_FAMILIES},
    "gamma_log": ("gamma", None),
    "gamma_skew": ("gamma", None),
    "gamma_inverse": ("gamma", "inverse"),
    "inverse_gaussian": ("inverse-gaussian", None),
    "lognormal_gaussian": ("gaussian", None),
    "lognormal_gamma": ("gamma", None),
    "student_t": ("student-t", None),
}
DESIGNS = ("p1", "p5", "p20", "te")
# Designs beyond the core grid, run by the Gaussian sweep plans: a tensor plus
# an additive smooth, and a factor-by smooth (one curve per level).
EXTRA_DESIGNS = ("te+s", "by")
BY_LEVELS = ("a", "b", "c")
INTERVAL_LEVEL = 0.95
TEST_SEED_OFFSET = 1000
PD_POINTS = 200
SAMPLE_DRAWS = 100
# The fuzz interval phase checks that intervals are finite on this many
# held-out rows; interval speed is measured by the core plans, not here.
FUZZ_INTERVAL_ROWS = 200

FloatArray = NDArray[np.float64]


def supports(lib: str, family: str) -> bool:
    return lib == "gamfit" or family in PYGAM_FAMILIES


def design_width(design: str) -> int | None:
    """Covariate count of an additive design ``p<k>``; ``None`` otherwise."""
    match = re.fullmatch(r"p([1-9][0-9]*)", design)
    return None if match is None else int(match.group(1))


def is_binomial(family: str) -> bool:
    return family == "binomial" or family in BINOMIAL_FAMILIES


def make_data(
    n: int, design: str, family: str, seed: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray | None]:
    """Draw ``(X, y, mu, weights)``: covariates, response, true response-scale
    mean, and the per-row trial counts (``None`` for families without them).

    Same generators as the pyGAM audit (bench/pygam_audit/speed/worker.py) so
    numbers stay comparable with the audit's speed.md tables. The extra designs:

      te+s  ``sin(2 pi x0) cos(2 pi x1) + sin(2 pi x2)``, fitted as
            ``te(x0, x1) + s(x2)``;
      by    ``sin(2 pi x0 + g) + (g - 1) / 2`` for a factor ``g`` with levels
            ``BY_LEVELS`` (column 1 of ``X`` holds its integer code), fitted
            as a factor-by smooth of ``x0``.
    """
    rng = np.random.default_rng(seed)
    if design == "te":
        X = rng.uniform(0.0, 1.0, (n, 2))
        eta = np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])
    elif design == "te+s":
        X = rng.uniform(0.0, 1.0, (n, 3))
        eta = np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])
        eta += np.sin(2 * np.pi * X[:, 2])
    elif design == "by":
        x = rng.uniform(0.0, 1.0, n)
        g = rng.integers(0, len(BY_LEVELS), n).astype(float)
        X = np.column_stack([x, g])
        eta = np.sin(2 * np.pi * x + g) + (g - 1) / 2
    elif design_width(design) is not None:
        p = design_width(design)
        X = rng.uniform(0.0, 1.0, (n, p))
        eta = np.zeros(n)
        for j in range(p):
            eta += np.sin(2 * np.pi * X[:, j] + j) / np.sqrt(p)
    else:
        raise ValueError(
            f"unknown design {design!r}; expected p<k> or one of te, {EXTRA_DESIGNS}"
        )
    if family == "gaussian":
        mu = eta
        y = eta + rng.normal(0.0, 0.5, n)
    elif family == "binomial":
        mu = special.expit(BINOMIAL_SLOPE * eta)
        y = (rng.uniform(size=n) < mu).astype(float)
    elif family in BINOMIAL_PREVALENCE:
        base = special.logit(BINOMIAL_PREVALENCE[family])
        mu = special.expit(base + BINOMIAL_SLOPE * eta)
        y = (rng.uniform(size=n) < mu).astype(float)
    elif family == "binomial_trials":
        mu = special.expit(BINOMIAL_SLOPE * eta)
        trials = rng.integers(1, TRIALS_MAX + 1, n).astype(float)
        y = rng.binomial(trials.astype(np.int64), mu) / trials
        return X, y, mu, trials
    elif family == "poisson":
        mu = np.exp(0.5 + 0.7 * eta)
        y = rng.poisson(mu).astype(float)
    elif family in ("gamma_log", "gamma_skew"):
        shape = GAMMA_SHAPE if family == "gamma_log" else GAMMA_SKEW_SHAPE
        level = 0.5 if family == "gamma_log" else GAMMA_SKEW_LEVEL
        mu = np.exp(level + 0.7 * eta)
        y = rng.gamma(shape, mu / shape)
    elif family == "gamma_inverse":
        mu = 1.0 / (1.0 + GAMMA_INVERSE_SLOPE * eta)
        y = rng.gamma(GAMMA_SHAPE, mu / GAMMA_SHAPE)
    elif family == "inverse_gaussian":
        mu = np.exp(0.5 + 0.7 * eta)
        # numpy's wald(mean, scale) has variance mean^3 / scale.
        y = rng.wald(mu, 1.0 / INVERSE_GAUSSIAN_PHI)
    elif family in ("lognormal_gaussian", "lognormal_gamma"):
        log_mean = 0.5 + 0.7 * eta
        log_y = log_mean + rng.normal(0.0, LOGNORMAL_SD, n)
        if family == "lognormal_gaussian":
            mu, y = log_mean, log_y
        else:
            mu, y = np.exp(log_mean + LOGNORMAL_SD**2 / 2), np.exp(log_y)
    elif family == "student_t":
        mu = eta
        y = eta + STUDENT_T_SCALE * rng.standard_t(STUDENT_T_DF, n)
    else:
        raise ValueError(f"unknown family {family!r}; expected one of {ALL_FAMILIES}")
    return X, y, mu, None


def mean_deviance(
    family: str, y: FloatArray, mu: FloatArray, trials: FloatArray | None = None
) -> float:
    """Mean unit deviance of held-out ``y`` at predicted mean ``mu``.

    A grouped binomial is the per-trial deviance: the rows are weighted by
    their trial counts.
    """
    if family in ("gaussian", "lognormal_gaussian", "student_t"):
        return float(np.mean((y - mu) ** 2))
    if family.startswith("gamma") or family == "lognormal_gamma":
        return float(np.mean(2 * (-np.log(y / mu) + (y - mu) / mu)))
    if family == "inverse_gaussian":
        return float(np.mean((y - mu) ** 2 / (mu**2 * y)))
    if is_binomial(family):
        unit = special.xlogy(y, y / mu) + special.xlogy(1 - y, (1 - y) / (1 - mu))
        return float(np.average(2 * unit, weights=trials))
    unit = special.xlogy(y, y / mu) - (y - mu)
    return float(np.mean(2 * unit))


def mean_logscore(
    family: str,
    y: FloatArray,
    mu: FloatArray,
    predictive_sd: FloatArray | None,
    trials: FloatArray | None = None,
) -> float | None:
    """Mean negative log predictive density of held-out ``y`` (lower is better).

    Binomial and Poisson are scored at the predicted mean; a grouped binomial
    scores the observed count out of its trials, per trial. Gaussian needs a
    predictive scale: the library's own 95% prediction interval gives it as
    ``(upper - lower) / (2 z_0.975)``, which prices both the fitted scale and
    the posterior variance of the mean. The positive families need their fitted
    shape / dispersion, which the libraries do not expose alike, so they report
    no log score.
    """
    if family in POSITIVE_FAMILIES:
        return None
    if trials is not None:
        counts = np.rint(y * trials)
        return float(-np.sum(stats.binom.logpmf(counts, trials, mu)) / np.sum(trials))
    if is_binomial(family):
        return float(-np.mean(special.xlogy(y, mu) + special.xlogy(1 - y, 1 - mu)))
    if family == "poisson":
        return float(-np.mean(stats.poisson.logpmf(y, mu)))
    if predictive_sd is None:
        raise ValueError("gaussian logscore needs a predictive sd")
    return float(-np.mean(stats.norm.logpdf(y, loc=mu, scale=predictive_sd)))


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [leaf for item in value for leaf in _flatten(item)]
    return [value]


def rss_peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class Timer:
    def __init__(self) -> None:
        self.wall = time.perf_counter()
        self.cpu = time.process_time()

    def stop(self) -> tuple[float, float]:
        return time.perf_counter() - self.wall, time.process_time() - self.cpu


class Adapter:
    """Library-specific fit / predict calls behind one interface."""

    version: str

    def fit(self, X: FloatArray, y: FloatArray, weights: FloatArray | None) -> None:
        """Fit ``y`` on ``X``; ``weights`` are binomial trial counts or ``None``."""
        raise NotImplementedError

    def predict(self, X: FloatArray) -> FloatArray:
        raise NotImplementedError

    def interval(
        self, X: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray | None]:
        """Return (lower, upper) of the 95% interval for the mean, and the
        Gaussian predictive sd (``None`` for other families)."""
        raise NotImplementedError

    def model_info(self) -> dict[str, Any]:
        raise NotImplementedError

    def postfit_ops(
        self, X: FloatArray, y: FloatArray, path: str
    ) -> dict[str, Callable[[], Any]]:
        """The post-fit operations timed under ``--postfit``, in run order.

        ``X`` / ``y`` are the training data (the posterior draws and the
        significance refits need the response); ``path`` is a scratch file
        for the save / load round trip, written by ``save`` before ``load``.
        """
        raise NotImplementedError


WEIGHTS_COLUMN = "trials"


class GamfitAdapter(Adapter):
    def __init__(self, family: str, design: str, p: int) -> None:
        self.gamfit: Any = importlib.import_module("gamfit")
        self.version = str(self.gamfit.__version__)
        self.family = family
        self.gamfit_family, self.link = GAMFIT_FAMILY.get(family, (family, None))
        self.names = [f"x{j}" for j in range(p)]
        self.factor = design == "by"
        if design == "te":
            self.formula = "y ~ te(x0, x1)"
        elif design == "te+s":
            self.formula = "y ~ te(x0, x1) + s(x2)"
        elif self.factor:
            self.names = ["x0"]
            self.formula = "y ~ s(x0, by=g)"
        else:
            self.formula = "y ~ " + " + ".join(f"s({nm})" for nm in self.names)
        self.model: Any = None

    def _table(self, X: FloatArray) -> dict[str, Any]:
        table: dict[str, Any] = {nm: X[:, j] for j, nm in enumerate(self.names)}
        if self.factor:
            table["g"] = np.asarray(BY_LEVELS)[X[:, 1].astype(int)]
        return table

    def fit(self, X: FloatArray, y: FloatArray, weights: FloatArray | None) -> None:
        data = self._table(X)
        data["y"] = y
        if weights is not None:
            data[WEIGHTS_COLUMN] = weights
        self.model = self.gamfit.fit(
            data,
            self.formula,
            family=self.gamfit_family,
            link=self.link,
            weights=None if weights is None else WEIGHTS_COLUMN,
        )

    def predict(self, X: FloatArray) -> FloatArray:
        return np.asarray(self.model.predict(self._table(X)), dtype=float).reshape(-1)

    def interval(
        self, X: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray | None]:
        gaussian = self.family == "gaussian"
        res = self.model.predict(
            self._table(X),
            interval=INTERVAL_LEVEL,
            observation_interval=gaussian,
            return_type="dict",
        )
        lo = np.asarray(res["posterior_mean_lower"], dtype=float)
        hi = np.asarray(res["posterior_mean_upper"], dtype=float)
        sd = None
        if gaussian:
            olo = np.asarray(res["observation_lower"], dtype=float)
            ohi = np.asarray(res["observation_upper"], dtype=float)
            sd = (ohi - olo) / (2 * stats.norm.ppf(0.5 + INTERVAL_LEVEL / 2))
        return lo, hi, sd

    def model_info(self) -> dict[str, Any]:
        summ = self.model.summary()
        conv = json.loads(json.dumps(summ.convergence, default=str))
        return {
            "edf": None if summ.edf_total is None else float(summ.edf_total),
            "ncoef": None if summ.coefficients is None else len(summ.coefficients),
            "iterations": self.model.outer_iterations,
            "inner_iterations": self.model.inner_iterations,
            "certified": conv.get("certified") if isinstance(conv, dict) else None,
            "convergence": conv,
        }

    def postfit_ops(
        self, X: FloatArray, y: FloatArray, path: str
    ) -> dict[str, Callable[[], Any]]:
        train = self._table(X)
        train["y"] = y
        term = self.formula.split("~ ", 1)[1].split(" + ", 1)[0]
        return {
            "pd": lambda: self.model.partial_dependence(term, n_points=PD_POINTS),
            "summary": self.model.summary,
            "save": lambda: self.model.save(path),
            "load": lambda: self.gamfit.load(path),
            "sample": lambda: self.model.sample(train, samples=SAMPLE_DRAWS, seed=0),
            "sig": lambda: self.model.smooth_significance(train),
        }


class PygamAdapter(Adapter):
    def __init__(self, family: str, design: str, p: int, gridsearch: bool) -> None:
        # pyGAM ships no type information; its API is used as Any.
        pygam: Any = importlib.import_module("pygam")
        self.pygam = pygam
        self.version = str(pygam.__version__)
        self.family = family
        if design == "te":
            self.terms: Any = pygam.te(0, 1)
        elif design == "te+s":
            self.terms = pygam.te(0, 1) + pygam.s(2)
        elif design == "by":
            # pyGAM's ``by=`` is a numeric multiplier only; its factor-by smooth
            # is the tensor of a spline in x0 with the categorical marginal of
            # the factor, which spans one curve per level with its level offset.
            self.terms = pygam.te(0, 1, dtype=["numerical", "categorical"])
        else:
            self.terms = pygam.s(0)
            for j in range(1, p):
                self.terms = self.terms + pygam.s(j)
        if family not in PYGAM_FAMILIES:
            raise ValueError(f"pyGAM has no {family!r} comparator")
        self.dist, self.link = {
            "gaussian": ("normal", "identity"),
            "binomial": ("binomial", "logit"),
            "poisson": ("poisson", "log"),
            "gamma_log": ("gamma", "log"),
            "gamma_skew": ("gamma", "log"),
            "gamma_inverse": ("gamma", "inverse"),
            "lognormal_gaussian": ("normal", "identity"),
            "lognormal_gamma": ("gamma", "log"),
        }["binomial" if is_binomial(family) else family]
        self.gridsearch = gridsearch
        self.model: Any = None

    def fit(self, X: FloatArray, y: FloatArray, weights: FloatArray | None) -> None:
        if self.family == "gaussian":
            # LinearGAM is GAM(normal, identity) plus prediction_intervals,
            # which the Gaussian log score needs.
            g = self.pygam.LinearGAM(self.terms)
        else:
            g = self.pygam.GAM(self.terms, distribution=self.dist, link=self.link)
        # A grouped binomial is the proportion with its trials as prior
        # weights: pyGAM's binomial IRLS weight is then ``m_i mu (1 - mu)``,
        # the grouped-binomial Fisher information.
        extra = {} if weights is None else {"weights": weights}
        if self.gridsearch:
            g.gridsearch(X, y, progress=False, **extra)
        else:
            g.fit(X, y, **extra)
        self.model = g

    def predict(self, X: FloatArray) -> FloatArray:
        return np.asarray(self.model.predict(X), dtype=float).reshape(-1)

    def interval(
        self, X: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray | None]:
        ci = np.asarray(
            self.model.confidence_intervals(X, width=INTERVAL_LEVEL), dtype=float
        )
        sd = None
        if self.family == "gaussian":
            pi = np.asarray(
                self.model.prediction_intervals(X, width=INTERVAL_LEVEL), dtype=float
            )
            sd = (pi[:, 1] - pi[:, 0]) / (2 * stats.norm.ppf(0.5 + INTERVAL_LEVEL / 2))
        return ci[:, 0], ci[:, 1], sd

    def model_info(self) -> dict[str, Any]:
        return {
            "edf": float(self.model.statistics_["edof"]),
            "ncoef": int(len(self.model.coef_)),
            "lam": [float(v) for v in _flatten(self.model.lam)],
        }

    def postfit_ops(
        self, X: FloatArray, y: FloatArray, path: str
    ) -> dict[str, Callable[[], Any]]:
        def pd() -> Any:
            grid = self.model.generate_X_grid(term=0, n=PD_POINTS)
            return self.model.partial_dependence(term=0, X=grid, width=INTERVAL_LEVEL)

        def summary() -> None:
            with contextlib.redirect_stdout(io.StringIO()):
                self.model.summary()

        def save() -> None:
            with open(path, "wb") as fh:
                pickle.dump(self.model, fh)

        def load() -> Any:
            with open(path, "rb") as fh:
                return pickle.load(fh)

        return {
            "pd": pd,
            "summary": summary,
            "save": save,
            "load": load,
            # Coefficient draws from pyGAM's Gaussian posterior approximation;
            # n_bootstraps=1 keeps it to the fitted smoothing parameters
            # instead of refitting on bootstrap resamples.
            "sample": lambda: self.model.sample(
                X, y, quantity="coef", n_draws=SAMPLE_DRAWS, n_bootstraps=1
            ),
        }


def run_fuzz(family: str, n: int, design: str, seed: int) -> dict[str, Any]:
    """One convergence-fuzz rep (design ``fz<case>``, gamfit only).

    Same phases as :func:`run`, plus what the fuzz triage needs: the formula,
    the certificate verdict, whether every prediction and interval came back
    finite, and the exception type of a phase that raised, so a typed
    build-time refusal can be told apart from a solver failure.
    """
    data = fuzz_terms.draw(int(design[2:]), n, family, seed)
    out: dict[str, Any] = {"base_rss_mb": rss_peak_mb(), "formula": data.formula}
    errors: dict[str, str] = {}
    error_types: dict[str, str] = {}

    def phase(name: str, fn: Callable[[], Any]) -> Any:
        t = Timer()
        try:
            value = fn()
        except Exception as exc:
            errors[name] = traceback.format_exc(limit=4)[-2000:]
            error_types[name] = type(exc).__name__
            return None
        out[f"{name}_s"], out[f"{name}_cpu_s"] = t.stop()
        return value

    gamfit: Any = phase("import", lambda: importlib.import_module("gamfit"))
    model: Any = None
    if gamfit is not None:
        out["lib_version"] = str(gamfit.__version__)
        out["lib_file"] = str(gamfit.__file__)
        train = fuzz_terms.as_frame(data.train, data.categorical)
        model = phase("fit", lambda: gamfit.fit(train, data.formula, family=family))
    if model is not None:
        out["rss_after_fit_mb"] = rss_peak_mb()
        test = fuzz_terms.as_frame(data.test, data.categorical)
        head = fuzz_terms.as_frame(
            {k: v[:FUZZ_INTERVAL_ROWS] for k, v in data.test.items()},
            data.categorical,
        )
        pred = phase(
            "pred", lambda: np.asarray(model.predict(test), dtype=float).reshape(-1)
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
        if pred is not None:
            out["pred_finite"] = bool(np.all(np.isfinite(pred)))
            if out["pred_finite"]:
                out["rmse_mu"] = float(np.sqrt(np.mean((pred - data.mu_test) ** 2)))
        if iv is not None:
            lo = np.asarray(iv["posterior_mean_lower"], dtype=float)
            hi = np.asarray(iv["posterior_mean_upper"], dtype=float)
            finite = np.isfinite(lo) & np.isfinite(hi)
            out["interval_finite"] = bool(np.all(finite))
            if out["interval_finite"]:
                mu = data.mu_test[: lo.shape[0]]
                out["coverage"] = float(np.mean((mu >= lo) & (mu <= hi)))
    out["peak_rss_mb"] = rss_peak_mb()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    out["cpu_user_s"] = usage.ru_utime
    out["cpu_sys_s"] = usage.ru_stime
    out["status"] = "error" if errors else "ok"
    if errors:
        out["errors"] = errors
        out["error_types"] = error_types
    return out


def run(
    lib: str,
    family: str,
    n: int,
    design: str,
    seed: int,
    n_predict: int | None = None,
    postfit: bool = False,
) -> dict[str, Any]:
    if lib not in LIBS:
        raise ValueError(f"unknown lib {lib!r}; expected one of {LIBS}")
    if fuzz_terms.is_fuzz_design(design):
        if lib != "gamfit":
            raise ValueError(f"fuzz design {design!r} is gamfit-only, got lib {lib!r}")
        return run_fuzz(family, n, design, seed)
    if fuzz_families.is_fuzz_design(design):
        if lib != "gamfit":
            raise ValueError(f"fuzz design {design!r} is gamfit-only, got lib {lib!r}")
        return fuzz_families.run(family, n, design, seed)
    X, y, _, w = make_data(n, design, family, seed)
    n_test = n if n_predict is None else n_predict
    Xt, yt, mut, wt = make_data(n_test, design, family, seed + TEST_SEED_OFFSET)
    out: dict[str, Any] = {"base_rss_mb": rss_peak_mb()}
    if n_predict is not None:
        out["n_predict"] = n_predict
    errors: dict[str, str] = {}

    def phase(name: str, fn: Callable[[], Any], timed: bool = True) -> Any:
        t = Timer()
        try:
            value = fn()
        except Exception:
            errors[name] = traceback.format_exc(limit=4)[-2000:]
            return None
        if timed:
            out[f"{name}_s"], out[f"{name}_cpu_s"] = t.stop()
        return value

    def make_adapter() -> Adapter:
        if lib == "gamfit":
            return GamfitAdapter(family, design, X.shape[1])
        return PygamAdapter(family, design, X.shape[1], gridsearch=lib == "pygam_gs")

    adapter: Adapter | None = phase("import", make_adapter)
    if adapter is not None:
        out["lib_version"] = adapter.version
        out["after_import_rss_mb"] = rss_peak_mb()
        phase("fit", lambda: adapter.fit(X, y, w))
    if adapter is not None and "fit" not in errors:
        phase("fit_warm", lambda: adapter.fit(X, y, w))
    if adapter is not None and not {"fit", "fit_warm"} & errors.keys():
        out["rss_after_fit_mb"] = rss_peak_mb()
        pred: FloatArray | None = phase("pred", lambda: adapter.predict(Xt))
        iv = phase("interval", lambda: adapter.interval(Xt))
        info = phase("info", adapter.model_info, timed=False)
        if info is not None:
            out.update(info)
        if pred is not None:
            out["rmse_mu"] = float(np.sqrt(np.mean((pred - mut) ** 2)))
            out["rmse_y"] = float(np.sqrt(np.mean((pred - yt) ** 2)))
            out["deviance"] = mean_deviance(family, yt, pred, wt)
        if iv is not None:
            lo, hi, sd = iv
            out["coverage"] = float(np.mean((mut >= lo) & (mut <= hi)))
            out["ci_width"] = float(np.mean(hi - lo))
            if pred is not None:
                out["logscore"] = phase(
                    "logscore", lambda: mean_logscore(family, yt, pred, sd, wt), timed=False
                )
        elif pred is not None and family != "gaussian":
            out["logscore"] = mean_logscore(family, yt, pred, None, wt)
        if postfit:
            path = os.path.abspath(f"postfit_{lib}_{os.getpid()}.model")
            for name, op in adapter.postfit_ops(X, y, path).items():
                phase(name, op)
            if os.path.exists(path):
                out["save_bytes"] = os.path.getsize(path)
                os.remove(path)
    out["peak_rss_mb"] = rss_peak_mb()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    out["cpu_user_s"] = usage.ru_utime
    out["cpu_sys_s"] = usage.ru_stime
    out["status"] = "error" if errors else "ok"
    if errors:
        out["errors"] = errors
    return out


def main(argv: list[str]) -> int:
    postfit = "--postfit" in argv
    args = [a for a in argv if a != "--postfit"]
    if len(args) not in (5, 6):
        print(__doc__, file=sys.stderr)
        return 2
    lib, family, n, design, seed = (
        args[0],
        args[1],
        int(float(args[2])),
        args[3],
        int(args[4]),
    )
    n_predict = int(float(args[5])) if len(args) == 6 else None
    out = run(lib, family, n, design, seed, n_predict=n_predict, postfit=postfit)
    # Non-finite floats become null: JSON has no NaN/Inf, and a report that
    # silently parses "NaN" would hide a broken metric.
    nonfinite = [
        k for k, v in out.items() if isinstance(v, float) and not np.isfinite(v)
    ]
    clean = {k: (None if k in nonfinite else v) for k, v in out.items()}
    if nonfinite:
        clean["nonfinite"] = nonfinite
    print("RESULT " + json.dumps(clean, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
