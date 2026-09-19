"""One chunk of calibration reps: one (family, n, null) cell, a run of seeds.

Usage: worker.py FAMILY N NULL SEED_START SEED_STOP LIBS

  FAMILY  gaussian | binomial | poisson | gamma | negbin
  NULL    smooth | linear | factor | ti | re | concurvity
  LIBS    comma-separated subset of gamfit,pygam,pygam_gs

Prints one ``RESULT {json}`` line per seed in ``range(SEED_START, SEED_STOP)``,
flushed as soon as that rep is done, so a chunk the driver's safety net kills
still keeps the reps it finished. The driver (``run.py``) launches this script
through ``bench/pygam_compare``'s policed runner: a pinned one-thread
environment and a scratch working directory.

Each rep draws two datasets from its seed, one under the cell's null and one
under the matched alternative, and fits every library to both. A rep records,
per hypothesis, the p-value each library reports for the tested term on each
surface it has:

  gamfit.wald  ``summary().smooth_terms[...]["p_value"]``: Wood's rank-truncated
               Wald test.
  gamfit.lr    ``smooth_significance(data)[...]["p_value_corrected"]``: the
               per-term likelihood-ratio test from a constrained null refit,
               the value that method documents as its headline.
  gamfit.coef  the coefficient row's own ``p_value`` when it has one. gamfit
               reports none today, so the harness reads the two-sided normal
               tail of ``estimate / std_error`` from the same row and labels
               the record ``coef_source = "z_from_std_error"``. That checks the
               reported standard error, which is what such a p-value would be
               built from. The row is located through ``model.term_blocks``.
  pygam.wald   pyGAM's ``statistics_["p_values"]`` for the tested term, with
  pygam_gs.*   the default fixed lambda and with ``gridsearch`` respectively.

A surface a library has no counterpart for is not in ``expected_surfaces``
(pyGAM has no ``ti``, random effect or negative binomial, and no LR test). A
surface that is expected but produced no p-value is recorded under ``missing``
with the reason, and a fit that raised under ``errors``. The report counts both
as unusable reps; it never drops them.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
import time
import traceback
import zlib
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats

LIBS = ("gamfit", "pygam", "pygam_gs")
FAMILIES = ("gaussian", "binomial", "poisson", "gamma", "negbin")
NULLS = ("smooth", "linear", "factor", "ti", "re", "concurvity")
HYPOTHESES = ("null", "alt")

FloatArray = NDArray[np.float64]

# --- data-generating process -------------------------------------------------
#
# eta = b0 + a1 * sin(2 pi x1) + [structure-specific nuisance] + delta * t,
#
# where t is the tested term, scaled to unit variance over the design, and
# delta = 0 under the null. The family constants below set the baseline mean
# and the nuisance smooth, the same shape the audit's calibration cells used
# (inference.md section 4). They are properties of the simulated world, not of
# any fitting method.
BASELINE: dict[str, tuple[float, float]] = {
    # family: (b0, a1)
    "gaussian": (0.0, 1.0),
    "binomial": (0.0, 1.0),
    "poisson": (0.5, 0.5),
    "gamma": (0.5, 0.5),
    "negbin": (1.0, 0.5),
}
GAUSSIAN_SD = 1.0
GAMMA_SHAPE = 2.0
NEGBIN_THETA = 2.0
# Correlation of the latent normals behind x1 and x2 in the ``concurvity``
# structure (the lane's "concurvity 0.9").
CONCURVITY_RHO = 0.9
FACTOR_LEVELS = ("a", "b", "c")
# The random-effect structure has one level per this many rows.
RE_ROWS_PER_GROUP = 10
# The alternative is a local one: delta = sqrt(ALT_NONCENTRALITY / (n * I)),
# with I the Fisher information for eta of one row at the baseline mean. A
# one-degree-of-freedom test of a unit-variance effect then has the same
# noncentrality at every family and n, so the power column compares like
# with like. 9 is a 3-sigma effect: power 0.85 for a 1-df test at 0.05.
ALT_NONCENTRALITY = 9.0

FORMULAS: dict[str, str] = {
    "smooth": "y ~ s(x1) + s(x2)",
    "linear": "y ~ s(x1) + x2",
    "factor": "y ~ s(x1) + g",
    "ti": "y ~ s(x1) + s(x2) + ti(x1, x2)",
    "re": "y ~ s(x1) + group(g)",
    "concurvity": "y ~ s(x1) + s(x2)",
}

GAMFIT_FAMILY = {
    "gaussian": "gaussian",
    "binomial": "binomial",
    "poisson": "poisson",
    "gamma": "gamma",
    "negbin": "negative-binomial",
}
# pyGAM (distribution, link); negative binomial has no pyGAM counterpart.
PYGAM_FAMILY = {
    "gaussian": ("normal", "identity"),
    "binomial": ("binomial", "logit"),
    "poisson": ("poisson", "log"),
    "gamma": ("gamma", "log"),
}
# pyGAM term structure (tested term is index 1). ti and random effects have no
# pyGAM counterpart: te() carries the main effects, and pyGAM has no random
# effects.
PYGAM_TERMS = {
    "smooth": ("s", "s"),
    "linear": ("s", "l"),
    "factor": ("s", "f"),
    "concurvity": ("s", "s"),
}


def expected_surfaces(lib: str, family: str, null: str) -> tuple[str, ...]:
    """The p-value surfaces ``lib`` has for the tested term of this cell."""
    if lib == "gamfit":
        if null == "linear":
            return ("coef",)
        # A factor or random effect is a penalized term, so both of gamfit's
        # smooth-term surfaces are expected to cover it. A surface that has no
        # row for it is recorded as missing, not left out of the table.
        return ("wald", "lr")
    if family not in PYGAM_FAMILY or null not in PYGAM_TERMS:
        return ()
    return ("wald",)


def fisher_information(family: str) -> float:
    """Per-row Fisher information for eta at the baseline mean ``exp(b0)``."""
    b0, _ = BASELINE[family]
    if family == "gaussian":
        return 1.0 / GAUSSIAN_SD**2
    if family == "binomial":
        p = special.expit(b0)
        return float(p * (1.0 - p))
    mu = math.exp(b0)
    if family == "poisson":
        return mu
    if family == "gamma":
        return GAMMA_SHAPE
    return mu / (1.0 + mu / NEGBIN_THETA)


def alt_delta(family: str, n: int) -> float:
    return math.sqrt(ALT_NONCENTRALITY / (n * fisher_information(family)))


def rep_rng(family: str, n: int, null: str, seed: int, hyp: str) -> np.random.Generator:
    """An independent stream per (cell, seed, hypothesis), stable across runs."""
    cell = zlib.crc32(f"{family}/{n}/{null}".encode())
    return np.random.default_rng([seed, cell, HYPOTHESES.index(hyp)])


def make_data(
    family: str, n: int, null: str, seed: int, hyp: str
) -> dict[str, Any]:
    rng = rep_rng(family, n, null, seed, hyp)
    b0, a1 = BASELINE[family]
    delta = 0.0 if hyp == "null" else alt_delta(family, n)
    if null == "concurvity":
        z = rng.standard_normal((n, 2))
        z2 = CONCURVITY_RHO * z[:, 0] + math.sqrt(1 - CONCURVITY_RHO**2) * z[:, 1]
        x1 = special.ndtr(z[:, 0])
        x2 = special.ndtr(z2)
    else:
        x1 = rng.uniform(0.0, 1.0, n)
        x2 = rng.uniform(0.0, 1.0, n)
    eta = b0 + a1 * np.sin(2 * np.pi * x1)
    data: dict[str, Any] = {"x1": x1, "x2": x2}
    if null in ("smooth", "concurvity"):
        t = math.sqrt(2.0) * np.cos(2 * np.pi * x2)
    elif null == "linear":
        t = math.sqrt(12.0) * (x2 - 0.5)
    elif null == "factor":
        codes = rng.integers(0, len(FACTOR_LEVELS), n)
        data["g"] = np.asarray(FACTOR_LEVELS, dtype=object)[codes]
        data["g_code"] = codes.astype(float)
        t = math.sqrt(1.5) * np.array([1.0, -1.0, 0.0])[codes]
    elif null == "ti":
        eta = eta + a1 * np.cos(2 * np.pi * x2)
        t = 2.0 * np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2)
    elif null == "re":
        groups = max(2, n // RE_ROWS_PER_GROUP)
        codes = rng.integers(0, groups, n)
        data["g"] = np.asarray([f"g{c}" for c in codes], dtype=object)
        t = rng.standard_normal(groups)[codes]
    else:
        raise ValueError(f"unknown null {null!r}; expected one of {NULLS}")
    eta = eta + delta * t
    if family == "gaussian":
        y = eta + rng.normal(0.0, GAUSSIAN_SD, n)
    elif family == "binomial":
        y = (rng.uniform(size=n) < special.expit(eta)).astype(float)
    elif family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    elif family == "gamma":
        y = rng.gamma(GAMMA_SHAPE, np.exp(eta) / GAMMA_SHAPE)
    elif family == "negbin":
        mu = np.exp(eta)
        y = rng.negative_binomial(NEGBIN_THETA, NEGBIN_THETA / (NEGBIN_THETA + mu))
        y = y.astype(float)
    else:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    data["y"] = y
    return data


# --- gamfit ------------------------------------------------------------------

# Name of the tested term's row in gamfit's tables (``summary().smooth_terms``,
# ``smooth_significance`` and ``model.term_blocks``), per null structure.
GAMFIT_TARGET = {
    "smooth": "s(x2)",
    "concurvity": "s(x2)",
    "ti": "ti(x1,x2)",
    "factor": "g",
    "re": "g",
    "linear": "x2",
}


def _norm(name: str) -> str:
    return name.replace(" ", "")


def _row(rows: list[dict[str, Any]], target: str) -> dict[str, Any]:
    want = _norm(target)
    for row in rows:
        if _norm(str(row["name"])) == want:
            return row
    raise LookupError(f"no row named {target!r}; rows are {[r['name'] for r in rows]}")


def _coefficient_row(model: Any, summary: Any, target: str) -> dict[str, Any]:
    want = _norm(target)
    blocks = [b for b in model.term_blocks if _norm(b.name) == want]
    if len(blocks) != 1 or blocks[0].end - blocks[0].start != 1:
        raise LookupError(f"no one-column term block named {target!r}: {model.term_blocks}")
    (row,) = (r for r in summary.coefficients if r["index"] == blocks[0].start)
    return dict(row)


def gamfit_pvalues(
    gamfit: Any, family: str, null: str, data: dict[str, Any]
) -> tuple[dict[str, float], dict[str, str], dict[str, Any]]:
    """Fit gamfit and read the tested term's p-value on each surface."""
    table = {k: v for k, v in data.items() if k != "g_code"}
    target = GAMFIT_TARGET[null]
    pvals: dict[str, float] = {}
    missing: dict[str, str] = {}
    extra: dict[str, Any] = {}
    model = gamfit.fit(table, FORMULAS[null], family=GAMFIT_FAMILY[family])
    summary = model.summary()
    for surface in expected_surfaces("gamfit", family, null):
        try:
            if surface == "wald":
                row = _row(list(summary.smooth_terms), target)
                extra["wald_edf"] = row.get("edf")
                p = row.get("p_value")
            elif surface == "lr":
                rows = model.smooth_significance(table)
                row = _row(list(rows), target)
                p = row["p_value_corrected"]
                extra["lr_source"] = row.get("reference_source")
                extra["lr_provenance"] = row.get("correction_provenance")
            else:
                row = _coefficient_row(model, summary, target)
                if row.get("p_value") is not None:
                    p = row["p_value"]
                    extra["coef_source"] = "reported"
                else:
                    se = row.get("std_error")
                    if se is None or not se > 0:
                        raise LookupError(f"coefficient row has std_error={se!r}")
                    p = 2.0 * stats.norm.sf(abs(row["estimate"]) / se)
                    extra["coef_source"] = "z_from_std_error"
        except Exception as exc:  # recorded, never dropped
            missing[surface] = f"{type(exc).__name__}: {exc}"[:500]
            continue
        if p is None or not math.isfinite(float(p)):
            missing[surface] = f"p_value={p!r}"
        else:
            pvals[surface] = float(p)
    return pvals, missing, extra


# --- pyGAM -------------------------------------------------------------------


def pygam_pvalue(pygam: Any, family: str, null: str, data: dict[str, Any], gs: bool) -> float:
    kinds = PYGAM_TERMS[null]
    col2 = data["g_code"] if null == "factor" else data["x2"]
    X = np.column_stack([data["x1"], col2])
    terms = getattr(pygam, kinds[0])(0) + getattr(pygam, kinds[1])(1)
    dist, link = PYGAM_FAMILY[family]
    if family == "gaussian":
        g = pygam.LinearGAM(terms)
    else:
        g = pygam.GAM(terms, distribution=dist, link=link)
    if gs:
        g.gridsearch(X, data["y"], progress=False)
    else:
        g.fit(X, data["y"])
    return float(g.statistics_["p_values"][1])


# --- driver ------------------------------------------------------------------


def run_rep(
    family: str, n: int, null: str, seed: int, libs: tuple[str, ...], mods: dict[str, Any]
) -> dict[str, Any]:
    t0 = time.process_time()
    rec: dict[str, Any] = {
        "expected_surfaces": {
            lib: list(expected_surfaces(lib, family, null)) for lib in libs
        },
        "alt_delta": alt_delta(family, n),
        "p": {},
        "missing": {},
        "errors": {},
        "extra": {},
    }
    for hyp in HYPOTHESES:
        data = make_data(family, n, null, seed, hyp)
        p: dict[str, float] = {}
        for lib in libs:
            surfaces = expected_surfaces(lib, family, null)
            if not surfaces:
                continue
            try:
                if lib == "gamfit":
                    got, miss, extra = gamfit_pvalues(mods[lib], family, null, data)
                    for s, v in got.items():
                        p[f"{lib}.{s}"] = v
                    for s, why in miss.items():
                        rec["missing"][f"{hyp}.{lib}.{s}"] = why
                    for k, v in extra.items():
                        rec["extra"][f"{hyp}.{k}"] = v
                else:
                    p[f"{lib}.wald"] = pygam_pvalue(
                        mods[lib], family, null, data, gs=lib == "pygam_gs"
                    )
            except Exception:
                rec["errors"][f"{hyp}.{lib}"] = traceback.format_exc(limit=4)[-2000:]
        rec["p"][hyp] = p
    rec["cpu_s"] = time.process_time() - t0
    rec["status"] = "error" if rec["errors"] else "ok"
    return rec


def main(argv: list[str]) -> int:
    if len(argv) != 6:
        print(__doc__, file=sys.stderr)
        return 2
    family, n, null = argv[0], int(argv[1]), argv[2]
    start, stop = int(argv[3]), int(argv[4])
    libs = tuple(argv[5].split(","))
    if family not in FAMILIES or null not in NULLS or set(libs) - set(LIBS):
        print(__doc__, file=sys.stderr)
        return 2
    mods: dict[str, Any] = {}
    versions: dict[str, str] = {}
    for lib in libs:
        name = "gamfit" if lib == "gamfit" else "pygam"
        mods[lib] = importlib.import_module(name)
        versions[lib] = str(mods[lib].__version__)
    for seed in range(start, stop):
        rec = run_rep(family, n, null, seed, libs, mods)
        rec.update(family=family, n=n, null=null, seed=seed, lib_versions=versions)
        print("RESULT " + json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
