"""Full conformal for non-Gaussian families through the public predict route.

``model.predict(interval="conformal", training_data=...)`` and ``gam predict
--conformal --training-data`` used to refuse every family except
Gaussian-identity, so a binomial, Poisson, negative-binomial or Gamma GAM had
no distribution-free prediction set, and any fit with an offset refused too.
The route now builds the full-conformal set of the augmented penalized GLM at
the fitted smoothing parameters (``gam_models::inference::full_conformal_glm``):

* binomial: exact enumeration of ``{0, 1}``;
* Poisson / negative binomial: enumeration up to a data-derived tail beyond
  which no count can conform;
* Gamma: a certified walk on the Pearson score ``|y / mu - 1|``.

Discrete ties are randomized with a seed drawn from the data, so the set is
exact (coverage ``1 - alpha`` on average) rather than conservative. A
prior-weighted fit has no exchangeable augmented problem, so it refuses with a
typed error that names split conformal. GLM rows report
``frozen_rho_certified`` 0 because only the Gaussian route certifies the frozen
smoothing parameters.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gamfit
from gamfit.errors import FitConvergenceError, InvalidConfigurationError

ALPHA = 0.1
# alpha * (n + 1) is an integer, so the conformal quantile has no rounding
# slack and marginal coverage is exactly 1 - alpha for an exact set.
N_TRAIN = 59
M_TEST = 8
REPS = 400


def _eta(x: np.ndarray) -> np.ndarray:
    return 0.8 * np.sin(2.0 * np.pi * x)


def _draw(family: str, rng: np.random.Generator, n: int) -> pd.DataFrame:
    x = rng.uniform(0.0, 1.0, n)
    eta = _eta(x)
    if family == "binomial":
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    elif family == "poisson":
        y = rng.poisson(np.exp(1.0 + eta)).astype(float)
    elif family == "negative_binomial":
        mu = np.exp(1.0 + eta)
        y = rng.negative_binomial(3.0, 3.0 / (3.0 + mu)).astype(float)
    elif family == "gamma":
        y = rng.gamma(4.0, np.exp(eta) / 4.0)
    else:
        raise AssertionError(family)
    return pd.DataFrame({"x": x, "y": y})


FAMILIES = ["binomial", "poisson", "negative_binomial", "gamma"]


@pytest.mark.parametrize("family", FAMILIES)
def test_glm_full_conformal_set_is_a_set_in_the_support(family: str) -> None:
    rng = np.random.default_rng(11)
    train = _draw(family, rng, 120)
    model = gamfit.fit(train, "y ~ s(x, k=8)", family=family)
    test = _draw(family, rng, 25)
    out = model.predict(
        test[["x"]],
        interval="conformal",
        training_data=train,
        conformal_level=1.0 - ALPHA,
        return_type="dict",
    )
    lo = np.asarray(out["posterior_mean_lower"], dtype=float)
    hi = np.asarray(out["posterior_mean_upper"], dtype=float)
    comps = np.asarray(out["conformal_set_components"], dtype=float)
    certified = np.asarray(out["frozen_rho_certified"], dtype=float)
    assert np.all(certified == 0.0), "only the Gaussian route certifies frozen rho"
    assert np.all(comps >= 1), f"{family}: empty set at alpha={ALPHA}"
    assert np.all(lo <= hi)
    if family == "gamma":
        assert np.all(lo >= 0.0)
        assert np.all(np.isfinite(hi))
    else:
        assert np.all(lo == np.round(lo)) and np.all(hi == np.round(hi))
        assert np.all(lo >= 0.0)
    if family == "binomial":
        assert np.all(hi <= 1.0)


def _penalty_model(family: str, rng: np.random.Generator):
    """A fit on an independent draw that supplies the frozen penalty.

    The draw is independent of the labeled and test rows, so redrawing it when
    the smoothing-parameter search refuses to certify an optimum leaves the
    n + 1 augmented rows exchangeable and the coverage unbiased.
    """

    refusals = []
    for _ in range(5):
        try:
            return gamfit.fit(_draw(family, rng, N_TRAIN), "y ~ s(x, k=6)", family=family)
        except FitConvergenceError as err:
            refusals.append(err)
    raise AssertionError(f"{family}: every penalty fit refused: {refusals}")


@pytest.mark.parametrize("family", FAMILIES)
def test_glm_full_conformal_covers_at_the_nominal_level(family: str) -> None:
    """Seeded Monte Carlo: coverage within 2 MCSE of 1 - alpha, both ways.

    Over-coverage past the 1/(n+1) granularity is a bug (a conservative rather
    than exact set); with alpha * (n + 1) integral the granularity is 0, and the
    randomized tie-break makes the discrete sets exact too.

    The model that supplies the frozen penalty is fitted on an independent draw,
    so the penalty is independent of the labeled and test rows and the n + 1
    augmented rows are exchangeable: coverage is then exactly 1 - alpha, which
    tests the set construction itself. With the penalty selected on the labeled
    rows (the usual call) the smoothing step sees the labeled rows but not the
    candidate, which costs O(1/n) coverage; ``frozen_rho_certified`` is 0 for
    every GLM row for that reason, and bench/pygam_audit/conformal_coverage.md
    reports that route's coverage.
    """

    rng = np.random.default_rng(20260919)
    per_rep = []
    fragmented = []
    for _ in range(REPS):
        model = _penalty_model(family, rng)
        train = _draw(family, rng, N_TRAIN)
        test = _draw(family, rng, M_TEST)
        out = model.predict(
            test[["x"]],
            interval="conformal",
            training_data=train,
            conformal_level=1.0 - ALPHA,
            return_type="dict",
        )
        lo = np.asarray(out["posterior_mean_lower"], dtype=float)
        hi = np.asarray(out["posterior_mean_upper"], dtype=float)
        comps = np.asarray(out["conformal_set_components"], dtype=float)
        # The envelope is the set only when it is one interval: a
        # multi-component row counts as covered when y is inside the envelope,
        # which is conservative, so such rows must be rare.
        fragmented.append(float(np.mean(comps > 1)))
        y = test["y"].to_numpy()
        per_rep.append(float(np.mean((y >= lo) & (y <= hi))))
    assert np.mean(fragmented) <= 0.05, f"{family}: sets are mostly fragmented"
    cov = float(np.mean(per_rep))
    mcse = float(np.std(per_rep, ddof=1) / np.sqrt(REPS))
    target = 1.0 - ALPHA
    assert abs(cov - target) <= 2.0 * mcse + 1e-12, (
        f"{family}: full-conformal coverage {cov:.4f} is outside "
        f"{target} +/- 2*MCSE ({2 * mcse:.4f})"
    )


def test_gaussian_offset_shifts_the_set_by_the_test_offset() -> None:
    rng = np.random.default_rng(5)
    n = 90
    x = rng.uniform(0.0, 1.0, n)
    o = rng.normal(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) + o + rng.normal(0.0, 0.3, n)
    train = pd.DataFrame({"x": x, "y": y, "o": o})
    shifted = pd.DataFrame({"x": x, "y": y - o})
    xt = rng.uniform(0.0, 1.0, 10)
    ot = rng.normal(0.0, 1.0, 10)

    with_offset = gamfit.fit(train, "y ~ s(x, k=8)", offset="o")
    without = gamfit.fit(shifted, "y ~ s(x, k=8)")
    a = with_offset.predict(
        pd.DataFrame({"x": xt, "o": ot}),
        interval="conformal",
        training_data=train,
        conformal_level=0.9,
        return_type="dict",
    )
    b = without.predict(
        pd.DataFrame({"x": xt}),
        interval="conformal",
        training_data=shifted,
        conformal_level=0.9,
        return_type="dict",
    )
    for key in ("posterior_mean_lower", "posterior_mean_upper"):
        np.testing.assert_allclose(
            np.asarray(a[key], dtype=float),
            np.asarray(b[key], dtype=float) + ot,
            rtol=1e-6,
            atol=1e-6,
            err_msg=key,
        )


def test_poisson_offset_is_an_exposure_in_the_set() -> None:
    rng = np.random.default_rng(8)
    n = 150
    x = rng.uniform(0.0, 1.0, n)
    exposure = rng.uniform(0.5, 20.0, n)
    y = rng.poisson(exposure * np.exp(0.5 * np.sin(2.0 * np.pi * x))).astype(float)
    train = pd.DataFrame({"x": x, "y": y, "logt": np.log(exposure)})
    model = gamfit.fit(train, "y ~ s(x, k=8)", family="poisson", offset="logt")
    out = model.predict(
        pd.DataFrame({"x": [0.25, 0.25], "logt": np.log([1.0, 40.0])}),
        interval="conformal",
        training_data=train,
        conformal_level=0.9,
        return_type="dict",
    )
    lo = np.asarray(out["posterior_mean_lower"], dtype=float)
    hi = np.asarray(out["posterior_mean_upper"], dtype=float)
    mean = np.asarray(out["mean_plugin"], dtype=float)
    # Forty times the exposure moves the set to ~40x the counts: the offset is
    # in the augmented fit, not dropped.
    assert hi[0] < 10.0, f"unit-exposure set {lo[0]}..{hi[0]} ignores the offset"
    assert lo[1] > hi[0], f"high-exposure set {lo[1]}..{hi[1]} ignores the offset"
    assert lo[1] <= mean[1] <= hi[1]


def test_prior_weights_refuse_and_name_split_conformal() -> None:
    rng = np.random.default_rng(3)
    train = _draw("poisson", rng, 80)
    # Integer weights: a Poisson fit reads them as frequency weights.
    train["w"] = rng.integers(1, 4, len(train)).astype(float)
    model = gamfit.fit(train, "y ~ s(x, k=6)", family="poisson", weights="w")
    with pytest.raises(InvalidConfigurationError, match="calibration="):
        model.predict(train[["x"]].head(3), interval="conformal", training_data=train)


def test_glm_saved_model_carries_no_training_rows() -> None:
    small = gamfit.fit(_draw("poisson", np.random.default_rng(1), 400), "y ~ s(x, k=8)", family="poisson")
    large = gamfit.fit(_draw("poisson", np.random.default_rng(2), 6400), "y ~ s(x, k=8)", family="poisson")
    assert len(large.dumps()) < 1.1 * len(small.dumps())
    # A reloaded model still builds the set from the labeled rows supplied now.
    train = _draw("poisson", np.random.default_rng(1), 400)
    reloaded = gamfit.loads(small.dumps())
    got = reloaded.predict(train[["x"]].head(4), interval="conformal", training_data=train, return_type="dict")
    want = small.predict(train[["x"]].head(4), interval="conformal", training_data=train, return_type="dict")
    for key in ("posterior_mean_lower", "posterior_mean_upper"):
        np.testing.assert_array_equal(np.asarray(got[key]), np.asarray(want[key]))


def _gam_binary() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    for candidate in (
        os.environ.get("GAM_BIN"),
        repo_root / "target" / "release" / "gam",
        repo_root / "target" / "debug" / "gam",
        shutil.which("gam"),
    ):
        if candidate and Path(candidate).exists():
            return str(candidate)
    # No skip: an unbuilt CLI is a real gap in this parity check.
    raise AssertionError(
        "no `gam` CLI binary found (GAM_BIN, target/release/gam, target/debug/gam, PATH)"
    )


def test_cli_full_conformal_matches_python_for_a_poisson_offset_fit(tmp_path: Path) -> None:
    rng = np.random.default_rng(21)
    n = 100
    x = rng.uniform(0.0, 1.0, n)
    logt = np.log(rng.uniform(0.5, 5.0, n))
    y = rng.poisson(np.exp(logt + _eta(x))).astype(float)
    train = pd.DataFrame({"x": x, "y": y, "logt": logt})
    test = pd.DataFrame({"x": rng.uniform(0.0, 1.0, 12), "logt": np.log(rng.uniform(0.5, 5.0, 12))})

    model = gamfit.fit(train, "y ~ s(x, k=8)", family="poisson", offset="logt")
    python = model.predict(
        test,
        interval="conformal",
        training_data=train,
        conformal_level=0.9,
        return_type="dict",
    )

    model_path = tmp_path / "m.gam"
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    out_path = tmp_path / "out.csv"
    model.save(model_path)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    subprocess.run(
        [
            _gam_binary(),
            "predict",
            str(model_path),
            str(test_path),
            "--out",
            str(out_path),
            "--conformal",
            "--level",
            "0.9",
            "--training-data",
            str(train_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with out_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(test)
    for key in (
        "posterior_mean_lower",
        "posterior_mean_upper",
        "conformal_set_components",
        "frozen_rho_certified",
    ):
        cli = np.array([float(r[key]) for r in rows])
        np.testing.assert_array_equal(cli, np.asarray(python[key], dtype=float), err_msg=key)
    cli_mean = np.array([float(r["posterior_mean"]) for r in rows])
    np.testing.assert_allclose(cli_mean, np.asarray(python["posterior_mean"], dtype=float), rtol=1e-9)
