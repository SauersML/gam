"""Inverse Gaussian scale and Gamma inverse link: gamfit vs pyGAM vs statsmodels.

    python -m bench.pygam_compare.inverse_gaussian_scale [--n N] [--reps R]

pyGAM stores every family's scale as ``sqrt(phi)`` (``pygam.py``: ``scale =
phi(...) ** 0.5``). ``InvGaussDist.log_pdf`` then evaluates
``scipy.stats.invgauss.logpdf(y, mu, scale=scale / w)``, which is the IG law
with mean ``mu * scale`` and shape ``scale``: the log-likelihood is taken at
the wrong mean and at shape ``sqrt(phi)`` instead of ``1/phi``, and the
prediction variance adds the constant ``scale**2`` instead of ``phi * mu^3``.
This script simulates ``y ~ IG(mu(x), phi)`` with a known ``phi`` and prints

* gamfit's estimated dispersion ``phi_hat`` (the MLE, ``V = phi mu^3``);
* pyGAM's ``statistics_['scale']`` and its square, with and without gridsearch;
* the statsmodels GLM dispersion (deviance / residual df) on the parametric
  no-smooth limit, where the gamfit coefficients must equal the GLM MLE.

It also fits the Gamma inverse link, which pyGAM accepts but whose
gridsearch fails on ordinary data (audit families.md section 2), and compares
the parametric gamfit fit against the statsmodels GLM.

Bench-only dependencies: ``pygam`` (bench/pygam_compare/requirements.txt) and
``statsmodels``.
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np

import gamfit


def _dispersion(model) -> float:
    def find(obj):
        if isinstance(obj, dict):
            if "EstimatedDispersion" in obj and isinstance(obj["EstimatedDispersion"], dict):
                return float(obj["EstimatedDispersion"]["phi"])
            for value in obj.values():
                found = find(value)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for value in obj:
                found = find(value)
                if found is not None:
                    return found
        return None

    return find(json.loads(model.dumps()))


def _ig_loglik(y: np.ndarray, mu: np.ndarray, phi: float) -> float:
    return float(
        np.sum(-0.5 * np.log(2.0 * np.pi * phi * y**3) - (y - mu) ** 2 / (2.0 * phi * mu**2 * y))
    )


def _smooth_rep(rng: np.random.Generator, n: int, phi: float) -> dict[str, float]:
    import pygam

    x = rng.uniform(0.0, 1.0, n)
    mu = np.exp(0.3 + 0.5 * np.sin(2.0 * np.pi * x))
    y = rng.wald(mu, 1.0 / phi)
    row: dict[str, float] = {"phi_true": phi}

    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="inverse-gaussian", link="log")
    fitted = np.asarray(model.predict({"x": x}), dtype=float).reshape(-1)
    row["gamfit_phi_hat"] = _dispersion(model)
    row["gamfit_rmse"] = float(np.sqrt(np.mean((fitted - mu) ** 2)))

    X = x[:, None]
    for label, gridsearch in (("pygam", False), ("pygam_gs", True)):
        gam = pygam.InvGaussGAM(pygam.s(0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if gridsearch:
                gam.gridsearch(X, y, progress=False)
            else:
                gam.fit(X, y)
        scale = float(gam.statistics_["scale"])
        mu_hat = gam.predict_mu(X)
        row[f"{label}_scale"] = scale
        row[f"{label}_scale_squared"] = scale * scale
        row[f"{label}_rmse"] = float(np.sqrt(np.mean((mu_hat - mu) ** 2)))
        # pyGAM's own log-likelihood against the IG log-likelihood of its
        # fitted mean at its own (squared) dispersion.
        row[f"{label}_loglik_reported"] = float(gam.statistics_["loglikelihood"])
        row[f"{label}_loglik_correct"] = _ig_loglik(y, mu_hat, scale * scale)
    return row


def _glm_rep(rng: np.random.Generator, n: int, family: str, link: str) -> dict[str, float]:
    import statsmodels.api as sm

    x = rng.uniform(0.0, 1.0, n)
    eta = 1.0 + 0.8 * x
    phi = 0.2
    if family == "inverse-gaussian":
        mu = eta**-0.5
        y = rng.wald(mu, 1.0 / phi)
        sm_family = sm.families.InverseGaussian(sm.families.links.InverseSquared())
    else:
        mu = 1.0 / eta
        y = rng.gamma(1.0 / phi, mu * phi)
        sm_family = sm.families.Gamma(sm.families.links.InversePower())
    model = gamfit.fit({"x": x, "y": y}, "y ~ linear(x, double_penalty=false)", family=family, link=link)
    ours = np.array([row["estimate"] for row in model.summary().coefficients])
    reference = sm.GLM(y, sm.add_constant(x), family=sm_family).fit(scale="dev", tol=1e-14)
    row = {
        "family": family,
        "link": link,
        "max_abs_coef_diff": float(np.max(np.abs(ours - reference.params))),
        "statsmodels_scale_dev_over_df": float(reference.scale),
    }
    if family == "inverse-gaussian":
        row["gamfit_phi_hat"] = _dispersion(model)
        row["phi_true"] = phi
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--phi", type=float, default=0.3)
    args = parser.parse_args()
    rng = np.random.default_rng(20260919)
    for rep in range(args.reps):
        print(json.dumps({"rep": rep, "kind": "ig_smooth", **_smooth_rep(rng, args.n, args.phi)}))
    for family, link in (("inverse-gaussian", "inverse-squared"), ("gamma", "inverse")):
        print(json.dumps({"kind": "glm_limit", **_glm_rep(rng, args.n, family, link)}))


if __name__ == "__main__":
    main()
