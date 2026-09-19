"""Head-to-head fit quality: gamfit 0.1.267 vs pyGAM 0.12.0 per response family.

Metric: RMSE of the fitted mean (response scale) against the known truth on a
held-out grid, 95% interval coverage of the true mean, wall time.  pyGAM is run
both at its default (lam=0.6 fixed) and with its recommended gridsearch.
Usage: python compare_fits.py [family ...] [--reps R]
"""
import sys, time, warnings, json
import numpy as np
from scipy import optimize, stats
import gamfit
import pygam
from pygam import LinearGAM, LogisticGAM, PoissonGAM, GammaGAM, InvGaussGAM, ExpectileGAM, GAM, s

warnings.filterwarnings("ignore")

def f1(x):
    return np.sin(2 * np.pi * x)

def f2(x):
    return 0.8 * (x - 0.5) ** 2 * 4 - 0.27

def design(rng, n):
    return rng.uniform(0, 1, n), rng.uniform(0, 1, n)

GRID = np.linspace(0.02, 0.98, 60)
GX1, GX2 = np.meshgrid(GRID, GRID[::7])
GX1, GX2 = GX1.ravel(), GX2.ravel()


def gam_fit_predict(train, family, test, fit_kw=None, test_extra=None):
    fit_kw = fit_kw or {}
    t0 = time.perf_counter()
    m = gamfit.fit(train, "y ~ s(x1) + s(x2)", family=family, **fit_kw)
    tt = time.perf_counter() - t0
    td = dict(test)
    if test_extra:
        td.update(test_extra)
    out = m.predict(td, interval=0.95, return_type="dict")
    return (np.asarray(out["posterior_mean"]), np.asarray(out["posterior_mean_lower"]),
            np.asarray(out["posterior_mean_upper"]), tt)


def pygam_fit_predict(model, X, y, Xt, gridsearch, fit_kw=None, pred_kw=None):
    fit_kw = fit_kw or {}
    pred_kw = pred_kw or {}
    t0 = time.perf_counter()
    if gridsearch:
        model.gridsearch(X, y, progress=False, **fit_kw)
    else:
        model.fit(X, y, **fit_kw)
    tt = time.perf_counter() - t0
    mu = model.predict(Xt, **pred_kw) if pred_kw else model.predict_mu(Xt)
    try:
        ci = model.confidence_intervals(Xt, width=0.95)
        lo, hi = ci[:, 0], ci[:, 1]
        if pred_kw and "exposure" in pred_kw:
            lo, hi = lo * pred_kw["exposure"], hi * pred_kw["exposure"]
    except Exception:
        lo = hi = np.full_like(mu, np.nan)
    return mu, lo, hi, tt


def std_normal_expectile(tau):
    # e solves tau * E[(Z-e)+] = (1-tau) * E[(e-Z)+]
    def g(e):
        up = stats.norm.pdf(e) - e * (1 - stats.norm.cdf(e))  # E[(Z-e)+]
        dn = e * stats.norm.cdf(e) + stats.norm.pdf(e)  # E[(e-Z)+]
        return tau * up - (1 - tau) * dn
    return optimize.brentq(g, -10, 10)


def make(family, rng, n):
    x1, x2 = design(rng, n)
    eta = f1(x1) + f2(x2)
    geta = f1(GX1) + f2(GX2)
    extra_train, extra_test, pyfit, pypred = {}, {}, {}, {}
    if family == "gaussian":
        y = eta + rng.normal(0, 0.5, n); truth = geta
    elif family == "binomial":
        y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float); truth = 1 / (1 + np.exp(-geta))
    elif family == "binomial_trials":
        trials = rng.integers(1, 21, n)
        cnt = rng.binomial(trials, 1 / (1 + np.exp(-eta)))
        y = cnt / trials; truth = 1 / (1 + np.exp(-geta))
        extra_train = {"w": trials.astype(float)}
        pyfit = {"weights": trials.astype(float)}
    elif family == "poisson_exposure":
        e = rng.uniform(0.2, 5.0, n)
        y = rng.poisson(e * np.exp(0.5 * eta)).astype(float)
        truth = np.exp(0.5 * geta)  # rate at unit exposure
        extra_train = {"loge": np.log(e)}
        extra_test = {"loge": np.zeros_like(GX1)}
        pyfit = {"exposure": e}
        pypred = {"exposure": np.ones_like(GX1)}
    elif family == "gamma":
        mu = np.exp(0.5 * eta + 1); shape = 3.0
        y = rng.gamma(shape, mu / shape); truth = np.exp(0.5 * geta + 1)
    elif family == "gamma_inverse":
        mu = 1 / (0.6 + 0.25 * eta); shape = 3.0
        y = rng.gamma(shape, mu / shape); truth = 1 / (0.6 + 0.25 * geta)
    elif family == "inv_gauss":
        mu = np.exp(0.5 * eta + 1); lam = 10.0
        y = rng.wald(mu, lam); truth = np.exp(0.5 * geta + 1)
    elif family == "expectile90":
        m = eta; sd = 0.2 + 0.8 * x1
        y = m + sd * rng.normal(0, 1, n)
        truth = geta + (0.2 + 0.8 * GX1) * std_normal_expectile(0.9)
    else:
        raise ValueError(family)
    train = {"x1": x1, "x2": x2, "y": y, **extra_train}
    test = {"x1": GX1, "x2": GX2, **extra_test}
    return train, test, truth, pyfit, pypred


def gam_spec(family):
    return {
        "gaussian": ("gaussian", {}),
        "binomial": ("binomial", {}),
        "binomial_trials": ("binomial", {"weights": "w"}),
        "poisson_exposure": ("poisson", {"offset": "loge"}),
        "gamma": ("gamma", {}),
        "gamma_inverse": ("gamma", {}),  # gamfit only has gamma(log): misspecified-link control
        "inv_gauss": (None, {}),  # unsupported
        "expectile90": ("expectile", {"expectile_tau": 0.9}),
    }[family]


def py_model(family):
    terms = s(0) + s(1)
    return {
        "gaussian": lambda: LinearGAM(terms),
        "binomial": lambda: LogisticGAM(terms),
        "binomial_trials": lambda: LogisticGAM(terms),
        "poisson_exposure": lambda: PoissonGAM(terms),
        "gamma": lambda: GammaGAM(terms),
        "gamma_inverse": lambda: GAM(terms, distribution="gamma", link="inverse"),
        "inv_gauss": lambda: InvGaussGAM(terms),
        "expectile90": lambda: ExpectileGAM(terms, expectile=0.9),
    }[family]


def run(family, reps, n):
    rows = {"gamfit": [], "pygam_default": [], "pygam_gridsearch": []}
    errors = {"gamfit": [], "pygam_default": [], "pygam_gridsearch": []}
    for r in range(reps):
        rng = np.random.default_rng(1000 + r)
        train, test, truth, pyfit, pypred = make(family, rng, n)
        X = np.c_[train["x1"], train["x2"]]
        Xt = np.c_[test["x1"], test["x2"]]
        fam, kw = gam_spec(family)
        if fam is not None:
            try:
                mu, lo, hi, tt = gam_fit_predict(train, fam, test, kw)
                rows["gamfit"].append((np.sqrt(np.mean((mu - truth) ** 2)),
                                       np.mean((lo <= truth) & (truth <= hi)), tt))
            except Exception as e:
                errors["gamfit"].append(f"{type(e).__name__}: {str(e)[:160]}")
        else:
            errors["gamfit"].append("unsupported family")
        for label, gs in (("pygam_default", False), ("pygam_gridsearch", True)):
            try:
                mu, lo, hi, tt = pygam_fit_predict(py_model(family)(), X, train["y"], Xt, gs, pyfit, pypred)
                rows[label].append((np.sqrt(np.mean((mu - truth) ** 2)),
                                    np.mean((lo <= truth) & (truth <= hi)), tt))
            except Exception as e:
                errors[label].append(f"{type(e).__name__}: {str(e)[:160]}")
    print(f"\n== {family} (n={n}, reps={reps})")
    for k, v in rows.items():
        if v:
            a = np.array(v)
            print(f"  {k:17s} RMSE {a[:,0].mean():.4f} (sd {a[:,0].std():.4f})  cover95 {np.nanmean(a[:,1]):.3f}  time {a[:,2].mean():.2f}s  ok={len(v)}")
        if errors[k]:
            print(f"  {k:17s} ERRORS ({len(errors[k])}): {errors[k][0]}")
    sys.stdout.flush()


if __name__ == "__main__":
    args = sys.argv[1:]
    reps = 6
    if "--reps" in args:
        i = args.index("--reps"); reps = int(args[i + 1]); del args[i:i + 2]
    fams = args or ["gaussian", "binomial", "binomial_trials", "poisson_exposure", "gamma",
                    "gamma_inverse", "inv_gauss", "expectile90"]
    for fam in fams:
        run(fam, reps, 600 if fam != "binomial" else 1500)
