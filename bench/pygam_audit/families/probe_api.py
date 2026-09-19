"""Probe which (family, link) cells gamfit 0.1.267 accepts through the Python API."""
import numpy as np, pandas as pd, gamfit, warnings
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
n = 400
x = rng.uniform(0, 1, n)
eta = 0.5 + np.sin(2 * np.pi * x) * 0.4
pos = np.exp(eta)
dfs = {
    "gauss": {k: np.asarray(v) for k, v in ({"x": x, "y": pos + rng.normal(0, 0.2, n)}).items()},
    "pos": {k: np.asarray(v) for k, v in ({"x": x, "y": rng.gamma(5, pos / 5)}).items()},
    "count": {k: np.asarray(v) for k, v in ({"x": x, "y": rng.poisson(pos * 3).astype(float)}).items()},
    "bin": {k: np.asarray(v) for k, v in ({"x": x, "y": rng.binomial(1, 1 / (1 + np.exp(-eta)), n).astype(float)}).items()},
}
cases = [
    ("gauss", dict(family="gaussian")),
    ("gauss", dict(family="gaussian", link="log")),
    ("gauss", dict(family="gaussian(log)")),
    ("gauss", dict(family="gaussian", link="inverse")),
    ("pos", dict(family="gamma")),
    ("pos", dict(family="gamma", link="inverse")),
    ("pos", dict(family="gamma", link="identity")),
    ("pos", dict(family="gamma(inverse)")),
    ("pos", dict(family="inverse-gaussian")),
    ("pos", dict(family="inverse_gaussian")),
    ("pos", dict(family="inv_gauss")),
    ("pos", dict(family="invgauss")),
    ("count", dict(family="poisson")),
    ("count", dict(family="poisson", link="identity")),
    ("count", dict(family="poisson", link="sqrt")),
    ("bin", dict(family="binomial")),
    ("bin", dict(family="binomial", link="probit")),
    ("bin", dict(family="binomial", link="cloglog")),
    ("bin", dict(family="binomial", link="loglog")),
    ("bin", dict(family="binomial", link="cauchit")),
    ("bin", dict(family="binomial", link="log")),
    ("bin", dict(family="binomial(cauchit)")),
    ("bin", dict(family="binomial", link="sas")),
    ("gauss", dict(family="expectile", expectile_tau=0.9)),
    ("gauss", dict(family="expectile(0.9)")),
    ("gauss", dict(family="quantile")),
    ("pos", dict(family="gaussian", link="identity")),
]
for dname, kw in cases:
    try:
        m = gamfit.fit(dfs[dname], "y ~ s(x)", **kw)
        fam = None
        for attr in ("family", "family_name", "likelihood"):
            if hasattr(m, attr):
                try:
                    fam = getattr(m, attr)
                    break
                except Exception:
                    pass
        print(f"OK   {dname:5s} {kw} -> {fam}")
    except Exception as e:
        print(f"FAIL {dname:5s} {kw} -> {type(e).__name__}: {str(e)[:220]}")
