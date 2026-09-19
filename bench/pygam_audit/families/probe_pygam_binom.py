"""pyGAM binomial trials: scalar levels only? per-row levels? proportions + weights?"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from pygam import GAM, LogisticGAM, s
from pygam.distributions import BinomialDist
rng = np.random.default_rng(0); n = 600
x = rng.uniform(0, 1, (n, 1)); p = 1/(1+np.exp(-np.sin(2*np.pi*x[:,0])))
tr = rng.integers(1, 21, n); cnt = rng.binomial(tr, p)
for label, fn in [
  ("levels=array (per-row trials)", lambda: GAM(s(0), distribution=BinomialDist(levels=tr), link="logit").fit(x, cnt)),
  ("levels=20 scalar, counts", lambda: GAM(s(0), distribution=BinomialDist(levels=20), link="logit").fit(x, cnt)),
  ("LogisticGAM prop + weights", lambda: LogisticGAM(s(0)).fit(x, cnt/tr, weights=tr)),
]:
    try:
        m = fn(); g = np.linspace(.05,.95,50)[:,None]
        mu = m.predict_mu(g); lev = m.distribution.levels
        if np.ndim(lev) == 0 and lev != 1: mu = mu/lev
        tru = 1/(1+np.exp(-np.sin(2*np.pi*g[:,0])))
        print("OK  ", label, "rmse", round(float(np.sqrt(np.mean((np.ravel(mu)[:50]-tru)**2))),4))
    except Exception as e:
        print("FAIL", label, type(e).__name__, str(e)[:200])
