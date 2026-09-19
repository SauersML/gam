"""pyGAM accepts incoherent (distribution, link) cells silently?"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from pygam import GAM, s
rng = np.random.default_rng(0); n = 500
x = rng.uniform(0, 1, (n, 1)); y = rng.poisson(np.exp(1 + np.sin(2*np.pi*x[:,0]))).astype(float)
yb = rng.binomial(1, 0.3, n).astype(float)
for dist, link, yy in [("poisson","logit",y), ("binomial","log",yb), ("normal","inv_squared",y+1),("poisson","identity",y),("gamma","identity",y+1), ("binomial","identity", yb)]:
    try:
        m = GAM(s(0), distribution=dist, link=link).fit(x, yy)
        mu = m.predict_mu(np.linspace(0,1,200)[:,None])
        print("accepted", dist, link, "mu range", float(mu.min()), float(mu.max()), "converged-ish stats edof", round(m.statistics_["edof"],2))
    except Exception as e:
        print("rejected", dist, link, type(e).__name__, str(e)[:120])
