"""pyGAM PoissonGAM.gridsearch(exposure=) double-applies exposure (pygam.py:2054 passes weights positionally into PoissonGAM.fit's `exposure` slot)."""
import numpy as np, warnings
from pygam import PoissonGAM, s
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0); n = 600
x = rng.uniform(0, 1, n); e = rng.uniform(0.2, 5, n)
y = rng.poisson(e * np.exp(0.5 * np.sin(2 * np.pi * x))).astype(float)
X = x[:, None]; g = np.linspace(.05, .95, 50)[:, None]; tru = np.exp(0.5 * np.sin(2 * np.pi * g[:, 0]))
a = PoissonGAM(s(0), lam=1.0).fit(X, y, exposure=e)
b = PoissonGAM(s(0)).gridsearch(X, y, exposure=e, lam=[1.0, 1.0], progress=False)
for name, m in (("fit(lam=1)", a), ("gridsearch(lam=[1,1])", b)):
    p = m.predict(g, exposure=np.ones(50))
    print(f"{name:22s} rmse vs truth {np.sqrt(np.mean((p - tru) ** 2)):.4f}  mean rate {p.mean():.3f} (truth {tru.mean():.3f})")
