"""Binomial rep 21 (mc.py seed 1021): the fit silently switches to Firth/Jeffreys; is that visible in summary()/predict()?"""
import os, re, time, warnings; os.environ["RAYON_NUM_THREADS"]="1"; warnings.simplefilter("ignore")
import numpy as np, gamfit
from mc import CELLS, f1, f3, inv_link, draw_y
fam, n, b0, a1, a3, sigma = CELLS["binom"]
rep = 21
rng = np.random.default_rng(1000 + rep); X = rng.uniform(0, 1, (n, 3))
y = draw_y(fam, inv_link(fam, b0 + f1(X[:, 0], a1) + f3(X[:, 2], a3)), sigma, rng)
d = dict(x1=X[:, 0], x2=X[:, 1], x3=X[:, 2], y=y)
t0 = time.time(); m = gamfit.fit(d, "y ~ s(x1) + s(x2) + s(x3)", family=fam); print("fit", round(time.time()-t0, 1))
sm = m.summary()
txt = repr(sm) + str(sm) + " ".join(f"{a}={getattr(sm, a)!r}" for a in dir(sm) if not a.startswith("_") and not callable(getattr(sm, a)))
print("summary mentions firth/jeffreys:", bool(re.search("firth|jeffreys", txt, re.I)))
p = m.predict(d, interval=0.95); print("predict keys:", list(p.keys()) if hasattr(p, "keys") else type(p))
print("predict mentions firth:", bool(re.search("firth|jeffreys", repr(p), re.I)))
print("model repr:", repr(m)[:300])
