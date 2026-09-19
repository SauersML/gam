import warnings

import numpy as np

import gamfit

warnings.simplefilter("ignore")
rng = np.random.default_rng(0)
n = 300
x = rng.uniform(0, 1, n)

print("== offset column that is all-zero at fit, then log(2) at predict")
y = rng.poisson(np.exp(0.5 + np.sin(6 * x))).astype(float)
d = {"x": x, "y": y, "off": np.zeros(n)}
m = gamfit.fit(d, "y ~ s(x)", family="poisson", offset="off")
g0 = {"x": np.array([0.2, 0.5]), "off": np.zeros(2)}
g1 = {"x": np.array([0.2, 0.5]), "off": np.full(2, np.log(2.0))}
try:
    print("ratio", m.predict(g1) / m.predict(g0))
except Exception as e:
    print("ERR", type(e).__name__, str(e)[:200])
print("== offset column with values {0, log 2} at fit (not 0/1) -> predict log 3")
d2 = dict(d, off=np.where(rng.uniform(size=n) > 0.5, np.log(2.0), 0.0))
m2 = gamfit.fit(d2, "y ~ s(x)", family="poisson", offset="off")
g3 = {"x": np.array([0.2, 0.5]), "off": np.full(2, np.log(3.0))}
print("ratio", m2.predict(g3) / m2.predict(g0))
print("== offset column {0,1} at fit -> predict 0.5")
d3 = dict(d, off=(rng.uniform(size=n) > 0.5).astype(float))
m3 = gamfit.fit(d3, "y ~ s(x)", family="poisson", offset="off")
try:
    print("ratio", m3.predict({"x": g0["x"], "off": np.full(2, 0.5)}) / m3.predict(g0), "expect", np.exp(0.5))
except Exception as e:
    print("ERR", type(e).__name__, str(e)[:200])

print("== expectile_tau with default family")
yg = np.sin(6 * x) + rng.normal(0, 0.3, n)
dg = {"x": x, "y": yg}
gg = {"x": np.linspace(0.05, 0.95, 5)}
base = gamfit.fit(dg, "y ~ s(x)").predict(gg)
for kw in [dict(expectile_tau=0.9), dict(expectile_tau=1.5), dict(family="gaussian", expectile_tau=0.9),
           dict(family="expectile", expectile_tau=0.9), dict(family="expectile(0.9)"),
           dict(family="expectile", expectile_tau=1.5)]:
    try:
        p = gamfit.fit(dg, "y ~ s(x)", **kw).predict(gg)
        print(kw, "max|p-base|", float(np.max(np.abs(p - base))), "mean(p>base)", float(np.mean(p > base)))
    except Exception as e:
        print(kw, "ERR", type(e).__name__, str(e)[:160])
