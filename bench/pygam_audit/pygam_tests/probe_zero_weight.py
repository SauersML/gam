"""Is a zero prior weight inert in the public Gaussian formula path?
Decisive check: two fits that differ ONLY in the response of zero-weight rows must agree exactly."""
import warnings

import numpy as np

import gamfit

warnings.simplefilter("ignore")
rng = np.random.default_rng(8)
n = 300
x = rng.uniform(0, 1, n)
y = np.sin(6 * x) + rng.normal(0, 0.3, n)
w = (rng.uniform(size=n) > 0.3).astype(float)
w[np.argmin(x)] = 1.0
w[np.argmax(x)] = 1.0
g = {"x": np.linspace(0.05, 0.95, 50)}
f = "y ~ s(x, knot_placement=uniform)"
for fam in ["gaussian", "poisson"]:
    if fam == "poisson":
        yy = rng.poisson(np.exp(0.5 + np.sin(6 * x))).astype(float)
        junk = yy.copy(); junk[w == 0] = 40.0
    else:
        yy = y
        junk = yy.copy(); junk[w == 0] = 1e3
    ma = gamfit.fit({"x": x, "y": yy, "w": w}, f, family=fam, weights="w")
    mb = gamfit.fit({"x": x, "y": junk, "w": w}, f, family=fam, weights="w")
    keep = w > 0
    md = gamfit.fit({"x": x[keep], "y": yy[keep]}, f, family=fam)
    sa, sb, sd = ma.summary(), mb.summary(), md.summary()
    print(fam, "lambda a/b/del", sa.lambdas, sb.lambdas, sd.lambdas)
    print(fam, "edf a/b/del", sa.edf_total, sb.edf_total, sd.edf_total, "n_obs", sa.n_obs, sd.n_obs)
    print(fam, "max|pred(y) - pred(junk y)|", float(np.max(np.abs(ma.predict(g) - mb.predict(g)))))
    print(fam, "max|pred(w) - pred(deleted)|", float(np.max(np.abs(ma.predict(g) - md.predict(g)))))
