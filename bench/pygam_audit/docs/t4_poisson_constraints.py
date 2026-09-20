"""pyGAM tour: PoissonGAM histogram smoothing (faithful), constraints (hepatitis-like / trees), te() interaction."""
import time, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gamfit
from pygam import PoissonGAM, LinearGAM, s, te

fig, ax = plt.subplots(2, 3, figsize=(15, 8))

# ---- Poisson: faithful eruptions histogram counts (pyGAM tour "PoissonGAM") ----
fa = pd.read_csv("data/faithful.csv")
y, edges = np.histogram(fa.eruptions, bins=200)
x = (edges[:-1] + edges[1:]) / 2
t = time.time(); pg = PoissonGAM().gridsearch(x[:, None], y, progress=False); print("pygam poisson s", round(time.time() - t, 2))
ax[0, 0].bar(x, y, width=x[1] - x[0], color="lightgray"); ax[0, 0].plot(x, pg.predict(x[:, None]), 'r')
ax[0, 0].set_title("pyGAM PoissonGAM")
D = {"x": x, "y": y.astype(float)}
t = time.time(); m = gamfit.fit(D, "y ~ s(x)", family="poisson"); print("gamfit poisson s", round(time.time() - t, 2), m.family_name)
p = m.predict({"x": x}, interval=0.95, return_type="dict")
ax[0, 1].bar(x, y, width=x[1] - x[0], color="lightgray"); ax[0, 1].plot(x, p["posterior_mean"], 'r')
ax[0, 1].fill_between(x, p["posterior_mean_lower"], p["posterior_mean_upper"], alpha=.4)
ax[0, 1].set_title("gamfit poisson s(x)")
ma = gamfit.fit(D, "y ~ s(x)")  # family auto on integer counts?
print("family auto on integer counts ->", ma.family_name)

# ---- Monotone constraint: pyGAM s(0, constraints='monotonic_inc') ----
rng = np.random.default_rng(1)
xm = np.sort(rng.uniform(0, 10, 300)); ym = np.log1p(xm) + 0.4 * np.sin(2 * xm) * 0 + rng.normal(0, .4, 300)
t = time.time(); pc = LinearGAM(s(0, constraints="monotonic_inc")).fit(xm[:, None], ym); print("pygam monotone s", round(time.time() - t, 2))
t = time.time(); mc = gamfit.fit({"x": xm, "y": ym}, "y ~ s(x, shape=monotone_increasing)"); print("gamfit monotone s", round(time.time() - t, 2))
pm = mc.predict({"x": xm}, interval=0.95, return_type="dict")
print("gamfit monotone interval columns:", list(pm.keys()))
ax[0, 2].scatter(xm, ym, s=4, c="gray"); ax[0, 2].plot(xm, pc.predict(xm[:, None]), 'g', label="pyGAM")
ax[0, 2].plot(xm, pm["posterior_mean"], 'r', label="gamfit")
if "posterior_mean_lower" in pm: ax[0, 2].fill_between(xm, pm["posterior_mean_lower"], pm["posterior_mean_upper"], alpha=.3, color="r")
ax[0, 2].legend(); ax[0, 2].set_title("monotone increasing")
print("gamfit monotone diffs >= 0:", bool(np.all(np.diff(pm["posterior_mean"]) >= -1e-9)))

# ---- te() interaction (pyGAM tour: LinearGAM(s(0)+s(1)+te(0,1)); plot 3D surface) ----
n = 1500
a, b = rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)
z = np.sin(a) * np.cos(b) + 0.5 * a * b + rng.normal(0, .3, n)
t = time.time(); pt = LinearGAM(te(0, 1)).fit(np.c_[a, b], z); print("pygam te s", round(time.time() - t, 2))
XX = pt.generate_X_grid(term=0, meshgrid=True)
Z = pt.partial_dependence(term=0, X=XX, meshgrid=True)
ax[1, 0].contourf(XX[0], XX[1], Z, 20); ax[1, 0].set_title("pyGAM te(0,1) partial dependence")
t = time.time(); mt = gamfit.fit({"a": a, "b": b, "z": z}, "z ~ te(a, b)"); print("gamfit te s", round(time.time() - t, 2))
print("te term blocks:", [bb.name for bb in mt.term_blocks])
g1, g2 = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
G = np.c_[g1.ravel(), g2.ravel()]
try:
    r = mt.partial_dependence("te(a, b)", {"a": a, "b": b, "z": z}, grid=G)
    ax[1, 1].contourf(g1, g2, r["predicted"].reshape(g1.shape), 20); ax[1, 1].set_title("gamfit te(a,b) partial_dependence")
    ax[1, 2].contourf(g1, g2, r["standard_error"].reshape(g1.shape), 20); ax[1, 2].set_title("gamfit te SE")
except Exception as e:
    print("te partial_dependence error:", type(e).__name__, str(e)[:300])
fig.tight_layout(); fig.savefig("t4_poisson_constraints_te.png", dpi=65)
