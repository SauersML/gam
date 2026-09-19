import numpy as np, gamfit
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(1); n = 1500
a, b = rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)
z = np.sin(a) * np.cos(b) + 0.5 * a * b + rng.normal(0, .3, n)
D = {"a": a, "b": b, "z": z}
mt = gamfit.fit(D, "z ~ te(a, b)")
g1, g2 = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
r = mt.partial_dependence("te(a, b)", D, grid=np.c_[g1.ravel(), g2.ravel()])
print({k: (np.shape(v) if hasattr(v, "__len__") and not isinstance(v, str) else v) for k, v in r.items()})
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
c = ax[0].contourf(g1, g2, np.asarray(r["predicted"]).reshape(g1.shape), 20); ax[0].set_title("gamfit te(a,b)")
ax[1].contourf(g1, g2, np.asarray(r["standard_error"]).reshape(g1.shape), 20); ax[1].scatter(a, b, s=1, c="k", alpha=.2); ax[1].set_title("SE")
fig.tight_layout(); fig.savefig("t4c_te_grid.png", dpi=65)
