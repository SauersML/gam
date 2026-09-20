import time, numpy as np, gamfit
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pygam import LinearGAM, te
rng = np.random.default_rng(1); n = 1500
a, b = rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)
z = np.sin(a) * np.cos(b) + 0.5 * a * b + rng.normal(0, .3, n)
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
t = time.time(); pt = LinearGAM(te(0, 1)).fit(np.c_[a, b], z); print("pygam te s", round(time.time()-t, 2), flush=True)
XX = pt.generate_X_grid(term=0, meshgrid=True); Z = pt.partial_dependence(term=0, X=XX, meshgrid=True)
ax[0].contourf(XX[0], XX[1], Z, 20); ax[0].set_title("pyGAM te(0,1)")
t = time.time(); mt = gamfit.fit({"a": a, "b": b, "z": z}, "z ~ te(a, b)"); print("gamfit te s", round(time.time()-t, 2), flush=True)
print("blocks", [bb.name for bb in mt.term_blocks], flush=True)
try:
    r = mt.partial_dependence(mt.term_blocks[-1].name, {"a": a, "b": b, "z": z})
    print("PD keys", list(r.keys()), "grid shape", np.shape(r["grid"]), "pred shape", np.shape(r["predicted"]), flush=True)
    g = np.asarray(r["grid"]); pr = np.asarray(r["predicted"]); se = np.asarray(r["standard_error"])
    k = int(round(np.sqrt(len(pr))))
    ax[1].tricontourf(g[:, 0], g[:, 1], pr, 20); ax[1].set_title("gamfit te PD")
    ax[2].tricontourf(g[:, 0], g[:, 1], se, 20); ax[2].set_title("gamfit te PD SE")
except Exception as e:
    print("te PD error:", type(e).__name__, str(e)[:300], flush=True)
truth = lambda A, B: np.sin(A)*np.cos(B) + .5*A*B
gg = np.random.default_rng(2).uniform(-2, 2, (2000, 2))
print("rmse vs truth pygam", np.sqrt(np.mean((pt.predict(gg) - truth(gg[:,0], gg[:,1]))**2)))
pm = np.asarray(mt.predict({"a": gg[:,0], "b": gg[:,1]}))
print("rmse vs truth gamfit", np.sqrt(np.mean((pm - truth(gg[:,0], gg[:,1]))**2)), flush=True)
fig.tight_layout(); fig.savefig("t4b_te.png", dpi=65); print("done", flush=True)
