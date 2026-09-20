"""Re-test the gam#1680 rationale for the 12-function default cap on the current solver.
Design (tests/regressions/misc/bug_hunt_1680_near_collinear_additive_recovery.rs):
n=120, x ~ U(-2,2)^4, x2,x3 = 0.985 x1 + sqrt(1-.985^2) U(-2,2) (collinear) or independent,
y = sin(1.5 x1) + 0.25 x4^2 + N(0, .3^2). Truth RMSE on 600 fresh points; default vs k=16/24."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[v] = "1"
import warnings, numpy as np, gamfit
warnings.simplefilter("ignore")
RHO = 0.985
def gen(n, seed, collinear):
    r = np.random.default_rng(seed)
    u = lambda: r.uniform(-2, 2, n)
    x1 = u(); c = np.sqrt(1 - RHO**2)
    x2 = RHO * x1 + c * u() if collinear else u()
    x3 = RHO * x1 + c * u() if collinear else u()
    x4 = u(); t = np.sin(1.5 * x1) + 0.25 * x4**2
    return {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": t + 0.3 * r.normal(size=n)}, t
for collinear in (True, False):
    test, tt = gen(600, 99, collinear)
    tdat = {k: v for k, v in test.items() if k != "y"}
    for k in (None, 16, 24):
        opt = "" if k is None else f", k={k}"
        f = "y ~ " + " + ".join(f"s(x{j}{opt})" for j in range(1, 5))
        rm = []
        for seed in range(4):
            tr, _ = gen(120, seed, collinear)
            try:
                m = gamfit.fit(tr, f)
                rm.append(np.sqrt(np.mean((m.predict(tdat) - tt) ** 2)))
            except Exception as e:
                rm.append(np.nan); print("  fail", type(e).__name__, str(e)[:120])
        print(f"collinear={collinear!s:5} k={k!s:4} truth RMSE per seed {np.round(rm,3)} mean {np.nanmean(rm):.3f}", flush=True)
