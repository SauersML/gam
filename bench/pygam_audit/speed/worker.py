"""One benchmark measurement in a fresh process.

Usage: worker.py LIB FAMILY N DESIGN SEED
  LIB     gamfit | gamfit_k20 | pygam | pygam_gs
  FAMILY  gaussian | binomial | poisson
  DESIGN  p1 | p5 | p20 | te
Prints one JSON line on stdout.

Phases timed (perf_counter):
  import   : import of the library (numpy already imported)
  fit1     : first (cold) fit in the process  -> "time-to-first-fit" minus import
  fit2     : second fit, same data, same process (warm: caches / page faults paid)
  pred1    : first predict on n fresh rows
  pred2    : second predict
Memory: ru_maxrss (peak RSS of the process, KiB on Linux) sampled after data
generation (baseline) and at the end; peak_delta = end - baseline.
"""
import json
import os
import resource
import sys
import time

import numpy as np

LIB, FAMILY, N, DESIGN, SEED = sys.argv[1], sys.argv[2], int(float(sys.argv[3])), sys.argv[4], int(sys.argv[5])


def rss_peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def make_data(n, design, family, seed):
    rng = np.random.default_rng(seed)
    if design == "te":
        X = rng.uniform(0, 1, (n, 2))
        eta = np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])
    else:
        p = int(design[1:])
        X = rng.uniform(0, 1, (n, p))
        eta = np.zeros(n)
        for j in range(p):
            eta += np.sin(2 * np.pi * X[:, j] + j) / np.sqrt(p)
    if family == "gaussian":
        y = eta + rng.normal(0, 0.5, n)
        mu = eta
    elif family == "binomial":
        mu = 1 / (1 + np.exp(-1.5 * eta))
        y = (rng.uniform(size=n) < mu).astype(float)
    elif family == "poisson":
        mu = np.exp(0.5 + 0.7 * eta)
        y = rng.poisson(mu).astype(float)
    else:
        raise SystemExit(f"bad family {family}")
    return X, y, mu


X, y, mu = make_data(N, DESIGN, FAMILY, SEED)
Xp, _, mup = make_data(N, DESIGN, FAMILY, SEED + 1000)
base_rss = rss_peak_mb()
out = dict(lib=LIB, family=FAMILY, n=N, design=DESIGN, seed=SEED, base_rss_mb=base_rss)

t0 = time.perf_counter(); c0 = time.process_time()
if LIB.startswith("gamfit"):
    import gamfit  # noqa: F401
else:
    import pygam
t1 = time.perf_counter(); c1 = time.process_time()
out["import_s"] = t1 - t0
out["import_cpu_s"] = c1 - c0
after_import_rss = rss_peak_mb()
out["after_import_rss_mb"] = after_import_rss

if LIB.startswith("gamfit"):
    names = [f"x{j}" for j in range(X.shape[1])]
    data = {nm: X[:, j] for j, nm in enumerate(names)}
    data["y"] = y
    pdata = {nm: Xp[:, j] for j, nm in enumerate(names)}
    k = "" if LIB == "gamfit" else (", k=20" if DESIGN != "te" else ", k=10")
    if DESIGN == "te":
        formula = f"y ~ te(x0, x1{k})"
    else:
        formula = "y ~ " + " + ".join(f"s({nm}{k})" for nm in names)

    def do_fit():
        return gamfit.fit(data, formula, family=FAMILY)

    def do_pred(m):
        return np.asarray(m.predict(pdata))
else:
    from pygam import GAM, s, te
    if DESIGN == "te":
        terms = te(0, 1)
    else:
        terms = s(0)
        for j in range(1, X.shape[1]):
            terms = terms + s(j)
    dist, link = dict(gaussian=("normal", "identity"), binomial=("binomial", "logit"),
                      poisson=("poisson", "log"))[FAMILY]

    def do_fit():
        g = GAM(terms, distribution=dist, link=link)
        if LIB == "pygam_gs":
            g.gridsearch(X, y, progress=False)
        else:
            g.fit(X, y)
        return g

    def do_pred(m):
        return np.asarray(m.predict(Xp))

t2 = time.perf_counter(); c2 = time.process_time()
m = do_fit()
t3 = time.perf_counter(); c3 = time.process_time()
out["fit1_s"] = t3 - t2
out["fit1_cpu_s"] = c3 - c2
out["rss_after_fit1_mb"] = rss_peak_mb()
REPEAT = os.environ.get("BENCH_WARM", "1") == "1"
if REPEAT:
    t4 = time.perf_counter(); c4 = time.process_time()
    m = do_fit()
    t5 = time.perf_counter(); c5 = time.process_time()
    out["fit2_s"] = t5 - t4
    out["fit2_cpu_s"] = c5 - c4
t6 = time.perf_counter(); c6 = time.process_time()
pr = do_pred(m)
t7 = time.perf_counter(); c7 = time.process_time()
out["pred1_cpu_s"] = c7 - c6
pr = do_pred(m)
t8 = time.perf_counter()
out["pred1_s"] = t7 - t6
out["pred2_s"] = t8 - t7
out["peak_rss_mb"] = rss_peak_mb()
out["peak_delta_mb"] = out["peak_rss_mb"] - after_import_rss
# sanity: accuracy of fitted mean on the held-out points vs truth
out["rmse_mu"] = float(np.sqrt(np.mean((pr.reshape(-1) - mup) ** 2)))
if LIB.startswith("gamfit"):
    try:
        txt = str(m.summary())
        for line in txt.splitlines():
            if "Outer iterations" in line:
                out["outer_iter"] = int(line.split(":")[1])
            if "Effective dof" in line:
                out["edf"] = float(line.split(":")[1])
            if "Coefficients:" in line:
                out["ncoef"] = int(line.split(":")[1])
    except Exception as e:  # pragma: no cover
        out["summary_err"] = repr(e)
else:
    out["edf"] = float(m.statistics_["edof"])
    out["ncoef"] = int(len(m.coef_))
    out["lam"] = [float(v) for v in np.ravel(m.lam)][:3]
out["cpu_user_s"] = resource.getrusage(resource.RUSAGE_SELF).ru_utime
out["cpu_sys_s"] = resource.getrusage(resource.RUSAGE_SELF).ru_stime
print("RESULT " + json.dumps(out), flush=True)
