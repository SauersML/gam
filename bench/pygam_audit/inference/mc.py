"""Monte Carlo coverage / calibration study: gamfit vs pyGAM.

Usage: python mc.py <cell> <nrep> <nworkers>
cells: gauss, gauss_small, binom, pois
Writes results/<cell>.json with per-replicate summaries.
"""
import os, sys, json, time, warnings, traceback
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

TWO_PI = 2 * np.pi
CELLS = {
    # name: (family, n, intercept, a1, a3, sigma)
    "gauss": ("gaussian", 200, 0.0, 1.0, 0.30, 1.0),
    "gauss_small": ("gaussian", 60, 0.0, 1.0, 0.30, 0.5),
    "binom": ("binomial", 400, 0.0, 1.5, 0.60, None),
    "pois": ("poisson", 200, 0.5, 0.8, 0.25, None),
}
N_TEST = 60
GRID = np.linspace(0.02, 0.98, 25)


def f1(x, a1):
    return a1 * np.sin(TWO_PI * x)


def f3(x, a3):
    return a3 * np.cos(TWO_PI * x)


def inv_link(fam, eta):
    if fam == "gaussian":
        return eta
    if fam == "binomial":
        return 1 / (1 + np.exp(-eta))
    return np.exp(eta)


def draw_y(fam, mu, sigma, rng):
    if fam == "gaussian":
        return mu + rng.normal(0, sigma, mu.shape)
    if fam == "binomial":
        return (rng.uniform(size=mu.shape) < mu).astype(float)
    return rng.poisson(mu).astype(float)


def one_rep(args):
    cell, rep = args
    warnings.simplefilter("ignore")
    import gamfit
    from pygam import LinearGAM, LogisticGAM, PoissonGAM, s

    fam, n, b0, a1, a3, sigma = CELLS[cell]
    trng = np.random.default_rng(12345)
    Xt = trng.uniform(0.02, 0.98, (N_TEST, 3))  # fixed test design
    rng = np.random.default_rng(1000 + rep)
    X = rng.uniform(0, 1, (n, 3))
    eta = b0 + f1(X[:, 0], a1) + f3(X[:, 2], a3)
    mu = inv_link(fam, eta)
    y = draw_y(fam, mu, sigma, rng)
    eta_t = b0 + f1(Xt[:, 0], a1) + f3(Xt[:, 2], a3)
    mu_t = inv_link(fam, eta_t)
    ynew = draw_y(fam, mu_t, sigma, rng)
    # centered truths for partial-dependence (sum-to-zero over training rows)
    truth_pd = {
        "x1": f1(GRID, a1) - f1(X[:, 0], a1).mean(),
        "x2": np.zeros_like(GRID),
        "x3": f3(GRID, a3) - f3(X[:, 2], a3).mean(),
    }
    out = {"rep": rep}
    d = dict(x1=X[:, 0], x2=X[:, 1], x3=X[:, 2], y=y)
    dt = dict(x1=Xt[:, 0], x2=Xt[:, 1], x3=Xt[:, 2])

    def summ_interval(lo, hi, truth):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        return {
            "cover": ((truth >= lo) & (truth <= hi)).astype(int).tolist(),
            "width": float(np.mean(hi - lo)),
        }

    # ---------------- gamfit ----------------
    g = {}
    try:
        t0 = time.time()
        fam_arg = "gaussian" if fam == "gaussian" else fam
        m = gamfit.fit(d, "y ~ s(x1) + s(x2) + s(x3)", family=fam_arg)
        g["fit_time"] = time.time() - t0
        p = m.predict(dt, interval=0.95, observation_interval=True)
        g["cov_src"] = p["covariance_source"]
        g["mean"] = summ_interval(p["posterior_mean_lower"], p["posterior_mean_upper"], mu_t)
        g["obs"] = summ_interval(p["observation_lower"], p["observation_upper"], ynew)
        pc = m.predict(dt, interval=0.95, covariance_mode="conditional")
        g["mean_cond"] = summ_interval(pc["posterior_mean_lower"], pc["posterior_mean_upper"], mu_t)
        pdd = {}
        for v in ("x1", "x2", "x3"):
            r = m.partial_dependence(f"s({v})", d, grid=GRID)
            est, se = np.asarray(r["predicted"]), np.asarray(r["standard_error"])
            pdd[v] = summ_interval(est - 1.959964 * se, est + 1.959964 * se, truth_pd[v])
            # simultaneous-ish: whole curve covered
            pdd[v]["all"] = int(all(pdd[v]["cover"]))
        g["pd"] = pdd
        st = m.summary()
        g["p_wald"] = {r["name"]: r["p_value"] for r in st.smooth_terms}
        g["edf"] = {r["name"]: r["edf"] for r in st.smooth_terms}
        try:
            lr = m.smooth_significance(d)
            g["p_lr"] = {r["name"]: r["p_value_corrected"] for r in lr}
            g["p_lr_unc"] = {r["name"]: r["p_value_uncorrected"] for r in lr}
        except Exception as e:  # noqa
            g["p_lr_err"] = repr(e)[:300]
    except Exception as e:
        g["error"] = repr(e)[:500]
    out["gamfit"] = g

    # ---------------- pyGAM ----------------
    cls = {"gaussian": LinearGAM, "binomial": LogisticGAM, "poisson": PoissonGAM}[fam]
    for label in ("pygam_default", "pygam_gridsearch"):
        q = {}
        try:
            t0 = time.time()
            gm = cls(s(0) + s(1) + s(2))
            if label == "pygam_default":
                gm.fit(X, y)
            else:
                gm.gridsearch(X, y, progress=False)
            q["fit_time"] = time.time() - t0
            q["lam"] = [float(np.ravel(l)[0]) for l in gm.lam]
            ci = gm.confidence_intervals(Xt, width=0.95)
            q["mean"] = summ_interval(ci[:, 0], ci[:, 1], mu_t)
            if fam == "gaussian":
                pi = gm.prediction_intervals(Xt, width=0.95)
                q["obs"] = summ_interval(pi[:, 0], pi[:, 1], ynew)
            pdd = {}
            for j, v in enumerate(("x1", "x2", "x3")):
                XG = np.full((len(GRID), 3), 0.5)
                XG[:, j] = GRID
                est, band = gm.partial_dependence(j, X=XG, width=0.95)
                # centre estimate by its training-row mean (generous to pyGAM,
                # whose term levels are not identified by a constraint)
                shift = gm.partial_dependence(j, X=X).mean()
                pdd[v] = summ_interval(band[:, 0] - shift, band[:, 1] - shift, truth_pd[v])
                pdd[v]["all"] = int(all(pdd[v]["cover"]))
                pdd[v]["uncentred_shift"] = float(shift)
            q["pd"] = pdd
            pv = gm.statistics_["p_values"]
            q["p_wald"] = {"s(x1)": float(pv[0]), "s(x2)": float(pv[1]), "s(x3)": float(pv[2])}
            q["edf_total"] = float(gm.statistics_["edof"])
        except Exception as e:
            q["error"] = repr(e)[:500] + traceback.format_exc()[-500:]
        out[label] = q
    return out


def main():
    cell, nrep, nw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs("results", exist_ok=True)
    from multiprocessing import Pool
    t0 = time.time()
    res = []
    with Pool(nw) as pool:
        for i, r in enumerate(pool.imap_unordered(one_rep, [(cell, k) for k in range(nrep)])):
            res.append(r)
            if (i + 1) % 20 == 0:
                print(f"{cell}: {i+1}/{nrep} {time.time()-t0:.0f}s", flush=True)
                json.dump(res, open(f"results/{cell}.json", "w"))
    json.dump(res, open(f"results/{cell}.json", "w"))
    print(f"{cell} done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
