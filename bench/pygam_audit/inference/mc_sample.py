"""Coverage of posterior-sample credible intervals for the mean: gamfit.sample vs pyGAM.sample.
Usage: python mc_sample.py <cell> <nrep> <nworkers>
"""
import os, sys, json, time, warnings, traceback
os.environ.setdefault("RAYON_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from mc import CELLS, N_TEST, f1, f3, inv_link, draw_y


def one_rep(args):
    cell, rep = args
    warnings.simplefilter("ignore")
    import gamfit
    from pygam import LinearGAM, LogisticGAM, PoissonGAM, s
    fam, n, b0, a1, a3, sigma = CELLS[cell]
    Xt = np.random.default_rng(12345).uniform(0.02, 0.98, (N_TEST, 3))
    rng = np.random.default_rng(1000 + rep)
    X = rng.uniform(0, 1, (n, 3))
    y = draw_y(fam, inv_link(fam, b0 + f1(X[:, 0], a1) + f3(X[:, 2], a3)), sigma, rng)
    mu_t = inv_link(fam, b0 + f1(Xt[:, 0], a1) + f3(Xt[:, 2], a3))
    d = dict(x1=X[:, 0], x2=X[:, 1], x3=X[:, 2], y=y)
    dt = dict(x1=Xt[:, 0], x2=Xt[:, 1], x3=Xt[:, 2])
    out = {"rep": rep}
    g = {}
    try:
        m = gamfit.fit(d, "y ~ s(x1) + s(x2) + s(x3)", family=fam)
        t0 = time.time()
        ps = m.sample(d, samples=500, seed=rep)
        g["time"] = time.time() - t0
        g["method"] = ps.method; g["cov_src"] = ps.covariance_source
        r = ps.predict(dt, level=0.95)
        lo, hi = np.asarray(r["posterior_mean_lower"]), np.asarray(r["posterior_mean_upper"])
        g["cover"] = float(np.mean((mu_t >= lo) & (mu_t <= hi))); g["width"] = float(np.mean(hi - lo))
        p = m.predict(dt, interval=0.95)
        g["pred_cover"] = float(np.mean((mu_t >= p["posterior_mean_lower"]) & (mu_t <= p["posterior_mean_upper"])))
        g["pred_width"] = float(np.mean(np.asarray(p["posterior_mean_upper"]) - p["posterior_mean_lower"]))
    except Exception as e:
        g["error"] = repr(e)[:400]
    out["gamfit"] = g
    q = {}
    try:
        cls = {"gaussian": LinearGAM, "binomial": LogisticGAM, "poisson": PoissonGAM}[fam]
        gm = cls(s(0) + s(1) + s(2)).gridsearch(X, y, progress=False)
        np.random.seed(rep)
        t0 = time.time()
        dr = gm.sample(X, y, quantity="mu", sample_at_X=Xt, n_draws=500)
        q["time"] = time.time() - t0
        lo, hi = np.percentile(dr, [2.5, 97.5], axis=0)
        q["cover"] = float(np.mean((mu_t >= lo) & (mu_t <= hi))); q["width"] = float(np.mean(hi - lo))
    except Exception as e:
        q["error"] = repr(e)[:400] + traceback.format_exc()[-300:]
    out["pygam_sample"] = q
    return out


if __name__ == "__main__":
    cell, nrep, nw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    from multiprocessing import Pool
    res = []
    with Pool(nw) as pool:
        for r in pool.imap_unordered(one_rep, [(cell, k) for k in range(nrep)]):
            res.append(r)
    json.dump(res, open(f"results/sample_{cell}.json", "w"))
    for k in ("gamfit", "pygam_sample"):
        ok = [r[k] for r in res if "error" not in r[k]]
        print(cell, k, "n_ok", len(ok), "cover", np.mean([o["cover"] for o in ok]) if ok else None,
              "width", np.mean([o["width"] for o in ok]) if ok else None, "time", np.median([o["time"] for o in ok]) if ok else None)
        if k == "gamfit" and ok:
            print("  gamfit predict() cover", np.mean([o["pred_cover"] for o in ok]), "width", np.mean([o["pred_width"] for o in ok]), ok[0]["method"], ok[0]["cov_src"])
        errs = [r[k]["error"] for r in res if "error" in r[k]]
        if errs:
            print("  errors", len(errs), errs[0][:300])
