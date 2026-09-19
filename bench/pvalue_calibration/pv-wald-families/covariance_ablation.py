"""The Wald statistic of the null s(x2) under the conditional covariance Vb
(what summary() uses) and under the smoothing-parameter-corrected Vc, with the
same Wood (2013) whitening and rank truncation.

Run:     python covariance_ablation.py run gaussian,poisson,binomial,gamma 200 500 out.jsonl
Analyze: python covariance_ablation.py analyze out.jsonl

The statistic is rebuilt from the public design_matrix() accessor. T_cond
reproduces summary()'s chi_sq to ~1e-12 for gaussian/poisson/binomial; gamma
uses unit weights here (the fit uses observed-information weights), so its
corrected statistic is taken as chi_sq * T_corr / T_cond. The reference
distribution is the shipped one (same ref_df; F with n - 12 denominator df for
the estimated-scale families), so the only thing that changes is the
covariance.
"""
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from null_calibration import FAMILIES, simulate  # noqa: E402

ESTIMATED_SCALE = {"gaussian", "gamma"}


def wald(beta, cov, gram, rank):
    ev, U = np.linalg.eigh(gram)
    keep = ev > len(ev) * np.finfo(float).eps * ev.max()
    R = (np.sqrt(ev[keep]) * U[:, keep]).T
    bw = R @ beta
    Vw = R @ cov @ R.T
    Vw = (Vw + Vw.T) / 2
    lam, Q = np.linalg.eigh(Vw)
    tol = len(lam) * np.finfo(float).eps * np.abs(lam).max()
    q, used = 0.0, 0
    for i in np.argsort(lam)[::-1]:
        if lam[i] <= tol:
            continue
        q += (bw @ Q[:, i]) ** 2 / lam[i]
        used += 1
        if used >= rank:
            break
    return q


def one(task):
    fam, n, rep = task
    import gamfit

    seed = 7_000_000 + 10_000_000 * FAMILIES.index(fam) + 1000 * n + rep
    d = simulate(fam, n, seed)
    fd = os.open(os.devnull, os.O_WRONLY)
    sv = os.dup(2)
    os.dup2(fd, 2)
    try:
        m = gamfit.fit(d, "y ~ s(x1) + s(x2)", family=fam)
        row = [r for r in m.summary().smooth_terms if r["name"] == "s(x2)"][0]
        blk = [b for b in m.term_blocks if b.name == "s(x2)"][0]
        a = m.design_matrix(d)
        eta = a.offset + a.matrix @ a.coefficients
        if fam in ("gaussian", "gamma"):
            w = np.ones(n)
        elif fam == "poisson":
            w = np.exp(eta)
        elif fam == "binomial":
            mu = 1 / (1 + np.exp(-eta))
            w = mu * (1 - mu)
        else:
            raise ValueError(fam)
        X = a.matrix[:, blk.start:blk.end]
        G = X.T @ (w[:, None] * X)
        b = a.coefficients[blk.start:blk.end]
        rank = int(min(max(round(row["edf"]), 1), X.shape[1]))
        sl = slice(blk.start, blk.end)
        t_cond = wald(b, a.covariance_conditional[sl, sl], G, rank)
        vc = a.covariance_smoothing_corrected
        t_corr = None if vc is None else wald(b, vc[sl, sl], G, rank)
        return dict(family=fam, n=n, rep=rep, T_reported=row["chi_sq"], T_cond=t_cond,
                    T_corr=t_corr, edf=row["edf"], ref_df=row["ref_df"], p=row["p_value"])
    except Exception as e:  # recorded, never dropped silently
        return dict(family=fam, n=n, rep=rep, error=str(e)[:200])
    finally:
        os.dup2(sv, 2)
        os.close(fd)


def analyze(path):
    rows = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if "error" in r or None in (r.get("p"), r.get("T_corr"), r.get("T_cond")):
            continue
        rows[r["family"]].append(r)
    for fam, rs in sorted(rows.items()):
        p_vb = np.array([r["p"] for r in rs])
        t_vc = np.array([r["T_reported"] * (r["T_corr"] / r["T_cond"] if r["T_cond"] > 0 else 1.0)
                         for r in rs])
        if fam in ESTIMATED_SCALE:
            p_vc = np.array([stats.f.sf(t / r["ref_df"], r["ref_df"], r["n"] - 12) for t, r in zip(t_vc, rs)])
        else:
            p_vc = np.array([stats.chi2.sf(t, r["ref_df"]) for t, r in zip(t_vc, rs)])
        for label, p in (("Vb (shipped)", p_vb), ("Vc corrected", p_vc)):
            s = [(p <= a).mean() for a in (0.10, 0.05, 0.01)]
            low = p[p < 0.5] / 0.5
            ks = stats.kstest(low, "uniform").pvalue if len(low) > 5 else float("nan")
            print(f"{fam:10s} {label:13s} m={len(rs):4d} size .10={s[0]:.3f} .05={s[1]:.3f} "
                  f".01={s[2]:.3f} KS<.5={ks:.3f} P(p<.5)={(p < 0.5).mean():.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "analyze":
        analyze(sys.argv[2])
    else:
        fams, n, reps, out = sys.argv[2].split(","), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
        tasks = [(f, n, i) for f in fams for i in range(reps)]
        with Pool(int(os.environ.get("NPROC", "2"))) as pool, open(out, "a") as fh:
            for r in pool.imap_unordered(one, tasks, chunksize=4):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
