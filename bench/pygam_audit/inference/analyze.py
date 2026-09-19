import json, sys, numpy as np
from scipy import stats

cells = sys.argv[1:] or ["gauss", "gauss_small", "pois", "binom"]
METH = ["gamfit", "pygam_default", "pygam_gridsearch"]


def mcse(p, n):
    return np.sqrt(p * (1 - p) / max(n, 1))


for cell in cells:
    try:
        R = json.load(open(f"results/{cell}.json"))
    except FileNotFoundError:
        continue
    print(f"\n## cell {cell}  (replicates={len(R)})")
    for m in METH:
        err = sum(1 for r in R if "error" in r[m])
        if err:
            print(f"  {m}: {err} errors, e.g. {[r[m]['error'][:200] for r in R if 'error' in r[m]][0]}")
    print("| method | mean cov | mean width | obs cov | obs width | PD cov x1 | PD cov x2 | PD cov x3 | PD whole-curve x1/x2/x3 | fit s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for m in METH + ["gamfit_cond"]:
        src = "gamfit" if m == "gamfit_cond" else m
        ok = [r[src] for r in R if "error" not in r[src]]
        if not ok:
            continue
        key = "mean_cond" if m == "gamfit_cond" else "mean"
        mc = np.mean([np.mean(o[key]["cover"]) for o in ok])
        mw = np.mean([o[key]["width"] for o in ok])
        if "obs" in ok[0] and m != "gamfit_cond":
            oc = np.mean([np.mean(o["obs"]["cover"]) for o in ok]); ow = np.mean([o["obs"]["width"] for o in ok])
            obs = f"{oc:.3f} | {ow:.3f}"
        else:
            obs = "- | -"
        if m != "gamfit_cond":
            pdc = [np.mean([np.mean(o["pd"][v]["cover"]) for o in ok]) for v in ("x1", "x2", "x3")]
            pda = [np.mean([o["pd"][v]["all"] for o in ok]) for v in ("x1", "x2", "x3")]
            pd = " | ".join(f"{c:.3f}" for c in pdc) + " | " + "/".join(f"{a:.2f}" for a in pda)
            ft = f"{np.median([o['fit_time'] for o in ok]):.3f}"
        else:
            pd = "- | - | - | -"; ft = "-"
        print(f"| {m} | {mc:.3f} | {mw:.3f} | {obs} | {pd} | {ft} |")
    # p-values
    print("\n| method | test | P(p<.05) null s(x2) | P(p<.01) null | KS-uniform p (null) | frac p>0.99 null | power s(x3) @.05 | power s(x1) @.05 |")
    print("|---|---|---|---|---|---|---|---|")
    for m, key in [("gamfit", "p_wald"), ("gamfit", "p_lr"), ("gamfit", "p_lr_unc"), ("pygam_default", "p_wald"), ("pygam_gridsearch", "p_wald")]:
        ok = [r[m] for r in R if "error" not in r[m] and key in r[m]]
        if not ok:
            continue
        f = lambda v: np.array([np.nan if o[key][v] is None else o[key][v] for o in ok], float)
        p2, p3, p1 = f("s(x2)"), f("s(x3)"), f("s(x1)")
        nn = int(np.isnan(p2).sum() + np.isnan(p3).sum() + np.isnan(p1).sum())
        if nn: print(f"  ({m} {key}: {nn} None p-values; null-term None={int(np.isnan(p2).sum())})")
        p2 = p2[~np.isnan(p2)]
        ks = stats.kstest(p2, "uniform").pvalue
        print(f"| {m} | {key} | {np.mean(p2<.05):.3f} | {np.mean(p2<.01):.3f} | {ks:.2g} | {np.mean(p2>.99):.2f} | {np.nanmean(p3<.05) if False else np.mean(p3[~np.isnan(p3)]<.05):.3f} | {np.mean(p1[~np.isnan(p1)]<.05):.3f} |")
    ok = [r["gamfit"] for r in R if "error" not in r["gamfit"]]
    e2 = np.array([o["edf"]["s(x2)"] for o in ok])
    print(f"\n gamfit edf s(x2): median {np.median(e2):.3g}, frac<0.05 {np.mean(e2<0.05):.2f}, mean {e2.mean():.3f}; edf s(x3) median {np.median([o['edf']['s(x3)'] for o in ok]):.2f}")
    print(f" MCSE for coverage at .95 over nrep*60 points (indep approx) ~ {mcse(.95, len(R)):.3f} per-replicate-level")
    for m in ("pygam_default", "pygam_gridsearch"):
        ok = [r[m] for r in R if "error" not in r[m]]
        if ok:
            print(f" {m}: edof total median {np.median([o['edf_total'] for o in ok]):.1f}; lam median {np.median([o['lam'][0] for o in ok]):.3g}; |uncentred shift x2| median {np.median([abs(o['pd']['x2']['uncentred_shift']) for o in ok]):.3f}")
    lre = sum(1 for o in ok if "p_lr_err" in o)
