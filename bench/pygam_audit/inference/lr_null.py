import os, sys, json, warnings
import numpy as np
from mc import CELLS, f1, f3, inv_link, draw_y
def one(rep):
    warnings.simplefilter("ignore"); import gamfit
    cell=sys.argv[1]; fam, n, b0, a1, a3, sigma = CELLS[cell]
    rng = np.random.default_rng(50000+rep)
    X = rng.uniform(0, 1, (n, 3))
    y = draw_y(fam, inv_link(fam, b0 + f1(X[:, 0], a1) + f3(X[:, 2], a3)), sigma, rng)
    d = dict(x1=X[:, 0], x2=X[:, 1], x3=X[:, 2], y=y)
    try:
        m = gamfit.fit(d, "y ~ s(x1) + s(x2) + s(x3)", family=fam)
        w = {r["name"]: r["p_value"] for r in m.summary().smooth_terms}
        lr = {r["name"]: (r["p_value_corrected"], r["p_value_conditional"], r["bartlett_factor"]) for r in m.smooth_significance(d)}
        return {"w": w, "lr": lr}
    except Exception as e:
        return {"err": str(e)[:200]}
if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(int(sys.argv[3])) as p: res = p.map(one, range(int(sys.argv[2])))
    json.dump(res, open(f"results/lrnull_{sys.argv[1]}.json", "w"))
    ok=[r for r in res if "err" not in r]
    p2=np.array([r["lr"]["s(x2)"][0] for r in ok if r["lr"]["s(x2)"][0] is not None]); pc=np.array([r["lr"]["s(x2)"][1] for r in ok if r["lr"]["s(x2)"][1] is not None]); pw=np.array([r["w"]["s(x2)"] for r in ok])
    se=lambda a: np.sqrt(a*(1-a)/len(p2))
    for name,a in (("lr",p2),("lr_conditional",pc),("wald",pw)):
        print(sys.argv[1], name, "n", len(a), "rej.05", np.mean(a<.05), "rej.01", np.mean(a<.01), "rej.10", np.mean(a<.10))
    print("mcse .05", se(.05), ".01", se(.01), "errors", len(res)-len(ok))
