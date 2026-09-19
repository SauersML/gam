"""Binomial reps 13/16/21 route sample() to NUTS (~6-7 min) instead of Polya-Gamma: check whether the fit used the automatic Firth rescue."""
import os, json, time, warnings; os.environ["RAYON_NUM_THREADS"]="1"; warnings.simplefilter("ignore")
import numpy as np, gamfit
from mc import CELLS, f1, f3, inv_link, draw_y
fam, n, b0, a1, a3, sigma = CELLS["binom"]
for rep in [13, 16, 21, 14]:
    rng = np.random.default_rng(1000 + rep); X = rng.uniform(0, 1, (n, 3))
    y = draw_y(fam, inv_link(fam, b0 + f1(X[:, 0], a1) + f3(X[:, 2], a3)), sigma, rng)
    d = dict(x1=X[:, 0], x2=X[:, 1], x3=X[:, 2], y=y)
    t0 = time.time(); m = gamfit.fit(d, "y ~ s(x1) + s(x2) + s(x3)", family=fam); ft = time.time() - t0
    js = m.to_json() if hasattr(m, "to_json") else None
    s = json.dumps(js) if not isinstance(js, str) else js
    import re
    hits = sorted(set(re.findall(r'"(firth[a-z_]*)"\s*:\s*(true|false|"[^"]{0,40}")', s)))
    sm = m.summary()
    conv = getattr(sm, "convergence", None)
    print(rep, "fit %.1fs" % ft, "firth keys:", hits[:6], "| convergence:", str(conv)[:300], flush=True)
