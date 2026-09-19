import os, re, time, warnings; os.environ["RAYON_NUM_THREADS"]="1"; warnings.simplefilter("ignore")
import numpy as np, gamfit
from mc import CELLS, f1, f3, inv_link, draw_y
fam, n, b0, a1, a3, sigma = CELLS["binom"]
for rep in [21, 14]:
    rng = np.random.default_rng(1000 + rep); X = rng.uniform(0, 1, (n, 3))
    y = draw_y(fam, inv_link(fam, b0 + f1(X[:, 0], a1) + f3(X[:, 2], a3)), sigma, rng)
    d = dict(x1=X[:, 0], x2=X[:, 1], x3=X[:, 2], y=y)
    m = gamfit.fit(d, "y ~ s(x1) + s(x2) + s(x3)", family=fam)
    s = m.dumps(); s = s.decode("utf-8", "replace") if isinstance(s, (bytes, bytearray)) else str(s)
    print(rep, sorted(set(re.findall(r'"([a-z_]*(?:firth|jeffreys)[a-z_]*)"\s*:\s*([^,}\]]{0,60})', s, re.I)))[:10], flush=True)
