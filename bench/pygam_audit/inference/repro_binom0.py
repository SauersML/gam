import numpy as np, gamfit, time, warnings
warnings.simplefilter("ignore")
from mc import CELLS, f1, f3, inv_link, draw_y
fam, n, b0, a1, a3, sigma = CELLS["binom"]
rng = np.random.default_rng(1000)
X = rng.uniform(0, 1, (n, 3))
y = draw_y(fam, inv_link(fam, b0 + f1(X[:, 0], a1) + f3(X[:, 2], a3)), sigma, rng)
d = dict(x1=X[:, 0], x2=X[:, 1], x3=X[:, 2], y=y)
t=time.time()
try:
    m = gamfit.fit(d, "y ~ s(x1) + s(x2) + s(x3)", family="binomial"); print("ok", time.time()-t, m.summary().smooth_terms)
except Exception as e:
    print("FAIL after", time.time()-t, "s"); print(str(e))
