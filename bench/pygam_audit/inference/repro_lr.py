import numpy as np, gamfit, warnings, json
warnings.simplefilter("ignore")
from mc import CELLS, f1, f3, inv_link, draw_y
fam, n, b0, a1, a3, sigma = CELLS["gauss"]
for rep in (35,):
    rng = np.random.default_rng(1000+rep)
    X = rng.uniform(0, 1, (n, 3))
    y = draw_y(fam, inv_link(fam, b0 + f1(X[:, 0], a1) + f3(X[:, 2], a3)), sigma, rng)
    d = dict(x1=X[:, 0], x2=X[:, 1], x3=X[:, 2], y=y)
    m = gamfit.fit(d, "y ~ s(x1) + s(x2) + s(x3)")
    for r in m.smooth_significance(d):
        print({k:v for k,v in r.items()})
