"""Accuracy + failure rate of monotone fits: gamfit default/k=20/unconstrained vs pyGAM."""
from common import *
from pygam import LinearGAM, s

truths = {
    "inc_step": ("monotone_increasing", "monotonic_inc", lambda x: 1 / (1 + np.exp(-30 * (x - 0.5)))),
    "inc_flat": ("monotone_increasing", "monotonic_inc", lambda x: np.where(x < 0.6, 0.0, (x - 0.6) * 3)),
    "dec_exp": ("monotone_decreasing", "monotonic_dec", lambda x: np.exp(-4 * x)),
}
grid = np.linspace(0, 1, 1001)
rows = []
for name, (gk, pk, f) in truths.items():
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        n = 200
        x = rng.uniform(0, 1, n)
        y = f(x) + rng.normal(0, 0.3, n)
        tr = f(grid)
        r = dict(truth=name, seed=seed)
        pg = LinearGAM(s(0, constraints=pk)).gridsearch(x[:, None], y, progress=False)
        r["pyg"] = np.sqrt(np.mean((pg.predict(grid[:, None]) - tr) ** 2))
        for lab, form in [("gf_unc", "y ~ s(x)"), ("gf_shape", f"y ~ s(x, shape={gk})"),
                          ("gf_shape_k20", f"y ~ s(x, k=20, shape={gk})")]:
            try:
                m = gamfit.fit(dict(x=x, y=y), form)
                r[lab] = np.sqrt(np.mean((gpred(m, dict(x=grid)) - tr) ** 2))
            except Exception as e:
                r[lab] = np.nan
        rows.append(r)
        print(r, flush=True); open("t03b.rows","a").write(repr(r)+"\n")
df = pd.DataFrame(rows)
print(df.groupby("truth").agg(lambda v: f"{np.nanmean(v):.4f} (fail {np.isnan(v).sum()})"))
df.to_csv("t03b.csv", index=False)
