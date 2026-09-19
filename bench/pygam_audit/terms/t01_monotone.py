"""Shape constraints: pyGAM soft penalty vs gamfit hard cone, dense-grid check,
in-sample and extrapolation, several seeds and truths."""
import sys
from common import *
from pygam import LinearGAM, s

truths = {
    "inc_step": ("inc", lambda x: 1 / (1 + np.exp(-30 * (x - 0.5)))),
    "inc_flat": ("inc", lambda x: np.where(x < 0.6, 0.0, (x - 0.6) * 3)),
    "dec_exp": ("dec", lambda x: np.exp(-4 * x)),
    "convex": ("convex", lambda x: (x - 0.3) ** 2 * 2),
    "concave": ("concave", lambda x: np.log1p(5 * x)),
}
pyg_kind = {"inc": "monotonic_inc", "dec": "monotonic_dec", "convex": "convex", "concave": "concave"}
gf_kind = {"inc": "monotone_increasing", "dec": "monotone_decreasing", "convex": "convex", "concave": "concave"}

grid_in = np.linspace(0, 1, 2001)
grid_out = np.linspace(-0.5, 1.5, 4001)
rows = []
for name, (kind, f) in truths.items():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        n = 150
        x = rng.uniform(0, 1, n)
        y = f(x) + rng.normal(0, 0.3, n)
        # pyGAM
        pg = LinearGAM(s(0, constraints=pyg_kind[kind])).gridsearch(x[:, None], y, progress=False)
        pin = pg.predict(grid_in[:, None]); pout = pg.predict(grid_out[:, None])
        # gamfit
        try:
            gm = gamfit.fit(dict(x=x, y=y), f"y ~ s(x, shape={gf_kind[kind]})")
            gin = gpred(gm, dict(x=grid_in)); gout = gpred(gm, dict(x=grid_out))
            gerr = None
        except Exception as e:  # noqa
            gin = gout = None; gerr = repr(e)[:120]
        truth = f(grid_in)
        row = dict(
            truth=name, seed=seed,
            pyg_viol_in=check_shape(pin, grid_in, kind),
            pyg_viol_out=check_shape(pout, grid_out, kind),
            pyg_rmse=np.sqrt(np.mean((pin - truth) ** 2)),
        )
        if gin is not None:
            row.update(
                gf_viol_in=check_shape(gin, grid_in, kind),
                gf_viol_out=check_shape(gout, grid_out, kind),
                gf_rmse=np.sqrt(np.mean((gin - truth) ** 2)),
            )
        else:
            row.update(gf_err=gerr)
        rows.append(row)
        print(row, flush=True)

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.groupby("truth").agg(["mean", "max"]).T)
df.to_csv("t01_monotone.csv", index=False)
