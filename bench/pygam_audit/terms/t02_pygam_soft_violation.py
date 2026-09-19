"""Show pyGAM's soft (1e9 quadratic) constraint penalty lets the fit violate
monotonicity once the data term is large (large n and/or large response scale),
while gamfit's hard cone does not."""
from common import *
from pygam import LinearGAM, s

grid = np.linspace(0, 1, 5001)
for scale, n in [(1.0, 2000), (1e3, 2000), (1e4, 20000), (1e5, 20000)]:
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, n)
    # monotone truth with a sharp dip in the data (local non-monotone bump)
    f = x + 0.4 * np.exp(-((x - 0.5) / 0.03) ** 2) * np.sign(x - 0.5)  # N-shaped wiggle -> down then up
    f = x - 0.6 * np.exp(-((x - 0.5) / 0.05) ** 2)
    y = scale * (f + rng.normal(0, 0.05, n))
    pg = LinearGAM(s(0, constraints="monotonic_inc", n_splines=30), lam=0.6).fit(x[:, None], y)
    pv = check_shape(pg.predict(grid[:, None]), grid, "inc")
    d = np.diff(pg.predict(grid[:, None]))
    gm = gamfit.fit(dict(x=x, y=y), "y ~ s(x, k=30, shape=monotone_increasing)")
    gv = check_shape(gpred(gm, dict(x=grid)), grid, "inc")
    print(f"scale={scale:g} n={n}: pyGAM worst decrease={pv:.4g} (rel {pv/scale:.2e}), "
          f"frac grid steps decreasing={np.mean(d < -1e-9*scale):.3f}; gamfit worst decrease={gv:.3g}", flush=True)
