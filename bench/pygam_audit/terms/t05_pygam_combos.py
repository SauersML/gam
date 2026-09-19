"""pyGAM: constraint combos that gamfit rejects. Do they work, and do they hold?"""
from common import *
from pygam import LinearGAM, s, te, f, l

rng = np.random.default_rng(3)
n = 1000
x = rng.uniform(0, 1, n)
z = rng.uniform(0, 1, n)
gnum = rng.integers(0, 2, n).astype(float)

# 1) monotone x-margin of te(x,z): truth increasing in x for every z
ftrue = lambda x, z: np.log1p(4 * x) * (1 + z) + np.sin(3 * z)
y = ftrue(x, z) + rng.normal(0, 0.3, n)
X = np.c_[x, z]
pg = LinearGAM(te(0, 1, constraints=["monotonic_inc", None])).gridsearch(X, y, progress=False)
gx = np.linspace(0, 1, 401)
worst = 0.0
for zz in np.linspace(0, 1, 41):
    p = pg.predict(np.c_[gx, np.full_like(gx, zz)])
    worst = max(worst, check_shape(p, gx, "inc"))
print(f"pyGAM te(monotonic_inc x-margin): worst x-decrease over z-slices = {worst:.3e}")
worst_out = 0.0
gxo = np.linspace(-0.5, 1.5, 801)
for zz in np.linspace(-0.2, 1.2, 29):
    p = pg.predict(np.c_[gxo, np.full_like(gxo, zz)])
    worst_out = max(worst_out, check_shape(p, gxo, "inc"))
print(f"pyGAM te(monotonic_inc x-margin) incl. extrapolation: worst = {worst_out:.3e}")

# 2) monotone smooth inside a numeric by= (varying coefficient)
y2 = np.log1p(4 * x) * gnum + rng.normal(0, 0.3, n)
pg2 = LinearGAM(s(0, by=1, constraints="monotonic_inc") + l(1)).gridsearch(np.c_[x, gnum], y2, progress=False)
p = pg2.predict(np.c_[gx, np.ones_like(gx)]) - pg2.predict(np.c_[gx, np.zeros_like(gx)])
print(f"pyGAM s(by=, monotonic_inc): worst decrease of by-curve = {check_shape(p, gx, 'inc'):.3e}")

# 3) several constraints on one term: monotone increasing + concave
y3 = np.log1p(8 * x) + rng.normal(0, 0.3, n)
pg3 = LinearGAM(s(0, constraints=["monotonic_inc", "concave"])).gridsearch(x[:, None], y3, progress=False)
p = pg3.predict(gx[:, None])
print(f"pyGAM s(constraints=[inc, concave]): inc viol={check_shape(p, gx, 'inc'):.3e} concave viol={check_shape(p, gx, 'concave'):.3e}")

# 4) periodic basis + monotone (nonsense request, pyGAM accepts?)
try:
    pg4 = LinearGAM(s(0, basis="cp", constraints="monotonic_inc")).fit(x[:, None], y3)
    p = pg4.predict(gx[:, None])
    print(f"pyGAM s(basis=cp, monotonic_inc) ACCEPTED; inc viol={check_shape(p, gx, 'inc'):.3e}, endpoints f(0)={p[0]:.3f} f(1)={p[-1]:.3f}")
except Exception as e:
    print("pyGAM cp+mono rejected:", e)

# 5) convex + concave simultaneously (contradiction -> must be linear)
try:
    pg5 = LinearGAM(s(0, constraints=["convex", "concave"])).fit(x[:, None], y3)
    p = pg5.predict(gx[:, None])
    print(f"pyGAM convex+concave accepted: max|f''| (grid) = {np.abs(np.diff(p,2)).max():.3e}")
except Exception as e:
    print("pyGAM convex+concave rejected:", e)
