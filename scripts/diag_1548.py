import numpy as np, gamfit, pandas as pd, json

def signal(t): return np.sin(2.0*t) + 0.3*t**2

def fit(x, y):
    df = pd.DataFrame({"y": y, "x": x})
    return gamfit.fit(df, 'y ~ s(x, bs="ps")', family="gaussian")

def fit_pred(x, y, grid):
    m = fit(x, y)
    p = np.asarray(m.predict(pd.DataFrame({"x": grid}))).ravel()
    s = m.summary()
    return p, s

rng = np.random.RandomState(2)
n = 400
x = rng.uniform(-2.0, 2.0, n)
y = signal(x) + 0.2*rng.standard_normal(n)
grid = np.linspace(-1.8, 1.8, 60)

m1 = fit(x, y); m2 = fit(-x, y)
s1, s2 = m1.summary(), m2.summary()
print("=== summary attrs ===")
print([a for a in dir(s1) if not a.startswith('_')])
p1 = np.asarray(m1.predict(pd.DataFrame({"x": grid}))).ravel()
p2 = np.asarray(m2.predict(pd.DataFrame({"x": -grid}))).ravel()
print("max drift:", np.max(np.abs(p1-p2)), "of signal range", np.ptp(signal(grid)))
print("lambdas  original:", np.asarray(s1.lambdas))
print("lambdas reflected:", np.asarray(s2.lambdas))
for nm in ("edf_total","reml_score","score","gcv","edf"):
    try:
        print(f"{nm}: orig={getattr(s1,nm)} refl={getattr(s2,nm)}")
    except Exception as e:
        pass
