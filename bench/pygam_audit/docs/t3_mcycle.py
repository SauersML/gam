"""pyGAM tour example 3: mcycle wiggly fit + prediction intervals + posterior draws."""
import time, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gamfit
from pygam import LinearGAM

df = pd.read_csv("data/mcycle.csv")[["times", "accel"]]
X, y = df[["times"]].values, df.accel.values

t = time.time()
gam = LinearGAM(n_splines=25).gridsearch(X, y, progress=False)
print("pygam s", round(time.time() - t, 2))
XX = gam.generate_X_grid(term=0, n=500)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(XX, gam.predict(XX), 'r--')
ax[0].plot(XX, gam.prediction_intervals(XX, width=.95), color='b', ls='--')
ax[0].scatter(X, y, facecolor='gray', edgecolors='none'); ax[0].set_title("pyGAM gridsearch, n_splines=25")

D = {"times": df.times.to_numpy(), "accel": y}
grid = {"times": np.linspace(df.times.min(), df.times.max(), 500)}
for j, (formula, kw, title) in enumerate([
    ("accel ~ s(times)", {}, "gamfit default s(times)"),
    ("accel ~ s(times)", {"noise_formula": "s(times)"}, "gamfit location-scale"),
]):
    t = time.time()
    m = gamfit.fit(D, formula, **kw)
    print(title, "fit s", round(time.time() - t, 2), "edf", m.summary().edf_total)
    p = m.predict(grid, interval=0.95, observation_interval=True, return_type="dict")
    print("  columns:", list(p.keys()))
    a = ax[j + 1]
    a.scatter(D["times"], y, facecolor='gray', edgecolors='none')
    a.plot(grid["times"], p["posterior_mean"], 'r-')
    a.fill_between(grid["times"], p["posterior_mean_lower"], p["posterior_mean_upper"], alpha=.4)
    if "observation_lower" in p:
        a.plot(grid["times"], p["observation_lower"], 'b--'); a.plot(grid["times"], p["observation_upper"], 'b--')
    a.set_title(title)
    # empirical coverage of the 95% observation interval on training rows
    q = m.predict(D, interval=0.95, observation_interval=True, return_type="dict")
    if "observation_lower" in q:
        cov = np.mean((y >= q["observation_lower"]) & (y <= q["observation_upper"]))
        print("  in-sample 95% observation-interval coverage:", round(cov, 3))
lo, hi = gam.prediction_intervals(X, width=.95).T
print("pygam in-sample 95% PI coverage:", round(np.mean((y >= lo) & (y <= hi)), 3))
fig.savefig("t3_mcycle.png", dpi=70)

# posterior draws, pyGAM-tour style: gam.sample(X, y, quantity='mu', n_draws=...)
t = time.time()
post = m.sample(D, seed=0)
print("sample s", round(time.time() - t, 2), post)
