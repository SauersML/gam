"""pyGAM tour example 1: wage ~ s(year) + s(age) + f(education), PDP with bands."""
import time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gamfit
from pygam import LinearGAM, s, f

df = pd.read_csv("data/Wage.csv")[["year", "age", "education", "wage"]]
D = {c: df[c].to_numpy() for c in df}  # pandas3 w/o pyarrow cannot be passed directly
X = np.column_stack([df.year, df.age, pd.Categorical(df.education).codes])
y = df.wage.values

# ---------------- pyGAM (tour verbatim, 8 lines) ----------------
t = time.time()
gam = LinearGAM(s(0) + s(1) + f(2)).fit(X, y)
print("pygam fit s", time.time() - t)
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
    ax.set_title(["year", "age", "education"][i])
fig.savefig("t1_wage_pygam.png", dpi=80)

# ---------------- gamfit ----------------
t = time.time()
model = gamfit.fit(D, "wage ~ s(year) + s(age) + education")
print("gamfit fit s", time.time() - t)
print("term_blocks:", [b.name for b in model.term_blocks])
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, term in zip(axs, ["s(year)", "s(age)"]):
    pd_ = model.partial_dependence(term, D)          # wheel 0.1.267 needs data
    g, m, se = pd_["grid"], pd_["predicted"], pd_["standard_error"]
    ax.plot(g, m); ax.fill_between(g, m - 1.96 * se, m + 1.96 * se, alpha=.3)
    ax.set_title(term)
# factor term: is partial_dependence usable?
try:
    r = model.partial_dependence("education", D)
    print("factor PD:", r)
except Exception as e:
    print("factor PD error:", type(e).__name__, e)
fig.savefig("t1_wage_gamfit.png", dpi=80)

# the built-in plot
try:
    ax = model.plot(D, x="age")
    ax.figure.savefig("t1_wage_gamfit_builtin_plot.png", dpi=80)
    print("model.plot ok")
except Exception as e:
    print("model.plot error:", type(e).__name__, e)
print(model.summary())
