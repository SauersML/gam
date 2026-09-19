"""pyGAM tour example 2: LogisticGAM on ISLR Default (student factor, balance, income)."""
import time, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gamfit
from pygam import LogisticGAM, s, f

df = pd.read_csv("data/Default.csv")[["default", "student", "balance", "income"]]
X = np.column_stack([(df.student == "Yes").astype(int), df.balance, df.income])
y = (df.default == "Yes").astype(int).values

t = time.time()
gam = LogisticGAM(f(0) + s(1) + s(2)).gridsearch(X, y, progress=False)
print("pygam gridsearch fit s", round(time.time() - t, 2), "acc", gam.accuracy(X, y))
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=.95)
    ax.plot(XX[:, i], pdep); ax.plot(XX[:, i], confi, c='r', ls='--')
    ax.set_title(["student", "balance", "income"][i])
fig.savefig("t2_default_pygam.png", dpi=80)

# ---- gamfit: 1) string response as-is ----
D = {c: df[c].to_numpy() for c in df}
try:
    m = gamfit.fit(D, "default ~ student + s(balance) + s(income)")
    print("string response OK; family =", m.family_name)
except Exception as e:
    print("string response fails:", type(e).__name__, str(e)[:300])

# ---- 2) 0/1 response ----
D["default01"] = y
t = time.time()
m = gamfit.fit(D, "default01 ~ student + s(balance) + s(income)")
print("gamfit fit s", round(time.time() - t, 2), "family", m.family_name)
p = m.predict(D)
print("gamfit acc", ((p > .5) == y).mean())
print("diagnose metrics", m.diagnose({k: D[k] for k in ["default01", "student", "balance", "income"]}).metrics)
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, term in zip(axs[1:], ["s(balance)", "s(income)"]):
    r = m.partial_dependence(term, D)
    g, mu, se = r["grid"], r["predicted"], r["standard_error"]
    ax.plot(g, mu); ax.fill_between(g, mu - 1.96 * se, mu + 1.96 * se, alpha=.3); ax.set_title(term)
fig.savefig("t2_default_gamfit.png", dpi=80)
print(m.summary().smooth_terms_frame())

# sklearn path
from gamfit.sklearn import GAMClassifier
Xdf = {k: D[k] for k in ["student", "balance", "income"]}
try:
    est = GAMClassifier(formula="default ~ student + s(balance) + s(income)").fit(Xdf, df["default"].to_numpy())
    print("GAMClassifier classes_", est.classes_, "AUC", est.score(Xdf, df["default"].to_numpy()))
except Exception as e:
    print("GAMClassifier fails:", type(e).__name__, str(e)[:300])
