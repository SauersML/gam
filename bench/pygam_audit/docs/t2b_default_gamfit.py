import time, numpy as np, pandas as pd, gamfit
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
df = pd.read_csv("data/Default.csv")
y = (df.default == "Yes").astype(int).values
D = {"student": df.student.to_numpy(), "balance": df.balance.to_numpy(), "income": df.income.to_numpy(), "d": y}
for fam in ["binomial"]:
    try:
        m0 = gamfit.fit({**D, "dstr": df.default.to_numpy()}, "dstr ~ s(balance)", family=fam); print("string Yes/No with family=binomial OK", m0.family_name, flush=True)
    except Exception as e: print("string+binomial fails:", type(e).__name__, str(e)[:200], flush=True)
t = time.time(); m = gamfit.fit(D, "d ~ student + s(balance) + s(income)"); print("gamfit fit s", round(time.time()-t,2), m.family_name, flush=True)
p = m.predict(D); print("acc", ((np.asarray(p) > .5) == y).mean(), flush=True)
print(m.summary().smooth_terms_frame(), flush=True)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, term in zip(axs, ["s(balance)", "s(income)"]):
    r = m.partial_dependence(term, D); g, mu, se = r["grid"], r["predicted"], r["standard_error"]
    ax.plot(g, mu); ax.fill_between(g, mu-1.96*se, mu+1.96*se, alpha=.3); ax.set_title(term)
fig.savefig("t2b_default_gamfit.png", dpi=80); print("done", flush=True)
