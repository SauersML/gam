"""Does the binomial SAS / beta-logistic link fit on ordinary logistic data?"""
import numpy as np, gamfit, warnings, sys
warnings.filterwarnings("ignore")
for seed in (0, 1, 2):
    rng = np.random.default_rng(seed)
    n = 1500
    x = rng.uniform(0, 1, n)
    eta = np.sin(2 * np.pi * x)
    y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    d = {"x": x, "y": y}
    for kw in (dict(family="binomial", link="sas"), dict(family="binomial", link="beta-logistic"),
               dict(family="binomial", link="logit")):
        try:
            m = gamfit.fit(d, "y ~ s(x)", **kw)
            p = m.predict({"x": np.linspace(0.05, 0.95, 50)})
            tru = 1 / (1 + np.exp(-np.sin(2 * np.pi * np.linspace(0.05, 0.95, 50))))
            print(seed, kw["link"], "OK", m.family_name, "rmse", round(float(np.sqrt(np.mean((p - tru) ** 2))), 4))
        except Exception as e:
            print(seed, kw["link"], "FAIL", type(e).__name__, str(e)[:300])
        sys.stdout.flush()
