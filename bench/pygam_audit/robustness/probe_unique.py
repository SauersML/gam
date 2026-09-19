import gamfit, numpy as np, warnings, sys
from cases import CASES
def tryfit(d, f, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            m = gamfit.fit(d, f, **kw); s=m.summary()
            print("OK", f, "edf", round(s.edf_total,3), "ncoef", len(s.coefficients) if s.coefficients is not None else None, "certified", s.convergence.get("certified"))
        except Exception as e:
            print("ERR", f, type(e).__name__, str(e)[:250])
        for x in w: print("   warn:", str(x.message)[:250])
c = CASES["tiny_n5"](); tryfit(c["data"], "y ~ s(x, k=20)")
c = CASES["two_unique"]()
for f in ["y ~ s(x)", "y ~ s(x, k=3)", "y ~ s(x, k=4)", "y ~ x"]: tryfit(c["data"], f)
c = CASES["three_unique"]()
for f in ["y ~ s(x)", "y ~ s(x, k=3)", "y ~ s(x, k=5)", "y ~ s(x, bs='tps')"]: tryfit(c["data"], f)
r=np.random.default_rng(0)
for nu in [4,5,6,8,10,16]:
    x = r.integers(0, nu, 400).astype(float); y = np.sin(x) + r.normal(0,.3,400)
    tryfit({"x":x,"y":y}, "y ~ s(x)")
