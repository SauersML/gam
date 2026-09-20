from common import *
f = lambda x: 1/(1+np.exp(-30*(x-0.5)))
grid = np.linspace(0,1,1001)
for seed in [2,3]:
    rng = np.random.default_rng(100+seed); n=200
    x = rng.uniform(0,1,n); y = f(x)+rng.normal(0,.3,n)
    for form in ["y ~ s(x)", "y ~ s(x, shape=monotone_increasing)", "y ~ s(x, double_penalty=false, shape=monotone_increasing)", "y ~ s(x, double_penalty=false)"]:
        try:
            m = gamfit.fit(dict(x=x,y=y), form)
            p = gpred(m, dict(x=grid))
            edf = getattr(m, "edf", None)
            try: edf = m.edf() if callable(edf) else edf
            except Exception: pass
            print(seed, form, "rmse=%.4f"%np.sqrt(np.mean((p-f(grid))**2)), "sp=", {k: round(float(v),4) for k,v in m.smoothing_parameters().items()}, "edf=", edf, "range=%.3f..%.3f"%(p.min(),p.max()), flush=True)
        except Exception as e:
            print(seed, form, "ERR", str(e)[:200], flush=True)
