import numpy as np, gamfit, warnings, traceback
warnings.simplefilter("ignore")
rng=np.random.default_rng(0)
n=300; x=rng.uniform(0,1,n); y=np.sin(6*x)+rng.normal(0,.3,n)
d={"x":x,"y":y}
def tryit(label, f):
    try:
        r=f(); print(f"[OK]  {label}: {r if not hasattr(r,'shape') else r.shape}")
    except Exception as e:
        print(f"[ERR] {label}: {type(e).__name__}: {str(e)[:200]}")
tryit("intercept only y~1", lambda: gamfit.fit(d,"y ~ 1").predict(d)[:2])
tryit("no terms y~0", lambda: gamfit.fit(d,"y ~ 0").predict(d)[:2])
tryit("no terms y~-1", lambda: gamfit.fit(d,"y ~ -1").predict(d)[:2])
tryit("s(x)-1", lambda: [b.name for b in gamfit.fit(d,"y ~ s(x) - 1").term_blocks])
tryit("0+s(x)", lambda: [b.name for b in gamfit.fit(d,"y ~ 0 + s(x)").term_blocks])
tryit("dup s(x)+s(x)", lambda: [b.name for b in gamfit.fit(d,"y ~ s(x) + s(x)").term_blocks])
tryit("te(x) single", lambda: [b.name for b in gamfit.fit(d,"y ~ te(x)").term_blocks])
d2=dict(d); d2["z"]=rng.uniform(size=n)
tryit("te k=[5]", lambda: gamfit.fit(d2,"y ~ te(x,z,k=[5])").term_blocks)
tryit("te k=[5,5,5]", lambda: gamfit.fit(d2,"y ~ te(x,z,k=[5,5,5])").term_blocks)
tryit("by missing", lambda: gamfit.fit(d,"y ~ s(x, by=w)").term_blocks)
dn=dict(d); dn["x"]=x.copy(); dn["x"][0]=np.nan
tryit("NaN in x fit", lambda: gamfit.fit(dn,"y ~ s(x)").summary().n_obs)
dn=dict(d); dn["y"]=y.copy(); dn["y"][0]=np.nan
tryit("NaN in y fit", lambda: gamfit.fit(dn,"y ~ s(x)").summary().n_obs)
dw=dict(d); dw["w"]=np.ones(n); dw["w"][0]=np.nan
tryit("NaN in w fit", lambda: gamfit.fit(dw,"y ~ s(x)",weights="w").summary().n_obs)
dw=dict(d); dw["w"]=np.ones(n); dw["w"][0]=-1
tryit("negative weight fit", lambda: gamfit.fit(dw,"y ~ s(x)",weights="w").summary().n_obs)
m=gamfit.fit(d,"y ~ s(x)")
dn=dict(d); dn["x"]=x.copy(); dn["x"][0]=np.nan
tryit("NaN predict", lambda: m.predict(dn)[:3])
tryit("NaN predict interval", lambda: m.predict(dn, interval=0.95)["posterior_mean"][:3])
tryit("NaN partial_dependence grid", lambda: m.partial_dependence("s(x)", d, grid=np.array([np.nan,0.5]))["predicted"])
tryit("NaN sample_replicates", lambda: m.sample_replicates(dn, 2)[:, :3])
tryit("inf predict", lambda: m.predict({"x":np.array([np.inf,0.5])}))
tryit("pdep bad term", lambda: m.partial_dependence("s(q)", d))
tryit("string y", lambda: gamfit.fit({"x":x,"y":np.array(["a"]*n)},"y ~ s(x)").family_name)
tryit("string x in s()", lambda: gamfit.fit({"x":np.array(["a","b","c"]*100),"y":y},"y ~ s(x)").family_name)
tryit("len mismatch", lambda: gamfit.fit({"x":x,"y":y[:-1]},"y ~ s(x)").family_name)
tryit("binomial y+0.1", lambda: gamfit.fit({"x":x,"y":(y>0)+0.1},"y ~ s(x)", family="binomial").family_name)
tryit("poisson negative", lambda: gamfit.fit({"x":x,"y":np.round(y*3)},"y ~ s(x)", family="poisson").family_name)
tryit("gamma nonpositive", lambda: gamfit.fit({"x":x,"y":y},"y ~ s(x)", family="gamma").family_name)
tryit("inverse-gaussian", lambda: gamfit.fit({"x":x,"y":np.exp(y)},"y ~ s(x)", family="inverse-gaussian").family_name)
tryit("gamma inverse link", lambda: gamfit.fit({"x":x,"y":np.exp(y)},"y ~ s(x)", family="gamma", link="inverse").family_name)
tryit("gamma identity link", lambda: gamfit.fit({"x":x,"y":np.exp(y)},"y ~ s(x)", family="gamma", link="identity").family_name)
tryit("fit_array 1-D X", lambda: gamfit.fit_array(x, y).predict_array(x)[:2])
tryit("fit_array 3-D X", lambda: gamfit.fit_array(np.ones((5,4,3)), np.ones(5)))
tryit("2-D column y (n,1)", lambda: gamfit.fit({"x":x,"y":y[:,None]},"y ~ s(x)").family_name)
tryit("sample samples=0", lambda: m.sample(d, samples=0).shape)
tryit("sample_replicates 0", lambda: m.sample_replicates(d, 0))
tryit("predict outside range", lambda: m.predict({"x":np.array([-1.0,-0.5,1.5,2.0])}))
