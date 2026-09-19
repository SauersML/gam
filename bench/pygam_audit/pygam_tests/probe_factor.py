import numpy as np, gamfit, warnings; warnings.simplefilter("ignore")
rng=np.random.default_rng(0)
lev=np.array(["a","b","c","d","e"]); g=np.repeat(lev,[3,3,3,3,3]); mu={"a":0,"b":1,"c":2,"d":3,"e":10}
y=np.array([mu[v] for v in g])+rng.normal(0,2.0,len(g))
d={"g":g,"y":y}
for f in ["y ~ factor(g)","y ~ g","y ~ group(g)"]:
    m=gamfit.fit(d,f)
    print(f, m.term_blocks)
    s=m.summary(); print("  lambdas",s.lambdas, "edf",s.edf_total)
    p=m.predict({"g":lev}); gm=np.array([y[g==v].mean() for v in lev])
    print("  pred",np.round(p,3)); print("  grp means",np.round(gm,3))
print("---- ways to get an unpenalized factor")
for f in ["y ~ linear(g)","y ~ factor(g, double_penalty=false)","y ~ C(g)"]:
    try:
        m=gamfit.fit(d,f); p=m.predict({"g":lev}); print(f, m.term_blocks, m.summary().lambdas, np.round(p,3))
    except Exception as e: print(f,"ERR",type(e).__name__, str(e)[:200])
m=gamfit.fit(d,"y ~ factor(g)")
try: print("unseen factor:", m.predict({"g":np.array(["zz"])}))
except Exception as e: print("unseen factor raises", type(e).__name__)
m=gamfit.fit(d,"y ~ group(g)")
try: print("unseen group:", m.predict({"g":np.array(["zz"])}))
except Exception as e: print("unseen group raises", type(e).__name__)
