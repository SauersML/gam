import numpy as np, gamfit, warnings; warnings.simplefilter("ignore")
rng=np.random.default_rng(0)
lev=np.array(["zeta","alpha","mid","beta"]); eff={"zeta":0.0,"alpha":5.0,"mid":0.0,"beta":5.0}
g=rng.choice(lev,400); y=np.array([eff[v] for v in g])+rng.normal(0,0.3,400)
d={"g":g,"y":y}
for f in ["y ~ factor(g, foo=1)","y ~ group(g, bogus=3)"]:
    try: m=gamfit.fit(d,f); print(f,"ACCEPTED", m.term_blocks)
    except Exception as e: print(f,"ERR",type(e).__name__,str(e)[:150])
for f in ["y ~ s(g)","y ~ linear(g)"]:
    m=gamfit.fit(d,f); p=m.predict({"g":lev}); print(f, m.term_blocks, "pred per level", dict(zip(lev,np.round(p,2))))
    print("   truth", eff, " rmse", np.sqrt(np.mean((m.predict(d)-y)**2)))
