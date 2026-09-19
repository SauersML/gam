import warnings, sys, time
warnings.simplefilter("always")
import numpy as np
with warnings.catch_warnings(record=True) as w_imp:
    warnings.simplefilter("always")
    import gamfit
print("import warnings:", [(x.category.__name__, str(x.message)[:120]) for x in w_imp])
rng=np.random.default_rng(0); n=500
x=rng.uniform(0,1,n); y=np.sin(2*np.pi*x)+rng.normal(0,.3,n)
data={"x":x,"y":y}
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    t=time.time(); m=gamfit.fit(data,"y ~ s(x)"); print("fit s", round(time.time()-t,2))
    p=m.predict({"x":x[:5]})
    s=m.summary()
print("fit warnings:", [(x.category.__name__, str(x.message)[:200], x.filename.split('/')[-1], x.lineno) for x in w])
print(type(m), type(p))
import pickle
try:
    b=pickle.dumps(m); m2=pickle.loads(b); print("pickle ok", len(b))
except Exception as e: print("pickle FAIL", type(e).__name__, str(e)[:200])
try:
    r=gamfit.plot(m); print("gamfit.plot(model) ->", type(r))
except Exception as e: print("gamfit.plot(model) FAIL", type(e).__name__, str(e)[:300])
# pygam
with warnings.catch_warnings(record=True) as w2:
    warnings.simplefilter("always")
    import pygam
    g=pygam.LinearGAM(pygam.s(0)).fit(x[:,None],y)
    g.summary()
print("pygam warnings:", [(x.category.__name__, str(x.message)[:200]) for x in w2])
