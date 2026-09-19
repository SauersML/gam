from common import *
import inspect
rng = np.random.default_rng(7); n=300
x = rng.uniform(0,1,n); y = np.log1p(5*x) + rng.normal(0,.3,n)
m = gamfit.fit(dict(x=x,y=y), "y ~ s(x, shape=monotone_increasing)")
g = np.linspace(0,1,501)
print("predict sig:", inspect.signature(m.predict))
for kw in [dict(interval=0.95), dict(se=True)]:
    try:
        out = m.predict(dict(x=g), **kw); print(kw, type(out), getattr(out,'shape',None) if not isinstance(out,dict) else list(out))
        if isinstance(out, dict):
            for k,v in out.items():
                v=np.asarray(v)
                if v.shape==g.shape: print(k, "worst decrease", check_shape(v,g,"inc"))
        elif isinstance(out,np.ndarray) and out.ndim==2:
            for j in range(out.shape[1]): print(j, "worst decrease", check_shape(out[:,j],g,"inc"))
    except Exception as e: print(kw, "ERR", type(e).__name__, str(e)[:200])
# explicit knots via smooths=
try:
    m2 = gamfit.fit(dict(x=x,y=y), "y ~ s(x)", smooths={"x": gamfit.BSpline(knots=np.linspace(-0.5,1.5,10))})
    print("smooths knots OK", gpred(m2, dict(x=np.array([-0.4,1.4]))))
except Exception as e: print("smooths knots ERR", type(e).__name__, str(e)[:300])
