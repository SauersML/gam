import pickle, copy, numpy as np, warnings
warnings.simplefilter("ignore")
import gamfit, gamfit.sklearn as gs, pygam
rng=np.random.default_rng(0); n=300
X=rng.uniform(0,1,(n,2)); y=np.sin(6*X[:,0])+X[:,1]+rng.normal(0,.3,n)
print([n for n in dir(gs) if not n.startswith('_')])
m=gamfit.fit({"x":X[:,0],"z":X[:,1],"y":y},"y ~ s(x) + s(z)")
for label,obj in [("gamfit.Model",m)]:
    for fn in (pickle.dumps, copy.deepcopy):
        try: fn(obj); print(label, fn.__name__, "OK")
        except Exception as e: print(label, fn.__name__, "FAIL", type(e).__name__, e)
est_cls=[getattr(gs,n) for n in dir(gs) if n.endswith('Regressor') or n.endswith('GAM')]
print(est_cls)
est=est_cls[0]()
est.fit(X,y)
for fn in (pickle.dumps, copy.deepcopy):
    try: fn(est); print(type(est).__name__, fn.__name__, "OK")
    except Exception as e: print(type(est).__name__, fn.__name__, "FAIL", type(e).__name__, e)
g=pygam.LinearGAM().fit(X,y)
b=pickle.dumps(g); print("pygam pickle OK", len(b), np.allclose(pickle.loads(b).predict(X), g.predict(X)))
b=m.save("/tmp/claude-0/-home-user-gam/02aeec89-32a7-52a0-8d71-90f383516996/scratchpad/audit/packaging/m.gam") if hasattr(m,'save') else None
m2=gamfit.load("/tmp/claude-0/-home-user-gam/02aeec89-32a7-52a0-8d71-90f383516996/scratchpad/audit/packaging/m.gam")
print("save/load roundtrip", np.allclose(np.asarray(m2.predict({"x":X[:,0],"z":X[:,1]})), np.asarray(m.predict({"x":X[:,0],"z":X[:,1]}))))
