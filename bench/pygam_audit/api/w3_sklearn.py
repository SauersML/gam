import numpy as np, warnings, pickle, io, joblib, time
warnings.filterwarnings("ignore")
rng=np.random.default_rng(0)
n=400
X=np.column_stack([rng.uniform(0,1,n), rng.uniform(-2,2,n), rng.integers(0,4,n)])
y=np.sin(2*np.pi*X[:,0]) + 0.5*X[:,1]**2 + np.array([0,0.5,-0.5,1.0])[X[:,2].astype(int)] + rng.normal(0,0.3,n)
yb=(y>np.median(y)).astype(int)
def T(label, f):
    t=time.time()
    try:
        r=f(); print(f"[OK ] {label} ({time.time()-t:.2f}s):", repr(r)[:250]); return r
    except Exception as e:
        print(f"[ERR] {label}: {type(e).__name__}: {str(e)[:350]}")
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from gamfit.sklearn import GAMRegressor, GAMClassifier
from pygam import LinearGAM, LogisticGAM, s, f
for lib in ["pygam","gamfit"]:
    print("=====",lib)
    mk = (lambda: LinearGAM(s(0)+s(1)+f(2))) if lib=="pygam" else (lambda: GAMRegressor(formula="s(x0)+s(x1)+factor(x2)"))
    mkc = (lambda: LogisticGAM(s(0)+s(1)+f(2))) if lib=="pygam" else (lambda: GAMClassifier(formula="s(x0)+s(x1)+factor(x2)"))
    e=mk()
    T("get_params", lambda: e.get_params())
    T("set_params", lambda: e.set_params(family="gaussian") if lib=="gamfit" else e.set_params(max_iter=50))
    T("clone", lambda: clone(e))
    T("hash(est)", lambda: hash(e))
    T("est == clone(est)", lambda: e == clone(e))
    T("Pipeline(StandardScaler, est).fit", lambda: make_pipeline(StandardScaler(), mk()).fit(X,y).score(X,y))
    T("cross_val_score", lambda: cross_val_score(mk(), X, y, cv=3))
    T("cross_val_score clf default", lambda: cross_val_score(mkc(), X, yb, cv=3))
    T("cross_val_score clf accuracy", lambda: cross_val_score(mkc(), X, yb, cv=3, scoring="accuracy"))
    T("cross_val_score clf neg_log_loss", lambda: cross_val_score(mkc(), X, yb, cv=3, scoring="neg_log_loss"))
    T("cross_val_score clf roc_auc", lambda: cross_val_score(mkc(), X, yb, cv=3, scoring="roc_auc"))
    if lib=="gamfit":
        grid={"formula":["s(x0)+s(x1)+factor(x2)","s(x0)+x1+factor(x2)"]}
    else:
        grid={"lam":[0.1,1.0]}
    T("GridSearchCV", lambda: GridSearchCV(mk(), grid, cv=3).fit(X,y).best_params_)
    T("GridSearchCV n_jobs=2", lambda: GridSearchCV(mk(), grid, cv=3, n_jobs=2).fit(X,y).best_params_)
    fit=mk().fit(X,y)
    T("pickle roundtrip", lambda: np.allclose(pickle.loads(pickle.dumps(fit)).predict(X[:5]), fit.predict(X[:5])))
    def jl():
        b=io.BytesIO(); joblib.dump(fit,b); b.seek(0); return np.allclose(joblib.load(b).predict(X[:5]), fit.predict(X[:5]))
    T("joblib roundtrip", jl)
    w=rng.uniform(0.5,2,n)
    T("fit(X,y,sample_weight=w)", lambda: mk().fit(X,y,sample_weight=w) if lib=="gamfit" else mk().fit(X,y,weights=w))
    T("fit(X,y,weights=w)", lambda: mk().fit(X,y,weights=w))
    T("Pipeline fit with sample_weight", lambda: make_pipeline(StandardScaler(), mk()).fit(X,y,**{("gamregressor" if lib=="gamfit" else "lineargam")+"__sample_weight":w}))
    T("cross_val_score with params sample_weight", lambda: cross_val_score(mk(), X, y, cv=3, params={"sample_weight":w}))
    T("predict 1D row", lambda: fit.predict(X[0]))
    T("predict wrong ncols", lambda: fit.predict(X[:5,:2]))
    Xn=X.copy(); Xn[0,0]=np.nan
    T("fit with NaN", lambda: mk().fit(Xn,y))
    T("predict with NaN", lambda: fit.predict(Xn[:3]))
    T("fit y wrong length", lambda: mk().fit(X,y[:-1]))
    T("fit y 2D column", lambda: mk().fit(X,y[:,None]).predict(X[:2]))
    Xs=X.astype(object); Xs[0,1]="abc"
    T("fit with string in numeric col", lambda: mk().fit(Xs,y))
    T("predict unseen level 7", lambda: fit.predict(np.array([[0.5,0.0,7.0]])))
    T("predict out of range x0=3", lambda: fit.predict(np.array([[3.0,0.0,1.0]])))
    T("fit X list of lists", lambda: mk().fit(X.tolist(), y.tolist()).predict(X[:2].tolist()))
    T("classifier string labels", lambda: mkc().fit(X, np.where(yb==1,"yes","no")).predict(X[:3]))
    T("clf predict_proba", lambda: mkc().fit(X,yb).predict_proba(X[:2]))
    T("clf decision_function", lambda: mkc().fit(X,yb).decision_function(X[:2]))
    T("clf score (what metric)", lambda: mkc().fit(X,yb).score(X,yb))
    T("multiclass", lambda: mkc().fit(X, X[:,2].astype(int)).predict(X[:3]))
