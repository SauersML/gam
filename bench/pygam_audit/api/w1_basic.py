import numpy as np, time, warnings, traceback
rng=np.random.default_rng(0)
n=500
X=np.column_stack([rng.uniform(0,1,n), rng.uniform(-2,2,n), rng.integers(0,4,n)])
y=np.sin(2*np.pi*X[:,0]) + 0.5*X[:,1]**2 + np.array([0,0.5,-0.5,1.0])[X[:,2].astype(int)] + rng.normal(0,0.3,n)
def T(label, f):
    t=time.time()
    try:
        r=f(); print(f"[OK ] {label} ({time.time()-t:.2f}s):", repr(r)[:300])
        return r
    except Exception as e:
        print(f"[ERR] {label}: {type(e).__name__}: {str(e)[:400]}")
print("==== pyGAM")
from pygam import LinearGAM, LogisticGAM, s, f
g=T("pygam fit", lambda: LinearGAM(s(0)+s(1)+f(2)).fit(X,y))
T("pygam predict", lambda: g.predict(X[:3]))
T("pygam predict_mu", lambda: g.predict_mu(X[:3]))
T("pygam score?", lambda: g.score(X,y))
T("pygam deviance_residuals", lambda: g.deviance_residuals(X,y)[:3])
T("pygam loglikelihood", lambda: g.loglikelihood(X,y))
T("pygam prediction_intervals", lambda: g.prediction_intervals(X[:2]))
T("pygam confidence_intervals", lambda: g.confidence_intervals(X[:2]))
T("pygam partial_dependence", lambda: g.partial_dependence(term=0, X=g.generate_X_grid(term=0))[:3])
T("pygam statistics_ keys", lambda: list(g.statistics_.keys()))
print("==== gamfit")
import gamfit
from gamfit.sklearn import GAMRegressor, GAMClassifier
m=T("gamfit.fit_array formula-less? (no formula arg)", lambda: gamfit.fit_array(X,y))
m=T("gamfit.fit_array", lambda: gamfit.fit_array(X,y,"y ~ s(x0)+s(x1)+x2"))
m=T("gamfit.fit_array with factor", lambda: gamfit.fit_array(X,y,"y ~ s(x0)+s(x1)+factor(x2)"))
m=T("gamfit.fit_array with C()", lambda: gamfit.fit_array(X,y,"y ~ s(x0)+s(x1)+C(x2)"))
m2=T("gamfit.fit numpy X + y? (fit(data,formula))", lambda: gamfit.fit(np.column_stack([X,y]),"x3 ~ s(x0)+s(x1)+x2"))
r=T("GAMRegressor(formula=...) numpy", lambda: GAMRegressor(formula="y ~ s(x0)+s(x1)+x2").fit(X,y))
r0=T("GAMRegressor() no formula", lambda: GAMRegressor().fit(X,y))
r1=T("GAMRegressor(formula='s(x0)+s(x1)') RHS only", lambda: GAMRegressor(formula="s(x0)+s(x1)+x2").fit(X,y))
if r is not None:
    T("reg.predict", lambda: r.predict(X[:3]))
    T("reg.score", lambda: r.score(X,y))
    T("reg.model_ predict default return", lambda: r.model_.predict(X[:3]))
    T("reg.model_ predict_array", lambda: r.model_.predict_array(X[:3]))
    T("reg.summary", lambda: str(r.summary())[:200])
    for attr in ["deviance_residuals","loglikelihood","predict_mu","predict_proba","statistics_","coef_","deviance","aic","edf","residuals","log_likelihood","n_iter_","converged"]:
        T(f"reg has {attr}", lambda: hasattr(r,attr) or hasattr(r.model_,attr))
    T("dir(model_)", lambda: [a for a in dir(r.model_) if not a.startswith('_')])
