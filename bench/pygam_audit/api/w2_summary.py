import numpy as np, warnings
rng=np.random.default_rng(0)
n=500
X=np.column_stack([rng.uniform(0,1,n), rng.uniform(-2,2,n), rng.integers(0,4,n)])
y=np.sin(2*np.pi*X[:,0]) + 0.5*X[:,1]**2 + np.array([0,0.5,-0.5,1.0])[X[:,2].astype(int)] + rng.normal(0,0.3,n)
import gamfit
from gamfit.sklearn import GAMRegressor
r=GAMRegressor(formula="s(x0)+s(x1)+factor(x2)").fit(X,y)
print("score", r.score(X,y))
print("-----summary str")
print(r.summary())
S=r.summary()
print(type(S), [a for a in dir(S) if not a.startswith('_')])
m=r.model_
print("predict full dict:", {k:(v[:2] if hasattr(v,'__len__') else v) for k,v in m.predict(X[:2], return_type='dict').items()})
print("predict interval:", m.predict(X[:2], interval=0.95))
print("model repr:", repr(m))
print("notes:", m.notes)
from pygam import LinearGAM,s,f
g=LinearGAM(s(0)+s(1)+f(2)).fit(X,y)
print("-----pygam summary")
g.summary()
