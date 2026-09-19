import sys, warnings, numpy as np
warnings.simplefilter("ignore")
import gamfit
rng=np.random.default_rng(0); n=400
x=rng.uniform(0,1,n); y=np.sin(6*x)+rng.normal(0,.3,n)
m=gamfit.fit({"x":x,"y":y},"y ~ s(x)")
p=m.predict({"x":x[:3]})
print(sys.version.split()[0], "numpy", np.__version__, "gamfit", gamfit.__version__, "pred", np.round(np.asarray(p),4))
yb=(rng.uniform(size=n)<1/(1+np.exp(-3*np.sin(6*x)))).astype(float)
mb=gamfit.fit({"x":x,"y":yb},"y ~ s(x)", family="binomial"); print("binomial ok", np.round(np.asarray(mb.predict({"x":x[:2]})),3))
