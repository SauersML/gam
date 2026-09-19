import numpy as np, gamfit
rng=np.random.default_rng(0); n=500
x=rng.uniform(0,1,n); y=np.sin(2*np.pi*x)+rng.normal(0,.3,n)
m=gamfit.fit({"x":x,"y":y},"y ~ s(x)")
m.summary()
print("done")
