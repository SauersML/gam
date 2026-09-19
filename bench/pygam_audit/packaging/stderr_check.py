import numpy as np, gamfit, sys
rng=np.random.default_rng(0); n=300
X=rng.uniform(0,1,(n,2)); y=np.sin(6*X[:,0])+X[:,1]+rng.normal(0,.3,n)
print("BEFORE", file=sys.stderr, flush=True)
m=gamfit.fit({"x":X[:,0],"z":X[:,1],"y":y},"y ~ s(x) + s(z)")
print("AFTER", file=sys.stderr, flush=True)
