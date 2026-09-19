import numpy as np, warnings
warnings.simplefilter("ignore")
from pygam import LinearGAM, LogisticGAM, PoissonGAM, s
rng=np.random.default_rng(0)
n=300
x=rng.uniform(0,1,n); z=rng.uniform(0,1,n)
y=np.sin(2*np.pi*x)+rng.normal(0,0.5,n)
X=np.c_[x,z]
g=LinearGAM(s(0)+s(1)).fit(X,y)
xt=np.c_[np.linspace(0.01,.99,5),np.full(5,.5)]
pd0,ci=g.partial_dependence(0,X=xt,width=.95)
print("pd",pd0); print("ci halfwidth",(ci[:,1]-ci[:,0])/2)
print("mean of term over training", g.partial_dependence(0,X=X).mean())
print("conf",g.confidence_intervals(xt)); print("pred",g.prediction_intervals(xt))
print(g.statistics_['p_values'], g.statistics_['edof'], g.lam)
g.summary()
gb=LogisticGAM(s(0)+s(1)).fit(X,(rng.uniform(size=n)<1/(1+np.exp(-2*np.sin(2*np.pi*x)))).astype(int))
print(gb.statistics_['pseudo_r2'], gb.statistics_['loglikelihood'])
