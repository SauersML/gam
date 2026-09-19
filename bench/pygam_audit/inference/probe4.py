import numpy as np, gamfit, warnings, time
warnings.simplefilter("ignore")
rng=np.random.default_rng(1)
n=200
x=rng.uniform(0,1,n); z=rng.uniform(0,1,n)
eta=-2+3*np.exp(-((x-0.3)/0.15)**2)
yb=rng.binomial(1,1/(1+np.exp(-eta)))
yp=rng.poisson(np.exp(0.5+np.sin(2*np.pi*x)))
xt=np.linspace(0.01,0.99,7)
for fam,y in [("binomial",yb),("poisson",yp)]:
    t=time.time()
    m=gamfit.fit(dict(x=x,z=z,y=y.astype(float)),"y ~ s(x) + s(z)",family=fam)
    print(fam,"fit",time.time()-t, m.family_name)
    p=m.predict(dict(x=xt,z=np.full(7,0.5)),interval=0.95,observation_interval=True)
    for k,v in p.items(): print(" ",k,np.round(v,3) if not isinstance(v,str) else v)
    print(m.summary().smooth_terms)
    pdp=m.partial_dependence("s(x)",dict(x=x,z=z,y=y.astype(float)),grid=xt)
    print(pdp)
