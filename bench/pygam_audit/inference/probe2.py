import numpy as np, gamfit, inspect, json, warnings
warnings.simplefilter("ignore")
print(inspect.signature(gamfit.Model.partial_dependence))
print(inspect.signature(gamfit.Model.sample))
print(inspect.signature(gamfit.Model.difference_smooth))
rng=np.random.default_rng(0)
n=300
x=rng.uniform(0,1,n); z=rng.uniform(0,1,n)
y=np.sin(2*np.pi*x)+rng.normal(0,0.5,n)
df=dict(x=x,z=z,y=y)
m=gamfit.fit(df,"y ~ s(x) + s(z)")
s=m.summary()
for k in ['coefficient_se_source','covariance_kind','deviance','edf_total','log_likelihood','lambdas','smooth_terms','extras','convergence','reml_score','n_obs']:
    print(k, getattr(s,k))
print(s.smooth_terms_frame if not callable(s.smooth_terms_frame) else s.smooth_terms_frame())
pdp=m.partial_dependence("s(x)", df) if 'data' in str(inspect.signature(m.partial_dependence)) else None
print({k:(v[:3] if hasattr(v,'__len__') and not isinstance(v,str) else v) for k,v in pdp.items()})
