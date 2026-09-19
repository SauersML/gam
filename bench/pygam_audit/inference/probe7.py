import numpy as np, gamfit, warnings, json
warnings.simplefilter("ignore")
rng=np.random.default_rng(3)
n=200
X=rng.uniform(0,1,(n,3))
y=np.sin(2*np.pi*X[:,0])+0.3*np.cos(2*np.pi*X[:,2])+rng.normal(0,0.7,n)
d=dict(x1=X[:,0],x2=X[:,1],x3=X[:,2],y=y)
m=gamfit.fit(d,"y ~ s(x1)+s(x2)+s(x3)")
b=m.dumps()
print(type(b), len(b), b[:20])
try:
    j=json.loads(b)
except Exception as e:
    print("not json",e); j=None
def walk(o,path=""):
    if isinstance(o,dict):
        for k,v in o.items():
            p=path+"/"+k
            if any(s in k.lower() for s in ("rho_post","k_hat","scale","dispersion","phi","sigma","adequacy","rho_cov","corrected")): print(p, str(v)[:200])
            walk(v,p)
    elif isinstance(o,list) and o and isinstance(o[0],(dict,list)):
        for i,v in enumerate(o[:3]): walk(v,path+f"[{i}]")
if j: walk(j)
