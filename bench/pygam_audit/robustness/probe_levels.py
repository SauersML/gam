import gamfit, numpy as np, warnings, sys, time
warnings.simplefilter("ignore")
L=int(sys.argv[1]); n=int(sys.argv[2]); form=sys.argv[3]
r=np.random.default_rng(18)
g=r.integers(0,L,n); eff=r.normal(0,1,L); x=r.uniform(0,1,n)
y=np.sin(6*x)+eff[g]+r.normal(0,.3,n)
gs=np.array([f"L{i}" for i in g],dtype=object)
t=time.time()
m=gamfit.fit({"x":x,"g":gs,"y":y},form)
s=m.summary()
print(L,n,form,"t=%.1f"%(time.time()-t),"edf",round(s.edf_total,1),"certified",s.convergence.get("certified"),"outer",s.convergence.get("outer_iterations"))
