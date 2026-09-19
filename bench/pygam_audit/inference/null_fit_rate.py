import numpy as np, gamfit, warnings, time, sys
warnings.simplefilter("ignore")
fam = sys.argv[1]; nrep=int(sys.argv[2]); k=int(sys.argv[3])
fails=0; msgs={}; ts=[]
for rep in range(nrep):
    rng=np.random.default_rng(7000+rep); n=200
    X=rng.uniform(0,1,(n,k))
    y = rng.normal(size=n) if fam=="gaussian" else (rng.poisson(2.0,size=n).astype(float) if fam=="poisson" else (rng.uniform(size=n)<.5).astype(float))
    d={f"x{j}":X[:,j] for j in range(k)}; d["y"]=y
    f="y ~ "+" + ".join(f"s(x{j})" for j in range(k))
    t=time.time()
    try:
        m=gamfit.fit(d,f,family=fam)
    except Exception as e:
        fails+=1; key=str(e)[:80]; msgs[key]=msgs.get(key,0)+1
    ts.append(time.time()-t)
print(fam, "k=",k, "fails", fails, "/", nrep, "median time", np.median(ts), "max", max(ts)); print(msgs)
