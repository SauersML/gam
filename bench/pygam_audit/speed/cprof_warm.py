import cProfile, pstats, sys, time, io
import numpy as np
n=int(sys.argv[1]); formula=sys.argv[2]
rng=np.random.default_rng(0)
X=rng.uniform(0,1,(n,5)); y=np.sin(2*np.pi*X[:,0])+rng.normal(0,0.3,n)
data={f"x{j}":X[:,j] for j in range(5)}; data['y']=y
import gamfit
t=time.perf_counter(); c=time.process_time(); m=gamfit.fit(data,formula,family="gaussian"); print("cold",time.perf_counter()-t, time.process_time()-c, file=sys.stderr)
pr=cProfile.Profile(); pr.enable()
t=time.perf_counter(); c=time.process_time(); m=gamfit.fit(data,formula,family="gaussian"); print("warm",time.perf_counter()-t,time.process_time()-c, file=sys.stderr)
t=time.perf_counter(); p=m.predict(data); print("pred",time.perf_counter()-t, file=sys.stderr)
pr.disable()
s=io.StringIO(); pstats.Stats(pr,stream=s).sort_stats('cumulative').print_stats(30); print(s.getvalue()[:6000])
