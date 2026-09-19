import warnings, sys, time, json, os
warnings.simplefilter("ignore")
os.environ.setdefault("RAYON_NUM_THREADS","1")
from sklearn.utils.estimator_checks import check_estimator
from gamfit.sklearn import GAMRegressor, GAMClassifier
which=sys.argv[1]; formula=sys.argv[2]
est = GAMRegressor(formula=formula) if which=="reg" else GAMClassifier(formula=formula)
t=time.time()
res=check_estimator(est, on_fail=None)
out=[]
for r in res:
    st=r["status"]; ex=r.get("exception")
    out.append((r["check_name"], st, (type(ex).__name__+": "+str(ex).replace("\n"," ")[:300]) if ex else ""))
print(f"{which} formula={formula!r} total={len(out)} time={time.time()-t:.0f}s")
from collections import Counter
print(Counter(s for _,s,_ in out))
for n,s,e in out:
    if s!="passed": print(f"  {s:8s} {n}: {e}")
