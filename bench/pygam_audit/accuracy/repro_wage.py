import os; os.environ["RAYON_NUM_THREADS"]="1"
import numpy as np, gamfit, warnings; warnings.filterwarnings("ignore")
import pygam.datasets.load_datasets as L; L.PATH=os.environ.get("PYGAM_DATA_DIR", os.path.expanduser("~/.cache/gamfit-bench/pygam_data"))
from sklearn.model_selection import KFold
X,y=L.wage(); y=np.asarray(y,float)
for k,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=0).split(X)):
    d={"year":X[tr,0],"age":X[tr,1],"edu":X[tr,2],"y":y[tr]}
    try:
        m=gamfit.fit(d,"y ~ s(year) + s(age) + factor(edu)",family="gaussian")
        print(k,"ok edf",m.summary().edf_total)
    except Exception as e:
        print(k,"FAIL",type(e).__name__,str(e)[:300])
        # try: which term triggers it
        for f in ["y ~ s(year)","y ~ s(age) + factor(edu)","y ~ s(year) + factor(edu)","y ~ s(year) + s(age)"]:
            try: gamfit.fit(d,f,family="gaussian"); print("   ",f,"ok")
            except Exception as e2: print("   ",f,"FAIL",type(e2).__name__)
        np.savez(f"wage_fail_fold{k}.npz", **d)
