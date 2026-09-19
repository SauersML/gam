import os; os.environ["RAYON_NUM_THREADS"]="1"
import numpy as np, gamfit, warnings; warnings.filterwarnings("ignore")
import pygam.datasets.load_datasets as L; L.PATH=os.environ.get("PYGAM_DATA_DIR", os.path.expanduser("~/.cache/gamfit-bench/pygam_data"))
from pygam import PoissonGAM, s
from sklearn.model_selection import KFold
X,y=L.coal(); x=X[:,0]; y=np.asarray(y,float)
def pdev(y,m):
    t=np.where(y>0,y*np.log(np.where(y>0,y,1)/m),0); return np.mean(2*(t-(y-m)))
res={}
for k,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=0).split(x)):
    for lab,fml in [("def","y ~ s(x)"),("k20","y ~ s(x, k=20)"),("k30","y ~ s(x, k=30)")]:
        m=gamfit.fit({"x":x[tr],"y":y[tr]},fml,family="poisson")
        res.setdefault(lab,[]).append(pdev(y[te],m.predict({"x":x[te]})))
        res.setdefault(lab+"_edf",[]).append(m.summary().edf_total)
    g=PoissonGAM(s(0)).gridsearch(x[tr,None],y[tr],progress=False)
    res.setdefault("pygam_grid",[]).append(pdev(y[te],g.predict_mu(x[te,None]))); res.setdefault("pg_edf",[]).append(g.statistics_['edof'])
for k,v in res.items(): print(k, np.round(v,3), np.mean(v))
