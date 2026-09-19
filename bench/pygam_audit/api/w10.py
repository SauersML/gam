import numpy as np, warnings; warnings.simplefilter("ignore")
from gamfit.sklearn import GAMClassifier
rng=np.random.default_rng(0); X=rng.normal(size=(60,1)); y=X[:,0]+rng.normal(size=60)
try: GAMClassifier(formula="s(x0)").fit(X,y)
except Exception as e: print("continuous y:", type(e).__name__, str(e)[:200])
try: GAMClassifier(formula="s(x0)").fit(X,None)
except Exception as e: print("y None:", type(e).__name__, str(e)[:200])
import scipy.sparse as sp
try: GAMClassifier(formula="s(x0)").fit(sp.csr_matrix(X),(y>0).astype(int))
except Exception as e: print("sparse:", type(e).__name__, str(e)[:200])
