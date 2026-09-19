import numpy as np, os
import pygam.datasets.load_datasets as L
L.PATH=os.environ.get("PYGAM_DATA_DIR", os.path.expanduser("~/.cache/gamfit-bench/pygam_data"))
for name in ['mcycle','coal','faithful','wage','trees','default','cake','hepatitis','toy_classification','head_circumference','chicago','toy_interaction']:
    kw={'n':5000} if name=='toy_interaction' else {}
    np.random.seed(0)
    X,y=getattr(L,name)(**kw)
    print(name, X.shape, y.shape, 'y uniq',len(np.unique(y)), 'ymin',y.min(),'ymax',y.max(), [len(np.unique(X[:,j])) for j in range(X.shape[1])])
