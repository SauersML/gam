import multiprocessing as mp, numpy as np, warnings
warnings.simplefilter("ignore")
import pygam
rng=np.random.default_rng(0); x=rng.uniform(0,1,400); y=np.sin(6*x)+rng.normal(0,.3,400)
def job(s): return float(pygam.LinearGAM(pygam.s(0)).fit(x[:,None],y).predict(x[:1,None])[0])
if __name__=="__main__":
    print("parent", job(0))
    with mp.get_context("fork").Pool(2) as p: print("fork children", p.map_async(job, range(4)).get(timeout=60))
