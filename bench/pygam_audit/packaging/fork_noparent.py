import multiprocessing as mp, numpy as np, warnings, sys
warnings.simplefilter("ignore")
import gamfit
rng=np.random.default_rng(0); n=400
x=rng.uniform(0,1,n); y=np.sin(6*x)+rng.normal(0,.3,n)
D={"x":x,"y":y}
def job(seed):
    m=gamfit.fit(D,"y ~ s(x)")
    return float(np.asarray(m.predict({"x":x[:3]}))[0])
if __name__=="__main__":
    print("no parent fit", flush=True)
    ctx=mp.get_context(sys.argv[1])
    with ctx.Pool(2) as p:
        r=p.map_async(job, range(4))
        print(sys.argv[1], "children", r.get(timeout=45), flush=True)
