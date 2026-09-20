"""Plain fit (no logging) for py-spy.  Usage: prof_fit.py FAMILY N DESIGN"""
import sys, time, threading, resource
import numpy as np
fam, n, design = sys.argv[1], int(float(sys.argv[2])), sys.argv[3]
rng = np.random.default_rng(0)
if design == "te":
    X = rng.uniform(0, 1, (n, 2)); eta = np.sin(2*np.pi*X[:, 0])*np.cos(2*np.pi*X[:, 1])
else:
    p = int(design[1:]); X = rng.uniform(0, 1, (n, p)); eta = np.zeros(n)
    for j in range(p): eta += np.sin(2*np.pi*X[:, j]+j)/np.sqrt(p)
y = {"gaussian": lambda: eta + rng.normal(0, 0.5, n),
     "binomial": lambda: (rng.uniform(size=n) < 1/(1+np.exp(-1.5*eta))).astype(float),
     "poisson": lambda: rng.poisson(np.exp(0.5+0.7*eta)).astype(float)}[fam]()
names = [f"x{j}" for j in range(X.shape[1])]
data = {nm: X[:, j] for j, nm in enumerate(names)}; data["y"] = y
formula = "y ~ te(x0, x1)" if design == "te" else "y ~ " + " + ".join(f"s({nm})" for nm in names)
import gamfit, pandas  # noqa
trace = []
stop = False
def poll():
    t0 = time.perf_counter()
    while not stop:
        with open("/proc/self/statm") as f:
            rss = int(f.read().split()[1]) * 4096 / 2**20
        trace.append((time.perf_counter() - t0, rss)); time.sleep(0.2)
th = threading.Thread(target=poll, daemon=True); th.start()
t = time.perf_counter(); c = time.process_time()
m = gamfit.fit(data, formula, family=fam)
tf = time.perf_counter() - t
print(f"fit wall={tf:.3f} cpu={time.process_time()-c:.3f} peak={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MB", file=sys.stderr)
stop = True
# RSS timeline, decimated
step = max(1, len(trace)//40)
print("rss_timeline", " ".join(f"{a:.1f}:{b:.0f}" for a, b in trace[::step]), file=sys.stderr)
