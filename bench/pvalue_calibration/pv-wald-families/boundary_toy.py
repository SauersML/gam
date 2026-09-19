"""One-direction REML boundary toy: z ~ N(0,1), REML shrinkage tau = max(1 - 1/z^2, 0),
T = tau z^2 = max(z^2 - 1, 0), p = chi2_1.sf(T). Exact size and conditional-KS power."""
import numpy as np
from scipy import stats
edges = np.array([0, .01, .05, .1, .2, .3, .4, .5])
def cdf(a):  # P(p <= a)
    return stats.chi2.sf(stats.chi2.isf(a, 1) + 1, 1)
print("size .10 .05 .01:", [round(cdf(a), 4) for a in (.1, .05, .01)], " P(p<.5)=", round(cdf(.5), 4))
c = np.array([cdf(a) if a > 0 else 0 for a in edges]) / cdf(.5)
print("toy obs/exp per bin:", " ".join(f"{v:5.2f}" for v in np.diff(c) / (np.diff(edges) / .5)))
rng = np.random.default_rng(1)
for m in (100, 150, 250):
    rej = 0
    for _ in range(2000):
        z = rng.standard_normal(3000)
        p = stats.chi2.sf(np.maximum(z**2 - 1, 0), 1)
        low = p[p < .5][:m] / .5
        rej += stats.kstest(low, "uniform").pvalue < .05
    print(f"KS(p<.5) rejection rate at {m} sub-.5 samples: {rej/2000:.3f}")
