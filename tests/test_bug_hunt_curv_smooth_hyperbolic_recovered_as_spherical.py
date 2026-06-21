import io, contextlib, math, numpy as np, pandas as pd
import gamfit

def curved(kappa_star, seed=1, n=600, radius=0.68, noise=0.02):
    rng = np.random.default_rng(seed); root = math.sqrt(abs(kappa_star))
    x1=[]; x2=[]; y=[]
    while len(y) < n:
        a, b = 2*rng.random()-1, 2*rng.random()-1
        if a*a+b*b > 1.0: continue
        u, v = a*radius, b*radius; r = math.hypot(u, v)
        if kappa_star < 0:   d = 2*math.atanh(min(root*r, 1-1e-9))/root   # hyperbolic
        elif kappa_star > 0: d = 2*math.atan(root*r)/root                 # spherical
        else:                d = 2*r
        x1.append(u); x2.append(v); y.append(2*math.exp(-d)-1 + noise*rng.standard_normal())
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})

for ks in (+2.0, -2.0):
    df = curved(ks)
    rep = gamfit.fit(df, "y ~ curv(x1, x2, centers=10)").curvature(df)[0]
    print(f"truth kappa*={ks:+}: kappa_hat={rep['kappa_hat']:+.4f} "
          f"ci=({rep['ci_lo']:+.3f},{rep['ci_hi']:+.3f}) verdict={rep['verdict']} flat_p={rep['flatness_p_value']:.3g}")
