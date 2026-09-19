# gamfit default basis cap (8 internal knots => 12 coefs) regardless of n: underfits sin(8 pi x) even at n=5000.
import warnings
import numpy as np, gamfit, io, contextlib
rng=np.random.default_rng(2)
for n in [500,5000]:
    x=rng.uniform(0,1,n); f=np.sin(8*np.pi*x); y=f+rng.normal(0,.3,n); d={"x":x,"y":y}
    with warnings.catch_warnings(record=True) as w, contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("always")
        m=gamfit.fit(d,"y ~ s(x)")
        bc=m.basis_check(d) if 'data' in m.basis_check.__code__.co_varnames else m.basis_check()
        m2=gamfit.fit(d,"y ~ s(x, k=30)")
    mu=np.asarray(m.predict(d)).ravel(); mu2=np.asarray(m2.predict(d)).ravel()
    print(f" n={n}: default rmse={np.sqrt(np.mean((mu-f)**2)):.4f} coefs={len(m.summary().coefficients)} | k=30 rmse={np.sqrt(np.mean((mu2-f)**2)):.4f}")
    print("  warnings:",[str(x.message)[:110] for x in w][:3])
    print("  basis_check:",str(bc)[:400])
