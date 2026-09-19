# gamfit Model.evidence / compare_models: pyffi recomputes -2ll+2edf (no scale dof, no Wood-Pya-Safken correction)
# although gam-inference/src/model_comparison.rs:447-450 computes aic_conditional (+scale_dof) and aic_corrected.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, gamfit, io, contextlib, sys
R=int(sys.argv[1]) if len(sys.argv)>1 else 1
rng=np.random.default_rng(7); n=200; picks=0
for r in range(R):
    x=rng.uniform(0,1,n); z=rng.uniform(0,1,n); y=np.sin(2*np.pi*x)+rng.normal(0,.5,n); d={"x":x,"z":z,"y":y}
    with contextlib.redirect_stdout(io.StringIO()):
        m0=gamfit.fit(d,"y ~ s(x)"); m1=gamfit.fit(d,"y ~ s(x) + s(z)")
    s0=m0.summary()
    if r==0:
        print(" conditional_aic =",m0.evidence," -2ll+2edf =",-2*s0.log_likelihood+2*s0.edf_total,
              " -2ll+2(edf+1) [Rust aic_conditional, Gaussian scale dof] =",-2*s0.log_likelihood+2*(s0.edf_total+1))
    picks+= m1.evidence < m0.evidence
print(" null s(z) model preferred by conditional_aic in %d/%d reps"%(picks,R))
