"""Reproducer: `gam fit --transformation-normal` fails to converge on Gamma
distributed y data, hitting 100 cycles of inner-solve non-convergence and
then erroring out.

Repro is deterministic (seed=7, n=500). Truth is y ~ Gamma(2, mean=1+0.7sin(2πx)/2)
— a skewed positive-y distribution that transformation-normal should
handle (its whole point is to monotonically map y to a normal latent).

Observed error message:
  error: transformation-normal fit failed: transformation fit failed:
  custom-family optimization error: outer smoothing optimization final
  inner refit did not converge after 100 cycles.

Updated diagnosis (after probing): bumping inner_max_cycles from 100 to
300 does NOT fix the failure — at cycle 299 the inner Newton is still
proposing step 0.9 (vs tol 0.004) with `beta_inf` ≈ 3713 (way beyond
the y range 0–6). The joint Newton is oscillating, not slowly
converging, almost certainly because the joint Hessian over
(β, monotone-ψ) is rank-deficient in some direction at extreme β.

Real root cause candidates (open for the next cycle):
1. Joint inner Hessian needs Levenberg–Marquardt damping in the joint
   Newton when ‖β‖ grows past a sane bound. Currently the step
   acceptance has no LM guard — every full step is accepted even when
   obj barely moves.
2. Initial response-basis ψ⁰ is mis-scaled when `max(y) ≫ mean(y)` —
   the monotone basis evaluated on the heavy tail produces extreme
   leverage rows. A heavy-tail-aware ψ⁰ (e.g. quantile-spaced knots
   from `log(1 + y)` rather than from raw y) would damp this.
3. Outer REML strategy is asking for ρ values that force β extreme;
   a ρ box-constraint or step trust radius would prevent the inner
   from being asked to solve an ill-posed sub-problem.

Once fixed, run this script and expect a successful fit + sensible
predictions.
"""
import subprocess
import tempfile
from pathlib import Path
import numpy as np

GAM = Path(__file__).resolve().parents[1] / "target" / "release" / "gam"
rng = np.random.default_rng(7)
n = 500
x = rng.uniform(0, 1, n)
mean_x = 1.0 + 0.7 * np.sin(2*np.pi*x)
y = rng.gamma(shape=2.0, scale=mean_x / 2.0)

tmp = Path(tempfile.mkdtemp())
train = tmp / "train.csv"
predd = tmp / "pred.csv"
train.write_text("x,y\n" + "\n".join(f"{a:.10g},{b:.10g}" for a, b in zip(x, y)))
predd.write_text("x,y\n" + "\n".join(f"{v},0" for v in [0.1, 0.25, 0.5, 0.75, 0.9]))

print(f"# y range: [{y.min():.3f}, {y.max():.3f}] mean {y.mean():.3f}")
print("# Fitting transformation-normal...")
fit = subprocess.run(
    [str(GAM), "fit", "--transformation-normal", str(train),
     "y ~ smooth(x)", "--out", str(tmp / "m.bin")],
    capture_output=True, text=True,
)
print(f"fit rc={fit.returncode}")
if fit.returncode != 0:
    print("STDERR (last 5 lines):")
    for ln in fit.stderr.strip().splitlines()[-5:]:
        print(" ", ln)
else:
    pred = subprocess.run(
        [str(GAM), "predict", str(tmp / "m.bin"), str(predd), "--out", str(tmp / "p.csv")],
        capture_output=True, text=True,
    )
    print(f"predict rc={pred.returncode}")
    if pred.returncode == 0:
        print("predictions:")
        print(open(tmp / "p.csv").read())
