# ACC-6: the ρ-marginal predictive mean, measured and not shipped

## The finding

`accuracy.md` scores coal (Poisson, n = 150) a TIE: gamfit 1.175 against
pyGAM 1.131 held-out deviance. It calls the coal optimum "a genuinely flat
LAML optimum" with one fold at edf 2, and proposes as an optional improvement
(M) to integrate the predictive mean over the ρ posterior:

    E[μ(x) | y] = ∫ E[μ(x) | ρ, y] p(ρ | y) dρ.

The brief says that if this integral gives no measurable improvement anywhere,
the evidence should be reported instead of shipping the complexity. **It gives
no consistent improvement.** On independent replicates the first-order
ρ-marginal mean is:

- measurably better on one generator: binom_sin2_n500, −0.24% truth-MSE
  (t = −3.71);
- measurably worse on another of the same family: binom_add4_n300, +0.33%
  (t = +2.31);
- indistinguishable from the shipped mean on the other five, with the
  coal-shaped generator leaning worse (+1.27%, t = +1.74).

The full integral, with the mean shift, is worse wherever its effect is
measurable, including on binom_sin2_n500. The shipped estimand is unchanged,
and `tests/rho_marginal_predictive_mean_acc6_test.py` pins it.

## What `predict` returns today

`posterior_mean` is the conditional posterior mean at ρ̂ (#398):

    E[g⁻¹(η)],  η ~ N(xᵀβ̂, xᵀ V_β x)

For the log link this is the closed form `exp(η + s²/2)`, and the logit
expectation is evaluated numerically. The battery below reconstructs the same
value from the fit's exported affine design (`Model.design_matrix`). It
matches the shipped `predict` to a maximum relative error of 5.6e-12 on every
fold of every case, so the comparison is against the exact shipped number.

## The two ρ-marginal estimators measured

### 1. First-order Laplace (`rho_marginal_bench.py`, gamfit itself)

Take the Laplace posterior ρ | y ≈ N(ρ̂, V_ρ), with V_ρ the inverse analytic
REML ρ-Hessian, and write δ = ρ − ρ̂ and J = ∂β̂/∂ρ. Then

    β̂(ρ) = β̂ + J δ + O(δ²)
    E[β | y]   = β̂ + O(tr V_ρ)
    Cov[β | y] = V_β + J V_ρ Jᵀ + O(‖V_ρ‖²)

This covariance is gamfit's smoothing-corrected `V_p`. So the first-order
ρ-marginal mean is the same inverse-link expectation with `V_p` in place of
`V_β`:

- It reuses the fit's existing factorizations and needs no extra solves.
- The cost is one quadratic form per prediction row, so it is O(n p²) at any n.
- The mean shift E[β̂(ρ)] − β̂ is second order.
- For an identity link the prediction is **exactly** the plug-in, so every
  Gaussian case is unchanged by construction.

Two details of the implementation:

- **The expectations.** The log link uses the closed form. The logit link uses
  the trapezoid rule on z ∈ [−12, 12] with 8001 nodes. The integrand is
  analytic in a strip, so the rule converges geometrically. The truncated
  normal tail is below 1e-32, and against adaptive quadrature the error is at
  most 5e-9 for s ∈ [0.01, 20] and η ∈ [−15, 15]. Gauss-Hermite is not used:
  the logistic's poles at η + s z = ±iπ limit it to about 1e-3 at s = 8.
- **What V_p contains.** On flat or near-boundary surfaces gamfit's `V_p` is
  already *more* than first order. The sigma-point cubature correction
  `compute_smoothing_correction_auto` (`crates/gam-solve/src/reml/eval.rs`)
  replaces the linearization with φ̂·E_ρ[H(ρ)⁻¹] + Cov_ρ[β̂(ρ)], using 2r
  calibrated spherical nodes of the same Laplace posterior. So this estimator
  is the "first-order correction where it is adequate, cubature where it is
  not" that the brief asks for. Those nodes already carry β̂(ρ_m), so if the
  mean shift were ever wanted, E_ρ[β̂] would come at no extra cost.

### 2. Full Laplace (`full_laplace_folds.py`, `full_laplace_replicates.py`)

The second estimator also keeps the second-order mean shift that the first one
drops. It is an independent REML/LAML re-implementation, for diagnostics only
and never shipped: FD ρ-Hessian, Nelder-Mead; the library never imports it.
The model is:

- 1-D cubic B-spline with k = 20 basis functions.
- A second-derivative penalty plus a null-space penalty, with a sum-to-zero
  constraint.
- A tensor Gauss-Hermite rule with 15 points per active ρ direction over
  N(ρ̂, V_ρ), exact for polynomials of degree ≤ 29 in the standardized ρ.
  Directions with Hessian eigenvalue ≤ 1e-8 are flat and carry no Laplace
  mass.
- At each node it evaluates the full conditional mean E[μ | ρ_m, y], so the
  mean shift is included.

This estimator is the estimand ACC-6 proposes, evaluated without the
first-order truncation. If the truncation were hiding a gain, this estimator
would show it.

## Results

### Coal, before and after

5-fold CV (the bench_accuracy splits), mean held-out deviance:

| estimator | coal dev | vs shipped |
|---|---:|---:|
| plug-in g⁻¹(η̂) | 1.17671 | |
| **shipped** (conditional, V_β) | **1.17691** | |
| first-order ρ-marginal (V_p) | 1.17551 | −0.13%, paired fold t = −0.61 |
| full Laplace, 15ʳ GH (independent fit) | +0.19% vs its own plug-in | worse |
| pyGAM (accuracy.md) | 1.131 | |

Per fold:

| fold | edf | plug-in | shipped | ρ-marginal (V_p) | median var increase V_p vs V_β | full Laplace plug-in | full Laplace GH |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.97 | 1.10782 | 1.10442 | 1.10437 | +0.7% | 1.09614 | 1.09614 |
| 1 | 5.81 | 0.94532 | 0.94641 | 0.94502 | +32.3% | 0.94629 | 0.94233 |
| 2 | 5.24 | 1.46139 | 1.46218 | 1.45962 | +37.7% | 1.55983 | 1.55983 |
| 3 | 5.38 | 1.24216 | 1.25555 | 1.26192 | +40.9% | 1.25537 | 1.27051 |
| 4 | 5.36 | 1.12686 | 1.11598 | 1.10663 | +48.9% | 1.17406 | 1.17406 |

The flat LAML direction is the direction toward infinite smoothing. On fold 0
(edf 1.97) the prediction variance grows by a median of only 0.7% under V_p,
because along that direction ∂β̂/∂ρ → 0: the surface is flat *because* the fit
no longer changes. Integrating over a direction in which the prediction does
not move cannot move the prediction. The folds where V_p does move it (+32% to
+49% median variance) go both ways: fold 4 improves and fold 3 gets worse.
The coal gap to pyGAM is not smoothing-parameter uncertainty.

On independent datasets drawn from a coal-shaped rate (`coal_like_n150`),
where the paired test is valid, gamfit's ρ-marginal mean is 1.27% *worse* in
truth-MSE than the shipped mean (t = +1.74; gamfit replicate table below).
With the full integral the median change is zero on the same generator. Its
mean (+15%) comes from four replicates on which the diagnostic's Laplace
approximation fails at a boundary optimum (see the note under the full-Laplace
replicate table).

### Whole `bench_accuracy.py` battery (first-order ρ-marginal vs shipped)

Paired over the folds where all three predictions exist. CV folds share
training data, so the fold t overstates significance; it is shown only to
separate noise from signal.

| case | family | n | metric | folds | plugin | conditional (shipped) | rho-marginal | change | paired t |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| badhealth | poisson | 1127 | dev | 5/5 | 3.11538 | 3.11515 | 3.11503 | -0.003% | -0.46 |
| binom_add4_n1000 | binomial | 1000 | truth_mse | 5/5 | 0.00418124 | 0.00426413 | 0.00428641 | +0.469% | +3.36 * |
| binom_add4_n300 | binomial | 300 | truth_mse | 4/5 | 0.0160346 | 0.0148109 | 0.0145836 | -1.505% | -1.55 |
| binom_sin2_n3000 | binomial | 3000 | truth_mse | 5/5 | 0.000798631 | 0.000799615 | 0.00079926 | -0.042% | -1.84 |
| binom_sin2_n500 | binomial | 500 | truth_mse | 5/5 | 0.00348469 | 0.00347541 | 0.00345914 | -0.444% | -3.41 * |
| chicago | poisson | 4863 | dev | 5/5 | 1.51094 | 1.5108 | 1.51072 | -0.006% | -1.02 |
| coal | poisson | 150 | dev | 5/5 | 1.17671 | 1.17691 | 1.17551 | -0.131% | -0.61 |
| faithful | poisson | 200 | dev | 5/5 | 1.29798 | 1.29977 | 1.30016 | +0.027% | +0.51 |
| gamma_add2_n2000 | gamma | 2000 | truth_mse | 5/5 | 0.0533839 | 0.0530536 | 0.0531111 | +0.092% | +0.31 |
| gamma_add2_n300 | gamma | 300 | truth_mse | 5/5 | 0.197985 | 0.215007 | 0.219347 | +2.077% | +5.10 * |
| haberman | binomial | 306 | dev | 4/5 | 1.09102 | 1.08744 | 1.08701 | -0.038% | -1.90 |
| heart_failure | binomial | 299 | dev | 3/5 | 1.04969 | 1.04263 | 1.0419 | -0.064% | -1.03 |
| nearsep_n200 | binomial | 200 | truth_mse | 1/5 | 0.0024787 | 0.00278859 | 0.00279569 |  |  |
| pois_add2_n2000 | poisson | 2000 | truth_mse | 5/5 | 0.0463528 | 0.0468199 | 0.0468365 | +0.033% | +0.88 |
| pois_add2_n300 | poisson | 300 | truth_mse | 5/5 | 0.154867 | 0.157076 | 0.156929 | -0.121% | -2.59 |
| pois_lowcount_n500 | poisson | 500 | truth_mse | 5/5 | 0.00500477 | 0.00490617 | 0.00490599 | -0.003% | -0.02 |
| prostate_pc | binomial | 654 | dev | 5/5 | 1.22828 | 1.22917 | 1.22943 | +0.022% | +3.26 * |
| trees | gamma | 31 | dev | 5/5 | 0.00724614 | 0.00729679 | 0.00737107 | +1.165% | +1.53 |

Flagged (|t| above the 5% quantile) improvements: 1; regressions: 3.

Gaussian cases (40), all identical to the shipped prediction: add4_n1000, add4_n200, add4_n5000, bike, bump2d_n1000, bump2d_n4000, cake, city_temp, g1d_doppler_n100, g1d_doppler_n2000, g1d_doppler_n500, g1d_sin1_n100, g1d_sin1_n2000, g1d_sin1_n500, g1d_sin3_n100, g1d_sin3_n10000, g1d_sin3_n2000, g1d_sin3_n500, g1d_sin6_n100, g1d_sin6_n10000, g1d_sin6_n2000, g1d_sin6_n500, gagurine, head_circumference, hepatitis, hetero_n3000, hetero_n500, lidar, mcycle, nottem, null3_n200, null3_n2000, outlier_n2000, outlier_n300, penguins_mass, quakes, quakes_space, sleepstudy, toy_interaction, wage.

Every Gaussian case is identical by construction (identity link). On the
non-Gaussian cases:

- The changes are at most ±2.1%.
- Their signs are mixed.
- The few with a large fold t go both ways: binom_sin2_n500 −0.44% is
  better, while binom_add4_n1000 +0.47%, gamma_add2_n300 +2.08% and
  prostate_pc +0.02% are worse.

The pois_add2, pois_lowcount and gamma_add2 rows were run on a build of the
merged main branch. The other rows were run on the branch point. Each row
compares the three predictions from the same fits, so the build only matters
within a row.
`nearsep_n1000` has no row, because its first fold never finished (see the
side findings).

### gamfit on independent replicates (`rho_marginal_replicates.py`, R = 40)

The fold t above is not a valid test. This one is: each replicate is a fresh
dataset from the same non-Gaussian generator (same truth, same n), gamfit is
fitted once, and truth-MSE is scored on 2000 fresh covariate draws. `*` marks
|t| above the two-sided 5% Student-t quantile. Replicates without an exported
`V_p` are left out of the pairing (none of the fits failed).

| generator | replicates with V_p | plug-in | conditional (shipped) | ρ-marginal (V_p) | marginal vs shipped | t | shipped vs plug-in | t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| binom_add4_n1000 | 39/40 | 0.0028489 | 0.002838 | 0.0028389 | +0.049% | +0.92 | +0.061% | +0.14 |
| binom_add4_n300 | 34/40 | 0.0082228 | 0.0081989 | 0.0082195 | +0.329% | +2.31 * | -0.562% | -0.65 |
| binom_sin2_n500 | 40/40 | 0.0030989 | 0.0030626 | 0.0030551 | -0.238% | -3.71 * | -0.983% | -1.43 |
| coal_like_n150 | 39/40 | 0.039221 | 0.039842 | 0.040129 | +1.270% | +1.74 | +2.836% | +1.30 |
| gamma_add2_n300 | 37/40 | 0.17755 | 0.17531 | 0.17548 | -0.017% | -0.05 | -1.250% | -0.97 |
| pois_add2_n300 | 40/40 | 0.30507 | 0.30767 | 0.30791 | +0.060% | +1.00 | +0.707% | +1.25 |
| pois_lowcount_n500 | 40/40 | 0.0066966 | 0.0067338 | 0.0067364 | +0.020% | +0.13 | +1.187% | +0.75 |

### Full Laplace on the CV folds (1-D cases, independent fit)

| case | family | full Laplace vs plug-in | paired fold t |
|---|---|---:|---:|
| coal | poisson | +0.19% | +0.57 |
| hepatitis | gaussian | -0.09% | +0.79 |
| g1d_sin1_n100 | gaussian | +2.89% | +2.19 |
| g1d_doppler_n500 | gaussian | +0.07% | +0.43 |
| hetero_n500 | gaussian | -1.11% | -0.89 |
| outlier_n300 | gaussian | -6.43% | -1.34 |
| binom_sin2_n3000 | binomial | -1.22% | -1.90 |
| faithful | poisson | -0.08% | -1.00 |
| lidar | gaussian | +0.09% | +0.68 |
| binom_sin2_n500 | binomial | -0.22% | +0.26 |
| pois_lowcount_n500 | poisson | -2.22% | -0.44 |

### Full Laplace on independent replicates (R = 40, truth-MSE on a 400-point grid)

These replicates are independent datasets, so the paired t is a valid test
here, unlike on overlapping folds.

| generator | replicates | first-order vs plug-in | t | median | full Laplace vs plug-in | t | median |
|---|---:|---:|---:|---:|---:|---:|---:|
| binom_sin2_n500 | 40/40 | -0.171% | -4.38 | -0.198% | +2.234% | +2.69 | +1.367% |
| coal_like_n150 | 39/40 (1 non-finite) | +0.070% | +0.63 | +0.000% | +15.453% | +2.12 | -0.000% |
| hetero_n500 | 40/40 | +0.000% | 0 (identity link) | +0.000% | +0.253% | +0.64 | +0.000% |
| outlier_n300 | 40/40 | +0.000% | 0 (identity link) | +0.000% | +7.767% | +1.25 | +0.520% |
| pois_lowcount_n500 | 40/40 | +0.160% | +0.94 | +0.251% | +2.070% | +2.95 | +2.062% |
| sin1_n100 | 40/40 | +0.000% | 0 (identity link) | +0.000% | +1.727% | +2.19 | +1.605% |

`coal_like_n150` needs a note. Its +15% mean is an artifact of this
diagnostic, not a property of the integral:

- **The median is zero and the mean is not.** The mean comes from four
  replicates (seeds 1017, 1025, 1027 and 1029) that lose 73% to 179%. Each one
  reproduces when rerun on its own.
- **Why those four lose.** On each of them ρ̂ lies on the diagnostic's lower
  bound for the null-space penalty (log λ₂ = −15, the linear trend left
  unpenalized).
  - Below that bound the LAML surface is flat, and above it the surface rises
    steeply: +41 LAML units at log λ₂ ≥ 7.4 on seed 1017, and +26 on seed
    1027.
  - The finite-difference Hessian across the bound sees a curvature near 1e-3
    and turns it into a Laplace standard deviation of 27 to 37 in log λ₂.
  - The Gaussian then puts a third of its mass on fits that shrink the linear
    trend away. That mass has posterior density e⁻²⁶ to e⁻⁴¹ relative to ρ̂,
    and on it the truth-MSE is 12 to 14 times the plug-in's.
  - The remaining nodes sit on the flat side, where the prediction does not
    move (truth-MSE ratio 1.000).
  - So the correct integral on these replicates is essentially the plug-in,
    and the loss measures the Gaussian approximation failing at a boundary
    optimum.
- **The non-finite replicate.** Seed 1028 gave a non-finite full-Laplace mean
  in the batch run, with two active directions. Its ρ̂ is on the same bound.
  - There the finite-difference Hessian depends on the inner solver's warm
    start: at the same ρ̂ its eigenvalues were [−0.057, 0.527] in one rerun
    and [0.069, 0.202] after a different fit history.
  - Rerun on its own, it has one active direction and is finite, equal to the
    plug-in to 2e-6.
  - It is excluded from the row above rather than replaced by the rerun.
- **The other generators do not have this problem.** On pois_lowcount_n500,
  binom_sin2_n500 and sin1_n100 the median and the mean agree, so their losses
  are broad and not driven by a few failed approximations.

## Conclusion

- **The first-order ρ-marginal mean.** It costs nothing to compute, but it has
  no consistent effect.
  - On gamfit's own independent replicates it is significantly better on one
    of seven generators (binom_sin2_n500, −0.24%) and significantly worse on
    another (binom_add4_n300, +0.33%).
  - Both are binomial, so there is no family or regime where it reliably
    wins.
  - On the coal-shaped generator it leans worse (+1.27%, t = +1.74), and on
    the rest it is within noise.
  - Making it the default would trade sub-percent changes of either sign.
- **The full Laplace integral, which includes the mean shift, is harmful
  wherever its effect is measurable.** On independent replicates:
  - pois_lowcount_n500: +2.07% (t = +2.95);
  - binom_sin2_n500: +2.23% (t = +2.69);
  - sin1_n100: +1.73% (t = +2.19).

  So the small first-order gain on binom_sin2_n500 is not the leading term of
  a larger gain from doing the integral properly. Doing the integral
  properly reverses it.
- **On the coal-shaped generator the full integral gains nothing.**
  - The median change over replicates is zero. The LAML surface is flat
    along directions in which the prediction does not move.
  - The large mean loss is the Laplace approximation failing where ρ̂ sits
    on a boundary, not the integral.
  - A shipped ρ-integral would need a posterior approximation that survives
    that case, to buy a change that is zero where the approximation holds.
- **That is expected.** REML's ρ̂ is a good point estimate for prediction.
  Averaging over ρ adds variance-driven bias through the curvature of g⁻¹
  without reducing error.

Nothing is shipped:

- `predict` keeps the conditional posterior mean, as SPEC requires (posterior
  mean, not MAP).
- No option is added.
- `V_p` stays where it belongs, in the interval widths.

## Side findings (not ACC-6; recorded for the relevant owners)

- **`default` (binomial, n = 10000).** A single fold did not finish within 31
  minutes of CPU time at HEAD on a contended 4-CPU box, and the run was
  stopped there. Its log shows:
  - Strong-Wolfe fallbacks at iterations 0 and 1.
  - ARC cost-stall escapes.
  - `[INDEF-HESS]` pair diagnostics.
  - "rho-posterior adequacy diagnostic refused at the converged rho: outer
    Hessian is not positive definite".
  - A `[CERTIFICATE]` line: "the criterion CONTRADICTS the reported negative
    curvature (λ_min = −1.26e-7)".

  The audit JSON has no timings, so this cannot be called a regression here.
- **`binom_add4_n300`, fold 4.** `gamfit.fit` raises `FitInputError: exact
  smoothing-corrected covariance unavailable: OuterHessianInverse … rho
  Hessian has negative curvature -2.511e-7 below the outer certificate's own
  bar 1.701e-7`. The fit converged, but the covariance export turns that into
  a fit failure.
- **`nearsep_n200`.** Folds 0 to 2 raise `RemlConvergenceError`: folds 0 and 2
  do not certify a stationary optimum, and fold 1 "declined a certified
  optimum that an evaluated state beats".
- **No `V_p` exported.** `covariance_smoothing_corrected` is `None` on some
  folds: haberman 0, heart_failure 0 and 4, nearsep_n200 3, null3_n2000 1 and
  wage 1. The same happens on 11 of the 280 gamfit replicate fits, all of
  which converged: binom_add4_n300 seeds 1000, 1001, 1005, 1024, 1029 and
  1038; gamma_add2_n300 seeds 1011, 1017 and 1037; binom_add4_n1000 seed
  1035; coal_like_n150 seed 1035. These replicates cannot be scored for the
  ρ-marginal mean, so they drop out of its pairing.
- **`nearsep_n1000` (binomial, n = 1000).** Its first fold never finished.
  - On the branch point it ran for over two hours of CPU time before the run
    was stopped.
  - On a build of merged main it ran for 22 minutes. Its log went silent at
    58 s, after an outer "cost-stall STUCK (NOT a flat valley)" line, and the
    run was stopped.
  - Three native stack samples all place the time inside the outer cost
    evaluation, in `RemlState::compute_cost_charging` →
    `block_local_quadrature_correction` →
    `BlockExcessTarget::excess_with_displaced_neg_score_batch` →
    `Gam784BlockTarget::likelihood_surface_at`.
  - The case is not measured here.
- **`toy_classification` (binomial, n = 5000, six terms).** Its 5 folds did
  not finish inside a 25-minute bench timeout on a 4-CPU box running about
  8 jobs, so it has no row in the battery table. The timeout belongs to this
  bench, not the library. The case is not measured here.
- **`V_p − V_β` can be indefinite.** This is not a bug. Under the cubature
  correction, E_ρ[H(ρ)⁻¹] can be smaller than H(ρ̂)⁻¹.

## Reproduce

The scripts import `bench_accuracy.py` from one directory up, which is the
audit branch's `bench/pygam_audit/accuracy/bench_accuracy.py`. Copy that file
into `bench/pygam_audit/accuracy/` first. Then run:

    python rho_marginal_bench.py [--only REGEX] [--no-big]   # -> results/<case>.json
    python rho_marginal_report.py                             # battery table
    python rho_marginal_replicates.py GENERATOR 40 1000       # -> results/gamfit_replicates_<GENERATOR>.json
    python full_laplace_folds.py 'REGEX'                      # -> results/full_laplace_folds.json
    python full_laplace_replicates.py GENERATOR 40 1000       # -> results/full_laplace_replicates_<GENERATOR>.json

The `results/` JSONs committed here are the raw per-fold and per-replicate
numbers behind every table above.
