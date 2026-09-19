# Smooth-term p-values of multi-predictor models

This study audits the smooth-term significance tests of every model whose
coefficients come in more than one predictor block. For each surface it records
what null is tested, which covariance the test uses, and whether the reference
distribution sees the other predictors. It then calibrates each surface's size
under a null by seeded Monte Carlo.

    python calibrate.py SCENARIO N REPS SEED results/OUT.json
    python analyze.py results/*.json

`results/` holds the runs reported below: 500 replications per scenario, at
n = 300 and n = 2000, one seed per run (listed in each JSON). A test is valid
when P(p ≤ a) ≤ a. The verdict is PASS when the rejection rate at every level
a in {0.10, 0.05, 0.01} is at most a + 2·MCSE, with MCSE = sqrt(a(1−a)/R).
With 500 replications the MCSE is 0.0134, 0.0097 and 0.0044. The KS p-value
against Uniform(0,1) is reported but is not the criterion. A penalized smooth
whose EDF shrinks to its null space returns p = 1 exactly, so a correctly sized
null distribution has an atom at 1 and a KS test rejects it whatever the
size.

The per-seed regression checks live in
`tests/pv_multi_predictor_smooth_significance_test.py`,
`tests/inference/misc/sbc_multinomial_{,joint_}smooth_significance_size_curve.rs`
and the `tests/quality/calibration/registry.rs` rows.

## The shared kernel

Every surface below feeds the same Wood (2013) rank-truncated Wald kernel,
`gam_terms::inference::smooth_test::wood_smooth_test`. It uses:

- the conditional covariance `V_b = H⁻¹` of the full stacked coefficient
  vector. That is the joint posterior covariance across every block, so
  cross-predictor correlation enters the test.
- the joint influence matrix `F`. The term EDF is `tr(F_JJ)`, and the
  reference df is `max(tr(F_JJ)²/tr(F_JJ²), r)`, where `r` is the truncation
  rank: `round(EDF)`, floored at the unpenalized dimension. Both are read over
  the tested coefficient set `J` of the joint `F`, so the other blocks'
  shrinkage enters through `F`'s joint solve.
- the design-whitening Gram `XᵀWX` in the same stacked layout.

### Fits that publish no influence matrix

The standard single-predictor lane publishes `F` and `XᵀWX`. The
custom-family and survival lanes do not: Gaussian and survival
location-scale, Royston-Parmar and Weibull survival. Their smooth table used to
fall back to the truncation rank `r` for the reference df and to the caller's
unweighted `XᵀX` for the whitening. In a location-scale fit the mean block's
curvature is `Xᵀ diag(1/σᵢ²) X`, so the unweighted metric kept the wrong
rank-`r` subspace. The mean-smooth test of a covariate that moves only the
scale then rejected above its level at n = 300 (0.078 at 0.05 before the fix).

Both blocks belong to the term alone, because no other term's penalty touches
its coefficients `J`:

- `F_JJ = I − (V_JJ/c)·S_JJ`, with `S_JJ = Σ_k λ_k S_k`, and
- `G_JJ = H_JJ − S_JJ`.

The summary now derives them when the fit leaves them out. A fit may report
`β` in other units than the design's penalties. A Gaussian location-scale fit
solves on a standardized response and reports the mean block rescaled. That
factor `d²` is identified from the fit's own unit-free per-penalty traces
`τ_k`: `tr(V_JJ λ_k S_k)/c = d²·τ_k` must give one `d²` for every penalty. The
resulting `tr(F_JJ)` must also equal the reported EDF. If either check fails,
the test keeps the old rank-only reference df rather than guessing. The
regression `tests/regressions/smooths/summary_smooth_test_without_published_influence.rs`
checks two things:

- a standard fit stripped of `F` and `XᵀWX` reports the same test as with
  them;
- a location-scale fit's reference df is Wood's, and does not depend on the
  response's units.

`V_b` is the smoothing-parameter-conditional covariance. The kernel
deliberately does not use the smoothing-corrected `V_c` (#2296), because Wood's
reference distribution is derived for `V_b`. No surface here substitutes
another covariance.

## Surfaces

| surface | tested predictor | null | before | after |
|---|---|---|---|---|
| multinomial, per class | class `a` block, `a·P + span` | term does not move log-odds of `a` vs the reference class | raw-covariance truncation, not whitened; oversized (table below) | two classes: whitened by the exact softmax curvature `XᵀW(β̂)X`, like the scalar summary. Three or more: refused with `penalty_couples_outside_tested_set` (see below) |
| multinomial, joint (new) | union of the term's span in every class block | term moves no class probability | did not exist | `MultinomialModel.joint_smooth_significance()`; reference-class invariant |
| Gaussian location-scale | Location block | term does not move the mean | table refused: "Refit the model" (λ count compared against all blocks) | tested at the Location block's offsets |
| survival location-scale | Threshold block, after the Time block | term does not move the threshold predictor | the TIME block's coefficients were tested under the covariate names (a null covariate read p ≈ 1e-17; the real one had no p-value) | tested at the Threshold block's offsets |
| survival transformation (Royston-Parmar) | covariate columns of the one Mean block `[time basis \| covariates]` | term does not move the log cumulative hazard | table refused: stale-save message | covariates located after the time columns and time penalties |
| survival Weibull | as transformation, time basis is one `log t` column | as transformation | every smooth read one column early (the `log t` coefficient folded into the first smooth) | as transformation |
| latent survival | Mean block, after the Time block | as location-scale survival | same offset defect as location-scale survival | fixed by the same offset; **not calibrated** (see below) |

The scale (and link-wiggle) blocks of the location-scale models get no rows.
The saved spec that the per-smooth table replays describes the mean predictor
only, so there is nothing to name a scale-block row by. Those blocks are
neither tested nor misreported.

A fit whose layout cannot be matched refuses with a typed reason. Examples are
a covariate penalty with a non-zero prior mean that the survival fit leaves out,
a block narrower than its design, or a competing-risks fit (whose layout is
per cause). The table then reports that reason in `smooth_terms_unavailable`
instead of a p-value for the wrong coefficients.

### Multinomial: which test to use

The per-class test asks whether the term moves the log-odds of one class
against the reference class. With three or more classes it has no valid
p-value on these fits, and every per-class row now reports
`penalty_couples_outside_tested_set` instead of one.

The reason is the penalty. Since #1587 the fit penalizes each class's
deviation from the all-class mean, `Σ_c λ_c ‖f_c − f̄‖²`, so that the fit does
not depend on which class is the reference. That penalty couples the class
blocks. The shrinkage target of class `a`'s coefficients is a share of the
other classes' fit, not `β_a = 0`. Under the per-class null, with another class
carrying the effect, the class-`a` estimate is biased toward that effect.
Wood's test assumes the penalty shrinks the tested block toward its null, and
no reference df corrects a biased estimate. The runs below show both failures:

- under the global null (`multinomial_null`), the a-vs-c row rejected 0.075
  (n = 300) and 0.087 (n = 2000) at 0.05;
- under `multinomial_power`, x moves class b only, so the a-vs-c null still
  holds. The a-vs-c row rejected 0.088 at 0.01 (n = 2000), almost nine times
  its level, and more as n grows.

A class block of the joint influence matrix is not an influence matrix of its
own either: its trace goes to zero or below when the fit ties class `a`'s term
to the others, so up to 10% of the null runs' rows had no EDF at all.

The refusal is structural. The test reads the fitted penalty and refuses any
coefficient set `J` whose rows reach outside `J`. A two-class fit has one class
block, nothing to couple, and keeps its per-class test
(`sbc_multinomial_smooth_significance_size_curve`).

The joint test, `joint_smooth_significance()`, tests every class block of the
term at once, and the penalty does not reach outside that set. Its null, that
the term moves no class probability, is the penalty's shrinkage target and
does not depend on the reference class. Its EDF is the trace of the term's
`F_JJ` over all classes. It is the test to use for "does this covariate
matter", and it holds its size at both sample sizes. Rejecting when any
per-class test rejects is not a valid substitute: it multiplies the size, as
the legacy table below shows.

## Results

| scenario | n | test | usable/R | p<=.10 | p<=.05 | p<=.01 | MCSE (.10/.05/.01) | KS p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| multinomial_null | 300 | joint (all classes) | 500/500 | 0.108 | 0.052 | 0.010 | 0.013/0.010/0.004 | 6.92e-101 | PASS |
| multinomial_null | 300 | per-class a vs c | 468/500 | 0.143 | 0.075 | 0.013 | 0.014/0.010/0.005 | 2.87e-77 | FAIL, now refused |
| multinomial_null | 300 | per-class b vs c | 448/500 | 0.127 | 0.054 | 0.007 | 0.014/0.010/0.005 | 8.87e-85 | PASS, now refused |
| multinomial_null | 2000 | joint (all classes) | 499/500 | 0.090 | 0.044 | 0.002 | 0.013/0.010/0.004 | 3.19e-111 | PASS |
| multinomial_null | 2000 | per-class a vs c | 494/500 | 0.158 | 0.087 | 0.012 | 0.013/0.010/0.004 | 1.94e-84 | FAIL, now refused |
| multinomial_null | 2000 | per-class b vs c | 455/500 | 0.121 | 0.057 | 0.007 | 0.014/0.010/0.005 | 1.61e-99 | PASS, now refused |
| multinomial_power | 300 | joint (all classes) | 500/500 | 0.290 | 0.176 | 0.066 | 0.013/0.010/0.004 | 6.11e-31 | power |
| multinomial_power | 300 | per-class a vs c | 407/500 | 0.174 | 0.088 | 0.037 | 0.015/0.011/0.005 | 1.34e-32 | now refused |
| multinomial_power | 300 | per-class b vs c | 477/500 | 0.277 | 0.155 | 0.061 | 0.014/0.010/0.005 | 1.24e-24 | now refused |
| multinomial_power | 2000 | joint (all classes) | 500/500 | 0.992 | 0.980 | 0.876 | 0.013/0.010/0.004 | 0 | power |
| multinomial_power | 2000 | per-class a vs c | 432/500 | 0.150 | 0.125 | 0.088 | 0.014/0.010/0.005 | 5.53e-15 | now refused |
| multinomial_power | 2000 | per-class b vs c | 500/500 | 0.980 | 0.970 | 0.858 | 0.013/0.010/0.004 | 0 | now refused |
| ls_scale_only | 300 | mean s(x) | 499/500 | 0.086 | 0.044 | 0.004 | 0.013/0.010/0.004 | 1.84e-146 | PASS |
| ls_scale_only | 2000 | mean s(x) | 498/500 | 0.058 | 0.026 | 0.006 | 0.013/0.010/0.004 | 1.44e-138 | PASS |
| ls_power | 300 | mean s(x) | 500/500 | 0.622 | 0.502 | 0.268 | 0.013/0.010/0.004 | 9.48e-144 | power |
| ls_power | 2000 | mean s(x) | 500/500 | 1.000 | 1.000 | 1.000 | 0.013/0.010/0.004 | 0 | power |
| weibull_null | 300 | s(noise) | 473/500 | 0.106 | 0.044 | 0.015 | 0.014/0.010/0.005 | 1.17e-88 | PASS |
| weibull_null | 2000 | s(noise) | 491/500 | 0.102 | 0.049 | 0.010 | 0.014/0.010/0.004 | 1.14e-53 | PASS |
| weibull_power | 300 | s(age) | 480/500 | 1.000 | 1.000 | 1.000 | 0.014/0.010/0.005 | 0 | power |
| weibull_power | 2000 | s(age) | 491/500 | 1.000 | 1.000 | 1.000 | 0.014/0.010/0.004 | 0 | power |
| transformation_null | 300 | s(noise) | 476/500 | 0.101 | 0.061 | 0.013 | 0.014/0.010/0.005 | 1.34e-109 | PASS |
| transformation_null | 2000 | s(noise) | 491/500 | 0.096 | 0.063 | 0.016 | 0.014/0.010/0.004 | 1.4e-54 | PASS |
| transformation_power | 300 | s(age) | 465/500 | 1.000 | 1.000 | 1.000 | 0.014/0.010/0.005 | 0 | power |
| transformation_power | 2000 | s(age) | 488/500 | 1.000 | 1.000 | 1.000 | 0.014/0.010/0.005 | 0 | power |
| survls_null | 300 | s(noise) | 500/500 | 0.064 | 0.034 | 0.004 | 0.013/0.010/0.004 | 1.79e-134 | PASS |
| survls_null | 2000 | s(noise) | 500/500 | 0.068 | 0.034 | 0.006 | 0.013/0.010/0.004 | 2.79e-110 | PASS |
| survls_power | 300 | s(age) | 500/500 | 1.000 | 1.000 | 1.000 | 0.013/0.010/0.004 | 0 | power |
| survls_power | 2000 | s(age) | 500/500 | 1.000 | 1.000 | 1.000 | 0.013/0.010/0.004 | 0 | power |

The multinomial per-class rows are the p-values the per-class test gave before
it was made to refuse. They are the evidence for the refusal described above;
the current build reports no per-class p-value for these three-class fits.
The joint rows are unchanged by the refusal, which only reads the penalty.

Failed fits are recorded, not dropped, and they are fitting failures rather
than p-value failures: the fit raised before any test ran.

| scenario | n = 300 | n = 2000 |
|---|---|---|
| multinomial_null | 0 | 1 |
| ls_scale_only | 1 | 2 |
| weibull_null / power | 27 / 20 | 9 / 9 |
| transformation_null / power | 24 / 35 | 9 / 12 |

Every other run fitted all 500 replications. The Weibull and transformation
failures are REML or inner P-IRLS convergence refusals on the survival path.

### Legacy multinomial per-class test (before)

This is the same null data-generating process on the pre-fix build, with 500
replications each. The class probabilities depend on z only, so x is null for
every class.

| n | class | p≤.10 | p≤.05 | p≤.01 |
|---|---|---|---|---|
| 300 | a vs c | 0.144 | 0.078 | 0.022 |
| 300 | b vs c | 0.126 | 0.080 | 0.024 |
| 2000 | a vs c | 0.158 | 0.098 | 0.034 |
| 2000 | b vs c | 0.116 | 0.066 | 0.026 |

Every cell exceeds a + 2·MCSE, and rejecting when any class rejects gave 0.112
(n = 300) and 0.116 (n = 2000) at 0.05.

## Not calibrated

A latent survival fit takes several minutes at n = 300, so 500 replications
per sample size are not feasible here. Its smooth table uses the same
primary-block offset as location-scale survival, whose calibration is above.
