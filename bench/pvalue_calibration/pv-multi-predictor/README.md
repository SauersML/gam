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

`V_b` is the smoothing-parameter-conditional covariance. The kernel
deliberately does not use the smoothing-corrected `V_c` (#2296), because Wood's
reference distribution is derived for `V_b`. No surface here substitutes
another covariance.

## Surfaces

| surface | tested predictor | null | before | after |
|---|---|---|---|---|
| multinomial, per class | class `a` block, `a·P + span` | term does not move log-odds of `a` vs the reference class | raw-covariance truncation, not whitened; oversized (table below) | whitened by the exact softmax curvature `XᵀW(β̂)X`, like the scalar summary |
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

## Results

RESULTS_TABLE

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
