# Summary smooth-term p-values: concurvity, large n, weights, offsets, trials, heteroscedasticity

Lane: `bench/pygam_audit/lanes/pv-concurvity-scale.md`.

The p-value studied is `summary().smooth_terms[s(x2)].p_value`: the
rank-truncated Wald test of Wood (2013) that the summary table publishes. In
every cell `s(x2)` is exactly null under H0; power is measured on the same
design with a real `x2` effect. `calibrate.py` holds the seeded data-generating
processes (seed `7_300_000 + 1000*cell_index + rep`, stream `[seed, H1]`).

```
python calibrate.py --list
python calibrate.py conc_rho90 weights_iv --workers 4   # appends to results.jsonl
```

`results_before_fix.jsonl` holds the baseline on main at `a5a71e5`, and
`results.jsonl` holds the run after the fixes below. MCSE is `sqrt(a(1-a)/R)`.
With R = 500 it is 0.0134, 0.0097 and 0.0045 at .10, .05 and .01; with R = 200 it
is 0.0212, 0.0154 and 0.0070.

## Results

Rejection rate under H0 at .10 / .05 / .01. The next column gives the excess at
.05 in MCSE units. After that come the Kolmogorov-Smirnov distance from U(0,1),
the share of null p-values above .99, and power at .05 (100 H1 reps).

| cell | n | R | .10 | .05 | .01 | .05 excess (MCSE) | KS D | p > .99 | power | fit s |
|---|---|---|---|---|---|---|---|---|---|---|
| conc_rho00 (reference) | 200 | 500 | .064 | .040 | .004 | −1.0 | .50 | .50 | .73 | 0.14 |
| conc_rho50 | 200 | 500 | .060 | .028 | .008 | −2.3 | .48 | .48 | .81 | 0.14 |
| conc_rho90 | 200 | 500 | .066 | .038 | .008 | −1.2 | .47 | .47 | .27 | 0.13 |
| conc_nonlinear, x2 = (2x1−1)² + N(0, .15²) | 200 | 500 | .046 | .018 | .004 | −3.3 | .51 | .51 | .13 | 0.14 |
| large_n_1e4 | 10⁴ | 500 | .046 | .020 | .006 | −3.1 | .46 | .47 | .48 | 0.66 |
| large_n_1e5 † | 10⁵ | 200 | .050 | .025 | .000 | −1.6 | .48 | .47 | .43 | 14.5 |
| weights_iv, inverse-variance w, σ² = 1 | 200 | 500 | .068 | .030 | .006 | −2.1 | .52 | .53 | .91 | 0.15 |
| poisson_offset, log-exposure offset | 200 | 500 | .054 | .026 | .008 | −2.5 | .62 | .62 | .81 | 0.94 |
| binomial_trials, trials 1..20 | 200 | 500 | .066 | .044 | .004 | −0.6 | .55 | .56 | .99 | 0.87 |
| hetero_constant, constant-variance model | 400 | 500 | **.134** | **.074** | **.024** | **+2.5** | .44 | .44 | .76 | 0.17 |
| hetero_locscale, `noise_formula="s(x2)"` | 400 | 497 ‡ | .074 | .036 | .014 | −1.4 | .53 | .38 | 1.00 | 1.16 |

† Not rerun after the fix. The 1e5 row is the baseline run; see the gauge fix
below for why it is unchanged.

‡ 4 of 600 fits (3 H0, 1 H1) raised `InnerModeConvergenceError`. The failures
are recorded and not dropped silently. On main this cell had no p-value at all
(see the second fix).

The weights_iv, poisson_offset, binomial_trials, hetero_constant and concurvity
cells were rerun after the fix. They reproduce the baseline sizes and power
exactly. One baseline `RemlConvergenceError` in binomial_trials did not recur.

KS rejects in every cell (p < 1e-40). That comes entirely from the point mass at
p ≈ 1 described below, not from the shape of the continuous part.

## What the numbers say

**Concurvity (ρ = .5, .9, nonlinear).** Concurvity does not make the test
anti-conservative. Size stays at or under nominal, and power falls as `s(x2)`
becomes harder to separate from `s(x1)` (.73 → .27 at ρ = .9, and .13 for the
nonlinear map). The lane asked whether the Vc covariance (Vb plus the
smoothing-parameter correction) should replace Vb under concurvity. That
premise is refuted. Vc was computed from the same fits and is *more*
conservative, not less:

| cell | Vb at .05 | Vc at .05 |
|---|---|---|
| conc_rho50 | .028 | .018 |
| conc_rho90 | .038 | .022 |
| conc_nonlinear | .018 | .008 |

Vb is therefore kept.

**Large n.** Size stays conservative at 1e4 and 1e5, with no drift toward
anti-conservativeness. Fit time is 0.66 s at 1e4 and 14.5 s at 1e5.

**Prior weights, offsets, trials.**
- Inverse-variance Gaussian weights with known σ² = 1 are calibrated like the
  unweighted reference.
- The Poisson log-exposure offset and binomial trials 1..20 as prior weights
  are calibrated too.
- The exact weights-equal-duplicated-rows property is covered by the regression
  tests, not by Monte Carlo.

**Heteroscedastic Gaussian: the expected failure.**
- A constant-variance model fitted to data whose sd grows 7-fold with x2 is
  anti-conservative: .074 at .05 and .024 at .01, i.e. +2.5 and +3.1 MCSE. Its
  Vb uses one pooled σ̂², so it understates the variance of the x2-direction
  contrasts where the noise is largest.
- This is not a bug in the test. It is the documented consequence of a
  misspecified variance model.
- The location-scale model that describes this response
  (`noise_formula="s(x2)"`) is calibrated: .074 / .036 / .014, all within
  2 MCSE, with power 1.00.

**The point mass at p ≈ 1 (every cell).**
- About half the null p-values exceed .99. The null `s(x2)` is shrunk to its
  boundary (edf ≈ 0), and the Wald statistic is then ≈ 0.
- Its null law is a mixture of a point mass at 0 and a continuous part, which
  is the boundary-null structure of a variance component. So the p-value is
  valid (conservative) but not uniform.
- The continuous part rejects at roughly twice the nominal rate: .040 / .50 at
  .05 in the reference cell. That partly compensates for the point mass.
- The cells that miss the two-sided 2-MCSE band (rho50, nonlinear, 1e4,
  weights_iv, poisson_offset) miss it on the *conservative* side, by 2.1–3.3
  MCSE. The reference cell shows the same pattern at .10 (−2.7 MCSE).
- The cause is this reference distribution at the boundary, not concurvity,
  scale, weights or offsets. That distribution belongs to the summary Wald test
  itself, which this lane does not own. It is recorded here as the open item,
  not patched with a factor.

## Root-cause fixes on this branch

1. **The Wald statistic depended on the smooth's centering gauge.**
   - Fitting with integer prior weights `w` and fitting on the duplicated rows
     are the same likelihood for a fixed-dispersion family. Both give the same
     fitted values, edf and λ̂.
   - However, the spline's sum-to-zero constraint was applied over the
     unweighted rows. The two fits therefore carried `s(x)` in different
     centering gauges, i.e. differing by a constant absorbed in the intercept.
   - The summary whitens each block by its Gram block `X_jᵀWX_j`, and that
     Gram gained a constant component that depended on the gauge. So the
     published χ² differed: 57.84 vs 57.95 for Poisson, 61.257 vs 61.263 for
     binomial.
   - `smooth_term_summary_rows` now whitens by the Gram with the intercept
     projected out, `G − G[:,0]G[0,:]/G₀₀`. This is the only part of the block
     that the smooth, rather than the intercept, identifies.
   - It is exactly a no-op when `1ᵀWX_j = 0`, which is the unweighted Gaussian
     training Gram. The concurvity and large-n cells, including the 1e5 row
     not rerun, are therefore unchanged.
2. **A location-scale fit got no smooth-term table.**
   - The rebuilt-layout check compared the Location block's penalties against
     the λ count of *all* blocks, including the scale predictor's. It therefore
     refused with "rebuilt design has 4 penalty blocks but the saved fit has 6
     smoothing parameters".
   - It now checks the Location block's own λ, and still refuses a layout in
     which that block is not the leading one.
   - Behind that, the summary failed outright on the information criteria,
     because a fit with several linear predictors has no scalar family or
     dispersion. The AIC fields are now `null`, with the reason in
     `aic_corrected_unavailable`.

Regression tests:
- `tests/test_summary_pvalue_weights_and_location_scale.py` (each test fails
  on main and passes here):
  - Poisson integer weights vs duplicated rows;
  - binomial trials vs expanded Bernoulli rows (edf, ref_df, χ², p to rel 1e-5);
  - the location-scale table.
- The Rust unit tests in `smooth_term_summary.rs`:
  - the statistic is invariant to the centering;
  - a design without a single intercept column keeps its Gram.
- The Rust unit test in `inference/model.rs`: the location-scale λ check.

## Gaussian weights are prior weights

For the Gaussian family a weight is an inverse-variance prior weight, and σ̂²
uses the row count (#1617 / #1618,
`tests/weighted_gaussian_is_prior_weight_not_frequency_weight_test.py`). So a
weighted Gaussian fit deliberately does *not* reproduce the duplicated-rows
standard errors. The exact weights-equal-duplicates identity is therefore
asserted only for the fixed-dispersion families (Poisson, binomial).

No sufficient-statistics path exists on main for the large-n cells to be
compared against. The 1e4 and 1e5 cells fit the full design.
