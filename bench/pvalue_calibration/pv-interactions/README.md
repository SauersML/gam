# pv-interactions: smooth-term p-values for te(), ti(), by= and 2-D smooths

Target: `summary().smooth_terms[*].p_value`, i.e. `wood_smooth_test`
(`crates/gam-terms/src/inference/smooth_test.rs`) on its design-whitened
branch, as the production summary walk calls it (conditional covariance `Vb`,
full coefficient influence `F`, whitening Gram `X'WX`).

## The defect and the fix

The whitened branch truncated the eigen-decomposition at the integer rank
`round(edf)` and referred the quadratic to `χ²` (or `F`) at
`max(tr(F)²/tr(F²), rank)` degrees of freedom. The statistic and its reference
law had different dimensions, so the test was conservative, most of all for the
terms of this lane (`ti`/`te` with several penalties, `by=` blocks and 2-D
smooths), whose edf sits well inside the block dimension.

The test now uses Wood (2013)'s fractional rank directly: `r = edf1 =
Σ_block (2F − F·F)_ii`, `k = ⌊r⌋`, `ν = r − k`, statistic
`T = Σ_{i≤k} u_i² + ν·u_{k+1}²` on the standardized whitened coordinates, and
its exact null law `χ²_k + ν·χ²₁` (over an independent `χ²_ρ/ρ` when the scale
is estimated), evaluated by `gam_math::fractional_rank::fractional_rank_sf`
through the relative-accuracy signed weighted chi-square inversion. There is
no tuning constant: every quantity is the fit's own `F`, `Vb`, `X'WX` and
residual df.

Regression test: `fractional_rank_test_holds_its_size_under_the_null`
(4000 seeded draws from a known-null block with fractional edf 2.4, known and
estimated scale, |size − α| ≤ 3 MCSE at .10/.05/.01). The previous code fails
it (known-scale size 0.021 at .10 and 0.0075 at .05).

## Design

`python calibrate.py CELL 500 OUT.json`; replicate `r` of cell `C` draws from
`numpy.random.default_rng([crc32(C), r])`. `x ~ U(0, 1)`, Gaussian noise
σ = 0.8, binomial is Bernoulli on `logistic(η)`.

| cell | formula | truth (η) |
|---|---|---|
| (a) ti | `y ~ s(x1) + s(x2) + ti(x1, x2)` | `sin 2πx1 + 0.8 cos 2πx2` (×1.2 binomial); power adds `a·sin 2πx1·cos πx2` |
| (b) by=factor | `y ~ s(x, by=g)`, 3 levels | level intercepts 0, .5, −.5; only level `a` carries `A·sin 2πx` |
| (c) te | `y ~ te(x1, x2) + s(x3)` | `sin 2πx3`; power adds `a·sin πx1·cos πx2` |
| 2-D iso | `y ~ s(x1, x2) + s(x3)` | as (c) |
| (d) by=numeric | `y ~ s(x) + s(x, by=z)`, `z ~ N(0, 1)` | `sin 2πx`; power adds `a·z·cos πx` |

Size bound: `α + 2·√(α(1−α)/m)`; with m = 500 it is .069 at .05 and .019 at
.01 (MCSE .0134/.0097/.0045 at .10/.05/.01). KS is run on `2p | p < .5`
against U(0, 1): a shrinkage penalty sends the null term to edf ≈ 0 in a
fraction of fits (last column), which puts a point mass at p = 1 that an
unconditional KS would count against a valid test.

## Results (500 replications per cell)

### Null size

| cell | null term | m (failed) | median edf | size .10 | size .05 | size .01 | bound .05 / .01 | KS 2p\|p<.5 | edf<.01 |
|---|---|---|---|---|---|---|---|---|---|
| ti_gauss_200 | `ti(x1, x2)` | 500 (0) | 0.62 | 0.114 | 0.050 | 0.004 | 0.069 / 0.019 | 0.817 | 0.26 |
| ti_gauss_1000 | `ti(x1, x2)` | 500 (0) | 0.61 | 0.082 | 0.036 | 0.002 | 0.069 / 0.019 | 0.042 | 0.27 |
| ti_binom_200 | `ti(x1, x2)` | 499 (1) | 0.54 | 0.076 | 0.042 | 0.012 | 0.070 / 0.019 | 0.381 | 0.39 |
| ti_binom_1000 | `ti(x1, x2)` | 500 (0) | 0.52 | 0.080 | 0.048 | 0.012 | 0.069 / 0.019 | 0.169 | 0.38 |
| by_gauss_300 | `s(x, by=g):by=g[b]` | 500 (0) | 0.26 | 0.094 | 0.036 | 0.016 | 0.069 / 0.019 | 0.306 | 0.43 |
| by_gauss_300 | `s(x, by=g):by=g[c]` | 500 (0) | 0.21 | 0.098 | 0.038 | 0.002 | 0.069 / 0.019 | 0.528 | 0.44 |
| by_binom_600 | `s(x, by=g):by=g[b]` | 496 (4) | 0.14 | 0.079 | 0.042 | 0.010 | 0.070 / 0.019 | 0.268 | 0.48 |
| by_binom_600 | `s(x, by=g):by=g[c]` | 496 (4) | 0.00 | 0.079 | 0.048 | 0.014 | 0.070 / 0.019 | 0.030 | 0.51 |
| te_gauss_300 | `te(x1, x2)` | 500 (0) | 1.14 | 0.134 | 0.068 | 0.018 | 0.069 / 0.019 | 0.246 | 0.13 |
| te_binom_600 | `te(x1, x2)` | 500 (0) | 1.06 | 0.126 | 0.064 | 0.012 | 0.069 / 0.019 | 0.186 | 0.16 |
| iso_gauss_300 | `s(x1, x2)` | 500 (0) | 0.67 | 0.028 | 0.016 | 0.004 | 0.069 / 0.019 | 0.000 | 0.36 |
| vc_gauss_300 | `s(x, by=z)` | 500 (0) | 0.41 | 0.098 | 0.048 | 0.010 | 0.069 / 0.019 | 0.116 | 0.41 |
| vc_binom_600 | `s(x, by=z)` | 500 (0) | 0.25 | 0.066 | 0.028 | 0.004 | 0.069 / 0.019 | 0.509 | 0.47 |

Every null cell is within the bound at .05 and .01. The real co-terms
(`s(x1)`, `s(x2)`, `s(x3)`, `s(x)`, level `a`) reject in 94–100% of fits at
.05 in every cell. The isotropic 2-D smooth is conservative (valid, not
exact); `ti_gauss_1000` and `by_binom_600[c]` KS rejections at the 5% level
come from p-values on the conservative side.

### Power (the interaction term is real)

| cell | term | amplitude | m (failed) | median edf | power .10 | power .05 | power .01 |
|---|---|---|---|---|---|---|---|
| ti_gauss_200_power | `ti(x1, x2)` | 0.6 | 500 (0) | 3.65 | 0.990 | 0.974 | 0.894 |
| ti_binom_1000_power | `ti(x1, x2)` | 0.8 | TI_BINOM_POWER |
| te_gauss_300_power | `te(x1, x2)` | 0.5 | 500 (0) | 3.71 | 0.988 | 0.986 | 0.948 |
| iso_gauss_300_power | `s(x1, x2)` | 0.5 | 500 (0) | 3.96 | 0.980 | 0.954 | 0.846 |
| vc_gauss_300_power | `s(x, by=z)` | 0.4 | 500 (0) | 1.91 | 1.000 | 0.998 | 0.988 |
| by_gauss_300 | `s(x, by=g):by=g[a]` | 1.5 | 500 (0) | 5.38 | 1.000 | 1.000 | 1.000 |
| by_binom_600 | `s(x, by=g):by=g[a]` | 2.0 | 496 (4) | 4.53 | 1.000 | 1.000 | 1.000 |

## Open: te() runs slightly liberal

Each te cell is inside its 2-MCSE bound, but pooled over both te cells
(m = 1000) the size is 0.130 / 0.066 / 0.015 at .10 / .05 / .01, i.e. +3.2,
+2.3 and +1.6 MCSE. The excess is not in one edf band:

| te edf band (pooled) | fits | reject .10 | reject .05 |
|---|---|---|---|
| < 0.01 | 146 | 0.000 | 0.000 |
| 0.01–1 | 333 | 0.129 | 0.075 |
| 1–2 | 227 | 0.115 | 0.079 |
| 2–3 | 160 | 0.188 | 0.050 |
| 3–5 | 107 | 0.271 | 0.131 |
| ≥ 5 | 27 | 0.074 | 0.037 |

With the smoothing parameters held fixed the whitened test is conservative
(the whitened `Vb`-standardized coordinates of `β̂` have variance ≤ 1), so the
excess is the effect of selecting te's three smoothing parameters (two
marginal penalties plus the null-space penalty) from the same data: REML can
rotate the anisotropy so the top whitened directions line up with the noise.
`Vb` is conditional on `λ̂` and does not carry that. This is the limitation
Wood (2013) and `mgcv::summary.gam` document for poorly identified smoothing
parameters; the remedy there is the smoothing-parameter-corrected covariance,
which this project deliberately does not use for p-values (#2142, #2296: it
reorders the eigen-directions the test truncates). No calibration factor is
applied. `ti()` (same penalty structure, additive truth) does not show it.

## Failed fits

Five of the 6500 fits above failed before any p-value was formed, all in the
outer smoothing-parameter optimizer (out of this lane's scope):
`ti_binom_200` rep 26 (`RemlConvergenceError`: certified optimum beaten by an
uncertified state), `by_binom_600` reps 220, 228, 274, 323 (`FitInputError`:
the smoothing-corrected covariance refused because the ρ-Hessian has negative
curvature just past its certificate bar, e.g. −6.5e-8 vs 6.4e-8).

`results.jsonl` holds, per cell, the summary above and every replicate's
`edf`, `ref_df`, `chi_sq` and `p_value` per term.
