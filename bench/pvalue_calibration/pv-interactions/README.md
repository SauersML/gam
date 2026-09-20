# pv-interactions: smooth-term p-values for te(), ti(), by= and 2-D smooths

Target: `summary().smooth_terms[*].p_value` for tensor products (`te`, `ti`),
factor `by=` smooths (one row per level), numeric `by=` (varying
coefficient) and isotropic 2-D smooths. The p-value is computed in Rust by
`gam_terms::inference::smooth_score_test`; the summary walk
(`gam_solve::estimate::smooth_term_summary`) only hands it the fit.

## Result

All 13 null terms pass the two-sided gate below: size at .10/.05/.01, the
upper-tail mass `P(p > 1 − a)` at the same levels, and KS against U(0, 1),
over 500 replicates each. The worst of the 91 checks has p = 0.054. The
p-values are neither liberal nor conservative anywhere on the unit interval.
Power against the interaction alternatives is 0.97–1.00 at .05.

## The test

A penalized smooth `f_j = X_j β_j` is the random effect
`β_j ~ N(0, τ·S_j⁻)`, and "no effect" is the boundary null `τ = 0`. The
variance-component score test (Lin 1997; Zhang & Lin 2003) reads everything
off the one full fit:

```text
b = Hβ̂ (= XᵀWz at convergence),   A = H_oo⁻¹ G_oj,
s = b_j − Aᵀ b_o,                  C = G_jj − G_jo A − Aᵀ G_oj + Aᵀ G_oo A,
Q = sᵀ K s / φ̂,                    Q ~ Σ w_i χ²₁,   w = eig(K^½ C K^½),
```

with `G = XᵀWX` the likelihood curvature, `H = G + S(λ)` and `o` every other
coefficient. When `φ` is estimated the tail is the signed weighted
chi-square `P(Σ w_i χ²₁ − (q/ρ) χ²_ρ > 0)`. The reported `ref_df` is
`(Σw)²/Σw²`, and the reported statistic is `Q` rescaled to that null mean.

Why a score test and not a Wald test on `β̂_j`: the Wald statistic depends on
the term's own fitted `λ_j`. Under the null `λ_j` runs to its boundary in a
large share of fits (edf < 0.01 in 13–51% of the null fits of the earlier run). That
puts an atom at p ≈ 1 in the Wald null law, and in the other fits the fitted
`λ_j` chooses the directions the statistic is spread over. The previous
fractional-rank Wald test was therefore conservative for
`s(x1, x2)` (size 0.016 at .05, KS p < 0.001). It ran liberal for `te()`
when the two cells were pooled (+2.3 MCSE at .05). The score test never looks
at `λ_j`, so its reference law is exact at the fitted `λ_o` across the whole
unit interval.

### The kernel K

`K = Σ_l K_l / tr(K_l C)`: one independent variance component per
structural penalty `S_l` of the term, each put on its own null scale. This
makes the test invariant to how each penalty was scaled when it was built.
It also gives the penalty null space the same standing as the wiggly part.

`K_l` is read off the penalties. Let `N_l` be an orthonormal basis of
`null(Σ_{k≠l} S_k)`, the directions that `S_l` alone charges. When these
subspaces together form a basis of the block, the prior splits exactly into
independent components at every λ:

```text
K_l = N_l (N_lᵀ S_l N_l)⁻¹ N_lᵀ.
```

A smooth with a null-space penalty is always this case. The formula
transforms with any change of coefficient chart `β = T·γ`, so the test does
not depend on the chart.

The Euclidean pseudo-inverse `S_l⁺` gives the same `K_l` only when the
penalty ranges are orthogonal. The B-spline null-space ridge charges the
**mean slope** through the end coefficients, so there `S_l⁺` points along an
end-concentrated curve instead of the linear null function. A level-specific
or varying-coefficient slope was then nearly invisible: χ² = 0.14 at a
noncentrality near 107, and a level curve of amplitude 2 under a binomial
response was found in 7% of fits.

When the subspaces do not form a basis (overlapping tensor marginals in
`te`/`ti`), no exact split exists. `K_l` is then penalty `l`'s share of the
prior covariance at a data-scaled base precision `P = Σ_l a_l S_l`, with
`a_l = 1/tr(G_jj⁺ S_l)`:

```text
K_l = P⁻¹ (a_l S_l) P⁻¹ = −∂(Σ_k λ_k S_k)⁻¹/∂ log λ_l  at λ = a.
```

This moves with the coefficient chart and does not depend on any penalty's
scale. On complementary ranges it is the exact formula above divided by
`a_l`, which the null-mean normalization cancels, so the one rule covers both
cases. `S_l⁺` moved with the chart. Any fixed `K` gives an exact null law, so
this choice affects power only, never size.

### When no p-value is reported

The row's `p_value` is `None`, and a typed `p_value_unavailable` label gives
the reason:

- `unpenalized_direction`: the penalties do not cover the block, so the
  block has a fixed effect that `τ = 0` does not remove.
- `not_identified`: the other terms' `H_oo` is not positive definite, or the
  term's score has no variance left once they are fitted.
- `indefinite_curvature`: the published curvature `G` leaves `C` with a
  negative eigenvalue. The score then has no variance law, and testing the
  positive part of the spectrum would test a different statistic. This
  happens on the custom-family (location-scale) lane: there `G = H − S(λ)`
  is the observed information, which need not be positive semi-definite.
- `residual_df_unavailable`: the estimated-scale reference needs `ρ`.

### Fit-side fix: the by-level null-space ridge

A factor-level `by=` B-spline smooth rebuilt its double-penalty ridge
differently in two places:

- At fit time, under the collection coefficient gauge, it was built as the
  Euclidean complement of the centred chart.
- In the frozen replay (the saved model), it is charged along the mean slope.

The saved model's summary therefore formed its test from penalties the fit
never used. The gauge path now charges the ridge along the mean slope, exactly
as the replay does. A regression test pins the fit and the frozen rebuild to
the same ridge.

## Design

Run `python calibrate.py CELL 500 OUT.json`. Replicate `r` of cell `C` draws
from `numpy.random.default_rng([crc32(C), r])`, with `x ~ U(0, 1)` and
Gaussian noise σ = 0.8. A binomial response is Bernoulli on `logistic(η)`.
Then run `python tabulate.py OUT*.json` to produce the tables and the gate
below.

| cell | formula | truth (η) |
|---|---|---|
| ti | `y ~ s(x1) + s(x2) + ti(x1, x2)` | `sin 2πx1 + 0.8 cos 2πx2` (×1.2 binomial); power adds `a·sin 2πx1·cos πx2` |
| by=factor | `y ~ s(x, by=g)`, 3 levels | level intercepts 0, .5, −.5; only level `a` carries `A·sin 2πx` |
| te | `y ~ te(x1, x2) + s(x3)` | `sin 2πx3`; power adds `a·sin πx1·cos πx2` |
| 2-D iso | `y ~ s(x1, x2) + s(x3)` | as te |
| by=numeric | `y ~ s(x) + s(x, by=z)`, `z ~ N(0, 1)` | `sin 2πx`; power adds `a·z·cos πx` |

**Gate.** Each null term gets seven checks:

- KS against U(0, 1), unconditional; the score test has no atom at p = 1 to
  condition away.
- At each a ∈ {.10, .05, .01}, an exact two-sided binomial test of
  `#{p ≤ a}` against `a`.
- At the same levels, an exact two-sided binomial test of `#{p > 1 − a}`
  against `a`.

A conservative test fails the lower-tail check exactly as a liberal one does,
and a p-value that piles up or thins out near 1 fails the upper-tail check.
The 91 checks (13 terms × 7) share a family-wise level of 0.01, so each check
runs at 0.01/91 = 1.1e-4.

At m = 500 the MCSE of a size is .0134, .0097 and .0045 at .10, .05 and .01.

## Results (500 replications per cell)

### Null: two-sided calibration

| cell | null term | m (failed) | median edf | size .10 | size .05 | size .01 | P(p>.90) | P(p>.95) | P(p>.99) | KS p | min check p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ti_gauss_200 | `ti(x1, x2)` | 500 (0) | 0.62 | 0.088 | 0.050 | 0.012 | 0.096 | 0.044 | 0.010 | 0.470 | 0.412 |
| ti_gauss_1000 | `ti(x1, x2)` | 500 (0) | 0.61 | 0.092 | 0.042 | 0.004 | 0.116 | 0.050 | 0.010 | 0.184 | 0.184 |
| ti_binom_200 | `ti(x1, x2)` | 500 (0) | 0.60 | 0.092 | 0.032 | 0.008 | 0.084 | 0.042 | 0.006 | 0.761 | 0.065 |
| ti_binom_1000 | `ti(x1, x2)` | 500 (0) | 0.57 | 0.110 | 0.058 | 0.012 | 0.092 | 0.062 | 0.010 | 0.942 | 0.217 |
| by_gauss_300 | `s(x, by=g):by=g[b]` | 499 (1) | 0.15 | 0.110 | 0.052 | 0.016 | 0.092 | 0.040 | 0.008 | 0.054 | 0.054 |
| by_gauss_300 | `s(x, by=g):by=g[c]` | 499 (1) | 0.15 | 0.094 | 0.034 | 0.010 | 0.094 | 0.048 | 0.008 | 0.259 | 0.122 |
| by_binom_600 | `s(x, by=g):by=g[b]` | 498 (2) | 0.22 | 0.100 | 0.036 | 0.008 | 0.092 | 0.044 | 0.010 | 0.096 | 0.096 |
| by_binom_600 | `s(x, by=g):by=g[c]` | 498 (2) | 0.05 | 0.090 | 0.056 | 0.008 | 0.114 | 0.056 | 0.008 | 0.656 | 0.295 |
| te_gauss_300 | `te(x1, x2)` | 500 (0) | 1.14 | 0.108 | 0.064 | 0.014 | 0.090 | 0.046 | 0.008 | 0.900 | 0.150 |
| te_binom_600 | `te(x1, x2)` | 500 (0) | 1.10 | 0.088 | 0.048 | 0.006 | 0.102 | 0.048 | 0.010 | 0.931 | 0.412 |
| iso_gauss_300 | `s(x1, x2)` | 500 (0) | 0.67 | 0.098 | 0.050 | 0.008 | 0.086 | 0.054 | 0.004 | 0.087 | 0.087 |
| vc_gauss_300 | `s(x, by=z)` | 500 (0) | 0.41 | 0.118 | 0.068 | 0.006 | 0.098 | 0.048 | 0.012 | 0.501 | 0.080 |
| vc_binom_600 | `s(x, by=z)` | 500 (0) | 0.29 | 0.082 | 0.044 | 0.006 | 0.090 | 0.052 | 0.006 | 0.399 | 0.205 |

The worst check has p = 0.054, against a per-check level of 1.1e-4. Even
without the multiplicity correction, no check rejects at 0.05. The real
co-terms (`s(x1)`, `s(x2)`, `s(x3)`, `s(x)`, level `a`) reject in 94–100% of
fits at .05.

### Power (the interaction term is real)

| cell | term | amplitude | m (failed) | median edf | power .10 | power .05 | power .01 |
|---|---|---|---|---|---|---|---|
| ti_gauss_200_power | `ti(x1, x2)` | 0.6 | 500 (0) | 3.65 | 0.988 | 0.972 | 0.916 |
| ti_binom_1000_power | `ti(x1, x2)` | 0.8 | 500 (0) | 3.84 | 0.992 | 0.978 | 0.908 |
| te_gauss_300_power | `te(x1, x2)` | 0.5 | 499 (1) | 3.71 | 0.994 | 0.984 | 0.966 |
| iso_gauss_300_power | `s(x1, x2)` | 0.5 | 500 (0) | 3.96 | 1.000 | 0.998 | 0.972 |
| vc_gauss_300_power | `s(x, by=z)` | 0.4 | 500 (0) | 1.91 | 0.998 | 0.998 | 0.986 |
| by_gauss_300 | `s(x, by=g):by=g[a]` | 1.5 | 499 (1) | 4.76 | 1.000 | 1.000 | 1.000 |
| by_binom_600 | `s(x, by=g):by=g[a]` | 2.0 | 498 (2) | 3.91 | 1.000 | 1.000 | 1.000 |

## Known limitation: a purely wiggly alternative with no linear part

The kernel weights each component's directions by its prior covariance, so
within the wiggly component the smoothest directions dominate. A curve that
is S-shaped and has no linear part lies mostly in rougher directions. It is
detected weakly: p ≈ .015 at a noncentrality near 107, where a test aimed at
that direction would be near-certain. The same happens for a plain `s(x)`,
so the cause is the prior-direction kernel in general, not the `by=` or
tensor terms. It costs power only; size is exact for any fixed `K`.

## Failed fits

Four of the 8000 fits failed before any p-value was formed. All four failed
in the outer smoothing-parameter optimizer, which is outside this lane's
scope:

| cell | rep | failure (`RemlConvergenceError`) |
|---|---|---|
| by_binom_600 | 16 | stationarity not certified (`\|g\|` = 5.2e-4 against a bound of 8.9e-6) |
| by_binom_600 | 297 | stationarity not certified (`\|g\|` = 5.9e-4 against a bound of 8.9e-6) |
| by_gauss_300 | 157 | a certified optimum was beaten by an uncertified checkpoint |
| te_gauss_300_power | 115 | Newton decrement 3.5e-6 above the band 2.2e-6 after polish |

`results.jsonl` holds one line per cell: the summary above plus every
replicate's `edf`, `ref_df`, `chi_sq` and `p_value` for each term.
