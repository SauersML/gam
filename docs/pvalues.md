# p-values and their calibration

gamfit reports p-values in three places. Each one tests a different null
hypothesis against a different reference distribution. This page says what
each one is, and shows how well each holds its nominal size in a seeded
simulation grid (`bench/pvalue_calibration`).

A p-value is **calibrated** when it is Uniform(0, 1) under its null
hypothesis: `P(p ≤ a) = a` at every level `a`. An anti-conservative p-value
(size above nominal) rejects a true null more often than its level promises.
A conservative one (size below nominal, or a point mass at 1) rejects it less
often, which hides real effects and misstates the evidence. Both are
miscalibrated.

## What gamfit reports

| where | tests | statistic | reference distribution |
|---|---|---|---|
| `model.summary().smooth_terms[i]["p_value"]` | the smooth is identically zero | Wood (2013) rank-truncated Wald statistic `chi_sq` | `χ²_{ref_df}` when the scale is known (binomial, Poisson, negative binomial); `F_{ref_df, n − edf}` on `chi_sq / ref_df` when the fit estimates it (Gaussian, Gamma) |
| `model.smooth_significance(data)[i]["p_value_corrected"]` | the smooth is identically zero | likelihood ratio `statistic_lr` from a constrained refit without the smooth | the statistic's exact null law `Σ w_j χ²₁`, Bartlett-corrected |
| `model.summary().basis_checks[i]["p_value"]` | the basis of the smooth is rich enough | penalized score (Rao) lack-of-fit statistic | `χ²_{enrichment_rank}` |

Some terms have no p-value on any surface:

- Parametric coefficients carry an estimate and a standard error
  (`summary().coefficients`), not a p-value. `model.term_blocks` gives the
  coefficient columns of each term.
- A random effect (`group(g)`) or a factor (`g`) has a `smooth_terms` row with
  `edf` only, and no row in `smooth_significance`.

### The Wald test in `summary()`

`summary()` needs only the saved model, so its p-value is the one available
everywhere. The statistic follows Wood (2013):

1. The coefficient block of the term, and its posterior covariance, are
   mapped into fitted-value space by the design whitening `R` (`RᵀR = XᵀWX`).
2. The whitened covariance is inverted with a spectral pseudo-inverse
   truncated at rank `round(edf)`.

The whitening matters. Truncating the raw covariance would keep the
heavily-penalized, signal-free directions and discard the fitted function.

`ref_df` is the influence-trace degrees of freedom
`tr(F_jj)² / tr(F_jj²)`, never below the rank actually used. The covariance
already includes the scale, so the F statistic is `chi_sq / ref_df` with no
further division by the dispersion.

This is a first-order reference. Under a penalty the Wald statistic is itself
a weighted sum of `χ²₁`, not a central `χ²` or `F`. For a term that REML shrinks
to the boundary, the reported p-value is conservative, often with a point mass
at 1.

### The likelihood-ratio test in `smooth_significance(data)`

`smooth_significance` refits the model once per penalized smooth with that
smooth removed. It needs the training data for this. The statistic is
`W = 2(ℓ_full − ℓ_null)`.

At fixed smoothing parameters, the penalized LR statistic has exactly the law
`Σ_j w_j χ²₁`, with `w = eig(2F_jj − F_jj²)` on the tested block of the
influence matrix. Its p-value is `P(Σ w_j χ²₁ > W)`, computed by Imhof
inversion of the characteristic function. `reference_source` says which
reference was used:

- `"null_spectrum"`: the exact law. `reference_weights` holds `w`.
- `"spectral_moment_match"`: only the first two moments of the spectrum were
  available, so the reference is `P(χ²_ν > W / g)`, a two-moment match.
- `"unit_weight_fallback"`.

When the scale is estimated (Gaussian), `W` is a monotone function of the
spectrum's quadratic form and of the residual sum of squares. The p-value
inverts that map exactly, as `P(Q − c(W)·V > 0)`, rather than scoring `W`
against the known-scale law. `reference_residual_df` and
`reference_deterministic_offset` carry the two constants of the map. The
F reference in `summary()` exists for the same reason.

The raw statistic is then Bartlett-corrected, `W* = W / c`, with the exact
Lawley factor `c`. The headline value is `p_value_corrected`.
`correction_provenance` says which factor was used:

- `"lawley_lr_estimated_lambda"`: the factor includes the sampling variation
  of the estimated smoothing parameters, so the p-value accounts for `λ̂`
  having been chosen from the same data. `p_value_conditional` is the tail at
  fixed `λ`, before that adjustment, and `rho_variation_shift` is the part of
  the factor due to it.
- `"lawley_lr_fixed_lambda"`: the fixed-`λ` factor alone, used when the
  estimated-`λ` terms are unavailable.
- `"none"`: the family has no closed-form cumulants or the null refit did not
  converge, so the uncorrected reference stands.

`p_value_uncorrected` is the tail of the raw `W`. `p_value_bound` is the
certified absolute accuracy of the p-values. `material` is `True` when the
correction moves the p-value or the factor by more than 10%, which means n is
small for this model.

### The basis check

`basis_checks` asks a different question: whether the residuals still carry
structure in a smooth's covariates that its basis cannot reach. A small
`p_value` means `k` is too small. The `basis_check()` section of
[Diagnostics](diagnostics.md) gives the construction. It is not a test of
whether the term matters.

## Calibration

`bench/pvalue_calibration` draws a seeded grid of datasets, fits each one,
and reads the p-value of one tested term under its null hypothesis and under
a matched local alternative.

The grid spans:

- **families:** Gaussian, binomial, Poisson, Gamma, negative binomial;
- **n:** 60, 200, 1000, 5000;
- **null structures:** a null smooth beside a real one, a null linear term, a
  null factor, a null `ti` interaction, a null random effect, and a null
  smooth whose covariate has latent correlation 0.9 with the real smooth's
  (concurvity).

The table below is the `quick` plan: every family and null structure at
n = 200, 100 reps per cell. The `ti` null runs for Gaussian only, because the
LR test on a `ti` model takes minutes per fit. The `nightly` plan runs the full
grid at 500 reps. `bench/pvalue_calibration/README.md` gives the plans and the
data-generating process.

The alternative has the same noncentrality (9, a 3-sigma effect) in every
cell, so the power column compares like with like across families and
sample sizes. The pyGAM columns fit pyGAM 0.12 to the same data wherever pyGAM
has a counterpart. pyGAM has no `ti`, random effect or negative binomial, and
no LR test.

For the linear null, gamfit has no coefficient p-value, so the table's
"coefficient" row is the two-sided normal tail of `estimate / std_error`.
That row tests the reported standard error, which is what such a p-value
would be built from.

How to read the table:

- **size@a** is the rejection rate at level `a` under the null, `± MCSE`, where
  MCSE = `sqrt(size (1 − size) / R)`.
- **KS D** is the Kolmogorov–Smirnov distance of the null p-values from
  uniform.
- **power@0.05** is the rejection rate under the matched alternative.
- **verdict** is **ANTI-CONSERVATIVE** at `a` when the rejection count
  exceeds what a calibrated p-value reaches except with probability
  `10⁻³ / (number of checks)`, and **CONSERVATIVE** at `a` when it falls
  below what a calibrated p-value stays above with the same probability. Both
  bounds are Binomial(R, a) quantiles, not hand-picked tolerances. It is
  **NOT UNIFORM** when the KS test of the null p-values against Uniform(0, 1)
  rejects at the same per-check level. A rep that produced no p-value takes
  the value worst for each check, so a row passes only if it would pass
  whatever those reps would have reported. A row passes as **calibrated** only
  when every check passes.

The table below is generated from the committed baseline
`bench/pvalue_calibration/baseline/quick/`; do not edit it by hand.
Regenerate it with:

```bash
python -m bench.pvalue_calibration.run quick --out bench/pvalue_calibration/baseline/quick
python -m bench.pvalue_calibration.report bench/pvalue_calibration/baseline/quick --docs docs/pvalues.md
```

<!-- BEGIN pvalue_calibration table (generated; do not edit) -->
<!-- END pvalue_calibration table -->
