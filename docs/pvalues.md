# p-values and their calibration

gamfit reports p-values in four places. Each one tests a different null
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
| `model.summary().smooth_terms[i]["p_value"]`, penalized smooth | the smooth is identically zero | variance-component score statistic `chi_sq` (Lin 1997; Zhang & Lin 2003) | its exact null law `Σ w_i χ²₁` when the scale is known (binomial, Poisson, negative binomial); `P(Σ w_i χ²₁ − (q/ρ)·χ²_ρ > 0)` when the fit estimates it (Gaussian, Gamma) |
| `model.summary().smooth_terms[i]["p_value"]`, random effect `group(g)` | the variance component is zero | variance-component score statistic `‖X̃_Rᵀv‖²` (Lin 1997) | its exact spectral law `Σ μ_j χ²₁` (Wood 2013), against the residual `χ²_ν` when the scale is estimated |
| `model.summary().parametric_terms[i]["p_value"]`, unpenalized coefficient | the coefficient is zero | Wald ratio `estimate / std_error` | `t` on the residual degrees of freedom when the fit estimates the scale, `N(0, 1)` when it is known |
| `model.summary().parametric_terms[i]["p_value"]`, ridged linear term (the default) | the slope is zero | variance-component score statistic of the slope's ridge, reported as its signed square root | `χ²₁` (known scale) or `F_{1, ν}` on the unpenalized residual (estimated scale); for a Gaussian fit it is exactly the partial `t` of the unpenalized slope |
| `model.smooth_significance(data)[i]["p_value_corrected"]` | the smooth is identically zero | likelihood ratio `statistic_lr` from a constrained refit without the smooth | the statistic's exact null law `Σ w_j χ²₁`, Bartlett-corrected |
| `model.summary().basis_checks[i]["p_value"]` | the basis of the smooth is rich enough | penalized score (Rao) lack-of-fit statistic | `χ²_{enrichment_rank}` |

A bare categorical factor (`g`) is lowered to a penalized random-effect
block. Like `group(g)`, its `smooth_terms` row carries the recorded
variance-component score test, or a typed reason why that test is unavailable.
It has no single-coefficient `parametric_terms` row and no
`smooth_significance` row.

### The score tests in `summary()`

`summary()` needs only the saved model, so its p-value is the one available
everywhere. A penalized smooth `f_j = X_j β_j` is the random effect
`β_j ~ N(0, τ·S_j⁻)`, and "no effect" is the boundary null `τ = 0`. The score
for `τ` there is a quadratic form in the term's score vector:

```text
s = b_j − G_jo H_oo⁻¹ b_o,        b = Hβ̂,        Q = sᵀ K s / φ̂,
```

with `G = XᵀWX`, `H = G + S(λ)`, `o` every other coefficient and
`K = Σ_l K_l / t_l` the term's structural penalties, each on its own null
scale. Under the null `Q` has exactly the law `Σ w_i χ²₁` with
`w = eig(K^½ C K^½)`, `C` the covariance of `s`. When the scale is
estimated, the tail is the signed weighted chi-square
`P(Σ w_i χ²₁ − (q/ρ)·χ²_ρ > 0)`. The reported `chi_sq` is scaled so that its
null mean is `ref_df = (Σw)² / Σw²`.

A Wald statistic on `β̂_j` is a function of the term's own fitted smoothing
parameter, which under the null runs to its boundary in a large share of
replicates and leaves an atom of p-values at 1. The score test never reads
`λ_j`: `s` fits the other terms only, and `K` is fixed by the basis. Its
reference law is therefore exact at the fitted `λ_o`.

A random effect `group(g)` carries the score test of its variance component
`σ²_b = 0` (Lin 1997): `T = ‖u‖²`, `u = X̃_Rᵀ v`, with `v` the working
residual of the model without the random effect and `X̃_R` the random-effect
design with every other column projected out. Its null law is the exact
spectral sum `Σ μ_j χ²₁` (Wood 2013). When the scale is estimated, `T` is
referred to the unpenalized residual, `P(Σ μ_j χ²₁ − t·χ²_ν > 0)`.

A row that cannot be scored has no `p_value` and carries a
`p_value_unavailable` reason instead.

### Linear coefficient tests in `parametric_terms`

An unpenalized coefficient carries its Wald statistic `estimate / std_error`
and its p-value. The reference is Student-t on the residual degrees of freedom
when the scale is estimated and N(0, 1) when it is known.

A linear term is ridged by default, and REML picks the ridge from the same
data, so its estimate is shrunk toward zero and shrinks furthest exactly when
the slope is null. The Wald ratio of that shrunk estimate is far below its
nominal law under the null: its p-values pile up near one. The ridged row
therefore reports the score test of the ridge's variance component, which
never reads the shrunk estimate (gam#3573). A ridged row on a fit that records
no such test (the location-scale, GAMLSS and expectile families) has no
statistic and no p-value, rather than the invalid Wald ratio.

A ridged row reports the signed square root of its score statistic, with
the sign of the fitted slope, and the score test's p-value.

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
estimated-scale score tests in `summary()` use the same signed form.

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

The committed baseline was measured at the git revision in the table's
header, before `summary()` carried the score tests and `parametric_terms`.
At that revision the "Wald (`summary`)" rows measure Wood's (2013)
rank-truncated Wald test, the "coefficient" rows measure the two-sided normal
tail of `estimate / std_error`, and a random effect had no p-value. The rows
describe the current `summary()` only once the baseline is regenerated (gam#3722).

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

Plan `quick` at git `4f17dc876bda`, gamfit=0.1.268, pygam=0.12.0, pygam_gs=0.12.0.

79 rows, 413 two-sided checks (size at each level, and KS), family-wise false-alarm rate 0.001 (Bonferroni: each check at 2.4e-06).

| cell | surface | usable | size@0.10 | size@0.05 | size@0.01 | KS D (p) | power@0.05 | verdict |
|---|---|---|---|---|---|---|---|---|
| binomial/n=200/concurvity | gamfit LR (`smooth_significance`) | 100/100 | 0.090 ± 0.029 | 0.050 ± 0.022 | 0.010 ± 0.010 | 0.289 (6.5e-08) | 0.300 | **NOT UNIFORM** |
| binomial/n=200/concurvity | gamfit Wald (`summary`) | 100/100 | 0.080 ± 0.027 | 0.070 ± 0.026 | 0.020 ± 0.014 | 0.577 (3.8e-32) | 0.160 | **NOT UNIFORM** |
| binomial/n=200/concurvity | pyGAM, fixed lam | 100/100 | 0.080 ± 0.027 | 0.030 ± 0.017 | 0.000 ± 0.000 | 0.072 (0.65) | 0.130 | calibrated |
| binomial/n=200/concurvity | pyGAM, gridsearch | 100/100 | 0.110 ± 0.031 | 0.060 ± 0.024 | 0.000 ± 0.000 | 0.106 (0.2) | 0.180 | calibrated |
| binomial/n=200/factor | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| binomial/n=200/factor | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| binomial/n=200/factor | pyGAM, fixed lam | 100/100 | 0.110 ± 0.031 | 0.040 ± 0.020 | 0.000 ± 0.000 | 0.056 (0.9) | 0.720 | calibrated |
| binomial/n=200/factor | pyGAM, gridsearch | 100/100 | 0.120 ± 0.032 | 0.030 ± 0.017 | 0.000 ± 0.000 | 0.074 (0.61) | 0.730 | calibrated |
| binomial/n=200/linear | gamfit coefficient | 100/100 | 0.010 ± 0.010 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.748 (2.5e-58) | 0.670 | **NOT UNIFORM** |
| binomial/n=200/linear | pyGAM, fixed lam | 100/100 | 0.050 ± 0.022 | 0.030 ± 0.017 | 0.000 ± 0.000 | 0.106 (0.2) | 0.810 | calibrated |
| binomial/n=200/linear | pyGAM, gridsearch | 100/100 | 0.050 ± 0.022 | 0.030 ± 0.017 | 0.000 ± 0.000 | 0.124 (0.084) | 0.820 | calibrated |
| binomial/n=200/re | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| binomial/n=200/re | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| binomial/n=200/smooth | gamfit LR (`smooth_significance`) | 100/100 | 0.100 ± 0.030 | 0.050 ± 0.022 | 0.030 ± 0.017 | 0.265 (1.1e-06) | 0.700 | **NOT UNIFORM** |
| binomial/n=200/smooth | gamfit Wald (`summary`) | 100/100 | 0.060 ± 0.024 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.555 (1.5e-29) | 0.510 | **NOT UNIFORM** |
| binomial/n=200/smooth | pyGAM, fixed lam | 100/100 | 0.100 ± 0.030 | 0.050 ± 0.022 | 0.010 ± 0.010 | 0.103 (0.22) | 0.290 | calibrated |
| binomial/n=200/smooth | pyGAM, gridsearch | 100/100 | 0.070 ± 0.026 | 0.030 ± 0.017 | 0.010 ± 0.010 | 0.065 (0.76) | 0.230 | calibrated |
| gamma/n=200/concurvity | gamfit LR (`smooth_significance`) | 100/100 | 0.200 ± 0.040 | 0.140 ± 0.035 | 0.100 ± 0.030 | 0.331 (2.9e-10) | 0.420 | **ANTI-CONSERVATIVE** at 0.01; **NOT UNIFORM** |
| gamma/n=200/concurvity | gamfit Wald (`summary`) | 100/100 | 0.040 ± 0.020 | 0.020 ± 0.014 | 0.010 ± 0.010 | 0.536 (2e-27) | 0.190 | **NOT UNIFORM** |
| gamma/n=200/concurvity | pyGAM, fixed lam | 100/100 | 0.150 ± 0.036 | 0.100 ± 0.030 | 0.020 ± 0.014 | 0.123 (0.089) | 0.160 | calibrated |
| gamma/n=200/concurvity | pyGAM, gridsearch | 100/100 | 0.120 ± 0.032 | 0.060 ± 0.024 | 0.030 ± 0.017 | 0.174 (0.0041) | 0.210 | calibrated |
| gamma/n=200/factor | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| gamma/n=200/factor | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| gamma/n=200/factor | pyGAM, fixed lam | 100/100 | 0.140 ± 0.035 | 0.080 ± 0.027 | 0.010 ± 0.010 | 0.095 (0.31) | 0.730 | calibrated |
| gamma/n=200/factor | pyGAM, gridsearch | 100/100 | 0.140 ± 0.035 | 0.070 ± 0.026 | 0.020 ± 0.014 | 0.104 (0.21) | 0.720 | calibrated |
| gamma/n=200/linear | gamfit coefficient | 100/100 | 0.040 ± 0.020 | 0.030 ± 0.017 | 0.030 ± 0.017 | 0.658 (5.5e-43) | 0.760 | **NOT UNIFORM** |
| gamma/n=200/linear | pyGAM, fixed lam | 100/100 | 0.100 ± 0.030 | 0.060 ± 0.024 | 0.030 ± 0.017 | 0.078 (0.55) | 0.880 | calibrated |
| gamma/n=200/linear | pyGAM, gridsearch | 100/100 | 0.080 ± 0.027 | 0.040 ± 0.020 | 0.020 ± 0.014 | 0.067 (0.74) | 0.890 | calibrated |
| gamma/n=200/re | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| gamma/n=200/re | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| gamma/n=200/smooth | gamfit LR (`smooth_significance`) | 100/100 | 0.150 ± 0.036 | 0.120 ± 0.032 | 0.060 ± 0.024 | 0.423 (1e-16) | 0.680 | **NOT UNIFORM** |
| gamma/n=200/smooth | gamfit Wald (`summary`) | 100/100 | 0.100 ± 0.030 | 0.030 ± 0.017 | 0.000 ± 0.000 | 0.508 (2e-24) | 0.520 | **NOT UNIFORM** |
| gamma/n=200/smooth | pyGAM, fixed lam | 100/100 | 0.160 ± 0.037 | 0.070 ± 0.026 | 0.030 ± 0.017 | 0.168 (0.0061) | 0.290 | calibrated |
| gamma/n=200/smooth | pyGAM, gridsearch | 100/100 | 0.110 ± 0.031 | 0.060 ± 0.024 | 0.000 ± 0.000 | 0.064 (0.79) | 0.270 | calibrated |
| gaussian/n=200/concurvity | gamfit LR (`smooth_significance`) | 100/100 | 0.080 ± 0.027 | 0.020 ± 0.014 | 0.000 ± 0.000 | 0.435 (9.3e-18) | 0.360 | **NOT UNIFORM** |
| gaussian/n=200/concurvity | gamfit Wald (`summary`) | 100/100 | 0.040 ± 0.020 | 0.030 ± 0.017 | 0.000 ± 0.000 | 0.633 (2.6e-39) | 0.200 | **NOT UNIFORM** |
| gaussian/n=200/concurvity | pyGAM, fixed lam | 100/100 | 0.070 ± 0.026 | 0.020 ± 0.014 | 0.000 ± 0.000 | 0.081 (0.5) | 0.120 | calibrated |
| gaussian/n=200/concurvity | pyGAM, gridsearch | 100/100 | 0.120 ± 0.032 | 0.040 ± 0.020 | 0.000 ± 0.000 | 0.083 (0.47) | 0.240 | calibrated |
| gaussian/n=200/factor | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| gaussian/n=200/factor | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| gaussian/n=200/factor | pyGAM, fixed lam | 100/100 | 0.100 ± 0.030 | 0.030 ± 0.017 | 0.010 ± 0.010 | 0.060 (0.84) | 0.730 | calibrated |
| gaussian/n=200/factor | pyGAM, gridsearch | 100/100 | 0.130 ± 0.034 | 0.050 ± 0.022 | 0.010 ± 0.010 | 0.062 (0.82) | 0.750 | calibrated |
| gaussian/n=200/linear | gamfit coefficient | 100/100 | 0.030 ± 0.017 | 0.020 ± 0.014 | 0.010 ± 0.010 | 0.690 (6.5e-48) | 0.740 | **NOT UNIFORM** |
| gaussian/n=200/linear | pyGAM, fixed lam | 100/100 | 0.080 ± 0.027 | 0.050 ± 0.022 | 0.010 ± 0.010 | 0.105 (0.2) | 0.830 | calibrated |
| gaussian/n=200/linear | pyGAM, gridsearch | 100/100 | 0.070 ± 0.026 | 0.030 ± 0.017 | 0.020 ± 0.014 | 0.052 (0.93) | 0.850 | calibrated |
| gaussian/n=200/re | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| gaussian/n=200/re | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| gaussian/n=200/smooth | gamfit LR (`smooth_significance`) | 99/100 | 0.121 ± 0.033 | 0.030 ± 0.017 | 0.000 ± 0.000 | 0.434 (1.7e-17) | 0.670 | **NOT UNIFORM**; 1 unusable |
| gaussian/n=200/smooth | gamfit Wald (`summary`) | 99/100 | 0.061 ± 0.024 | 0.051 ± 0.022 | 0.000 ± 0.000 | 0.485 (6e-22) | 0.420 | **NOT UNIFORM**; 1 unusable |
| gaussian/n=200/smooth | pyGAM, fixed lam | 100/100 | 0.050 ± 0.022 | 0.010 ± 0.010 | 0.000 ± 0.000 | 0.081 (0.51) | 0.250 | calibrated |
| gaussian/n=200/smooth | pyGAM, gridsearch | 100/100 | 0.010 ± 0.010 | 0.010 ± 0.010 | 0.000 ± 0.000 | 0.142 (0.032) | 0.220 | calibrated |
| gaussian/n=200/ti | gamfit LR (`smooth_significance`) | 91/100 | 0.110 ± 0.033 | 0.044 ± 0.021 | 0.011 ± 0.011 | 0.257 (9e-06) | 0.418 | **ANTI-CONSERVATIVE** at 0.01; **NOT UNIFORM**; 9 unusable |
| gaussian/n=200/ti | gamfit Wald (`summary`) | 91/100 | 0.088 ± 0.030 | 0.033 ± 0.019 | 0.022 ± 0.015 | 0.370 (1e-11) | 0.187 | **ANTI-CONSERVATIVE** at 0.01; **NOT UNIFORM**; 9 unusable |
| negbin/n=200/concurvity | gamfit LR (`smooth_significance`) | 100/100 | 0.100 ± 0.030 | 0.060 ± 0.024 | 0.010 ± 0.010 | 0.272 (4.8e-07) | 0.350 | **NOT UNIFORM** |
| negbin/n=200/concurvity | gamfit Wald (`summary`) | 100/100 | 0.070 ± 0.026 | 0.070 ± 0.026 | 0.040 ± 0.020 | 0.698 (2.8e-49) | 0.180 | **NOT UNIFORM** |
| negbin/n=200/factor | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| negbin/n=200/factor | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| negbin/n=200/linear | gamfit coefficient | 100/100 | 0.030 ± 0.017 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.737 (3.9e-56) | 0.690 | **NOT UNIFORM** |
| negbin/n=200/re | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| negbin/n=200/re | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| negbin/n=200/smooth | gamfit LR (`smooth_significance`) | 100/100 | 0.140 ± 0.035 | 0.080 ± 0.027 | 0.030 ± 0.017 | 0.242 (1.3e-05) | 0.700 | calibrated |
| negbin/n=200/smooth | gamfit Wald (`summary`) | 100/100 | 0.070 ± 0.026 | 0.040 ± 0.020 | 0.010 ± 0.010 | 0.606 (8.9e-36) | 0.610 | **NOT UNIFORM** |
| poisson/n=200/concurvity | gamfit LR (`smooth_significance`) | 100/100 | 0.070 ± 0.026 | 0.010 ± 0.010 | 0.000 ± 0.000 | 0.226 (5.9e-05) | 0.310 | calibrated |
| poisson/n=200/concurvity | gamfit Wald (`summary`) | 100/100 | 0.080 ± 0.027 | 0.040 ± 0.020 | 0.010 ± 0.010 | 0.578 (3.3e-32) | 0.170 | **NOT UNIFORM** |
| poisson/n=200/concurvity | pyGAM, fixed lam | 100/100 | 0.060 ± 0.024 | 0.010 ± 0.010 | 0.010 ± 0.010 | 0.067 (0.74) | 0.110 | calibrated |
| poisson/n=200/concurvity | pyGAM, gridsearch | 100/100 | 0.100 ± 0.030 | 0.050 ± 0.022 | 0.000 ± 0.000 | 0.155 (0.015) | 0.220 | calibrated |
| poisson/n=200/factor | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| poisson/n=200/factor | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| poisson/n=200/factor | pyGAM, fixed lam | 100/100 | 0.120 ± 0.032 | 0.070 ± 0.026 | 0.010 ± 0.010 | 0.112 (0.15) | 0.780 | calibrated |
| poisson/n=200/factor | pyGAM, gridsearch | 100/100 | 0.120 ± 0.032 | 0.060 ± 0.024 | 0.010 ± 0.010 | 0.065 (0.76) | 0.800 | calibrated |
| poisson/n=200/linear | gamfit coefficient | 100/100 | 0.010 ± 0.010 | 0.010 ± 0.010 | 0.000 ± 0.000 | 0.648 (1.4e-41) | 0.680 | **NOT UNIFORM** |
| poisson/n=200/linear | pyGAM, fixed lam | 100/100 | 0.100 ± 0.030 | 0.020 ± 0.014 | 0.010 ± 0.010 | 0.088 (0.39) | 0.820 | calibrated |
| poisson/n=200/linear | pyGAM, gridsearch | 100/100 | 0.060 ± 0.024 | 0.030 ± 0.017 | 0.010 ± 0.010 | 0.094 (0.32) | 0.830 | calibrated |
| poisson/n=200/re | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| poisson/n=200/re | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| poisson/n=200/smooth | gamfit LR (`smooth_significance`) | 100/100 | 0.120 ± 0.032 | 0.040 ± 0.020 | 0.010 ± 0.010 | 0.214 (0.00016) | 0.810 | calibrated |
| poisson/n=200/smooth | gamfit Wald (`summary`) | 100/100 | 0.070 ± 0.026 | 0.060 ± 0.024 | 0.020 ± 0.014 | 0.537 (1.6e-27) | 0.610 | **NOT UNIFORM** |
| poisson/n=200/smooth | pyGAM, fixed lam | 100/100 | 0.060 ± 0.024 | 0.030 ± 0.017 | 0.010 ± 0.010 | 0.051 (0.95) | 0.340 | calibrated |
| poisson/n=200/smooth | pyGAM, gridsearch | 100/100 | 0.050 ± 0.022 | 0.010 ± 0.010 | 0.010 ± 0.010 | 0.066 (0.75) | 0.340 | calibrated |

<!-- END pvalue_calibration table -->
