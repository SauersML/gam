# pv-model-comparison: calibration of `basis_check` and `compare_models`

pyGAM audit lane `pv-model-comparison`. The study covers two surfaces:

1. **`basis_check`.** This is the penalized score (Rao) lack-of-fit p-value in
   `Summary.basis_checks` and `Model.basis_check`.
   - With an adequate basis, it must be U(0, 1) across its whole range. A
     conservative p-value is as miscalibrated as an anti-conservative one, so
     every null cell is read in both tails: the size at 0.10, 0.05 and 0.01
     against its MCSE, and a two-sided KS test against U(0, 1).
   - With a basis too small for the truth, it must reject.
2. **`compare_models`.** It must offer a nested-model p-value only if a
   calibrated reference exists.

## How to run

```
python calibrate.py --jobs 4 --out results.json   # about 45 min on 4 cores
```

Every cell has its own fixed seed, recorded in its JSON row, so each cell
reproduces exactly on its own. The tables below are one such run.

Data-generating process:
- `x ~ U(0, 1)`, truth on the linear predictor.
- Gaussian: noise σ = 0.5. Binomial: logit link. Poisson: log link.
- Null cells fit the default `y ~ s(x)` to the truth `sin(2πx)`. The binomial
  and Poisson null cells repeat at `sin(2πx) − 1` (success probabilities
  0.12–0.5, Poisson means 0.14–1.0) and `sin(2πx) − 2` (28 and 34 expected
  events in 200 rows).
- Power cells fit `y ~ s(x, k=4)` to the truth `sin(6x)`.

MCSE is `sqrt(a(1 − a)/tested)`. For a 1000-rep null cell that is 0.0095, 0.0069
and 0.0031 at a = 0.10, 0.05 and 0.01. The seeded regression is
`tests/test_basis_check_pvalue_calibration.py`.

The numbers were measured on a release wheel of this branch (Rust at
`ff14bb59` merged with main at `0b0d0120`). The "product reference" column is
the same build with the conditional reference switched off, so every row
runs the χ²_r reference main ships for binomial and Poisson. The two columns
share seeds, so they see the same data.

## `basis_check`: the estimated-scale defect and its fix

With an estimated scale, the p-value compared `T/r` with `F(r, ν)`:
- `T = UᵀV⁻U/φ̂` is the score statistic.
- `r` is the enrichment rank.
- `ν` is the residual d.f. that `φ̂` was estimated on.

That is not an F ratio. `νφ̂` contains the numerator's own share `T·φ̂`, so
`T/r` is `(ν/r)·Beta(r/2, (ν − r)/2)`, which is under-dispersed relative to
`F(r, ν)`: the test was conservative. It now refers `(T/r)·(ν − r)/(ν − T)` to
`F(r, ν − r)`, the classical added-variable F test of the enrichment columns
appended to the fit (PR #3079). For a Gaussian least-squares fit that is
exactly the nested F test; the Rust test
`estimated_scale_p_value_is_the_exact_added_variable_f_test` pins the identity
to 1e-9.

## `basis_check`: the conditional reference for binomial and Poisson

For a canonical binomial or Poisson fit, the χ²_r reference for the score at
the penalized fit is only first order. Its error at n = 200 is not small, and
it is not one-sided (the product-reference column below: conservative for
binomial, KS p = 3e-5 for low-count Poisson). The score is evaluated at an
estimated β, so its mean, covariance and fourth cumulant are each off by
`O(1/n)` per direction, which is `O(r/n)` on the statistic.

The fix (`crates/gam-terms/src/inference/basis_adequacy.rs`, module header)
takes the score at the unpenalized null MLE and refers it to its law
conditional on the sufficient statistic `Xᵀ(w∘y)`. A canonical link makes that
law free of β, and every weight built from the null MLE is constant under the
conditioning. The mean `δ`, covariance `Σ` and fourth cumulant `K₄` of the
score are expanded to the order the χ² reference misses, and
`T_c = (u − δ)ᵀΣ⁻¹(u − δ)` is referred to `c·χ²_{r/c}` with `c = 1 + K₄/(2r)`,
the scaled χ² that matches both moments.

Where the expansion leaves its range of validity (`Σ` not positive definite,
or `c ≤ 0`), the row reports provenance `conditional_reference_unavailable`
and no p-value. That happens at high-leverage rows with an extreme fitted
mean, which is where the expansion's small parameter is large. It is the
honest answer there: forcing `Σ` positive definite or shrinking the
enrichment until the expansion holds would report a number whose calibration
nobody measured. Those rows are counted in the tables and left out of
`tested`.

## `basis_check`: null cells (adequate basis, 1000 reps each)

Sizes are at 0.10 / 0.05 / 0.01. "Not measured" counts
`conditional_reference_unavailable` rows; "fit refused" counts replicates whose
fit raised `FitError` (no basis check is published for them).

| family | truth | n | tested | not measured | fit refused | sizes | KS p | product reference: sizes | KS p | seed |
|---|---|---|---|---|---|---|---|---|---|---|
| gaussian | `sin 2πx` | 200 | 1000 | — | 0 | 0.101 / 0.046 / 0.006 | 0.60 | — | — | 20260920 |
| gaussian | `sin 2πx` | 2000 | 1000 | — | 0 | 0.099 / 0.050 / 0.006 | 0.90 | — | — | 20260921 |
| binomial | `sin 2πx` | 200 | 934 | 66 | 0 | 0.099 / 0.050 / 0.010 | 0.054 | 0.079 / 0.032 / 0.003 | 0.084 | 20260922 |
| binomial | `sin 2πx` | 2000 | 1000 | 0 | 0 | 0.119 / 0.069 / 0.016 | 0.31 | 0.114 / 0.069 / 0.018 | 0.62 | 20260923 |
| poisson | `sin 2πx` | 200 | 995 | 5 | 0 | 0.119 / 0.063 / 0.016 | 0.026 | 0.103 / 0.050 / 0.007 | 0.9988 | 20260924 |
| poisson | `sin 2πx` | 2000 | 1000 | 0 | 0 | 0.106 / 0.056 / 0.009 | 0.63 | 0.102 / 0.052 / 0.009 | 0.92 | 20260925 |
| binomial | `sin 2πx − 1` | 200 | 810 | 190 | 0 | 0.121 / 0.047 / 0.011 | 0.29 | 0.083 / 0.028 / 0.001 | 0.071 | 20260926 |
| binomial | `sin 2πx − 1` | 2000 | 1000 | 0 | 0 | 0.110 / 0.051 / 0.009 | 0.90 | | | 20260927 |
| poisson | `sin 2πx − 1` | 200 | 851 | 149 | 0 | 0.108 / 0.059 / 0.014 | 0.75 | 0.079 / 0.043 / 0.006 | 3e-5 | 20260928 |
| poisson | `sin 2πx − 1` | 2000 | 1000 | 0 | 0 | 0.110 / 0.049 / 0.013 | 0.86 | | | 20260929 |
| binomial | `sin 2πx − 2` | 200 | 332 | 666 | 2 | 0.105 / 0.066 / 0.018 | **0.038** | 0.057 / 0.024 / 0.002 | 1e-9 | 20260930 |
| binomial | `sin 2πx − 2` | 2000 | 999 | 1 | 0 | 0.099 / 0.047 / 0.010 | 0.77 | | | 20260931 |
| poisson | `sin 2πx − 2` | 200 | 339 | 657 | 3 | **0.062 / 0.030 / 0.006** | **2e-4** | 0.040 / 0.019 / 0.004 | 5e-22 | 20260932 |
| poisson | `sin 2πx − 2` | 2000 | 1000 | 0 | 0 | 0.102 / 0.044 / 0.009 | 0.9991 | | | 20260933 |

The product-reference column for the n = 2000 `sin 2πx` rows is main's run at
the same seeds; the low-rate n = 2000 cells were not rerun on it. At n = 2000
the conditional corrections are `O(r/n)` and the `sin 2πx` rows read alike.
The Gaussian rows carry no product column: their reference is the
added-variable F in both builds. The Poisson `sin 2πx − 2` row also has one
`enrichment_budget_below_realized_width` row.

### Replications on fresh seeds

Three cells above sit more than 2 MCSE from nominal at some level. Each was
rerun on independent seeds to tell a reference error from a fluctuation.
Each run is `calibrate.basis_check_cell(family, n, "y ~ s(x)", truth, reps,
seed)` with the seed named, 1000 replicates per seed unless stated.

- **Binomial, n = 2000, `sin 2πx`** (0.069 at 0.05 above). The product
  reference showed the same excess at this seed, and on four fresh seeds it
  was sized. The conditional reference at seed 81: 1000 tested,
  0.097 / 0.047 / 0.009, KS p = 0.91. At seed 82: 0.082 / 0.043 / 0.010,
  KS p = 0.35. Pooled over both, 2000 replicates: 0.090 / 0.045 / 0.010, KS
  p = 0.43.
- **Poisson, n = 200, `sin 2πx`** (0.063 at 0.05, KS 0.026 above). Seeds 13,
  71 and 72, 3000 replicates: 2975 tested, 0.101 / 0.052 / 0.008, KS p = 0.82.
- **Binomial, n = 200, `sin 2πx − 1`** (0.121 at 0.10 above). Seeds 51–54,
  500 replicates each: 1567 tested (433 not measured), 0.100 / 0.047 / 0.009,
  per-seed KS p 0.08–0.45.
- The Poisson `sin 2πx − 1` cell, for the same regime: seeds 51–54, 500
  replicates each: 1714 tested (286 not measured), 0.101 / 0.047 / 0.009, KS
  p = 0.61.

So at `sin 2πx` and `sin 2πx − 1`, both n, the conditional reference is
U(0, 1) to the resolution of these runs, where the product reference was
conservative (binomial) or failed KS (low-count Poisson).

### Open failure: rare events at n = 200

At `sin 2πx − 2` with n = 200 (28 binomial, 34 Poisson expected events against
a design and enrichment of about 35 directions) the conditional reference is
**not calibrated**:
- About two-thirds of rows are refused as `conditional_reference_unavailable`.
- The Poisson rows it does measure are conservative: 0.062 / 0.030 / 0.006,
  KS p = 2e-4.
- The binomial rows are off too: 0.105 / 0.066 / 0.018 (KS p = 0.038) at the
  bench seed, and 0.088 / 0.039 / 0.010 (489 tested of 1500, KS p = 0.012) at
  seed 41.

The product reference is further off in the same cells (KS p = 1e-9 binomial, 5e-22
Poisson), but that does not make this row acceptable. The expansion's small
parameter is not small in this regime. Resolving it needs the next order of
the expansion or an exact conditional sampler, not a tolerance. At n = 2000
the same truth is sized.

## `basis_check`: power (`y ~ s(x, k=4)`, truth `sin(6x)`, 500 reps each)

| family | n | tested | power 0.10 | power 0.05 | power 0.01 | seed |
|---|---|---|---|---|---|---|
| gaussian | 200 | 500 | 0.152 | 0.086 | 0.024 | 20260934 |
| gaussian | 2000 | 500 | 0.924 | **0.862** | 0.696 | 20260935 |
| binomial | 200 | 500 | 0.106 | 0.060 | 0.008 | 20260936 |
| binomial | 2000 | 500 | 0.154 | 0.052 | 0.010 | 20260937 |
| poisson | 200 | 500 | 0.116 | 0.068 | 0.020 | 20260938 |
| poisson | 2000 | 500 | 0.384 | 0.268 | 0.114 | 20260939 |

The Poisson n = 2000 row reads lower than main's run at another seed
(0.432 / 0.328 / 0.140, seed 20260931), a gap of about 2 MCSE at 0.05 that
was not resolved further here. The product reference's power is not a
target: its null at n = 200 was miscalibrated.

The acceptance demonstration is the Gaussian n = 2000 row: the check flags a
k = 4 basis against `sin(6x)` 86% of the time at 0.05.

A k = 4 cubic basis approximates `sin(6x)` on [0, 1] closely, so the lack of
fit is small against the noise, and power tracks the information in the data.
A Gaussian row carries about 18 times the information a Bernoulli row does, so
power at n = 200, and for binary data at either n, is low. The check cannot
flag structure the data cannot resolve.

## Determinism

The check draws on no random state. The enrichment, the row selection and the
reference are all deterministic:
- A refit of the same rows publishes bit-identical `basis_checks`.
- `Model.basis_check(data)` recomputes from the training rows by re-solving the
  inner problem at the stored smoothing parameters. It agrees with the persisted
  row to solver tolerance (about 1e-9 relative), not to the bit.

`test_basis_check_is_deterministic_and_matches_the_summary` pins both.

## `compare_models`: no nested p-value

`compare_models` ranks fits by the smoothing-corrected AIC and returns **no
p-value** (the criterion was changed in PR #3043, pyGAM audit d11).

The naive likelihood-ratio χ² on the EDF difference is anti-conservative
(Wood, Pya & Saefken 2016), because it ignores two things:
- penalization, since the EDF is not a parameter count;
- smoothing-parameter selection, since the null sits on the boundary of the
  larger model's smoothing parameter.

No implemented reference corrects for both, so none is offered.
`test_compare_models_offers_no_nested_p_value` pins that no key or value in
the comparison document names a p-value.

**The SE of the AIC difference is not added either.** A Vuong-style SE needs
each fit's per-row log-likelihood contributions. A saved model does not carry
those, and `compare_models` accepts saved bytes. For *nested* models under the
null the per-row differences also tend to zero, so the Vuong variance
degenerates in exactly the case this lane is about. The ranking reports
`delta_aic` and the Akaike evidence ratio, with no sampling-error claim.

### Null-nested cells (`s(x)` against `s(x) + s(z)`, z pure noise, 200 reps each)

| family | n | ranked | noise model selected | unranked | seed |
|---|---|---|---|---|---|
| gaussian | 200 | 200 | 0.145 | — | 20260940 |
| gaussian | 2000 | 200 | 0.210 | — | 20260941 |
| binomial | 200 | 198 | 0.167 | 2 correction refused | 20260942 |
| binomial | 2000 | 200 | 0.110 | — | 20260943 |
| poisson | 200 | 187 | 0.107 | 13 correction refused | 20260944 |
| poisson | 2000 | 199 | 0.161 | 1 correction refused | 20260945 |

No cell's document carried a p-value field.

An information criterion is not a test, and the rate at which it picks a
superfluous model is not a size. For reference, classical AIC adds one free
parameter with probability `P(χ²₁ > 2) = 0.157`. "Correction refused" is the
typed refusal from #3043: the corrected AIC is not reported when the smoothing
correction the fit retained is not the first-order identified-subspace
correction it is defined from.

## Coordination

The null cells use the default basis of `s(x)`. Lane `pygam-basis-size` owns
that default. If it changes, rerun `calibrate.py` so the null cells describe
the shipped default. The power cells pin `k=4` and are unaffected.
