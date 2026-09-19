# pv-model-comparison: calibration of `basis_check` and `compare_models`

pyGAM audit lane `pv-model-comparison`. The study covers two surfaces:

1. **`basis_check`.** This is the penalized score (Rao) lack-of-fit p-value in
   `Summary.basis_checks` and `Model.basis_check`.
   - With an adequate basis, it must be uniform or conservative.
   - With a basis too small for the truth, it must reject.
2. **`compare_models`.** It must offer a nested-model p-value only if a valid
   reference exists.

## How to run

```
python calibrate.py --jobs 4 --out results.json   # main study, ~1 h on 4 cores
python replicate.py --out replicate.json          # binomial n=2000 null replication
```

Every cell has its own fixed seed, recorded in its JSON row, so each cell
reproduces exactly on its own.

Data-generating process:
- `x ~ U(0, 1)`, truth on the linear predictor.
- Gaussian: noise σ = 0.5. Binomial: logit link. Poisson: log link.
- Null cells fit the default `y ~ s(x)` to the truth `sin(2πx)`.
- Power cells fit `y ~ s(x, k=4)` to the truth `sin(6x)`.

MCSE is `sqrt(a(1 − a)/tested)`. For a 1000-rep null cell that is 0.0095, 0.0069
and 0.0031 at a = 0.10, 0.05 and 0.01. The seeded regression is
`tests/test_basis_check_pvalue_calibration.py`.

## `basis_check`: the defect and the fix

With an estimated scale, the p-value compared `T/r` with `F(r, ν)`:
- `T = UᵀV⁻U/φ̂` is the score statistic.
- `r` is the enrichment rank.
- `ν` is the residual d.f. that `φ̂` was estimated on.

That is not an F ratio. `νφ̂` is the residual sum of squares of the fit that
excludes the enrichment, so it contains the numerator's own share `T·φ̂`. The
numerator and denominator are therefore not independent: `T/r` is
`(ν/r)·Beta(r/2, (ν − r)/2)`. That distribution is under-dispersed relative to
`F(r, ν)`, so the test was conservative. Over 600 simulated replicates, the
measured `var(T/r)` was 0.070–0.076, against the 0.076 the Beta law predicts.

The fix (`crates/gam-terms/src/inference/basis_adequacy.rs`) refers
`(T/r)·(ν − r)/(ν − T)` to `F(r, ν − r)`. This is the classical added-variable F
test of the enrichment columns appended to the fit. When `ν ≤ r` or `T ≥ ν`,
there are no independent residual d.f., and the check reports no p-value.

For a Gaussian least-squares fit, the new p-value is exactly the nested F test
`((RSS₀ − RSS₁)/r)/(RSS₁/(ν − r))`. The Rust test
`estimated_scale_p_value_is_the_exact_added_variable_f_test` pins that identity
to 1e-9. It failed before the fix: p = 0.5666 against the exact 0.5739.
Known-scale families (binomial, Poisson) go through the χ²_r branch and are
unchanged.

## `basis_check`: null cells (adequate basis, 1000 reps each)

| family | n | tested | refused | size 0.10 | size 0.05 | size 0.01 | KS p | seed |
|---|---|---|---|---|---|---|---|---|
| gaussian, **before fix** | 200 | 1000 | 0 | 0.065 | 0.023 | 0.001 | 0.006 | 20260920 |
| gaussian, **before fix** | 2000 | 1000 | 0 | 0.092 | 0.048 | 0.005 | 0.862 | 20260921 |
| gaussian | 200 | 1000 | 0 | 0.101 | 0.046 | 0.006 | 0.596 | 20260920 |
| gaussian | 2000 | 1000 | 0 | 0.099 | 0.050 | 0.006 | 0.896 | 20260921 |
| binomial | 200 | 1000 | 0 | 0.079 | 0.032 | 0.003 | 0.084 | 20260922 |
| binomial | 2000 | 1000 | 0 | 0.114 | 0.069 | 0.018 | 0.620 | 20260923 |
| binomial, replication | 2000 | 2000 | 0 | 0.0985 | 0.0485 | 0.0055 | 0.64–0.73 per 500 | 20261001–20261004 |
| poisson | 200 | 998 | 2 | 0.103 | 0.050 | 0.007 | 0.9996 | 20260924 |
| poisson | 2000 | 1000 | 0 | 0.102 | 0.052 | 0.009 | 0.917 | 20260925 |

What the table shows:

- **Gaussian.** After the fix, every level is within 1 MCSE of nominal at both n,
  and the n = 200 KS p-value rises from 0.006 to 0.60. Before the fix, n = 200
  rejected at half its level (0.023 at 0.05, 0.001 at 0.01).
- **Binomial, n = 200: conservative.**
  - At 0.10 and 0.05 the size is more than 2 MCSE *below* nominal, with KS p =
    0.084.
  - The lane contract permits "U(0, 1) or conservative", so this is valid, but the
    test loses some power at this size.
  - The fix does not touch it: known scale, and identical numbers before and after
    at the same seed.
  - The cause is not diagnosed here. It is a small-sample property of the
    Bernoulli score test at 200 rows.
- **Binomial, n = 2000: chance excess.**
  - The main-study cell (seed 20260923) is more than 2 MCSE *above* nominal at
    0.05 (0.069) and 0.01 (0.018).
  - The four-seed, 2000-rep replication on fresh seeds (`replicate.json`) is sized:
    0.0985 / 0.0485 / 0.0055, with every level within 1 MCSE.
  - Pooled over all 3000 reps, the size at 0.05 is 0.0553 (MCSE 0.0040) and at
    0.01 it is 0.0097. So the main cell's excess did not replicate: it is a
    seed-level fluctuation, not a miscalibrated reference.
- **Poisson** is sized at both n.
- **Refusals.** The two Poisson n = 200 refusals are `gamfit.errors.FitError`, where the
  outer optimizer could not certify the fit. A refused fit publishes no basis
  check, so it is excluded from `tested`. Every tested row had provenance
  `radial_enrichment`.

## `basis_check`: power (`y ~ s(x, k=4)`, truth `sin(6x)`, 500 reps each)

| family | n | tested | refused | power 0.10 | power 0.05 | power 0.01 | seed |
|---|---|---|---|---|---|---|---|
| gaussian | 200 | 500 | 0 | 0.154 | 0.080 | 0.022 | 20260926 |
| gaussian | 2000 | 500 | 0 | 0.928 | **0.846** | 0.650 | 20260927 |
| binomial | 200 | 500 | 0 | 0.082 | 0.034 | 0.004 | 20260928 |
| binomial | 2000 | 500 | 0 | 0.136 | 0.074 | 0.012 | 20260929 |
| poisson | 200 | 500 | 0 | 0.118 | 0.054 | 0.012 | 20260930 |
| poisson | 2000 | 500 | 0 | 0.432 | 0.328 | 0.140 | 20260931 |

The acceptance demonstration is the Gaussian n = 2000 row. The check flags a
k = 4 basis against `sin(6x)` 85% of the time at 0.05.

A k = 4 cubic basis approximates `sin(6x)` on [0, 1] closely enough that the
lack of fit is small against the noise. Power therefore tracks the information
in the data:
- A Gaussian row carries about 18 times the information a Bernoulli row does.
- Power at n = 200, and for binary data at either n, is correspondingly low.

That is the check behaving as a test, not a defect. It cannot flag structure
the data cannot resolve.

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
p-value** (the criterion was changed in PR #3043, pyGAM audit d11). That is
the correct answer for this lane.

The naive likelihood-ratio χ² on the EDF difference is anti-conservative
(Wood, Pya & Saefken 2016), because it ignores two things:
- penalization, since the EDF is not a parameter count;
- smoothing-parameter selection, since the null sits on the boundary of the
  larger model's smoothing parameter.

No implemented reference corrects for both, so none is offered.
`test_compare_models_offers_no_nested_p_value` pins that no key or value in
the comparison document names a p-value.

**The SE of the AIC difference is not added either.** The lane allowed a
cAIC difference with its SE as the fallback. A Vuong-style SE needs each fit's
per-row log-likelihood contributions. A saved model does not carry those, and
`compare_models` accepts saved bytes. Also, for *nested* models under the
null, the per-row differences tend to zero, and the Vuong variance degenerates
at exactly the case this lane is about. An SE that is uninformative when the
question matters most is not reported. The ranking reports `delta_aic` and the
Akaike evidence ratio, with no sampling-error claim attached.

### Null-nested cells (`s(x)` against `s(x) + s(z)`, z pure noise, 200 reps each)

| family | n | ranked | noise model selected | unranked | seed |
|---|---|---|---|---|---|
| gaussian | 200 | 200 | 0.150 | — | 20260932 |
| gaussian | 2000 | 191 | 0.188 | 8 correction refused, 1 fit refused | 20260933 |
| binomial | 200 | 182 | 0.346 | 18 correction refused | 20260934 |
| binomial | 2000 | 187 | 0.257 | 13 correction refused | 20260935 |
| poisson | 200 | 184 | 0.152 | 16 correction refused | 20260936 |
| poisson | 2000 | 192 | 0.224 | 8 correction refused | 20260937 |

No cell's document carried a p-value field.

**Selection rates.** An information criterion is not a test, and the rate at
which it picks a superfluous model is not a size. For reference, classical AIC
adds one free parameter with probability `P(χ²₁ > 2) = 0.157`. The binomial
n = 200 rate (0.35) is well above that.

**Refusals.** "Correction refused" is the typed refusal from #3043. The
corrected AIC is not reported when the smoothing correction the fit retained is
not the first-order identified-subspace correction it is defined from. Between
0% and 9% of the fits in each cell hit it. Both belong to the criterion from
#3043, not to this lane.

## Coordination

The null cells use the default basis of `s(x)`. Lane `pygam-basis-size` owns
that default. If it changes, rerun `calibrate.py` so the null cells describe
the shipped default. The power cells pin `k=4` and are unaffected.
