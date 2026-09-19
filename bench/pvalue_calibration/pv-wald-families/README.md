# pv-wald-families: the smooth-term Wald p-value across response families

Target: `summary().smooth_terms[*].p_value`, i.e. `wood_smooth_test`
(`crates/gam-terms/src/inference/smooth_test.rs`) as it is called by the
production summary walk `smooth_term_summary_rows`.

Design: `y ~ s(x1) + s(x2)`, `x1, x2 ~ U(0, 1)` independent,
`f1 = sin(2πx1)` enters the linear predictor, `x2` never does (power runs
add `0.4·sin(2πx2)`). Seven families:

| family | DGP (linear predictor η) | scale in the test |
|---|---|---|
| gaussian | `y = f1 + N(0, 0.5²)` | estimated → `F(ref_df, n − edf)` |
| poisson | `exp(0.5 + 0.5 f1)` | known (φ = 1) → `χ²(ref_df)` |
| binomial | `logistic(f1)`, Bernoulli | known → `χ²` |
| gamma | shape 3, `μ = exp(1 + 0.5 f1)` | estimated shape → `F` |
| negative-binomial | θ = 2, `μ = exp(1 + 0.5 f1)` | θ estimated, φ = 1 → `χ²` |
| tweedie(p=1.5) | φ = 1, `μ = exp(0.5 + 0.5 f1)` | estimated φ → `F` |
| beta | φ = 10, `μ = logistic(0.5 f1)` | estimated φ → `F` |

Acceptance (from the lane): size ≤ α + 2·MCSE at α = .05 and .01 over
≥ 500 seeded replications at n ≥ 200; KS of `p/0.5 | p < 0.5` against U(0,1).

## Results

### Null size (true-null `s(x2)`)

`m` = fits that returned a p-value (failed fits, all outer-optimizer
certification errors, are in parentheses and never counted). Bound =
`α + 2·√(α(1−α)/m)`. Bold = over the bound, or KS rejected at 5%.

| family | n | m (failed) | size .10 | size .05 | size .01 | bound .05 / .01 | KS p<.5 |
|---|---|---|---|---|---|---|---|
| gaussian | 60 | 500 (0) | 0.062 | 0.026 | 0.004 | 0.069 / 0.019 | 0.420 |
| poisson | 60 | 500 (0) | 0.062 | 0.034 | 0.006 | 0.069 / 0.019 | 0.719 |
| binomial | 60 | 500 (0) | 0.062 | 0.034 | 0.000 | 0.069 / 0.019 | 0.158 |
| gamma | 60 | 500 (0) | 0.096 | 0.048 | 0.014 | 0.069 / 0.019 | **0.012** |
| negative-binomial | 60 | 500 (0) | 0.124 | 0.066 | **0.020** | 0.069 / 0.019 | **0.000** |
| tweedie | 60 | 440 (60) | 0.120 | **0.077** | 0.018 | 0.071 / 0.019 | **0.002** |
| beta | 60 | 493 (7) | 0.091 | 0.047 | 0.008 | 0.070 / 0.019 | **0.022** |
| gaussian | 200 | 500 (0) | 0.064 | 0.040 | 0.006 | 0.069 / 0.019 | 0.630 |
| poisson | 200 | 500 (0) | 0.072 | 0.034 | 0.002 | 0.069 / 0.019 | 0.528 |
| binomial | 200 | 500 (0) | 0.052 | 0.026 | 0.002 | 0.069 / 0.019 | 0.737 |
| gamma | 200 | 500 (0) | 0.090 | 0.038 | 0.012 | 0.069 / 0.019 | **0.018** |
| negative-binomial | 200 | 500 (0) | 0.076 | 0.028 | 0.010 | 0.069 / 0.019 | **0.006** |
| tweedie | 200 | 484 (16) | 0.074 | 0.043 | 0.012 | 0.070 / 0.019 | **0.002** |
| beta | 200 | 500 (0) | 0.056 | 0.030 | 0.010 | 0.069 / 0.019 | 0.739 |
| gaussian | 2000 | 300 (0) | 0.030 | 0.017 | 0.007 | 0.075 / 0.021 | 0.349 |
| poisson | 2000 | 300 (0) | 0.050 | 0.027 | 0.000 | 0.075 / 0.021 | 0.880 |
| binomial | 2000 | 300 (0) | 0.073 | 0.040 | 0.003 | 0.075 / 0.021 | 0.146 |
| gamma | 2000 | 300 (0) | 0.073 | 0.030 | 0.007 | 0.075 / 0.021 | 0.238 |
| negative-binomial | 2000 | 300 (0) | 0.053 | 0.033 | 0.003 | 0.075 / 0.021 | 0.607 |
| tweedie | 2000 | 297 (3) | 0.051 | 0.020 | 0.000 | 0.075 / 0.022 | 0.570 |
| beta | 2000 | 300 (0) | 0.053 | 0.030 | 0.007 | 0.075 / 0.021 | 0.271 |

At the acceptance sizes (n ≥ 200) every family is within `α + 2·MCSE` at
.10, .05 and .01. n = 2000 ran 300 replications per family (fits there cost
~10× the n = 200 ones); the bound is widened accordingly and nothing is near
it. KS on `p | p < .5` rejects for gamma, negative-binomial and tweedie at
n = 200 and for no family at n = 2000; see "Why KS rejects" below.

### Power (`s(x2)` carries `0.4·sin(2πx2)`, n = 200, 200 replications)

| family | m (failed) | power .10 | power .05 | power .01 |
|---|---|---|---|---|
| gaussian | 200 (0) | 1.000 | 1.000 | 1.000 |
| poisson | 200 (0) | 0.995 | 0.990 | 0.950 |
| binomial | 200 (0) | 0.330 | 0.240 | 0.100 |
| gamma | 200 (0) | 1.000 | 1.000 | 1.000 |
| negative-binomial | 200 (0) | 0.960 | 0.905 | 0.740 |
| tweedie | 185 (15) | 0.978 | 0.957 | 0.832 |
| beta | 200 (0) | 1.000 | 1.000 | 1.000 |

(Binomial is the low-information family: a ±0.4 swing on the logit scale
from 200 Bernoulli draws.)

### By fitted edf of the null term (n = 200, all families pooled)

| edf band | fits | min p | min ref_df | reject at .05 |
|---|---|---|---|---|
| [0, 0.01) | 1885 | 0.932 | 1.000 | 0.000 |
| [0.01, 0.1) | 42 | 0.741 | 1.000 | 0.000 |
| [0.1, 0.5) | 411 | 0.320 | 1.000 | 0.000 |
| [0.5, 1) | 683 | 0.00017 | 1.000 | 0.105 |
| [1, 2) | 386 | 0.0084 | 1.617 | 0.057 |
| [2, ∞) | 77 | 0.0016 | 2.937 | 0.325 |

REML switches the null term off in ~80% of fits (edf < 0.5, where p ≥ .32).
The rejections concentrate in the fits where noise happened to look like a
curve and REML kept it — a selection effect; the unconditional size above is
the operating characteristic that matters.

### Conditional `Vb` vs corrected `Vc` (n = 200, `covariance_ablation.py`)

| family | m | covariance | size .10 | size .05 | size .01 | P(p < .5) |
|---|---|---|---|---|---|---|
| gaussian | 500 | Vb (shipped) | 0.064 | 0.040 | 0.006 | 0.266 |
| | | Vc | 0.044 | 0.024 | 0.002 | 0.226 |
| poisson | 469 | Vb (shipped) | 0.062 | 0.030 | 0.002 | 0.249 |
| | | Vc | 0.038 | 0.021 | 0.002 | 0.222 |
| binomial | 444 | Vb (shipped) | 0.047 | 0.027 | 0.002 | 0.257 |
| | | Vc | 0.034 | 0.025 | 0.000 | 0.227 |
| gamma | 104 | Vb (shipped) | 0.096 | 0.038 | 0.010 | 0.337 |
| | | Vc | 0.106 | 0.019 | 0.010 | 0.298 |

(The run was stopped at 1632 of 2000 rows to free the machine; the rows
present are a seeded prefix, not a selection.)

## Answers to the lane questions

**χ² vs F, and which residual df.** Known-scale families (poisson, binomial,
negative-binomial) use `χ²(ref_df)`. Estimated-scale families (gaussian,
gamma, tweedie, beta) use `F(T/ref_df; ref_df, n − edf_total)` where
`edf_total` is the whole model's edf (`wald_residual_degrees_of_freedom`,
`result_types.rs`). That is mgcv's `summary.gam` convention (`residual.df`
= `n − sum(edf)`). mgcv refers `betar` to `χ²` (it treats φ as a family
parameter, `scale = 1`); gamfit refers it to `F`. With `n − edf ≈ 190` the two
differ by < 1% in p and the `F` choice is the (slightly) conservative one — the
tables above show beta sized well under nominal. Not changed.

**Estimated shape / θ / φ / power.** The Wald statistic is a quadratic form
in β̂. In a GLM the Fisher information is block-diagonal between β and the
dispersion φ (and between β and the NB θ): `E[∂²ℓ/∂β∂φ] = 0`. So plugging in
φ̂ or θ̂ changes `Var(β̂)` only at second order, and the first-order Wald
reference needs no extra term; what finite-sample slack exists for an
estimated scale is exactly what the `F` reference's denominator df absorbs.
The Tweedie variance power is never estimated (SPEC: the family string must
name it), so there is nothing to propagate. The per-family size rows above are
the empirical check: negative-binomial (θ̂ plugged in, `χ²`) and gamma/beta/
tweedie (φ̂ plugged in, `F`) are all within `α + 2·MCSE`.

**Conditional `Vb` vs smoothing-parameter-corrected `Vc`.** The codebase
tests with `Vb` (#2142/#2296). The evidence here supports keeping it:
`covariance_ablation.py` recomputes the identical whitened, rank-truncated
statistic under `Vc` and refers it to the identical reference. `Vc` inflates
the covariance along exactly the directions REML shrinks, which drives the
statistic down: sizes roughly halve (gaussian .040 → .024 at .05, poisson
.030 → .021, gamma .038 → .019) and the p < .5 mass drops, i.e. `Vc` makes an
already conservative boundary test more conservative and loses power, while
`Vb` is not oversized in any family. mgcv's `summary.gam` likewise hands
`testStat` the conditional Bayesian covariance `Vp`, not `Vc`: the Wood
(2013) p-values are defined conditional on λ̂.

**edf → 0.** `ref_df = max(tr(F)²/tr(F²), rank_used) ≥ 1` whenever a
p-value is produced, so it is defined for edf < 1 (no replicate at any n had
`ref_df < 1`). A term REML shrinks to edf < 0.5 is tested on its top
whitened direction against `χ²₁`/`F₁`, and the statistic vanishes with the
shrunk coefficients, so p ≥ .32 in every such replicate and p > .93 for
edf < 0.01 (see the edf-band table). A block with no estimable direction
returns `None` (surfaced as `p_value = None`), never a fabricated number.

## Why KS rejects for some families, and why that is not a defect

On a null term REML lands on (or next to) the zero-penalty-direction
boundary, and conditional on λ̂ the statistic is not `χ²`. The
one-direction case is exact (`boundary_toy.py`): `z ~ N(0,1)`, REML
shrinkage `τ = max(1 − 1/z², 0)`, `T = τz² = max(z² − 1, 0)` against
`χ²₁`:

| | α = .10 | .05 | .01 | P(p < .5) |
|---|---|---|---|---|
| exact size | .0542 | .0278 | .0057 | .2277 |

It is conservative at every level, but *below* 0.5 the p-value is
front-loaded — observed/expected per bin over
`[0, .01, .05, .1, .2, .3, .4, .5]` is 1.26, 1.21, 1.16, 1.09, 1.00, 0.91,
0.80 — so a KS test on `p | p < .5` has real power against it even though
the test is well sized: 24% rejection with 100 sub-.5 p-values, 35% with 150,
51% with 250. The null runs have ~110–170 sub-.5 p-values per family. mgcv's
`testStat` refers a rank-< 1 term to the same top direction and `χ²₁` (its
`k = 0` branch), so it shares the shape.

The toy does not explain everything: at n = 200 the gamma, negative-binomial
and tweedie histograms are more front-loaded than the toy (first-bin
obs/exp 1.8–2.2 vs 1.26, on ~6 counts against ~5 expected), and three
rejections of seven is more than the toy's ~35% per family. Those are exactly
the families whose nuisance parameter (shape, θ, φ) is estimated by a
plug-in that is biased at small n (next section), and the extra
front-loading is gone at n = 2000, where no family rejects. It is a
finite-sample effect that sits below the size bound at n ≥ 200.

Making `p | p < .5` uniform would need a λ̂-unconditional reference
(a mixture over the boundary), which is a different test than Wood (2013) —
the lane rules exclude an alternative p-value variant, and the one principled
covariance change available (`Vc`) moves the wrong way (above). The lane's KS
criterion is therefore reported, not met, for the families listed; the size
criterion, the one that decides whether the test can be trusted, is met
everywhere.

## n = 60: two families oversized, and why

Below the acceptance sample size two families exceed the bound:
negative-binomial at .01 (0.020 vs 0.019) and tweedie at .05 (0.077 vs
0.071, with 60 of 500 fits failing). Neither is in the Wald test itself;
both are the plug-in nuisance estimate feeding `Vb`:

- **Tweedie φ̂ divides by n, not n − edf.**
  `estimate_tweedie_phi_from_eta` (`crates/gam-solve/src/pirls/dispersion.rs`)
  returns `Σ wᵢ(yᵢ − μᵢ)²/μᵢᵖ / Σ wᵢ`, and it is frozen at the first
  converged λ-search solve. The `F(ref_df, n − edf)` reference presumes the
  Pearson estimator on `n − edf` degrees of freedom (mgcv's, and the one
  `run_sample_generate_report.rs` already uses for Tweedie φ in residual
  reports). With ~6 total edf at n = 60, `φ̂` is ~10% low, `Vb` ~10% small,
  `T` ~10% large. Rescaling the recorded statistics by `(n − 6)/n` moves
  tweedie at n = 60 from .120/.077/.018 to .102/.059/.011 — inside the bound —
  and changes n = 200 by ≤ .004. The same divisor question applies to the
  Beta precision moment estimate.
- **Negative-binomial θ̂ is the ML estimate**, which at n = 60 understates
  overdispersion; the test refers to `χ²` (as mgcv's `nb` does), so there is
  no denominator df to absorb it.

The divisor lives in the dispersion estimator that REML, every standard error
and every interval share, not in `wood_smooth_test`, and changing it moves the
fit, so it is reported here as a follow-up rather than patched from this
lane. At n ≥ 200 the effect is inside Monte-Carlo error for every family.

**What the trace-ratio `ref_df` floor buys.** `analyze.py --ablation`
recomputes each p-value against `max(1, round(edf))` (the rank summed,
without `tr(F)²/tr(F²)`): that is oversized in five families at .05 (see the
`ref_df=rank` rows), so the Wood reference df is doing real work. At 500
replications the bench detects its removal; the 200-replication CI gate does
not (below).

## Regression test

`tests/inference/misc/sbc_wood_smooth_test_family_size_curve.rs`: one test
per family, n = 200, 200 seeded replications (rayon-parallel; the 500-rep
run above is the bench, and 500 reps × 7 families overruns the per-test CI
kill), gate
`size ≤ α + 2·√(α(1−α)/m)` at α ∈ {.10, .05, .01}; also `ref_df ≥ 1` and
finite for every replicate and `p > .5` for edf < 0.01. It uses the Rust
simulator (same DGPs, different RNG stream from the Python bench).

What it catches, checked by mutating the `ref_df` line of `wood_smooth_test`
locally and running the gate (the mutations were never committed):

| mutation of `ref_df` | gate result |
|---|---|
| `edf` (the #1360 reference, pre-floor) | **fails**: gaussian and gamma, e.g. `Gamma rep 0: ref_df 4.9e-7 undefined or below one at edf 4.9e-7` |
| `rank_used` (drop the trace ratio) | passes in all 7. Sizes rise (poisson .044→.075 and binomial .032→.080 at .05; poisson .064→.140 at .10), but every one stays under the 200-rep bound (.0808 / .1424) |
| raw trace ratio, without the `rank_used` floor | passes, and should: for PSD `F`, `tr(F)²/tr(F²) ≥ 1`, so the floor binds only when round(edf) ≥ 2 exceeds the ratio, which these null terms don't reach |

So the gate protects the boundary behaviour (`ref_df` defined and ≥ 1 as
edf → 0, and no small p for a switched-off term) and any gross size
failure. At 200 replications its MCSE is too coarse to detect a size of
~.075 at nominal .05; that sensitivity is the job of the 500-replication
bench, not of CI.

## Out of lane, noted

- Tweedie fits fail the outer-optimizer certification in 60/500 (n = 60),
  16/500 (n = 200) and 3/300 (n = 2000) null replications, and beta in 7/500
  at n = 60 ("did not certify a stationary optimum … Newton-decrement above
  tolerance after polish" / "declined a certified optimum that an evaluated
  state beats"). Those are errors, not p-values; sizes are over the fits that
  converged. Solver issue, not the Wald test.
- Tweedie φ̂ / Beta φ̂ divisor (n vs n − edf), above.

## Reproduce

```
cd bench/pvalue_calibration/pv-wald-families
F=gaussian,poisson,binomial,gamma,negative-binomial,tweedie,beta
NPROC=4 python null_calibration.py $F 200 500 0.0 results/null200.jsonl
NPROC=4 python null_calibration.py $F 60 500 0.0 results/null60.jsonl
NPROC=4 python null_calibration.py $F 2000 100 0.0 results/null2000.jsonl
NPROC=4 python null_calibration.py $F 200 200 0.4 results/power200.jsonl
python analyze.py --ablation results/null200.jsonl
python analyze.py results/null60.jsonl results/null2000.jsonl results/power200.jsonl
NPROC=4 python covariance_ablation.py run gaussian,poisson,binomial,gamma 200 500 results/vc200.jsonl
python covariance_ablation.py analyze results/vc200.jsonl
python boundary_toy.py
./build.sh test --test inference sbc_wood_smooth_test_family_size_curve -- --nocapture
```
