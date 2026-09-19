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

__RESULTS__

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
51% with 250. The null runs have ~110–170 sub-.5 p-values per family, so KS
rejecting in two or three of seven families is what this intrinsic shape
predicts. mgcv's `testStat` refers a rank-< 1 term to the same top direction
and `χ²₁` (its `k = 0` branch), so it shares the shape.

Making `p | p < .5` uniform would need a λ̂-unconditional reference
(a mixture over the boundary), which is a different test than Wood (2013) —
the lane rules exclude an alternative p-value variant, and the one principled
covariance change available (`Vc`) moves the wrong way (above). The lane's KS
criterion is therefore reported, not met, for the families listed; the size
criterion, the one that decides whether the test can be trusted, is met
everywhere.

**What the trace-ratio `ref_df` floor buys.** `analyze.py --ablation`
recomputes each p-value against `max(1, round(edf))` (the rank summed,
without `tr(F)²/tr(F²)`): that is oversized in five families at .05 (see the
`ref_df=rank` rows), so the Wood reference df is doing real work and the new
regression test fails when it is removed (below).

## Regression test

`tests/inference/misc/sbc_wood_smooth_test_family_size_curve.rs`: one test
per family, n = 200, 500 seeded replications (rayon-parallel), gate
`size ≤ α + 2·√(α(1−α)/m)` at α ∈ {.10, .05, .01}; also `ref_df ≥ 1` and
finite for every replicate and `p > .5` for edf < 0.01. It uses the Rust
simulator (same DGPs, different RNG stream from the Python bench).

__MUTATION__

## Out of lane, noted

- Tweedie at n = 200: 16 of 500 fits fail with "Outer smoothing-parameter
  optimization did not certify a stationary optimum … Newton-decrement above
  tolerance after polish". Those are errors, not p-values; the size is
  computed over the 484 that converged. Solver issue, not the Wald test.

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
