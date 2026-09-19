# Random-effect and double-penalty smooth p-values on the variance-component boundary

Lane: `bench/pygam_audit/lanes/pv-random-effects.md`.

Two summary rows test a null that sits on the boundary of the parameter
space:

* a `group(g)` / `re(g)` block, a ridge-penalized one-hot block whose variance
  `σ²_b` is zero under "no group effect";
* a default `s(x)`, which is double-penalized (a wiggliness penalty plus a
  ridge on its polynomial null space), so "no effect" is every variance
  component of the term at zero.

Neither a Wald statistic nor a likelihood ratio referred to `χ²` is calibrated
there. The random-effect row used to carry no p-value at all. The smooth row
carried the Wood (2013) rank-truncated Wald test. That test is computed from
coefficients REML shrank on the same data, so about half of all null fits
shrank the term to `edf ≈ 0` and reported `p ≈ 1`. The result was a point mass
that made the test conservative at every level (the "before" table below).

Both rows now carry the variance-component score test of Lin (1997), scored
against its exact finite-sample null law (Wood 2013, *Biometrika* 100, "A
simple test for random effects in regression models"). The code is in
`crates/gam-terms/src/inference/variance_component_test.rs`, and its module
docs carry the derivation.

## The test

The block's design is `X_R`. `X̃_R` is `X_R` with every other column of the
fit projected out in the fit's own curvature metric. With `v` the working
residual of the model without the block:

```
u = X̃_Rᵀv,    V = X̃_RᵀW X̃_R,    u ~ N(0, φV) under H₀
T = uᵀΣ₀u,    T/φ ~ Σ_i μ_i χ²₁,   μ = eig(Σ₀^{1/2} V Σ₀^{1/2})
```

When the scale is estimated, `T` is ratioed against the unpenalized residual,
and the p-value is the signed weighted-chi-square tail
`P(Σ μ_i χ²₁ − t·χ²_ν > 0)`. Both tails are evaluated by exact inversion. There
is no moment matching, no halving and no chi-bar-square mixture weight to
estimate. `T > 0` almost surely, so there is no atom at `p = 1`.

The direction `Σ₀` is fixed by the design, the weights and the penalty
structure, never by `y`, `b̂` or `λ̂`. That is what makes the reference law
exact:

```
Σ₀ = Σ_j K_j / tr(K_j V),    K_j = P⁻¹(a_j S_j)P⁻¹,
P = Σ_j a_j S_j,             a_j = 1/tr(G_u⁻ S_j),   G_u = X_RᵀW X_R
```

`K_j` is penalty `j`'s share of the prior covariance (`Σ_j K_j = P⁻¹`).
Normalizing by `tr(K_j V)` gives every component unit null mean. As a result,
the rank-2 linear null space of a smoothing penalty is not swamped by its
rank-(k−2) wiggly range, and a purely linear effect and a purely wiggly one are
both seen. A `group()` ridge gives `Σ₀ ∝ I` and the Lin statistic `‖u‖²`.

`K_j` is covariant under a reparametrization `X_R → X_R Z`, `S_j → ZᵀS_jZ`, so
the p-value does not depend on the chart the basis factory realized. The first
version of this change used the pseudo-inverses `S_j⁺` instead, and that
mattered in practice. In the realized chart of a default smooth
(identifiability-constrained, centred and reparametrized), the double penalty's
two ranges are complementary but not Euclidean-orthogonal. For one binomial
replicate, the linear direction `c` of `s(x2)` had `cᵀS_wiggle c ≈ 0` but
`cᵀS_ridge c / ‖c‖² = 0.35`. The `S_j⁺` direction then scored a chart-dependent
mixture of the components, and on that replicate it read a linear score of
`z = −2.98` as `p = .33`. Null calibration was fine, but power collapsed (the
"S⁺ direction" row below). With `K_j` the same replicate gives `p = .0018`.

A block with a direction no penalty touches is refused with the typed reason
`variance_component_test_unpenalized_directions`, and the smooth keeps its
Wald row. A block with no penalty at all (a factor `by=` main effect) gets the
classical fixed-effect score `F` under its own hypothesis label.

## Design

`null_calibration.py` holds the seeded data-generating processes.

**Random-effect cells:** `y ~ s(x1) + group(g)`, with a real `sin(2πx1)`.
Under the null there is no group effect; under the alternative `b_g ~ N(0, .25²)`.

* `L ∈ {5, 20, 200}` levels, balanced (round-robin) or unbalanced
  (Dirichlet(½) level shares, every level seen).
* `n = max(400, 5L)`.
* 500 null reps and 100 power reps per cell.
* Seed `11_000_000 + 10⁶·family + 10⁴·L + 5000·balanced + rep`.

**Double-penalty cells:** `y ~ s(x1) + s(x2)`, where `s(x2)` is null or
`EFFECT·sin(2πc·x2)`.

* `n = 400`.
* 500 null reps, and 200 power reps at `c = 1, EFFECT = .4` (`dp`) and at
  `c = 2, EFFECT = .5` (`dp2`).
* Seed `13_000_000 + 10⁶·family + 10·n + rep`.

The p-value read is `gamfit.fit(...).summary().smooth_terms`, which is the
production row the CLI and Rust summaries share. MCSE is `√(α(1−α)/m)`. A null
cell is flagged ANTI or CONS when its size falls outside `α ± 2·MCSE` at any
`α ∈ {.10, .05, .01}`, and KS when the two-sided KS test against `U(0,1)`
rejects at .01. Conservative is a failure exactly like anti-conservative.

## Results: random effects

`results/re_null.jsonl` holds reps 0–499 of every cell, 500 p-values each.
The one exception is binomial `L = 5` balanced, where one fit failed outer
certification. It is reported as a failed fit, not dropped, and 499 p-values
remain.

| family | L | balance | size .10 | size .05 | size .01 | KS p |
|---|---:|---|---:|---:|---:|---:|
| gaussian | 5 | unbalanced | .088 | .050 | .006 | .240 |
| gaussian | 5 | balanced | .092 | .054 | .008 | .431 |
| gaussian | 20 | unbalanced | .084 | .042 | .006 | .662 |
| gaussian | 20 | balanced | .110 | .048 | .014 | .566 |
| gaussian | 200 | unbalanced | .096 | .044 | .006 | .727 |
| gaussian | 200 | balanced | .102 | .052 | .016 | .690 |
| binomial | 5 | unbalanced | .106 | .058 | .010 | .566 |
| binomial | 5 | balanced | .124 | .062 | .006 | .686 |
| binomial | 20 | unbalanced | .114 | .050 | .006 | .460 |
| binomial | 20 | balanced | .084 | .046 | .008 | .232 |
| binomial | 200 | unbalanced | .078 | .038 | .008 | .104 |
| binomial | 200 | balanced | .078 | .032 | .006 | .026 |
| poisson | 5 | unbalanced | .058 | .026 | .006 | .037 |
| poisson | 5 | balanced | .100 | .042 | .008 | .832 |
| poisson | 20 | unbalanced | .124 | .072 | .008 | .424 |
| poisson | 20 | balanced | .096 | .056 | .008 | .961 |
| poisson | 200 | unbalanced | .106 | .048 | .014 | .778 |
| poisson | 200 | balanced | .114 | .048 | .010 | .755 |

At 500 reps the 2-MCSE bands are `.10 ± .027`, `.05 ± .019` and `.01 ± .009`.
Two of the 108 (cell, α) checks fall outside them:

* poisson `L = 5` unbalanced reads `.026` at .05 (below).
* poisson `L = 20` unbalanced reads `.072` at .05 (above).

No KS test rejects at .01. With 36 cells × 3 levels × 2 sides, a couple of
excursions are expected by chance. To tell chance from a real miscalibration,
the four Poisson `L ∈ {5, 20}` cells were rerun on 1000 fresh seeds (reps
500–1499, `results/re_null_fresh_seeds.jsonl`):

| family | L | balance | size .10 | size .05 | size .01 | KS p |
|---|---:|---|---:|---:|---:|---:|
| poisson | 5 | unbalanced | .113 | .053 | .014 | .839 |
| poisson | 5 | balanced | .104 | .049 | .012 | .473 |
| poisson | 20 | unbalanced | .108 | .052 | .008 | .397 |
| poisson | 20 | balanced | .091 | .050 | .009 | .259 |

All four are inside the 1000-rep bands (`.10 ± .019`, `.05 ± .014`,
`.01 ± .006`), so the two flags were chance.

Power comes from `results/re_power.jsonl`, with `b_g ~ N(0, .25²)` and 100
reps per cell. The table gives the rejection rate at .05, where the null band
is `.05 ± .044`.

| family | L=5 U | L=5 B | L=20 U | L=20 B | L=200 U | L=200 B |
|---|---:|---:|---:|---:|---:|---:|
| gaussian | .87 | .98 | .96 | 1.00 | 1.00 | 1.00 |
| poisson | .46 | .47 | .47 | .45 | .43 | .30 |
| binomial | .32 | .33 | .28 | .13 | .13 | .05 |

Binomial `L = 200` has 5 Bernoulli rows per level at `σ_b = .25`, which carries
almost no information about `σ_b`, so low power there is expected. Before this
change the random-effect row had no p-value at all, so there is no "before"
power to compare against.

## Results: double-penalty smooth

`s(x2)` in `y ~ s(x1) + s(x2)`, `n = 400`. There are 500 null reps per family
in `results/dp_null_*.jsonl` and 200 power reps per family and alternative in
`results/dp_power_*.jsonl`. No fit failed in any of the nine runs. The 2-MCSE
null bands are `.10 ± .027`, `.05 ± .019` and `.01 ± .009`.

**Null.** "Before" is `main`'s Wood Wald row (`dp_null_before.jsonl`). "`S⁺`"
is the first version of this change, whose direction used the penalty
pseudo-inverses (`dp_null_s_plus.jsonl`). "After" is this change, with the
chart-covariant `K_j` direction (`dp_null_after.jsonl`).

| family | version | size .10 | size .05 | size .01 | share of p > .99 | KS p |
|---|---|---:|---:|---:|---:|---:|
| gaussian | before | .074 | .024 | .006 | .460 | < .001 |
| gaussian | `S⁺` | .100 | .050 | .014 | .008 | .534 |
| gaussian | after | .110 | .034 | .004 | .010 | .823 |
| binomial | before | .064 | .032 | .012 | .484 | < .001 |
| binomial | `S⁺` | .100 | .046 | .006 | .012 | .897 |
| binomial | after | .094 | .046 | .010 | .018 | .604 |
| poisson | before | .080 | .032 | .010 | .498 | < .001 |
| poisson | `S⁺` | .084 | .044 | .010 | .008 | .242 |
| poisson | after | .092 | .046 | .008 | .016 | .671 |

`main`'s row put about half of all null p-values within .01 of 1 and was
conservative at .10 and .05 in every family. After this change, every cell is
inside its band at every level and no KS test rejects.

**Power.** The table gives rejection rates at .10 / .05 / .01, 200 reps each.
`dp` is `0.4·sin(2πx2)` and `dp2` is `0.5·sin(4πx2)`.

| family | alternative | before | `S⁺` | after |
|---|---|---|---|---|
| gaussian | `dp` | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 |
| binomial | `dp` | .625 / .525 / .255 | .130 / .080 / .010 | .645 / .530 / .325 |
| poisson | `dp` | .820 / .730 / .465 | .240 / .120 / .015 | .835 / .725 / .540 |
| gaussian | `dp2` | 1 / 1 / 1 | .920 / .805 / .480 | 1 / 1 / 1 |
| binomial | `dp2` | .550 / .420 / .170 | .130 / .075 / .015 | .280 / .165 / .040 |
| poisson | `dp2` | .840 / .780 / .575 | .175 / .105 / .025 | .450 / .305 / .145 |

On the one-period alternative, the calibrated test matches or beats `main`'s
Wald row at every level, although the Wald row had a point mass at 1.

**Power loss on the two-period alternative.** On `dp2` in binomial and
Poisson, the new test has less than half of `main`'s power. This is a real
cost of this change, not a Monte Carlo artefact. The reason is the direction
itself:

* `Σ₀` weights each basis direction by its prior variance under the
  penalty. The wiggliness penalty's prior variance falls off steeply with
  frequency, so the statistic is locally most powerful against smooth,
  low-frequency departures.
* `sin(4πx2)` puts most of its mass on higher-frequency directions, which the
  fixed direction downweights.
* The Wald row adapts to the data: it scores the coefficients at the REML
  `λ̂` chosen on that same data. That adaptivity is also why it is not
  calibrated. Its null law depends on `λ̂`, and it has the atom at `p = 1`
  shown above.

A test that keeps the exact reference law must fix its direction before seeing
`y`. Any fixed direction trades power between frequencies, and the one used
here is the prior's own. Recovering `main`'s `dp2` power without giving up
calibration would need a different statistic, for example a combination over
several fixed directions with an exact joint reference law. That is not part
of this change.

## Gaussian location-scale mean smooths

A fit with a `noise_formula` goes through its own route. The test is scored
from that route's own row state:

* The Fisher weight is `wᵢ/σ̂ᵢ²` and the score is `wᵢ(yᵢ − μ̂ᵢ)/σ̂ᵢ²`.
* The mean/log-σ block of the Fisher information is zero, so the log-σ
  coefficients need no projection.
* The level of `σ̂` is treated as estimated, and the score is referred to the
  residual `χ²_ν` of the standardized model. `σ̂` is fitted to the residuals
  the score is built from, so taking the level as known reads the score too
  large.

The Rust gate is `gaussian_location_scale_null_mean_smooth_size_is_within_monte_carlo_error`.
It fits `y ~ s(x1) + s(x2)` with noise `~ s(x1)`, `n = 200`, and a noise SD
`0.3·e^{x1}` that nearly triples across `x1`. It draws 1000 seeded null
replicates:

| version | reps | size .10 | size .05 | size .01 | KS p |
|---|---:|---:|---:|---:|---:|
| scale taken as known (`φ = 1`) | 200 | .110 | .075 | .035 | .767 |
| scale estimated | 200 | .105 | .065 | .030 | .945 |
| scale estimated, fresh seeds 200–1199 | 1000 | .095 | .050 | .012 | .852 |
| scale estimated, the gate (seeds 0–999) | 1000 | .093 | .050 | .015 | .926 |

* The first two rows share their seeds. Taking the scale as known was the
  first version of this route. It failed at .01, and treating the scale as
  estimated moved all three levels towards nominal.
* At 200 reps the .01 band is `.01 ± .014`, and the estimated-scale version
  still sat outside it (6 of 200). A 1000-rep run on fresh seeds read `.012`,
  inside its band.
* The gate therefore runs 1000 reps (band `.01 ± .0063`). It passes at every
  level, and the size at .01 is `.015`, near the top of the band. That is 1.6
  Monte Carlo standard errors above `.01`, which is not significant.
  It is the one cell in this lane that reads high at .01 on both of its
  seed sets, so it is disclosed here rather than rounded away.
* Power: `0.5·sin(2πx2)` is found in 40 of 40 fits at .01.

The route is skipped (the field is `None`, and the smooths keep the Wald row)
when the fit carries a link wiggle, whose extra mean-side coefficients the
test does not model.

## Rust CI gates

The seeded Rust tests are smaller than this bench, so they run in CI. Each
one fails when size falls outside `α ± 2·MCSE` on **either** side at .10, .05
or .01, or when a two-sided KS test against `U(0,1)` rejects at .01.

* `tests/inference/misc/sbc_double_penalty_smooth_family_size_curve.rs`
  covers the double-penalty null in 7 families (`n = 200`, 200 reps), the
  Gaussian location-scale mean smooth, and power controls.
* `tests/inference/misc/sbc_random_effect_variance_component_size_curve.rs`
  covers the random-effect null in 3 families, plus power at `σ_b = 1`.
* The unit tests in `crates/gam-terms/src/inference/variance_component_test.rs`
  cover exact identities (the one-way ANOVA `F`), null uniformity with a known
  and an estimated scale, invariance to penalty rescaling and to block
  reparametrization, and the typed refusals.

**Failed fits** are the fits the outer optimizer refuses to certify. SPEC.md:
"A fit object must only ever come from a converged optimization." They are
counted and reported, never scored as p-values, and a gate fails outright when
more than 5% of its fits fail.

## Routes that keep the Wald row

`FitArtifacts::variance_component_tests` is `Some(records)` on the routes that
run the test. These are the standard single-block fit and the Gaussian
location-scale mean block (without a link wiggle). On those routes, a smooth
whose penalties cover every direction, or a `group()` block, reads its
p-value from the record.

On every other route the field is `None`. These are the binomial and
dispersion location-scale fits, the survival location-scale fit, fits with a
link wiggle, and payloads saved before this change. On those routes a smooth
keeps `main`'s Wood Wald row, and a `group()` block reports
`variance_component_test_not_recorded`. Those routes still carry the
conservative Wald behaviour measured above. Extending the test to them needs
each route's own score and curvature for the tested block. That is open
follow-up work.

## Reproduce

```sh
source .venv/bin/activate && ./build.sh maturin
cd bench/pvalue_calibration/pv-random-effects
export NPROC=4
for L in 5 20 200; do
  python null_calibration.py re gaussian,binomial,poisson $L 500 0.0 results/re_null.jsonl
  python null_calibration.py re gaussian,binomial,poisson $L 100 0.25 results/re_power.jsonl
done
for L in 5 20; do
  python null_calibration.py re poisson $L 500:1500 0.0 results/re_null_fresh_seeds.jsonl
done
python null_calibration.py dp gaussian,binomial,poisson 400 500 0.0 results/dp_null_after.jsonl
python null_calibration.py dp gaussian,binomial,poisson 400 200 0.4 results/dp_power_after.jsonl
python null_calibration.py dp2 gaussian,binomial,poisson 400 200 0.5 results/dp_power_after.jsonl
python analyze.py results/re_null.jsonl results/re_power.jsonl
python analyze.py results/dp_null_after.jsonl results/dp_power_after.jsonl
```

The `before` and `S⁺` files came from the same commands run on `main` and on
the first version of this change.

Rust gates:

```sh
./build.sh test --test inference -- sbc_double_penalty_smooth sbc_random_effect gaussian_location_scale_ --nocapture
./build.sh test -p gam-terms --lib variance_component_test
```
