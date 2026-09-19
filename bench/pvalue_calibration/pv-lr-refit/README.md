# pv-lr-refit: calibration of `Model.smooth_significance`

A seeded Monte Carlo study of the per-term likelihood-ratio p-value over the
pyGAM audit's three inference cells (`bench/pygam_audit`). Each cell fits
`y ~ s(x1) + s(x2) + s(x3)`, where `s(x1)` is a strong effect, `s(x3)` a weak
one and `s(x2)` has no effect. Replicate `r` is seeded `default_rng(1000 + r)`,
so it is the same dataset as the audit's replicate `r`.

| cell    | family   | n   | eta                                  |
|---------|----------|-----|--------------------------------------|
| `gauss` | gaussian | 200 | `sin(2πx1) + 0.30 cos(2πx3)`, σ = 1  |
| `binom` | binomial | 400 | `1.5 sin(2πx1) + 0.60 cos(2πx3)`     |
| `pois`  | poisson  | 200 | `0.5 + 0.8 sin(2πx1) + 0.25 cos(2πx3)` |

## Running

```sh
python run.py <cell> <first_rep> <last_rep_exclusive> <out.jsonl> [<stall_seconds>]
python run.py --summarize <out.jsonl> [<out.jsonl> ...]
```

Shards append one JSON line per replicate, so a cell can be split over several
processes and summarized together. With `stall_seconds`, each replicate runs in
a worker process. A replicate that has not returned by then is recorded as
`stalled` and the worker is replaced. That lets the study finish; it is not a
result. The summary lists stalled replicates separately and never scores them
(see handoff (c)). The results below used `stall_seconds = 300`.

## What is measured

* **Availability.** Every row of a converged fit must carry exactly one of
  `p_value`, `p_value_upper_bound` or `unavailable_reason`. The summary counts
  each kind and any malformed rows.
* **Size** of the null term `s(x2)` at α = .10/.05/.01, with MCSE
  `sqrt(α(1−α)/R)`: .0134 / .0097 / .0044 at R = 500.
* **Uniformity** of its p-values over the whole of (0, 1). A conservative
  p-value is as wrong as an anti-conservative one, so the verdict is the
  two-sided one-sample KS test against U(0, 1). The summary also prints the
  one-sided `D+` (the empirical CDF above the uniform's somewhere, i.e.
  anti-conservative at some level) to say which way a failure leans, and the
  share with p = 1. The decile counts and their χ²₉ test are reported
  alongside.
* **Power** of `s(x1)` and `s(x3)`. A `p < p_value_upper_bound` row rejects at
  every α above the bound.

## Results (500 replicates per cell)

Every cell: 500 of 500 fits converged, no fit errors, no stalls, and every
row carries a p-value or a bound (no `unavailable_reason`, no malformed rows).

Null term `s(x2)`, MCSE .0134 / .0097 / .0044:

| cell    | size @.10 | size @.05 | size @.01 | KS D (p)       | D+ (p)       | P(p = 1) | deciles (expected 50 each)             | χ²₉ (p)    |
|---------|-----------|-----------|-----------|----------------|--------------|----------|----------------------------------------|------------|
| `gauss` | .1060     | .0440     | .0140     | .0259 (.883)   | .0259 (.504) | 0        | 53 48 51 55 50 40 46 49 57 51          | 4.1 (.903) |
| `binom` | .0940     | .0520     | .0140     | .0353 (.550)   | .0169 (.743) | 0        | 47 57 49 49 41 45 54 48 49 61          | 6.2 (.724) |
| `pois`  | .0940     | .0440     | .0120     | .0500 (.161)   | .0081 (.931) | 0        | 47 36 54 67 41 47 47 53 50 58          | 13.6 (.136)|

**The null p-values are calibrated in all three cells**: every size is within
one MCSE of nominal, on both sides, at all three levels, and neither the
two-sided KS test nor the decile χ² rejects uniformity at 5%. There is no
mass at p = 1 and no pile-up anywhere in (0, 1).

Power:

| cell    | term    | @.10  | @.05  | @.01  |
|---------|---------|-------|-------|-------|
| `gauss` | `s(x1)` | 1     | 1     | 1     |
| `gauss` | `s(x3)` | .8340 | .6980 | .4180 |
| `binom` | `s(x1)` | 1     | 1     | 1     |
| `binom` | `s(x3)` | .9220 | .8720 | .6960 |
| `pois`  | `s(x1)` | 1     | 1     | 1     |
| `pois`  | `s(x3)` | .9080 | .8340 | .5420 |

Rows published as a bound: 27 / 70 / 56 of 1500 (Gaussian / binomial /
Poisson). Almost all are the strong term, whose tail is far below the
selection replay's resolution; three are null-term rows (below).

### How it got there

Null term `s(x2)`, 500 replicates per cell at each stage. Before the lane is
the branch point; each later stage adds one fix (the sections below).

| stage                       | cell    | R   | size @.10/.05/.01   | KS D (p)     | D+ (p)       | P(p = 1) |
|-----------------------------|---------|-----|---------------------|--------------|--------------|----------|
| before the lane             | `gauss` | 500 | .108 / .048 / .016  | .480 (0)     | .480 (0)     | 0        |
|                             | `binom` | 488 | .125 / .072 / .035  | .239 (0)     | .183 (0)     | .218     |
|                             | `pois`  | 500 | .084 / .044 / .008  | .256 (0)     | .124 (0)     | .249     |
| + estimated-scale support   | `gauss` | 500 | .108 / .048 / .016  | .440 (0)     | .440 (0)     | .012     |
| + nested null at `ρ̂`        | `gauss` | 500 | .106 / .046 / .014  | .441 (0)     | .441 (0)     | 0        |
|                             | `binom` | 479 | .100 / .048 / .015  | .215 (0)     | .022 (.619)  | .013     |
|                             | `pois`  | 500 | .088 / .044 / .010  | .281 (0)     | .003 (.992)  | 0        |
| + rescored selection tail   | `gauss` | 500 | .106 / .044 / .014  | .026 (.883)  | .026 (.504)  | 0        |
|                             | `binom` | 365 | .085 / .044 / .014  | .225 (0)     | .009 (.938)  | .003     |
|                             | `pois`  | 439 | .084 / .039 / .011  | .339 (0)     | .004 (.984)  | .002     |
| + no double Bartlett shift  | all     | 500 | table above         |              |              |          |

The two partial rows (binomial 365, Poisson 439) were stopped once the p ≈ 1
mass they showed had been traced to the Bartlett double count; the remaining
replicates were run on the fixed build instead. The Gaussian cell rerun on
the fixed build is identical to its rescored row, replicate for replicate
(`c = 1` there, so the Bartlett fix cannot move it). The binomial and Poisson
reference laws have known scale, so the estimated-scale fix does not touch
them.

## Findings

### B3 (missing LR p-values): no `None` on main; the wrong values were the defect

I scanned 43 of the audit's seeded replicates on main before this lane:
binomial 0–2, Gaussian 0–9 and Poisson 0–29. Every one returned an LR row with
a p-value for every smooth term; none returned `None`. So there was no missing
value to restore. What this lane changes is the contract, so that the
behaviour cannot silently regress: every tested smooth term yields either a
typed inference row or a typed `SmoothLrUnavailable` reason, never a skip:

`full_refit_failed`, `null_fit_not_converged`, `null_fit_unsupported`,
`null_log_likelihood_not_finite`, `empty_coefficient_block`,
`degenerate_reference`, `tail_not_computable`.

A shape-constrained term has no LR reference at all. It is not tested and
goes through main's route instead: a row with `p_value_unavailable` and an
`explanation`, from `smooth_term_lr_unavailable_forspec`.

A reduced model that does not converge is `null_fit_not_converged`, not a NaN.
The saved-model `summary()` is Wald-only and has no data to refit with, and
the CLI has no LR surface. The reason is surfaced where the LR test lives: in
the `smooth_significance` row, as `unavailable_reason` plus
`unavailable_message`.

What the full study did find is worse than a missing value: **wrong tiny
p-values for the null term.** Before this lane, 11 of 488 binomial replicates
published `s(x2)` (no effect) at p < 1e-3 from a statistic `W ≤ ~1e-4`,
several astronomically:

* replicate 14: `p < 1.1e-237` from `W = 7.4e-6`;
* replicate 204: `p < 9.0e-54` from `W = 8.9e-5`.

That inflated the binomial size at α = .01 to .035 (8 MCSE above nominal).

**Root cause.** The reduced model was refitted from scratch: REML re-ran for
every surviving smoothing parameter. When the tested term sits at its penalty
null space the full and reduced REML optima differ only by the outer search's
convergence tolerance, and that tolerance became the statistic. `W ~ 1e-6 …
1e-4` was scored against a null law concentrated at `~1e-8`.

**Fix.** The nested null is now the full problem with the tested coefficient
block fixed at zero, solved by the same P-IRLS at the full fit's `ρ̂` with no
outer search (`gam_solve::estimate::fit_nested_at_fitted_log_lambdas`). The
only difference between the two optima is the constraint. Profiled nuisance
parameters are handled the way the full fit handles them: Gamma shape and
Beta precision are re-profiled on the reduced model; a jointly estimated link
shape has no nested reduced model at fixed `ρ̂` and is the typed reason
`null_fit_unsupported`, as is a penalty straddling the tested block, a
constraint that needs the block, and the bounded-linear route.

Before/after, null term `s(x2)`:

| binomial, `s(x2)`       | R   | p < 1e-3 | smallest p    | size @.01 |
|-------------------------|-----|----------|---------------|-----------|
| from-scratch null refit | 488 | 12       | 1.1e-237      | .035      |
| nested null at `ρ̂`      | 479 | 1        | 1.4e-4        | .015      |
| final                   | 500 | 1        | 1.1e-4        | .014      |

Under the null, 500 replicates expect 0.5 below 1e-3.

The regression tests are a splitmix64-seeded Rust fixture driven through the
real `smooth_term_lr_inference_forspec`
(`fit_orchestration/smooth_lr_nested_null_tests.rs`, seed 20 used to publish
`p < 2.3e-28` from `W = 2.9e-4`) and the audit's binomial replicates 14 and 204
through the Python API (`tests/smooth_significance_nested_null_test.py`).

### L4 (tail floor): reproduced; the floor itself is gone on main

On the branch point the tail came from Imhof inversion, which is accurate only
in absolute terms (`p_value_bound` ≈ 1e-13). A strong term's tail lies below
that accuracy, and the row published the inversion's rounding residue as the
p-value:

* Poisson replicate 5, `s(x1)`, W = 90.8: `p_value_corrected = 0.0`;
* Gaussian replicate 34, `s(x1)`: `5.55e-16`.

While this lane was open, pv-tails-numerics replaced Imhof on main with a
saddle-point inversion of the moment generating function
(`gam_math::probability::signed_weighted_chi_square_sf`), accurate relative to
the tail. That removes the floor at its root. This lane keeps the part of L4
that is about the row: a p-value is published as a point value only when its
own certified accuracy (`p_value_bound`, the inversion's error plus twice the
selection replay's Monte-Carlo standard error) separates it from zero.
Otherwise the row is `p < p_value_upper_bound`, the top of the certified
interval, and never `0.0`. After the merge a huge effect is a finite tiny
point value within its accuracy of the exact tail (Rust test
`a_huge_effect_is_a_finite_tiny_p_value_at_the_exact_tail`, a closed-form
`F` tail down to ~1e-90), and the audit's Poisson replicate 5 and Gaussian
replicate 34 are pinned in `tests/smooth_significance_typed_p_value_test.py`.

The bound branch still fires where the replay's noise, not the inversion, is
what limits the row. Three null replicates of 1500 in the final study
(binomial 206, Poisson 205 and 206) had `W ≈ 14.0–15.3`. There the certified
accuracy `p_value_bound ≈ 0.0028–0.0041` was larger than the corrected tail,
and the row reads `p < 0.0050–0.0072`. That is a true statement at the
replay's resolution, not a point value it cannot support. The size counts
such a row as rejecting at α only when its bound is below α; the KS test
uses the point values alone, hence its n of 499 (binomial) and 498 (Poisson).

### Estimated-scale statistic clamped to zero (found and fixed here)

With a profiled scale, `W = n·ln(1 + Q/V) + B` with `B < 0`. `W` used to be
clamped to 0, which scored every statistic in `(B, 0)` as `P(W > 0)` instead
of its own tail. Against the from-scratch null refit, 241 of the 500 Gaussian
null replicates landed there. With the nested null at a shared `λ̂` the two
residual degrees of freedom differ only by the dropped term's edf, `B` is
correspondingly small, and one Gaussian replicate of 500 (427) has `W < 0`.
The support is pinned by the Rust test
`profiled_scale_reference_tests::a_statistic_between_the_offset_and_zero_is_scored_where_it_is`.

### Atom at p ≈ 0.5: the observation was read on the wrong scale (fixed here)

With the nested null in place the sizes were right but the Gaussian KS still
rejected at D = .44. A null term that REML shrinks away got p ≈ 0.50–0.54
instead of spreading over (0.5, 1]: the ECDF jumped from F(.50) = .519 to
F(.55) = .984.

**Root cause.** The λ̂-selection replay draws a whitened score `z` for the
tested block, re-selects `λ` on each draw and records the pair
`(W_sel, W_cond)`: the statistic at the draw's own selection, and at the
fitted `λ̂`. The correction added to the exact conditional tail was
`P̂(W_sel ≥ x) − P̂(W_cond ≥ x)`, with `x` the data's statistic in both.
But `x` is the observation at the fitted `λ̂`, the `W_cond` scale; the
`W_sel` arm needs the same observation *under the replay's selection*.
For a shrunk term the replay's shrunk draws land at `W_sel ≈ 0`, the data's
`x` sat just above them, and the correction subtracted about half the atom.

**Fix.** The observed block is scored the way the replay scores a draw: its
whitened score `z_obs = Wᵀ g` at the nested null (divided by
`sqrt(D_f / E[V])` when the scale is profiled) is taken through both of the
replay's statistic maps, and

```text
x_sel = x · W_sel(z_obs) / W_cond(z_obs)
p     = p_cond(x) + P̂(W_sel ≥ x_sel) − P̂(W_cond ≥ x).
```

At the identity `W_cond(z_obs) = x` the ratio is exactly `W_sel(z_obs)`. A
score that cannot be formed is the typed replay decline
`observed_score_unusable`, never a silent fallback. The Rust test
`the_rescored_selection_tail_is_uniform_on_both_sides` draws observations in
the replay's own world and checks the rescored p is U(0, 1) on both sides
(sizes and KS, with both the observations' and the replay's Monte-Carlo
error carried), and that the unrescored version fails the same test. After
the fix the Gaussian cell is uniform (KS D = .026, p = .88).

### Bartlett factor near 70 on a shrunk null term (found and fixed here)

The rescoring made the Gaussian cell uniform but left the binomial and
Poisson cells failing KS (D = .22 and .34) with 21% and 26% of null p-values
above .99. Poisson replicate 0, `s(x2)`: `p_value_uncorrected = 0.626`, but the
published p was 0.9994, with `bartlett_factor = 73.9`.

**Root cause.** With known scale the statistic is Lawley-corrected,
`W* = W / c`. The estimated-λ factor adds the ρ-variation mean shift to the
fixed-λ one, `c = (ref_df + shift) / ref_df`. For a term REML shrinks to its
null space `ref_df → 0` (here 1.96e-5), so a shift of .0014 made `c ≈ 74`
and moved every such term to p ≈ 1. The shift is a first-order account of
what estimating `λ` does to `W`. The selection replay already carries the
whole of that in the reference law (it re-selects `λ` on every draw), so
adding the shift counted λ estimation twice. The Gaussian cell was
unaffected because its profiled-scale reference has `c = 1`.

**Fix.** When a selection replay is present only the fixed-λ factor applies
(`correction_provenance = "lawley_lr_fixed_lambda"`); the estimated-λ shift is
used only when no replay carries λ estimation. Pinned in Python by
`test_poisson_null_term_is_not_bartlett_corrected_for_selection_twice`
(Poisson replicate 0: the fixed-λ lane, a factor within 1e-2 of 1, and the
p-value equal to the uncorrected one within its accuracy). After the fix the
binomial and Poisson cells are uniform (table above).

## Handoffs (owned by pv-lr-selection)

**(b) A replay decline is not surfaced.** When the selection replay declines
(a `SmoothLrSelectionDecline`), the row publishes the conditional tail and
does not say so; the decline label is not carried into the FFI row. The case
that exposed it (Gaussian replicate 427, `p = 9.0e-4` for a term with
`ref_df` 2.8e-6) now reads `p = 0.990` under the nested null, but the missing
label is independent of that.

**(c) Unbounded per-draw descent.** `SmoothLrSelectionReplay::generate_multiscale`
→ `select_draw` (the coordinatewise certified descent run per replay draw for
multi-penalty smooths) has no bound on work and can run for tens of minutes
on one replicate. With the 300 s harness limit, 12 binomial replicates
stalled before this lane and 21 under the nested-null build; none of the
1500 in the final study did (slowest: 39 s). Nothing in this lane bounds the
descent, so that is a change in which draws reach it, not a fix. Before this
lane Poisson replicate 263 took 2330 s. The stack is in `faer::linalg::svd` ← `generate_multiscale` ←
`lr_null_reference` ← `smooth_term_lr_inference_forspec`. The rest of each
cell takes 3–12 s per replicate.
