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
* **Uniformity** of its p-values: a two-sided one-sample KS test against
  U(0, 1), and the one-sided `D+`, where the empirical CDF lies above the
  uniform's somewhere (anti-conservative at some level). Also the share with
  p = 1.
* **Power** of `s(x1)` and `s(x3)`. A `p < p_value_upper_bound` row rejects at
  every α above the bound.

## Results (500 replicates per cell)

RESULTS_PLACEHOLDER

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

BEFORE_AFTER_PLACEHOLDER

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
what limits the row. Three null replicates of 1000 in the study below (Poisson
205 and 206, Gaussian 213) had `W ≈ 14–17`. There the replay's `2·se ≈ 0.003–0.005`
was larger than the corrected tail, and the row reads `p < 0.005–0.009`. That
is a true statement at the replay's resolution, not a point value it cannot
support.

### Estimated-scale statistic clamped to zero (found and fixed here)

With a profiled scale, `W = n·ln(1 + Q/V) + B` with `B < 0`. `W` used to be
clamped to 0, which scored every statistic in `(B, 0)` as `P(W > 0)` instead
of its own tail. Against the from-scratch null refit, 241 of the 500 Gaussian
null replicates landed there. With the nested null at a shared `λ̂` the two
residual degrees of freedom differ only by the dropped term's edf, `B` is
correspondingly small, and one Gaussian replicate of 500 (427) has `W < 0`.
The support is pinned by the Rust test
`profiled_scale_reference_tests::a_statistic_between_the_offset_and_zero_is_scored_where_it_is`.

## Handoffs (owned by pv-lr-selection)

**(a) Boundary-mismatch atom at p ≈ 0.5.** A null term that REML shrinks away
gets p ≈ 0.50–0.54 instead of spreading over (0.5, 1]: the Gaussian ECDF jumps
from F(.50) = .519 to F(.55) = .984. The selection replay's shrunk draws land
at `W_sel ≈ 0`, while the data's own `W` sits just above them, so the
correction `P̂(W_sel ≥ x) − P̂(W_cond ≥ x)` subtracts about half the atom. This
is all of the remaining KS rejection; the size at α ≤ .10 is unaffected. It
is unchanged by the nested null.

**(b) A replay decline is not surfaced.** When the selection replay declines
(a `SmoothLrSelectionDecline`), the row publishes the conditional tail and
does not say so; the decline label is not carried into the FFI row. The case
that exposed it (Gaussian replicate 427, `p = 9.0e-4` for a term with
`ref_df` 2.8e-6) now reads `p = 0.990` under the nested null, but the missing
label is independent of that.

**(c) Unbounded per-draw descent.** `SmoothLrSelectionReplay::generate_multiscale`
→ `select_draw` (the coordinatewise certified descent run per replay draw for
multi-penalty smooths) has no bound on work and can run for tens of minutes
on one replicate. STALLS_PLACEHOLDER Before this lane Poisson replicate 263
took 2330 s. The stack is in `faer::linalg::svd` ← `generate_multiscale` ←
`lr_null_reference` ← `smooth_term_lr_inference_forspec`. The rest of each
cell takes 3–12 s per replicate.
