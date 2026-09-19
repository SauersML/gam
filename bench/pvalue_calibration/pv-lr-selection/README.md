# pv-lr-selection: the smooth-term LR p-value's λ̂-selection correction

Target: `smooth_significance(...)[*]["p_value_corrected"]`, the per-term
likelihood-ratio test (`crates/gam-models/src/fit_orchestration/drivers/smooth_term_lr.rs`).

Design (`lr_null_mc.py`): truth `η = b0 + a1·sin(2πx1) + a3·cos(2πx3)`,
`x ~ U(0,1)³`; model `y ~ s(x1) + s(x2) + s(x3)`. `s(x2)` is null (size),
`s(x3)` is weak (power). Replicate `r` draws its data from
`default_rng(50000 + r)`, so every column below is on the same datasets.

| cell | family | n | b0, a1, a3 | σ |
|---|---|---|---|---|
| `gauss_small` | gaussian (scale profiled) | 60 | 0, 1.0, 0.30 | 0.5 |
| `gauss` | gaussian (scale profiled) | 200 | 0, 1.0, 0.30 | 1.0 |
| `pois` | poisson | 200 | 0.5, 0.8, 0.25 | — |
| `binom` | binomial | 400 | 0, 1.5, 0.60 | — |

```
python lr_null_mc.py <cell> 600 <workers> out.json
```

KS: the two-sided Kolmogorov-Smirnov test of every null p-value against
U(0, 1). On the corrected reference the null p-value has no material atom
(a shrunk null block is scored where its statistic falls, not sent to one), so
the whole range is held to uniformity, and a mass piled near one fails it as
an excess near zero does.

## Results (600 replicates per cell, MCSE at .10 = 0.0122, .05 = 0.0089, .01 = 0.0041)

`before` = origin/main df02753c; `after` = this branch merged with it. Size
is `P(p ≤ α)`; `z` is its distance from α in MCSE, on either side.

| cell | | size .10 (z) | size .05 (z) | size .01 (z) | KS p | power .05 |
|---|---|---|---|---|---|---|
| gauss_small | before | 0.1117 (+0.95) | 0.0633 (+1.50) | 0.0150 (+1.23) | 0.608 | 0.7333 |
| gauss_small | after | 0.1100 (+0.82) | 0.0583 (+0.94) | 0.0117 (+0.41) | 0.726 | 0.7333 |
| gauss | before | 0.0950 (−0.41) | 0.0400 (−1.12) | 0.0100 (0.00) | 0.271 | 0.6783 |
| gauss | after | 0.0933 (−0.54) | 0.0417 (−0.94) | 0.0083 (−0.41) | 0.244 | 0.6833 |
| pois | before | 0.0883 (−0.95) | 0.0433 (−0.75) | 0.0067 (−0.82) | 0.865 | 0.8183 |
| pois | after | 0.0883 (−0.95) | 0.0433 (−0.75) | 0.0067 (−0.82) | 0.865 | 0.8183 |
| binom | before | 0.1057 (+0.46) | 0.0352 (−1.65) | 0.0050 (−1.22) | 0.610 | 0.9094 |
| binom | after | 0.1057 (+0.46) | 0.0352 (−1.65) | 0.0050 (−1.22) | 0.610 | 0.9094 |

Every cell is inside two MCSE on both sides at every level and none rejects
uniformity. The binom cell is on 596 of 600 replicates, before and after:
replicates 241, 366, 498 and 552 fail the fit itself (the outer REML
optimizer does not certify a stationary optimum with λ on its rail), which is
upstream of the LR test and not this lane's.

pois and binom agree with main replicate for replicate (largest null
p-value difference 5e-4 on pois, 1.4e-5 on binom). Their known scale never
enters the profiled-scale path this lane changes, and on these fits the null
block's Lawley factor stays within 1.000 to 1.023, where the bounded factor
and the `mean_w / ref_df` ratio it replaced are the same number to 8e-5. The
ratio's blow-up (L1) needs `ref_df → 0`, which the tests in
`gam-terms/src/inference/lawley.rs` construct directly.

The ρ-conditional reference (`p_value_conditional`, no selection correction)
on the same datasets, for scale: gauss 0.1817 / 0.0917 / 0.0167, binom
0.1946 / 0.1057 / 0.0168.

Power: gauss_small gains 2 and loses 2 weak-term rejections of 600, gauss
gains 4 and loses 1; the null p-values correlate at 0.997 with main's.

What main already fixed. The finding's figures (n = 60 Gaussian 0.135 /
0.070 / 0.020) predate main's own rework of this reference; at df02753c the
n = 60 cell is already within two MCSE. What this lane still changes is the
construction, not a measured miss: the selection shift was evaluated at
`E[V]` and the replay selected with the known-scale criterion, both wrong in
principle for a profiled scale; with them fixed the n = 60 cell moves toward
nominal at every level (+1.50 → +0.94 MCSE at .05, +1.23 → +0.41 at .01).

## What changed and why

With a profiled scale the LR event `W ≥ w` is `Q ≥ c·V`, `V` the residual
law. The replay's selection shift was evaluated at `c·E[V]`; the shift is a
difference of two tails, not linear in the threshold, so its value at the mean
threshold is not its mean. It is now integrated over `V` (paired, the unit
block in closed form when the selection never saw it).

The replay also selected every draw's `λ` with the KNOWN-scale criterion. A
Gaussian fit selects with the profiled one, `m·ln D_p + log|I+T| − log|T|₊`,
whose data term depends on the whole penalized deviance
`D_p = R + Σ c²s`: `R = χ²_h + χ²_{r − r_j}` (the residual directions no
column reaches, and the other terms' penalized directions, each `χ²_1` under
the Gaussian law REML's fixed point is the mean of). A large residual
flattens the data term and moves `λ̂`, and that SAME `χ²_h` is the weight-one
block of the draw's `V`. The replay now draws `R` per draw, selects with the
profiled criterion on it, and reads the tail-shift event on the draw's own
`χ²_h`. Nothing is tuned: `h = n₊ − p`, `r` is the model's balanced-penalty
structural rank, `r_j` the tested block's.

The Lawley correction (L1). `bartlett_factor_from_mean` returned
`mean_w / ref_df`, which blows up as a shrunk term's `ref_df → 0`. The
fixed-λ Lawley term is now applied as a mean shift on the spectral
reference, per component: `w_j → w_j + Δε·w_j/d`. That is the scale
`c = 1 + Δε/d` on the reference, and `c` stays bounded as the block is
absorbed (`d → 0`; `fixed_lambda_factor_stays_bounded_as_the_block_is_absorbed`
in `gam-terms/src/inference/lawley.rs`). The driver scores
`W / c` against the replay reference `L`, which is the same as scoring `W`
against `c·L`.

The estimated-λ location term `δ_ρ = ½ tr(H_Δε Cov ρ̂)` is not applied. The
selection replay inside `L` already integrates over `λ̂`, so adding it would
count the `λ̂` variation twice. Its quadratic expansion is also unbounded
where `Δε` itself is bounded (λ on a rail). On binom it reached 14.9 and sent
6 of 600 null replicates to `p = 1`, which is a conservative bug.
`lawley_lr_correction_estimated_lambda` still reports it (scale and
location) through the inference instrument.

## Diagnostics

`LR_MC_FORMULA` / `LR_MC_SIGNAL` (gauss_small, earlier build):

* `y ~ s(x2) + s(x3)`, no signal anywhere: 0.1250 / 0.0633 / 0.0150,
  `KS | p < .5` 0.228. The miss is not caused by the signal in `s(x1)`.

Toy replays of the reference's own assumptions (`n = 60`, profiled-REML
truth, `P(p < α)` at .10 / .05 / .01):

| construction | no signal elsewhere | signal in the other terms |
|---|---|---|
| truth | 0.0527 at .05 | 0.0967 / 0.0520 / 0.0107 |
| known-scale selection, `V` independent (before) | 0.0587 at .05 | 0.0840 / 0.0413 / 0.0087 |
| profiled selection, shared `χ²_h` (after) | 0.0510 at .05 | 0.0810 / 0.0387 / 0.0067 |
| profiled selection, true noncentral other-term law | — | 0.0980 / 0.0527 / 0.0107 |

With no signal elsewhere the coupled replay is the truth law. With signal in
the other terms the other-term deviance is noncentral; neither the replay nor
the analytic reference (which is `Σ v_r χ²_1 + χ²_h`, central) sees that, and
the remaining gap in the toy is that, not the selection replay. On the real
cells it is below MC resolution.
