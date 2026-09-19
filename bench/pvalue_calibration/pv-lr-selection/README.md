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

KS: a double-penalty null smooth is shrunk to nothing on about half the
datasets, so a calibrated p has an atom at `1 − m` and is uniform below it. The
full-range KS rejects that by construction; `KS | p < .5` tests `2p` on that
event against U(0, 1) and is the shape check.

## Results (600 replicates per cell, MCSE at .05 = 0.0089, at .01 = 0.0041)

`before` = origin/main b0efd45a; `after` = this branch. Size is
`P(p ≤ α)`; `z` is its distance from α in MCSE.

| cell | | size .10 (z) | size .05 (z) | size .01 (z) | KS \| p<.5 | power .05 |
|---|---|---|---|---|---|---|
| gauss_small | before | 0.1200 (+1.63) | 0.0700 (**+2.25**) | 0.0167 (+1.64) | 0.191 | 0.7233 |
| gauss_small | after | 0.1217 (+1.77) | 0.0667 (+1.87) | 0.0167 (+1.64) | 0.125 | 0.7200 |
| gauss | before | 0.1050 (+0.41) | 0.0450 (−0.56) | 0.0067 (−0.82) | 0.555 | 0.6817 |
| gauss | after | 0.1083 (+0.68) | 0.0483 (−0.19) | 0.0067 (−0.82) | 0.706 | 0.6817 |
| pois | before | 0.0967 (−0.27) | 0.0400 (−1.12) | 0.0067 (−0.82) | 0.868 | 0.8167 |
| pois | after | 0.0967 (−0.27) | 0.0400 (−1.12) | 0.0067 (−0.82) | 0.880 | 0.8167 |
| binom | before | 0.1100 (+0.82) | 0.0400 (−1.12) | 0.0050 (−1.23) | 0.074 | 0.8950 |
| binom | after | BINOM_AFTER |

`KS | p < .5` is the KS p-value of `2p` given `p < .5`, from the saved
p-values.

The ρ-conditional reference (`p_value_conditional`, no selection correction)
on the same datasets, for scale: gauss_small 0.2050 / 0.1167 / 0.0383,
gauss 0.2033 / 0.1033 / 0.0250, pois 0.1850 / 0.0933 / 0.0200.

Power: on gauss_small the weak term loses 3 rejections and gains 1 of 600
(McNemar exact p ≈ 0.63) while the null size at .05 falls by 2 of 600; every
other cell's power is unchanged.

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

## Open: binom mid-range excess (not fixed here)

The binom cell's `KS | p < .5` rejects both before and after this lane. The
`p_value_uncorrected` column (known-scale reference, no Lawley term) is
identical to origin/main replicate for replicate. This lane does not touch
the known-scale path, so the defect is pre-existing. On 1200 replicates
(`default_rng(50000 + r)`, r = 0..1199):

| | size .10 (z) | size .05 (z) | size .01 (z) | KS \| p<.5 |
|---|---|---|---|---|
| corrected | 0.1167 (+1.92) | 0.0492 (−0.13) | 0.0100 (0.00) | 0.000 |
| uncorrected | 0.1208 (+2.41) | 0.0533 (+0.53) | 0.0108 (+0.29) | 0.000 |

Histogram of `2p` given `p < .5`, in tenths: 59 81 64 73 72 58 57 42 38 48.
The size at .05 and .01 is right. The excess sits at `p ∈ (.05, .25)`, where
`W` is 3 to 6 and `ref_df` is 1.4 to 2: there the replay under-corrects the
selection. Two candidate causes are not established:

* The replay selects with the Gaussian-proxy criterion. LAML's non-Gaussian
  terms (the working-weight derivatives in the log-determinant) are missing
  from it.
* The other terms' λ re-selection and their noncentral deviance. These are
  pv-lr-refit's null refit, not this lane's.

The main-branch corrected column's `KS | p < .5` of 0.074 on r = 0..599 came
from the `Δε/ref_df` ratio, not from calibration. It is gone with L1.

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
