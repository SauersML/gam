# Parametric-term p-value calibration

`calibrate.py` fits `y ~ x1 + g + s(x2)` to seeded data with a real smooth in
`x2`. Under the null neither the linear coefficient `x1` nor the 4-level factor
`g` enters. Under the alternative both do. Every p-value it reads is the one
the summary payload carries: `x1`'s row in `parametric_terms`, and `g`'s joint
Wald row in `parametric_term_tests`. The CLI's `gam summary` prints the same
payload, bit for bit (`tests/test_parametric_pvalue_calibration.py`).

```
python bench/pvalue_calibration/pv-parametric/calibrate.py --reps 1000 \
    --out bench/pvalue_calibration/pv-parametric/results.json
```

The run takes about 9 minutes on 4 cores; the tables below are that run's
`results.json` (bench results are not committed). With 1000 reps the Monte Carlo SE is 0.0095 at 0.10, 0.0069 at 0.05 and 0.0031
at 0.01.

For the Gaussian cells the script also runs the exact oracle on every dataset:
the least-squares t or F test of the same hypothesis, with the true smooth shape
as a known covariate. Its p-values are exactly U(0, 1) under the null, so its
rate on the same datasets is the Monte Carlo baseline for that seed set.

## Reference distribution

- Estimated scale (Gaussian): Student-t for a coefficient, and F(q, n − edf) for
  a term.
- Known scale (binomial, Poisson): N(0, 1) and χ²_q.

The fit decides which reference applies; no caller chooses it.

## Before

A linear term carries its own REML ridge. Its Wald statistic was scaled by the
posterior SD, which charges the ridge prior's own variance to the estimate. At
a true null the p-values piled up near 1.

| cell | `x1` size @0.05 | KS p |
|---|---|---|
| gaussian n=30 | 0.014 | ~1e-250 |
| gaussian n=200 | 0.012 | ~1e-250 |

(500 reps.) A categorical term is a ridge-penalized level block. It had no
p-value at all: its smooth-table row carries EDF only.

## After (1000 reps, default seeds)

The statistic is now scaled by the conditional covariance of the estimate with
the term's own ridge prior removed, `φ[H⁻¹(H − S_J)H⁻¹]_JJ`. That makes it the
statistic of the fit with the term unpenalized, whatever the ridge's λ is.

A factor is tested once, on its L − 1 contrasts. The level mean is aliased with
the intercept and set by the ridge, so only the contrasts carry information.
The derivation is in `crates/gam-solve/src/estimate/parametric_term_summary.rs`.

| cell | term | size @0.10 | size @0.05 | size @0.01 | KS p | power @0.05 | oracle @0.10 / 0.05 / 0.01 |
|---|---|---|---|---|---|---|---|
| gaussian n=30 | x1 | 0.081 | 0.042 | 0.014 | 0.18 | 0.883 | 0.082 / 0.038 / 0.010 |
| gaussian n=30 | g (F, 3 df) | 0.101 | 0.060 | 0.012 | 0.80 | 0.964 | 0.099 / 0.054 / 0.008 |
| gaussian n=200 | x1 | 0.112 | 0.065 | 0.015 | 0.48 | 0.486 | 0.111 / 0.058 / 0.016 |
| gaussian n=200 | g (F, 3 df) | 0.121 | 0.046 | 0.007 | 0.059 | 0.574 | 0.109 / 0.046 / 0.007 |
| binomial n=400 | x1 | 0.077 | 0.037 | 0.007 | 0.25 | 0.703 | — |
| binomial n=400 | g (χ², 3 df) | 0.081 | 0.041 | 0.007 | 0.62 | 0.794 | — |
| poisson n=200 | x1 | 0.096 | 0.041 | 0.009 | 0.94 | 0.930 | — |
| poisson n=200 | g (χ², 3 df) | 0.092 | 0.043 | 0.008 | 0.19 | 0.987 | — |

No fit failed and no p-value was withheld.

At 0.05 and 0.01, 15 of the 16 sizes are within 2 MCSE of the level. The one
outside is gaussian n=200 `x1` at 0.05: 0.065, or 2.2 MCSE. The exact oracle
rejects 0.058 on the same 1000 datasets, so most of that excess comes from the
seed set. The engine's p-values track the oracle closely: correlation
0.96, and a median −log p ratio of 1.007.

The binomial Wald tests are somewhat conservative at 0.10, which is valid
(P(p ≤ a) ≤ a). No KS test rejects uniformity at 0.05.
