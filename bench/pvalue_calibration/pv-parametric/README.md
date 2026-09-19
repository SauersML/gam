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

Every dataset is also tested by an oracle that knows the true smooth shape
`sin(2πx2)` and fits it as an unpenalized covariate:

- Gaussian: the least-squares t or F test of the same hypothesis. Its p-values
  are exactly U(0, 1) under the null, so its rate on the same datasets is the
  Monte Carlo baseline for that seed set.
- Binomial and Poisson: the maximum-likelihood Wald z / χ² test and the
  likelihood-ratio test of the unpenalized GLM. Neither is exact, but both are
  the standard large-sample tests. Their rates on the same datasets separate
  what the seed set does from what the engine does.

Acceptance is two-sided: a conservative size is a defect exactly as an
anti-conservative one is.

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
| binomial n=400 | x1 | 0.077 | 0.037 | 0.007 | 0.25 | 0.703 | 0.076 / 0.039 / 0.011 |
| binomial n=400 | g (χ², 3 df) | 0.081 | 0.041 | 0.007 | 0.62 | 0.794 | 0.078 / 0.040 / 0.009 |
| poisson n=200 | x1 | 0.096 | 0.041 | 0.009 | 0.94 | 0.930 | 0.103 / 0.042 / 0.009 |
| poisson n=200 | g (χ², 3 df) | 0.092 | 0.043 | 0.008 | 0.19 | 0.987 | 0.091 / 0.044 / 0.009 |

No fit failed and no p-value was withheld.

The oracle column for binomial and Poisson is the Wald test; the
likelihood-ratio oracle agrees with it to within 0.004 at every level.

At 0.05 and 0.01, 15 of the 16 sizes are within 2 MCSE of the level. The one
outside is gaussian n=200 `x1` at 0.05: 0.065, or 2.2 MCSE. The exact oracle
rejects 0.058 on the same 1000 datasets, so most of that excess comes from the
seed set. The engine's p-values track the oracle closely: correlation
0.96, and a median −log p ratio of 1.007.

On these 1000 seeds the four known-scale tests reject below the level at 0.05
(0.037 to 0.043), and binomial `x1` rejects 0.077 at 0.10 (−2.4 MCSE). The
oracle rejects just as rarely on the same datasets (0.039 to 0.044, and 0.076),
so the seed set is low here, not the engine. The check below separates the two.

## Known-scale cells at 11000 reps

The binomial and Poisson null arm run to 11000 reps (the null half of
`--reps 11000 --cells binom,pois`). The MCSE is 0.0029 at 0.10, 0.0021 at 0.05
and 0.00095 at 0.01. Each entry is the size and, in brackets, its distance
from the level in MCSE.

| cell | term | size @0.10 | size @0.05 | size @0.01 | KS p | oracle @0.10 / 0.05 / 0.01 |
|---|---|---|---|---|---|---|
| binomial n=400 | x1 | 0.0990 (−0.3) | 0.0499 (−0.0) | 0.0091 (−1.0) | 0.68 | 0.0993 / 0.0500 / 0.0100 |
| binomial n=400 | g (χ², 3 df) | 0.1024 (+0.8) | 0.0523 (+1.1) | 0.0102 (+0.2) | 0.50 | 0.1020 / 0.0519 / 0.0097 |
| poisson n=200 | x1 | 0.0944 (−2.0) | 0.0479 (−1.0) | 0.0103 (+0.3) | 0.13 | 0.0962 / 0.0481 / 0.0104 |
| poisson n=200 | g (χ², 3 df) | 0.1030 (+1.0) | 0.0525 (+1.2) | 0.0099 (−0.1) | 0.96 | 0.1042 / 0.0520 / 0.0098 |

No fit failed and no p-value was withheld.

Every size is within 2 MCSE of its level on both sides. The closest to the
edge is Poisson `x1` at 0.10, at −1.96 MCSE; the oracle is at −1.3 on the same
datasets. On the same datasets the engine and the oracle reject together. A
paired McNemar test of the datasets on which exactly one of them rejects finds
no difference at any level in either cell: every p ≥ 0.10, and the Poisson `x1`
0.10 split is 123 against 143, p = 0.24. The engine's p-values correlate with
the oracle's at 0.95 to 0.99.

No test in any cell is conservative or anti-conservative beyond Monte Carlo
error.
