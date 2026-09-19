# pv-bands: simultaneous difference-smooth bands and the no-difference test

Seeded Monte Carlo calibration of `difference_smooth(..., simultaneous=True)`
bands and of the whole-curve "no difference" p-value (the row `p_value`).

## Design

- Model: `y ~ g + s(x, by=g, k=10)`, `x ~ U(0, 1)`. The groups are unequal:
  A is the 30% minority.
- Linear predictor: `eta = 0.2 + 0.8 sin(2 pi x)`. For the difference arm,
  group B adds `d(x) = 0.6 cos(pi x)` on the eta scale.
- Families: Gaussian (sd 0.5), Poisson (log link) and binomial (logit link),
  each at n = 100, 400 and 2000.
- Each cell runs 500 replicates of each arm, so the MCSE at 0.95 is at most
  about 0.0104.
- Grid: 50 points spanning the training x range.
- **Coverage:** whether the band contains `d` at every grid point. It is
  checked for the simultaneous band at 0.90/0.95/0.99 and for the pointwise
  band at 0.95, which is the G2 comparison.
- **Size and KS:** both groups share one curve. Size at alpha is
  `P(p_value <= alpha)`. A one-sample Kolmogorov-Smirnov test checks the null
  p-values for uniformity.
- **Power:** `P(p_value <= 0.05)` under `d`.
- Seeds:
  - Replicate r uses `100_000 * (1 + family index) + 10 n + r`, and the null
    arm adds `5_000_000`.
  - The band's max|Z| law uses 10 000 draws with seed 12 345. Both are fixed
    in Rust and reported on every row as `n_sim` and `seed`.
- Refused fits are counted with their message, never dropped silently.

Reproduce (about 32 minutes on a 4-core container):

```
python bench/pvalue_calibration/pv-bands/run.py --reps 500 --out bench/pvalue_calibration/pv-bands/results.json
python bench/pvalue_calibration/pv-bands/run.py --summarize bench/pvalue_calibration/pv-bands/results.json
```

`results.json` is git-ignored by repository policy (`bench/**/results.json`).
The summary below is copied from it verbatim.

## What the band is

- The band sits on the linear-predictor scale. The contrast is
  `C = X_B - X_A`, and its covariance is `C V C^T`.
- `V` is the covariance the fit publishes:
  - smoothing-corrected when the fit carries the correction;
  - conditional when the correction is typed unavailable. Before this change,
    the band refused such fits.
- The critical value is the `level` quantile of `max_i |Z_i|`, where
  `Z ~ N(0, R)` and `R` is the correlation of `C V C^T`.
- The whole-curve p-value is `(1 + #{M_s >= T}) / (S + 1)`, where
  `T = max_i |diff_i| / se_i` and `M_s` are the same simulated maxima. The
  p-value and the band are therefore dual: the band at `level` excludes 0
  somewhere exactly when `p_value <= 1 - level`.
- There is no Bonferroni fallback. The dead `multi_point_joint` Bonferroni
  field is deleted.

## Results (500 replicates per cell)

### Whole-curve coverage of d(x)

The target is the level. `R` is the number of usable replicates, `err` the
number of refused fits, and `conditional` the number of replicates that
published the conditional covariance.

```
family        n    R err  sim@0.90  sim@0.95  sim@0.99  pw@0.95  MCSE@.95  conditional
gaussian    100  498   2     0.924     0.950     0.978    0.707   0.0098            0
gaussian    400  500   0     0.964     0.982     0.992    0.716   0.0097            0
gaussian   2000  500   0     0.978     0.994     1.000    0.652   0.0097            0
poisson     100  456  44     0.761     0.840     0.930    0.500   0.0102          305
poisson     400  496   4     0.913     0.944     0.978    0.692   0.0098          444
poisson    2000  496   4     0.935     0.964     0.996    0.649   0.0098          265
binomial    100  443  57     0.702     0.795     0.885    0.540   0.0104          159
binomial    400  497   3     0.803     0.891     0.954    0.565   0.0098          183
binomial   2000  486  14     0.930     0.969     0.994    0.722   0.0099          156
```

### No-difference test: size under a shared curve, power under d(x)

```
family        n    R err  size@.10  size@.05  size@.01  MCSE@.05  KS D     KS p    power@.05
gaussian    100  498   2     0.106     0.074     0.032    0.0098  0.2012  0.0000     0.606
gaussian    400  500   0     0.068     0.038     0.008    0.0097  0.2218  0.0000     0.998
gaussian   2000  497   3     0.066     0.026     0.004    0.0098  0.1628  0.0000     1.000
poisson     100  480  20     0.210     0.133     0.077    0.0099  0.1143  0.0000     0.575
poisson     400  498   2     0.080     0.050     0.012    0.0098  0.0997  0.0001     0.952
poisson    2000  491   9     0.053     0.026     0.010    0.0098  0.1628  0.0000     1.000
binomial    100  467  33     0.156     0.090     0.017    0.0101  0.1170  0.0000     0.334
binomial    400  497   3     0.171     0.121     0.056    0.0098  0.0796  0.0035     0.441
binomial   2000  486  14     0.062     0.031     0.006    0.0099  0.2871  0.0000     0.733
```

Almost all refused fits are REML outer-optimizer refusals ("did not certify a
stationary optimum"). They occur mostly at n = 100 for binomial (87) and
Poisson (63), with 28 at binomial n = 2000. `--summarize` lists every refusal.

## Reading the numbers

**The band construction is calibrated.**

- The seeded Rust tests in `crates/gam-inference/src/effects.rs` draw the
  estimate exactly from `N(truth, C V C^T)`, with S = 10 000 max|Z| draws.
- In those tests:
  - whole-curve coverage is within 2 MCSE of the level;
  - the no-difference p-value is KS-uniform;
  - the p-value agrees with the band's zero-exclusion at every level.
- The max|Z| machinery is therefore exact for the covariance it is given.
  The simulation count, seeding, correlation and eta-scale construction are
  not the source of any miscalibration.

**G2 is fixed where the band is used.**

- Pointwise 95% bands cover the whole curve only 0.50-0.72 of the time in
  every cell.
- The simultaneous band is the one to use for whole-curve statements.

**The fitted-model numbers only partly meet the acceptance target**, which is
within 2 MCSE of 0.95 coverage and 0.05 size.

- Poisson n = 400 hits both targets (coverage 0.944, size 0.050), though its
  KS still rejects.
- Gaussian n = 100 hits the coverage target at 0.950. Its size of 0.074 is
  2.4 MCSE high.
- Gaussian is conservative at larger n: coverage is 0.982 and 0.994, and size
  is 0.038 and 0.026. Its KS rejects.
  - The published smoothing-corrected covariance is wider than the sampling
    spread of the contrast under this truth.
  - That is a property of the covariance and the REML estimator (the
    rho-uncertainty lane), not of the band.
- Small-n Poisson and binomial are anti-conservative. Coverage at n = 100 is
  0.84 and 0.80, and size is 0.13 and 0.09.
  - Many of those fits publish the conditional covariance, because their
    correction is typed unavailable.
  - At n = 100 there is also smoothing bias from heavy penalization.
  - Both effects shrink with n. At n = 2000, coverage is 0.964 and 0.969 and
    size is 0.026 and 0.031.
- Binomial n = 400 stays anti-conservative, with coverage 0.891 and size
  0.121.

**No correction was added.** There are no fudge factors, inflation constants
or Bonferroni fallback. The remaining gap belongs to the covariance and to the
REML estimator, and it should close in those lanes; this bench re-measures it
without code changes.

The fast seeded regression slices live in
`tests/test_difference_smooth_band_calibration.py` and in the Rust unit tests
above.
