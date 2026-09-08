Further verified progress, committed as `30a462b62` and `caf457207`:

- Cubature now keeps its positive spherical proposal inside the data-derived
  rho domain with one common scale. Centered moment accumulation preserves
  small covariance when coefficient means have large offsets. Numerical
  integration failures retain their errors rather than substituting a
  first-order covariance.
- The original 30-replicate Gaussian experiment now gives **95.4889% marginal
  coverage** at nominal 95%, versus the earlier 97.0000%. Conditional coverage
  is unchanged at 95.1889%. All 9,000 marginal widths are wider in this
  experiment; ratios range from 1.00167608 to 1.10764761.
- All five original prostate holdout fits succeed, including previously
  failing fold 3 in **1.256 seconds**. The five fits total **3.340 seconds**.
  Their frozen-design predictions pass both original absolute quality bars
  and both reference margins:

| Metric | GAM | EBM | pyGAM |
| --- | ---: | ---: | ---: |
| Mean five-fold AUC | 0.7072979814 | 0.6991852711 | 0.7064062130 |
| Fold-0 holdout NLL | 0.6215316618 | 0.6210594769 | 0.6212567175 |

The Poisson tensor fit also returns successfully on the integrated source.
Its RMSE remains **0.240018379199** against the original reference's
**0.156515492583**; the Gaussian interaction RMSE remains **0.0346950027463**.
Those statistical gaps have not been resolved.

[Exact predictions, scripts, versions, timings, and calibration logs](https://github.com/SauersML/gam/tree/main/bench/measurements/issue_1561)
are retained. All computation ran on MSI with at most four assigned CPUs;
the diagnostics reused the built libraries. These remain integrated shared-tree
measurements, not a fresh complete-suite significance result. **Issue stays
open.**
