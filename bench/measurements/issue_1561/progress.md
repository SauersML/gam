# Issue 1561: measurement integrity and current reproductions

MSI acn112, 2026-09-08. CPU execution only; shared scratch build caches.

The closure requirement is a one-sided paired Wilcoxon test on **each test's
median log error ratio**, with GAM better at p < 0.05. The existing aggregator
instead ranked individual metric rows. Multiple channels of one experiment
therefore received multiple votes. It also matched failed tests to emitters by
module name, allowing an emitting sibling to conceal a failed case.

The revised telemetry carries the executing libtest case, independently of the
human channel label. Aggregation takes the median within each case before
ranking. Invalid metrics invalidate the case; inconsistent duplicate results
refuse aggregation; failed or unrecorded execution blocks closure. The rank
variance now uses the same exact tie groups as the ranks, and even-sized
medians average both central observations. Old logs without case identities
must be regenerated rather than guessed into experimental units.

Validation so far:

- Python 3.12 on MSI: 17 regression tests passed in 0.024 seconds. The gate
  now counts the exact tied-rank sign-randomization distribution, rather than
  using a normal approximation. An exhaustive five-observation sign panel
  independently checks all 32 outcomes. Four unanimous wins cannot yield
  p < 0.05 (their exact one-sided p is 1/16).
- Closure additionally requires the complete case manifest from the same
  binary's successful `--list`. Missing, unexpected, or failed executions
  block closure. The workflow retains that manifest, and failed discovery
  cannot be converted into a partial successful listing. Workflow YAML parsed
  successfully on MSI. The aggregator returns a failing status for failed
  closure, while the workflow still retains its explicitly non-gating report.
- Rust telemetry regression: the executing case survives multiple channel
  labels and the reported metric round-trips to the original f64 bits.
- Pending positive-measure covariance implementation: seven analytical tests
  passed in 0.02 seconds, including varying conditional covariance and
  between-direction movement of coefficient means.
- Its full solver response-scale equivariance test passed in 0.13 seconds.
- Existing warm quality binary reproduced the Poisson tensor loss in 23.45
  seconds: GAM RMSE 0.2400183792880, mgcv RMSE 0.1565154925828; EDF
  10.073 versus 10.828. R 4.2.2, mgcv 1.9-1 from the shared R library.

Exact-zero errors are retained as ties or limiting infinite log ratios, without
an arbitrary epsilon. A case with an undefined median is invalidated. The
ArviZ LOO telemetry was removed from the superiority aggregate because it
compared held-out GAM predictions with LOO on different training rows from
GAM's own posterior. Its original quality assertions and diagnostics remain.

## Empirical covariance check

The model-service reproduction of the existing fixed-design Gaussian experiment
completed all 30 replicates, 300 rows each. Conditional 95% coverage was
**0.951889**, marginal coverage **0.970000**. All 9,000 marginal widths were
wider; their ratio to conditional widths ranged from **1.05368213 to
1.38683553**. This reproduces the old conditional coverage and passes the
fixture's empirical coverage and ordering requirements. It does not certify
the accuracy of the marginalization: direct integration remains necessary,
and total covariance does not mathematically require widths to exceed those
conditional at the mode.

## Tensor model audit

The exported CSVs retain the exact original observations and known truth. The
standalone R audit crosses natural-cubic/P-spline margins with and without
null-space shrinkage; original quality thresholds are unchanged.

| Model | Gaussian interaction RMSE | Poisson mean RMSE |
| --- | ---: | ---: |
| GAM, current model-service reproduction | 0.034695002792 | fit refused |
| mgcv, P-spline, no null shrinkage (original reference) | 0.028519483236 | 0.156515492583 |
| mgcv, P-spline, null shrinkage | 0.029133802933 | 0.237700958072 |
| mgcv, natural cubic, no null shrinkage | 0.029011539874 | 0.167101388564 |
| mgcv, natural cubic, null shrinkage | 0.034694911561 | 0.231753400302 |

GAM's Gaussian predictions differ from mgcv's natural-cubic, `select=TRUE`
predictions by at most **5.96e-7**. Thus this discrepancy is attributable to
the statistical basis and shrinkage choices, not evidence of a defective
Gaussian optimizer. The fixture fits an interaction-only model to observations
that also contain main effects; those omitted effects contribute to estimated
dispersion and smoothing. Its documentation previously asserted an incorrect
universal REML interpolation floor. That explanation has been corrected.

The current Poisson fit fails before returning predictions:
`positive smoothing cubature proposal extends beyond the resolved rho domain`.
The older executable's 0.240018379288 RMSE is a baseline reproduction, **not**
validation of the current cubature implementation. The binomial comparison
using the older executable exceeded a 90-second diagnostic bound and was
terminated; no passing binomial result was obtained.

## Verification limits and remaining work

The root quality target could not build because its repository audit rejects
unrelated in-progress trait argument and test-unwrap changes. Focused examples
used the model-service API and the existing MSI warm libraries. Their outputs
are retained alongside this report; they are not a complete quality-suite run.
A redundant rebuild was stopped, and the small diagnostic was compiled directly
against the already-built libraries to avoid rebuilding the model stack.

The cubature domain refusal and its integration accuracy, the statistical
quality shortfalls, the binomial fit, and the complete reference-quality
comparison remain unresolved. In particular, the cubature's use of the fitting
criterion needs to be reconciled with the proper distribution correction used
by the rho-posterior sampler before claiming both integrate the same posterior.
**Issue #1561 remains open. No fresh complete-suite significance result exists.**

## Independent Gaussian integration and density correction

The first replicate is now exported as `gaussian-integration-problem.json`.
`gaussian_posterior_audit.py` reconstructs its Gaussian REML criterion,
coefficient mode, and conditional covariance independently with SciPy. Errors
are respectively 2.01e-8 absolute, 4.13e-10 maximum coefficient error, and
1.71e-9 relative covariance error. The criterion reconstruction accounts for
the determinant of the solver's unpenalized-column scaling.

Integration uses positive Gauss-Legendre quadrature on the two independent
PC-prior CDF coordinates. This covers unbounded log precision without choosing
rho bounds or a Gaussian proposal. Dispersion stays fixed at the exported
estimate, matching the production cubature's estimand. Orders 17, 33, 65, and
129 refine the calculation; the final refinement changes prediction widths by
at most **1.04224e-7 relative**. This is numerical convergence evidence for this
two-penalty fixture, not a general quadrature error certificate.

| Covariance calculation | Maximum width error against independent integral |
| --- | ---: |
| Original exported shared-tree model | 23.822566% |
| Original covariance with its premature bias transform undone | 2.045509% |
| Shared-tree replay with raw covariance and consistent distribution prior | 0.145918% |

The original covariance had been transformed by `A V Aᵀ`, where
`A = I + H⁻¹S`, despite the reported coefficients being the original beta.
The removal of that obsolete frequentist machinery is already in main as
`360c9bb1c` (#2670); the shared working tree used for these experiments still
carried it. This audit independently supports the removal and does not claim
it as a new main-line fix. The earlier 97% coverage result cannot establish
integration accuracy, and the 23.8% error must not be attributed entirely to
quadrature: most of it was a mismatched coefficient/covariance estimand.

Commit `c286f7439` additionally makes calibration and cubature weights consume
the same declared density as the rho sampler. Flat criterion coordinates get
the existing proper PC distribution prior. Explicit Gamma precision priors
get their missing log-precision Jacobian; Normal and PC priors are already
rho densities. Malformed and improper priors are rejected. This does not
change the REML fitting criterion. Ten shared-prior tests passed on MSI,
including three distribution regressions; the solver test build also passed
those three tests. The seven positive-measure analytical checks and solver
response-scale equivariance check passed against the integrated working tree.

The solver-only replay uses the original converged smoothing parameters as
seeds and verifies `Vp = Vb + smoothing_correction` to **3.5891e-18 relative**.
The replayed rho and beta match the original fit, so the improvement comes
from inference rather than a different optimum. Its export is
`gaussian-integration-corrected.json`. Raw oracle refinements and solver
diagnostics accompany this report. The 0.145918% figure describes the tested
shared-tree positive cubature implementation, not an isolated-main benchmark.

## Fresh public-model reproductions

Commit `30a462b62` carries the tested positive cubature proposal and stable
accumulator. The proposal now uses the data-derived resolvability domain,
averages feasible chord midpoints, and scales all its spherical nodes together
to keep them inside that domain. Failed numerical integration reports its
error rather than substituting a first-order covariance.

After rebuilding the model library, the same 30 Gaussian replicates give
conditional coverage **0.951889**, marginal coverage **0.954889**, and 9,000
wider marginal intervals with width ratios **1.00167608–1.10764761**. These
are the original data, seeds, nominal level, and coverage bar. The shared-tree
prediction test also passes all four combinations of covariance mode and
requested estimator transform; that test exercises machinery already removed
from main and is not a new main-line test requirement.

The current Poisson tensor model now returns successfully. Its RMSE is still
**0.240018379199** against the original mgcv P-spline reference's
**0.156515492583**. The Gaussian interaction RMSE remains
**0.0346950027463**. Successful execution resolves the observed refusal on this
integrated source, not these statistical quality gaps.

`binomial_holdout_1561` replays the original five prostate folds, starting with
the previously failing fold 3. All five fits succeed: fold times are 1.256,
0.584, 0.538, 0.651, and 0.311 seconds (execution order 3, 0, 1, 2, 4).
Predictions use the fitted frozen design. The independent Python audit checks
that each of the 654 observations is held out exactly once, that fold IDs and
labels match, and then runs the original additive reference models and bars.

| Metric | GAM | EBM | pyGAM |
| --- | ---: | ---: | ---: |
| Mean five-fold AUC | 0.7072979814 | 0.6991852711 | 0.7064062130 |
| Fold-0 held-out NLL | 0.6215316618 | 0.6210594769 | 0.6212567175 |

Both absolute bars and both match-or-beat margins pass. Reference versions:
interpret-core 0.7.8, pyGAM 0.12.0, scikit-learn 1.9.0, NumPy 2.2.6. EBM uses
four workers to respect the shared node; its statistical settings and seed
match the original fixture. These checks are still not a complete-suite
significance result.
