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
