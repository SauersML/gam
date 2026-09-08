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

- Python 3.12 on MSI: 12 regression tests passed in 0.005 seconds.
- Rust telemetry regression: the executing case survives multiple channel
  labels and the reported metric round-trips to the original f64 bits.
- Pending positive-measure covariance implementation: seven analytical tests
  passed in 0.02 seconds, including varying conditional covariance and
  between-direction movement of coefficient means.
- Its full solver response-scale equivariance test passed in 0.13 seconds.
- Existing warm quality binary reproduced the Poisson tensor loss in 23.45
  seconds: GAM RMSE 0.2400183792880, mgcv RMSE 0.1565154925828; EDF
  10.073 versus 10.828. R 4.2.2, mgcv 1.9-1 from the shared R library.

This is not evidence for closing the issue. Current end-to-end interval
calibration, the tensor failures, the binomial fit refusal, and the complete
reference-quality comparison still need verified resolution.
