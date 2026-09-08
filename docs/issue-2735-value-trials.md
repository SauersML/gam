The current spatial driver promotes `OuterEvalOrder::Value` requests to
`ValueAndGradient` inside `SpatialJointContext::eval_full`. Consequently a
value-only trial builds the spatial derivative directions and computes a full
gradient. The subsequent gradient request can hit that completed cache entry,
matching the expensive-Value/zero-cost-gradient pattern reported here. This is
source-level attribution, not yet a measurement of how much of the reported
324-second trial it accounts for.

The working-tree repair routes Value requests through the existing
`eval_cost` implementation. That evaluator retains the converged inner solve
and exact criterion while avoiding spatial derivative assembly. Its cache stores
only the value; an accepted-point gradient request must still compute its own
derivatives. The outer fast-path accounting now uses the requested derivative
order too.

The new regression
`spatial_value_trials_do_not_compute_or_cache_a_gradient_2735` exercises the real
spatial context on a Duchon/probit problem. It checks that a value trial does not
populate the derivative cache, compares value-only and gradient-path costs,
checks the analytic spatial derivative against differences of value-only calls,
and checks the gradient again after intervening value probes.

Published as `10d011668` on main. Current MSI verification: the new regression
passed in **0.42 seconds**, covering the value/gradient agreement, finite
difference check, and cache behavior above. Its result is recorded in
`bench/measurements/issue_triage_20260908/spatial-value-current.log`.

The original large-scale acceptance case remains outstanding. The issue stays
open; no measured end-to-end speedup or full resolution is claimed. No local
build or test was run.
