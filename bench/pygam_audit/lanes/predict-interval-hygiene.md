# predict-interval-hygiene

TITLE: Student-t intervals for estimated-scale families and delete dead PredictOptions.multi_point_joint
WORK ITEM: Read audit/inference.md L2 and L3. posterior-sd and partial-effects touch interval code, but neither changes the quantile or removes the dead option.
L2 evidence: Gaussian/estimated-scale intervals use a z quantile (crates/gam-predict/src/interval_policy.rs:98 standard_normal_quantile; gam-predict lib.rs:2839). gauss_small (n=60) 95% coverage: corrected 0.934, conditional 0.921, obs 0.937.
Fix (SPEC: posterior mean default, no magic constants): use the Student-t quantile with df = n - edf, from marginalizing phi. Apply it only to families with estimated scale; known-scale families keep z.
L3: PredictOptions.multi_point_joint (lib.rs:1478-1485, 1689-1697) is always false at every construction site (1557, 3298, 3525, 3563, 3893, 3938). Delete it per SPEC "delete unnecessary options". Joint bands already go through effect_report (gam-inference effects.rs:260).
Coordinate: posterior-sd (same interval_policy.rs; agree on the ordering), partial-effects, rho-marginal-predict.
Acceptance: an MC test (500 reps, gauss_small n=60) asserts 95% coverage in [0.94, 0.96] for the corrected intervals. A unit test asserts the quantile equals t_{n-edf}. A grep test asserts multi_point_joint is absent from crates/. Coverage and grep both fail at HEAD.
