Published [73dbcab7a](https://github.com/SauersML/gam/commit/73dbcab7a9bb145f7b730f6a55e355cb70446f5f). **This issue is still not fixed.**

The SAE repairs retain the existing regression assertions:

- Preserve row, gradient, and device-frame allocation identity across accepted iterations while rebuilding their numerical contents.
- Carry the amplitude barrier's scalar curvature through the structured operator, restoring the small framed device path without dropping curvature. A new analytic-reference test checks CPU/device operands.
- Correct the sphere, function-Gram, inactive entropy-coordinate, and requested-rank fixtures.

MSI focused result: **15 passed, 1 failed, 0 ignored, 1,252 filtered out, 2.67 seconds**. The allocation-reuse and curvature checks pass. The remaining large-border regression now reaches its intended 2,048-coordinate system, but still fails because device operands are absent. Its assertion remains active. [Full output](https://github.com/SauersML/gam/blob/73dbcab7a9bb145f7b730f6a55e355cb70446f5f/bench/measurements/issue_triage_20260907/sae-2627-final.log).

Separately, [CI run 34253714873](https://github.com/SauersML/gam/actions/runs/34253714873), at `7c3c5b43b`, completed the gam-pyffi population: **124 run, 120 passed, 4 failed, 0 skipped, 12.449 seconds**. Failures are circle-latent convergence, the isotropic-noise mixture control, ring-of-clusters certification, and shared-tangent output-rotation equivariance. The workspace archive build passed; all ten Rust shards and both Python jobs were still running when checked.

The MSI receipt is from the shared development source, including other uncommitted changes; it is not an exact-main or full-workspace certification. Only reviewed changes were applied to main. [Verification ledger and remaining failures](https://github.com/SauersML/gam/blob/73dbcab7a9bb145f7b730f6a55e355cb70446f5f/docs/issue-2627-progress.md).
