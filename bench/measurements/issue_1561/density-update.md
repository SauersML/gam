Committed and pushed `c286f7439`: cubature calibration and weights now use the
same declared rho distribution as posterior sampling. This adds the existing
proper PC prior on unset coordinates and the missing change-of-variables
Jacobian for explicit Gamma precision priors. The REML fitting criterion is
unchanged. Malformed/improper priors are rejected, and the obsolete mask path
is removed. MSI: 10 prior-engine tests pass, including the three new
distribution tests; the solver build also passes those three tests.

The new independent Gaussian audit in `6d7c78cf1` explains the earlier interval
result. SciPy reconstructs the exported 300-row, 13-coefficient problem to
4.13e-10 coefficient error and 1.71e-9 relative covariance error. Positive
quadrature in the two prior-CDF coordinates covers unbounded rho support.
Refining from 65 to 129 nodes per axis changes widths by at most 1.04224e-7
relative.

| Shared-tree inference calculation | Maximum width error against independent integral |
| --- | ---: |
| Original exported covariance | 23.822566% |
| Original with the premature bias transform undone | 2.045509% |
| Raw covariance plus consistent posterior density | 0.145918% |

Most of the original discrepancy was an estimand mismatch: the covariance of
bias-corrected coefficients accompanied the original coefficient mean. Main
already removed this obsolete machinery in `360c9bb1c` (#2670), while the
shared working tree used for these experiments still carried it. This audit
independently supports that removal; it is not a claim of a second new fix.
The replay preserves the original REML optimum and verifies
`Vp = Vb + smoothing_correction` to 3.59e-18 relative error.

The [exact exports, audit scripts, and raw results](https://github.com/SauersML/gam/tree/main/bench/measurements/issue_1561)
are retained. The 0.145918% result measures the integrated shared-tree cubature
implementation, not an isolated-main benchmark. Seven analytical covariance
checks and the solver response-scale check also pass on MSI.

This is progress on inference correctness, **not closure evidence**. The
tensor quality gaps, binomial reproduction, and complete-suite significance
require further verification. No fresh complete-suite superiority claim is
being made; #1561 remains open.
