The spatial value-trial optimization in `10d011668` does not resolve the
anisotropic gradient problem. A separate current model test fails on MSI:
`aniso_psi_duchon_gaussian_joint_gradient_matches_finite_difference_2735`.

The 80-row two-dimensional Gaussian fixture evaluates five parameter vectors.
All 25 smoothing-weight derivative comparisons agree. All ten spatial-coordinate
comparisons disagree, with no unresolved numerical comparisons. At the zero
vector, the first spatial derivative is approximately -518.19 analytically and
+20.78717 by central differences; the second is -563.28 versus -14.22436.
Value and derivative evaluations agree on the criterion within 3.8e-12.
The run fails in 1.44 seconds.

The isotropic Gaussian control passes all 25 comparisons in 1.31 seconds,
with worst relative spatial derivative discrepancy 1.25e-6. This narrows the
investigation to the per-axis derivative path. A component-level regression now
compares the analytic design and penalty derivatives to the actual rebuilt
objects on both spatial axes; its execution is pending.

These runs use the canonical model test binary
`.buildd/gam_models-4bea90744dbcb863` in the MSI validation source. The compiled
driver, derivative construction, and test source hashes match the current
remote files. The binary's exact dependency and source hashes are retained in
`.buildd/survival-test-native-current.provenance.json`. The failing receipt is
`bench/measurements/issue_triage_20260908/spatial-aniso-current.log`.

The basis component oracle isolated a logarithmic homogeneity error. For a
partial-fraction kernel with logarithmic Riesz blocks, the identity
`K_u = r K_r + delta K` omits the polynomial
`D = -sum(a_m c_m r^(2m-d))`. The raw-axis first derivative needs `D/d`;
the second derivative needs `(D_a + D_b)/d`. The exact collision carrier
already included that contribution, so correcting only non-collision pairs
restores a consistent derivative across the whole design.

The implementation now carries this polynomial through the coefficient chart,
the fixed row-space projector, first/second matrix actions, and the amplified
kernel chart. It evaluates bounded row chunks and shares the data/center
storage; it does not materialize all axis derivative matrices.

The public owning-crate oracle passes both the ordinary length-scale fixture
and an amplified kernel fixture at length scale 1e-6. It checks rebuilt values
using Richardson differences, diagonal and mixed second derivatives, forward
and transpose actions, fixed row projection, and single-row access. Both tests
pass on MSI in 0.68 seconds. In the ordinary fixture, first derivative relative
errors are 1.23e-11 and 1.46e-11; second derivative errors are at most 2.62e-8.
The amplified fixture has first derivative errors below 2.67e-12 and second
derivative errors below 1.89e-9. Receipts and dependency provenance are retained
in `bench/measurements/issue_triage_20260908/duchon-anisotropic-corrected*`.

The combined model-level rerun still fails nine spatial comparisons, now with
worst relative error 0.2836 rather than the original wrong-sign magnitude.
Its component check finds 17.6% design error and 26.6% native Gram penalty
error; the three differential-operator penalties agree to about 1.5e-10.
Thus the logarithmic correction is necessary but does not establish closure.

A second public regression identifies a discontinuity at zero anisotropy:
the forward basis replaces an explicit `[0, 0]` by approximately
`[0.09414416, -0.09414416]`, whereas arbitrarily small nonzero contrasts are
honored literally. All five parameter vectors in the full gradient gate have
equal raw spatial coordinates, which decode to exactly zero contrasts. The
new regression rejects this implicit reseeding in 0.18 seconds on the old
library. With forward construction and native penalty construction honoring
literal centered contrasts, all three basis tests pass in 1.24 seconds,
including the zero-contrast case. Its first derivatives agree to 1.36e-11
and its second derivatives to 2.61e-8. Full-model verification is pending.

The issue remains open pending the full criterion derivative gate and original
full-size fit. No large-scale timing or recovery result is claimed here.
