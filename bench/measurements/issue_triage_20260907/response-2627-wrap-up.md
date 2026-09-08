# Issue 2627 wrap-up, September 8

The user requested an immediate wrap-up. The full issue remains unresolved.

Published SAE test correction: 98ea20eb3e574b8934c66ccc1504b15ec41e37c8.
Eight focused posterior-orbit checks passed, none failed or ignored. The commit
also preserves the completed shard 9 census, not a workspace-wide verdict.

Unpublished shared-tangent work remains in
`crates/gam-models/src/response_geometry.rs` and
`crates/gam-models/src/response_geometry_rotation_tests.rs`.
It freezes the penalty range, uses canonical penalty roots and factor solves
for analytic derivatives, and accumulates isotropic statistics with streamed QR.
The existing FFI regression and its tolerances remain unchanged.

The last MSI build succeeded in 247 seconds. The resulting seven core checks
finished in 0.30 seconds: six passed, one failed, none ignored, 1,557 filtered.
Receipt: `response-2627-certificate.log` in this directory.

Fixed-smoothing coefficient rotation error is 2.1649348980190553e-15.
Optimized coefficient rotation error is 3.446964402797903e-7, exceeding
the unchanged 1e-7 assertion; prediction error is 2.984501534797346e-12.
At either returned smoothing vector, coefficients from the two response frames
agree after rotation within 7.24e-14, and analytic gradients agree closely.
The base fit's gradient norm is 9.275306506372683e-9; the rotated fit's is
3.061144327128992e-7. Both returned a curvature-resolvability certificate,
with bounds approximately 5.90e-7 and 6.02e-7 respectively. The latter exceeds
the requested projected-gradient cap of 2 * sqrt(f64::EPSILON).
The next investigation is why that explicit cap does not constrain the returned
certificate. This is a measured discrepancy, not a completed optimizer fix.

All owned builds and focused tests have finished. No new broad runs were started
for this wrap-up. Existing CI jobs and other users' jobs were left alone.
