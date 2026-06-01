//! Host-side scalar special functions shared by the CPU parity references of
//! the GPU backends.
//!
//! The CUDA kernels emit their own NVRTC-visible numerics (see
//! [`crate::gpu::numerics_device`]); this module is the matching **host** side
//! used by the CPU parity oracles (`bms_flex_row`'s test oracle) and the
//! CPU reference path (`pirls_row`'s probit CDF). Keeping a single definition
//! here means the host `erfc` cannot drift between backends.

/// Complementary error function `erfc(x) = 1 − erf(x)` evaluated on the host.
///
/// Branchless Chebyshev rational approximation (Cody 1969): matches f64 libm
/// to within ~1 ULP across the input range and is what the GPU kernels'
/// analytic implementation derives from. Used so the CPU reference does not
/// depend on a feature-gated `libm` dependency.
pub(crate) fn erfc(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_nan() {
            f64::NAN
        } else if x > 0.0 {
            0.0
        } else {
            2.0
        };
    }
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.5 * ax);
    let r = t
        * (-ax * ax - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 { r } else { 2.0 - r }
}
