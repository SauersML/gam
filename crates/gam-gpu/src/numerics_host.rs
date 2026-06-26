//! Host-side scalar special functions shared by the CPU parity references of
//! the GPU backends.
//!
//! The CUDA kernels emit their own NVRTC-visible numerics (see
//! [`crate::numerics_device`]); this module is the matching **host** side
//! used by the CPU parity oracles (`bms_flex_row`'s test oracle) and the
//! CPU reference path (`pirls_row`'s probit CDF). Keeping a single definition
//! here means the host `erfc` cannot drift between backends.

/// Complementary error function `erfc(x) = 1 − erf(x)` evaluated on the host.
///
/// Routes to `libm::erfc`, the SunOS msun double-precision implementation
/// (accurate to within ~1 ulp across the entire real line). The CUDA kernel
/// side calls device `erfc`, which is itself msun-derived, so the host CPU
/// reference matches the device path to within a ULP. The previous
/// branchless Cody 1969 Chebyshev rational here was only ~1.2e-7 accurate
/// in relative terms; that ate seven digits of every probit `Mills =
/// φ/Φ = pdf / (½·erfc(-x/√2))` evaluation and made any sufficiently
/// tight finite-difference probe of `∂neglog/∂e = -w·s·Mills` (which the
/// analytic side computes from this same `cdf`, while the FD side
/// differences `log cdf` and cancels the erfc bias) break against itself
/// at the ~2e-7 floor instead of the genuine 5-point-stencil truncation
/// floor near 1e-12.
pub fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}
