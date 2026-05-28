//! Shared device-side probit numerics for NVRTC kernels.
//!
//! [`PROBIT_NUMERICS_CU`] is prepended to every NVRTC kernel source that needs
//! stable probit/normal-CDF arithmetic.  Keeping one copy here means a
//! numerics fix is a one-line change instead of a coordination problem across
//! multiple kernel source strings.
//!
//! Covered device functions (all `__device__ __forceinline__`, double precision):
//!   - `erfcx_nonnegative(x)`   — scaled complementary error function for x ≥ 0
//!   - `log_ndtr(x)`            — log Φ(x), numerically stable in the deep left tail
//!   - `log_ndtr_and_mills(x, *log_cdf, *lambda)` — joint (log Φ(x), φ(x)/Φ(x))

/// Device-side probit numerics injected at the top of every NVRTC kernel that
/// needs them.  Prepend this string to a kernel-specific body before passing to
/// `cudarc::nvrtc::compile_ptx` or `PtxModuleCache::get_or_compile`.
pub const PROBIT_NUMERICS_CU: &str = r#"
// -------- shared probit numerics -----------------------------------------
// All math in double precision.  No --use_fast_math.
//
// `log_ndtr(x)` = log Φ(x).  For x < 0 uses the erfcx representation
//   log Φ(x) = -u² + log(½ · erfcx(u)),   u = -x / √2
// which preserves digits all the way into the deep left tail (matches
// the CPU `normal_logcdf`).  For x ≥ 0 falls back to log1p(-½·erfc(x/√2)).
//
// `log_ndtr_and_mills(x, *log_cdf, *lambda)` returns both log Φ(x) and the
// Mills ratio φ(x)/Φ(x) in a single pass.  For x < 0 the erfcx path keeps
// the ratio stable even when Φ(x) underflows to zero.

#ifndef PROBIT_NUMERICS_INCLUDED
#define PROBIT_NUMERICS_INCLUDED

#define INV_SQRT_2PI 0.3989422804014327
#define SQRT_2       1.4142135623730951

extern "C" __device__ __forceinline__ double erfcx_nonnegative(double x) {
    if (!isfinite(x)) {
        return (x > 0.0) ? 0.0 : (1.0 / 0.0);
    }
    if (x <= 0.0) return 1.0;
    if (x < 26.0) {
        double xx = x * x;
        if (xx > 700.0) xx = 700.0;
        return exp(xx) * erfc(x);
    }
    // 4-term asymptotic expansion of erfcx for large x.
    double inv  = 1.0 / x;
    double inv2 = inv * inv;
    double poly = 1.0
                - 0.5      * inv2
                + 0.75     * inv2 * inv2
                - 1.875    * inv2 * inv2 * inv2
                + 6.5625   * inv2 * inv2 * inv2 * inv2;
    const double inv_sqrt_pi = 0.5641895835477563; // 1/√π
    return inv * poly * inv_sqrt_pi;
}

extern "C" __device__ __forceinline__ double log_ndtr(double x) {
    if (x ==  (1.0 / 0.0)) return 0.0;
    if (x == -(1.0 / 0.0)) return -(1.0 / 0.0);
    if (isnan(x)) return x;
    if (x < 0.0) {
        double u   = -x / SQRT_2;
        double ex  = erfcx_nonnegative(u);
        if (ex < 1e-300) ex = 1e-300;
        return -u * u + log(0.5 * ex);
    } else {
        double c = 0.5 * erfc(-x / SQRT_2);
        if (c < 1e-300) c = 1e-300;
        if (c > 1.0)    c = 1.0;
        return log(c);
    }
}

// Returns (log Φ(x), φ(x)/Φ(x)).
extern "C" __device__ __forceinline__ void
log_ndtr_and_mills(double x, double *log_cdf, double *lambda) {
    if (x ==  (1.0 / 0.0)) { *log_cdf = 0.0;            *lambda = 0.0;            return; }
    if (x == -(1.0 / 0.0)) { *log_cdf = -(1.0 / 0.0);   *lambda = (1.0 / 0.0);    return; }
    if (isnan(x))          { *log_cdf = x;              *lambda = x;              return; }
    if (x < 0.0) {
        double u   = -x / SQRT_2;
        double ex  = erfcx_nonnegative(u);
        if (ex < 1e-300) ex = 1e-300;
        *log_cdf = -u * u + log(0.5 * ex);
        const double sqrt_2_over_pi = 0.7978845608028654; // √(2/π)
        *lambda  = sqrt_2_over_pi / ex;
    } else {
        double cdf = 0.5 * erfc(-x / SQRT_2);
        if (cdf < 1e-300) cdf = 1e-300;
        if (cdf > 1.0)    cdf = 1.0;
        double pdf = INV_SQRT_2PI * exp(-0.5 * x * x);
        *log_cdf = log(cdf);
        *lambda  = pdf / cdf;
    }
}

#endif // PROBIT_NUMERICS_INCLUDED
"#;
