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
//!   - `log_ndtr_mills_curvature(...)` — also returns `−d² log Φ(x)/dx²`

/// Device-side probit numerics injected at the top of every NVRTC kernel that
/// needs them.  Prepend this string to a kernel-specific body before passing to
/// `cudarc::nvrtc::compile_ptx` or `PtxModuleCache::get_or_compile`.
pub const PROBIT_NUMERICS_CU: &str = r#"
// -------- shared probit numerics -----------------------------------------
// All math in double precision; fast-math is disabled at compile time
// (see `device_cache`'s `--fmad=false`) and the source is kept free of any
// fast-math / single-precision intrinsic, guarded by the numerics_host tests.
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
#define LN_2         0.6931471805599453

extern "C" __device__ __forceinline__ double erfcx_nonnegative(double x) {
    if (isnan(x) || x < 0.0) return nan("");
    if (isinf(x)) return 0.0;
    if (x < 26.0) {
        return exp(x * x) * erfc(x);
    }
    // Six-correction asymptotic expansion of erfcx for large x. At x=26,
    // the first omitted term is below 2e-17 relative to the leading term.
    double inv  = 1.0 / x;
    double inv2 = inv * inv;
    double poly = 1.0
                + inv2 * (-0.5
                + inv2 * (0.75
                + inv2 * (-1.875
                + inv2 * (6.5625
                + inv2 * (-29.53125
                + inv2 * 162.421875)))));
    const double inv_sqrt_pi = 0.5641895835477563; // 1/√π
    return inv * poly * inv_sqrt_pi;
}

extern "C" __device__ __forceinline__ double log_ndtr(double x) {
    if (isnan(x)) return x;
    if (isinf(x)) return (x > 0.0) ? 0.0 : x;
    if (x < 0.0) {
        double u   = -x / SQRT_2;
        double ex  = erfcx_nonnegative(u);
        return -u * u + log(ex) - LN_2;
    } else {
        double upper_tail = 0.5 * erfc(x / SQRT_2);
        return log1p(-upper_tail);
    }
}

// Returns (log Φ(x), φ(x)/Φ(x)).
extern "C" __device__ __forceinline__ void
log_ndtr_and_mills(double x, double *log_cdf, double *lambda) {
    if (isnan(x))          { *log_cdf = x;              *lambda = x;              return; }
    if (isinf(x)) {
        if (x > 0.0) { *log_cdf = 0.0; *lambda = 0.0; }
        else         { *log_cdf = x;   *lambda = -x;  }
        return;
    }
    if (x < 0.0) {
        double u   = -x / SQRT_2;
        double ex  = erfcx_nonnegative(u);
        *log_cdf = -u * u + log(ex) - LN_2;
        const double sqrt_2_over_pi = 0.7978845608028654; // √(2/π)
        *lambda  = sqrt_2_over_pi / ex;
    } else {
        double upper_tail = 0.5 * erfc(x / SQRT_2);
        double cdf = 1.0 - upper_tail;
        double pdf = INV_SQRT_2PI * exp(-0.5 * x * x);
        *log_cdf = log1p(-upper_tail);
        *lambda  = pdf / cdf;
    }
}

// Joint log Φ(x), Mills ratio, and positive negated log-CDF curvature
// `-d² log Φ(x)/dx²`. The direct `lambda * (x + lambda)` spelling loses the
// unit left-tail limit when x and lambda cancel, so the deep tail differentiates
// the Laplace continued fraction used by the CPU kernel.
extern "C" __device__ __forceinline__ void
log_ndtr_mills_curvature(double x, double *log_cdf, double *lambda, double *curvature) {
    log_ndtr_and_mills(x, log_cdf, lambda);
    if (isnan(x)) { *curvature = x; return; }
    if (isinf(x)) { *curvature = (x > 0.0) ? 0.0 : 1.0; return; }
    if (x <= -4.0) {
        double t = -x;
        double q = 0.0;
        double q_first = 0.0;
        for (int n = 32; n >= 1; --n) {
            double denominator = t + q;
            double value = ((double)n) / denominator;
            q_first = -value * (1.0 + q_first) / denominator;
            q = value;
        }
        *curvature = 1.0 + q_first;
    } else {
        *curvature = *lambda * (x + *lambda);
    }
}

#endif // PROBIT_NUMERICS_INCLUDED
"#;
