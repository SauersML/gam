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

// ── Host oracle for the shared device probit numerics (issue #1175) ──────────
//
// The functions below are the CPU-side, device-free mirror of the CUDA source
// in [`crate::numerics_device::PROBIT_NUMERICS_CU`]. They are written
// LINE-FOR-LINE against that kernel source — the SAME branch structure, the
// SAME asymptotic `erfcx` polynomial, and the SAME constants — differing only
// in that they call the host `libm`
// transcendentals (`erfc`/`exp`/`log`) where the kernel calls the device
// `erfc`/`exp`/`log`. Both sides are the SunOS *msun* double-precision
// implementations, so the host oracle matches the device to within ~1 ULP per
// transcendental (issue #1175 items 4–5). This mirrors the #1017
// `emulate_certified_encode_row` pattern: a CPU emulator that is BOTH the
// fallback and the exactness oracle a device launch is pinned to.
//
// Correctness *without a GPU* (CPU-verifiable): the test harness below asserts
// (a) these constants are bit-identical to the literals in the kernel source
// (the "constants cannot drift" lock, #1175 item 4), (b) the kernel source uses
// only msun transcendentals and no fast-math intrinsics (transcendental-parity
// intent), and (c) the host oracle satisfies the defining probit identities to
// a stated ULP bound. Confirming a *device launch* reproduces this oracle to
// round-off still needs CUDA hardware.

/// `1/√(2π)`, matching `INV_SQRT_2PI` in the kernel source bit-for-bit.
pub const INV_SQRT_2PI: f64 = 0.3989422804014327;
/// `√2`, matching `SQRT_2` in the kernel source bit-for-bit.
pub const SQRT_2: f64 = 1.4142135623730951;
/// `ln(2)`, matching `LN_2` in the kernel source bit-for-bit.
pub const LN_2: f64 = 0.6931471805599453;
/// `1/√π`, matching `inv_sqrt_pi` in the kernel source bit-for-bit.
pub const INV_SQRT_PI: f64 = 0.5641895835477563;
/// `√(2/π)`, matching `sqrt_2_over_pi` in the kernel source bit-for-bit.
pub const SQRT_2_OVER_PI: f64 = 0.7978845608028654;

/// Scaled complementary error function `erfcx(x) = exp(x²)·erfc(x)` for `x ≥ 0`,
/// the host oracle for the device `erfcx_nonnegative`. Returns `0.0` at `+∞`;
/// negative inputs and `NaN` return `NaN` because they violate the restricted
/// domain. For `0 ≤ x < 26` evaluates `exp(x²)·erfc(x)` directly; beyond that
/// it switches to the same six-correction asymptotic expansion as the kernel.
pub fn erfcx_nonnegative(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return 0.0;
    }
    if x < 26.0 {
        return libm::exp(x * x) * erfc(x);
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    let poly = 1.0
        + inv2
            * (-0.5
                + inv2
                    * (0.75
                        + inv2
                            * (-1.875
                                + inv2
                                    * (6.5625 + inv2 * (-29.53125 + inv2 * 162.421875)))));
    inv * poly * INV_SQRT_PI
}

/// `log Φ(x)` for the standard normal CDF, the host oracle for the device
/// `log_ndtr`. For `x < 0` uses the `erfcx` representation
/// `log Φ(x) = −u² + log(½·erfcx(u))`, `u = −x/√2`, keeping digits into the
/// deep left tail; for `x ≥ 0` uses `log1p(−½·erfc(x/√2))`, retaining the
/// negative tail after the CDF rounds to one. Propagates `±∞`/`NaN` exactly as
/// the device path does.
pub fn log_ndtr(x: f64) -> f64 {
    if x == f64::INFINITY {
        return 0.0;
    }
    if x == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if x.is_nan() {
        return x;
    }
    if x < 0.0 {
        let u = -x / SQRT_2;
        let ex = erfcx_nonnegative(u);
        -u * u + libm::log(ex) - LN_2
    } else {
        let upper_tail = 0.5 * erfc(x / SQRT_2);
        libm::log1p(-upper_tail)
    }
}

/// Joint `(log Φ(x), Mills ratio φ(x)/Φ(x))`, the host oracle for the device
/// `log_ndtr_and_mills`. The `x < 0` branch computes the Mills ratio as
/// `√(2/π)/erfcx(u)`, which stays finite even when `Φ(x)` underflows; the
/// `x ≥ 0` branch forms `pdf/cdf` directly. Boundary values mirror the kernel:
/// `(+0, +0)` at `+∞`, `(−∞, +∞)` at `−∞`, `(NaN, NaN)` at `NaN`.
pub fn log_ndtr_and_mills(x: f64) -> (f64, f64) {
    if x == f64::INFINITY {
        return (0.0, 0.0);
    }
    if x == f64::NEG_INFINITY {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    if x.is_nan() {
        return (x, x);
    }
    if x < 0.0 {
        let u = -x / SQRT_2;
        let ex = erfcx_nonnegative(u);
        let log_cdf = -u * u + libm::log(ex) - LN_2;
        let lambda = SQRT_2_OVER_PI / ex;
        (log_cdf, lambda)
    } else {
        let upper_tail = 0.5 * erfc(x / SQRT_2);
        let cdf = 1.0 - upper_tail;
        let pdf = INV_SQRT_2PI * libm::exp(-0.5 * x * x);
        let log_cdf = libm::log1p(-upper_tail);
        let lambda = pdf / cdf;
        (log_cdf, lambda)
    }
}

#[cfg(test)]
mod probit_parity_tests {
    //! CPU-verifiable floating-point-order & transcendental parity harness for
    //! the shared probit numerics (issue #1175). Everything here runs without a
    //! GPU: it pins the host oracle constants to the kernel-source literals,
    //! audits the kernel source for msun-only transcendentals (no fast-math),
    //! and checks the host oracle against the defining probit identities within
    //! stated ULP bounds. A *device* reproducing this oracle to round-off still
    //! requires CUDA hardware and is asserted by the on-device parity gates.
    use super::*;
    use crate::numerics_device::PROBIT_NUMERICS_CU;

    const EPS: f64 = f64::EPSILON; // 2.220446049250313e-16

    /// Relative error of `got` vs `want`, expressed in ULP of `want`.
    fn ulp(got: f64, want: f64) -> f64 {
        if want == 0.0 {
            (got - want).abs() / EPS
        } else {
            (got - want).abs() / (EPS * want.abs())
        }
    }

    /// Extract the first f64 literal appearing after `needle` in `src`.
    fn literal_after(src: &str, needle: &str) -> f64 {
        let start = src
            .find(needle)
            .unwrap_or_else(|| panic!("kernel source is missing marker {needle:?}"))
            + needle.len();
        let tail = &src[start..];
        // Skip separators between the marker and the number ('=', whitespace).
        let num_start = tail
            .find(|c: char| c == '-' || c == '.' || c.is_ascii_digit())
            .unwrap_or_else(|| panic!("no numeric literal follows {needle:?}"));
        let rest = &tail[num_start..];
        let end = rest
            .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-')))
            .unwrap_or(rest.len());
        rest[..end]
            .parse::<f64>()
            .unwrap_or_else(|e| panic!("failed to parse literal after {needle:?}: {e}"))
    }

    /// #1175 item 4 pattern ("constants cannot drift"): every constant the host
    /// oracle uses is bit-identical to the literal baked into the kernel source.
    /// A one-bit edit on either side fails this immediately.
    #[test]
    fn host_constants_match_kernel_source_bit_for_bit() {
        for (needle, host) in [
            ("#define INV_SQRT_2PI", INV_SQRT_2PI),
            ("#define SQRT_2", SQRT_2),
            ("#define LN_2", LN_2),
            ("inv_sqrt_pi =", INV_SQRT_PI),
            ("sqrt_2_over_pi =", SQRT_2_OVER_PI),
        ] {
            let device = literal_after(PROBIT_NUMERICS_CU, needle);
            assert_eq!(
                device.to_bits(),
                host.to_bits(),
                "constant {needle:?} drifted: kernel={device:?} host={host:?}"
            );
        }
    }

    /// Transcendental-parity intent: the kernel evaluates its transcendentals
    /// through the msun `erfc`/`exp`/`log` (which the host `libm` mirrors) and
    /// contains NO fast-math intrinsic or single-precision variant. FMA
    /// contraction is separately disabled at compile time via
    /// `device_cache`'s `--fmad=false`; this guards the source itself.
    #[test]
    fn kernel_source_uses_msun_transcendentals_only() {
        for good in ["erfc(", "exp(", "log(", "log1p("] {
            assert!(
                PROBIT_NUMERICS_CU.contains(good),
                "kernel source should call msun `{good}`"
            );
        }
        for bad in [
            "__expf",
            "__logf",
            "expf(",
            "logf(",
            "erfcf(",
            "__fdividef",
            "__frcp",
            "use_fast_math",
            "ffast-math",
            "__dmul_",
            "__dadd_",
            "__fmaf",
        ] {
            assert!(
                !PROBIT_NUMERICS_CU.contains(bad),
                "kernel source must not use fast-math / single-precision `{bad}`"
            );
        }
    }

    /// `erfc` boundary + symmetry: `erfc(0)=1` exactly and
    /// `erfc(-x) = 2 - erfc(x)` to ≤ 2 ULP across a moderate grid.
    #[test]
    fn erfc_boundary_and_symmetry() {
        assert_eq!(erfc(0.0), 1.0);
        let mut worst = 0.0_f64;
        for i in 0..300 {
            let x = i as f64 * 0.01;
            worst = worst.max(ulp(erfc(-x), 2.0 - erfc(x)));
        }
        assert!(worst <= 2.0, "erfc symmetry drift {worst:.3} ULP > 2");
    }

    /// Defining identity `erfcx(x)·exp(-x²) = erfc(x)` to ≤ 4 ULP for
    /// `0 < x < 26` (the direct branch of the host oracle).
    #[test]
    fn erfcx_matches_definition() {
        assert_eq!(erfcx_nonnegative(0.0), 1.0);
        assert!(erfcx_nonnegative(-3.0).is_nan());
        assert!(erfcx_nonnegative(f64::NEG_INFINITY).is_nan());
        assert!(erfcx_nonnegative(f64::NAN).is_nan());
        assert_eq!(erfcx_nonnegative(f64::INFINITY), 0.0);
        let mut worst = 0.0_f64;
        let mut x = 0.1;
        while x < 25.0 {
            worst = worst.max(ulp(erfcx_nonnegative(x) * libm::exp(-x * x), erfc(x)));
            x += 0.1;
        }
        assert!(worst <= 4.0, "erfcx definition drift {worst:.3} ULP > 4");
    }

    #[test]
    fn erfcx_asymptotic_switch_and_subnormal_contract_match_device_source() {
        let switch = 26.0_f64;
        let direct = libm::exp(switch * switch) * erfc(switch);
        assert!(
            (erfcx_nonnegative(switch) / direct - 1.0).abs() < 5.0e-14,
            "erfcx switch disagrees with direct finite identity"
        );
        let tail = erfcx_nonnegative(f64::MAX);
        assert!(tail > 0.0 && tail.is_subnormal(), "erfcx(MAX)={tail:e}");

        for required in [
            "isnan(x) || x < 0.0",
            "inv2 * 162.421875",
            "log1p(-upper_tail)",
        ] {
            assert!(
                PROBIT_NUMERICS_CU.contains(required),
                "device source lost shared tail contract `{required}`"
            );
        }
        for forbidden in [
            "1e-300",
            "if (xx > 700.0)",
            "if (cdf > 1.0)",
            "1.0 / 0.0",
        ] {
            assert!(
                !PROBIT_NUMERICS_CU.contains(forbidden),
                "device source reintroduced numerical projection `{forbidden}`"
            );
        }
    }

    /// `log_ndtr` boundary + bulk identity `log Φ(x) = log(½·erfc(-x/√2))` to
    /// ≤ 2 ULP for `|x| ≤ 3`, and `Φ(x)+Φ(-x)=1` to ≤ 4e-16.
    #[test]
    fn log_ndtr_matches_log_cdf_and_reflects() {
        assert_eq!(log_ndtr(0.0), libm::log(0.5));
        assert_eq!(log_ndtr(f64::INFINITY), 0.0);
        assert_eq!(log_ndtr(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert!(log_ndtr(f64::NAN).is_nan());
        assert!(log_ndtr(10.0) < 0.0);

        let mut worst_bulk = 0.0_f64;
        for i in -30..=30 {
            let x = i as f64 * 0.1;
            let cdf = 0.5 * erfc(-x / SQRT_2);
            worst_bulk = worst_bulk.max(ulp(log_ndtr(x), libm::log(cdf)));
        }
        assert!(
            worst_bulk <= 2.0,
            "log_ndtr vs log-cdf drift {worst_bulk:.3} ULP > 2"
        );

        let mut worst_refl = 0.0_f64;
        for i in 0..60 {
            let x = i as f64 * 0.1;
            let s = libm::exp(log_ndtr(x)) + libm::exp(log_ndtr(-x));
            worst_refl = worst_refl.max((s - 1.0).abs());
        }
        assert!(
            worst_refl <= 4e-16,
            "Φ(x)+Φ(-x) reflection drift {worst_refl:e} > 4e-16"
        );
    }

    /// `log_ndtr_and_mills` agrees with `log_ndtr` on the log-CDF channel and
    /// satisfies the Mills identity `λ(x)·Φ(x) = φ(x)` to ≤ 32 ULP for
    /// `|x| ≤ 5`; the deep left tail stays finite (no `-∞`/`NaN`).
    #[test]
    fn log_ndtr_and_mills_identity_and_deep_tail() {
        for i in -50..=50 {
            let x = i as f64 * 0.1;
            let (log_cdf, lambda) = log_ndtr_and_mills(x);
            assert_eq!(
                log_cdf.to_bits(),
                log_ndtr(x).to_bits(),
                "joint log-CDF channel diverged from log_ndtr at x={x}"
            );
            let phi = libm::exp(log_cdf);
            let pdf = INV_SQRT_2PI * libm::exp(-0.5 * x * x);
            assert!(
                ulp(lambda * phi, pdf) <= 32.0,
                "Mills identity drift {:.3} ULP > 32 at x={x}",
                ulp(lambda * phi, pdf)
            );
        }
        for &x in &[-10.0, -20.0, -30.0, -38.0] {
            let (log_cdf, lambda) = log_ndtr_and_mills(x);
            assert!(
                log_cdf.is_finite() && log_cdf < 0.0,
                "deep-tail log Φ({x}) not finite-negative: {log_cdf}"
            );
            assert!(
                lambda.is_finite() && lambda > x.abs() * 0.9,
                "deep-tail Mills({x}) should track |x|: {lambda}"
            );
        }
        assert_eq!(log_ndtr_and_mills(f64::INFINITY), (0.0, 0.0));
        assert_eq!(
            log_ndtr_and_mills(f64::NEG_INFINITY),
            (f64::NEG_INFINITY, f64::INFINITY)
        );
    }
}
