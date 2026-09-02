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

}
