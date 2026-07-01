use super::*;
use approx::assert_abs_diff_eq;

/// The overflow-free von-Mises normaliser must (a) agree with the naive
/// `bessel_i0(η).ln()` / `bessel_i1(η)/bessel_i0(η)` on moderate η where the
/// naive form is still finite, and (b) stay finite for the large η a
/// dispersion-inflated ARD seed reaches on a large-norm / ill-conditioned
/// checkpoint (#1113), where the naive form overflows to `inf` and divides
/// to `NaN`.
#[test]
pub(crate) fn bessel_log_and_ratio_is_finite_and_matches_naive() {
    // Moderate η: naive forms are finite, so the stable helper must match.
    for &eta in &[0.0_f64, 0.5, 1.0, 3.0, 3.75, 5.0, 20.0, 100.0, 300.0] {
        let (log_i0, ratio) = bessel_i0_log_and_ratio(eta);
        let naive_log = bessel_i0(eta).ln();
        let naive_ratio = bessel_i1(eta) / bessel_i0(eta);
        assert!(naive_log.is_finite(), "naive log finite at η={eta}");
        assert!(naive_ratio.is_finite(), "naive ratio finite at η={eta}");
        assert_abs_diff_eq!(log_i0, naive_log, epsilon = 1e-9);
        assert_abs_diff_eq!(ratio, naive_ratio, epsilon = 1e-9);
    }

    // Large η (past the `e^{η}` overflow threshold ≈ 709). The stable helper
    // must stay finite where `bessel_i0(η) = inf`, and the ratio I1/I0 → 1⁻.
    for &eta in &[710.0_f64, 1.0e3, 1.0e6, 1.0e12, 1.0e300] {
        assert!(
            !bessel_i0(eta).is_finite(),
            "naive I0 expected to overflow at η={eta} (guards the regression)"
        );
        let (log_i0, ratio) = bessel_i0_log_and_ratio(eta);
        assert!(log_i0.is_finite(), "stable log I0 finite at η={eta}");
        assert!(ratio.is_finite(), "stable I1/I0 finite at η={eta}");
        // I1/I0 ∈ (0, 1) and → 1 as η → ∞; the ρ-gradient term n·η·(ratio−1)
        // must therefore be finite, never `inf·NaN`.
        assert!(ratio > 0.0 && ratio <= 1.0, "ratio in (0,1] at η={eta}");
    }
}
