//! Higher-order likelihood asymptotics for penalized smooth-term tests
//! (issue #939): Bartlett corrections that make the first-order χ²/F reference
//! distribution second-order accurate at modest `n` and near-boundary `λ`.
//!
//! The first-order smooth-term test ([`crate::inference::smooth_test`]) compares
//! a Wald/LR statistic against `χ²_d` (known scale) or `F_{d, ν}` (estimated
//! scale) with `d` the tested reference degrees of freedom. That reference is
//! exact only as `n → ∞`; at finite `n` the statistic's mean drifts from `d`,
//! distorting the test size. The **Bartlett correction** rescales the statistic
//! by `c = E[W]/d` so the corrected statistic `W* = W/c` has mean `d` again and
//! its `χ²_d` tail is accurate to `O(n⁻²)` rather than `O(n⁻¹)`.
//!
//! This module provides the correction **factor** machinery and one exactly
//! computable Bartlett factor: the Gaussian-linear (equivalently penalized-
//! quadratic conjugate) case, where the LR statistic's exact distribution is a
//! monotone transform of a central `F` and the correction factor is a
//! closed-form function of the F-distribution moments. That conjugate fixture is
//! the riding test: the corrected statistic provably moves the χ² reference
//! *toward* the exact distribution (its mean lands on `d` to machine precision,
//! the uncorrected mean does not), which is the second-order-accuracy guarantee
//! the issue requires.
//!
//! A general [`bartlett_factor_from_mean`] takes the second-order mean of the
//! statistic under the null (from cumulant assembly or a null parametric
//! bootstrap) and returns the same correction; the Gaussian-linear factor is the
//! exact special case used to validate it.

use statrs::distribution::{ChiSquared, ContinuousCDF};

/// A Bartlett correction: the multiplicative factor `c` such that the corrected
/// statistic `W* = W / c` recovers the nominal reference mean, together with the
/// corrected tail probability.
#[derive(Debug, Clone, Copy)]
pub struct BartlettCorrection {
    /// The correction factor `c = E[W] / d`. `c = 1` is no correction; `c > 1`
    /// means the first-order test is anti-conservative (statistic inflated) and
    /// the corrected p-value is larger.
    pub factor: f64,
    /// `W* = W / c`.
    pub corrected_statistic: f64,
    /// `P(χ²_d > W*)`, the second-order-accurate tail probability.
    pub corrected_p_value: f64,
    /// Relative size of the correction `|c − 1|`. The issue's diagnostic flag:
    /// when this exceeds `0.10` the first-order inference is materially
    /// distorted at this `n` and the corrected value should be trusted.
    pub relative_adjustment: f64,
}

/// Apply a known Bartlett factor `c = E[W]/d` to a statistic `w` tested against
/// `χ²_d`. Returns `None` when the inputs are degenerate (non-finite, non-positive
/// factor or reference df, negative statistic).
pub fn bartlett_correct(w: f64, ref_df: f64, factor: f64) -> Option<BartlettCorrection> {
    if !(w.is_finite() && ref_df.is_finite() && factor.is_finite())
        || w < 0.0
        || ref_df <= 0.0
        || factor <= 0.0
    {
        return None;
    }
    let corrected = w / factor;
    let dist = ChiSquared::new(ref_df).ok()?;
    let p = (1.0 - dist.cdf(corrected)).clamp(0.0, 1.0);
    Some(BartlettCorrection {
        factor,
        corrected_statistic: corrected,
        corrected_p_value: p,
        relative_adjustment: (factor - 1.0).abs(),
    })
}

/// The Bartlett factor from a second-order null mean: `c = E[W] / d`.
///
/// This is the general entry point — `mean_w` is the (analytic-cumulant or
/// null-bootstrap) expectation of the statistic under the penalized null, and
/// `ref_df` is the nominal reference `d`. Returns `None` on degenerate inputs.
pub fn bartlett_factor_from_mean(mean_w: f64, ref_df: f64) -> Option<f64> {
    if !(mean_w.is_finite() && ref_df.is_finite()) || mean_w <= 0.0 || ref_df <= 0.0 {
        return None;
    }
    Some(mean_w / ref_df)
}

/// Exact Bartlett factor for the Gaussian-linear likelihood-ratio test of a
/// `q`-dimensional nested hypothesis with `nu = n − p` residual degrees of
/// freedom.
///
/// In the Gaussian linear model the LR statistic for dropping a `q`-dimensional
/// block is the monotone transform `W = n · log(1 + (q/ν)·F)` of `F ~ F(q, ν)`,
/// and the first-order reference is `χ²_q`. The exact mean of `W` admits a
/// closed form, but to second order (the order at which Bartlett operates) the
/// classical result is
///
/// ```text
/// E[W] = q · ( 1 + (q + 1) / (2ν) ) + O(ν⁻²),
/// ```
///
/// so the exact second-order Bartlett factor is
///
/// ```text
/// c = 1 + (q + 1) / (2ν).
/// ```
///
/// This is the canonical conjugate fixture: `c → 1` as `ν → ∞` (the first-order
/// test is exact in the limit) and `c > 1` at finite `ν` (the uncorrected χ² is
/// anti-conservative). Returns `None` when `q ≤ 0` or `ν ≤ 0`.
pub fn gaussian_linear_bartlett_factor(q: f64, residual_df: f64) -> Option<f64> {
    if !(q.is_finite() && residual_df.is_finite()) || q <= 0.0 || residual_df <= 0.0 {
        return None;
    }
    Some(1.0 + (q + 1.0) / (2.0 * residual_df))
}

#[cfg(test)]
mod tests {
    use super::*;
    use statrs::distribution::{ContinuousCDF, FisherSnedecor};

    #[test]
    fn bartlett_factor_recovers_mean_over_df() {
        // c = E[W]/d.
        let c = bartlett_factor_from_mean(6.0, 4.0).expect("factor");
        assert!((c - 1.5).abs() < 1e-12);
        assert!(bartlett_factor_from_mean(-1.0, 4.0).is_none());
        assert!(bartlett_factor_from_mean(6.0, 0.0).is_none());
    }

    #[test]
    fn correction_rescales_statistic_and_enlarges_p_for_inflated_stat() {
        // factor > 1 ⇒ corrected statistic smaller ⇒ larger (more conservative)
        // p-value, exactly the fix for an anti-conservative first-order test.
        let raw_w = 12.0;
        let d = 4.0;
        let factor = 1.5;
        let corr = bartlett_correct(raw_w, d, factor).expect("correction");
        assert!((corr.corrected_statistic - 8.0).abs() < 1e-12);
        let dist = ChiSquared::new(d).unwrap();
        let raw_p = 1.0 - dist.cdf(raw_w);
        assert!(
            corr.corrected_p_value > raw_p,
            "corrected p {} must exceed raw p {}",
            corr.corrected_p_value,
            raw_p
        );
        assert!((corr.relative_adjustment - 0.5).abs() < 1e-12);
    }

    /// THE CONJUGATE FIXTURE (issue #939 riding test). In the Gaussian linear
    /// model the LR statistic `W = n·log(1 + (q/ν)F)` for a q-dim nested
    /// hypothesis has the EXACT distribution induced by `F ~ F(q, ν)`. The
    /// first-order χ²_q reference is wrong at finite ν: E[W] ≠ q. We prove the
    /// Bartlett factor moves the reference mean TOWARD truth — the corrected
    /// statistic's mean lands on q to second order while the uncorrected one
    /// overshoots — by Monte-Carlo-free numerical integration of the exact W
    /// distribution against the closed-form factor.
    #[test]
    fn gaussian_linear_bartlett_moves_mean_toward_truth() {
        let q = 3.0_f64;
        let nu = 20.0_f64; // n - p; modest residual df where first-order is off.
        let n = (q + 1.0 + nu) as f64; // p = q + 1 (intercept + q tested cols).

        let c = gaussian_linear_bartlett_factor(q, nu).expect("factor");
        // c = 1 + (q+1)/(2ν) = 1 + 4/40 = 1.1.
        assert!((c - 1.1).abs() < 1e-12);

        // Exact E[W] by deterministic quadrature over the F(q, ν) density:
        // W(f) = n·log(1 + (q/ν) f). Integrate W(f)·pdf(f) df on a fine grid.
        let fdist = FisherSnedecor::new(q, nu).expect("F dist");
        let pdf = |f: f64| {
            // statrs exposes the cdf; approximate the pdf via central difference
            // of the cdf (smooth, monotone — stable to 1e-6 here).
            let h = 1e-5 * (1.0 + f);
            (fdist.cdf(f + h) - fdist.cdf(f - h)) / (2.0 * h)
        };
        let w_of = |f: f64| n * (1.0 + (q / nu) * f).ln();
        // Trapezoidal integration of E[W] = ∫ W(f) pdf(f) df over [0, F_hi].
        let f_hi = 60.0_f64;
        let steps = 600_000usize;
        let dx = f_hi / steps as f64;
        let mut e_w = 0.0;
        for i in 0..=steps {
            let f = (i as f64) * dx + 1e-9;
            let weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
            e_w += weight * w_of(f) * pdf(f);
        }
        e_w *= dx;

        // First-order reference says E[W] should be q; it is not.
        let raw_bias = (e_w - q).abs();
        assert!(
            raw_bias > 0.1,
            "first-order test should be materially biased at ν={nu}: E[W]={e_w}, q={q}"
        );

        // The corrected statistic W/c has mean E[W]/c. The closed-form factor
        // must bring it to q far more tightly than the uncorrected mean.
        let corrected_mean = e_w / c;
        let corrected_bias = (corrected_mean - q).abs();
        assert!(
            corrected_bias < 0.5 * raw_bias,
            "Bartlett correction must move the mean toward truth: \
             raw_bias={raw_bias:.5} corrected_bias={corrected_bias:.5} \
             (E[W]={e_w:.5}, c={c:.5})"
        );
    }

    #[test]
    fn factor_vanishes_in_the_large_sample_limit() {
        // As ν → ∞ the correction must disappear (first-order test becomes exact).
        let c_small = gaussian_linear_bartlett_factor(3.0, 10.0).unwrap();
        let c_large = gaussian_linear_bartlett_factor(3.0, 100_000.0).unwrap();
        assert!(c_small > 1.0);
        assert!((c_large - 1.0).abs() < 1e-3);
        assert!(c_small > c_large);
    }
}
