//! Higher-order likelihood asymptotics for likelihood-ratio tests (issue #939):
//! Bartlett corrections that make the first-order χ² reference distribution
//! second-order accurate at modest `n` and near-boundary `λ`.
//!
//! For a likelihood-ratio statistic `W` with first-order reference `χ²_d`, the
//! statistic's mean can drift from `d` at finite `n`, distorting the test size.
//! The **Bartlett correction** rescales the statistic by `c = E[W]/d` so the
//! corrected statistic `W* = W/c` has mean `d` again and its `χ²_d` tail is
//! accurate to `O(n⁻²)` rather than `O(n⁻¹)`.
//!
//! A general [`bartlett_factor_from_mean`] takes the second-order mean of the
//! statistic under the null (from cumulant assembly or a null parametric
//! bootstrap) and returns the correction factor.

/// The Bartlett factor from a second-order null mean: `c = E[W] / d`.
///
/// `mean_w` is the (analytic-cumulant or null-bootstrap) expectation of the
/// statistic under the penalized null, and `ref_df` is the nominal reference
/// `d`. Returns `None` on degenerate inputs.
///
/// The ratio is a rescale of the whole reference, so it is only the right
/// correction for a mean shift that scales with `d` — a fixed-λ shift taken at
/// the λ the reference was built at. A shift that survives `d → 0` (Lawley's
/// ρ̂-variation term) divided by `d` is unbounded; that one is an additive
/// location, see `crate::inference::lawley::LawleyLrCorrection`.
pub fn bartlett_factor_from_mean(mean_w: f64, ref_df: f64) -> Option<f64> {
    if !(mean_w.is_finite() && ref_df.is_finite()) || mean_w <= 0.0 || ref_df <= 0.0 {
        return None;
    }
    Some(mean_w / ref_df)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bartlett_factor_recovers_mean_over_df() {
        // c = E[W]/d.
        let c = bartlett_factor_from_mean(6.0, 4.0).expect("factor");
        assert!((c - 1.5).abs() < 1e-12);
        assert!(bartlett_factor_from_mean(-1.0, 4.0).is_none());
        assert!(bartlett_factor_from_mean(6.0, 0.0).is_none());
    }
}
