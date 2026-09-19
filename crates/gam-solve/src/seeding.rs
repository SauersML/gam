//! The outer search enters from one derived start
//! (`rho_optimizer::run_plan::outer_start_point`). This module holds the
//! validated ρ interval every start is clamped into and the one data-derived
//! start every REML path shares.

pub use gam_problem::OrderedRhoBounds;

use ndarray::ArrayView1;

/// Commensurate-curvature start for one smoothing coordinate (mgcv
/// `initial.sp`): `ρ = ln( Σ_{c ∈ support} (XᵀWX)_cc / tr(S) )`.
///
/// At this ρ the penalty `λ S` and the data curvature `XᵀWX` enter the
/// penalized Hessian at the same scale over the penalty's support, so the
/// start sits where the REML criterion trades fit against smoothness rather
/// than on either rail. Returns `None` when either trace is not strictly
/// positive and finite (no data support, or an empty penalty); the caller then
/// keeps its own anchor for that coordinate.
pub(crate) fn commensurate_curvature_rho(
    gram_diag: ArrayView1<'_, f64>,
    support: impl IntoIterator<Item = usize>,
    penalty_trace: f64,
) -> Option<f64> {
    let data_trace: f64 = support
        .into_iter()
        .filter_map(|c| gram_diag.get(c).copied())
        .sum();
    if !(penalty_trace > 0.0 && penalty_trace.is_finite() && data_trace > 0.0) {
        return None;
    }
    let rho = (data_trace / penalty_trace).ln();
    rho.is_finite().then_some(rho)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn commensurate_start_balances_data_and_penalty_traces_over_the_support() {
        let gram_diag = array![100.0, 3.0, 5.0, 1000.0];
        // Only the penalty's support (columns 1..3) enters the data trace.
        let rho = commensurate_curvature_rho(gram_diag.view(), 1..3, 2.0)
            .expect("positive traces give a start");
        assert_eq!(rho, (8.0_f64 / 2.0).ln());
        // At that ρ the penalty trace λ·tr(S) equals the supported data trace.
        assert!((rho.exp() * 2.0 - 8.0).abs() < 1e-12);
    }

    #[test]
    fn commensurate_start_declines_an_empty_or_degenerate_trace() {
        let gram_diag = array![1.0, 2.0];
        assert_eq!(commensurate_curvature_rho(gram_diag.view(), 0..2, 0.0), None);
        assert_eq!(commensurate_curvature_rho(gram_diag.view(), 0..2, f64::INFINITY), None);
        assert_eq!(commensurate_curvature_rho(gram_diag.view(), 0..0, 1.0), None);
        assert_eq!(commensurate_curvature_rho(array![0.0, 0.0].view(), 0..2, 1.0), None);
    }
}
