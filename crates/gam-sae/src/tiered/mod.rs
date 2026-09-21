//! #2023 tiered SAE spine.
//!
//! **The "tiers" are an OPTIMIZER SCHEDULE, not a model structure.** There is ONE
//! model: an intercept `μ` (the fixed effect — [`Tier0Mean`]) plus a sparse
//! dictionary of complexity-priced 1-D atoms, where a *linear* atom is just the
//! rank-1 / `b₂=0` special case of the curved (trig) atom and the rank-charge
//! criterion (`Δloss > ½·d_eff·log n`, MP floor) selects each atom's complexity.
//! The dictionary width `K` is engineered CAPACITY; the number of *certified*
//! atoms is an evidence **output** of that criterion, never a hand-set width
//! target (no PCA-energy cutoff sets it). "Migration" between linear and curved
//! is therefore not a statistical border — it is per-atom rank re-selection under
//! the same criterion.
//!
//! This module owns the coordinate-descent machinery + fixed effect for fitting
//! that one model at large `K`:
//! - [`Tier0Mean`] — the shared intercept `μ` (schedule stage "Tier-0").
//! - a linear sparse-dictionary bulk (schedule stage "Tier-1") whose atoms are
//!   the criterion-selected rank-1 special case.
//! - an evidence-selected curved refinement (schedule stage "Tier-2") fit on the
//!   RAW Tier-1 residual — higher per-atom complexity where the criterion pays
//!   for it. (No projector weight: curvature lives INSIDE the linear span, so a
//!   `Q⊥` whitener would blind the fit; anti-rechasing is the criterion's job.)
//!
//! This module owns the **spine-level** types every schedule stage hangs off:
//!   * [`Tier0Mean`] — the single shared mean μ. Moving the DC out of every atom
//!     into ONE Tier-0 mean is the structural kill of the co-collapse-to-mean
//!     class (issue #10 / #1893): on the de-meaned data the all-atoms-equal-to-
//!     mean state reconstructs zero, so it is EV-invisible and gets pruned rather
//!     than rewarded and PC-reseeded.
//!
//! The Mode-A per-block scale-out (one K=1 curved chart per orthonormal Tier-1
//! block) consumes the block frames on the block-sparse fit directly; see
//! `sparse_dict::block`.

mod code_space;
mod fit;
mod promotion;
pub use code_space::{
    CensusPairVerdict, CodeSpacePromotionReport, PairChartFit, fit_pair_chart,
    harvest_code_space_pair_promotions, harvest_code_space_promotions, linear_distortion_floor,
};
pub use promotion::{InstalledChart, PromotionInstall, PromotionRefusal, PromotionRefusalReason};
pub use fit::{
    LinearPeel, LinearPeelConfig, TieredFitConfig, TieredFitReport, fit_tiered, linear_bulk_census,
};

use ndarray::{Array1, Array2, ArrayView2, Axis};

/// Tier-0: the single shared mean μ (length `p`). The global DC lives here, not
/// duplicated across `K` per-atom intercepts.
#[derive(Clone, Debug)]
pub struct Tier0Mean {
    /// The shared mean, length `p`.
    pub mean: Array1<f64>,
}

impl Tier0Mean {
    /// Fit Tier-0 as the column mean of `z` (`N×P`). This is the train-split mean;
    /// hold it fixed and reuse it for out-of-sample de-meaning and for the EV
    /// baseline so held-out EV is measured against the same Tier-0 constant.
    pub fn fit(z: ArrayView2<'_, f64>) -> Result<Self, String> {
        if z.nrows() == 0 || z.ncols() == 0 {
            return Err("Tier0Mean::fit requires a non-empty (N, P) matrix".to_string());
        }
        let mean = z
            .mean_axis(Axis(0))
            .ok_or_else(|| "Tier0Mean::fit: mean_axis returned None".to_string())?;
        Ok(Self { mean })
    }

    /// De-mean: `R0 = z − μ` (row-broadcast). The Tier-1 bulk is fit on this.
    pub fn apply(&self, z: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        if z.ncols() != self.mean.len() {
            return Err(format!(
                "Tier0Mean::apply: z has P={} but μ has length {}",
                z.ncols(),
                self.mean.len()
            ));
        }
        Ok(&z - &self.mean.view().insert_axis(Axis(0)))
    }

    /// Add μ back to a de-meaned reconstruction (`recon + μ`), row-broadcast.
    pub fn reconstruct(&self, recon: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        if recon.ncols() != self.mean.len() {
            return Err(format!(
                "Tier0Mean::reconstruct: recon has P={} but μ has length {}",
                recon.ncols(),
                self.mean.len()
            ));
        }
        Ok(&recon + &self.mean.view().insert_axis(Axis(0)))
    }
}

/// `1 − RSS/TSS` — the one definition of explained variance in the SAE stack.
///
/// Returns `NaN` when `tss` is not positive. With no variance to explain the
/// ratio is genuinely undefined, and `NaN` says so; a reporting surface that
/// wants to show `0.0` for a constant target has to substitute it deliberately
/// at the point of display rather than inherit the choice by accident.
///
/// This exists because the same three-line policy was written out four times
/// across the SAE stack, and the copies had already drifted on exactly that
/// degenerate case — two returned `0.0`, one returned `NaN`, for the same
/// question about the same quantity.
pub(crate) fn explained_variance_from_sums(rss: f64, tss: f64) -> f64 {
    if tss > 0.0 { 1.0 - rss / tss } else { f64::NAN }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn tier0_mean_roundtrips() {
        let z = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let t0 = Tier0Mean::fit(z.view()).expect("fit");
        assert!((t0.mean[0] - 3.0).abs() < 1e-12);
        assert!((t0.mean[1] - 4.0).abs() < 1e-12);
        let demeaned = t0.apply(z.view()).expect("apply");
        // Column means of the de-meaned data are ~0.
        let cm = demeaned.mean_axis(Axis(0)).unwrap();
        assert!(cm[0].abs() < 1e-12 && cm[1].abs() < 1e-12);
        // reconstruct(apply(z)) == z.
        let back = t0.reconstruct(demeaned.view()).expect("reconstruct");
        for (a, b) in back.iter().zip(z.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn explained_variance_is_undefined_when_there_is_no_variance_to_explain() {
        // The shared policy answers "undefined", not "explains nothing" — the
        // two are different claims and the reporting surface has to pick.
        assert!(explained_variance_from_sums(0.0, 0.0).is_nan());
        assert!(explained_variance_from_sums(1.0, 0.0).is_nan());
        assert!(explained_variance_from_sums(0.0, -1.0).is_nan());
    }

    #[test]
    fn explained_variance_propagates_a_non_finite_residual() {
        // A NaN arriving through RSS means the fit went non-finite. That must
        // survive as NaN: reporting surfaces substitute 0.0 for the UNDEFINED
        // case (tss = 0), and if they keyed that substitution off `is_nan()`
        // instead they would disguise a broken fit as a merely useless one.
        assert!(explained_variance_from_sums(f64::NAN, 1.0).is_nan());
        assert!(explained_variance_from_sums(f64::INFINITY, 1.0).is_infinite());
    }

    #[test]
    fn explained_variance_recovers_a_known_fraction() {
        // Ground truth rather than self-consistency: a reconstruction that
        // leaves exactly a quarter of the centered energy must score 0.75.
        assert!((explained_variance_from_sums(0.25, 1.0) - 0.75).abs() < 1e-15);
        // A reconstruction no better than the mean scores 0; one worse than the
        // mean scores negative, which is meaningful and must not be clamped.
        assert!((explained_variance_from_sums(1.0, 1.0) - 0.0).abs() < 1e-15);
        assert!(explained_variance_from_sums(2.0, 1.0) < 0.0);
    }

}
