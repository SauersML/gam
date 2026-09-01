//! General exact orthogonal reparameterization of overlapping design blocks
//! (universal robustness — the "orthogonalize" stage).
//!
//! # Why this lives in the shared solver layer (not in a family)
//!
//! Several families build a linear predictor from two (or more) design blocks
//! whose column spans *overlap* by construction. The canonical case is the
//! Bernoulli/survival marginal-slope index
//!
//! ```text
//!     η(x) = M·β_m  +  diag(z)·S·β_s
//! ```
//!
//! where `M` is the marginal baseline surface and `S` is the score-weighted
//! ("slope") surface. Because the exposure `z` correlates with the same PC
//! smooths that both `M` and `S` are built from, a component of `M·β_m` can be
//! explained almost equally well by `diag(z)·S·β_s`. That structural confound
//! makes the *joint* design rank-soft: the inner Newton sees a near-singular
//! cross-block Hessian and the outer REML never settles.
//!
//! An earlier solver papered over this with pinned ridges (a penalty mass aimed
//! at the confounded direction, now deleted). That *penalizes* the confound.
//! This module instead *resolves* it by construction: it reparameterizes the
//! confound block so its
//! columns are **exactly** orthogonal (in a chosen row metric `W`) to the
//! primary block's column span. After the transform the cross-block Gram is
//! exactly zero, so no ridge is needed for identification — and the transform is
//! a pure change of basis, so the original-basis coefficients are recovered
//! **exactly** for prediction and reporting.
//!
//! The mechanism is family-general: it operates only on dense design columns and
//! a per-row weight vector, so any family that can hand over a `(primary,
//! confound, W)` triple inherits it. Activating it for BMS is fine, but nothing
//! here is BMS-specific.
//!
//! # The math (exact, no approximation)
//!
//! Let `M` (`n × p_m`) be the primary block, `C` (`n × p_c`) the confound block,
//! and `W = diag(w)` a non-negative row metric (`w_i ≥ 0`). Define the
//! W-projection coefficients
//!
//! ```text
//!     B = (MᵀW M + ε I)⁻¹ MᵀW C          (p_m × p_c)
//! ```
//!
//! and the orthogonalized confound design
//!
//! ```text
//!     C̃ = C − M·B.
//! ```
//!
//! Then `Mᵀ W C̃ = MᵀW C − (MᵀW M)·B = MᵀW C − MᵀW C = 0` (exactly, up to the ε
//! ridge that only acts when `MᵀW M` is rank-deficient), i.e. `C̃` is W-orthogonal
//! to `span(M)`.
//!
//! Crucially this is just a **shear** of the joint coefficient vector. The linear
//! predictor is invariant:
//!
//! ```text
//!     M·β̃_m + C̃·β_c = M·β̃_m + (C − M·B)·β_c
//!                    = M·(β̃_m − B·β_c) + C·β_c,
//! ```
//!
//! so if the solver fits `(β̃_m, β_c)` in the reparameterized basis, the
//! original-basis coefficients are recovered **exactly** by
//!
//! ```text
//!     β_m = β̃_m − B·β_c,      β_c (unchanged).
//! ```
//!
//! The confound coefficients are untouched; only the primary coefficients absorb
//! the shear `B·β_c`. [`OrthogonalReparam::recover_original`] performs exactly
//! this map, and [`OrthogonalReparam::reparameterized_confound`] returns `C̃`.
//!
//! Robustness is unconditional: the construction entry point
//! [`OrthogonalReparam::build_unconditional`] always builds the exact reparam.
//! The caller decides whether there is a confound block to orthogonalize.

use ndarray::{Array1, Array2, ArrayView2};

use faer::Side;
use gam_linalg::faer_ndarray::{
    FaerArrayView, factorize_symmetricwith_fallback, fast_ab, fast_xt_diag_x, fast_xt_diag_y,
};
use gam_linalg::matrix::FactorizedSystem;

/// Relative ridge (vs. the largest weighted primary-Gram diagonal) added to
/// `MᵀW M` before forming the projection coefficients `B`. It only regularizes a
/// rank-deficient primary design (a dropped/aliased column leaves a zero pivot)
/// and is negligible against a well-conditioned Gram, so the orthogonality
/// `MᵀW C̃ ≈ 0` holds to working precision. Matches the magnitude used by the
/// §3 influence projection so the two share a numerical regime.
pub const ORTHOGONAL_PROJECTION_RELATIVE_RIDGE: f64 = 1.0e-10;

/// An exact orthogonal reparameterization of one confound block against one
/// primary block's column span in a fixed row metric `W`.
///
/// Holds the shear matrix `B` (`p_m × p_c`) and the reparameterized confound
/// design `C̃ = C − M·B` (`n × p_c`). The transform is a pure change of basis, so
/// it is fully described by `B`; `C̃` is cached because the solver needs the new
/// design and recomputing it is wasteful.
///
/// Build with [`OrthogonalReparam::build_unconditional`]. The round-trip
/// [`recover_original`](Self::recover_original) maps fitted reparameterized
/// coefficients back to the original basis exactly.
#[derive(Debug, Clone)]
pub struct OrthogonalReparam {
    /// W-projection / shear matrix `B = (MᵀWM + εI)⁻¹ MᵀW C` (`p_m × p_c`).
    shear: Array2<f64>,
    /// Reparameterized confound design `C̃ = C − M·B` (`n × p_c`).
    confound_orthogonal: Array2<f64>,
}

impl OrthogonalReparam {

}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    /// Build a primary design `M` and a confound `C` that genuinely overlaps it
    /// (a couple of `C`'s columns are `M` columns plus small noise), and verify
    /// the W-orthogonality `MᵀW C̃ ≈ 0` holds to working precision.
    #[test]
    fn orthogonalized_confound_is_w_orthogonal_to_primary() {
        let n = 50;
        let mut m = Array2::<f64>::zeros((n, 3));
        let mut c = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 / n as f64;
            m[[i, 0]] = 1.0;
            m[[i, 1]] = t;
            m[[i, 2]] = (t * 6.0).sin();
            // C overlaps M: col0 ≈ M col1 (confound), col1 has a fresh direction.
            c[[i, 0]] = t + 0.01 * (t * 13.0).cos();
            c[[i, 1]] = (t * 3.0).cos();
        }
        let w = Array1::<f64>::from_elem(n, 1.0);
        let reparam = OrthogonalReparam::build_unconditional(m.view(), c.view(), &w)
            .expect("build should succeed");

        let c_tilde = reparam.reparameterized_confound().to_owned();
        // MᵀW C̃ should be ~0.
        let cross = fast_xt_diag_y(&m, &w, &c_tilde);
        let max_abs = cross.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(
            max_abs < 1e-8,
            "MᵀW C̃ not orthogonal: max |entry| = {max_abs:e}"
        );
    }

    /// EXACT round-trip: fit (synthetically) in the reparameterized basis, then
    /// recover original coefficients and confirm the predictor is identical.
    #[test]
    fn coefficient_round_trip_is_exact() {
        let n = 40;
        let mut m = Array2::<f64>::zeros((n, 2));
        let mut c = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 / n as f64;
            m[[i, 0]] = 1.0;
            m[[i, 1]] = (t * 4.0).sin();
            c[[i, 0]] = t; // overlaps the linear-ish part of M
            c[[i, 1]] = (t * 2.0).cos();
        }
        let w = Array1::<f64>::from_elem(n, 1.0);
        let reparam = OrthogonalReparam::build_unconditional(m.view(), c.view(), &w)
            .expect("build should succeed");

        // Pretend the solver returned these reparameterized-basis coefficients.
        let beta_m_reparam = Array1::from_vec(vec![0.7, -1.3]);
        let beta_c = Array1::from_vec(vec![2.1, 0.4]);

        // Predictor in the reparameterized basis: M·β̃_m + C̃·β_c.
        let c_tilde = reparam.reparameterized_confound().to_owned();
        let eta_reparam = m.dot(&beta_m_reparam) + c_tilde.dot(&beta_c);

        // Recover original coefficients and form the predictor in the ORIGINAL
        // basis: M·β_m + C·β_c. Must match to tight tolerance.
        let (beta_m, beta_c_out) = reparam
            .recover_original(&beta_m_reparam, &beta_c)
            .expect("recover should succeed");
        let eta_original = m.dot(&beta_m) + c.dot(&beta_c_out);

        let max_diff = (&eta_reparam - &eta_original)
            .iter()
            .fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(
            max_diff < 1e-10,
            "predictor changed under round-trip: max |Δη| = {max_diff:e}"
        );
        // Confound coefficients are untouched by the reparameterization.
        let cdiff = (&beta_c_out - &beta_c)
            .iter()
            .fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(cdiff == 0.0, "confound coeffs changed: {cdiff:e}");

        // Forward map is the exact inverse of recover_original.
        let back = reparam
            .to_reparameterized(&beta_m, &beta_c)
            .expect("forward should succeed");
        let fdiff = (&back - &beta_m_reparam)
            .iter()
            .fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(fdiff < 1e-10, "forward/inverse mismatch: {fdiff:e}");
    }

    /// When the confound does NOT overlap the primary span, predictions are
    /// unchanged AND the orthogonal design equals the raw confound (no shear),
    /// confirming the pass touches nothing it should not.
    #[test]
    fn absent_confound_leaves_design_and_predictions_unchanged() {
        let n = 30;
        // Primary spans constant + linear; confound is a pure quadratic deviation
        // built to be Euclidean-orthogonal to span{1, t} by centering.
        let mut m = Array2::<f64>::zeros((n, 2));
        let mut raw_quad = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            m[[i, 0]] = 1.0;
            m[[i, 1]] = t;
            raw_quad.push(t * t);
        }
        // Residualize the quadratic against {1, t} by hand so the confound column
        // is genuinely orthogonal to span(M) under W = I (the "confound absent"
        // regime). Use the very pass we are testing as the residualizer would be
        // circular; instead do an explicit least-squares residual.
        let w = Array1::<f64>::from_elem(n, 1.0);
        // Solve M b = quad in LS, residual = quad - M b is ⊥ span(M).
        let gram = fast_xt_diag_x(&m, &w);
        let quad = Array1::from_vec(raw_quad);
        let cross = m.t().dot(&quad);
        let gview = FaerArrayView::new(&gram);
        let factor = factorize_symmetricwith_fallback(gview.as_ref(), Side::Lower).expect("factor");
        let b = FactorizedSystem::solve(&factor, &cross).expect("solve");
        let resid = &quad - &m.dot(&b);
        let mut c = Array2::<f64>::zeros((n, 1));
        c.column_mut(0).assign(&resid);

        let reparam = OrthogonalReparam::build_unconditional(m.view(), c.view(), &w)
            .expect("build should succeed");
        // No overlap ⇒ shear ≈ 0 ⇒ C̃ ≈ C.
        let shear_max = reparam.shear().iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(shear_max < 1e-8, "expected ~zero shear, got {shear_max:e}");
        let c_tilde = reparam.reparameterized_confound().to_owned();
        let design_diff = (&c_tilde - &c).iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(
            design_diff < 1e-8,
            "orthogonalized design drifted from raw when confound absent: {design_diff:e}"
        );
    }

    /// Empty primary span ⇒ confound returned unchanged (nothing to project out).
    #[test]
    fn empty_primary_returns_raw_confound() {
        let n = 8;
        let m = Array2::<f64>::zeros((n, 0));
        let mut c = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            c[[i, 0]] = i as f64;
            c[[i, 1]] = 1.0;
        }
        let w = Array1::<f64>::from_elem(n, 1.0);
        let reparam = OrthogonalReparam::build_unconditional(m.view(), c.view(), &w)
            .expect("build should succeed");
        let c_tilde = reparam.reparameterized_confound().to_owned();
        let diff = (&c_tilde - &c).iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(diff == 0.0, "empty primary must return raw confound");
    }
}
