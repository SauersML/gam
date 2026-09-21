//! The one producer of a fit's Tierney-Kadane null-space normalizer metadata
//! (`FitArtifacts::null_space_dim` and `FitArtifacts::null_space_logdet`, #2627).
//!
//! [`crate::topology_selector::comparable_reml_score`] adds
//! `½·log|H_null| − ½·q·log 2π` to a fit's raw criterion. Every route that
//! publishes a `reml_score` calls this function where it forms its artifacts,
//! so the normalizer's dimension and determinant come from one computation.

use crate::model_types::UnifiedFitResult;
use faer::Side;
use gam_linalg::faer_ndarray::FaerCholesky;
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array2, s};

/// `(q, log|C'HC|)` over the raw penalty null space of one fit.
///
/// `C` spans the null space of the summed realized penalty blocks, the
/// directions the criterion's `½ log|S|₊` omits. A fit with no penalty block
/// leaves every raw coefficient unpenalized, so `C` is the identity. `C` is
/// pulled back through the fit's coefficient gauge into the active frame of
/// `fit.penalized_hessian()`, where the restriction is formed.
pub fn null_space_normalizer_metadata(
    coefficient_dim: usize,
    penalties: &[BlockwisePenalty],
    fit: &UnifiedFitResult,
) -> Result<(usize, f64), String> {
    let hessian = fit
        .penalized_hessian()
        .ok_or_else(|| "null-space Hessian logdet requires fitted penalized Hessian".to_string())?;
    let hessian_dim = hessian.nrows();
    if hessian.ncols() != hessian_dim {
        return Err(format!(
            "null-space Hessian logdet requires a square Hessian, got {}x{}",
            hessian.nrows(),
            hessian.ncols()
        ));
    }
    let p = coefficient_dim;
    let null_basis = if penalties.is_empty() {
        // No penalty block touches any coefficient, so the whole raw space is the
        // penalty null space and the normalizer runs over every direction. This
        // used to return q = 0, which switched the normalizer off for exactly the
        // fits whose null space is largest: `y ~ 1` scored without the intercept
        // direction that `y ~ x` carries.
        Array2::<f64>::eye(p)
    } else {
        penalty_null_basis(penalties, p)?
    };
    let q = null_basis.ncols();
    if q == 0 {
        return Ok((0, 0.0));
    }

    // The saved Hessian lives in the active coordinates declared by the fit's
    // gauge, while `null_basis` is expressed in the design's raw coordinates.
    // Pull every raw null-space direction N back through the injective lift
    // `T`: solve `T C = N`, then restrict as `C' H_active C`. Treating a
    // rectangular active Hessian as if it were raw curvature was the hidden
    // identity-gauge assumption exposed by exact smoothing boundaries (#2623).
    let active_null_basis = if let Some(geometry) = fit.geometry.as_ref() {
        let gauge = &geometry.coefficient_gauge;
        if gauge.raw_total() != p || gauge.reduced_total() != hessian_dim {
            return Err(format!(
                "null-space Hessian logdet gauge mismatch: realized penalty topology has {p} raw columns, gauge \
                 maps {} raw from {} active coordinates, Hessian is {hessian_dim}x{hessian_dim}",
                gauge.raw_total(),
                gauge.reduced_total(),
            ));
        }
        let t = &gauge.t_full;
        let raw_gram = t.t().dot(t);
        let gram = (&raw_gram + &raw_gram.t().to_owned()) * 0.5;
        let chol = gram.cholesky(Side::Lower).map_err(|error| {
            format!(
                "null-space Hessian logdet coefficient gauge is not injective: {error}"
            )
        })?;
        let coordinates = chol.solve_mat(&t.t().dot(&null_basis));
        let residual = t.dot(&coordinates) - &null_basis;
        let residual_max = residual
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        let basis_max = null_basis
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let backward_error = residual_max / basis_max;
        let roundoff_limit = f64::EPSILON.sqrt() * p.max(hessian_dim).max(1) as f64;
        if !backward_error.is_finite() || backward_error > roundoff_limit {
            return Err(format!(
                "null-space Hessian logdet raw penalty null space is not contained in the \
                 fitted active gauge: relative residual {backward_error:.6e}, numerical limit \
                 {roundoff_limit:.6e}"
            ));
        }
        coordinates
    } else {
        if hessian_dim != p {
            return Err(format!(
                "null-space Hessian logdet design/Hessian mismatch without a coefficient \
                 gauge: design has {p} columns but Hessian is {hessian_dim}x{hessian_dim}"
            ));
        }
        null_basis
    };
    let projected = hessian.dot(&active_null_basis);
    let mut restricted = active_null_basis.t().dot(&projected);
    restricted = (&restricted + &restricted.t()) * 0.5;
    let chol = restricted
        .cholesky(Side::Lower)
        .map_err(|err| format!("null-space Hessian is not positive definite: {err}"))?;
    let logdet = 2.0 * chol.diag().iter().map(|value| value.ln()).sum::<f64>();
    if logdet.is_finite() {
        Ok((q, logdet))
    } else {
        Err(format!("null-space Hessian logdet is not finite: {logdet}"))
    }
}

/// A basis for the raw null space of the summed realized penalty blocks.
fn penalty_null_basis(penalties: &[BlockwisePenalty], p: usize) -> Result<Array2<f64>, String> {
    let mut penalty = Array2::<f64>::zeros((p, p));
    for (idx, block) in penalties.iter().enumerate() {
        let range = block.col_range.clone();
        if range.start > range.end
            || range.end > p
            || block.local.nrows() != range.len()
            || block.local.ncols() != range.len()
        {
            return Err(format!(
                "null-space Hessian logdet penalty {idx} shape mismatch: range {}..{}, local {}x{}, p={p}",
                range.start,
                range.end,
                block.local.nrows(),
                block.local.ncols()
            ));
        }
        penalty
            .slice_mut(s![range.clone(), range])
            .scaled_add(1.0, &block.local);
    }
    let (null_basis, _) = gam_linalg::faer_ndarray::rrqr_nullspace_basis(&penalty)
        .map_err(|err| format!("failed to compute penalty null-space basis: {err}"))?;
    Ok(null_basis)
}
