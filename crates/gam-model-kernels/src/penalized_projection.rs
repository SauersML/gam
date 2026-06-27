//! Penalized weighted least-squares projection.
//!
//! Family-agnostic helper that solves a single penalized WLS system
//! `(XᵀWX + Σ λ_k S_k + ridge) β = Xᵀ W (target_eta − offset)` through the
//! design operator's stabilized policy solve. Carved out of the gamlss family
//! stack under #1521 because it carries no gamlss-specific type — its inputs are
//! the lower-tier [`DesignMatrix`] / [`PenaltyMatrix`] / [`RidgePolicy`]
//! primitives — and is consumed across families (gamlss block warm starts and
//! the transformation-normal warm start). Error text is emitted as `String`
//! verbatim, byte-identical to the previous `GamlssError`-coerced output.

use gam_linalg::matrix::DesignMatrix;
use gam_linalg::types::RidgePolicy;
use gam_problem::PenaltyMatrix;
use ndarray::{Array1, Array2};

pub fn solve_penalizedweighted_projection(
    design: &DesignMatrix,
    offset: &Array1<f64>,
    target_eta: &Array1<f64>,
    weights: &Array1<f64>,
    penalties: &[PenaltyMatrix],
    log_lambdas: &Array1<f64>,
    ridge_floor: f64,
) -> Result<Array1<f64>, String> {
    let n = design.nrows();
    let p = design.ncols();
    if offset.len() != n || target_eta.len() != n || weights.len() != n {
        return Err("solve_penalizedweighted_projection dimension mismatch".to_string());
    }
    if penalties.len() != log_lambdas.len() {
        return Err(format!(
            "solve_penalizedweighted_projection lambda mismatch: penalties={}, log_lambdas={}",
            penalties.len(),
            log_lambdas.len()
        ));
    }

    let y_star = target_eta - offset;
    let xtwy = design.compute_xtwy(weights, &y_star)?;
    let mut penalty_system = if penalties.is_empty() {
        None
    } else {
        Some(Array2::<f64>::zeros((p, p)))
    };
    for (k, s) in penalties.iter().enumerate() {
        let lambda = log_lambdas[k].exp();
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(format!(
                "solve_penalizedweighted_projection encountered invalid lambda at index {k}: {}",
                log_lambdas[k]
            ));
        }
        if s.nrows() != p || s.ncols() != p {
            return Err(format!(
                "solve_penalizedweighted_projection penalty shape mismatch at index {k}: \
                 penalty is {}x{} but design has {} columns",
                s.nrows(),
                s.ncols(),
                p
            ));
        }
        if let Some(system) = penalty_system.as_mut() {
            s.add_scaled_to(lambda, system);
        }
    }

    let beta = design.solve_systemwith_policy(
        weights,
        &xtwy,
        penalty_system.as_ref(),
        ridge_floor.max(1e-12),
        RidgePolicy::explicit_stabilization_pospart(),
    )?;
    if beta.iter().any(|v| !v.is_finite()) {
        return Err(
            "solve_penalizedweighted_projection produced non-finite coefficients".to_string(),
        );
    }
    Ok(beta)
}
