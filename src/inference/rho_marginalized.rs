//! ρ-marginalized predictive variance: propagate smoothing-parameter
//! uncertainty into prediction intervals.
//!
//! Standard GAM/GAMLSS posterior bands are *conditional on the fitted
//! smoothing parameters* `ρ̂`: they report `Var[f(x) | ρ̂] = xᵀ Σ_β x`. That
//! understates uncertainty for smooths, adaptive penalties, flexible links and
//! survival surfaces, because `ρ̂` is itself estimated. A second-order
//! correction adds the variance that flows through `ρ̂`:
//!
//! ```text
//! Var[f(x)] ≈ xᵀ Σ_β x  +  (∂f/∂ρ)ᵀ V_ρ (∂f/∂ρ),
//! ```
//!
//! where `V_ρ` is the covariance of `ρ̂` (the inverse outer REML/LAML Hessian)
//! and the sensitivity of the prediction to the smoothing parameters comes from
//! the implicit-function derivative of the penalized fit,
//!
//! ```text
//! ∂β/∂ρ_k = −H_ββ⁻¹ (∂H_ββ/∂ρ_k) β = −λ_k H_ββ⁻¹ S_k β,
//! ∂f/∂ρ_k = xᵀ (∂β/∂ρ_k),
//! ```
//!
//! with `H_ββ = XᵀWX + Σ_k λ_k S_k` the penalized coefficient Hessian and `S_k`
//! the (full-coefficient-space embedded) penalty of block `k`. This is the
//! Kass–Steffey / Wood–Pya–Säfken smoothing-parameter-uncertainty correction;
//! it generally improves interval coverage more than any cosmetic change.
//!
//! These functions take the already-computed `H_ββ⁻¹`, penalties, `β` and `V_ρ`
//! as inputs so they stay decoupled from the specific solver that produced
//! them (closed-form Gaussian REML, PIRLS+LAML, survival, …) and are exercised
//! directly by unit tests.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// `∂β/∂ρ_k = −λ_k H_ββ⁻¹ S_k β` — the sensitivity of the penalized
/// coefficients to the log smoothing parameter of one penalty block.
///
/// `h_beta_inv` is `H_ββ⁻¹` (`p×p`), `penalty` is the full-coefficient-space
/// embedded `S_k` (`p×p`, zero outside block `k`), `beta` is the fitted `β`
/// (`p`), and `lambda` is `λ_k = e^{ρ_k}`.
pub fn dbeta_drho(
    h_beta_inv: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    beta: ArrayView1<'_, f64>,
    lambda: f64,
) -> Array1<f64> {
    let s_beta = penalty.dot(&beta);
    h_beta_inv.dot(&s_beta).mapv(|v| -lambda * v)
}

/// `∂f/∂ρ = (∂β/∂ρ)ᵀ x` for a single prediction row `x`, given the stacked
/// `∂β/∂ρ` columns (`p×m`, column `k` is `∂β/∂ρ_k`). Returns the length-`m`
/// gradient of the linear predictor with respect to the log smoothing
/// parameters.
pub fn df_drho(x_row: ArrayView1<'_, f64>, dbeta_drho_cols: ArrayView2<'_, f64>) -> Array1<f64> {
    dbeta_drho_cols.t().dot(&x_row)
}

/// ρ-marginalized predictive variance for a single point:
/// `conditional_var + (∂f/∂ρ)ᵀ V_ρ (∂f/∂ρ)`.
///
/// `conditional_var` is `xᵀ Σ_β x` (the usual band variance at fixed `ρ̂`),
/// `df_drho_vec` is `∂f/∂ρ` (length `m`), and `rho_cov` is `V_ρ` (`m×m`,
/// the inverse outer REML Hessian). The added term is a non-negative
/// quadratic form whenever `V_ρ` is PSD, so the marginalized variance never
/// drops below the conditional one.
pub fn rho_marginalized_variance(
    conditional_var: f64,
    df_drho_vec: ArrayView1<'_, f64>,
    rho_cov: ArrayView2<'_, f64>,
) -> f64 {
    let correction = df_drho_vec.dot(&rho_cov.dot(&df_drho_vec));
    conditional_var + correction
}

/// Batched [`rho_marginalized_variance`]: `conditional_vars` length `npred`,
/// `df_drho_rows` shape `npred×m` (row `i` is `∂f_i/∂ρ`), `rho_cov` `m×m`.
pub fn rho_marginalized_variances(
    conditional_vars: ArrayView1<'_, f64>,
    df_drho_rows: ArrayView2<'_, f64>,
    rho_cov: ArrayView2<'_, f64>,
) -> Array1<f64> {
    let npred = conditional_vars.len();
    let mut out = Array1::<f64>::zeros(npred);
    for i in 0..npred {
        let g = df_drho_rows.row(i);
        out[i] = conditional_vars[i] + g.dot(&rho_cov.dot(&g));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    // Diagonal H_ββ = diag(1 + λ s_i) (orthonormal design, XᵀWX = I; diagonal
    // penalty S = diag(s)). Then H⁻¹ and β are closed-form, so we can check the
    // analytic ∂β/∂ρ against a central finite difference with no linear-algebra
    // dependency.
    fn diag_setup(lambda: f64) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
        let s = [0.5_f64, 2.0, 0.0, 4.0];
        let b = [1.3_f64, -0.7, 2.1, 0.4]; // XᵀWy
        let p = s.len();
        let mut penalty = Array2::<f64>::zeros((p, p));
        let mut h_inv = Array2::<f64>::zeros((p, p));
        let mut beta = Array1::<f64>::zeros(p);
        let b_arr = Array1::from(b.to_vec());
        for i in 0..p {
            penalty[[i, i]] = s[i];
            let h_ii = 1.0 + lambda * s[i];
            h_inv[[i, i]] = 1.0 / h_ii;
            beta[i] = b_arr[i] / h_ii;
        }
        (penalty, h_inv, beta, b_arr)
    }

    fn beta_at_rho(rho: f64, s: &[f64], b: &[f64]) -> Array1<f64> {
        let lambda = rho.exp();
        Array1::from_iter(
            s.iter()
                .zip(b.iter())
                .map(|(&si, &bi)| bi / (1.0 + lambda * si)),
        )
    }

    #[test]
    fn dbeta_drho_matches_central_finite_difference() {
        let rho = 0.37_f64;
        let lambda = rho.exp();
        let (penalty, h_inv, beta, _b) = diag_setup(lambda);
        let analytic = dbeta_drho(h_inv.view(), penalty.view(), beta.view(), lambda);

        let s = [0.5_f64, 2.0, 0.0, 4.0];
        let b = [1.3_f64, -0.7, 2.1, 0.4];
        let h = 1e-6;
        let plus = beta_at_rho(rho + h, &s, &b);
        let minus = beta_at_rho(rho - h, &s, &b);
        let fd = (&plus - &minus).mapv(|v| v / (2.0 * h));

        let max_err = (&analytic - &fd)
            .mapv(f64::abs)
            .fold(0.0_f64, |acc, &v| acc.max(v));
        assert!(max_err < 1e-6, "max |analytic - fd| = {max_err}");
    }

    #[test]
    fn df_drho_is_x_dot_dbeta() {
        // Two smoothing params -> two ∂β/∂ρ columns.
        let p = 3;
        let mut cols = Array2::<f64>::zeros((p, 2));
        cols[[0, 0]] = 1.0;
        cols[[1, 0]] = -2.0;
        cols[[2, 0]] = 0.5;
        cols[[0, 1]] = 0.3;
        cols[[1, 1]] = 0.4;
        cols[[2, 1]] = -1.0;
        let x = Array1::from(vec![2.0_f64, 1.0, -1.0]);
        let g = df_drho(x.view(), cols.view());
        assert_eq!(g.len(), 2);
        assert!((g[0] - (2.0 * 1.0 + 1.0 * -2.0 + -1.0 * 0.5)).abs() < 1e-12);
        assert!((g[1] - (2.0 * 0.3 + 1.0 * 0.4 + -1.0 * -1.0)).abs() < 1e-12);
    }

    #[test]
    fn marginalized_variance_adds_nonnegative_quadratic_form() {
        // V_ρ PSD (built as LᵀL) ⇒ correction ≥ 0 and total ≥ conditional.
        let l = Array2::from_shape_vec((2, 2), vec![0.8, 0.0, -0.3, 0.5]).unwrap();
        let rho_cov = l.t().dot(&l);
        let g = Array1::from(vec![1.2_f64, -0.7]);
        let cond = 0.25_f64;
        let total = rho_marginalized_variance(cond, g.view(), rho_cov.view());
        assert!(total >= cond - 1e-15, "total {total} < conditional {cond}");
        // exact value: cond + gᵀ V g
        let expect = cond + g.dot(&rho_cov.dot(&g));
        assert!((total - expect).abs() < 1e-12);
    }

    #[test]
    fn batched_matches_scalar_per_row() {
        let rho_cov = Array2::from_shape_vec((2, 2), vec![1.0, 0.2, 0.2, 0.7]).unwrap();
        let conditional = Array1::from(vec![0.1_f64, 0.3, 0.05]);
        let rows =
            Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.5, -0.5, -1.0, 2.0]).unwrap();
        let batched = rho_marginalized_variances(conditional.view(), rows.view(), rho_cov.view());
        for i in 0..3 {
            let scalar =
                rho_marginalized_variance(conditional[i], rows.row(i), rho_cov.view());
            assert!((batched[i] - scalar).abs() < 1e-12);
        }
    }

    #[test]
    fn zero_smoothing_uncertainty_recovers_conditional_variance() {
        let rho_cov = Array2::<f64>::zeros((2, 2));
        let g = Array1::from(vec![3.0_f64, -4.0]);
        let total = rho_marginalized_variance(0.42, g.view(), rho_cov.view());
        assert!((total - 0.42).abs() < 1e-15);
    }
}
