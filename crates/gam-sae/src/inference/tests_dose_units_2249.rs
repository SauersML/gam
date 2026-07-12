//! #2249 unit-validation harness — pins the **units** of `RowMetric::fisher_mass`
//! against a Fisher information that is known in closed form, with no fit, no
//! LLM, and no [`crate::inference::steering`] machinery in the loop.
//!
//! The dose contract in `steering.rs` is `predicted_nats = 0.5 * fisher_mass(row,
//! δ)`. That equation is only trustworthy if `fisher_mass` really returns
//! `δᵀFδ` for the *true* output Fisher `F`, in the *same* nats a KL computation
//! would report. This test builds the one case where `F` is known exactly: a
//! categorical distribution over `K` tokens with logits equal to the `p`-dim
//! output vector (identity Jacobian), so the output-Fisher metric IS the
//! logit-space Fisher, no pullback involved.
//!
//! For `y ~ Categorical(softmax(z))`, the score is `s_i = 𝟙[y = i] − p_i`
//! (`p = softmax(z)`), and the Fisher is `F = Cov(s) = Cov(𝟙_y) = diag(p) − p pᵀ`
//! — the standard multinomial/categorical Fisher in the natural (logit)
//! parameterization. That covariance decomposes over the `K` possible outcomes
//! as `F = Σ_k p_k (e_k − p)(e_k − p)ᵀ`, a sum of `K` rank-1 terms — which is
//! exactly the low-rank factor layout [`RowMetric::output_fisher`] takes
//! (`U_n ∈ ℝ^{p×rank}` with `M_n = U_n U_nᵀ`), letting the metric be built with
//! **no fit and no harvest**, straight from the closed-form probabilities.
//!
//! Two independent checks:
//! 1. **Units identity**: the analytic quadratic form `0.5·δᵀFδ`, computed
//!    from the closed-form `diag(p) − ppᵀ` (a completely separate code path
//!    from the `U_n` factor construction), matches `0.5 · fisher_mass(0, δ)`
//!    read off the constructed [`RowMetric`] to `1e-10` relative.
//! 2. **Nats identity**: at small `‖δ‖`, the exact KL `KL(p ‖ softmax(z+δ))`
//!    summed directly over the `K` outcomes matches the quadratic prediction
//!    `0.5·δᵀFδ` to `<1%` relative — pinning that the "nats" `fisher_mass`
//!    reports are actually KL-nats, not merely a same-named quadratic form.

use gam_problem::RowMetric;
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// `softmax(z)` for a small fixed-size logit vector.
fn softmax(z: &[f64]) -> Vec<f64> {
    let max_z = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = z.iter().map(|&v| (v - max_z).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&v| v / sum).collect()
}

/// `KL(p ‖ q)` for two probability vectors of the same length.
fn kl(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| if pi > 0.0 { pi * (pi / qi).ln() } else { 0.0 })
        .sum()
}

/// Build the `RowMetric::output_fisher` factor stack for a single row from the
/// closed-form categorical Fisher decomposition `F = Σ_k p_k (e_k − p)(e_k − p)ᵀ`.
/// Column `k` is `u_k = √p_k · (e_k − p)`, so `U Uᵀ = Σ_k p_k (e_k−p)(e_k−p)ᵀ = F`
/// exactly — a construction independent of, and never touching, the
/// `diag(p) − ppᵀ` formula used as the reference in the assertions below.
fn categorical_fisher_row_metric(p_probs: &[f64]) -> RowMetric {
    let k = p_probs.len();
    // Row-major (n_rows=1, p*rank) with rank = k: U[i, c] at flat index i*k + c.
    let mut flat = vec![0.0_f64; k * k];
    for c in 0..k {
        let sqrt_pc = p_probs[c].sqrt();
        for i in 0..k {
            let e_ci = if i == c { 1.0 } else { 0.0 };
            flat[i * k + c] = sqrt_pc * (e_ci - p_probs[i]);
        }
    }
    let u = Array2::from_shape_vec((1, k * k), flat).unwrap();
    RowMetric::output_fisher(Arc::new(u), k, k).unwrap()
}

/// Closed-form `F = diag(p) − ppᵀ` applied as a quadratic form, computed
/// independently of the `U_n` factor construction above.
fn analytic_fisher_quad_form(p_probs: &[f64], delta: &[f64]) -> f64 {
    let k = p_probs.len();
    let mut acc = 0.0;
    for i in 0..k {
        for j in 0..k {
            let f_ij = if i == j {
                p_probs[i] - p_probs[i] * p_probs[j]
            } else {
                -p_probs[i] * p_probs[j]
            };
            acc += delta[i] * f_ij * delta[j];
        }
    }
    acc
}

#[test]
fn fisher_mass_units_match_analytic_quadratic_form() {
    let p_probs = [0.5_f64, 0.3, 0.2];
    let delta = [0.01_f64, -0.02, 0.005];

    let metric = categorical_fisher_row_metric(&p_probs);
    assert_eq!(metric.p_out(), 3);
    assert_eq!(metric.n_rows(), 1);

    let delta_arr = Array1::from_vec(delta.to_vec());
    let predicted_nats = 0.5 * metric.fisher_mass(0, delta_arr.view());
    let analytic_nats = 0.5 * analytic_fisher_quad_form(&p_probs, &delta);

    assert!(analytic_nats > 0.0, "sanity: quadratic form must be positive");
    let rel_err = (predicted_nats - analytic_nats).abs() / analytic_nats.abs();
    assert!(
        rel_err < 1e-10,
        "fisher_mass-based predicted_nats {predicted_nats} vs analytic 0.5*deltaT*F*delta \
         {analytic_nats}: rel_err {rel_err} >= 1e-10"
    );
}

#[test]
fn fisher_mass_quadratic_matches_exact_kl_at_small_delta() {
    let p_probs = [0.5_f64, 0.3, 0.2];
    // z = ln(p) satisfies softmax(z) = p exactly (p already sums to 1).
    let z: Vec<f64> = p_probs.iter().map(|p| p.ln()).collect();
    let dir = [1.0_f64, -1.7, 0.4];

    let metric = categorical_fisher_row_metric(&p_probs);

    // Shrink epsilon and confirm the quadratic prediction converges to the
    // exact KL, with the smallest epsilon comfortably under 1% relative error.
    let mut last_rel_err = f64::INFINITY;
    for &eps in &[1e-2_f64, 1e-3, 1e-4] {
        let delta: Vec<f64> = dir.iter().map(|d| eps * d).collect();
        let z_perturbed: Vec<f64> = z.iter().zip(delta.iter()).map(|(zi, di)| zi + di).collect();
        let q_probs = softmax(&z_perturbed);

        let exact_kl = kl(&p_probs, &q_probs);
        let delta_arr = Array1::from_vec(delta.clone());
        let predicted_nats = 0.5 * metric.fisher_mass(0, delta_arr.view());

        assert!(exact_kl > 0.0, "sanity: exact KL must be positive for nonzero delta");
        last_rel_err = (predicted_nats - exact_kl).abs() / exact_kl.abs();
    }

    assert!(
        last_rel_err < 1e-2,
        "quadratic fisher_mass prediction vs exact KL at small delta: \
         rel_err {last_rel_err} >= 1% — the fisher_mass units are not KL-nats"
    );
}
