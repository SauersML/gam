//! CTN ψψ pairwise oracle.
//!
//! Builds a tiny CTN problem (n=4, p_resp=2, p_cov=2, ψ_dim=2) using the same
//! fixture data as the inline transformation_normal tests, calls the existing
//! pairwise body `exact_newton_joint_psisecond_order_terms(i, j)` for every
//! (i, j) pair, and writes the per-pair likelihood pieces (a, g, b_mat) plus
//! a contraction `Σ_j v_j · pair(i, j)` for a fixed direction v to JSON.
//!
//! Run:
//!     cargo run --release --example ctn_pairwise_oracle
//!
//! Writes:
//!     /tmp/ctn_pairwise_oracle.json
//!
//! This is one of three independent verification paths for the CTN HVP work:
//!   (1) sympy proof shadow (scripts/ctn_hvp_groundtruth.py)
//!   (2) THIS oracle — current pairwise body, before Phase 2 directional version
//!   (3) ctn-hvp-phase2's directional CtnOuterHessianOperator (when commit 2 lands)
//! All three must agree.

use ndarray::{Array1, Array2, array, s};
use serde_json::json;
use std::sync::Arc;

use gam::families::custom_family::{
    CustomFamily, CustomFamilyBlockPsiDerivative, ParameterBlockState,
};
use gam::families::transformation_normal::{
    TransformationNormalConfig, TransformationNormalFamily, build_tensor_psi_derivatives,
};
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};

/// Recreate the toy ψ-dependent covariate design and per-axis derivatives
/// from `transformation_normal.rs::tests::toy_covariate_design_and_derivs`.
/// 4×2 covariate basis built as a quadratic-in-ψ Taylor expansion around ψ=0.
fn toy_covariate_design_and_derivs(
    psi: &Array1<f64>,
) -> (Array2<f64>, Vec<CustomFamilyBlockPsiDerivative>) {
    let x0 = array![[1.00, 0.40], [1.10, 0.35], [1.20, 0.45], [0.95, 0.50]];
    let x_a = array![[0.10, -0.02], [0.08, 0.01], [0.12, -0.01], [0.09, 0.03]];
    let x_b = array![[-0.04, 0.06], [-0.02, 0.05], [-0.03, 0.04], [-0.01, 0.07]];
    let x_aa = array![[0.02, 0.00], [0.01, 0.01], [0.02, -0.01], [0.01, 0.02]];
    let x_ab = array![[0.01, -0.01], [0.00, 0.02], [0.01, 0.01], [0.00, -0.01]];
    let x_bb = array![[-0.01, 0.02], [-0.02, 0.01], [-0.01, 0.00], [-0.02, 0.02]];
    let design = &x0
        + &(x_a.clone() * psi[0])
        + &(x_b.clone() * psi[1])
        + &(x_aa.clone() * (0.5 * psi[0] * psi[0]))
        + &(x_ab.clone() * (psi[0] * psi[1]))
        + &(x_bb.clone() * (0.5 * psi[1] * psi[1]));
    let d_a = &x_a + &(x_aa.clone() * psi[0]) + &(x_ab.clone() * psi[1]);
    let d_b = &x_b + &(x_ab.clone() * psi[0]) + &(x_bb.clone() * psi[1]);
    let deriv_a = CustomFamilyBlockPsiDerivative::new(
        None,
        d_a,
        Array2::zeros((0, 0)),
        None,
        Some(vec![x_aa.clone(), x_ab.clone()]),
        None,
        None,
    );
    let deriv_b = CustomFamilyBlockPsiDerivative::new(
        None,
        d_b,
        Array2::zeros((0, 0)),
        None,
        Some(vec![x_ab, x_bb]),
        None,
        None,
    );
    (design, vec![deriv_a, deriv_b])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let psi = array![0.15, -0.10];
    let beta = array![0.15, -0.05, 0.80, 0.30];
    let v = array![0.4, -0.7]; // contraction direction (length ψ_dim)

    let response_val_basis = array![[1.0, -1.0], [1.0, -0.2], [1.0, 0.6], [1.0, 1.3]];
    let response_deriv_basis = array![[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]];
    let weights = Array1::from_elem(response_val_basis.nrows(), 1.0);
    let offset = Array1::zeros(response_val_basis.nrows());
    let (cov_design, cov_derivs) = toy_covariate_design_and_derivs(&psi);

    let family = TransformationNormalFamily::from_prebuilt_response_basis(
        response_val_basis.clone(),
        response_deriv_basis.clone(),
        vec![],
        Array1::zeros(0),
        1,
        Array2::zeros((0, 0)),
        &weights,
        &offset,
        DesignMatrix::Dense(DenseDesignMatrix::from(cov_design.clone())),
        vec![],
        &TransformationNormalConfig {
            double_penalty: false,
            ..TransformationNormalConfig::default()
        },
        None,
    )?;

    let derivative_blocks = vec![build_tensor_psi_derivatives(&family, &cov_derivs)?];
    let h_prime = family.x_deriv_kron.forward_mul(&beta);
    if !h_prime.iter().all(|val| *val > 0.25) {
        return Err(format!("toy beta must keep h' positive, got {:?}", h_prime).into());
    }

    let state = ParameterBlockState {
        beta: beta.clone(),
        eta: Array1::zeros(h_prime.len()),
    };
    let spec = family.block_spec();
    let block_states = vec![state.clone()];
    let specs = vec![spec.clone()];

    let psi_dim = psi.len();
    let p_total = beta.len();
    let n_obs = response_val_basis.nrows();

    println!(
        "[oracle] toy CTN: n={n_obs}, p_resp=2, p_cov=2, p_total={p_total}, ψ_dim={psi_dim}"
    );
    println!("[oracle] β = {:?}", beta.as_slice().unwrap());
    println!("[oracle] ψ = {:?}", psi.as_slice().unwrap());
    println!("[oracle] v = {:?}", v.as_slice().unwrap());
    println!();

    // Per-pair pairwise body — likelihood pieces only (no penalty/logdet).
    let mut pairs_json = Vec::new();
    for i in 0..psi_dim {
        for j in 0..psi_dim {
            let terms_opt = family.exact_newton_joint_psisecond_order_terms(
                &block_states,
                &specs,
                &derivative_blocks,
                i,
                j,
            )?;
            let Some(terms) = terms_opt else {
                println!("[oracle] (i={i}, j={j}) -> no pair returned");
                continue;
            };
            println!(
                "[oracle] pair (i={i}, j={j}): a={:.10}, ‖g‖∞={:.6e}, ‖b_mat‖∞={:.6e}",
                terms.objective_psi_psi,
                terms.score_psi_psi.iter().fold(0.0f64, |m, x| m.max(x.abs())),
                terms
                    .hessian_psi_psi
                    .iter()
                    .fold(0.0f64, |m, x| m.max(x.abs())),
            );
            pairs_json.push(json!({
                "i": i,
                "j": j,
                "a": terms.objective_psi_psi,
                "g": terms.score_psi_psi.to_vec(),
                "b_mat": terms.hessian_psi_psi.iter().copied().collect::<Vec<f64>>(),
                "b_mat_shape": [terms.hessian_psi_psi.nrows(), terms.hessian_psi_psi.ncols()],
            }));
        }
    }

    // Directional contraction: Σ_j v_j · pair(i, j) for each i.
    // a_dir(i) = Σ_j v_j · pair(i, j).a
    // g_dir(i) = Σ_j v_j · pair(i, j).g     (length-p vector)
    // b_dir(i) = Σ_j v_j · pair(i, j).b_mat (p×p matrix)
    let mut a_dir = Array1::<f64>::zeros(psi_dim);
    let mut g_dir = Array2::<f64>::zeros((psi_dim, p_total));
    let mut b_dir = vec![Array2::<f64>::zeros((p_total, p_total)); psi_dim];
    for i in 0..psi_dim {
        for j in 0..psi_dim {
            let terms_opt = family.exact_newton_joint_psisecond_order_terms(
                &block_states,
                &specs,
                &derivative_blocks,
                i,
                j,
            )?;
            let Some(terms) = terms_opt else { continue };
            a_dir[i] += v[j] * terms.objective_psi_psi;
            let mut g_row = g_dir.slice_mut(s![i, ..]);
            g_row.scaled_add(v[j], &terms.score_psi_psi);
            b_dir[i].scaled_add(v[j], &terms.hessian_psi_psi);
        }
    }

    println!();
    println!("[oracle] directional contraction Σ_j v_j · pair(i, j):");
    for i in 0..psi_dim {
        println!(
            "[oracle]   i={i}: a_dir={:.10}, ‖g_dir‖∞={:.6e}, ‖b_dir‖∞={:.6e}",
            a_dir[i],
            g_dir.row(i).iter().fold(0.0f64, |m, x| m.max(x.abs())),
            b_dir[i].iter().fold(0.0f64, |m, x| m.max(x.abs())),
        );
    }

    let directional_json: Vec<_> = (0..psi_dim)
        .map(|i| {
            json!({
                "i": i,
                "a_dir": a_dir[i],
                "g_dir": g_dir.row(i).to_vec(),
                "b_dir": b_dir[i].iter().copied().collect::<Vec<f64>>(),
                "b_dir_shape": [p_total, p_total],
            })
        })
        .collect();

    let blob = json!({
        "config": {
            "n": n_obs,
            "p_resp": 2,
            "p_cov": 2,
            "p_total": p_total,
            "psi_dim": psi_dim,
            "beta": beta.to_vec(),
            "psi": psi.to_vec(),
            "v": v.to_vec(),
            "weights": weights.to_vec(),
            "response_val_basis": response_val_basis.iter().copied().collect::<Vec<f64>>(),
            "response_deriv_basis": response_deriv_basis.iter().copied().collect::<Vec<f64>>(),
            "cov_design_at_psi": cov_design.iter().copied().collect::<Vec<f64>>(),
        },
        "pairwise": pairs_json,
        "directional_contraction": directional_json,
        "note": "Likelihood-only pieces from exact_newton_joint_psisecond_order_terms. \
                 Penalty/logdet contributions are added by the unified evaluator's \
                 outer_hessian_entry. Cross-check this against sympy-shadow's symbolic \
                 derivation of the same likelihood quantities at the same toy config.",
    });

    let path = "/tmp/ctn_pairwise_oracle.json";
    std::fs::write(path, serde_json::to_string_pretty(&blob)?)?;
    let _ = Arc::<()>::new(()); // silence unused-import lint if Arc wasn't otherwise used
    println!();
    println!("[oracle] wrote {path}");

    Ok(())
}
