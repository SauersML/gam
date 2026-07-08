//! #2133 — SURE within-basin deflation dof for the SAE reconstruction `φ̂`.
//!
//! The per-row latent coordinate is an incidental parameter (its count grows
//! with N), so the Gauss-Newton coordinate `coord_edf` — which drops the
//! second-order residual-curvature term of the basin-selecting MAP — MIS-counts
//! the flexibility the coordinate actually spends and biases `φ̂` LOW
//! (Neyman-Scott under-dispersion). [`SaeManifoldTerm::coordinate_sure_deflation_correction`]
//! restores the exact within-basin SURE divergence
//!   div = htt / (htt + f''ᵀr_code + V''),   r_code = f(θ̂) − y,
//! derived by the implicit-function theorem on the penalized stationarity.
//!
//! Two acceptance checks, both from the issue:
//!  (1) FD-ORACLE — on a curved chart (a radius-`R` circle, where the exact
//!      per-row divergence is analytically `R/ρ`), the analytic correction turns
//!      the GN coordinate edf into the finite-difference divergence `tr(∂μ̂/∂y)`
//!      of the ACTUAL per-row MAP estimator, strictly better than GN alone.
//!  (2) φ̂ COVERAGE — on planted curved data with a known dispersion `φ`, the
//!      corrected `φ̂` is closer to the truth than the (systematically low) GN
//!      `φ̂`, and does not over-shoot.

use crate::manifold::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};
use std::f64::consts::TAU;
use std::sync::Arc;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}
fn lcg_normal(s: &mut u64) -> f64 {
    let u1 = lcg(s).max(1e-12);
    let u2 = lcg(s);
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}

/// The circle chart `f(t) = (R·cos 2πt, R·sin 2πt, 0, …)` — matching the decoder
/// installed below (basis `[1, cos 2πt, sin 2πt]`, `B[1,0]=B[2,1]=R`).
fn f_circle(t: f64, radius: f64, p: usize) -> Vec<f64> {
    let mut out = vec![0.0; p];
    out[0] = radius * (TAU * t).cos();
    out[1] = radius * (TAU * t).sin();
    out
}

/// Grid argmin of the (essentially unpenalized, α≈0) per-row reconstruction
/// deviance `½‖x_row − f(t)‖²` — the actual per-row MAP the engine's inner solve
/// converges to. A dense grid stands in for the inner Newton so the estimator is
/// identical to the one the divergence is taken of.
fn grid_map_t(x_row: &[f64], radius: f64, grid: &[f64]) -> f64 {
    let mut best_t = grid[0];
    let mut best = f64::INFINITY;
    for &t in grid {
        let f = f_circle(t, radius, x_row.len());
        let d: f64 = x_row.iter().zip(f.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
        if d < best {
            best = d;
            best_t = t;
        }
    }
    best_t
}

/// Build an UNGATED single-atom circle term (fixed unit gate `a_k = 1`, so the
/// coordinate is the only per-row latent — a clean 1-D MAP estimator) whose
/// coordinates are set to `coords_t`. `log_ard = -10` makes the ARD precision
/// `α ≈ 4.5e-5` negligible, so the estimator is the near-ML projection onto the
/// circle whose divergence is analytically `R/ρ` (strong extrinsic curvature).
fn ungated_circle_term_at(
    coords_t: &[f64],
    radius: f64,
    p: usize,
) -> (SaeManifoldTerm, SaeManifoldRho) {
    let n = coords_t.len();
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(r, _)| coords_t[r]);
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let mut decoder = Array2::<f64>::zeros((3, p));
    decoder[[1, 0]] = radius;
    decoder[[2, 1]] = radius;
    let atom = SaeManifoldAtom::new(
        "circle".to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());
    let logits = Array2::<f64>::from_elem((n, 1), 6.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ibp_map(0.7, 1.0, false),
    )
    .unwrap()
    .with_ungated(vec![true])
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    term.set_guards_enabled(false);
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::from_elem(1, -10.0)]);
    (term, rho)
}

/// (1) FD-ORACLE: the analytic SURE correction reconstructs the finite-difference
/// divergence `tr(∂μ̂/∂y)` of the actual per-row MAP, strictly better than GN.
#[test]
fn sure_correction_matches_fd_divergence_2133() {
    let (n, p, radius, phi_true) = (200usize, 6usize, 1.0, 0.09_f64);
    let mut s = 0x2133_5A1E_0000_0001u64;
    let grid: Vec<f64> = (0..2000).map(|i| i as f64 / 2000.0).collect();

    // Plant curved data and take the actual per-row MAP coordinate.
    let mut x = Array2::<f64>::zeros((n, p));
    let mut t_hat = vec![0.0_f64; n];
    for i in 0..n {
        let theta = TAU * lcg(&mut s);
        let clean = f_circle(theta / TAU, radius, p);
        let mut xrow = vec![0.0; p];
        for c in 0..p {
            xrow[c] = clean[c] + phi_true.sqrt() * lcg_normal(&mut s);
            x[[i, c]] = xrow[c];
        }
        t_hat[i] = grid_map_t(&xrow, radius, &grid);
    }

    let (term, rho) = ungated_circle_term_at(&t_hat, radius, p);
    // Residual r_code = f(θ̂) − y (the convention the correction contracts).
    let residual = term.reconstruction_residual(x.view(), &rho).unwrap();

    // FD divergence of the estimator: perturb each y-entry, re-solve the MAP.
    let eps = 1e-4;
    let mut fd_div = 0.0_f64;
    for i in 0..n {
        let base_t = t_hat[i];
        let f_base = f_circle(base_t, radius, p);
        let mut xrow: Vec<f64> = (0..p).map(|c| x[[i, c]]).collect();
        for c in 0..p {
            xrow[c] += eps;
            let t_pert = grid_map_t(&xrow, radius, &grid);
            xrow[c] -= eps;
            let f_pert = f_circle(t_pert, radius, p);
            fd_div += (f_pert[c] - f_base[c]) / eps;
        }
    }

    // Analytic GN baseline and exact divergence (a_k = 1, α ≈ 0):
    //   htt = ‖f'‖² = (2πR)²,   c = f''ᵀr = −(2πR)²·fᵀr/R² ... computed directly,
    //   GN_div = Σ htt/(htt+V''),   exact = Σ htt/(htt+c+V'').
    let htt = (TAU * radius).powi(2);
    let alpha = SaeManifoldRho::stable_exp_strength(-10.0);
    let mut gn_div = 0.0_f64;
    let mut exact_div = 0.0_f64;
    for i in 0..n {
        let t = t_hat[i];
        let f = f_circle(t, radius, p);
        // f'' = −(2π)²·f ⇒ c = f''ᵀ(f − y) = −(2π)²·Σ f·(f − y).
        let c: f64 = (0..p)
            .map(|k| -(TAU * TAU) * f[k] * (f[k] - x[[i, k]]))
            .sum();
        let v_pp = (alpha * (TAU * t).cos()).max(0.0);
        let denom_gn = htt + v_pp;
        // Same PD floor the production correction applies.
        let denom_full =
            (htt + c + v_pp).max(SaeManifoldTerm::SURE_DIVERGENCE_PD_FLOOR * denom_gn);
        gn_div += htt / denom_gn;
        exact_div += htt / denom_full;
    }

    let correction = term
        .coordinate_sure_deflation_correction(residual.view(), &rho)
        .unwrap();

    eprintln!(
        "[#2133 FD] fd_div={fd_div:.3} exact_div={exact_div:.3} gn_div={gn_div:.3} \
         correction={correction:.3} (exact−gn={:.3})",
        exact_div - gn_div
    );

    // The production method returns exactly the analytic (exact − GN) delta.
    assert!(
        (correction - (exact_div - gn_div)).abs() < 1e-6,
        "correction {correction} != analytic delta {}",
        exact_div - gn_div
    );
    // Exact divergence reproduces the FD oracle of the real estimator.
    let exact_err = (exact_div - fd_div).abs();
    let gn_err = (gn_div - fd_div).abs();
    assert!(
        exact_err < 0.02 * fd_div,
        "exact-div {exact_div} not within 2% of FD divergence {fd_div}"
    );
    // …and does so STRICTLY better than the Gauss-Newton baseline (which the
    // issue documents as systematically under-counting on a curved chart).
    assert!(
        exact_err < 0.25 * gn_err,
        "GN+correction (err {exact_err}) not decisively better than GN alone (err {gn_err})"
    );
}

/// (2) φ̂ COVERAGE end-to-end through `reconstruction_dispersion`: on planted
/// curved data with known `φ`, the corrected `φ̂` is closer to the truth than the
/// GN `φ̂` and does not over-shoot.
#[test]
fn sure_corrected_phi_hat_improves_coverage_2133() {
    let (n, p, radius, phi_true) = (400usize, 6usize, 1.0, 0.09_f64);
    let mut s = 0x2133_C0FF_EE00_0002u64;
    let grid: Vec<f64> = (0..2000).map(|i| i as f64 / 2000.0).collect();

    let mut x = Array2::<f64>::zeros((n, p));
    let mut t_hat = vec![0.0_f64; n];
    for i in 0..n {
        let theta = TAU * lcg(&mut s);
        let clean = f_circle(theta / TAU, radius, p);
        let mut xrow = vec![0.0; p];
        for c in 0..p {
            xrow[c] = clean[c] + phi_true.sqrt() * lcg_normal(&mut s);
            x[[i, c]] = xrow[c];
        }
        t_hat[i] = grid_map_t(&xrow, radius, &grid);
    }

    let (mut term, rho) = ungated_circle_term_at(&t_hat, radius, p);
    // Assemble the arrow-Schur cache at the installed MAP state (0 inner iters).
    let (_v, loss, cache) = term
        .reml_criterion_with_cache(x.view(), &rho, None, 0, 1.0, 1e-6, 1e-6)
        .expect("reml assemble at MAP state");
    let residual = term.reconstruction_residual(x.view(), &rho).unwrap();

    let phi_gn = term
        .reconstruction_dispersion(&loss, &cache, &rho, None)
        .unwrap();
    let phi_sure = term
        .reconstruction_dispersion(&loss, &cache, &rho, Some(residual.view()))
        .unwrap();

    let gn_ratio = phi_gn / phi_true;
    let sure_ratio = phi_sure / phi_true;
    eprintln!(
        "[#2133 coverage] φ_true={phi_true}  φ̂_gn/φ={gn_ratio:.3}  φ̂_sure/φ={sure_ratio:.3}"
    );

    // GN is biased low (the #2133 under-dispersion).
    assert!(gn_ratio < 0.99, "GN φ̂/φ={gn_ratio} expected < 0.99 (under-dispersed)");
    // The correction adds coordinate dof ⇒ raises φ̂ toward the truth.
    assert!(
        phi_sure > phi_gn,
        "corrected φ̂ {phi_sure} must exceed GN φ̂ {phi_gn}"
    );
    // Strictly closer to the planted φ, with no over-shoot.
    assert!(
        (sure_ratio - 1.0).abs() < (gn_ratio - 1.0).abs(),
        "corrected φ̂/φ={sure_ratio} not closer to 1 than GN φ̂/φ={gn_ratio}"
    );
    assert!(
        sure_ratio < 1.08,
        "corrected φ̂/φ={sure_ratio} over-shoots (bar 1.08)"
    );
}
