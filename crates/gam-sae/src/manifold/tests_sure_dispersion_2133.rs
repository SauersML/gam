//! #2133 — SURE within-basin deflation dof for the SAE reconstruction `φ̂`.
//!
//! The per-row latent coordinate is an incidental parameter (its count grows
//! with N), so the Gauss-Newton coordinate `coord_edf` — which drops the
//! second-order residual-curvature term of the basin-selecting MAP — MIS-counts
//! the flexibility the coordinate spends and biases `φ̂` LOW (Neyman-Scott
//! under-dispersion). [`SaeManifoldTerm::coordinate_sure_deflation_correction`]
//! restores the exact within-basin SURE divergence
//!   div = htt / (htt + f''ᵀr_code + V''),   r_code = f(θ̂) − y,
//! derived by the implicit-function theorem on the penalized stationarity
//! `g = f'ᵀ(f−y) + V' = 0`  (`∂g/∂θ = htt + f''ᵀr_code + V''`, `∂g/∂y = −f'`).
//!
//! Tests, on a REAL converged circle fit (self-consistent basis/coords/residual):
//!  (1) PLUMBING+STABILITY — the production correction equals an independent
//!      hand recomputation from the term's own jet/residual primitives, the
//!      term's residual is consistent with its basis on every row (so the
//!      correction cannot desync), and the per-row dof correction stays bounded.
//!  (2) FD-ORACLE — perturbing the target and re-solving each row's 1-D
//!      coordinate MAP (holding the fitted decoder/gate fixed) gives the true
//!      divergence `tr(∂μ̂/∂y)`; `GN_edf + correction` reproduces it, decisively
//!      better than the Gauss-Newton baseline alone.

use crate::manifold::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    ArdAxisPrior, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
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

/// A REAL joint-fit K=1 rank-2 circle on dims (0,1): cos→e0, sin→e1, radius `R`,
/// isotropic noise `σ`. Returns the fitted term, its rho, and the target `x`.
/// Because it is the engine's own converged state, the atom basis, the
/// assignment coordinates, and `reconstruction_residual` are all mutually
/// consistent — the setting the dispersion correction actually runs in.
fn fitted_circle(
    n: usize,
    p: usize,
    radius: f64,
    sigma: f64,
    seed: u64,
) -> (SaeManifoldTerm, SaeManifoldRho, Array2<f64>) {
    let mut s = seed;
    let theta: Vec<f64> = (0..n).map(|_| TAU * lcg(&mut s)).collect();
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] += radius * theta[i].cos();
        x[[i, 1]] += radius * theta[i].sin();
        for j in 0..p {
            x[[i, j]] += sigma * lcg_normal(&mut s);
        }
    }
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(r, _)| theta[r] / TAU);
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
    let logits = Array2::<f64>::from_elem((n, 1), 3.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ibp_map(0.7, 1.0, false),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    term.set_guards_enabled(false);
    let mut rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    term.run_joint_fit_arrow_schur(x.view(), &mut rho, None, 80, 1.0, 1e-7, 1e-7)
        .expect("K=1 circle joint fit");
    (term, rho, x)
}

/// Independent hand recomputation of the correction from the term's OWN
/// primitives (the same objects the production method reads), returning the total
/// and the maximum absolute per-row contribution. Mirrors the production formula
/// exactly, so a match validates the wiring (indexing, `a_k`, second-jet
/// contraction, PD floor) end-to-end.
fn hand_correction(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    residual: &Array2<f64>,
) -> (f64, f64) {
    let p = term.output_dim();
    let n = term.n_obs();
    let sj = term.atom_second_jets().unwrap();
    let periods = term.assignment.coords[0].effective_axis_periods();
    let mut g1 = vec![0.0; p];
    let mut g2 = vec![0.0; p];
    let mut a_row = vec![0.0; term.atoms.len()];
    let mut total = 0.0_f64;
    let mut max_abs = 0.0_f64;
    for i in 0..n {
        term.assignment
            .try_assignments_row_for_rho_into(i, rho, &mut a_row)
            .unwrap();
        let a_k = a_row[0];
        let t = term.assignment.coords[0].row(i)[0];
        let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[0][0]);
        let v_pp = ArdAxisPrior::eval(alpha, t, periods[0])
            .hess
            .max(0.0);
        term.atoms[0].fill_decoded_derivative_row(i, 0, &mut g1);
        term.atoms[0].fill_decoded_second_derivative_row(&sj[0], i, 0, &mut g2);
        let htt = a_k * a_k * g1.iter().map(|v| v * v).sum::<f64>();
        let denom_gn = htt + v_pp;
        if !(denom_gn > 0.0) {
            continue;
        }
        let c = a_k
            * g2.iter()
                .zip((0..p).map(|k| residual[[i, k]]))
                .map(|(a, b)| a * b)
                .sum::<f64>();
        let denom_full =
            (htt + c + v_pp).max(SaeManifoldTerm::SURE_DIVERGENCE_PD_FLOOR * denom_gn);
        let delta = htt / denom_full - htt / denom_gn;
        total += delta;
        max_abs = max_abs.max(delta.abs());
    }
    (total, max_abs)
}

/// (1) PLUMBING + STABILITY on a real converged fit.
#[test]
fn sure_correction_wiring_and_stability_2133() {
    let (n, p) = (160usize, 6usize);
    let (term, rho, x) = fitted_circle(n, p, 1.0, 0.30, 0x2133_5A1E_0000_0001);
    let residual = term.reconstruction_residual(x.view(), &rho).unwrap();

    // Residual is consistent with the atom basis on EVERY row (fitted from the
    // decoded primitives equals reconstruction_residual + x). This is exactly the
    // invariant a converged fit guarantees and that the correction relies on.
    let mut a_row = vec![0.0; term.atoms.len()];
    let mut decoded = vec![0.0; p];
    let mut max_incons = 0.0_f64;
    for i in 0..n {
        term.assignment
            .try_assignments_row_for_rho_into(i, &rho, &mut a_row)
            .unwrap();
        term.atoms[0].fill_decoded_row(i, &mut decoded);
        for k in 0..p {
            let fitted_k = a_row[0] * decoded[k];
            max_incons = max_incons.max((fitted_k - (residual[[i, k]] + x[[i, k]])).abs());
        }
    }
    assert!(
        max_incons < 1e-9,
        "residual/basis desync on converged fit (max {max_incons:.2e})"
    );

    // Production correction == independent hand recomputation (wiring correct).
    let correction = term
        .coordinate_sure_deflation_correction(residual.view(), &rho)
        .unwrap();
    let (hand, max_abs_row) = hand_correction(&term, &rho, &residual);
    eprintln!(
        "[#2133 wiring] correction={correction:.6} hand={hand:.6} max|Δedf/row|={max_abs_row:.4}"
    );
    assert!(
        (correction - hand).abs() < 1e-9,
        "method {correction} != hand replica {hand}"
    );
    // Per-row dof correction is bounded (no floor blow-up / instability).
    assert!(
        max_abs_row < 1.5,
        "per-row dof correction {max_abs_row} unphysically large"
    );
    // Total correction is a modest fraction of the row count (a per-row edf tweak).
    assert!(
        correction.abs() < n as f64,
        "total correction {correction} exceeds row count {n}"
    );
}

/// (2) FD-ORACLE: `GN_edf + correction` reproduces the true divergence
/// `tr(∂μ̂/∂y)` of the per-row coordinate MAP (decoder/gate held at the fit),
/// decisively better than the Gauss-Newton baseline.
#[test]
fn sure_correction_matches_fd_divergence_2133() {
    let (n, p, radius) = (160usize, 6usize, 1.0);
    let (term, rho, x) = fitted_circle(n, p, radius, 0.30, 0x2133_C0FF_EE00_0002);
    let residual = term.reconstruction_residual(x.view(), &rho).unwrap();
    let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[0][0]);
    let periods = term.assignment.coords[0].effective_axis_periods();

    // Evaluate the atom's fitted image f(t) = a_k·Φ(t)·B and its coordinate MAP by
    // a fine 1-D grid + local refine, using the FITTED gate a_k (frozen decoder).
    let mut a_row = vec![0.0; term.atoms.len()];
    // Per-row fixed gate.
    let a_of: Vec<f64> = (0..n)
        .map(|i| {
            term.assignment
                .try_assignments_row_for_rho_into(i, &rho, &mut a_row)
                .unwrap();
            a_row[0]
        })
        .collect();
    let ev = PeriodicHarmonicEvaluator::new(3).unwrap();
    let decoder = term.atoms[0].decoder_coefficients.clone();
    // f, f', f'' at arbitrary t (a·Φ·B and its coordinate derivatives).
    let jets_at = |t: f64, a: f64| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let coords = Array2::<f64>::from_elem((1, 1), t.rem_euclid(1.0));
        let (phi, jet) = ev.evaluate(coords.view()).unwrap();
        let sj = crate::manifold::SaeBasisSecondJet::second_jet(&ev, coords.view()).unwrap();
        let (mut f, mut fp, mut fpp) = (vec![0.0; p], vec![0.0; p], vec![0.0; p]);
        for b in 0..decoder.nrows() {
            for c in 0..p {
                f[c] += a * phi[[0, b]] * decoder[[b, c]];
                fp[c] += a * jet[[0, b, 0]] * decoder[[b, c]];
                fpp[c] += a * sj[[0, b, 0, 0]] * decoder[[b, c]];
            }
        }
        (f, fp, fpp)
    };
    let f_at = |t: f64, a: f64| -> Vec<f64> { jets_at(t, a).0 };
    // Penalized deviance ½‖y−f‖² + V(t); V = (α/κ²)(1−cos κt), κ=2π:
    //   J'(t)  = f'ᵀ(f−y) + α·sin(2πt)/(2π),   J''(t) = f''ᵀ(f−y)+‖f'‖² + α·cos(2πt).
    let grid: Vec<f64> = (0..2000).map(|g| g as f64 / 2000.0).collect();
    let map_t = |y: &[f64], a: f64| -> f64 {
        // Coarse grid seed, then Newton-refine to a CONTINUOUS minimum so the FD
        // divergence is smooth (a grid argmin quantizes ∂t/∂y).
        let mut best_t = 0.0;
        let mut best = f64::INFINITY;
        for &t in &grid {
            let f = f_at(t, a);
            let d = 0.5 * y.iter().zip(f.iter()).map(|(u, v)| (u - v) * (u - v)).sum::<f64>()
                + (alpha / (TAU * TAU)) * (1.0 - (TAU * t).cos());
            if d < best {
                best = d;
                best_t = t;
            }
        }
        let mut t = best_t;
        for _ in 0..40 {
            let (f, fp, fpp) = jets_at(t, a);
            let r: Vec<f64> = (0..p).map(|c| f[c] - y[c]).collect();
            let jp = (0..p).map(|c| fp[c] * r[c]).sum::<f64>()
                + alpha * (TAU * t).sin() / TAU;
            let jpp = (0..p).map(|c| fpp[c] * r[c] + fp[c] * fp[c]).sum::<f64>()
                + alpha * (TAU * t).cos();
            if jpp.abs() < 1e-12 {
                break;
            }
            let step = jp / jpp;
            t -= step;
            if step.abs() < 1e-13 {
                break;
            }
        }
        t
    };

    // FD divergence + analytic GN baseline and exact (GN+correction) per row.
    let sj = term.atom_second_jets().unwrap();
    let mut g1 = vec![0.0; p];
    let mut g2 = vec![0.0; p];
    let (mut fd_div, mut gn_div, mut exact_div) = (0.0_f64, 0.0_f64, 0.0_f64);
    let eps = 1e-4;
    for i in 0..n {
        let a = a_of[i];
        let y: Vec<f64> = (0..p).map(|c| x[[i, c]]).collect();
        let t0 = map_t(&y, a);
        let f0 = f_at(t0, a);
        let mut yp = y.clone();
        for c in 0..p {
            yp[c] += eps;
            let fp = f_at(map_t(&yp, a), a);
            yp[c] -= eps;
            fd_div += (fp[c] - f0[c]) / eps;
        }
        // Analytic divergence at the fitted coordinate (same primitives as prod).
        let t = term.assignment.coords[0].row(i)[0];
        let v_pp = ArdAxisPrior::eval(alpha, t, periods[0])
            .hess
            .max(0.0);
        term.atoms[0].fill_decoded_derivative_row(i, 0, &mut g1);
        term.atoms[0].fill_decoded_second_derivative_row(&sj[0], i, 0, &mut g2);
        let htt = a * a * g1.iter().map(|v| v * v).sum::<f64>();
        let denom_gn = htt + v_pp;
        let c = a * g2.iter().zip((0..p).map(|k| residual[[i, k]])).map(|(u, v)| u * v).sum::<f64>();
        let denom_full =
            (htt + c + v_pp).max(SaeManifoldTerm::SURE_DIVERGENCE_PD_FLOOR * denom_gn);
        gn_div += htt / denom_gn;
        exact_div += htt / denom_full;
    }

    eprintln!(
        "[#2133 FD] fd_div={fd_div:.2} gn_div={gn_div:.2} exact_div(GN+corr)={exact_div:.2}"
    );
    let exact_err = (exact_div - fd_div).abs();
    let gn_err = (gn_div - fd_div).abs();
    assert!(
        exact_err < 0.05 * fd_div.abs().max(1.0),
        "GN+correction {exact_div} not within 5% of FD divergence {fd_div}"
    );
    assert!(
        exact_err < 0.5 * gn_err,
        "GN+correction (err {exact_err:.3}) not decisively better than GN alone (err {gn_err:.3})"
    );
}
