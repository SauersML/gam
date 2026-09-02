//! Penalized-objective finite-difference gradient contract tests, split verbatim
//! out of `tests.rs` to keep that tracked file under the #780 10k-line gate.
//! Declared as a sibling `#[cfg(test)] mod` in `mod.rs`.
//!
//! Covers the assembled-gradient-vs-central-FD checks for the analytic penalty
//! family (`sae_assembled_gradient_matches_penalized_objective_central_fd`,
//! `sae_d1_assembled_gradient_matches_loss_central_fd`,
//! `sae_reml_extra_penalty_energy_counts_live_isometry_once`) and their shared
//! `SaeFd*` / `sae_fd_*` / `sae_pen_*` fixtures. All production symbols are in
//! scope via `super::*`.
#![cfg(test)]

use super::*;
use ndarray::{Array2, array};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(crate) struct SaeFdWorst {
    pub(crate) index: usize,
    pub(crate) analytic: f64,
    pub(crate) finite_difference: f64,
    pub(crate) absolute_error: f64,
    pub(crate) relative_error: f64,
}

impl SaeFdWorst {
    pub(crate) fn new() -> Self {
        Self {
            index: 0,
            analytic: 0.0,
            finite_difference: 0.0,
            absolute_error: 0.0,
            relative_error: 0.0,
        }
    }

    pub(crate) fn observe(&mut self, index: usize, analytic: f64, finite_difference: f64) {
        let absolute_error = (analytic - finite_difference).abs();
        let scale = analytic.abs().max(finite_difference.abs()).max(1.0e-9);
        let relative_error = absolute_error / scale;
        if relative_error > self.relative_error {
            self.index = index;
            self.analytic = analytic;
            self.finite_difference = finite_difference;
            self.absolute_error = absolute_error;
            self.relative_error = relative_error;
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SaeFdBlockReport {
    pub(crate) label: String,
    pub(crate) base_loss: f64,
    pub(crate) coord: SaeFdWorst,
    pub(crate) decoder: SaeFdWorst,
}

pub(crate) fn sae_fd_decoder(n_basis: usize, p_out: usize) -> Array2<f64> {
    let mut decoder = Array2::<f64>::zeros((n_basis, p_out));
    for basis in 0..n_basis {
        for out_col in 0..p_out {
            let phase = 0.73 * ((basis + 1) as f64) + 1.17 * ((out_col + 1) as f64);
            decoder[[basis, out_col]] = 0.16 * phase.sin() + 0.05 * (1.9 * phase).cos();
        }
    }
    decoder
}

pub(crate) fn sae_fd_target(n_obs: usize, p_out: usize) -> Array2<f64> {
    let mut target = Array2::<f64>::zeros((n_obs, p_out));
    for row in 0..n_obs {
        for out_col in 0..p_out {
            let x = (row as f64) + 1.0;
            let y = (out_col as f64) + 1.0;
            target[[row, out_col]] =
                0.21 * (0.31 * x + 0.47 * y).sin() - 0.13 * (0.19 * x * y).cos();
        }
    }
    target
}

pub(crate) fn sae_fd_coords(label: &str, n_obs: usize) -> Array2<f64> {
    let mut coords = Array2::<f64>::zeros((n_obs, 1));
    for row in 0..n_obs {
        let x = row as f64;
        coords[[row, 0]] = match label {
            "periodic_d1" => 0.07 + 0.043 * x + 0.004 * (1.3 * x).sin(),
            "euclidean_d1" => -0.46 + 0.048 * x + 0.006 * (1.7 * x).cos(),
            other => panic!("unknown SAE FD case label {other}"),
        };
    }
    coords
}

pub(crate) fn sae_fd_term(label: &str) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n_obs = 20usize;
    let p_out = 3usize;
    let coords = sae_fd_coords(label, n_obs);
    let (basis_kind, phi, jet, n_basis, atom) = match label {
        "periodic_d1" => {
            let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
            let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
            let n_basis = phi.ncols();
            let atom = SaeManifoldAtom::new_with_provided_function_gram(
                "periodic_d1",
                SaeAtomBasisKind::Periodic,
                1,
                phi.clone(),
                jet.clone(),
                sae_fd_decoder(n_basis, p_out),
                Array2::<f64>::eye(n_basis),
            )
            .unwrap()
            .with_basis_second_jet(evaluator);
            (SaeAtomBasisKind::Periodic, phi, jet, n_basis, atom)
        }
        "euclidean_d1" => {
            let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).unwrap());
            let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
            let n_basis = phi.ncols();
            let atom = SaeManifoldAtom::new_with_provided_function_gram(
                "euclidean_d1",
                SaeAtomBasisKind::EuclideanPatch,
                1,
                phi.clone(),
                jet.clone(),
                sae_fd_decoder(n_basis, p_out),
                Array2::<f64>::eye(n_basis),
            )
            .unwrap()
            .with_basis_second_jet(evaluator);
            (SaeAtomBasisKind::EuclideanPatch, phi, jet, n_basis, atom)
        }
        other => panic!("unknown SAE FD case label {other}"),
    };
    assert_eq!(
        basis_kind.latent_manifold(1),
        atom.basis_kind().latent_manifold(1)
    );
    assert_eq!(phi.dim(), (n_obs, n_basis));
    assert_eq!(jet.dim(), (n_obs, n_basis, 1));

    let manifold = atom.basis_kind().latent_manifold(1);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n_obs, 1)),
        vec![coords],
        vec![manifold],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = sae_fd_target(n_obs, p_out);
    let rho = SaeManifoldRho::new(0.0, 1.0e-4_f64.ln(), vec![array![-30.0]]);
    (term, target, rho)
}

pub(crate) fn sae_fd_refresh(term: &mut SaeManifoldTerm) {
    let coords = term.assignment.coords[0].as_matrix();
    term.atoms[0].refresh_basis(coords.view()).unwrap();
}

pub(crate) fn sae_fd_set_coord(term: &mut SaeManifoldTerm, row: usize, value: f64) {
    let mut flat = term.assignment.coords[0].as_flat().clone();
    flat[row] = value;
    term.assignment.coords[0].set_flat(flat.view());
    sae_fd_refresh(term);
}

pub(crate) fn sae_fd_total_loss(
    term: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> f64 {
    term.loss(target.view(), rho).unwrap().total()
}

pub(crate) fn sae_fd_check_case(label: &str) -> SaeFdBlockReport {
    let epsilon = 1.0e-6;
    let (term, target, rho) = sae_fd_term(label);
    // The declared reference-function Gram is fixed across the base value and
    // every perturbation, so the finite difference and analytic assembly price
    // exactly the same quadratic objective.
    let mut term = term;
    sae_fd_refresh(&mut term);
    let term = term;
    let base_loss = sae_fd_total_loss(&term, &target, &rho);
    assert!(base_loss.is_finite(), "{label}: base loss is not finite");

    let mut assembled = term.clone();
    sae_fd_refresh(&mut assembled);
    let sys = assembled
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    assert_eq!(sys.rows.len(), term.n_obs());
    assert_eq!(sys.gb.len(), term.beta_dim());
    for row in 0..term.n_obs() {
        assert_eq!(
            sys.rows[row].gt.len(),
            1,
            "{label}: K=1 softmax d=1 should expose exactly one row coordinate gradient"
        );
    }

    let mut coord = SaeFdWorst::new();
    let base_coords = term.assignment.coords[0].as_flat().clone();
    for row in 0..term.n_obs() {
        let mut plus = term.clone();
        sae_fd_set_coord(&mut plus, row, base_coords[row] + epsilon);
        let loss_plus = sae_fd_total_loss(&plus, &target, &rho);

        let mut minus = term.clone();
        sae_fd_set_coord(&mut minus, row, base_coords[row] - epsilon);
        let loss_minus = sae_fd_total_loss(&minus, &target, &rho);

        let finite_difference = (loss_plus - loss_minus) / (2.0 * epsilon);
        coord.observe(row, sys.rows[row].gt[0], finite_difference);
    }

    let mut decoder = SaeFdWorst::new();
    let beta = term.flatten_beta();
    for beta_idx in 0..beta.len() {
        let mut beta_plus = beta.clone();
        beta_plus[beta_idx] += epsilon;
        let mut plus = term.clone();
        plus.set_flat_beta(beta_plus.view()).unwrap();
        sae_fd_refresh(&mut plus);
        let loss_plus = sae_fd_total_loss(&plus, &target, &rho);

        let mut beta_minus = beta.clone();
        beta_minus[beta_idx] -= epsilon;
        let mut minus = term.clone();
        minus.set_flat_beta(beta_minus.view()).unwrap();
        sae_fd_refresh(&mut minus);
        let loss_minus = sae_fd_total_loss(&minus, &target, &rho);

        let finite_difference = (loss_plus - loss_minus) / (2.0 * epsilon);
        decoder.observe(beta_idx, sys.gb[beta_idx], finite_difference);
    }

    SaeFdBlockReport {
        label: label.to_string(),
        base_loss,
        coord,
        decoder,
    }
}

#[test]
pub(crate) fn sae_d1_assembled_gradient_matches_loss_central_fd() {
    let reports = vec![
        sae_fd_check_case("euclidean_d1"),
        sae_fd_check_case("periodic_d1"),
    ];
    let relative_tolerance = 3.0e-5;
    let absolute_tolerance = 3.0e-7;
    let mut all_blocks_match = true;
    for report in &reports {
        let coord_ok = report.coord.relative_error <= relative_tolerance
            || report.coord.absolute_error <= absolute_tolerance;
        let decoder_ok = report.decoder.relative_error <= relative_tolerance
            || report.decoder.absolute_error <= absolute_tolerance;
        let metadata_ok = !report.label.is_empty() && report.base_loss.is_finite();
        all_blocks_match = all_blocks_match && metadata_ok && coord_ok && decoder_ok;
    }
    assert!(
        all_blocks_match,
        "SAE d=1 assembled gradient does not match central finite difference: {reports:#?}"
    );
}
