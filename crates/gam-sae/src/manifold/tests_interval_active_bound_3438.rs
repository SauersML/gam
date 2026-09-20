#![cfg(test)]
//! #3438 probe: an Interval atom whose fitted coordinates pile up at a bound.

use super::*;
use crate::basis::EuclideanPatchEvaluator;
use gam_terms::latent::LatentManifold;
use ndarray::{Array2, array};
use std::sync::Arc;

fn interval_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 8usize;
    let p = 2usize;
    let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).expect("patch basis"));
    // Generating latent: several rows sit beyond the interval's upper end.
    let truth: [f64; 8] = [-0.7, -0.3, 0.1, 0.4, 1.6, 1.8, 2.0, 0.8];
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| 0.8 * truth[row].clamp(-0.9, 0.9));
    let (phi, jet) = evaluator.evaluate(coords.view()).expect("coords evaluate");
    let width = phi.ncols();
    let decoder = Array2::<f64>::from_shape_fn((width, p), |(b, o)| {
        [[0.1, -0.2], [1.0, 0.6], [0.2, -0.3]][b][o]
    });
    let mut target = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let s = truth[row];
        for o in 0..p {
            target[[row, o]] = decoder[[0, o]] + decoder[[1, o]] * s + decoder[[2, o]] * s * s
                + 0.03 * (1.7 * row as f64 + 0.9 * o as f64).sin();
        }
    }
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "interval".to_string(),
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(width),
    )
    .expect("atom shapes agree")
    .with_basis_second_jet(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Interval { lo: -1.0, hi: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");
    let rho = SaeManifoldRho::new(0.0, 1.0, vec![array![-3.0]]);
    (term, target, rho)
}

#[test]
fn probe_interval_active_bound_3438() {
    let (mut term, target, rho) = interval_fixture();
    let result = term.penalized_quasi_laplace_criterion_with_cache(
        target.view(),
        &rho,
        None,
        60,
        0.4,
        1.0e-8,
        1.0e-8,
    );
    eprintln!("[3438] criterion result: {:?}", result.as_ref().map(|r| r.0));
    eprintln!("[3438] coords {:?}", term.assignment.coords[0].as_matrix().column(0).to_vec());
    let system = term.assemble_arrow_schur(target.view(), &rho, None).expect("assemble");
    for (i, row) in system.rows.iter().enumerate() {
        eprintln!("[3438] row {i} gt={:?} htt={:?}", row.gt.to_vec(), row.htt.iter().copied().collect::<Vec<_>>());
    }
    let options = term.evidence_factor_options();
    let factored = solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options);
    eprintln!("[3438] factor ok={}", factored.is_ok());
    if let Err(e) = &factored {
        eprintln!("[3438] factor err={e:?}");
    }
    if let Ok((_, _, cache)) = factored {
        match term.materialize_exact_hessian_dense_with_gap_border(&rho, target.view(), &cache) {
            Ok((a, _)) => {
                for i in 0..target.nrows() {
                    let off = cache.row_offsets[i];
                    let dim = cache.row_dims[i];
                    let row: Vec<f64> = (0..dim).map(|c| a[[off, off + c]]).collect();
                    eprintln!("[3438] A row {i} t-slot diag-block {row:?} (dim {dim})");
                }
            }
            Err(e) => eprintln!("[3438] dense A err={e}"),
        }
        match term.exact_observed_information_log_dets(&rho, target.view(), &cache) {
            Ok(v) => eprintln!("[3438] log|A| {v:?}"),
            Err(e) => eprintln!("[3438] log|A| err={e}"),
        }
    }
}
