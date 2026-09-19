//! Integer row weights of the multinomial Laplace criterion are row repetitions.
//!
//! A row carrying weight `k` contributes `k` copies of its log-likelihood, score
//! and information, so at every fixed joint rho the weighted family and the
//! family with that row repeated `k` times (and dropped at `k = 0`) have the same
//! criterion value, outer gradient and outer Hessian, with and without the
//! Jeffreys term. The sklearn contract checks this equivalence end to end
//! (`check_sample_weight_equivalence_on_dense_data`); this pins it at the
//! criterion, where any disagreement is a weighting defect rather than an
//! optimizer path.

use gam_custom_family::{
    BlockwiseFitOptions, CustomFamilyHyperLayout, CustomFamilyJointHyperResult,
    evaluate_custom_family_joint_hyper,
};
use gam_models::MultinomialFamily;
use gam_problem::{EvalMode, HessianValue, PenaltyMatrix};
use ndarray::{Array1, Array2, array};
use std::sync::Arc;

const N_ROWS: usize = 15;
const WEIGHTS: [usize; N_ROWS] = [3, 0, 1, 4, 2, 0, 1, 3, 2, 4, 1, 0, 2, 3, 1];
const CLASSES: [usize; N_ROWS] = [0, 1, 2, 2, 0, 1, 1, 0, 2, 1, 0, 2, 1, 0, 2];

fn covariates(row: usize) -> (f64, f64) {
    // Deterministic, irregular points in the unit square.
    let x0 = ((row as f64 + 0.5) * 0.618_033_988_749_895).fract();
    let x1 = ((row as f64 + 0.5) * 0.414_213_562_373_095).fract();
    (x0, x1)
}

fn family(rows: &[usize], weights: Array1<f64>, armed: bool) -> MultinomialFamily {
    let n = rows.len();
    let mut design = Array2::zeros((n, 3));
    let mut response = Array2::zeros((n, 3));
    for (i, &row) in rows.iter().enumerate() {
        let (x0, x1) = covariates(row);
        design[[i, 0]] = 1.0;
        design[[i, 1]] = x0;
        design[[i, 2]] = x1;
        response[[i, CLASSES[row]]] = 1.0;
    }
    let penalties = vec![
        PenaltyMatrix::Dense(array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        PenaltyMatrix::Dense(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    ];
    MultinomialFamily::new(response, weights, 3, Arc::new(design), Arc::new(penalties))
        .expect("finite three-class linear design")
        .with_joint_jeffreys_term(armed)
}

fn weighted_family(armed: bool) -> MultinomialFamily {
    let rows: Vec<usize> = (0..N_ROWS).collect();
    let weights = Array1::from_iter(WEIGHTS.iter().map(|&w| w as f64));
    family(&rows, weights, armed)
}

fn repeated_family(armed: bool) -> MultinomialFamily {
    let rows: Vec<usize> = (0..N_ROWS)
        .flat_map(|row| std::iter::repeat_n(row, WEIGHTS[row]))
        .collect();
    let n = rows.len();
    family(&rows, Array1::ones(n), armed)
}

fn evaluate(family: &MultinomialFamily, rho: &Array1<f64>) -> CustomFamilyJointHyperResult {
    let specs = family.build_block_specs();
    let layout = CustomFamilyHyperLayout::new(vec![vec![]; specs.len()], vec![], Array1::zeros(0))
        .expect("rho-only layout");
    let joint_specs =
        Arc::new(family.equivariant_class_penalty_specs().expect("class-function penalties"));
    let total_p = specs.iter().map(|spec| spec.design.ncols()).sum();
    let options = BlockwiseFitOptions {
        inner_tol: 1e-12,
        use_remlobjective: true,
        use_outer_hessian: true,
        compute_covariance: false,
        joint_penalties: Some(Arc::new(
            gam_problem::JointPenaltyBundle::new(joint_specs, rho.to_vec(), total_p)
                .expect("finite joint penalty strengths"),
        )),
        ..BlockwiseFitOptions::default()
    };
    let evaluation = evaluate_custom_family_joint_hyper(
        family,
        &specs,
        &options,
        &Array1::zeros(0),
        &layout,
        None,
        EvalMode::ValueGradientHessian,
    )
    .expect("converged fixed joint-rho Laplace mode");
    assert!(evaluation.inner_converged);
    evaluation
}

fn relative_gap(a: f64, b: f64) -> f64 {
    (a - b).abs() / (1.0 + a.abs().max(b.abs()))
}

fn check_weights_equal_repeated_rows(armed: bool) {
    let weighted = weighted_family(armed);
    let repeated = repeated_family(armed);
    let dimension = weighted.joint_smoothing_dimension();
    assert_eq!(dimension, repeated.joint_smoothing_dimension());
    // Both families solve the inner mode to `inner_tol`; the criterion agrees to
    // what that mode accuracy leaves in it.
    let tolerance = 1e-8;
    for shift in [0.0, -4.0] {
        let rho = Array1::from_shape_fn(dimension, |i| shift + (i as f64 - 2.5) * 0.7);
        let w = evaluate(&weighted, &rho);
        let r = evaluate(&repeated, &rho);
        eprintln!(
            "armed={armed} shift={shift} objective weighted={:.12e} repeated={:.12e}",
            w.objective, r.objective
        );
        eprintln!("  gradient weighted={:?}\n  gradient repeated={:?}", w.gradient, r.gradient);
        assert!(
            relative_gap(w.objective, r.objective) <= tolerance,
            "armed={armed}, rho={rho:?}: weighted objective {} != repeated {}",
            w.objective,
            r.objective
        );
        for axis in 0..dimension {
            assert!(
                relative_gap(w.gradient[axis], r.gradient[axis]) <= tolerance,
                "armed={armed}, rho={rho:?}, gradient axis {axis}: weighted {} != repeated {}",
                w.gradient[axis],
                r.gradient[axis]
            );
        }
        let (HessianValue::Dense(wh), HessianValue::Dense(rh)) = (&w.outer_hessian, &r.outer_hessian)
        else {
            panic!("small multinomial must supply dense exact outer curvature");
        };
        for (index, (a, b)) in wh.iter().zip(rh.iter()).enumerate() {
            assert!(
                relative_gap(*a, *b) <= tolerance,
                "armed={armed}, rho={rho:?}, Hessian entry {index}: weighted {a} != repeated {b}"
            );
        }
    }
}

#[test]
fn unaugmented_criterion_weights_equal_repeated_rows() {
    check_weights_equal_repeated_rows(false);
}

#[test]
fn jeffreys_criterion_weights_equal_repeated_rows() {
    check_weights_equal_repeated_rows(true);
}
