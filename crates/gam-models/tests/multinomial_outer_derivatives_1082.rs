//! Finite differences of the production joint-penalty multinomial Laplace criterion.
//! Joint rho coordinates travel in the penalty bundle; block-local rho is empty.

use gam_custom_family::{
    BlockwiseFitOptions, CustomFamilyHyperLayout, evaluate_custom_family_joint_hyper,
};
use gam_models::MultinomialFamily;
use gam_problem::{EvalMode, HessianValue, PenaltyMatrix};
use ndarray::{Array1, Array2, array};
use std::sync::Arc;

fn quasi_separated_family(armed: bool) -> MultinomialFamily {
    let n = 90;
    let mut design = Array2::zeros((n, 3));
    let mut response = Array2::zeros((n, 3));
    for row in 0..n {
        let x = -2.0 + 4.0 * row as f64 / (n - 1) as f64;
        design[[row, 0]] = 1.0;
        design[[row, 1]] = x;
        design[[row, 2]] = x * x - 4.0 / 3.0;
        let class = if row % 13 == 0 {
            (row / 13) % 3
        } else if x < -0.5 {
            0
        } else if x > 0.5 {
            1
        } else {
            2
        };
        response[[row, class]] = 1.0;
    }
    let penalties = vec![
        PenaltyMatrix::Dense(array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        PenaltyMatrix::Dense(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    ];
    MultinomialFamily::new(response, Array1::ones(n), 3, Arc::new(design), Arc::new(penalties))
        .expect("finite three-class polynomial design")
        .with_joint_jeffreys_term(armed)
}

fn check_outer_derivatives(armed: bool) {
    faer::set_global_parallelism(faer::Par::rayon(0));
    let family = quasi_separated_family(armed);
    let specs = family.build_block_specs();
    let options = BlockwiseFitOptions {
        inner_tol: 1e-10,
        use_remlobjective: true,
        use_outer_hessian: true,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let layout = CustomFamilyHyperLayout::new(vec![vec![]; specs.len()], vec![], Array1::zeros(0))
        .expect("rho-only layout");
    let joint_specs = Arc::new(family.equivariant_class_penalty_specs().expect("class-function penalties"));
    let total_p = specs.iter().map(|spec| spec.design.ncols()).sum();
    let evaluate = |rho: &Array1<f64>,
                    warm: Option<&gam_custom_family::CustomFamilyWarmStart>,
                    mode: EvalMode| {
        let mut point_options = options.clone();
        point_options.joint_penalties = Some(Arc::new(
            gam_problem::JointPenaltyBundle::new(Arc::clone(&joint_specs), rho.to_vec(), total_p)
                .expect("finite joint penalty strengths"),
        ));
        evaluate_custom_family_joint_hyper(
            &family, &specs, &point_options, &Array1::zeros(0), &layout, warm, mode,
        ).expect("converged fixed joint-rho Laplace mode")
    };
    let dimension = family.joint_smoothing_dimension();
    assert_eq!(dimension, 6, "two penalties across three centered class contrasts");
    for shift in [0.0, -3.0] {
        let rho = Array1::from_shape_fn(dimension, |i| shift + (i as f64 - 2.5) * 0.3);
        let center = evaluate(&rho, None, EvalMode::ValueGradientHessian);
        assert_eq!(center.gradient.len(), dimension);
        assert!(center.inner_converged);
        let HessianValue::Dense(hessian) = &center.outer_hessian else {
            panic!("small multinomial must supply dense exact outer curvature");
        };
        let h = 2e-4;
        for axis in 0..dimension {
            let mut plus_rho = rho.clone();
            let mut minus_rho = rho.clone();
            plus_rho[axis] += h;
            minus_rho[axis] -= h;
            let plus = evaluate(&plus_rho, Some(&center.warm_start), EvalMode::ValueAndGradient);
            let minus = evaluate(&minus_rho, Some(&center.warm_start), EvalMode::ValueAndGradient);
            assert!(plus.inner_converged && minus.inner_converged);
            let fd = (plus.objective - minus.objective) / (2.0 * h);
            let analytic = center.gradient[axis];
            eprintln!("armed={armed} shift={shift} axis={axis} gradient={analytic:.10e} fd={fd:.10e}");
            assert!((analytic - fd).abs() <= 1e-5 * (1.0 + analytic.abs().max(fd.abs())),
                "armed={armed}, rho={rho:?}, gradient axis {axis}: analytic={analytic}, FD={fd}");
            for row in 0..dimension {
                let fd = (plus.gradient[row] - minus.gradient[row]) / (2.0 * h);
                let analytic = hessian[[row, axis]];
                assert!((analytic - fd).abs() <= 2e-4 * (1.0 + analytic.abs().max(fd.abs())),
                    "armed={armed}, rho={rho:?}, Hessian ({row},{axis}): analytic={analytic}, FD={fd}");
            }
        }
    }
}

#[test]
fn unaugmented_outer_gradient_and_hessian_match_the_criterion() {
    check_outer_derivatives(false);
}

#[test]
fn jeffreys_outer_gradient_and_hessian_match_the_criterion() {
    check_outer_derivatives(true);
}
