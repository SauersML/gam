//! Reproduce the #2668 seed/certificate disagreement through the solver crate,
//! so an inner-accuracy edit can be checked without rebuilding model families.

use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_solve::estimate::{FitOptions, fit_gamwith_heuristic_lambdas};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2, array};

fn main() {
    gam_runtime::test_support::install_diagnostic_logger();
    let x = array![[1.0, -1.0], [1.0, -0.2], [1.0, 0.4], [1.0, 1.2]];
    let y = array![0.0, 0.0, 1.0, 1.0];
    let weights = Array1::ones(4);
    let offset = Array1::zeros(4);
    let penalties = vec![BlockwisePenalty::new(0..2, Array2::eye(2))];
    let options = FitOptions {
        compute_inference: true,
        max_iter: 40,
        nullspace_dims: vec![0],
        ..FitOptions::default()
    };
    let fit = fit_gamwith_heuristic_lambdas(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &penalties,
        Some(&[2.5]),
        LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        &options,
    )
    .expect("the seeded fit must pass the terminal REML certificate");
    assert_eq!(fit.log_lambdas.len(), 1);
    assert!(fit.log_lambdas[0].is_finite() && fit.deviance.is_finite());
    println!("rho={:.16e} deviance={:.16e}", fit.log_lambdas[0], fit.deviance);
}
