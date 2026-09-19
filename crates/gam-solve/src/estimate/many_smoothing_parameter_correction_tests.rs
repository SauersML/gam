//! The smoothing-parameter correction `V_c = V_β + J V_ρ Jᵀ` has no dimension
//! gate: a model with many smoothing parameters carries it exactly as a model
//! with one does.
//!
//! `J = ∂β̂/∂ρ` comes from implicit differentiation of the inner optimality
//! condition and `V_ρ` is the certified inverse of the LAML Hessian in ρ, so
//! neither needs sampling and neither has a cost that grows beyond the fit's
//! own. This pins that on a ten-term additive model (ten smoothing
//! parameters): the published corrected covariance differs from the
//! conditional one, the first-order term is the positive semidefinite
//! `J V_ρ Jᵀ`, and it is not confined to the few terms a cost gate would keep.

use super::*;
use crate::model_types::SmoothingCorrectionMethod;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N: usize = 400;
const TERMS: usize = 10;
const BASIS: usize = 6;
const NOISE_SD: f64 = 0.6;

fn box_muller(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-12);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// A mix of wiggly, linear, bump and null effects, so the smoothing
/// parameters range from well identified to weakly identified.
fn truth(term: usize, x: f64) -> f64 {
    match term % 4 {
        0 => (2.5 * x + 0.2 * term as f64).sin(),
        1 => 0.6 * x,
        2 => 0.8 * (-4.0 * (x - 0.1 * (term % 3) as f64).powi(2)).exp() - 0.35,
        _ => 0.0,
    }
}

/// Ten-term additive design: an unpenalized intercept, then per term a cosine
/// basis `cos(πk(x+1)/2)`, `k = 1..=BASIS`, penalized by `k⁴` — the
/// integrated squared second derivative of that basis, up to a constant.
fn additive_fixture() -> (Array2<f64>, Array1<f64>, Vec<BlockwisePenalty>) {
    let mut rng = StdRng::seed_from_u64(20_260_919);
    let p = 1 + TERMS * BASIS;
    let mut x = Array2::<f64>::zeros((N, p));
    let mut y = Array1::<f64>::zeros(N);
    for i in 0..N {
        x[[i, 0]] = 1.0;
        let mut mean = 0.0;
        for term in 0..TERMS {
            let covariate = 2.0 * rng.random::<f64>() - 1.0;
            mean += truth(term, covariate);
            for k in 1..=BASIS {
                let column = 1 + term * BASIS + (k - 1);
                x[[i, column]] =
                    (std::f64::consts::PI * k as f64 * (covariate + 1.0) / 2.0).cos();
            }
        }
        y[i] = mean + NOISE_SD * box_muller(&mut rng);
    }
    let penalties = (0..TERMS)
        .map(|term| {
            let start = 1 + term * BASIS;
            let mut s = Array2::<f64>::zeros((BASIS, BASIS));
            for k in 1..=BASIS {
                s[[k - 1, k - 1]] = (k as f64).powi(4);
            }
            BlockwisePenalty::new(start..start + BASIS, s)
        })
        .collect();
    (x, y, penalties)
}

#[test]
fn ten_smoothing_parameters_carry_the_rho_correction() {
    let (x, y, penalties) = additive_fixture();
    let weights = Array1::<f64>::ones(N);
    let offset = Array1::<f64>::zeros(N);
    let opts = FitOptions {
        compute_inference: true,
        max_iter: 200,
        tol: 1e-10,
        nullspace_dims: vec![0; TERMS],
        ..FitOptions::default()
    };
    let fit = fit_gamwith_heuristic_log_lambdas(
        x,
        y.view(),
        weights.view(),
        offset.view(),
        &penalties,
        None,
        LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        &opts,
    )
    .expect("ten-term Gaussian additive fit");
    assert_eq!(fit.lambdas.len(), TERMS, "one smoothing parameter per term");

    let conditional = fit.beta_covariance().expect("conditional covariance V_β");
    let corrected = fit
        .beta_covariance_corrected()
        .expect("a ten-smoothing-parameter fit must publish V_c");
    let p = conditional.nrows();
    assert_eq!(corrected.dim(), (p, p));

    // V_c ≠ V_β: the correction is present, not a zero placeholder.
    let difference: f64 = corrected
        .iter()
        .zip(conditional.iter())
        .map(|(c, v)| (c - v).powi(2))
        .sum::<f64>()
        .sqrt();
    let scale: f64 = conditional.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        difference > 1e-3 * scale,
        "V_c must differ from V_β with {TERMS} smoothing parameters: \
         ‖V_c − V_β‖_F = {difference:.3e}, ‖V_β‖_F = {scale:.3e}"
    );

    // The first-order term is J V_ρ Jᵀ over the identified ρ subspace: exactly
    // symmetric positive semidefinite and computed at every dimension.
    let first_order = fit
        .smoothing_correction_first_order()
        .expect("first-order J V_ρ Jᵀ is retained at every rho dimension");
    match fit.smoothing_correction_method_first_order() {
        Some(SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
            active_rank,
            rho_dimension,
        }) => {
            assert_eq!(rho_dimension, TERMS);
            assert!(active_rank >= 1 && active_rank <= TERMS);
        }
        other => panic!("first-order method must be the identified-subspace IFT, got {other:?}"),
    }
    let first_order_scale: f64 = first_order.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(first_order_scale > 1e-3 * scale);
    for a in 0..p {
        for b in 0..p {
            assert!(
                (first_order[[a, b]] - first_order[[b, a]]).abs()
                    <= 1e-12 * first_order_scale,
                "J V_ρ Jᵀ must be symmetric"
            );
        }
    }
    let (eigenvalues, _) = first_order
        .eigh(faer::Side::Lower)
        .expect("symmetric eigendecomposition");
    let most_negative = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        most_negative >= -1e-10 * first_order_scale,
        "J V_ρ Jᵀ must be positive semidefinite, smallest eigenvalue {most_negative:.3e}"
    );

    // Not confined to a handful of terms: every term whose smoothing parameter
    // is resolved gets variance inflation on its own block. Count the terms
    // whose block trace moves by more than roundoff.
    let inflated_terms = (0..TERMS)
        .filter(|term| {
            let start = 1 + term * BASIS;
            let trace: f64 = (start..start + BASIS).map(|j| first_order[[j, j]]).sum();
            let base: f64 = (start..start + BASIS).map(|j| conditional[[j, j]]).sum();
            trace > 1e-6 * base
        })
        .count();
    assert!(
        inflated_terms > 4,
        "the correction must reach more terms than a four-dimensional gate would \
         keep: only {inflated_terms} of {TERMS} term blocks are inflated"
    );
}
