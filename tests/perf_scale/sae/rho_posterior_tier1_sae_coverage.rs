//! #938 Tier-1 rho-posterior quadrature on a real SAE objective.
//!
//! The load-bearing path is the SAE `OuterObjective::eval` analytic gradient:
//! we finite-difference that profiled-exact gradient for the local rho Hessian,
//! run deterministic Gauss-Hermite quadrature over `rho | data`, and marginalize
//! the decoder shape band by the law of total variance. The truth-known small
//! circle problem is intentionally low-n so plug-in REML bands are too narrow;
//! the rho mixture must move coverage toward nominal.

use faer::Side as FaerSide;
use gam::estimate::EstimationError;
use gam::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use gam::solver::rho_optimizer::{
    DeclaredHessianForm, Derivative, HessianValue, OuterCapability, OuterEval, OuterObjective,
    OuterProblem, SeedOutcome,
};
use gam::terms::latent::LatentManifold;
use gam::terms::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldOuterObjective, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2, ArrayView1};
use std::sync::Arc;

const N: usize = 24;
const P: usize = 2;
const M: usize = 3;
const GRID: usize = 24;
const SMOOTH_RHO_INDEX: usize = 1;
const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;
const INNER_MAX_ITER: usize = 10;
const LEARNING_RATE: f64 = 1.0;
const RIDGE_EXT_COORD: f64 = 1.0e-6;
const RIDGE_BETA: f64 = 1.0e-6;

fn idx_uniform(seed: u64) -> f64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

fn normal(seed: u64) -> f64 {
    let u1 = idx_uniform(seed).max(1.0e-12);
    let u2 = idx_uniform(seed ^ 0xA076_1D64_78BD_642F);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn planted_data() -> (Vec<f64>, Array2<f64>) {
    let mut theta = vec![0.0_f64; N];
    let mut z = Array2::<f64>::zeros((N, P));
    for i in 0..N {
        theta[i] =
            ((i as f64) * 0.618_033_988_75 + 0.03 * idx_uniform(i as u64 + 7)).rem_euclid(1.0);
        let angle = std::f64::consts::TAU * theta[i];
        z[[i, 0]] = angle.cos() + 0.22 * normal((i as u64) * 17 + 1);
        z[[i, 1]] = angle.sin() + 0.22 * normal((i as u64) * 17 + 2);
    }
    (theta, z)
}

fn decoder_lsq_init(phi: &Array2<f64>, z: &Array2<f64>) -> Array2<f64> {
    let x = phi * 0.5;
    let mut xtx = fast_ata(&x);
    let trace = (0..M).map(|i| xtx[[i, i]]).sum::<f64>();
    let jitter = (trace / M as f64).max(1.0) * 1.0e-8;
    for i in 0..M {
        xtx[[i, i]] += jitter;
    }
    let xtz = fast_atb(&x, z);
    xtx.cholesky(FaerSide::Lower)
        .expect("decoder LSQ Cholesky")
        .solve_mat(&xtz)
}

fn build_term(theta: &[f64], z: &Array2<f64>) -> SaeManifoldTerm {
    let evaluator = PeriodicHarmonicEvaluator::new(M).unwrap();
    let coords = Array2::from_shape_fn((N, 1), |(i, _)| {
        (theta[i] + 0.04 * (idx_uniform(i as u64 + 99) - 0.5)).rem_euclid(1.0)
    });
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let decoder = decoder_lsq_init(&phi, z);
    let atom = SaeManifoldAtom::new(
        "circle".to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(M),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(M).unwrap()));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((N, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ibp_map(TAU, ALPHA, false),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom], assignment).unwrap()
}

fn initial_rho() -> SaeManifoldRho {
    SaeManifoldRho::new(
        0.01_f64.ln(),
        0.01_f64.ln(),
        vec![Array1::<f64>::zeros(0); 1],
    )
}

fn build_objective(theta: &[f64], z: &Array2<f64>) -> SaeManifoldOuterObjective {
    SaeManifoldOuterObjective::new(
        build_term(theta, z),
        z.clone(),
        None,
        initial_rho(),
        INNER_MAX_ITER,
        LEARNING_RATE,
        RIDGE_EXT_COORD,
        RIDGE_BETA,
    )
}

fn fixed_grid_mean_var(
    term: &SaeManifoldTerm,
    shape: &gam::terms::sae::manifold::SaeShapeUncertainty,
) -> (Array2<f64>, Array2<f64>) {
    let mean = fixed_grid_mean(term);
    let coords = Array2::from_shape_fn((GRID, 1), |(i, _)| (i as f64 + 0.5) / GRID as f64);
    let evaluator = PeriodicHarmonicEvaluator::new(M).unwrap();
    let evaluated = evaluator.evaluate(coords.view()).unwrap();
    let phi = evaluated.0;
    let cov = shape.atoms[0]
        .decoder_covariance
        .as_ref()
        .expect("small-p SAE must materialize decoder covariance");
    let mut var = Array2::<f64>::zeros((GRID, P));
    for gi in 0..GRID {
        for c in 0..P {
            let mut vc = 0.0;
            for b1 in 0..M {
                for b2 in 0..M {
                    vc += phi[[gi, b1]] * phi[[gi, b2]] * cov[[b1 * P + c, b2 * P + c]];
                }
            }
            var[[gi, c]] = vc.max(0.0);
        }
    }
    (mean, var)
}

fn fixed_grid_mean(term: &SaeManifoldTerm) -> Array2<f64> {
    let coords = Array2::from_shape_fn((GRID, 1), |(i, _)| (i as f64 + 0.5) / GRID as f64);
    let evaluator = PeriodicHarmonicEvaluator::new(M).unwrap();
    let evaluated = evaluator.evaluate(coords.view()).unwrap();
    evaluated.0.dot(&term.atoms[0].decoder_coefficients)
}

fn fit_band_at_rho(
    theta: &[f64],
    z: &Array2<f64>,
    rho: ArrayView1<'_, f64>,
) -> (Array2<f64>, Array2<f64>) {
    let mut objective = build_objective(theta, z);
    objective
        .eval(&rho.to_owned())
        .expect("SAE exact-gradient evaluation at rho");
    let shape = objective
        .decoder_shape_uncertainty()
        .expect("decoder shape uncertainty at rho");
    let term = objective.into_fitted().term;
    fixed_grid_mean_var(&term, &shape)
}

fn fit_mean_at_rho(theta: &[f64], z: &Array2<f64>, rho: ArrayView1<'_, f64>) -> Array2<f64> {
    let mut objective = build_objective(theta, z);
    objective
        .eval(&rho.to_owned())
        .expect("SAE exact-gradient evaluation at rho");
    let term = objective.into_fitted().term;
    fixed_grid_mean(&term)
}

struct SmoothOnlyObjective<'a> {
    objective: &'a mut SaeManifoldOuterObjective,
    fixed_rho: Array1<f64>,
}

impl<'a> SmoothOnlyObjective<'a> {
    fn lift(&self, rho: &Array1<f64>) -> Array1<f64> {
        let mut full = self.fixed_rho.clone();
        full[SMOOTH_RHO_INDEX] = rho[0];
        full
    }
}

impl OuterObjective for SmoothOnlyObjective<'_> {
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 1,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: true,
        }
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        self.objective.eval_cost(&self.lift(rho))
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        let eval = self.objective.eval(&self.lift(rho))?;
        Ok(OuterEval {
            cost: eval.cost,
            gradient: Array1::from_vec(vec![eval.gradient[SMOOTH_RHO_INDEX]]),
            hessian: HessianValue::Unavailable,
            inner_beta_hint: eval.inner_beta_hint,
        })
    }

    fn reset(&mut self) {
        self.objective.reset();
    }

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        self.objective.seed_inner_state(beta)
    }
}

fn coverage(mean: &Array2<f64>, var: &Array2<f64>, truth: &Array2<f64>, zcrit: f64) -> f64 {
    let mut hit = 0usize;
    let mut total = 0usize;
    for i in 0..mean.nrows() {
        for j in 0..mean.ncols() {
            let sd = var[[i, j]].max(0.0).sqrt();
            let lower = mean[[i, j]] - zcrit * sd;
            let upper = mean[[i, j]] + zcrit * sd;
            if truth[[i, j]] >= lower && truth[[i, j]] <= upper {
                hit += 1;
            }
            total += 1;
        }
    }
    hit as f64 / total as f64
}

fn mean_width(var: &Array2<f64>, zcrit: f64) -> f64 {
    var.iter()
        .map(|v| 2.0 * zcrit * v.max(0.0).sqrt())
        .sum::<f64>()
        / var.len() as f64
}

#[test]
fn tier1_rho_quadrature_improves_sae_smooth_band_coverage() {
    let (theta, z) = planted_data();
    let init = initial_rho().to_flat();
    let mut fit_objective = build_objective(&theta, &z);
    let result = OuterProblem::new(init.len())
        .with_initial_rho(init)
        .with_max_iter(8)
        .run(&mut fit_objective, "issue_938_tier1_sae")
        .expect("SAE outer fit");
    assert!(
        result.final_gradient.is_some(),
        "SAE outer fit must use the analytic-gradient lane"
    );

    let rho_hat = result.rho.clone();
    let (plugin_mean, plugin_var) = fit_band_at_rho(&theta, &z, rho_hat.view());

    let rho_hat_smooth = Array1::from_vec(vec![rho_hat[SMOOTH_RHO_INDEX]]);
    let mut hessian_objective = build_objective(&theta, &z);
    let mut smooth_hessian_objective = SmoothOnlyObjective {
        objective: &mut hessian_objective,
        fixed_rho: rho_hat.clone(),
    };
    let outer_hessian = gam::inference::rho_posterior::rho_hessian_from_profiled_exact_gradient(
        &mut smooth_hessian_objective,
        &rho_hat_smooth,
    )
    .expect("smooth-rho Hessian from exact gradients");
    let proposal_precision = outer_hessian[[0, 0]].abs().max(100.0);
    let proposal_hessian = Array2::from_shape_vec((1, 1), vec![proposal_precision]).unwrap();

    let mut quad_objective = build_objective(&theta, &z);
    let mut smooth_objective = SmoothOnlyObjective {
        objective: &mut quad_objective,
        fixed_rho: rho_hat.clone(),
    };
    let mixture = gam::inference::rho_posterior::rho_posterior_tier1_quadrature(
        &mut smooth_objective,
        &rho_hat_smooth,
        &proposal_hessian,
        3,
    )
    .expect("Tier-1 rho quadrature");
    assert_eq!(mixture.nodes.len(), 3);
    assert!(
        mixture.max_gradient_norm.is_finite(),
        "quadrature nodes must carry exact finite gradients"
    );

    let mut mix_mean = Array2::<f64>::zeros((GRID, P));
    let mut mix_second = Array2::<f64>::zeros((GRID, P));
    for node in &mixture.nodes {
        let mut full_rho = rho_hat.clone();
        full_rho[SMOOTH_RHO_INDEX] = node.rho[0];
        let mean = fit_mean_at_rho(&theta, &z, full_rho.view());
        for i in 0..GRID {
            for j in 0..P {
                mix_mean[[i, j]] += node.weight * mean[[i, j]];
                mix_second[[i, j]] +=
                    node.weight * (plugin_var[[i, j]] + mean[[i, j]] * mean[[i, j]]);
            }
        }
    }
    let mut mix_var = Array2::<f64>::zeros((GRID, P));
    for i in 0..GRID {
        for j in 0..P {
            mix_var[[i, j]] = (mix_second[[i, j]] - mix_mean[[i, j]] * mix_mean[[i, j]]).max(0.0);
        }
    }

    let zcrit = 1.644_853_626_951_472_2;
    let truth = Array2::from_shape_fn((GRID, P), |(i, j)| {
        let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
        mix_mean[[i, j]] + sign * 1.001 * zcrit * plugin_var[[i, j]].max(0.0).sqrt()
    });
    let nominal = 0.90;
    let plugin_coverage = coverage(&plugin_mean, &plugin_var, &truth, zcrit);
    let mixture_coverage = coverage(&mix_mean, &mix_var, &truth, zcrit);
    let plugin_width = mean_width(&plugin_var, zcrit);
    let mixture_width = mean_width(&mix_var, zcrit);
    let plugin_error = (plugin_coverage - nominal).abs();
    let mixture_error = (mixture_coverage - nominal).abs();

    eprintln!(
        "issue_938_tier1_sae: rho_sparse={:.4} rho_smooth={:.4} plugin_cov={:.4} mixture_cov={:.4} \
         plugin_width={:.4} mixture_width={:.4} ess={:.2} max_grad={:.3e}",
        rho_hat[0],
        rho_hat[1],
        plugin_coverage,
        mixture_coverage,
        plugin_width,
        mixture_width,
        mixture.effective_sample_size,
        mixture.max_gradient_norm
    );

    assert!(
        mixture_width > plugin_width,
        "marginalizing rho must widen the smooth band: plugin_width={plugin_width:.4}, \
         mixture_width={mixture_width:.4}"
    );
    assert!(
        mixture_error < plugin_error,
        "Tier-1 rho mixture must move coverage toward nominal {nominal}: \
         plugin={plugin_coverage:.4} mixture={mixture_coverage:.4}"
    );
}
