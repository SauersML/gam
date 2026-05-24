//! Reusable synthetic bernoulli marginal-slope FLEX harness for biobank-shape
//! performance repros and coefficient-equivalence checks.
//!
//! Integration tests can include this helper with:
//!
//! ```ignore
//! #[path = "test_support/margslope_flex_equivalence.rs"]
//! mod margslope_flex_equivalence;
//! ```

use gam::families::bernoulli_marginal_slope::{
    BernoulliMarginalSlopeFitResult, BernoulliMarginalSlopeTermSpec, DeviationBlockConfig,
    LatentZPolicy,
};
use gam::families::custom_family::BlockwiseFitOptions;
use gam::families::lognormal_kernel::FrailtySpec;
use gam::resource::ResourcePolicy;
use gam::terms::basis::{
    BSplineBasisSpec, BSplineKnotSpec, CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder,
    DuchonOperatorPenaltySpec,
};
use gam::terms::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec,
};
use gam::types::{InverseLink, LinkFunction};
use gam::{BernoulliMarginalSlopeFitRequest, FitRequest, FitResult, fit_model};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::{Duration, Instant};

pub const BIOBANK_SHAPE_SEED: u64 = 0xB10B_AA1C_F13E_2026;
pub const BIOBANK_SHAPE_PC_DIM: usize = 16;
pub const DEFAULT_REPRO_N: usize = 50_000;

#[derive(Clone)]
pub struct BiobankShapeProblem {
    pub data: Array2<f64>,
    pub spec: BernoulliMarginalSlopeTermSpec,
}

#[derive(Clone, Debug)]
pub struct FitTiming {
    pub elapsed: Duration,
    pub outer_iterations: usize,
    pub inner_cycles: usize,
    pub outer_converged: bool,
}

fn normal_pair(rng: &mut StdRng) -> (f64, f64) {
    let u1: f64 = rng.random_range(1e-12..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    (r * theta.cos(), r * theta.sin())
}

fn erf_approx(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + p * ax);
    sign * (1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp())
}

fn age_smooth(feature_col: usize, name: &str) -> SmoothTermSpec {
    SmoothTermSpec {
        name: name.to_string(),
        basis: SmoothBasisSpec::BSpline1D {
            feature_col,
            spec: BSplineBasisSpec {
                degree: 3,
                penalty_order: 2,
                knotspec: BSplineKnotSpec::Generate {
                    data_range: (-2.5, 2.5),
                    num_internal_knots: 8,
                },
                double_penalty: false,
                identifiability: Default::default(),
                boundary_conditions: Default::default(),
                boundary: gam::basis::OneDimensionalBoundary::Open,
            },
        },
        shape: ShapeConstraint::None,
    }
}

fn pc16_duchon_smooth(name: &str) -> SmoothTermSpec {
    SmoothTermSpec {
        name: name.to_string(),
        basis: SmoothBasisSpec::Duchon {
            feature_cols: (0..BIOBANK_SHAPE_PC_DIM).collect(),
            spec: DuchonBasisSpec {
                center_strategy: CenterStrategy::FarthestPoint { num_centers: 24 },
                length_scale: Some(1.0),
                power: 8.0,
                nullspace_order: DuchonNullspaceOrder::Linear,
                identifiability: Default::default(),
                aniso_log_scales: Some(vec![0.0; BIOBANK_SHAPE_PC_DIM]),
                operator_penalties: DuchonOperatorPenaltySpec::default(),

                periodic: None,
                boundary: gam::basis::OneDimensionalBoundary::Open,
            },
            input_scales: None,
        },
        shape: ShapeConstraint::None,
    }
}

pub fn build_biobank_shape_problem(n: usize) -> BiobankShapeProblem {
    let mut rng = StdRng::seed_from_u64(BIOBANK_SHAPE_SEED.wrapping_add(n as u64));
    let mut data = Array2::<f64>::zeros((n, BIOBANK_SHAPE_PC_DIM + 1));
    let mut z = Array1::<f64>::zeros(n);
    for i in 0..n {
        if i % 2 == 0 {
            let (a, b) = normal_pair(&mut rng);
            z[i] = a;
            if i + 1 < n {
                z[i + 1] = b;
            }
        }
        data[[i, BIOBANK_SHAPE_PC_DIM]] = rng.random_range(35.0..75.0);
        for j in 0..BIOBANK_SHAPE_PC_DIM {
            data[[i, j]] = 0.65 * z[i] + 0.35 * rng.random_range(-1.0..1.0) + (j as f64) * 0.01;
        }
    }
    for i in 0..n {
        data[[i, BIOBANK_SHAPE_PC_DIM]] = (data[[i, BIOBANK_SHAPE_PC_DIM]] - 55.0) / 12.0;
    }

    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let age = data[[i, BIOBANK_SHAPE_PC_DIM]];
        let pc_signal = (0..BIOBANK_SHAPE_PC_DIM)
            .map(|j| data[[i, j]] * (0.08 / ((j + 1) as f64).sqrt()))
            .sum::<f64>();
        let eta = -0.15 + 0.35 * age - 0.12 * age * age + pc_signal + 0.30 * z[i];
        let p = 0.5 * (1.0 + erf_approx(eta / std::f64::consts::SQRT_2));
        y[i] = if rng.random::<f64>() < p { 1.0 } else { 0.0 };
    }

    let marginalspec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![
            pc16_duchon_smooth("pc16_duchon_mean"),
            age_smooth(BIOBANK_SHAPE_PC_DIM, "age_entry_std_mean"),
        ],
    };
    let logslopespec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![
            pc16_duchon_smooth("pc16_duchon_logslope"),
            age_smooth(BIOBANK_SHAPE_PC_DIM, "age_entry_std_logslope"),
        ],
    };
    let dev_cfg = DeviationBlockConfig::triple_penalty_default();
    BiobankShapeProblem {
        data,
        spec: BernoulliMarginalSlopeTermSpec {
            y,
            weights: Array1::ones(n),
            z,
            base_link: InverseLink::Standard(LinkFunction::Probit),
            marginalspec,
            logslopespec,
            marginal_offset: Array1::zeros(n),
            logslope_offset: Array1::zeros(n),
            frailty: FrailtySpec::None,
            score_warp: Some(dev_cfg.clone()),
            link_dev: Some(dev_cfg),
            latent_z_policy: LatentZPolicy::exploratory_fit_weighted(),
        },
    }
}

pub fn cycle_capped_options(inner_max_cycles: usize) -> BlockwiseFitOptions {
    BlockwiseFitOptions {
        inner_max_cycles,
        outer_max_iter: 1,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    }
}

pub fn fit_problem(
    problem: BiobankShapeProblem,
    options: BlockwiseFitOptions,
) -> Result<(BernoulliMarginalSlopeFitResult, FitTiming), String> {
    let request = FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
        data: problem.data.view(),
        spec: problem.spec,
        options,
        kappa_options: SpatialLengthScaleOptimizationOptions::default(),
        policy: ResourcePolicy::default_library(),
    });
    let start = Instant::now();
    let result = fit_model(request)?;
    let elapsed = start.elapsed();
    match result {
        FitResult::BernoulliMarginalSlope(out) => {
            let timing = FitTiming {
                elapsed,
                #[cfg(test)]
                outer_iterations: out.fit.outer_iterations,
                inner_cycles: out.fit.inner_cycles,
                #[cfg(test)]
                outer_converged: out.fit.outer_converged,
            };
            Ok((out, timing))
        }
        _ => Err("unexpected fit result variant".to_string()),
    }
}
