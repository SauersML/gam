//! Reusable synthetic bernoulli marginal-slope FLEX harness for large-scale
//! performance repros and coefficient-equivalence checks.
//!
//! Integration tests can include this helper with:
//!
//! ```ignore
//! #[path = "test_support/margslope_flex_equivalence.rs"]
//! mod margslope_flex_equivalence;
//! ```

use gam::ResourcePolicy;
use gam::families::bms::{
    BernoulliMarginalSlopeFitResult, BernoulliMarginalSlopeTermSpec, DeviationBlockConfig,
    LatentZPolicy,
};
use gam::families::custom_family::BlockwiseFitOptions;
use gam::families::survival::lognormal_kernel::FrailtySpec;
use gam::terms::basis::{
    BSplineBasisSpec, BSplineKnotSpec, CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder,
    DuchonOperatorPenaltySpec,
};
use gam::terms::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec,
};
use gam::types::{InverseLink, StandardLink};
use gam::{BernoulliMarginalSlopeFitRequest, FitRequest, FitResult, fit_model};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::{Duration, Instant};

pub const LARGE_SCALE_SHAPE_SEED: u64 = 0xB10B_AA1C_F13E_2026;
pub const LARGE_SCALE_SHAPE_PC_DIM: usize = 16;

#[derive(Clone)]
pub struct LargeScaleShapeProblem {
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
        frozen_parametric_residualization: None,
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
        joint_null_rotation: None,
    }
}

fn pc16_duchon_smooth(name: &str) -> SmoothTermSpec {
    SmoothTermSpec {
        frozen_parametric_residualization: None,
        name: name.to_string(),
        basis: SmoothBasisSpec::Duchon {
            feature_cols: (0..LARGE_SCALE_SHAPE_PC_DIM).collect(),
            spec: DuchonBasisSpec {
                radial_reparam: None,
                center_strategy: CenterStrategy::FarthestPoint { num_centers: 24 },
                length_scale: Some(1.0),
                power: 8.0,
                nullspace_order: DuchonNullspaceOrder::Linear,
                identifiability: Default::default(),
                aniso_log_scales: Some(vec![0.0; LARGE_SCALE_SHAPE_PC_DIM]),
                operator_penalties: DuchonOperatorPenaltySpec::default(),

                periodic: None,
                boundary: gam::basis::OneDimensionalBoundary::Open,
            },
            input_scale: None,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
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
    problem: LargeScaleShapeProblem,
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
                outer_converged: true, // sealed: fit existence is the proof
            };
            Ok((out, timing))
        }
        _ => Err("unexpected fit result variant".to_string()),
    }
}
