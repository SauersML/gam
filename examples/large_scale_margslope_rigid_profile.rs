//! Large-scale RIGID Bernoulli marginal-slope repro for #979.
//!
//! This reproduces the `rust_margslope_aniso_duchon16d_rigid` CI benchmark
//! shape — the one that hits the 2400s Large-scale timeout (rc=124) — which is
//! NOT covered by `large_scale_margslope_repro` (that guard exercises the
//! order=1 / power=8 / linkwiggle path that converges in 300s).
//!
//! Rigid differs from the linkwiggle path in three ways that route it through
//! the `hessian=Unavailable -> outer BFGS` first-order bridge + the slow
//! joint-Newton phase:
//!   * `DuchonNullspaceOrder::Zero` (order=0, affine/StandardNormal latent),
//!   * `power = 9.0`,
//!   * NO `score_warp` / NO `link_dev` (no linkwiggle blocks).
//!
//! This is a profiling/repro harness, not a CI budget gate. It lives as a
//! Cargo example (`cargo run --release --example large_scale_margslope_rigid_profile`)
//! so it stays compile-checked without entering the `cargo test` budget — the
//! repo bans `#[ignore]` (build.rs gate), so an ignored test is not an option;
//! an example is the sanctioned "run explicitly on a compute node" home.

use gam::basis::{
    BSplineBasisSpec, BSplineKnotSpec, CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder,
    DuchonOperatorPenaltySpec, SpatialIdentifiability,
};
use gam::custom_family::BlockwiseFitOptions;
use gam::families::bms::{BernoulliMarginalSlopeTermSpec, LatentZPolicy};
use gam::families::survival::lognormal_kernel::FrailtySpec;
use gam::smooth::{
    LinearCoefficientGeometry, LinearTermSpec, ShapeConstraint, SmoothBasisSpec, SmoothTermSpec,
    SpatialLengthScaleOptimizationOptions, TermCollectionSpec,
};
use gam::types::{InverseLink, StandardLink};
use gam::{BernoulliMarginalSlopeFitRequest, FitRequest, FitResult, ResourcePolicy, fit_model};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::sync::Once;
use std::time::Instant;

const SEED: u64 = 0xB10B_AA5C_A1E5_2026;
// Large enough to enter the slow regime, small enough to eu-stack on a node.
const PROFILE_N: usize = 150_000;

struct StderrInfoLogger;
impl log::Log for StderrInfoLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{}", record.args());
        }
    }
    fn flush(&self) {}
}
static LOGGER: StderrInfoLogger = StderrInfoLogger;
static INIT_LOGGER: Once = Once::new();

fn init() {
    gam::init_parallelism();
    INIT_LOGGER.call_once(|| {
        if log::set_logger(&LOGGER).is_ok() {
            log::set_max_level(log::LevelFilter::Info);
        }
    });
}

fn normal_cdf_approx(x: f64) -> f64 {
    0.5 * (1.0 + statrs::function::erf::erf(x / std::f64::consts::SQRT_2))
}

fn linear(name: &str, feature_col: usize) -> LinearTermSpec {
    LinearTermSpec {
        name: name.to_string(),
        feature_col,
        feature_cols: vec![feature_col],
        double_penalty: true,
        coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
        coefficient_min: None,
        coefficient_max: None,
    }
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
                    data_range: (-4.5, 4.5),
                    num_internal_knots: 8,
                },
                double_penalty: false,
                identifiability: Default::default(),
                boundary: Default::default(),
                boundary_conditions: Default::default(),
            },
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    }
}

/// Rigid duchon: order=0 (Zero nullspace), power=9.
fn duchon_pc_smooth_rigid(name: &str) -> SmoothTermSpec {
    SmoothTermSpec {
        name: name.to_string(),
        basis: SmoothBasisSpec::Duchon {
            feature_cols: (2..18).collect(),
            spec: DuchonBasisSpec {
                center_strategy: CenterStrategy::FarthestPoint { num_centers: 24 },
                length_scale: Some(1.0),
                power: 9.0,
                nullspace_order: DuchonNullspaceOrder::Zero,
                identifiability: SpatialIdentifiability::default(),
                aniso_log_scales: Some(vec![0.0; 16]),
                operator_penalties: DuchonOperatorPenaltySpec::default(),
                periodic: None,
                boundary: Default::default(),
            },
            input_scales: None,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    }
}

fn build_rigid_problem(n: usize) -> (Array2<f64>, BernoulliMarginalSlopeTermSpec) {
    let mut rng = StdRng::seed_from_u64(SEED.wrapping_add(n as u64));
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut data = Array2::<f64>::zeros((n, 18));
    let mut z = Array1::<f64>::zeros(n);
    let mut y = Array1::<f64>::zeros(n);

    for i in 0..n {
        let sex = if rng.random::<f64>() < 0.5 { 1.0 } else { 0.0 };
        let age = normal.sample(&mut rng);
        data[[i, 0]] = sex;
        data[[i, 1]] = age;
        let mut pc_signal = 0.0;
        for j in 0..16 {
            let pc = normal.sample(&mut rng);
            data[[i, 2 + j]] = pc;
            pc_signal += pc * ((j + 1) as f64).recip() * if j % 2 == 0 { 1.0 } else { -1.0 };
        }
        let score = normal.sample(&mut rng);
        z[i] = score;
        let eta = -0.6744897501960817 + 0.18 * sex + 0.12 * age + 0.035 * pc_signal;
        let p = normal_cdf_approx(eta + 0.08 * score).clamp(1e-9, 1.0 - 1e-9);
        y[i] = if rng.random::<f64>() < p { 1.0 } else { 0.0 };
    }

    let marginalspec = TermCollectionSpec {
        linear_terms: vec![linear("sex", 0)],
        random_effect_terms: vec![],
        smooth_terms: vec![
            age_smooth(1, "smooth_age_entry_std_mean"),
            duchon_pc_smooth_rigid("duchon_pc16_mean"),
        ],
    };
    let logslopespec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![
            age_smooth(1, "smooth_age_entry_std_logslope"),
            duchon_pc_smooth_rigid("duchon_pc16_logslope"),
        ],
    };

    let spec = BernoulliMarginalSlopeTermSpec {
        y,
        weights: Array1::ones(n),
        z,
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginalspec,
        logslopespec,
        marginal_offset: Array1::zeros(n),
        logslope_offset: Array1::zeros(n),
        frailty: FrailtySpec::None,
        // Rigid: no score warp, no link deviation (this is what makes the
        // latent z-map affine and routes the outer loop through the
        // hessian=Unavailable -> BFGS first-order bridge).
        score_warp: None,
        link_dev: None,
        latent_z_policy: LatentZPolicy::exploratory_fit_weighted(),
        score_influence_jacobian: None,
    };
    (data, spec)
}

fn main() {
    init();
    let (data, spec) = build_rigid_problem(PROFILE_N);
    let request = FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
        data: data.view(),
        spec,
        options: BlockwiseFitOptions::default(),
        kappa_options: SpatialLengthScaleOptimizationOptions::default(),
        policy: ResourcePolicy::default_library(),
    });
    let start = Instant::now();
    let fit = match fit_model(request).expect("rigid large-scale fit should not error") {
        FitResult::BernoulliMarginalSlope(out) => out,
        _ => panic!("wrong fit result variant"),
    };
    let elapsed = start.elapsed();
    eprintln!(
        "[RIGID-PROFILE] n={} elapsed_s={:.3} outer_iters={} inner_cycles={} converged={}",
        PROFILE_N,
        elapsed.as_secs_f64(),
        fit.fit.outer_iterations,
        fit.fit.inner_cycles,
        fit.fit.outer_converged,
    );
}
