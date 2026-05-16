//! Ignored biobank-shape repro for the Bernoulli marginal-slope FLEX lane.
//!
//! Reproduces the CI lane formulas from `bench/biobank_scale/runner.py`:
//! mean:     `phenotype ~ link(type=probit) + sex + smooth(age_entry_std) + duchon(pc1_std, ..., pc16_std, centers=24, order=1, power=8, length_scale=1) + linkwiggle(internal_knots=8)`
//! logslope: `smooth(age_entry_std) + duchon(pc1_std, ..., pc16_std, centers=24, order=1, power=8, length_scale=1) + linkwiggle(internal_knots=8)`

use gam::basis::{
    BSplineBasisSpec, BSplineKnotSpec, CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder,
    DuchonOperatorPenaltySpec, SpatialIdentifiability,
};
use gam::bernoulli_marginal_slope::{
    BernoulliMarginalSlopeTermSpec, DeviationBlockConfig, LatentZPolicy,
};
use gam::custom_family::BlockwiseFitOptions;
use gam::families::lognormal_kernel::FrailtySpec;
use gam::smooth::{
    LinearCoefficientGeometry, LinearTermSpec, ShapeConstraint, SmoothBasisSpec, SmoothTermSpec,
    SpatialLengthScaleOptimizationOptions, TermCollectionSpec,
};
use gam::types::{InverseLink, LinkFunction, WigglePenaltyConfig};
use gam::{BernoulliMarginalSlopeFitRequest, FitRequest, FitResult, ResourcePolicy, fit_model};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::sync::Once;
use std::time::{Duration, Instant};

const SEED: u64 = 0xB10B_AA5C_A1E5_2026;
const FULL_N: usize = 60_000;
const SMALL_N: usize = 2_000;
const WALLCLOCK_BUDGET: Duration = Duration::from_secs(300);
const HESSIAN_BUILD_BUDGET: Duration = Duration::from_secs(30);

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
        let _ = log::set_logger(&LOGGER);
        log::set_max_level(log::LevelFilter::Info);
    });
}

struct BiobankProblem {
    data: Array2<f64>,
    spec: BernoulliMarginalSlopeTermSpec,
}

fn normal_cdf_approx(x: f64) -> f64 {
    0.5 * (1.0 + statrs::function::erf::erf(x / std::f64::consts::SQRT_2))
}

fn linkwiggle8() -> DeviationBlockConfig {
    WigglePenaltyConfig {
        num_internal_knots: 8,
        ..WigglePenaltyConfig::cubic_triple_operator_default()
    }
    .into()
}

fn linear(name: &str, feature_col: usize) -> LinearTermSpec {
    LinearTermSpec {
        name: name.to_string(),
        feature_col,
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
            },
        },
        shape: ShapeConstraint::None,
    }
}

fn duchon_pc_smooth(name: &str) -> SmoothTermSpec {
    SmoothTermSpec {
        name: name.to_string(),
        basis: SmoothBasisSpec::Duchon {
            feature_cols: (2..18).collect(),
            spec: DuchonBasisSpec {
                center_strategy: CenterStrategy::FarthestPoint { num_centers: 24 },
                length_scale: Some(1.0),
                power: 8,
                nullspace_order: DuchonNullspaceOrder::Linear,
                identifiability: SpatialIdentifiability::default(),
                aniso_log_scales: Some(vec![0.0; 16]),
                operator_penalties: DuchonOperatorPenaltySpec::default(),
            },
            input_scales: None,
        },
        shape: ShapeConstraint::None,
    }
}

fn build_problem(n: usize) -> BiobankProblem {
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
            duchon_pc_smooth("duchon_pc16_mean"),
        ],
    };
    let logslopespec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![
            age_smooth(1, "smooth_age_entry_std_logslope"),
            duchon_pc_smooth("duchon_pc16_logslope"),
        ],
    };

    BiobankProblem {
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
            score_warp: Some(linkwiggle8()),
            link_dev: Some(linkwiggle8()),
            latent_z_policy: LatentZPolicy::exploratory_fit_weighted(),
        },
    }
}

fn fit_biobank_problem(
    n: usize,
    options: BlockwiseFitOptions,
) -> gam::bernoulli_marginal_slope::BernoulliMarginalSlopeFitResult {
    let problem = build_problem(n);
    let request = FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
        data: problem.data.view(),
        spec: problem.spec,
        options,
        kappa_options: SpatialLengthScaleOptimizationOptions::default(),
        policy: ResourcePolicy::default_library(),
    });
    match fit_model(request).expect("biobank-shape Bernoulli marginal-slope fit should succeed") {
        FitResult::BernoulliMarginalSlope(out) => out,
        _ => panic!("wrong fit result variant"),
    }
}

#[test]
fn biobank_margslope_repro_completes_under_ci_budget() {
    init();
    let options = BlockwiseFitOptions::default();
    let start = Instant::now();
    let fit = fit_biobank_problem(FULL_N, options);
    let elapsed = start.elapsed();
    eprintln!(
        "[BIOBANK-MARGSLOPE-REPRO] n={} elapsed_s={:.3} outer_iters={} inner_cycles={} converged={}",
        FULL_N,
        elapsed.as_secs_f64(),
        fit.fit.outer_iterations,
        fit.fit.inner_cycles,
        fit.fit.outer_converged
    );
    assert!(
        elapsed < WALLCLOCK_BUDGET,
        "n={FULL_N} fit took {elapsed:?}, budget {WALLCLOCK_BUDGET:?}"
    );
    assert!(
        fit.fit.outer_converged,
        "expected the biobank-shape fit to converge within budget"
    );
    assert!(
        fit.fit.inner_cycles >= 1,
        "expected at least one logged joint-Newton cycle, got {}",
        fit.fit.inner_cycles
    );
}

#[test]
fn biobank_margslope_hessian_build_regression_baseline() {
    init();
    let options = BlockwiseFitOptions {
        outer_max_iter: 1,
        inner_max_cycles: 1,
        use_remlobjective: true,
        use_outer_hessian: true,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let start = Instant::now();
    let fit = fit_biobank_problem(FULL_N, options);
    let elapsed = start.elapsed();
    eprintln!(
        "[BIOBANK-MARGSLOPE-HESSIAN] n={} elapsed_s={:.3} inner_cycles={} beta_dim={}",
        FULL_N,
        elapsed.as_secs_f64(),
        fit.fit.inner_cycles,
        fit.fit.beta.len()
    );
    assert!(
        elapsed < HESSIAN_BUILD_BUDGET,
        "one Hessian-active outer probe took {elapsed:?}, budget {HESSIAN_BUILD_BUDGET:?}"
    );
}

#[test]
fn biobank_margslope_known_good_beta_numerical_fixture() {
    init();
    let options = BlockwiseFitOptions {
        outer_max_iter: 3,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let fit_a = fit_biobank_problem(SMALL_N, options.clone());
    let fit_b = fit_biobank_problem(SMALL_N, options);
    let beta_a = fit_a.fit.beta;
    let beta_b = fit_b.fit.beta;
    assert_eq!(beta_a.len(), beta_b.len());
    let max_abs = beta_a
        .iter()
        .zip(beta_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let checksum: f64 = beta_a
        .iter()
        .enumerate()
        .map(|(i, b)| (i as f64 + 1.0) * b)
        .sum();
    eprintln!(
        "[BIOBANK-MARGSLOPE-BETA] n={} beta_dim={} checksum={:.17e} repeated_fit_max_abs={:.3e}",
        SMALL_N,
        beta_a.len(),
        checksum,
        max_abs
    );
    assert!(
        max_abs <= 1e-9,
        "deterministic n={SMALL_N} beta fixture drifted across repeated fits: max_abs={max_abs:.3e}"
    );
}
