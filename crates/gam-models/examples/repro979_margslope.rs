//! Full-fit reproduction for #979 through the model service. Mirrors the integration test
//! `tests/bug_hunt_979_margslope_matern_slope_slowdown.rs`.
//!
//! Run: `cargo run -p gam-models --profile release-dev --example repro979_margslope -- 1500 4 1`

use gam_runtime::resource::ResourcePolicy;
use gam_models::bms::{BernoulliMarginalSlopeTermSpec, LatentZPolicy};
use gam_custom_family::BlockwiseFitOptions;
use gam_models::survival::lognormal_kernel::FrailtySpec;
use gam_terms::basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu};
use gam_terms::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec,
};
use gam_spec::{InverseLink, StandardLink};
use gam_models::fit_orchestration::{BernoulliMarginalSlopeFitRequest, FitRequest, FitResult, fit_model};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

fn matern_smooth(name: &str, centers: usize, kappa_auto: bool) -> SmoothTermSpec {
    SmoothTermSpec {
        frozen_parametric_residualization: None,
        name: name.to_string(),
        basis: SmoothBasisSpec::Matern {
            feature_cols: vec![0, 1],
            spec: MaternBasisSpec {
                center_strategy: CenterStrategy::EqualMass {
                    num_centers: centers,
                },
                periodic: None,
                length_scale: if kappa_auto {
                    gam_terms::basis::MaternLengthScale::auto()
                } else {
                    gam_terms::basis::MaternLengthScale::fixed(1.0)
                },
                nu: MaternNu::ThreeHalves,
                include_intercept: false,
                double_penalty: false,
                identifiability: MaternIdentifiability::default(),
                aniso_log_scales: None,
            },
            input_scale: None,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    }
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
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

fn build(
    n: usize,
    centers: usize,
    kappa_auto: bool,
) -> (Array2<f64>, BernoulliMarginalSlopeTermSpec) {
    let mut rng = StdRng::seed_from_u64(0x9797_0001);
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        data[[i, 0]] = rng.random_range(-2.0..2.0);
        data[[i, 1]] = rng.random_range(-2.0..2.0);
    }
    let mut z = Array1::<f64>::zeros(n);
    let mut i = 0usize;
    while i < n {
        let u1: f64 = rng.random_range(1e-12..1.0);
        let u2: f64 = rng.random_range(0.0..1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        z[i] = r * theta.cos();
        if i + 1 < n {
            z[i + 1] = r * theta.sin();
        }
        i += 2;
    }
    let true_eta: Array1<f64> = Array1::from_iter((0..n).map(|i| {
        let p1 = data[[i, 0]];
        let p2 = data[[i, 1]];
        let f = (0.8 * p1).sin() + 0.5 * (0.6 * p2).cos();
        let slope = 0.3 + 0.2 * p1;
        f + slope * z[i]
    }));
    let y = Array1::from_iter(true_eta.iter().map(|&eta| {
        let p = 0.5 * (1.0 + erf_approx(eta / std::f64::consts::SQRT_2));
        if rng.random::<f64>() < p { 1.0 } else { 0.0 }
    }));

    let marginalspec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![matern_smooth("f_pc", centers, kappa_auto)],
    };
    let slopespec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![matern_smooth("ls_pc", centers, kappa_auto)],
    };
    let spec = BernoulliMarginalSlopeTermSpec {
        y,
        weights: Array1::ones(n),
        z,
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginalspec,
        slopespec,
        marginal_offset: Array1::<f64>::zeros(n),
        slope_offset: Array1::<f64>::zeros(n),
        frailty: FrailtySpec::None,
        score_warp: None,
        link_dev: None,
        latent_z_policy: LatentZPolicy::exploratory_fit_weighted(),
        score_influence_jacobian: None,
    };
    (data, spec)
}

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

/// Positional CLI args: `repro979_margslope [n] [centers]`. Absent args keep the
/// defaults so the example still runs bare. (Avoids `env::var`, which the
/// repo-wide scanner bans — examples read tuning knobs from argv.)
fn arg_usize(idx: usize, default: usize) -> Result<usize, std::num::ParseIntError> {
    std::env::args().nth(idx).map_or(Ok(default), |s| s.parse())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    faer::set_global_parallelism(faer::Par::rayon(0));
    match log::set_logger(&LOGGER) {
        Ok(()) => log::set_max_level(log::LevelFilter::Info),
        Err(err) => eprintln!("keeping the already-installed logger: {err}"),
    }
    let n = arg_usize(1, 1500)?;
    let centers = arg_usize(2, 4)?;
    let kappa_auto = match arg_usize(3, 1)? {
        0 => false,
        1 => true,
        _ => return Err("kappa_auto must be 0 or 1".into()),
    };
    if n == 0 || std::env::args().len() > 4 {
        return Err("usage: repro979_margslope [positive n] [centers] [kappa_auto: 0|1]".into());
    }
    let (data, spec) = build(n, centers, kappa_auto);
    let request = FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
        data: data.view(),
        spec,
        options: BlockwiseFitOptions::default(),
        kappa_options: SpatialLengthScaleOptimizationOptions::default(),
        policy: ResourcePolicy::default_library(),
    });
    let start = Instant::now();
    let result = fit_model(request);
    let elapsed = start.elapsed().as_secs_f64();
    match result {
        Ok(FitResult::BernoulliMarginalSlope(out)) => {
            eprintln!(
                "[979-REPRO] n={n} centers={centers} kappa_auto={kappa_auto} total_s={elapsed:.2} outer_iters={} inner_cycles={} converged=certified",
                out.fit.outer_iterations, out.fit.inner_cycles
            );
        }
        Ok(_) => return Err("[979-REPRO] wrong FitResult variant".into()),
        Err(e) => {
            return Err(format!("[979-REPRO] kappa_auto={kappa_auto} fit failed after {elapsed:.2}s: {e}").into());
        }
    }
    Ok(())
}
