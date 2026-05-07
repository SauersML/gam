//! Fast local reproducer for the FLEX bernoulli marginal-slope cycle-0 cost
//! cliff seen in the biobank-scale lane.
//!
//! Manual invocation (intentionally ignored; do not put this in normal CI):
//!
//! ```text
//! cargo test --release --test margslope_flex_biobank_repro \
//!     -- --ignored --nocapture margslope_flex_biobank_repro
//! ```
//!
//! The synthetic shape keeps the production code path active: probit
//! bernoulli marginal slope, score-warp and link-deviation FLEX blocks,
//! a joint 16D Duchon PC smooth (`centers=24`, `order=1`, `power=8`, `length_scale=1.0`), a
//! separate smooth age term, and a standard-normal latent `z`.  The fit is
//! capped at one outer evaluation/inner cycle so the printed wall time is a
//! local proxy for the silent `[PIRLS/blockwise joint-Newton] cycle 0/100`
//! region.  Profile this ignored test under `cargo test --release` to verify
//! whether `cubic_cell_kernel::bivariate_normal_cdf` remains dominant.
//!
//! May 2026 cache-fix validation notes from this container: the ignored repro
//! compiles with `cargo test --test margslope_flex_biobank_repro --no-run`;
//! collect release wall timings with the manual command above on a warmed build.

use gam::families::bernoulli_marginal_slope::{
    BernoulliMarginalSlopeTermSpec, DeviationBlockConfig, LatentZPolicy,
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
use std::time::Instant;

const SEED: u64 = 0xB10B_AA1C_F13E_2026;
const D_PC: usize = 16;

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

fn build_problem(n: usize) -> (Array2<f64>, BernoulliMarginalSlopeTermSpec) {
    let mut rng = StdRng::seed_from_u64(SEED.wrapping_add(n as u64));
    let mut data = Array2::<f64>::zeros((n, D_PC + 1));
    let mut z = Array1::<f64>::zeros(n);
    for i in 0..n {
        if i % 2 == 0 {
            let (a, b) = normal_pair(&mut rng);
            z[i] = a;
            if i + 1 < n {
                z[i + 1] = b;
            }
        }
        data[[i, D_PC]] = rng.random_range(35.0..75.0);
        for j in 0..D_PC {
            data[[i, j]] = 0.65 * z[i] + 0.35 * rng.random_range(-1.0..1.0) + (j as f64) * 0.01;
        }
    }
    let age_mean = 55.0;
    let age_sd = 12.0;
    for i in 0..n {
        data[[i, D_PC]] = (data[[i, D_PC]] - age_mean) / age_sd;
    }

    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let age = data[[i, D_PC]];
        let pc_signal = (0..D_PC)
            .map(|j| data[[i, j]] * (0.08 / ((j + 1) as f64).sqrt()))
            .sum::<f64>();
        let eta = -0.15 + 0.35 * age - 0.12 * age * age + pc_signal + 0.30 * z[i];
        let p = 0.5 * (1.0 + erf_approx(eta / std::f64::consts::SQRT_2));
        y[i] = if rng.random::<f64>() < p { 1.0 } else { 0.0 };
    }

    let pc_smooth = SmoothTermSpec {
        name: "pc16_duchon".to_string(),
        basis: SmoothBasisSpec::Duchon {
            feature_cols: (0..D_PC).collect(),
            spec: DuchonBasisSpec {
                center_strategy: CenterStrategy::FarthestPoint { num_centers: 24 },
                length_scale: Some(1.0),
                power: 8,
                nullspace_order: DuchonNullspaceOrder::Linear,
                identifiability: Default::default(),
                aniso_log_scales: Some(vec![0.0; D_PC]),
                operator_penalties: DuchonOperatorPenaltySpec::default(),
            },
            input_scales: None,
        },
        shape: ShapeConstraint::None,
    };
    let age_smooth = SmoothTermSpec {
        name: "age_entry_std".to_string(),
        basis: SmoothBasisSpec::BSpline1D {
            feature_col: D_PC,
            spec: BSplineBasisSpec {
                degree: 3,
                penalty_order: 2,
                knotspec: BSplineKnotSpec::Generate {
                    data_range: (-2.5, 2.5),
                    num_internal_knots: 8,
                },
                double_penalty: false,
                identifiability: Default::default(),
            },
        },
        shape: ShapeConstraint::None,
    };

    let marginalspec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![pc_smooth, age_smooth],
    };
    let logslopespec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![],
    };
    let dev_cfg = DeviationBlockConfig::triple_penalty_default();
    let spec = BernoulliMarginalSlopeTermSpec {
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
    };
    (data, spec)
}

#[test]
#[ignore]
fn margslope_flex_biobank_repro() {
    gam::init_parallelism();
    let n = std::env::var("GAM_MARGSLOPE_REPRO_N")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(6_000);
    let (data, spec) = build_problem(n);
    let options = BlockwiseFitOptions {
        inner_max_cycles: 1,
        outer_max_iter: 1,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let request = FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
        data: data.view(),
        spec,
        options,
        kappa_options: SpatialLengthScaleOptimizationOptions::default(),
        policy: ResourcePolicy::default_library(),
    });
    let start = Instant::now();
    let result = fit_model(request).expect("biobank-shape FLEX margslope repro fit");
    let elapsed = start.elapsed().as_secs_f64();
    let FitResult::BernoulliMarginalSlope(out) = result else {
        panic!("unexpected fit result variant");
    };
    eprintln!(
        "[MS-FLEX-BIOBANK-REPRO] n={} elapsed_s={:.3} outer_iters={} inner_cycles={} converged={}",
        n, elapsed, out.fit.outer_iterations, out.fit.inner_cycles, out.fit.outer_converged
    );
    assert!(out.fit.inner_cycles >= 1);
}
