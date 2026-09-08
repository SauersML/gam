//! #2735 probe: is the held-out reconstruction of `large_scale_reml_stress_main`
//! bounded by the ANISOTROPY the fit is never allowed to learn?
//!
//! The fixture seeds `aniso_log_scales: Some(vec![0.0; d])` — the "initialize
//! me" sentinel — and `auto_seed_aniso_contrasts` replaces it, on every Duchon
//! basis build, with `initial_aniso_contrasts(centers)`: contrasts derived from
//! the per-axis SPREAD OF THE KNOT CLOUD. On this design the inputs are iid
//! `N(0, I)`, so that spread is isotropic up to sampling noise and the seeded η
//! is noise. `spatial_term_uses_per_axis_psi` then returns `false` for every
//! `SmoothBasisSpec::Duchon`, so η is never enrolled as an outer ψ coordinate
//! and no REML iteration can move it.
//!
//! The truth's non-linear content is NOT isotropic: `0.4·sin(π x₀)` lives
//! entirely on axis 0. So the question this probe answers is whether the REML
//! criterion itself PREFERS a per-axis η that the outer loop cannot reach.
//!
//! Run:
//!   cargo run --release --example probe_2735_aniso_axis -- [n] [k] [d] [sweep]
//!
//! For each η on the sweep it prints the fit's REML criterion, its outer
//! iteration count against the configured cap, the held-out `rel_l2`, and the
//! η the design actually froze (which is NOT the η requested when the request
//! is the all-zero sentinel).

use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    duchon_max_active_operator_derivative_order, resolve_duchon_orders,
};
use gam::estimate::FitOptions;
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec, build_term_collection_design, fit_term_collection_forspec,
    fit_term_collectionwith_spatial_length_scale_optimization,
    freeze_term_collection_from_design, get_spatial_aniso_log_scales, get_spatial_length_scale,
};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2, ArrayView2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::time::Instant;

const NOISE_SD: f64 = 0.30;
const SEED_BASE: u64 = 0xB10B_0001_0001_0001;
const HYBRID_LENGTH_SCALE: f64 = 1.0;

/// `large_scale_reml_stress.rs::truth`, verbatim.
fn truth(row: &[f64]) -> f64 {
    let mut linear = 0.0;
    let coefs = [0.55, -0.40, 0.30, 0.20, -0.15, 0.10];
    for (j, &xj) in row.iter().enumerate() {
        if j < coefs.len() {
            linear += coefs[j] * xj;
        }
    }
    let mut dist2 = 0.0;
    for (j, &xj) in row.iter().enumerate() {
        let cj = match j {
            0 => 0.30,
            1 => -0.20,
            2 => 0.10,
            _ => 0.0,
        };
        let d = xj - cj;
        dist2 += d * d;
    }
    let radial_bump = 1.0 * (-dist2 / (2.0 * 0.8 * 0.8)).exp();
    let sinusoid = 0.4 * (std::f64::consts::PI * row[0]).sin();
    linear + radial_bump + sinusoid
}

/// `large_scale_reml_stress.rs::simulate`, verbatim.
fn simulate(n: usize, pc_dim: usize, seed: u64) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("normal params must be valid");
    let noise = Normal::new(0.0, NOISE_SD).expect("noise params must be valid");
    let mut x = Array2::<f64>::zeros((n, pc_dim));
    let mut y = Array1::<f64>::zeros(n);
    let mut y_true = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut row = vec![0.0_f64; pc_dim];
        for j in 0..pc_dim {
            let v = normal.sample(&mut rng);
            x[[i, j]] = v;
            row[j] = v;
        }
        let f = truth(&row);
        y_true[i] = f;
        y[i] = f + noise.sample(&mut rng);
    }
    (x, y, y_true)
}

fn relative_l2(pred: &Array1<f64>, truth: &Array1<f64>) -> f64 {
    let mean_t = truth.mean().unwrap_or(0.0);
    let mut num = 0.0;
    let mut den = 0.0;
    for (p, t) in pred.iter().zip(truth.iter()) {
        num += (p - t) * (p - t);
        den += (t - mean_t) * (t - mean_t);
    }
    (num / den.max(1e-30)).sqrt()
}

fn duchon_spec(pc_dim: usize, k_centers: usize, eta: &[f64]) -> TermCollectionSpec {
    let operator_penalties = DuchonOperatorPenaltySpec::default();
    let (nullspace_order, power) = resolve_duchon_orders(
        pc_dim,
        DuchonNullspaceOrder::Linear,
        duchon_max_active_operator_derivative_order(&operator_penalties),
        Some(HYBRID_LENGTH_SCALE),
    );
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "duchon_pc_probe".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: (0..pc_dim).collect(),
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    center_strategy: CenterStrategy::FarthestPoint {
                        num_centers: k_centers,
                    },
                    length_scale: Some(HYBRID_LENGTH_SCALE),
                    power: power as f64,
                    nullspace_order,
                    identifiability: gam::basis::SpatialIdentifiability::default(),
                    aniso_log_scales: Some(eta.to_vec()),
                    operator_penalties,
                    periodic: None,
                    boundary: gam::basis::OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

fn fit_options(max_iter: usize) -> FitOptions {
    FitOptions {
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: true,
        max_iter,
        tol: 1e-5,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    }
}

fn one_fit(
    x_train: ArrayView2<'_, f64>,
    y_train: &Array1<f64>,
    x_holdout: ArrayView2<'_, f64>,
    y_true_holdout: &Array1<f64>,
    k_centers: usize,
    eta: &[f64],
    label: &str,
) {
    let pc_dim = x_train.ncols();
    let spec = duchon_spec(pc_dim, k_centers, eta);
    let weights = Array1::ones(x_train.nrows());
    let offset = Array1::<f64>::zeros(x_train.nrows());
    let start = Instant::now();
    let fitted = match fit_term_collection_forspec(
        x_train,
        y_train.view(),
        weights.view(),
        offset.view(),
        &spec,
        LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        &fit_options(40),
    ) {
        Ok(f) => f,
        Err(err) => {
            println!("[{label}] FIT FAILED: {err}");
            return;
        }
    };
    let elapsed = start.elapsed().as_secs_f64();
    let frozen =
        freeze_term_collection_from_design(&spec, &fitted.design).expect("freeze trained spec");
    let holdout_design =
        build_term_collection_design(x_holdout, &frozen).expect("holdout design build");
    let dense = holdout_design.design.to_dense();
    let pred = dense.dot(&fitted.fit.beta);
    let rel = relative_l2(&pred, y_true_holdout);
    let frozen_eta = get_spatial_aniso_log_scales(&frozen, 0).unwrap_or_default();
    let parts: Vec<String> = frozen_eta.iter().map(|v| format!("{v:+.4}")).collect();
    let spread = frozen_eta
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        - frozen_eta.iter().cloned().fold(f64::INFINITY, f64::min);
    let requested: Vec<String> = eta.iter().map(|v| format!("{v:+.4}")).collect();
    println!(
        "[{label}] requested_eta=[{}] frozen_eta=[{}] spread={spread:.4} \
         reml={:?} outer_iter={} rel_l2={rel:.4} wall={elapsed:.1}s",
        requested.join(","),
        parts.join(","),
        fitted.fit.reml_score(),
        fitted.fit.outer_iterations,
    );
}

/// One fit through the PRODUCTION entry — the one `StandardFitRequest` uses —
/// so the ψ outer solve runs and the geometry is learned rather than pinned.
fn production_fit(
    x_train: ArrayView2<'_, f64>,
    y_train: &Array1<f64>,
    x_holdout: ArrayView2<'_, f64>,
    y_true_holdout: &Array1<f64>,
    k_centers: usize,
    label: &str,
) {
    let pc_dim = x_train.ncols();
    let spec = duchon_spec(pc_dim, k_centers, &vec![0.0; pc_dim]);
    let weights = Array1::ones(x_train.nrows());
    let offset = Array1::<f64>::zeros(x_train.nrows());
    let start = Instant::now();
    let fitted = match fit_term_collectionwith_spatial_length_scale_optimization(
        x_train,
        y_train.clone(),
        weights,
        offset,
        &spec,
        LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        &fit_options(40),
        &SpatialLengthScaleOptimizationOptions {
            pilot_subsample_threshold: 0,
            ..SpatialLengthScaleOptimizationOptions::default()
        },
    ) {
        Ok(f) => f,
        Err(err) => {
            println!("[{label}] FIT FAILED: {err}");
            return;
        }
    };
    let elapsed = start.elapsed().as_secs_f64();
    let frozen = fitted.resolvedspec.clone();
    let holdout_design =
        build_term_collection_design(x_holdout, &frozen).expect("holdout design build");
    let dense = holdout_design.design.to_dense();
    let pred = dense.dot(&fitted.fit.beta);
    let rel = relative_l2(&pred, y_true_holdout);
    let frozen_eta = get_spatial_aniso_log_scales(&frozen, 0).unwrap_or_default();
    let parts: Vec<String> = frozen_eta.iter().map(|v| format!("{v:+.4}")).collect();
    let spread = frozen_eta
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        - frozen_eta.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "[{label}] learned_eta=[{}] spread={spread:.4} learned_length_scale={:?} \
         reml={:?} outer_iter={} rel_l2={rel:.4} wall={elapsed:.1}s",
        parts.join(","),
        get_spatial_length_scale(&frozen, 0),
        fitted.fit.reml_score(),
        fitted.fit.outer_iterations,
    );
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

fn main() {
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(6000);
    let k: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(150);
    let d: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(6);
    let sweep: Vec<f64> = args
        .get(4)
        .map(|v| {
            v.split(',')
                .filter_map(|piece| piece.parse::<f64>().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.5]);

    let (x_train, y_train, _) = simulate(n, d, SEED_BASE);
    let (x_holdout, _, y_true_holdout) = simulate(4000, d, SEED_BASE.wrapping_add(0xDEAD));

    println!("# n={n} k={k} d={d}");
    // The fixture's own request: the all-zero sentinel, replaced on every build
    // by knot-cloud geometry contrasts.
    one_fit(
        x_train.view(),
        &y_train,
        x_holdout.view(),
        &y_true_holdout,
        k,
        &vec![0.0; d],
        "sentinel",
    );
    // The production entry, which runs the ψ outer solve: with #2735's
    // enrollment the per-axis η is one of its coordinates.
    production_fit(
        x_train.view(),
        &y_train,
        x_holdout.view(),
        &y_true_holdout,
        k,
        "production",
    );
    // Axis-0 contrast sweep: c > 0 gives axis 0 a SHORTER correlation range
    // (larger metric weight) and every other axis an equal share of the
    // opposite shift, so Σ η = 0 by construction.
    for &c in &sweep {
        let mut eta = vec![-c / (d as f64 - 1.0); d];
        eta[0] = c;
        let label = format!("axis0_c={c:.2}");
        one_fit(
            x_train.view(),
            &y_train,
            x_holdout.view(),
            &y_true_holdout,
            k,
            &eta,
            &label,
        );
    }
}
