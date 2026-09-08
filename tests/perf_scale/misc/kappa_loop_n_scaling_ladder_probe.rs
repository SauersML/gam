//! gam#2760 decomposition probe: read `|Pg|` and its bound at EVERY rung of one
//! `n`-ladder, from ONE commit, on the same fixture the failing arms use.
//!
//! The issue transcribes two numbers taken from two different builds
//! (`n = 1000` pre-`24f3d24cd`, `n = 4000` post-), so their ratio is not a
//! scaling law. The question it poses — does `|Pg|` grow with `n`, does the
//! bound tighten with `n`, or both — needs both ends measured at one commit.
//!
//! `run_fit` in the sibling module reports only the refusal STRING, and only on
//! the rungs that fail; a converged rung carries its certificate silently in
//! `artifacts.criterion_certificate`. This probe reads that certificate on
//! success and the refusal text on failure, so every rung contributes a
//! `(|Pg|, bound, rung, railed)` row whether or not it converged.
//!
//! Report-only by construction: the deliverable is the printed table.

use gam::{
    FitRequest, FitResult, StandardFitRequest,
    basis::{
        CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
        OneDimensionalBoundary, SpatialIdentifiability,
    },
    estimate::FitOptions,
    smooth::{
        ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
        TermCollectionSpec,
    },
    types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink},
};
use ndarray::{Array1, Array2};

/// The sibling module's fixture, verbatim: a gentle noiseless `sin` on a
/// uniform grid. Duplicated rather than imported because the point of this
/// probe is that it measures the SAME fixture the failing arms measure, and a
/// shared helper that drifted would silently break that.
fn simulate_1d_gaussian(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::<f64>::zeros((n, 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = (i as f64) / (n as f64 - 1.0) * 6.0 - 3.0;
        x[[i, 0]] = t;
        y[i] = t.sin();
    }
    (x, y)
}

fn spec_1d() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "duchon_1d".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    periodic: None,
                    length_scale: Some(1.0),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::all_active(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

fn fit_options() -> FitOptions {
    FitOptions {
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: false,
        max_iter: 30,
        tol: 1e-6,
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

/// Everything the solver says about the outer trajectory, to a file.
///
/// The sibling module's logger keeps four `[KAPPA-*]` prefixes; the question
/// here is why the SEARCH stopped, so this one keeps the certificate and
/// stationarity lanes too. `nextest` gives each test binary its own process and
/// `set_logger` is idempotent-by-swallow, so installing both is safe: whichever
/// runs first owns the process, and this probe reports the file it wrote.
struct LadderTrace {
    file: std::sync::Mutex<std::fs::File>,
}

impl log::Log for LadderTrace {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Debug
    }
    fn log(&self, record: &log::Record) {
        use std::io::Write;
        let msg = format!("{}", record.args());
        let keep = [
            "[KAPPA-",
            "[CERTIFICATE",
            "[NFREE-RESET",
            "[spatial-kappa",
            "[OUTER",
            "[RAIL",
            "[COST-STALL",
            "[PSI-GRAM-INSTALL",
        ]
        .iter()
        .any(|prefix| msg.starts_with(prefix));
        if !keep {
            return;
        }
        if let Ok(mut f) = self.file.lock()
            && let Err(error) = writeln!(f, "{msg}").and_then(|()| f.flush())
        {
            eprintln!("[2760-ladder] dropped a trace record: {error}");
        }
    }
    fn flush(&self) {}
}

static LADDER_TRACE: std::sync::OnceLock<LadderTrace> = std::sync::OnceLock::new();

fn install_trace() {
    let logger = LADDER_TRACE.get_or_init(|| LadderTrace {
        file: std::sync::Mutex::new(
            std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open("/tmp/gam2760_ladder.log")
                .expect("open ladder trace log"),
        ),
    });
    drop(log::set_logger(logger));
    log::set_max_level(log::LevelFilter::Debug);
}

/// One rung: the certificate the fit minted, or the refusal it died on.
fn rung(n: usize) -> Result<String, String> {
    let (x, y) = simulate_1d_gaussian(n);
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let kappa_options = SpatialLengthScaleOptimizationOptions {
        enabled: true,
        max_outer_iter: 15,
        rel_tol: 1e-5,
        log_step: std::f64::consts::LN_2,
        min_length_scale: 1e-2,
        max_length_scale: 1e2,
        pilot_subsample_threshold: 0,
    };
    let result = gam::fit_model(FitRequest::Standard(StandardFitRequest {
        data: gam::solver::fit_orchestration::StandardFitData::shared(x),
        y: std::sync::Arc::new(y),
        weights: std::sync::Arc::new(weights),
        offset: std::sync::Arc::new(offset),
        spec: spec_1d(),
        family: LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        options: fit_options(),
        kappa_options,
        wiggle: None,
        coefficient_groups: Vec::new(),
        penalty_block_gamma_priors: Vec::new(),
        latent_coord: None,
        estimate_tweedie_p: false,
    }))
    .map_err(|e| format!("{e:?}"))?;
    match result {
        FitResult::Standard(s) => {
            let cert = s
                .fit
                .convergence_evidence()
                .outer_certificate()
                .map(|c| c.summary())
                .unwrap_or_else(|| "<no criterion certificate>".to_string());
            Ok(format!(
                "V={:?} rho={:?} outer_iters={} | {cert}",
                s.fit.reml_score(),
                s.fit.log_lambdas.to_vec(),
                s.fit.outer_iterations,
            ))
        }
        _ => Err("expected Standard fit result".to_string()),
    }
}

#[test]
fn probe_2760_pg_and_bound_at_every_rung() {
    install_trace();
    for &n in &[1_000usize, 2_000, 4_000, 8_000, 16_000] {
        eprintln!("[2760-ladder] ==== rung n={n} ====");
        log::info!("[KAPPA-RUNG] ================ n={n} ================");
        let t0 = std::time::Instant::now();
        match rung(n) {
            Ok(line) => eprintln!(
                "[2760-ladder] n={n:>6} OK   ({:.1}s) {line}",
                t0.elapsed().as_secs_f64()
            ),
            Err(reason) => eprintln!(
                "[2760-ladder] n={n:>6} FAIL ({:.1}s) {reason}",
                t0.elapsed().as_secs_f64()
            ),
        }
    }
}
