//! gam#2994 / gam#2995: a survival marginal-slope model solves one objective
//! whichever route the spatial-joint driver takes.
//!
//! The default `linear` baseline target has no auxiliary outer coordinate, so
//! the driver takes its fast path: one fit at the stated smoothing start. The
//! `weibull` target adds its log-scale and log-shape as two auxiliary outer
//! coordinates, so the driver runs the full exact-joint θ search. Both routes
//! fit the unarmed member first and refit with the Jeffreys/Firth prior armed
//! only on typed evidence (#979 ruling (b)); that evidence is published on
//! `FitArtifacts::jeffreys_arming_evidence`.
//!
//! What a fit solved is read from the inner solver's own per-cycle record,
//! `[979-PROBE] … firth_armed=…`, which this binary captures with its own
//! logger. That is what tells an armed solve from an unarmed one: a member
//! built armed without evidence (the literal `jeffreys_armed: true` the
//! exact-joint route used to carry) would publish no evidence and still solve
//! the armed objective, and only its cycles show it.
//!
//! Planted Weibull data, nothing separating: both routes solve only the
//! unarmed objective, the pilot included (gam#2994), and publish no evidence.
//! The arming step itself, typed evidence to one armed refit, is covered by
//! `gam-custom-family`'s `arm_on_evidence` tests.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_linalg::utils::splitmix64;
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use std::sync::atomic::{AtomicUsize, Ordering};

const SLOPE: f64 = 0.85;
const BASELINE_LOG_SCALE: f64 = 1.202;
const BASELINE_LOG_SHAPE: f64 = 0.516;
const COVARIATE_EFFECT: f64 = 0.4;

/// Inner-solver cycles seen, and how many of them evaluated the Jeffreys term.
static PROBE_CYCLES: AtomicUsize = AtomicUsize::new(0);
static ARMED_CYCLES: AtomicUsize = AtomicUsize::new(0);

struct ProbeCounter;

impl log::Log for ProbeCounter {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &log::Record<'_>) {
        let line = record.args().to_string();
        if line.starts_with("[979-PROBE]") {
            PROBE_CYCLES.fetch_add(1, Ordering::Relaxed);
            if line.contains("firth_armed=true") {
                ARMED_CYCLES.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn flush(&self) {}
}

static PROBE_COUNTER: ProbeCounter = ProbeCounter;

fn initialize() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        log::set_logger(&PROBE_COUNTER).expect("this binary installs the only logger");
        log::set_max_level(log::LevelFilter::Debug);
        drop(
            gam_problem::laplace_sampler_contract::set_laplace_marginal_corrector(Box::new(
                gam_inference::hmc_io::HmcIoLaplaceMarginalCorrector,
            )),
        );
        drop(gam_problem::rho_posterior::set_rho_posterior_escalator(
            Box::new(gam_inference::rho_posterior::HmcIoRhoPosteriorEscalator),
        ));
        // Match the public fitting startup's stack allowance for survival jets.
        rayon::ThreadPoolBuilder::new()
            .stack_size(64 << 20)
            .build_global()
            .expect("initialize the worker pool");
    });
    #[cfg(target_os = "macos")]
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::Off);
}

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gaussian(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn normal_cdf(x: f64) -> f64 {
    gam_math::probability::normal_cdf(x)
}

/// Standard-normal quantile by bisection on `Φ`, independent of the crate.
fn normal_quantile(p: f64) -> f64 {
    let (mut low, mut high) = (-12.0_f64, 12.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if normal_cdf(mid) < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

/// Event time from `Φ(−η(T)) = u`, `η = q(t, x)·√(1 + b²) + b·z` with the
/// Weibull chart's `q(t) = −Φ⁻¹(exp(−(t/λ)^k))` shifted by the covariate.
fn planted_event_time(u: f64, z: f64, location_shift: f64) -> f64 {
    let target = -normal_quantile(u);
    let baseline_index = (target - SLOPE * z) / (1.0 + SLOPE * SLOPE).sqrt() - location_shift;
    let cumulative_hazard = -normal_cdf(-baseline_index).ln();
    BASELINE_LOG_SCALE.exp() * cumulative_hazard.powf((-BASELINE_LOG_SHAPE).exp())
}

/// A covariate `x`, a frozen score `z`, uniform censoring, and a group
/// indicator `g` on about one row in six.
fn dataset(n: usize, seed: u64) -> gam_data::EncodedDataset {
    let headers = ["time", "event", "z", "x", "g"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let mut state = seed;
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let z = next_gaussian(&mut state);
        let x = next_gaussian(&mut state);
        let u = next_unit(&mut state).clamp(1e-6, 1.0 - 1e-6);
        let group = u8::from(next_unit(&mut state) < 1.0 / 6.0);
        let event_time = planted_event_time(u, z, COVARIATE_EFFECT * x);
        let censor = 0.35 + 5.0 * next_unit(&mut state);
        let (time, event) = if event_time <= censor {
            (event_time, 1u8)
        } else {
            (censor, 0u8)
        };
        let time = time.clamp(1e-3, 1e3);
        rows.push(StringRecord::from(vec![
            format!("{time:.17e}"),
            event.to_string(),
            format!("{z:.17e}"),
            format!("{x:.17e}"),
            group.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode the gam#2995 fixture")
}

fn config(baseline_target: &str) -> FitConfig {
    FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        slope_formula: Some("1".to_string()),
        z_column: Some("z".to_string()),
        frozen_score: true,
        baseline_target: baseline_target.to_string(),
        time_num_internal_knots: 3,
        ..FitConfig::default()
    }
}

/// What one fit solved: its published arming evidence, and the inner cycles
/// that did and did not evaluate the Jeffreys term.
#[derive(Debug)]
struct Solved {
    evidence: Option<gam_problem::jeffreys_arming::JeffreysArmingEvidence>,
    probe_cycles: usize,
    armed_cycles: usize,
}

fn solve(label: &str, formula: &str, data: &gam_data::EncodedDataset, baseline_target: &str) -> Solved {
    PROBE_CYCLES.store(0, Ordering::Relaxed);
    ARMED_CYCLES.store(0, Ordering::Relaxed);
    let result = fit_from_formula(formula, data, &config(baseline_target))
        .unwrap_or_else(|error| panic!("[2995 {label}] `{formula}` must fit: {error}"));
    let FitResult::SurvivalMarginalSlope(fit) = result else {
        panic!("[2995 {label}] expected a SurvivalMarginalSlope fit result");
    };
    let solved = Solved {
        evidence: fit.fit.artifacts.jeffreys_arming_evidence.clone(),
        probe_cycles: PROBE_CYCLES.load(Ordering::Relaxed),
        armed_cycles: ARMED_CYCLES.load(Ordering::Relaxed),
    };
    eprintln!(
        "[2995 {label}] target={baseline_target} {solved:?} log_lik={:.6e} outer_iterations={}",
        fit.fit.log_likelihood, fit.fit.outer_iterations
    );
    assert!(
        solved.probe_cycles > 0,
        "[2995 {label}] the inner solver's per-cycle record must be captured"
    );
    solved
}

#[test]
fn both_routes_solve_the_unarmed_objective_without_evidence_2995() {
    initialize();
    let data = dataset(800, 0x2930_0000_0001);
    for target in ["linear", "weibull"] {
        let solved = solve("no evidence", "Surv(time, event) ~ x + g", &data, target);
        assert!(
            solved.evidence.is_none(),
            "[{target}] nothing separates, so no evidence arms: {solved:?}"
        );
        assert_eq!(
            solved.armed_cycles, 0,
            "[{target}] without evidence no solve on either route, the pilot included, may \
             evaluate the Jeffreys term: {solved:?}"
        );
    }
}
