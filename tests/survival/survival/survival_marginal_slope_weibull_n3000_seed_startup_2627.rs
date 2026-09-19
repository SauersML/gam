//! #2627: the survival marginal-slope fit must actually START its outer
//! ρ-search on a well-conditioned Weibull frame.
//!
//! The Python suite covers this shape (`tests/test_python_api.py`,
//! `test_survival_marginal_slope_weibull_n3000_returns_under_60s`) and it
//! aborts there with the outer startup-validation signature:
//!
//! ```text
//! no candidate seeds passed outer startup validation (custom family):
//!   generated=5, screened=5, exact_validated=3, solver_started=0
//!   rejection breakdown: rejected_by_kkt=0, rejected_by_domain=0,
//!   rejected_by_objective=3, rejected_by_budget=0, rejected_other=0 (total=3)
//!   seed 0 (validation): custom-family inner solve did not converge after
//!   8 cycle(s) [joint-Newton terminal cycle 11:
//!   stationarity_residual=1.404164e3 (tol=1.359314e-8),
//!   step_inf=5.298744e-1, resolvable_negative_curvature=true,
//!   best_stationarity_residual=1.404164e3 (last improved 3 cycle(s) before)]
//! ```
//!
//! `solver_started = 0` means EVERY candidate ρ seed was rejected, so the
//! outer smoothing-parameter search never began — a fit in that state has
//! measured none of its own subject. The stationarity residual is 1.4e3
//! against a tolerance of 1.4e-8 and stops improving three cycles before the
//! budget runs out, with confirmed negative curvature in the feasible face:
//! that is a starting point in the wrong basin, not a tolerance, budget or
//! line-search problem.
//!
//! There was no Rust-side gate for this, which is why the defect could only
//! be seen in a Python artifact. This test reproduces the same fit shape at
//! the same n from Rust: same formula, same likelihood mode, same z column,
//! same slope formula, and a Weibull data-generating process with the
//! same structure as the Python fixture's `_make_weibull_survival`.
//!
//! Related: `survival_marginal_slope_stall` reproduces a DIFFERENT failure on
//! the same family (an n=195,780 PC-Duchon design whose joint-Newton trust
//! region is the subject). This one is the small, well-conditioned frame — if
//! this fit cannot start, nothing on this family can.

use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
use csv::StringRecord;
use std::sync::Once;

const N: usize = 3000;

struct StderrInfoLogger;

impl log::Log for StderrInfoLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Debug
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
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);
    init_parallelism();
    INIT_LOGGER.call_once(|| {
        if log::set_logger(&LOGGER).is_ok() {
            log::set_max_level(log::LevelFilter::Debug);
        }
    });
}

use gam::utils::splitmix64;

#[inline]
fn next_unit(state: &mut u64) -> f64 {
    let bits = splitmix64(state) >> 11;
    (bits as f64) * (1.0_f64 / ((1u64 << 53) as f64))
}

#[inline]
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    r * theta.cos()
}

/// Standard Weibull(shape) variate: `(-ln U)^(1/shape)`. Matches numpy's
/// `Generator.weibull(a)`, which is what the Python fixture draws.
#[inline]
fn next_weibull(state: &mut u64, shape: f64) -> f64 {
    let u = next_unit(state).max(f64::MIN_POSITIVE);
    (-u.ln()).powf(1.0 / shape)
}

/// Same structure as the Python fixture's `_make_weibull_survival(n=3000)`:
/// two continuous covariates driving a Weibull scale, one standard-normal
/// latent score column, staggered entry, and independent uniform censoring.
/// Roughly half the rows are events, so neither the event nor the censoring
/// channel is degenerate.
fn build_dataset() -> gam::inference::data::EncodedDataset {
    let headers = vec![
        "entry".to_string(),
        "exit".to_string(),
        "event".to_string(),
        "bmi".to_string(),
        "hba1c".to_string(),
        "age".to_string(),
    ];

    let mut state = 0x2627_C0FFEE_u64;
    let mut bmi = Vec::with_capacity(N);
    let mut hba1c = Vec::with_capacity(N);
    let mut age = Vec::with_capacity(N);
    let mut entry = Vec::with_capacity(N);
    for _ in 0..N {
        bmi.push(27.0 + 4.5 * next_gauss(&mut state));
        hba1c.push(5.8 + 0.7 * next_gauss(&mut state));
        age.push(next_gauss(&mut state));
        entry.push(40.0 + 25.0 * next_unit(&mut state));
    }

    let standardize = |v: &[f64]| -> Vec<f64> {
        let mean = v.iter().copied().sum::<f64>() / v.len() as f64;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / v.len() as f64;
        let sd = var.sqrt();
        v.iter().map(|x| (x - mean) / sd).collect()
    };
    let bmi_z = standardize(&bmi);
    let hba1c_z = standardize(&hba1c);

    const SHAPE: f64 = 1.45;
    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);
    for i in 0..N {
        let log_scale = 18.0_f64.ln() - 0.18 * bmi_z[i] - 0.22 * hba1c_z[i] - 0.10 * age[i];
        let event_gap = log_scale.exp() * next_weibull(&mut state, SHAPE);
        let censor_gap = 6.0 + 26.0 * next_unit(&mut state);
        let observed = event_gap.min(censor_gap);
        let event = if event_gap <= censor_gap { "1" } else { "0" };
        rows.push(StringRecord::from(vec![
            entry[i].to_string(),
            (entry[i] + observed).to_string(),
            event.to_string(),
            bmi[i].to_string(),
            hba1c[i].to_string(),
            age[i].to_string(),
        ]));
    }

    encode_recordswith_inferred_schema(headers, rows)
        .expect("encode synthetic weibull survival marginal-slope dataset")
}

#[test]
fn survival_marginal_slope_weibull_n3000_starts_its_outer_search() {
    init();

    let data = build_dataset();
    let slope_formula = "smooth(bmi) + smooth(hba1c)".to_string();
    let formula = format!("Surv(entry, exit, event) ~ {slope_formula}");

    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("age".to_string()),
        slope_formula: Some(slope_formula),
        gpu_policy: if cfg!(target_os = "macos") {
            gam::gpu::GpuPolicy::Off
        } else {
            gam::gpu::GpuPolicy::Auto
        },
        ..FitConfig::default()
    };

    match fit_from_formula(&formula, &data, &config) {
        Ok(_) => {}
        Err(err) => {
            let message = err.to_string();
            // Name the startup-validation abort explicitly. Any other error is
            // a different defect and must not be silently folded into this one.
            assert!(
                !message.contains("no candidate seeds passed outer startup validation"),
                "the outer rho-search never started: every candidate seed was \
                 rejected, so this fit measured none of its own subject. \
                 Read the `rejection breakdown` and per-seed reasons below — the \
                 category determines whether convergence work can address it at \
                 all.\n{message}"
            );
            panic!("survival marginal-slope weibull n={N} fit failed: {message}");
        }
    }
}
