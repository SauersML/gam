//! Red test: reproduce the PIRLS joint-Newton residual-stall early-exit
//! observed in production survival_marginal_slope fits at biobank scale.
//!
//! Production signature (n=195,780, p=33, 5/5 outer seeds failed):
//!
//!   - cycle 0: |prop|∞ ≈ 2e5, TR clamps to |δ|∞=20, |β|∞=20.0
//!   - cycles 1..25: linearized_rel ratio ≈ 0.97 for 15+ cycles
//!   - residual-stall early-exit triggers
//!   - budget-exhausted dump shows ALL gradient in the time block:
//!         block_widths   = [12, 11, 10]
//!         block_beta_inf = [2.3e-4, 15.3, 20.0]
//!         block_grad_inf = [5.6e8,  1.5e3, 2.3e3]
//!
//! Hypothesis under test: the joint-Newton trust region uses an isotropic
//! L2-norm step constraint over the concatenated δ. When one block (the
//! time block) has near-singular curvature, the unconstrained Newton step
//! has huge norm in that direction; the global L2 clamp rescales the
//! ENTIRE δ uniformly, so the marginal/logslope blocks receive an
//! arbitrarily small fraction of their "fair" step and the time gradient
//! stays large forever.
//!
//! This test induces the same regime on small synthetic data by:
//!   * over-parameterising the time block (many internal knots over a
//!     narrow age window),
//!   * making the upper part of the time axis nearly event-free so
//!     several time-basis columns carry almost no Fisher information,
//!   * disabling the time smoothing prior (very small lambda) so that
//!     prior is not what saves us.
//!
//! Current expectation: this test FAILS — the outer optimizer rejects
//! every seed because the inner joint-Newton stalls (residual-stall
//! early-exit). When the bug is fixed (per-block / preconditioned
//! anisotropic TR), the assertion `outer_converged == true` will pass.

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
use std::sync::Once;
use std::time::Instant;

const SEED: u64 = 0xC0FF_EE15_F00D_BA75;
const N: usize = 195_780;
const N_PCS: usize = 3;

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
    init_parallelism();
    INIT_LOGGER.call_once(|| {
        if log::set_logger(&LOGGER).is_ok() {
            log::set_max_level(log::LevelFilter::Info);
        }
    });
}

// ── Inline RNG (splitmix64 + Box-Muller, no external state) ───────────────

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn next_unit(state: &mut u64) -> f64 {
    let bits = splitmix64(state) >> 11;
    (bits as f64) * (1.0_f64 / ((1u64 << 53) as f64))
}

#[inline]
fn next_gauss(state: &mut u64) -> f64 {
    // Box-Muller with safe lower bound on u1.
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    r * theta.cos()
}

/// Build a small synthetic biobank-shape survival dataset whose time block
/// is structurally ill-conditioned: most events occur in the lower part
/// of the age window, leaving the upper-time basis columns with very
/// little Fisher information.  This mirrors a real survey/cohort where
/// the upper age strata are thinly sampled relative to the spline density.
fn build_dataset() -> gam::inference::data::EncodedDataset {
    let mut headers = vec![
        "entry_age".to_string(),
        "exit_age".to_string(),
        "event".to_string(),
        "sex".to_string(),
        "prs_z".to_string(),
    ];
    for i in 0..N_PCS {
        headers.push(format!("PC{}", i + 1));
    }

    let mut state = SEED;
    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);

    for _ in 0..N {
        let sex = if next_unit(&mut state) < 0.5 { 1.0_f64 } else { 0.0_f64 };
        // Age window [40, 80] but heavily back-loaded entries
        let entry_age: f64 = 40.0 + 30.0 * next_unit(&mut state);
        // Exit-age increment is short and skewed small so events cluster
        // in the lower-mid age band — the time basis sees a flat tail.
        let dt = 0.5 + 5.0 * next_unit(&mut state).powf(2.0);
        let exit_age = (entry_age + dt).min(80.0);

        // Latent prs_z and 10 PCs, all standard normal.
        let prs_z = next_gauss(&mut state);
        let mut pcs = [0.0_f64; N_PCS];
        for j in 0..N_PCS {
            pcs[j] = next_gauss(&mut state);
        }
        // Build a hazard that depends weakly on sex/prs/pcs.  Keep the
        // marginal event-rate low (≈ 4–6 %) so the time-block Fisher
        // matrix is structurally rank-deficient at the high-knot density
        // configured below.  This is the regime that triggers the
        // production failure: huge unconstrained Newton step in the
        // time block dominates the joint L2 trust-region clamp.
        let pc_signal: f64 = pcs.iter().enumerate()
            .map(|(j, p)| p * ((j + 1) as f64).recip() * if j % 2 == 0 { 1.0 } else { -1.0 })
            .sum();
        let log_hazard = -3.2 + 0.15 * sex + 0.10 * prs_z + 0.03 * pc_signal
            + 0.04 * (entry_age - 60.0);
        let lambda = log_hazard.exp().min(5e-2);
        // Sample event indicator from per-unit-time hazard over dt.
        let p_event = 1.0 - (-lambda * dt).exp();
        let event = if next_unit(&mut state) < p_event { 1.0 } else { 0.0 };

        let mut record: Vec<String> = vec![
            entry_age.to_string(),
            exit_age.to_string(),
            event.to_string(),
            sex.to_string(),
            prs_z.to_string(),
        ];
        for j in 0..N_PCS {
            record.push(pcs[j].to_string());
        }
        rows.push(StringRecord::from(record));
    }

    encode_recordswith_inferred_schema(headers, rows)
        .expect("encode synthetic survival marginal-slope dataset")
}

/// Reproduce the PIRLS joint-Newton residual-stall early-exit observed
/// at biobank scale.  This test is currently expected to FAIL — assert
/// `outer_converged == true`; the production code returns false because
/// the inner joint-Newton stalls on the time block.
#[test]
fn survival_marginal_slope_stall_reproduces_residual_stall_early_exit() {
    init();

    let data = build_dataset();

    // Build the PC-Duchon log-slope formula with the same PC dimensionality
    // and center count as the failing biobank fit:
    // `duchon(PC1, PC2, PC3, centers=10, order=1)` on both sides.
    let pcs: Vec<String> = (0..N_PCS).map(|i| format!("PC{}", i + 1)).collect();
    let duchon_term = format!("duchon({}, centers=10, order=1)", pcs.join(", "));
    // Time block configured to be over-parameterised relative to the
    // event-bearing portion of the age window: 12 internal knots over
    // a 40-unit range with a near-flat tail produces a near-singular
    // exit-time design.  `time_smooth_lambda = 1e-8` disables the
    // structural penalty's stabilising contribution so the bug is not
    // hidden by the prior.
    let formula = format!("Surv(entry_age, exit_age, event) ~ {} + sex", duchon_term);

    let config = FitConfig {
        survival_likelihood: "marginal-slope".to_string(),
        z_column: Some("prs_z".to_string()),
        logslope_formula: Some(duchon_term),
        baseline_target: "linear".to_string(),
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: 12,
        time_smooth_lambda: 1e-8,
        ..FitConfig::default()
    };

    eprintln!(
        "[SURVIVAL-MGS-STALL] starting fit: n={} time_internal_knots={} time_smooth_lambda={:.1e} formula={:?}",
        N, config.time_num_internal_knots, config.time_smooth_lambda, formula
    );

    let start = Instant::now();
    let outcome = fit_from_formula(&formula, &data, &config);
    let elapsed = start.elapsed();
    eprintln!(
        "[SURVIVAL-MGS-STALL] fit_from_formula returned in {:.3}s ok={}",
        elapsed.as_secs_f64(),
        outcome.is_ok()
    );

    let result = outcome
        .expect("survival marginal-slope fit returned an integration error (not the stall we're hunting)");

    let FitResult::SurvivalMarginalSlope(fit) = result else {
        panic!("expected SurvivalMarginalSlope FitResult variant");
    };

    let unified = &fit.fit;
    eprintln!(
        "[SURVIVAL-MGS-STALL] outer_converged={} outer_iters={} inner_cycles={} pirls_status={:?}",
        unified.outer_converged,
        unified.outer_iterations,
        unified.inner_cycles,
        unified.pirls_status
    );

    // RED ASSERTION: we expect the inner PIRLS joint-Newton stall to
    // make `outer_converged == false`.  When the isotropic-TR bug is
    // fixed, this assertion will pass.
    assert!(
        unified.outer_converged,
        "expected the survival marginal-slope fit to converge; got outer_converged=false \
         (pirls_status={:?}, outer_iters={}, inner_cycles={}) — this reproduces the \
         production joint-Newton residual-stall early-exit.",
        unified.pirls_status, unified.outer_iterations, unified.inner_cycles
    );
}
