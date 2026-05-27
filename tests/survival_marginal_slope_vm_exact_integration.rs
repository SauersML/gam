//! Integration test for the V+M-exact construction-site cutover (team
//! `smgs-vm-exact`, sub-task J).
//!
//! Aim: prove end-to-end that on a synthetic biobank-shape survival
//! marginal-slope problem with intentional cross-block parametric
//! aliasing, the V+M cutover engages, the fit converges, and predict
//! still runs on raw-width β.
//!
//! Aliasing is induced by the formula choosing a `duchon` log-slope
//! surface that includes the constant (parametric intercept) column,
//! while the time block carries its own structural baseline. The shared
//! constant forces the V+M compile to drop at least one column, which
//! triggers the `[smgs phase-4b active] applying per-term V: ...` log
//! line in `src/families/survival_marginal_slope.rs`.
//!
//! Hard contract (see teammate brief):
//!   * fit converges (`outer_converged == true`),
//!   * β block widths after lift equal RAW widths (not compiled widths),
//!     proving `SmgsLiftViaT::lift_block_betas_via_t` ran,
//!   * predictions are finite,
//!   * the active V log line is emitted at least once.
//!
//! NOTE: this test is written *ahead* of the sibling V+M cutover
//! landing. Until those land it will fail (or refuse to compile against
//! a sibling-renamed API). Do not run with `cargo test`; sibling work
//! will run it once the cutover is in.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use std::sync::{Arc, Mutex, Once, OnceLock};

const N: usize = 300;

#[derive(Default)]
struct CapturedLogs {
    lines: Mutex<Vec<String>>,
}

impl CapturedLogs {
    fn push(&self, line: String) {
        if let Ok(mut guard) = self.lines.lock() {
            guard.push(line);
        }
    }

    fn snapshot(&self) -> Vec<String> {
        self.lines
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|_| Vec::new())
    }
}

struct CapturingLogger {
    sink: Arc<CapturedLogs>,
}

impl log::Log for CapturingLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let line = format!("{}", record.args());
        eprintln!("{line}");
        self.sink.push(line);
    }
    fn flush(&self) {}
}

static INIT_LOGGER: Once = Once::new();
static LOG_SINK: OnceLock<Arc<CapturedLogs>> = OnceLock::new();

fn log_sink() -> &'static Arc<CapturedLogs> {
    LOG_SINK.get_or_init(|| Arc::new(CapturedLogs::default()))
}

fn install_logger() {
    INIT_LOGGER.call_once(|| {
        let logger = Box::leak(Box::new(CapturingLogger {
            sink: log_sink().clone(),
        }));
        if log::set_logger(logger).is_ok() {
            log::set_max_level(log::LevelFilter::Info);
        }
    });
}

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
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    r * theta.cos()
}

fn build_dataset() -> gam::inference::data::EncodedDataset {
    let headers = vec![
        "entry_age".to_string(),
        "exit_age".to_string(),
        "event".to_string(),
        "prs_z".to_string(),
        "x1".to_string(),
        "x2".to_string(),
        "x3".to_string(),
    ];

    let mut state = 0xC0DE_C0DE_u64;
    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);

    for _ in 0..N {
        let x1 = next_gauss(&mut state) * 0.5;
        let x2 = next_gauss(&mut state) * 0.5;
        let x3 = next_gauss(&mut state) * 0.5;
        let prs = next_gauss(&mut state);
        let entry = 40.0 + 5.0 * next_unit(&mut state);
        let followup = 0.5 + 8.0 * next_unit(&mut state);
        let exit = entry + followup;
        // mild signal so events are non-degenerate
        let score = 0.3 * prs + 0.4 * x1 - 0.3 * x2 + 0.2 * x3 + 0.2 * next_gauss(&mut state);
        let event = if score > 0.0 { 1 } else { 0 };
        rows.push(StringRecord::from(vec![
            entry.to_string(),
            exit.to_string(),
            event.to_string(),
            prs.to_string(),
            x1.to_string(),
            x2.to_string(),
            x3.to_string(),
        ]));
    }

    encode_recordswith_inferred_schema(headers, rows)
        .expect("encode synthetic V+M integration dataset")
}

#[test]
fn survival_marginal_slope_v_plus_m_exact_engages_and_lifts_beta_to_raw_width() {
    install_logger();
    init_parallelism();
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    let data = build_dataset();

    // Marginal side carries a `duchon` smooth over three covariates plus
    // a parametric intercept. The log-slope side carries the SAME duchon
    // surface — the shared constant column (intercept) between marginal
    // and log-slope forces the V+M parametric compile to drop at least
    // one column, which engages phase-4b.
    let duchon = "duchon(x1, x2, x3, centers=6, order=1)".to_string();
    let formula = format!(
        "Surv(entry_age, exit_age, event) ~ {} + 1",
        duchon
    );

    let config = FitConfig {
        survival_likelihood: "marginal-slope".to_string(),
        z_column: Some("prs_z".to_string()),
        logslope_formula: Some(duchon.clone()),
        baseline_target: "linear".to_string(),
        gpu_policy: if cfg!(target_os = "macos") {
            gam::gpu::GpuPolicy::Off
        } else {
            gam::gpu::GpuPolicy::Auto
        },
        ..FitConfig::default()
    };

    let outcome = fit_from_formula(&formula, &data, &config)
        .expect("V+M-exact synthetic survival marginal-slope fit should succeed");

    let result = match outcome {
        FitResult::SurvivalMarginalSlope(r) => r,
        other => panic!(
            "expected FitResult::SurvivalMarginalSlope, got {:?}",
            std::mem::discriminant(&other)
        ),
    };

    assert!(
        result.fit.outer_converged,
        "V+M-exact cutover must converge; outer_converged=false (iters={}, reml={:.6})",
        result.fit.outer_iterations, result.fit.reml_score
    );

    let logs = log_sink().snapshot();
    let active_marker = "[smgs phase-4b active] applying per-term V:";
    let active_seen = logs.iter().any(|line| line.contains(active_marker));
    assert!(
        active_seen,
        "expected log line containing {:?}; this proves the V+M cutover engaged. \
         Captured {} log lines.",
        active_marker,
        logs.len(),
    );

    // After the lift, β block widths must match the RAW design widths
    // (i.e. `marginal_design.ncols()` / `logslope_design.ncols()`), not
    // the compiled (post-drop) widths. If lift were skipped, block β
    // would still be at compiled width.
    let raw_marginal_width = result.marginal_design.ncols();
    let raw_logslope_width = result.logslope_design.ncols();

    // Find the marginal block and the logslope block in the fitted
    // result. Survival marginal-slope concatenates blocks in order
    // [time, marginal, logslope, (deviation blocks...)]. The exact
    // role label depends on `BlockRole` definitions but at minimum
    // the block widths in order must contain the raw marginal width
    // and the raw logslope width as distinct entries — we assert
    // structural width preservation rather than role naming so the
    // test is robust to the role-enum churn around the cutover.
    let block_widths: Vec<usize> = result.fit.blocks.iter().map(|b| b.beta.len()).collect();

    assert!(
        block_widths.contains(&raw_marginal_width),
        "no fitted block matches raw marginal design width {}: widths={:?}",
        raw_marginal_width,
        block_widths
    );
    assert!(
        block_widths.contains(&raw_logslope_width),
        "no fitted block matches raw log-slope design width {}: widths={:?}",
        raw_logslope_width,
        block_widths
    );

    // Sanity: every joint β coefficient is finite, no NaN/Inf leaked
    // through the lift.
    for (idx, block) in result.fit.blocks.iter().enumerate() {
        for (j, &coef) in block.beta.iter().enumerate() {
            assert!(
                coef.is_finite(),
                "β block {idx} coef {j} non-finite: {coef}"
            );
        }
    }

    // Predictions on the training data: the joint design × β + offsets
    // must produce a finite linear predictor in every row. We use the
    // fitted marginal design and joint β for the marginal portion as a
    // lightweight finite-ness probe; the full survival predict path is
    // exercised by `predict_*` tests elsewhere.
    let marginal_beta_block = result
        .fit
        .blocks
        .iter()
        .find(|b| b.beta.len() == raw_marginal_width)
        .expect("locate marginal β block");
    let pred_marginal = result.marginal_design.apply(&marginal_beta_block.beta);
    for (i, &v) in pred_marginal.iter().enumerate() {
        assert!(v.is_finite(), "marginal η[{i}] non-finite: {v}");
    }
}
