//! Integration test for the V+M-exact construction-site cutover (team
//! `smgs-vm-exact`, sub-task J).
//!
//! Aim: prove end-to-end that on a synthetic large-scale survival
//! marginal-slope problem with intentional cross-block parametric
//! aliasing, the V+M cutover engages, the fit converges, and predict
//! still runs on raw-width β.
//!
//! Aliasing is induced by the formula choosing a `duchon` log-slope
//! surface that includes the constant (parametric intercept) column,
//! while the time block carries its own structural baseline. The shared
//! constant forces the V+M compile to drop at least one column, which
//! triggers the `[smgs phase-4b compiled-map] applying CompiledMap T: ...` log
//! line in `src/families/survival_marginal_slope.rs`.
//!
//! Hard contract (see teammate brief):
//!   * fit converges (a minted fit is the sealed convergence proof, SPEC 20),
//!   * β block widths after lift equal RAW widths (not compiled widths),
//!     proving the result-time `Gauge::lift_block_betas` ran,
//!   * predictions are finite,
//!   * the compiled-map log line is emitted at least once (proves the
//!     channel-aware closed-form path ran, not the old per-term fallback).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use std::sync::{Arc, Mutex, Once, OnceLock};

use crate::fixtures::Splitmix64;

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

    let mut rng = Splitmix64::new(0xC0DE_C0DE_u64);
    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);

    for _ in 0..N {
        let x1 = rng.next_gauss() * 0.5;
        let x2 = rng.next_gauss() * 0.5;
        let x3 = rng.next_gauss() * 0.5;
        let prs = rng.next_gauss();
        let entry = 40.0 + 5.0 * rng.next_unit();
        let followup = 0.5 + 8.0 * rng.next_unit();
        let exit = entry + followup;
        // mild signal so events are non-degenerate
        let score = 0.3 * prs + 0.4 * x1 - 0.3 * x2 + 0.2 * x3 + 0.2 * rng.next_gauss();
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
    let formula = format!("Surv(entry_age, exit_age, event) ~ {} + 1", duchon);

    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
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

    // Fit existence is the sealed convergence proof (SPEC 20).

    let logs = log_sink().snapshot();
    // After T13's channel-aware Gram migration the closed-form
    // `compile_from_raw_grams` path handles the shared-constant alias.
    // The compiled-map log line proves the closed-form cutover engaged.
    let compiled_map_marker = "[smgs phase-4b compiled-map] applying CompiledMap T:";
    let compiled_map_seen = logs.iter().any(|line| line.contains(compiled_map_marker));
    assert!(
        compiled_map_seen,
        "expected log line containing {:?}; this proves the closed-form \
         compiled-map cutover engaged. Captured {} log lines.",
        compiled_map_marker,
        logs.len(),
    );

    // After the lift, β block widths must match the RAW design widths
    // (i.e. `marginal_design.ncols()` / `logslope_design.ncols()`), not
    // the compiled (post-drop) widths. If lift were skipped, block β
    // would still be at compiled width.
    let raw_marginal_width = result.marginal_design.design.ncols();
    let raw_logslope_width = result.logslope_design.design.ncols();

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
    let pred_marginal = result
        .marginal_design
        .design
        .apply(&marginal_beta_block.beta);
    for (i, &v) in pred_marginal.iter().enumerate() {
        assert!(v.is_finite(), "marginal η[{i}] non-finite: {v}");
    }
}
