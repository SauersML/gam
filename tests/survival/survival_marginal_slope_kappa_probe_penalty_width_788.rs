//! Regression test for #788 — survival marginal-slope penalty-block shape
//! mismatch (`block i penalty 0 must be NxN, got KxK`).
//!
//! Root cause: when the V+M-exact cutover fires it compiles the marginal /
//! log-slope designs into a *reduced* coordinate system and pulls each block's
//! penalty back to a matching **compiled**-width `Vᵀ S V`. The outer
//! spatial-length-scale (κ) optimizer, however, re-materialises **raw**-width
//! designs from the boot specs on every probe and routes them back through
//! `build_blocks`. The old code substituted the compiled-width `*_penalties_vm`
//! onto whatever design `build_blocks` received, guessing applicability from a
//! `penalty.shape() == design.shape()` width coincidence. On a κ probe that
//! paired a raw `K×K` design with a compiled `2×2` penalty and aborted the fit
//! with `IntegrationError: ... block 1 penalty 0 must be KxK, got 2x2`.
//!
//! The fix tags each `build_blocks` call site with an explicit
//! `BlockDesignCoords` provenance: construction-site calls (`PostCutover`)
//! install the compiled penalties; κ-probe calls (`RematerializedRaw`) keep the
//! raw design-derived penalties. This test reproduces the failing geometry:
//!
//!   * a `matern` surface — which carries a length scale, so the κ optimizer
//!     engages and the re-materialise-raw probe path runs (the bug locus), and
//!   * the SAME `matern` surface on the log-slope side plus a parametric
//!     intercept, which forces the V+M compile to drop a shared column and so
//!     fires the cutover that produces the compiled-width `*_penalties_vm`.
//!
//! Before the fix this fit aborts with the penalty-shape mismatch. After it,
//! the fit runs to completion with finite coefficients.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use std::sync::{Arc, Mutex, Once, OnceLock};

#[path = "../common/fixtures.rs"]
mod fixtures;
use fixtures::Splitmix64;

const N: usize = 360;

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
        self.sink.push(format!("{}", record.args()));
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
        "pc1".to_string(),
        "pc2".to_string(),
        "pc3".to_string(),
        "sex".to_string(),
    ];

    let mut rng = Splitmix64::new(0x788D_E517_u64);
    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);
    for _ in 0..N {
        let pc1 = rng.next_gauss();
        let pc2 = rng.next_gauss();
        let pc3 = rng.next_gauss();
        let prs = rng.next_gauss();
        let sex = if rng.next_unit() < 0.5 { 0.0 } else { 1.0 };
        let entry = 40.0 + 5.0 * rng.next_unit();
        let followup = 0.5 + 8.0 * rng.next_unit();
        let exit = entry + followup;
        // Mild signal in the marginal axis (prs) and the PC surface so events
        // are non-degenerate and the marginal-slope coupling is exercised.
        let score = 0.4 * prs + 0.3 * pc1 - 0.2 * pc2 + 0.15 * sex + 0.2 * rng.next_gauss();
        let event = if score > 0.0 { 1 } else { 0 };
        rows.push(StringRecord::from(vec![
            entry.to_string(),
            exit.to_string(),
            event.to_string(),
            prs.to_string(),
            pc1.to_string(),
            pc2.to_string(),
            pc3.to_string(),
            sex.to_string(),
        ]));
    }

    encode_recordswith_inferred_schema(headers, rows)
        .expect("encode synthetic #788 marginal-slope dataset")
}

#[test]
fn survival_marginal_slope_kappa_probe_keeps_raw_penalty_widths_788() {
    install_logger();
    init_parallelism();
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    let data = build_dataset();

    // `matern` carries a length scale → the spatial-length-scale (κ) optimizer
    // engages → the raw-design re-materialise probe path runs. The SAME matern
    // surface on the log-slope side plus the marginal intercept forces the V+M
    // compile to drop a shared column, firing the cutover that produces the
    // compiled-width `*_penalties_vm` whose mis-installation is #788.
    let matern = "matern(pc1, pc2, pc3, centers=8)".to_string();
    let formula = format!("Surv(entry_age, exit_age, event) ~ {} + sex", matern);

    let config = FitConfig {
        survival_likelihood: "marginal-slope".to_string(),
        z_column: Some("prs_z".to_string()),
        logslope_formula: Some(matern.clone()),
        baseline_target: "linear".to_string(),
        gpu_policy: if cfg!(target_os = "macos") {
            gam::gpu::GpuPolicy::Off
        } else {
            gam::gpu::GpuPolicy::Auto
        },
        ..FitConfig::default()
    };

    let outcome = fit_from_formula(&formula, &data, &config);

    // #788 is precisely the penalty-block *shape* invariant violation
    // (`block i penalty 0 must be NxN, got KxK`) — a compiled-width `_vm`
    // penalty installed onto a raw-width κ-probe design. The fix makes that
    // invariant hold; whether this small synthetic problem then *converges* is
    // governed by the separate, explicitly-related nullspace cluster
    // (#785 / #754 / #787 — the unpenalised baseline/surface polynomial null
    // space), so this test deliberately does NOT gate on convergence. It
    // asserts only that the shape mismatch never resurfaces, and — if the fit
    // does succeed — that no corrupted penalty leaked into the coefficients.
    if let Err(err) = &outcome {
        let msg = err.to_string();
        assert!(
            !(msg.contains("must be") && msg.contains(", got ") && msg.contains("penalty")),
            "#788 regression: survival marginal-slope κ probe installed a \
             compiled-width penalty onto a raw-width design (shape-mismatch \
             assertion fired): {msg}"
        );
    }

    // The guard above is only meaningful if the run actually exercised the bug
    // locus: (a) the V+M cutover fired — so compiled-width `_vm` penalties
    // exist that *could* be mis-installed — and (b) the κ optimizer ran an
    // outer eval/probe — so raw-width designs flowed back into `build_blocks`.
    // Assert both so the test can never silently stop covering #788.
    let logs = log_sink().snapshot();
    let cutover_seen = logs
        .iter()
        .any(|l| l.contains("[smgs phase-4b compiled-map] applying CompiledMap T:"));
    let outer_probe_seen = logs.iter().any(|l| {
        l.contains("[survival-marginal-slope/outer-inner-fit]")
            || l.contains("[survival-marginal-slope/outer-eval]")
    });
    assert!(
        cutover_seen,
        "expected the V+M compiled-map cutover to engage (so compiled-width \
         penalties exist to be mis-installed). Captured {} log lines.",
        logs.len()
    );
    assert!(
        outer_probe_seen,
        "expected the κ optimizer outer eval/probe to run (so raw-width designs \
         flow back into build_blocks). Captured {} log lines.",
        logs.len()
    );

    // If the fit did complete, no corrupted penalty may have leaked into β.
    if let Ok(FitResult::SurvivalMarginalSlope(result)) = &outcome {
        for (idx, block) in result.fit.blocks.iter().enumerate() {
            for (j, &coef) in block.beta.iter().enumerate() {
                assert!(
                    coef.is_finite(),
                    "β block {idx} coef {j} non-finite: {coef}"
                );
            }
        }
    }
}
