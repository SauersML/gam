//! Large-scale repro for the V+M-exact cutover (team `smgs-vm-exact`).
//!
//! Original failure: at n = 195_780 a survival marginal-slope fit using
//! the formulas
//!     Surv(entry_age, exit_age, event) ~ duchon(PC1, PC2, PC3, centers=10, order=1)
//!                                        + sex + linkwiggle()
//!     logslope: duchon(PC1, PC2, PC3, centers=10, order=1) + linkwiggle()
//! stalled in "pilot ignored-error" after ~56 minutes. Under V-only the
//! canonicalize gate failed closed because the duplicated duchon term
//! across the marginal and log-slope formulas produced a multi-dimensional
//! null-space among shared-constant columns. Under V+M the joint design
//! reduces correctly and the fit completes.
//!
//! This test rebuilds the same *structural* shape (duplicated duchon +
//! linkwiggle on both formulas) at n = 400 and centers = 4 to keep RAM
//! safe, and asserts:
//!   * the fit converges (a minted fit is the sealed convergence proof, SPEC 20),
//!   * the V+M log line `[smgs phase-4b compiled-map] applying CompiledMap T: ...`
//!     fires with at least one drop reported (closed-form channel-aware path),
//!   * fitted β block widths still match RAW design widths (T-lift ran),
//!   * every β coefficient is finite,
//!   * predictions on training rows are finite.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

use crate::fixtures::Splitmix64;
use std::sync::{Arc, Mutex, Once, OnceLock};
use std::time::Instant;

const N: usize = 400;
// Duchon order=1 in 3D has a 4-dimensional polynomial null space
// (constant + 3 linear coords), so `centers` must exceed 4 for the
// smooth's reproducing-kernel block to add any new directions on top
// of the parametric nullspace. Picking 6 keeps p_total small for a
// fast test while still letting V+M residualization see a real drop.
const CENTERS: usize = 6;

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
        "prs".to_string(),
        "PC1".to_string(),
        "PC2".to_string(),
        "PC3".to_string(),
        "sex".to_string(),
    ];

    let mut rng = Splitmix64::new(0xB10B_A11C_5EED_2026_u64);
    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);

    for _ in 0..N {
        let pc1 = rng.next_gauss() * 0.5;
        let pc2 = rng.next_gauss() * 0.5;
        let pc3 = rng.next_gauss() * 0.5;
        let prs = rng.next_gauss();
        let sex = if rng.next_unit() < 0.5 { 1.0 } else { 0.0 };
        let entry = 40.0 + 5.0 * rng.next_unit();
        let followup = 0.5 + 8.0 * rng.next_unit();
        let exit = entry + followup;
        let score =
            0.3 * prs + 0.4 * pc1 - 0.3 * pc2 + 0.2 * pc3 + 0.15 * sex + 0.2 * rng.next_gauss();
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
        .expect("encode synthetic large-scale survival marginal-slope dataset")
}

#[test]
fn survival_marginal_slope_large_scale_repro_vm_exact_engages_and_converges() {
    install_logger();
    init_parallelism();
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    let data = build_dataset();

    // The duplicated duchon term across mean and log-slope formulas, plus
    // a parametric `sex` term and a `linkwiggle()` deviation block,
    // recreates the original large-scale failure shape that overwhelmed the
    // V-only canonicalize path. centers=CENTERS keeps p_total small.
    let duchon = format!("duchon(PC1, PC2, PC3, centers={}, order=1)", CENTERS);
    let formula = format!(
        "Surv(entry_age, exit_age, event) ~ {} + sex + linkwiggle()",
        duchon
    );
    let logslope = format!("{} + linkwiggle()", duchon);

    let config = FitConfig {
        survival_likelihood: "marginal-slope".to_string(),
        z_column: Some("prs".to_string()),
        logslope_formula: Some(logslope),
        baseline_target: "linear".to_string(),
        gpu_policy: if cfg!(target_os = "macos") {
            gam::gpu::GpuPolicy::Off
        } else {
            gam::gpu::GpuPolicy::Auto
        },
        ..FitConfig::default()
    };

    // gam#979 regression guard: the survival `marginal-slope` path used to
    // HANG (rc=124) — the inner coupled joint-Newton ground to ~1000-1200
    // cycles and the generic pairwise `p(p+1)/2` Jeffreys completion turned
    // every near-converged polishing cycle into tens of seconds of full-data
    // row work. After the Moré–Sorensen trust-region step + the
    // contracted-hook gate on the completion (the generic pairwise fallback
    // no longer fires at production scale), this structural shape converges
    // in a handful of cycles. Assert a hard wall-clock budget so a
    // re-introduction of the hang/grind fails loudly instead of timing out
    // silently. The budget is deliberately generous relative to the observed
    // sub-second fit at this size (n=400, centers=6) to stay non-flaky under
    // CI contention while still being orders of magnitude below the old hang.
    let fit_start = Instant::now();
    let outcome = fit_from_formula(&formula, &data, &config)
        .expect("large-scale survival marginal-slope fit should succeed under V+M");
    let fit_elapsed = fit_start.elapsed();
    const SURVIVAL_MARGSLOPE_BUDGET_SECS: f64 = 120.0;
    assert!(
        fit_elapsed.as_secs_f64() < SURVIVAL_MARGSLOPE_BUDGET_SECS,
        "survival marginal-slope fit took {:.1}s, exceeding the {:.0}s budget \
         (gam#979 hang/grind regression): the structural duplicated-duchon + \
         linkwiggle shape at n={N}, centers={CENTERS} must converge promptly",
        fit_elapsed.as_secs_f64(),
        SURVIVAL_MARGSLOPE_BUDGET_SECS,
    );

    let result = match outcome {
        FitResult::SurvivalMarginalSlope(r) => r,
        other => panic!(
            "expected FitResult::SurvivalMarginalSlope, got {:?}",
            std::mem::discriminant(&other)
        ),
    };

    // Fit existence is the sealed convergence proof (SPEC 20).

    let logs = log_sink().snapshot();
    // After T13's channel-aware Gram migration, the closed-form path
    // handles the shared-column alias. Assert the compiled-map log fires.
    let compiled_map_marker = "[smgs phase-4b compiled-map] applying CompiledMap T:";
    let compiled_map_lines: Vec<&String> = logs
        .iter()
        .filter(|line| line.contains(compiled_map_marker))
        .collect();
    assert!(
        !compiled_map_lines.is_empty(),
        "expected log line containing {:?}; this proves the closed-form compiled-map \
         cutover engaged on the large-scale formula. Captured {} log lines.",
        compiled_map_marker,
        logs.len(),
    );

    // The compiled-map log reports per-channel drops; at least one block
    // must report a non-zero drop on this duplicated-duchon shape.
    let saw_drop = compiled_map_lines.iter().any(|line| {
        // Log format: "drops time=N, marginal=M, logslope=P"
        if let Some(idx) = line.find("drops ") {
            let tail = &line[idx..];
            tail.chars().any(|c| matches!(c, '1'..='9'))
        } else {
            false
        }
    });
    assert!(
        saw_drop,
        "no compiled-map log line reported a non-zero drop on the duplicated-duchon \
         large-scale repro; captured {} compiled-map lines, e.g. {:?}",
        compiled_map_lines.len(),
        compiled_map_lines.first()
    );

    // After lift, β block widths must match RAW design widths.
    let raw_marginal_width = result.marginal_design.design.ncols();
    let raw_logslope_width = result.logslope_design.design.ncols();
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

    for (idx, block) in result.fit.blocks.iter().enumerate() {
        for (j, &coef) in block.beta.iter().enumerate() {
            assert!(
                coef.is_finite(),
                "β block {idx} coef {j} non-finite: {coef}"
            );
        }
    }

    // Lightweight predict probe on training rows: η_marginal = X_marg β_marg.
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
