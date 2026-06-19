//! #1271 DIAGNOSTIC (not a quality gate): dump the REML logdet internals of the
//! thin-plate `bs="tp"` fit on PURELY LINEAR data so we can see WHERE the cost
//! surface picks a wiggly interior optimum (EDF ~5.34) instead of the linear
//! truth (mgcv EDF ~2.10).
//!
//! Mechanism under test: the combined-penalty `log|S|₊` rank used in the REML
//! cost vs. the count of `log|H|` eigenvalues that actually grow with λ. If
//! penalty_rank stays constant while `log|H|` keeps climbing as λ→∞, there is a
//! spurious `Δrank·ρ` slope -> false interior minimum -> over-fit.
//!
//! Output channel: `objective.rs` emits one `log::info!` line per dense REML
//! evaluation (`[#1271-diag] ...`). This test installs a process-global logger
//! that captures those records into a buffer, runs the fit, prints the captured
//! trace + a final summary, then PANICS with the summary so that nextest dumps
//! the captured stderr even when the harness does not pass `--nocapture`.

use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use std::io::Write;
use std::sync::Mutex;

static DIAG_LINES: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct CaptureLogger;

impl log::Log for CaptureLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record) {
        let msg = format!("{}", record.args());
        if msg.contains("#1271-diag") {
            if let Ok(mut g) = DIAG_LINES.lock() {
                g.push(msg);
            }
        }
    }
    fn flush(&self) {}
}

static LOGGER: CaptureLogger = CaptureLogger;

fn make_csv(seed: u64) -> std::path::PathBuf {
    // EXACT #1271 DGP (copied from quality_vs_mgcv_thin_plate_1d.rs):
    // y = 2 + 3x + 0.15*N(0,1), x = linspace(0,1,800), deterministic LCG+Box-Muller.
    let n = 800usize;
    let mut state = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    let mut next_unit = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut normal = || {
        let u1 = next_unit().max(1e-12);
        let u2 = next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    let path = std::env::temp_dir().join(format!("gam_tp_1271_diag_seed{seed}.csv"));
    let mut f = std::fs::File::create(&path).expect("create csv");
    writeln!(f, "x,y").unwrap();
    for i in 0..n {
        let x = i as f64 / (n as f64 - 1.0);
        let y = 2.0 + 3.0 * x + 0.15 * normal();
        writeln!(f, "{x:.12},{y:.12}").unwrap();
    }
    path
}

#[test]
fn diag_1271_dump_reml_logdet_internals() {
    init_parallelism();

    // Install the capture logger (max level Info so objective.rs emits).
    // `set_logger` fails if another logger is already installed; ignore that
    // (still works if the other logger forwards, and the panic-dump fallback
    // below prints whatever we have).
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Info);

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let path = make_csv(1);
    let ds = load_csvwith_inferred_schema(&path).expect("load csv");
    let result = fit_from_formula("y ~ s(x, bs=\"tp\", k=20)", &ds, &cfg).expect("gam tp fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };

    let inf = fit.fit.inference.as_ref().expect("inference present");
    let edf_total = inf.edf_total;
    let edf_by_block = inf.edf_by_block.clone();
    let block_trace = inf.penalty_block_trace.clone();

    // Path-independent decisive numbers from the public fit: the SELECTED
    // smoothing parameters (small λ = under-smoothing/genuine cost bug;
    // large λ + high EDF = EDF-reporting bug) and the per-penalty influence
    // trace tr_kk = λ_kk·tr(H⁻¹ S_kk) (which penalty does the EDF work).
    let lambdas: Vec<f64> = fit.fit.lambdas.to_vec();
    let log_lambdas: Vec<f64> = fit.fit.log_lambdas.to_vec();

    // Penalty inventory: how many penalties ship, their col_range/dim, plus the
    // per-penalty λ, ρ=ln λ, and trace aligned 1:1 (lambdas / penalty_block_trace
    // are in penalty order, same as fit.design.penalties).
    let mut pen_summary = String::new();
    for (i, bp) in fit.design.penalties.iter().enumerate() {
        let (r, c) = bp.local.dim();
        let lam = lambdas.get(i).copied().unwrap_or(f64::NAN);
        let rho = log_lambdas.get(i).copied().unwrap_or(f64::NAN);
        let tr = block_trace.get(i).copied().unwrap_or(f64::NAN);
        pen_summary.push_str(&format!(
            "\n    penalty[{i}] col_range={:?} dim={r}x{c} lambda={lam:.6e} rho={rho:.4} trace={tr:.4}",
            bp.col_range
        ));
    }
    pen_summary.push_str(&format!(
        "\n    ALL lambdas={lambdas:?}\n    ALL log_lambdas(rho)={log_lambdas:?}\n    ALL block_trace={block_trace:?}"
    ));

    // Drain the captured per-evaluation diagnostic trace.
    let lines = DIAG_LINES.lock().map(|g| g.clone()).unwrap_or_default();

    let mut report = String::new();
    report.push_str("\n==================== #1271 REML LOGDET PROBE ====================\n");
    report.push_str(&format!("num_penalties={}{}\n", fit.design.penalties.len(), pen_summary));
    report.push_str(&format!("FINAL edf_total={edf_total:.6} edf_by_block={edf_by_block:?}\n"));
    report.push_str(&format!("captured {} dense REML evaluations:\n", lines.len()));
    for l in &lines {
        report.push_str("  ");
        report.push_str(l);
        report.push('\n');
    }
    // The LAST captured evaluation is the one at (or nearest) the selected
    // optimum that produced the reported EDF.
    if let Some(last) = lines.last() {
        report.push_str("\n--- LAST (≈selected optimum) ---\n  ");
        report.push_str(last);
        report.push('\n');
    }
    report.push_str("================================================================\n");

    // Force nextest to surface the captured output even without --nocapture by
    // failing with the full report as the panic message. This test is a probe,
    // not a gate; the "failure" is the report.
    panic!("{report}");
}
