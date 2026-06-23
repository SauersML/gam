//! #1266 DIAGNOSTIC: capture the REML logdet trace for the DEFAULT `s(x)`
//! (penalized B-spline `bs="ps"`, double penalty) fit on PURELY LINEAR data and
//! dump WHICH (penalty_rank, log|S|, log|H|) the optimizer visits. The default
//! `s()` over-fits linear data to EDF ≈ 5 instead of the mgcv `select=TRUE`
//! optimum EDF ≈ 2.1; this probe shows whether the dense `[#1271-diag]` channel
//! even fires (i.e. whether the ps double penalty uses the dense path) and, if
//! so, whether `½(log|H| − log|S|₊)` keeps decreasing as the optimizer raises
//! λ_bend (→ EDF 2, score bug if it stops short) or whether the analytic
//! gradient halts the optimizer at the inflated point.
//!
//! Report-only; panics at the end so the captured trace is surfaced.

use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use std::io::Write;
use std::sync::Mutex;

static DIAG_LINES_1266: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct CaptureLogger1266;

impl log::Log for CaptureLogger1266 {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record) {
        let msg = format!("{}", record.args());
        if msg.contains("#1271-diag") {
            if let Ok(mut g) = DIAG_LINES_1266.lock() {
                g.push(msg);
            }
        }
    }
    fn flush(&self) {}
}

static LOGGER_1266: CaptureLogger1266 = CaptureLogger1266;

fn make_csv(seed: u64) -> std::path::PathBuf {
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
    let path = std::env::temp_dir().join(format!("gam_ps_1266_diag_seed{seed}.csv"));
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
fn diag_1266_dump_ps_double_penalty_logdet() {
    init_parallelism();
    let _ = log::set_logger(&LOGGER_1266);
    log::set_max_level(log::LevelFilter::Info);

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let path = make_csv(1);
    let ds = load_csvwith_inferred_schema(&path).expect("load csv");
    // Default double penalty (the issue's `s(x)`); bs="ps" explicit.
    let result = fit_from_formula("y ~ s(x, bs=\"ps\", k=20)", &ds, &cfg).expect("gam ps fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };

    let inf = fit.fit.inference.as_ref().expect("inference present");
    let edf_total = inf.edf_total;
    let edf_by_block = inf.edf_by_block.clone();
    let lambdas: Vec<f64> = fit.fit.lambdas.to_vec();
    let log_lambdas: Vec<f64> = fit.fit.log_lambdas.to_vec();

    let mut pen_summary = String::new();
    for (i, bp) in fit.design.penalties.iter().enumerate() {
        let (r, c) = bp.local.dim();
        let lam = lambdas.get(i).copied().unwrap_or(f64::NAN);
        let rho = log_lambdas.get(i).copied().unwrap_or(f64::NAN);
        pen_summary.push_str(&format!(
            "\n    penalty[{i}] col_range={:?} dim={r}x{c} lambda={lam:.6e} rho={rho:.4}",
            bp.col_range
        ));
    }

    let lines = DIAG_LINES_1266.lock().map(|g| g.clone()).unwrap_or_default();

    let mut report = String::new();
    report.push_str("\n============ #1266 ps DOUBLE-PENALTY LOGDET PROBE ============\n");
    report.push_str(&format!("edf_total={edf_total:.4} (mgcv select=TRUE ≈ 2.10)\n"));
    report.push_str(&format!("edf_by_block={edf_by_block:?}\n"));
    report.push_str(&format!("lambdas={lambdas:?}\n"));
    report.push_str(&format!("log_lambdas(rho)={log_lambdas:?}\n"));
    report.push_str(&format!("penalties:{pen_summary}\n"));
    report.push_str(&format!(
        "\n[#1271-diag] lines captured: {} (0 => ps double NOT on dense path)\n",
        lines.len()
    ));
    for l in lines.iter() {
        report.push_str(l);
        report.push('\n');
    }
    report.push_str("============================================================\n");
    // `println!` (NOT eprintln) so the trace shows under `--nocapture` without a
    // banned debug-format eprintln and without a permanent-red panic.
    println!("{report}");
    assert!(edf_total.is_finite(), "ps double-penalty fit edf must be finite");
}
