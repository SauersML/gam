//! #979 LIVE hang repro at TINY scale: the lifelines survival marginal-slope
//! quality fit (heart-failure real data, n_train≈133) that times out at
//! 1400s+ on MSI (job 11417349 rc=124). Mirrors the fit in
//! tests/quality/quality_vs_lifelines_cox_like_marginal.rs exactly, minus the
//! comparator/concordance scoring, so RUST_LOG=info captures the inner-solve
//! stall trace locally in seconds-to-minutes instead of a 23-min MSI cycle.
//!
//! Run:
//!   RUST_LOG=info cargo run --profile release-dev --example repro979_heart_margslope 2>&1 | tee /tmp/heart979.log

use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;
use std::time::Instant;

const HEART_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/heart_failure_clinical_records_dataset.csv"
);

struct StderrInfoLogger;
impl log::Log for StderrInfoLogger {
    fn enabled(&self, m: &log::Metadata<'_>) -> bool {
        m.level() <= log::Level::Info
    }
    fn log(&self, r: &log::Record<'_>) {
        if self.enabled(r.metadata()) {
            eprintln!("{}", r.args());
        }
    }
    fn flush(&self) {}
}
static LOGGER: StderrInfoLogger = StderrInfoLogger;

fn main() {
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);
    init_parallelism();
    let _ = log::set_logger(&LOGGER).map(|()| log::set_max_level(log::LevelFilter::Info));

    let mut ds = load_csvwith_inferred_schema(Path::new(HEART_CSV)).expect("load heart csv");
    let n_full = ds.values.nrows();
    assert_eq!(n_full, 299, "expected n=299");

    // Same #1082 bounded slice as the test: keep rows where i%3 != 2 → n=200.
    let analysis_rows: Vec<usize> = (0..n_full).filter(|&i| i % 3 != 2).collect();
    let mut analysis_values = Array2::<f64>::zeros((analysis_rows.len(), ds.headers.len()));
    for (out_row, &src_row) in analysis_rows.iter().enumerate() {
        analysis_values
            .row_mut(out_row)
            .assign(&ds.values.row(src_row));
    }
    ds.values = analysis_values;
    let n = ds.values.nrows();
    assert_eq!(n, 200);

    // Same deterministic train split: hold out every i%3==0 → n_train≈133.
    let train_rows: Vec<usize> = (0..n).filter(|&i| i % 3 != 0).collect();
    let n_train = train_rows.len();
    let mut train_values = Array2::<f64>::zeros((n_train, ds.headers.len()));
    for (r, &src) in train_rows.iter().enumerate() {
        train_values.row_mut(r).assign(&ds.values.row(src));
    }
    let mut ds_train = ds.clone();
    ds_train.values = train_values;

    let cfg = FitConfig {
        survival_likelihood: "marginal-slope".to_string(),
        z_column: Some("ejection_fraction".to_string()),
        logslope_formula: Some("s(age, bs='tp', k=4)".to_string()),
        baseline_target: "linear".to_string(),
        gpu_policy: gam::gpu::GpuPolicy::Off,
        ..FitConfig::default()
    };

    eprintln!(
        "[979-HEART] starting survival marginal-slope fit n_train={n_train} (HEAD #979 hang repro)"
    );
    let start = Instant::now();
    let result = fit_from_formula("Surv(time, DEATH_EVENT) ~ sex + age", &ds_train, &cfg);
    let elapsed = start.elapsed().as_secs_f64();
    match result {
        Ok(FitResult::SurvivalMarginalSlope(fit)) => {
            eprintln!(
                "[979-HEART] DONE n_train={n_train} total_s={elapsed:.2} outer_iters={} outer_converged={} reml={:.4}",
                fit.fit.outer_iterations, fit.fit.outer_converged, fit.fit.reml_score
            );
        }
        Ok(_) => eprintln!("[979-HEART] wrong variant after {elapsed:.2}s"),
        Err(e) => {
            let m = e.to_string();
            let short: String = m.chars().take(300).collect();
            eprintln!("[979-HEART] n_train={n_train} total_s={elapsed:.2} ERR: {short}");
        }
    }
}
