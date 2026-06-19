//! Scratch repro for #979: time the survival marginal-slope fit that the
//! quality test `gam_marginal_slope_heldout_concordance_matches_or_beats_lifelines_coxph`
//! exercises, WITHOUT the python/lifelines part. Mirrors the test's DGP exactly:
//! the heart-failure dataset, n=200 #1082-bounded slice, deterministic train split,
//! survival marginal-slope, logslope s(age, bs='tp', k=4), marginal sex+age.

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

fn main() {
    init_parallelism();
    let _ = log::set_logger(&LOGGER).map(|()| log::set_max_level(log::LevelFilter::Info));
    let mut ds = load_csvwith_inferred_schema(Path::new(HEART_CSV)).expect("load heart-failure csv");
    let n_full = ds.values.nrows();
    let analysis_rows: Vec<usize> = (0..n_full).filter(|&i| i % 3 != 2).collect();
    let mut analysis_values = Array2::<f64>::zeros((analysis_rows.len(), ds.headers.len()));
    for (out_row, &src_row) in analysis_rows.iter().enumerate() {
        analysis_values.row_mut(out_row).assign(&ds.values.row(src_row));
    }
    ds.values = analysis_values;
    let n = ds.values.nrows();

    let is_test = |i: usize| i % 3 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
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
        ..FitConfig::default()
    };
    eprintln!("[repro] starting fit n_train={n_train}");
    let t = Instant::now();
    let result = fit_from_formula("Surv(time, DEATH_EVENT) ~ sex + age", &ds_train, &cfg)
        .expect("gam survival marginal-slope fit");
    let elapsed = t.elapsed();
    let FitResult::SurvivalMarginalSlope(fit) = result else {
        panic!("expected SurvivalMarginalSlope");
    };
    eprintln!(
        "[repro] DONE elapsed={:.2}s outer_iters={} inner_cycles={} outer_converged={} p={} reml={:.4}",
        elapsed.as_secs_f64(),
        fit.fit.outer_iterations,
        fit.fit.inner_cycles,
        fit.fit.outer_converged,
        fit.fit.beta.len(),
        fit.fit.reml_score,
    );
}
