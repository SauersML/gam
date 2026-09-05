//! Reproduce #2827 through the public formula API on a bounded CSV prefix.
//!
//! Usage: spatial_tensor_profile_2827 DATA.csv ROWS [FORMULA]
//! Every log line carries elapsed seconds so tensor setup can be distinguished
//! from the subsequent optimizer. ROWS is explicit to keep iteration bounded.

use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use std::sync::OnceLock;
use std::time::Instant;

static START: OnceLock<Instant> = OnceLock::new();
struct ElapsedLogger;
static LOGGER: ElapsedLogger = ElapsedLogger;

impl log::Log for ElapsedLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!(
                "[{:.6}s] {} {}",
                START.get().expect("logger clock initialized").elapsed().as_secs_f64(),
                record.level(),
                record.args(),
            );
        }
    }

    fn flush(&self) {}
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: spatial_tensor_profile_2827 DATA.csv ROWS [FORMULA]".into());
    }
    let rows = args[2].parse::<usize>().map_err(|error| error.to_string())?;
    if rows == 0 {
        return Err("ROWS must be positive".into());
    }
    let formula = args.get(3).map(String::as_str).unwrap_or(
        "yg ~ duchon(x1,x2,x3,x4,x5,x6,centers=100,length_scale=1)",
    );
    let mut reader = csv::Reader::from_path(&args[1]).map_err(|error| error.to_string())?;
    let headers = reader.headers().map_err(|error| error.to_string())?
        .iter().map(str::to_string).collect();
    let records = reader.records().take(rows).collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    let actual_rows = records.len();
    let data = encode_recordswith_inferred_schema(headers, records)
        .map_err(|error| error.to_string())?;
    let config = FitConfig { family: Some("gaussian".into()), ..FitConfig::default() };
    eprintln!("[2827-public] rows={actual_rows} formula={formula}");
    let started = Instant::now();
    let result = fit_from_formula(formula, &data, &config).map_err(|error| error.to_string())?;
    let FitResult::Standard(result) = result else {
        return Err("expected a standard Gaussian fit".into());
    };
    eprintln!(
        "[2827-public] rows={actual_rows} fit_seconds={:.6} objective={:?} deviance={:.12} outer_iterations={} log_lambdas={:?} kappa_timing={:?}",
        started.elapsed().as_secs_f64(), result.fit.reml_score(),
        result.fit.deviance, result.fit.outer_iterations, result.fit.log_lambdas,
        result.kappa_timing,
    );
    Ok(())
}

fn main() -> Result<(), String> {
    START.set(Instant::now()).expect("initialize logger clock once");
    log::set_logger(&LOGGER).map_err(|error| error.to_string())?;
    log::set_max_level(log::LevelFilter::Info);
    std::thread::Builder::new().name("spatial-profile-2827".into())
        .stack_size(64 << 20).spawn(run).map_err(|error| error.to_string())?
        .join().map_err(|_| "fit worker panicked".to_string())?
}
