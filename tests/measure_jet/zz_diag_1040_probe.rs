//! #1040 diagnostic probe (temporary). Installs a stderr info logger and runs a
//! small survival marginal-slope fit so the inner joint-Newton convergence
//! trace (`[PIRLS/joint-Newton convergence]`, `[PIRLS/JN/#1040-cond]`, the
//! Newton-decrement certificate line) is observable. Deleted after diagnosis.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use std::sync::Once;
use std::time::Instant;

const N: usize = 300;
const CENTERS: usize = 4;

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

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn build_dataset() -> gam::inference::data::EncodedDataset {
    let headers = ["time", "event", "z", "PC1", "PC2"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let mut state: u64 = 0x5CA1_AB1E_1040_C0DE_u64;
    let mut z_raw: Vec<f64> = Vec::with_capacity(N);
    let mut scratch: Vec<[f64; 3]> = Vec::with_capacity(N);
    for _ in 0..N {
        let pc1 = next_gauss(&mut state) * 0.5;
        let pc2 = next_gauss(&mut state) * 0.5;
        let z = next_gauss(&mut state);
        let u_event = next_unit(&mut state).max(1e-9);
        z_raw.push(z);
        scratch.push([pc1, pc2, u_event]);
    }
    let z_mean = z_raw.iter().sum::<f64>() / N as f64;
    let z_var = z_raw.iter().map(|v| (v - z_mean).powi(2)).sum::<f64>() / N as f64;
    let z_sd = z_var.sqrt().max(1e-12);
    let z_std: Vec<f64> = z_raw.iter().map(|v| (v - z_mean) / z_sd).collect();
    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);
    for (i, s) in scratch.iter().enumerate() {
        let [pc1, pc2, u_event] = *s;
        let z = z_std[i];
        let lin = 0.4 * pc1 - 0.3 * pc2 + 0.35 * z;
        let rate = lin.exp().max(1e-6);
        let t_event = -u_event.ln() / rate;
        let t_cens = 1.5 * next_unit(&mut state);
        let (time, event) = if t_event <= t_cens {
            (t_event, 1u8)
        } else {
            (t_cens, 0u8)
        };
        let time = time.max(1e-4);
        rows.push(StringRecord::from(vec![
            time.to_string(),
            event.to_string(),
            z.to_string(),
            pc1.to_string(),
            pc2.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode probe dataset")
}

#[test]
fn zz_diag_1040_probe() {
    INIT_LOGGER.call_once(|| {
        log::set_logger(&LOGGER).ok();
        log::set_max_level(log::LevelFilter::Info);
    });
    init_parallelism();
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    let data = build_dataset();
    let matern = format!("matern(PC1, PC2, centers={CENTERS})");
    let formula = format!("Surv(time, event) ~ {matern}");
    let cfg = FitConfig {
        survival_likelihood: "marginal-slope".to_string(),
        z_column: Some("z".to_string()),
        logslope_formula: Some(matern.clone()),
        baseline_target: "linear".to_string(),
        ..FitConfig::default()
    };
    let start = Instant::now();
    let result = fit_from_formula(&formula, &data, &cfg);
    let elapsed = start.elapsed().as_secs_f64();
    match result {
        Ok(FitResult::SurvivalMarginalSlope(fit)) => {
            eprintln!(
                "[PROBE-DONE] t={elapsed:.2}s outer_iters={} inner_cycles={} converged={}",
                fit.fit.outer_iterations, fit.fit.inner_cycles, fit.fit.outer_converged
            );
            // #1040 acceptance: the survival marginal-slope outer/inner loop must
            // reach its stopping criterion (not grind the inner joint-Newton cycle
            // budget on the full-rank-but-ill-conditioned H_pen). The inner solve
            // never exhausting its budget is the bug this probe pins.
            assert!(
                fit.fit.inner_cycles < gam::custom_family::DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES,
                "survival-MS inner joint-Newton ground its full {} cycle budget without \
                 converging (gam#1040 ill-conditioning hang); cycles={}",
                gam::custom_family::DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES,
                fit.fit.inner_cycles,
            );
        }
        Ok(_) => panic!("[PROBE] t={elapsed:.2}s unexpected non-survival-MS variant"),
        Err(e) => panic!("[PROBE-ERR] t={elapsed:.2}s err={e}"),
    }
}
