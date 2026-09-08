//! Full-fit reproduction for #979 survival arm through the model service.
//! Mirrors `tests/survival/survival_marginal_slope_stall.rs` (the n=195,780
//! residual-stall repro) but parameterized by positional arguments so the smallest scale that
//! still grinds/hangs can be bisected without rebuilding.
//!
//! Run:
//!   cargo run -p gam-models --profile release-dev --example repro979_survival_margslope -- 600 10

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::{FitConfig, fit_from_formula};
use std::time::Instant;

const N_PCS: usize = 3;

// Inline splitmix64 + Box-Muller (matches the stall test's DGP).
use gam_linalg::utils::splitmix64;
#[inline]
fn next_unit(state: &mut u64) -> f64 {
    let bits = splitmix64(state) >> 11;
    (bits as f64) * (1.0_f64 / ((1u64 << 53) as f64))
}
#[inline]
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

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

fn build_dataset(n: usize) -> EncodedDataset {
    let mut headers = vec![
        "entry_age".to_string(),
        "exit_age".to_string(),
        "event".to_string(),
        "prs_z".to_string(),
    ];
    for i in 0..N_PCS {
        headers.push(format!("PC{}", i + 1));
    }
    headers.push("sex".to_string());

    let mut st: u64 = 0xD0E1_2345_6789_ABCD;
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for _ in 0..n {
        let pcs: Vec<f64> = (0..N_PCS).map(|_| next_gauss(&mut st) * 0.5).collect();
        let prs = next_gauss(&mut st);
        let sex = if next_unit(&mut st) < 0.5 { 1.0 } else { 0.0 };
        let entry = 40.0 + 5.0 * next_unit(&mut st);
        let followup = 0.5 + 8.0 * next_unit(&mut st);
        let exit = entry + followup;
        let score = 0.3 * prs + 0.4 * pcs[0] - 0.3 * pcs[1]
            + 0.2 * pcs[2]
            + 0.15 * sex
            + 0.2 * next_gauss(&mut st);
        let event = if score > 0.0 { 1 } else { 0 };
        let mut rec = vec![
            entry.to_string(),
            exit.to_string(),
            event.to_string(),
            prs.to_string(),
        ];
        for p in &pcs {
            rec.push(p.to_string());
        }
        rec.push(sex.to_string());
        rows.push(StringRecord::from(rec));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode survival margslope dataset")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "macos")]
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::Off);
    faer::set_global_parallelism(faer::Par::rayon(0));
    // Raise the level only if this call is the one that installed LOGGER;
    // losing the race means someone else owns the level too.
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }

    // Positional CLI args: `repro979_survival_margslope [n] [centers]`. Absent
    // args keep the defaults. (Avoids `env::var`, banned by the repo scanner.)
    let n: usize = std::env::args()
        .nth(1)
        .map_or(Ok(600), |s| s.parse())?;
    let centers: usize = std::env::args()
        .nth(2)
        .map_or(Ok(10), |s| s.parse())?;
    if n == 0 || std::env::args().len() > 3 {
        return Err("usage: repro979_survival_margslope [positive n] [centers]".into());
    }

    let data = build_dataset(n);
    let pcs: Vec<String> = (0..N_PCS).map(|i| format!("PC{}", i + 1)).collect();
    let duchon_term = format!("duchon({}, centers={}, order=1)", pcs.join(", "), centers);
    let formula = format!("Surv(entry_age, exit_age, event) ~ {} + sex", duchon_term);
    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("prs_z".to_string()),
        slope_formula: Some(duchon_term),
        baseline_target: "linear".to_string(),
        gpu_policy: gam_gpu::GpuPolicy::Off,
        ..FitConfig::default()
    };

    eprintln!("[979-SURV] starting n={n} centers={centers} formula={formula:?}");
    let start = Instant::now();
    let outcome = fit_from_formula(&formula, &data, &config);
    let elapsed = start.elapsed().as_secs_f64();
    match outcome {
        Ok(_) => eprintln!("[979-SURV] n={n} centers={centers} total_s={elapsed:.2} OK converged"),
        Err(e) => {
            return Err(format!("[979-SURV] n={n} centers={centers} total_s={elapsed:.2} ERR: {e}").into());
        }
    }
    Ok(())
}
