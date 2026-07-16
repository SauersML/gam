//! Red test: reproduce the PIRLS joint-Newton residual-stall early-exit
//! observed in production survival_marginal_slope fits at large scale.
//!
//! Production signature (n=195,780, p=33, 5/5 outer seeds failed):
//!
//!   - cycle 0: |prop|∞ ≈ 2e5, TR clamps to |δ|∞=20, |β|∞=20.0
//!   - cycles 1..25: linearized_rel ratio ≈ 0.97 for 15+ cycles
//!   - residual-stall early-exit triggers
//!   - budget-exhausted dump shows ALL gradient in the time block:
//!         block_widths   = [12, 11, 10]
//!         block_beta_inf = [2.3e-4, 15.3, 20.0]
//!         block_grad_inf = [5.6e8,  1.5e3, 2.3e3]
//!
//! Hypothesis under test: the joint-Newton trust region uses an isotropic
//! L2-norm step constraint over the concatenated δ. When one block (the
//! time block) has near-singular curvature, the unconstrained Newton step
//! has huge norm in that direction; the global L2 clamp rescales the
//! ENTIRE δ uniformly, so the marginal/logslope blocks receive an
//! arbitrarily small fraction of their "fair" step and the time gradient
//! stays large forever.
//!
//! This test induces the same regime on small synthetic data by:
//!   * over-parameterising the time block (many internal knots over a
//!     narrow age window),
//!   * making the upper part of the time axis nearly event-free so
//!     several time-basis columns carry almost no Fisher information,
//!   * disabling the time smoothing prior (very small lambda) so that
//!     prior is not what saves us.
//!
//! Current expectation: this test FAILS — the outer optimizer rejects
//! every seed because the inner joint-Newton stalls (residual-stall
//! early-exit). When the bug is fixed (per-block / preconditioned
//! anisotropic TR), the assertion `outer_converged == true` will pass.

use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
use std::sync::Once;
use std::time::Instant;

const N: usize = 195_780;
const N_PCS: usize = 3;

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

fn init() {
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);
    init_parallelism();
    INIT_LOGGER.call_once(|| {
        if log::set_logger(&LOGGER).is_ok() {
            log::set_max_level(log::LevelFilter::Info);
        }
    });
}

// ── Inline RNG (splitmix64 + Box-Muller, no external state) ───────────────

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn next_unit(state: &mut u64) -> f64 {
    let bits = splitmix64(state) >> 11;
    (bits as f64) * (1.0_f64 / ((1u64 << 53) as f64))
}

#[inline]
fn next_gauss(state: &mut u64) -> f64 {
    // Box-Muller with safe lower bound on u1.
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    r * theta.cos()
}

fn next_gamma_alpha_ge_one(state: &mut u64, alpha: f64, scale: f64) -> f64 {
    assert!(alpha >= 1.0);
    let d = alpha - 1.0 / 3.0;
    let c = (1.0 / (9.0 * d)).sqrt();
    loop {
        let x = next_gauss(state);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v3 = v * v * v;
        let u = next_unit(state);
        if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return scale * d * v3;
        }
    }
}

#[inline]
fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

/// Build the same frame shape and first-order column statistics as the
/// failing large-scale survival-marginal-slope run:
///
///   rows=195,780 columns=[entry_age, exit_age, event, sex, prs_z, PC1..PC3]
///   event mean=0.5, sex mean≈0.391, entry_age mean≈45, exit_age mean≈53
///
/// This deliberately avoids the old low-event synthetic fixture, which
/// reaches a different startup-validation error before the coupled
/// exact-joint residual-stall path.
fn build_dataset() -> gam::inference::data::EncodedDataset {
    let mut headers = vec![
        "entry_age".to_string(),
        "exit_age".to_string(),
        "event".to_string(),
        "sex".to_string(),
        "prs_z".to_string(),
    ];
    for i in 0..N_PCS {
        headers.push(format!("PC{}", i + 1));
    }

    let mut state = 0xA0B10B_u64;
    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);

    let mut pc1 = Vec::with_capacity(N);
    let mut pc2 = Vec::with_capacity(N);
    let mut pc3 = Vec::with_capacity(N);
    let mut sex = Vec::with_capacity(N);
    let mut prs_z = Vec::with_capacity(N);
    let mut entry_age = Vec::with_capacity(N);
    let mut exit_age = Vec::with_capacity(N);

    for _ in 0..N {
        pc1.push(clip(
            0.024_352_2 + 0.042 * next_gauss(&mut state),
            -0.339_466,
            0.118_391,
        ));
        pc2.push(clip(
            0.080_796_3 + 0.030 * next_gauss(&mut state),
            -0.186_896,
            0.144_561,
        ));
        pc3.push(clip(
            -0.008_917_45 + 0.036 * next_gauss(&mut state),
            -0.313_383,
            0.056_200_3,
        ));
        sex.push(if next_unit(&mut state) < 0.391_184 {
            1.0_f64
        } else {
            0.0_f64
        });
        prs_z.push(next_gauss(&mut state));
        let entry = clip(45.082_7 + 18.0 * next_gauss(&mut state), 1.546_89, 121.963);
        let followup = next_gamma_alpha_ge_one(&mut state, 1.7, 4.4) + 0.05;
        entry_age.push(entry);
        exit_age.push((entry + followup).min(122.47).max(entry + 1.0e-3));
    }

    let prs_mean = prs_z.iter().copied().sum::<f64>() / N as f64;
    let prs_var = prs_z
        .iter()
        .map(|x| {
            let d = *x - prs_mean;
            d * d
        })
        .sum::<f64>()
        / N as f64;
    let prs_sd = prs_var.sqrt();
    for z in &mut prs_z {
        *z = (*z - prs_mean) / prs_sd;
    }

    let pc1_mean = pc1.iter().copied().sum::<f64>() / N as f64;
    let pc2_mean = pc2.iter().copied().sum::<f64>() / N as f64;
    let pc3_mean = pc3.iter().copied().sum::<f64>() / N as f64;
    let mut event_score = Vec::with_capacity(N);
    for i in 0..N {
        event_score.push(
            0.34 * prs_z[i] + 0.15 * sex[i] + 2.4 * (pc1[i] - pc1_mean) - 1.6 * (pc2[i] - pc2_mean)
                + 1.9 * (pc3[i] - pc3_mean)
                + next_gauss(&mut state),
        );
    }
    let mut sorted_score = event_score.clone();
    sorted_score.sort_by(|a, b| a.total_cmp(b));
    let event_median = sorted_score[N / 2];

    for i in 0..N {
        let mut record: Vec<String> = vec![
            entry_age[i].to_string(),
            exit_age[i].to_string(),
            if event_score[i] >= event_median {
                "1"
            } else {
                "0"
            }
            .to_string(),
            sex[i].to_string(),
            prs_z[i].to_string(),
        ];
        record.push(pc1[i].to_string());
        record.push(pc2[i].to_string());
        record.push(pc3[i].to_string());
        rows.push(StringRecord::from(record));
    }

    encode_recordswith_inferred_schema(headers, rows)
        .expect("encode synthetic survival marginal-slope dataset")
}

/// Reproduce the PIRLS joint-Newton residual-stall early-exit observed
/// at large scale.  This test is currently expected to FAIL — assert
/// `outer_converged == true`; the production code returns false because
/// the inner joint-Newton stalls on the time block.
#[test]
fn survival_marginal_slope_stall_reproduces_residual_stall_early_exit() {
    init();

    let data = build_dataset();

    // Build the PC-Duchon log-slope formula with the same PC dimensionality
    // and center count as the failing large-scale fit:
    // `duchon(PC1, PC2, PC3, centers=10, order=1)` on both sides.
    let pcs: Vec<String> = (0..N_PCS).map(|i| format!("PC{}", i + 1)).collect();
    let duchon_term = format!("duchon({}, centers=10, order=1)", pcs.join(", "));
    let formula = format!("Surv(entry_age, exit_age, event) ~ {} + sex", duchon_term);

    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("prs_z".to_string()),
        logslope_formula: Some(duchon_term),
        baseline_target: "linear".to_string(),
        gpu_policy: if cfg!(target_os = "macos") {
            gam::gpu::GpuPolicy::Off
        } else {
            gam::gpu::GpuPolicy::Auto
        },
        ..FitConfig::default()
    };

    eprintln!(
        "[SURVIVAL-MGS-STALL] starting exact startup-failure repro: n={} formula={:?}",
        N, formula
    );

    let start = Instant::now();
    let outcome = fit_from_formula(&formula, &data, &config);
    let elapsed = start.elapsed();
    eprintln!(
        "[SURVIVAL-MGS-STALL] fit_from_formula returned in {:.3}s ok={}",
        elapsed.as_secs_f64(),
        outcome.is_ok()
    );

    let err = match outcome {
        Ok(_) => panic!("expected exact startup validation failure"),
        Err(err) => err,
    };
    let message = err.to_string();
    assert!(
        message.contains("outer smoothing optimization failed after exhausting strategy fallbacks"),
        "missing outer fallback exhaustion error: {message}"
    );
    assert!(
        message.contains("no candidate seeds passed outer startup validation (custom family)"),
        "missing seed validation error: {message}"
    );
    assert!(
        message.contains(
            "coupled exact-joint inner solve exited the joint Newton path before convergence"
        ),
        "missing coupled exact-joint convergence error: {message}"
    );
    panic!("replicated exact production startup-validation error: {message}");
}
