//! Discriminating diagnostic for issue #1040: does the survival marginal-slope
//! outer REML/LAML loop fail to converge because of an objective↔gradient
//! DESYNC (a bug — analytic gradient disagrees with a finite-difference of the
//! criterion, so the trust region chases a phantom descent direction forever),
//! or because of weak IDENTIFIABILITY (a genuinely flat valley — analytic ≈ FD
//! everywhere but a near-zero outer-Hessian eigenvalue)?
//!
//! The fork is the central-difference of the outer gradient at a fixed θ,
//! component by component. The audit machinery lives in
//! `outer_strategy::outer_gradient_fd_audit` and is invoked automatically by
//! `optimize_spatial_length_scale_exact_joint` on diagnostic-sized problems
//! (this one: n=800, centers=4). It emits `[OUTER-FD-AUDIT/...]` log lines
//! including per-block analytic gradient norms, per-coordinate analytic-vs-FD
//! gaps, the outer-Hessian eigenvalue spectrum, and a single VERDICT line. This
//! test drives a small survival-MS fit, captures those lines, and asserts the
//! verdict is emitted and self-consistent.
//!
//! The audit runs at θ₀ BEFORE the (potentially non-terminating) outer loop, so
//! `outer_max_iter` is capped low to keep the test cheap regardless of whether
//! the underlying loop would converge.

use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
use std::sync::{Mutex, Once};

const N: usize = 800;

static CAPTURE: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct CapturingLogger;
impl log::Log for CapturingLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record<'_>) {
        let line = format!("{}", record.args());
        if line.contains("OUTER-FD-AUDIT") {
            if let Ok(mut g) = CAPTURE.lock() {
                g.push(line.clone());
            }
            eprintln!("{line}");
        } else if line.contains("survival-marginal-slope/outer")
            || line.contains("[joint-newton-tr]")
            || line.contains("KAPPA-PHASE")
            || line.contains("startup")
        {
            eprintln!("[trace] {line}");
        }
    }
    fn flush(&self) {}
}

static LOGGER: CapturingLogger = CapturingLogger;
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

#[inline]
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
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}
fn next_gamma_alpha_ge_one(state: &mut u64, alpha: f64, scale: f64) -> f64 {
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = next_gauss(state);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = next_unit(state).max(1.0e-12);
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return scale * d * v;
        }
    }
}
fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

fn build_dataset() -> gam::inference::data::EncodedDataset {
    let headers = vec![
        "entry_age".to_string(),
        "exit_age".to_string(),
        "event".to_string(),
        "sex".to_string(),
        "prs_z".to_string(),
        "PC1".to_string(),
        "PC2".to_string(),
    ];
    let mut state = 0xA0B10B_u64;
    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);

    let mut pc1 = Vec::with_capacity(N);
    let mut pc2 = Vec::with_capacity(N);
    let mut sex = Vec::with_capacity(N);
    let mut prs_z = Vec::with_capacity(N);
    let mut entry_age = Vec::with_capacity(N);
    let mut exit_age = Vec::with_capacity(N);

    for _ in 0..N {
        pc1.push(clip(0.024 + 0.042 * next_gauss(&mut state), -0.34, 0.12));
        pc2.push(clip(0.081 + 0.030 * next_gauss(&mut state), -0.19, 0.15));
        sex.push(if next_unit(&mut state) < 0.39 {
            1.0
        } else {
            0.0
        });
        prs_z.push(next_gauss(&mut state));
        let entry = clip(45.08 + 18.0 * next_gauss(&mut state), 1.55, 121.96);
        let followup = next_gamma_alpha_ge_one(&mut state, 1.7, 4.4) + 0.05;
        entry_age.push(entry);
        exit_age.push((entry + followup).min(122.47).max(entry + 1.0e-3));
    }

    let prs_mean = prs_z.iter().copied().sum::<f64>() / N as f64;
    let prs_var = prs_z.iter().map(|x| (x - prs_mean).powi(2)).sum::<f64>() / N as f64;
    let prs_sd = prs_var.sqrt().max(1.0e-9);
    for z in &mut prs_z {
        *z = (*z - prs_mean) / prs_sd;
    }
    let pc1_mean = pc1.iter().copied().sum::<f64>() / N as f64;
    let pc2_mean = pc2.iter().copied().sum::<f64>() / N as f64;

    let mut event_score = Vec::with_capacity(N);
    for i in 0..N {
        event_score.push(
            0.34 * prs_z[i] + 0.15 * sex[i] + 2.4 * (pc1[i] - pc1_mean) - 1.6 * (pc2[i] - pc2_mean)
                + next_gauss(&mut state),
        );
    }
    let mut sorted = event_score.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = sorted[N / 2];

    for i in 0..N {
        let record: Vec<String> = vec![
            entry_age[i].to_string(),
            exit_age[i].to_string(),
            if event_score[i] >= median { "1" } else { "0" }.to_string(),
            sex[i].to_string(),
            prs_z[i].to_string(),
            pc1[i].to_string(),
            pc2[i].to_string(),
        ];
        rows.push(StringRecord::from(record));
    }
    encode_recordswith_inferred_schema(headers, rows)
        .expect("encode small survival marginal-slope dataset")
}

fn field(line: &str, key: &str) -> String {
    match line.find(key) {
        Some(p) => {
            let rest = &line[p + key.len()..];
            rest.split_whitespace().next().unwrap_or("").to_string()
        }
        None => String::new(),
    }
}

fn parse_components(lines: &[String]) -> Vec<(String, usize, f64, f64, f64)> {
    let mut out = Vec::new();
    for l in lines {
        if !l.contains(" analytic=") || !l.contains(" fd=") || !l.contains(" gap=") {
            continue;
        }
        let block = field(l, "block=");
        let i: usize = field(l, "i=").parse().unwrap_or(usize::MAX);
        let a: f64 = field(l, "analytic=").parse().unwrap_or(f64::NAN);
        let fd: f64 = field(l, "fd=").parse().unwrap_or(f64::NAN);
        let gap: f64 = field(l, "gap=").parse().unwrap_or(f64::NAN);
        if i != usize::MAX {
            out.push((block, i, a, fd, gap));
        }
    }
    out
}

fn run_basis(basis_term: &str) {
    init();
    if let Ok(mut g) = CAPTURE.lock() {
        g.clear();
    }
    let data = build_dataset();
    let formula = format!("Surv(entry_age, exit_age, event) ~ {basis_term} + sex");
    let config = FitConfig {
        survival_likelihood: "marginal-slope".to_string(),
        z_column: Some("prs_z".to_string()),
        logslope_formula: Some(basis_term.to_string()),
        baseline_target: "linear".to_string(),
        outer_max_iter: Some(2),
        gpu_policy: if cfg!(target_os = "macos") {
            gam::gpu::GpuPolicy::Off
        } else {
            gam::gpu::GpuPolicy::Auto
        },
        ..FitConfig::default()
    };

    eprintln!("[FD-DIAG] starting survival-MS fit: n={N} formula={formula:?}");
    let _ = fit_from_formula(&formula, &data, &config);

    let lines = CAPTURE.lock().unwrap().clone();
    let verdict: Vec<&String> = lines.iter().filter(|l| l.contains("VERDICT=")).collect();
    assert!(
        !verdict.is_empty(),
        "expected an [OUTER-FD-AUDIT] VERDICT line for basis {basis_term:?}; captured {} audit lines: {:#?}",
        lines.len(),
        lines
    );

    let comps = parse_components(&lines);
    assert!(
        !comps.is_empty(),
        "expected per-coordinate analytic-vs-FD lines for basis {basis_term:?}; captured: {:#?}",
        lines
    );

    let worst = comps
        .iter()
        .max_by(|a, b| a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal))
        .cloned()
        .unwrap();
    eprintln!(
        "[FD-DIAG] basis={basis_term} worst component: block={} i={} analytic={:.6e} fd={:.6e} gap={:.3e}",
        worst.0, worst.1, worst.2, worst.3, worst.4
    );
    for v in &verdict {
        eprintln!("[FD-DIAG] {v}");
    }

    for (block, i, a, fd, _gap) in &comps {
        assert!(
            a.is_finite() && fd.is_finite(),
            "non-finite gradient component for basis {basis_term:?}: block={block} i={i} analytic={a} fd={fd}"
        );
    }
}

#[test]
fn survival_marginal_slope_outer_gradient_fd_audit_matern() {
    run_basis("matern(PC1, PC2, centers=4)");
}

#[test]
fn survival_marginal_slope_outer_gradient_fd_audit_duchon() {
    run_basis("duchon(PC1, PC2, centers=4, order=1)");
}
