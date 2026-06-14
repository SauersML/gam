//! #1122 / #901 iso-κ gradient class — the FULL joint-REML objective gradient
//! w.r.t. the Matérn isotropic length-scale coordinate `ψ = log κ` must match a
//! central finite difference of the COMPLETE outer criterion, not just the
//! penalty block.
//!
//! The penalty-only basis test
//! (`basis_matern_double_penalty_log_kappa_derivative_fd`) checks only
//! `∂(penalty matrices)/∂log_κ`. The full profiled REML objective
//!   V(κ) = data-fit(deviance) + ½log|H+Sλ| − ½log|Sλ|₊ + penalty-quad
//! also depends on κ through the Matérn DESIGN columns (and hence the deviance
//! and `H = XᵀWX`) and through the H-side of the `½log|H+Sλ|` term. A stall with
//! a large residual gradient at the iteration cap is the signature of an
//! objective↔gradient DESYNC: the optimizer follows a gradient inconsistent with
//! the objective it is minimizing.
//!
//! The root cause was the double-penalty nullspace-shrinkage decision (and the
//! identifiability transform `Z`) NOT being frozen across the κ-optimizer's
//! per-trial value rebuilds: the κ-DEPENDENT spectral test
//! (`build_nullspace_shrinkage_penalty`, tolerance ∝ λ_max(A(κ))) flips the
//! shrinkage block `P/√r` discontinuously as κ moves, so V(κ) is
//! piecewise-discontinuous while the analytic gradient (assembled in a fixed
//! frozen eigenbasis) is smooth. The fix freezes BOTH `Z` and the
//! shrinkage decision into a `FrozenTransform` at the first per-trial rebuild,
//! and mirrors that freeze onto the collection spec the analytic ψ-gradient
//! reads, so value and gradient share one fixed `Z` and one fixed null
//! dimension `r` at every trial.
//!
//! The outer driver (`spatial-exact-joint`) runs
//! `solver::outer_strategy::outer_gradient_fd_audit` automatically at θ₀,
//! central-differencing the *outer criterion* component-by-component and
//! comparing it to the analytic outer gradient. The Matérn log-κ coordinate is
//! labelled `psi_kappa[..]`. This test fits an ordinary Gaussian 2-D surface
//! with a single `matern(x1, x2)` smooth (default double-penalty), captures the
//! audit, and asserts the analytic gradient w.r.t. log-κ matches the central FD
//! of the full criterion (no DESYNC verdict, finite per-coordinate analytic/fd,
//! small relative gap on the κ block).
//!
//! Reference-as-truth: every assertion is against the analytic FD of gam's own
//! profiled REML criterion — never another tool's output.

use gam::{FitConfig, encode_recordswith_inferred_schema};
use std::sync::{Mutex, Once};

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
        }
    }
    fn flush(&self) {}
}

static LOGGER: CapturingLogger = CapturingLogger;
static INIT_LOGGER: Once = Once::new();

fn init() {
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);
    gam::init_parallelism();
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

fn truth(a: f64, b: f64) -> f64 {
    (2.0 * std::f64::consts::PI * a).sin() * (2.0 * std::f64::consts::PI * b).sin()
}

fn build_dataset(n: usize, sigma: f64, seed: u64) -> gam::inference::data::EncodedDataset {
    let mut st = seed;
    let mut header = String::from("y,x1,x2\n");
    let mut body = String::new();
    for _ in 0..n {
        let a = next_unit(&mut st);
        let b = next_unit(&mut st);
        let y = truth(a, b) + sigma * next_gauss(&mut st);
        body.push_str(&format!("{y:.6},{a:.6},{b:.6}\n"));
    }
    header.push_str(&body);
    let mut rdr = csv::ReaderBuilder::new().from_reader(header.as_bytes());
    let records: Vec<csv::StringRecord> = rdr.records().map(|r| r.unwrap()).collect();
    let headers = vec!["y".to_string(), "x1".to_string(), "x2".to_string()];
    encode_recordswith_inferred_schema(headers, records).expect("encode dataset")
}

fn field(line: &str, key: &str) -> String {
    if let Some(pos) = line.find(key) {
        let rest = &line[pos + key.len()..];
        rest.split_whitespace().next().unwrap_or("").to_string()
    } else {
        String::new()
    }
}

fn parse_components(lines: &[String]) -> Vec<(String, f64, f64, f64)> {
    let mut out = Vec::new();
    for l in lines {
        if !l.contains(" analytic=") || !l.contains(" fd=") || !l.contains(" gap=") {
            continue;
        }
        let block = field(l, "block=");
        let a: f64 = field(l, "analytic=").parse().unwrap_or(f64::NAN);
        let fd: f64 = field(l, "fd=").parse().unwrap_or(f64::NAN);
        let gap: f64 = field(l, "gap=").parse().unwrap_or(f64::NAN);
        if !block.is_empty() {
            out.push((block, a, fd, gap));
        }
    }
    out
}

/// MERGE GATE (#1122 / #901): the analytic outer REML gradient w.r.t. the
/// Matérn log-κ coordinate matches a central finite difference of the FULL
/// profiled REML criterion at θ₀ — data-fit + logdet (both Sλ-side AND H-side) +
/// penalty-quad — with the default double-penalty (nullspace shrinkage) active.
#[test]
fn matern_2d_iso_kappa_outer_gradient_matches_fd() {
    init();
    if let Ok(mut g) = CAPTURE.lock() {
        g.clear();
    }
    let data = build_dataset(150, 0.05, 0x9A7E_7212_0001u64);
    let formula = "y ~ matern(x1, x2)";
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        // The outer FD audit fires at θ₀ during the fit, before the (capped)
        // outer loop; cap the loop so the test stays fast — the gate we assert
        // is the captured audit, independent of the fit's eventual outcome.
        outer_max_iter: Some(2),
        gpu_policy: if cfg!(target_os = "macos") {
            gam::gpu::GpuPolicy::Off
        } else {
            gam::gpu::GpuPolicy::Auto
        },
        ..FitConfig::default()
    };
    match gam::fit_from_formula(formula, &data, &config) {
        Ok(_) => eprintln!("[FD-DIAG] matern(x1,x2) fit returned Ok"),
        Err(e) => eprintln!("[FD-DIAG] matern(x1,x2) fit returned Err (audit still ran): {e}"),
    }

    let lines = CAPTURE.lock().unwrap().clone();
    let gate: Vec<&String> = lines
        .iter()
        .filter(|l| l.contains("gate eligible="))
        .collect();
    assert!(
        !gate.is_empty(),
        "expected an [OUTER-FD-AUDIT] gate line; captured {} lines: {:#?}",
        lines.len(),
        lines
    );
    let psi_dim: usize = gate
        .iter()
        .filter_map(|l| field(l, "psi_dim=").parse::<usize>().ok())
        .max()
        .unwrap_or(0);
    assert!(
        psi_dim >= 1,
        "matern(x1,x2) must enroll log-κ as a ψ-coordinate (psi_dim≥1); gate lines: {gate:#?}"
    );

    let comps = parse_components(&lines);
    assert!(
        !comps.is_empty(),
        "expected per-coordinate analytic-vs-FD lines; captured: {:#?}",
        lines
    );

    // No DESYNC verdict: the FULL-objective analytic gradient agrees with FD.
    let desync: Vec<&String> = lines
        .iter()
        .filter(|l| l.contains("VERDICT=DESYNC"))
        .collect();
    assert!(
        desync.is_empty(),
        "Matérn iso-κ outer gradient DESYNCs from the FD of the full criterion: {desync:#?}"
    );

    // The log-κ block(s) specifically: finite and analytic≈fd to tolerance.
    let kappa_comps: Vec<&(String, f64, f64, f64)> = comps
        .iter()
        .filter(|(block, ..)| block.starts_with("psi_kappa"))
        .collect();
    assert!(
        !kappa_comps.is_empty(),
        "expected a psi_kappa[..] audit component; got blocks: {:?}",
        comps.iter().map(|c| &c.0).collect::<Vec<_>>()
    );
    for (block, a, fd, gap) in &kappa_comps {
        assert!(
            a.is_finite() && fd.is_finite(),
            "non-finite Matérn log-κ gradient component {block}: analytic={a} fd={fd}"
        );
        let scale = a.abs().max(fd.abs()).max(1e-6);
        assert!(
            gap / scale < 5e-2,
            "Matérn iso-κ outer-gradient analytic≠FD on {block}: analytic={a:.6e} \
             fd={fd:.6e} gap={gap:.3e} rel={:.3e}",
            gap / scale
        );
    }
}
