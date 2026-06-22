//! Diagnostic for #1426 (shared with #1373): surface the Gamma/log REML
//! determinant-pair internals on the seeds that ship the near-full-basis
//! overfit (EDF≈24 vs mgcv EDF≈8), to attribute the bad Occam pair to the
//! correct sub-term.
//!
//! Background (hand-derived; see the #1426 investigation). The Gamma/log LAML
//! criterion minimized over ρ=log λ is
//!   V(ρ) = −ℓ(β̂) + ½ β̂ᵀSβ̂ + ½ log|H| − ½ log|S|₊
//! with φ=1/k̂ frozen per inner solve. An asymptotic analysis showed the overfit
//! guard is the Occam determinant pair `½ log|H| − ½ log|S|₊` (the data-fidelity
//! term self-cancels under ML shape estimation). The acn116 basin run found the
//! TOTAL cost is genuinely lower at the overfit (cost≈524 at EDF≈24 vs ≈835 at
//! EDF≈8): the killer is the effective pair, ≈ −288 at the overfit vs +21 at the
//! healthy fit. But the public-API basin probe could only recover `log|S|₊`
//! CONTAMINATED by `hessian_logdet_correction` (the #901 intrinsic pseudo-logdet
//! folded into `log|H|`), so it could not say WHICH sub-term is pathological.
//!
//! This diagnostic resolves that by capturing the existing `[#1271-diag]` REML
//! evaluation trace already emitted (at `log::Level::Info`) by
//! `src/solver/reml/objective.rs` on every dense evaluation. That line reports,
//! per visited ρ, the GENUINE internal scalars:
//!   - `logS`  = `penalty_logdet.value` = the true `log|S|₊` (NO correction),
//!   - `logH`  = `hessian_op.logdet()`   = `log|H|`  (NO correction),
//!   - `penalty_rank`, `nullspace_dim`, the H eigenvalue spread, and `lambda`.
//! With those, the decision is unambiguous:
//!   - If `logS` does NOT grow with λ (or `penalty_rank` shrinks as λ→small),
//!     the penalty pseudo-logdet `½log|λS|₊` is mis-computed (the −½log|S|₊
//!     term fails to forbid the small-λ overfit) → penalty-logdet bug.
//!   - If `logS` grows correctly with λ but the pair is still inverted, the
//!     `½log|H|` / `hessian_logdet_correction` side is the culprit.
//!
//! Capturing-logger pattern mirrors `tests/quality/diag_1271_logdet_probe.rs`
//! (no env access). Diagnostic only — asserts false at the end to surface the
//! captured trace under nextest; never a quality gate.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
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

// ───────────────────────── data generators (bit-identical to the failing tests)

/// Deterministic LCG used by `tests/issue_1426_gammalog_recovery.rs`
/// (`LCG_SEED = 7`). Gamma(shape=2, scale) = −scale·(ln u1 + ln u2).
fn build_data_lcg(seed: u64, n: usize) -> gam::data::EncodedDataset {
    let mut state: u64 = seed;
    let mut nxt = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((state >> 11) as f64) / ((1u64 << 53) as f64);
        u.clamp(1e-12, 1.0 - 1e-12)
    };
    let true_mu = |x: f64| (0.6 * (2.0 * std::f64::consts::PI * x).sin() + 0.6).exp();
    let mut x: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        x.push(nxt());
    }
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for &xi in &x {
        let scale = true_mu(xi) / 2.0;
        let u1 = nxt();
        let u2 = nxt();
        y.push(-scale * (u1.ln() + u2.ln()));
    }
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![x[i].to_string(), y[i].to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode lcg gammalog dataset")
}

/// Deterministic SplitMix64 used by
/// `tests/bug_hunt_1426_gamma_log_reml_flat_valley_overfit.rs` (seeds 900006 /
/// 900000). Gamma(shape=2, scale) = scale·(Exp1 + Exp1), Exp1 = −ln(1−u).
fn build_data_splitmix(seed: u64, n: usize) -> gam::data::EncodedDataset {
    let mut state = seed;
    let mut next_unit = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let true_mu = |x: f64| (0.6 * (2.0 * std::f64::consts::PI * x).sin() + 0.6).exp();
    let mut x: Vec<f64> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = next_unit();
        let scale = true_mu(xi) / 2.0;
        let e1 = -(1.0 - next_unit()).ln();
        let e2 = -(1.0 - next_unit()).ln();
        y.push(scale * (e1 + e2));
        x.push(xi);
    }
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![x[i].to_string(), y[i].to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode splitmix gammalog dataset")
}

/// log|H| via a plain Cholesky (H = L Lᵀ ⇒ log|H| = 2 Σ log L_ii). NaN when H is
/// not numerically PD; NaN flows through the printed components so it is visible.
fn chol_logdet(h: &ndarray::Array2<f64>) -> f64 {
    let n = h.nrows();
    if n == 0 || h.ncols() != n {
        return f64::NAN;
    }
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = h[[i, j]];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return f64::NAN;
                }
                l[i * n + j] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut ld = 0.0;
    for i in 0..n {
        ld += 2.0 * l[i * n + i].ln();
    }
    ld
}

/// Fit `y ~ s(x)` Gamma/log on `data`, print the public summary plus the FIRST /
/// MID / LAST captured `[#1271-diag]` trace lines (the LAST is the converged ρ
/// the fit ships). Returns the total EDF for the closing summary line.
fn dump_components(tag: &str, data: &gam::data::EncodedDataset) -> f64 {
    if let Ok(mut g) = DIAG_LINES.lock() {
        g.clear();
    }
    let cfg = FitConfig {
        family: Some("gamma".to_string()),
        link: Some("log".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x)", data, &cfg).expect("gamma/log gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit"); // SAFETY: y~s(x) gamma/log is always a standard GAM; any other variant is an engine contract break worth surfacing.
    };
    let u = &fit.fit;

    let edf = u.edf_total().unwrap_or(f64::NAN);
    let phi = u.dispersion_phi(); // Gamma: φ = 1/k̂
    let k_hat = if phi > 0.0 { 1.0 / phi } else { f64::NAN };
    let neg_ll = -u.log_likelihood;
    let pen_quad = 0.5 * u.stable_penalty_term;
    let half_logdet_h_geom = 0.5
        * u.geometry
            .as_ref()
            .map(|g| chol_logdet(g.penalized_hessian.as_array()))
            .unwrap_or(f64::NAN);
    let reml = u.reml_score;
    let gnorm = u.outer_gradient_norm.unwrap_or(f64::NAN);
    let converged = if u.outer_converged { 1.0 } else { 0.0 };
    let lam_min = u.lambdas.iter().cloned().fold(f64::INFINITY, f64::min);
    let lam_max = u.lambdas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    eprintln!(
        "[#1426 basin] tag={tag} edf={edf:.4} k_hat={k_hat:.5} phi={phi:.6} \
         neg_ll={neg_ll:.5} pen_quad={pen_quad:.5} half_logdetH_geom={half_logdet_h_geom:.5} \
         cost_reml={reml:.5} outer_converged={converged:.0} outer_gnorm={gnorm:.6e} \
         n_lambda={} lambda_min={lam_min:.4e} lambda_max={lam_max:.4e}",
        u.lambdas.len(),
    );

    // The `[#1271-diag]` trace carries the GENUINE log|S|₊ / log|H| split (no
    // #901 correction folded in) at every visited ρ. Print first/mid/last so the
    // logS-vs-λ trend the cost actually used is visible; LAST is the shipped ρ.
    if let Ok(g) = DIAG_LINES.lock() {
        let n = g.len();
        eprintln!("[#1426 basin] tag={tag} captured_1271diag_lines={n}");
        if n > 0 {
            eprintln!("[#1426 basin] tag={tag} FIRST {}", g[0]);
            if n > 2 {
                eprintln!("[#1426 basin] tag={tag} MID {}", g[n / 2]);
            }
            eprintln!("[#1426 basin] tag={tag} LAST {}", g[n - 1]);
        }
    }

    edf
}

#[test]
fn diag_1426_gammalog_basin_dump() {
    init_parallelism();
    // Install the capturing logger so the `[#1271-diag]` Info trace from the REML
    // evaluator lands in `DIAG_LINES` (no env access). Ignore an Err return: a
    // global logger may already be installed by another test in the same binary.
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }

    let edf_lcg7 = dump_components("lcg_seed7_OVERFIT", &build_data_lcg(7, 1500));
    let edf_sm900006 = dump_components("splitmix_900006_OVERFIT", &build_data_splitmix(900006, 1500));
    let edf_sm900000 = dump_components("splitmix_900000_HEALTHY", &build_data_splitmix(900000, 1500));

    eprintln!(
        "[#1426 basin] SUMMARY edf_lcg7={edf_lcg7:.3} edf_splitmix900006={edf_sm900006:.3} \
         edf_splitmix900000_healthy={edf_sm900000:.3} (mgcv recovers EDF≈8 on this DGP). \
         READ the [#1271-diag] LAST lines: logS must GROW with lambda; if it does \
         not (or penalty_rank shrinks as lambda→small), the penalty pseudo-logdet \
         is the bug; otherwise the log|H|/correction side is."
    );

    // Intentional failure so nextest surfaces the eprintln trace above. This is a
    // DIAGNOSTIC dump, never a quality gate.
    assert!(
        false,
        "diag_1426_gammalog_basin: diagnostic dump only — read the [#1426 basin] and \
         [#1271-diag] lines above to attribute the inverted Occam pair to the penalty \
         pseudo-logdet (logS not growing with lambda) vs the log|H|/correction side."
    );
}
