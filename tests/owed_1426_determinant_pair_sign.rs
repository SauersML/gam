//! Regression for #1426: the Gamma/log LAML determinant pair must have the
//! correct OCCAM SIGN — it must be LARGER at small λ (the wiggly/overfit corner)
//! than at large λ (the well-penalized corner).
//!
//! The criterion minimized over ρ=log λ is
//!   V(ρ) = −ℓ(β̂) + ½ β̂ᵀSβ̂ + ½ log|H| − ½ log|S|₊,   H = X'WX + S(λ).
//! The determinant pair D(λ) = ½(log|H| − log|S|₊) is the Bayesian Occam
//! complexity penalty: on each penalized direction it equals
//!   ½ log(1 + γ_j/(λ s_j)),
//! which is strictly DECREASING in λ — i.e. D is MAXIMAL at small λ (more
//! effective parameters ⇒ bigger complexity penalty) and → 0 as λ → ∞. A
//! correct criterion therefore charges the overfit MORE through D, so the
//! over-fit does not win. The #1426 bug ships the pair INVERTED (hugely negative
//! at the overfit), which makes the criterion prefer the near-full-basis fit
//! (cost ≈ 524 at EDF≈24 vs ≈ 835 at EDF≈8).
//!
//! This test pins the sign directly. It captures the `[#1271-diag]` REML
//! evaluation trace already emitted (at `log::Level::Info`) by
//! `src/solver/reml/objective.rs` on every dense evaluation — each line carries
//! the GENUINE internal scalars `logS` (= `penalty_logdet.value`), `logH`
//! (= `hessian_op.logdet()`), `half_diff` (= ½(logH − logS), the pair), and the
//! per-penalty `lambda=[…]`. Across the ρ-points the outer optimizer visits on
//! the Gamma/log fit, the pair at the SMALLEST visited λ must exceed the pair at
//! the LARGEST visited λ, and (the decisive penalty-side check) `logS` must GROW
//! with λ (`log|λS|₊ = rank·log λ + const` is monotone increasing in λ).
//!
//! If this gate is RED, the determinant pair is still inverted — which (given
//! the rank-guarded H-side fix in `intrinsic_hessian_pseudo_logdet_parts`)
//! localizes the remaining defect to the PENALTY pseudo-logdet side. The
//! capturing-logger pattern mirrors `tests/quality/diag_1271_logdet_probe.rs`
//! (no env access).

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

/// Deterministic LCG (the `tests/issue_1426_gammalog_recovery.rs` `LCG_SEED = 7`
/// generator): Gamma(shape=2, scale) = −scale·(ln u1 + ln u2), μ(x)=exp(0.6 sin
/// 2πx + 0.6). Reproduces the EDF≈24 overfit ship on this fixture.
fn build_data(seed: u64, n: usize) -> gam::data::EncodedDataset {
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
    encode_recordswith_inferred_schema(headers, rows).expect("encode gammalog dataset")
}

/// Parse a `key=<f64>` token out of a `[#1271-diag]` line (e.g. `logS=-3.21`).
fn parse_scalar(line: &str, key: &str) -> Option<f64> {
    let needle = format!("{key}=");
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let end = rest
        .find(|c: char| c == ' ' || c == '[')
        .unwrap_or(rest.len());
    rest[..end].trim().parse::<f64>().ok()
}

/// Parse the maximum λ out of the `lambda=[a,b,…]` list on a `[#1271-diag]` line.
fn parse_lambda_max(line: &str) -> Option<f64> {
    let start = line.find("lambda=[")? + "lambda=[".len();
    let rest = &line[start..];
    let end = rest.find(']')?;
    rest[..end]
        .split(',')
        .filter_map(|t| t.trim().parse::<f64>().ok())
        .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v))))
}

#[test]
fn gammalog_determinant_pair_has_correct_occam_sign() {
    init_parallelism();
    // Install the capturing logger so the `[#1271-diag]` Info trace lands in
    // `DIAG_LINES`. Ignore Err: a global logger may already be installed by
    // another test sharing this binary.
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
    if let Ok(mut g) = DIAG_LINES.lock() {
        g.clear();
    }

    let data = build_data(7, 1500);
    let cfg = FitConfig {
        family: Some("gamma".to_string()),
        link: Some("log".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x)", &data, &cfg).expect("gamma/log gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit"); // SAFETY: y~s(x) gamma/log is always a standard GAM; any other variant is an engine contract break worth surfacing.
    };
    let edf = fit.fit.edf_total().expect("gam reports total edf");

    // Collect (lambda_max, logS, half_diff) from every captured evaluation.
    let mut points: Vec<(f64, f64, f64)> = Vec::new();
    if let Ok(g) = DIAG_LINES.lock() {
        for line in g.iter() {
            if let (Some(lam), Some(log_s), Some(half_diff)) = (
                parse_lambda_max(line),
                parse_scalar(line, "logS"),
                parse_scalar(line, "half_diff"),
            ) {
                if lam.is_finite() && lam > 0.0 && log_s.is_finite() && half_diff.is_finite() {
                    points.push((lam, log_s, half_diff));
                }
            }
        }
    }

    assert!(
        points.len() >= 2,
        "captured only {} usable [#1271-diag] evaluation lines for the gamma/log fit; need >= 2 \
         distinct λ points to test the determinant-pair sign (logger may not be installed, or the \
         REML path changed its trace format)",
        points.len()
    );

    // Smallest-λ vs largest-λ visited evaluation.
    let lo = points
        .iter()
        .copied()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .expect("non-empty");
    let hi = points
        .iter()
        .copied()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .expect("non-empty");
    let (lam_lo, log_s_lo, pair_lo) = lo;
    let (lam_hi, log_s_hi, pair_hi) = hi;

    eprintln!(
        "[#1426 sign] edf={edf:.3} n_points={} \
         lo: lambda={lam_lo:.4e} logS={log_s_lo:.4} pair={pair_lo:.4} | \
         hi: lambda={lam_hi:.4e} logS={log_s_hi:.4} pair={pair_hi:.4} | \
         d_pair(lo-hi)={:.4} d_logS(hi-lo)={:.4}",
        points.len(),
        pair_lo - pair_hi,
        log_s_hi - log_s_lo,
    );

    // (1) PENALTY-SIDE MONOTONICITY: log|λS|₊ = rank·log λ + const is strictly
    //     increasing in λ. With lam_hi ≫ lam_lo, logS at the large-λ point must
    //     exceed logS at the small-λ point. An INVERSION here is the #1426
    //     penalty-pseudo-logdet defect (Derivation B).
    assert!(
        lam_hi > lam_lo * 1.5,
        "test fixture issue: visited λ range too narrow to test sign (lam_lo={lam_lo:.4e}, \
         lam_hi={lam_hi:.4e}); the outer optimizer did not explore distinct smoothing scales"
    );
    assert!(
        log_s_hi > log_s_lo,
        "Gamma/log penalty pseudo-logdet is INVERTED in λ (#1426): logS={log_s_hi:.4} at \
         λ={lam_hi:.4e} is NOT greater than logS={log_s_lo:.4} at λ={lam_lo:.4e}, yet \
         log|λS|₊ = rank·log λ + const must grow with λ. The −½log|S|₊ term then fails to \
         forbid the small-λ overfit — fix the PenaltyPseudologdet path."
    );

    // (2) OCCAM-PAIR SIGN: the determinant pair ½(log|H| − log|S|₊) is the
    //     complexity penalty; it must be LARGER at the small-λ (wiggly) corner
    //     than at the large-λ (smooth) corner. The #1426 bug ships it inverted
    //     (hugely negative at small λ), which rewards the overfit.
    assert!(
        pair_lo > pair_hi,
        "Gamma/log LAML determinant pair has the WRONG Occam sign (#1426): pair={pair_lo:.4} at \
         the small-λ overfit corner (λ={lam_lo:.4e}) is NOT greater than pair={pair_hi:.4} at the \
         large-λ smooth corner (λ={lam_hi:.4e}). The complexity penalty ½(log|H|−log|S|₊) must \
         charge the wiggly fit MORE, not less — an inverted pair makes the criterion prefer the \
         near-full-basis overfit."
    );
}
