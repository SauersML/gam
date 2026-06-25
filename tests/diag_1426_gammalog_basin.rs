//! Diagnostic for #1426: surface the Gamma/log REML cost decomposition on the
//! seeds that ship the near-full-basis overfit (EDF≈24 vs mgcv EDF≈8), so the
//! A/C-vs-B fork can be decided from real numbers.
//!
//! Background (verified by hand-derivation, see the #1426 investigation). The
//! Gamma/log LAML criterion minimized over ρ=log λ is
//!   V(ρ) = −ℓ(β̂) + ½ β̂ᵀSβ̂ + ½ log|H| − ½ log|S|₊
//! with the Gamma dispersion φ=1/k̂ frozen per inner solve (k̂ the ML shape).
//! An asymptotic analysis showed the *overfit guard* is the Occam determinant
//! pair `½ log|H| − ½ log|S|₊` (already present), NOT a missing saturated-
//! likelihood term (adding that term is wrong-signed). The data-fidelity term
//! `D/(2φ) = (k̂/2)·Σw·d` is self-cancelling under ML shape estimation
//! (`k̂·d̄ → ½`), so it is ≈ n_eff/4 at every fit. The remaining question this
//! diagnostic answers from real numbers:
//!
//!   - If the shipped fit reports `outer_converged = true` with a small outer
//!     gradient norm AND EDF≈24, the criterion's *stationary point* genuinely
//!     is the overfit → determinant-computation bug (A/C): the Occam pair is
//!     too weak / mis-ranked for Gamma/log.
//!   - If the shipped fit reports `outer_converged = false` (or a large outer
//!     gradient norm) yet still ships EDF≈24, the objective is fine but the
//!     optimizer never reached the well-penalized basin → optimizer/seeding
//!     acceptance bug (B).
//!
//! This test prints NAMED f64 components (no debug formatting) for the two
//! reported failing datasets and one healthy contrast seed, then fails via an
//! assertion so the captured output surfaces under nextest. It is a DIAGNOSTIC,
//! never a quality gate.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

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
/// 900000). Gamma(shape=2, scale) = scale·(Exp1 + Exp1).
fn build_data_splitmix(seed: u64, n: usize) -> gam::data::EncodedDataset {
    let mut state = seed;
    // Single mutable-state generator; no nested mutable-closure captures.
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
        // Gamma(shape=2, scale) = scale·(Exp1 + Exp1), Exp1 = −ln(1−u).
        let e1 = -(1.0 - next_unit()).ln();
        let e2 = -(1.0 - next_unit()).ln();
        let yi = scale * (e1 + e2);
        x.push(xi);
        y.push(yi);
    }
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![x[i].to_string(), y[i].to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode splitmix gammalog dataset")
}

/// log|H| via a plain Cholesky (H = L Lᵀ ⇒ log|H| = 2 Σ log L_ii). Returns NaN
/// if H is not numerically PD (shouldn't happen at a converged penalized mode);
/// NaN flows through the printed components so a non-PD Hessian is visible.
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

/// Fit `y ~ s(x)` Gamma/log on `data` and print the named REML cost components.
/// Returns the total EDF for the post-loop summary line.
fn dump_components(tag: &str, data: &gam::data::EncodedDataset) -> f64 {
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
    let neg_ll = -u.log_likelihood; // −ℓ(β̂)  (β-dependent scaled-deviance form)
    let pen_quad = 0.5 * u.stable_penalty_term; // ½ β̂ᵀSβ̂  (incl. solver ridge)
    let half_logdet_h = 0.5
        * u.geometry
            .as_ref()
            .map(|g| chol_logdet(g.penalized_hessian.as_array()))
            .unwrap_or(f64::NAN);
    let reml = u.reml_score; // total minimized criterion V(ρ̂)
    // V = −ℓ + ½βᵀSβ + ½log|H| − ½log|S|₊  ⇒  ½log|S|₊ = (−ℓ + ½βᵀSβ + ½log|H|) − V.
    // This recovers the (otherwise non-public) penalty pseudo-determinant from
    // the public components; it is exact only if the engine's V uses exactly
    // this term layout (the documented Fixed-dispersion LAML form). A large
    // residual here itself diagnoses an extra/dropped objective term.
    let half_logdet_s = (neg_ll + pen_quad + half_logdet_h) - reml;
    // D/(2φ): the data-fidelity term. The codebase folds k̂ into its deviance
    // (`deviance = Σ w·2k̂·(d/2)`), so `−ℓ = −½·deviance` already equals the
    // k̂-scaled fit term; D/(2φ) in mgcv's k̂-free convention is `(k̂/2)·Σw·d =
    // ½·deviance_codebase = −neg_ll`. Print the codebase deviance and its half
    // so the self-cancellation `k̂·d̄ → ½` is checkable.
    let deviance = u.deviance;
    let d_over_2phi = 0.5 * deviance;

    let gnorm = u.outer_gradient_norm.unwrap_or(f64::NAN);
    let converged = if u.outer_converged { 1.0 } else { 0.0 };
    let railed = u.lambdas.iter().filter(|&&l| l <= 1e-6).count();
    let lam_min = u.lambdas.iter().cloned().fold(f64::INFINITY, f64::min);
    let lam_max = u.lambdas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    eprintln!(
        "[#1426 basin] tag={tag} edf={edf:.4} k_hat={k_hat:.5} phi={phi:.6} \
         neg_ll={neg_ll:.5} pen_quad={pen_quad:.5} half_logdetH={half_logdet_h:.5} \
         half_logdetS={half_logdet_s:.5} D_over_2phi={d_over_2phi:.5} \
         occam_pair={:.5} cost_reml={reml:.5} \
         outer_converged={converged:.0} outer_gnorm={gnorm:.6e} \
         n_lambda={} railed_lambda={railed} lambda_min={lam_min:.4e} lambda_max={lam_max:.4e}",
        half_logdet_h - half_logdet_s,
        u.lambdas.len(),
    );

    edf
}

#[test]
fn diag_1426_gammalog_basin_dump() {
    init_parallelism();

    // The two reported failing datasets (overfit, EDF≈24) ...
    let edf_lcg7 = dump_components("lcg_seed7_OVERFIT", &build_data_lcg(7, 1500));
    let edf_sm900006 = dump_components(
        "splitmix_900006_OVERFIT",
        &build_data_splitmix(900006, 1500),
    );
    // ... and a healthy contrast (the bug-hunt control seed lands EDF≈9.4).
    let edf_sm900000 = dump_components(
        "splitmix_900000_HEALTHY",
        &build_data_splitmix(900000, 1500),
    );

    eprintln!(
        "[#1426 basin] SUMMARY edf_lcg7={edf_lcg7:.3} edf_splitmix900006={edf_sm900006:.3} \
         edf_splitmix900000_healthy={edf_sm900000:.3} (mgcv recovers EDF≈8 on this DGP)"
    );

    // Intentional failure so nextest surfaces the eprintln components above.
    // This is a DIAGNOSTIC dump, never a quality gate: the decision (A/C vs B)
    // is read off the printed `outer_converged` / `outer_gnorm` / cost columns,
    // not asserted here.
    assert!(
        false,
        "diag_1426_gammalog_basin: diagnostic dump only — read the [#1426 basin] component \
         lines above to decide A/C (converged stationary overfit ⇒ Occam-pair bug) vs B \
         (non-converged / large outer gradient ⇒ optimizer-basin bug)."
    );
}
