//! Regression: the stored Gamma dispersion `φ = 1/shape` must reflect the
//! *conditional* noise of the response, not the spread of the conditional mean.
//!
//! For a Gamma response `Var(Y|x) = φ·μ(x)²`, so `φ` is a property of the
//! conditional noise alone and is invariant to how spread out `μ(x)` is. The
//! engine previously estimated the Gamma shape **once** from an early, far-from-
//! converged warm-start linear predictor and then froze it for the remainder of
//! the inner P-IRLS solve (`gamma_shape_locked = true`). The final dedicated fit
//! at the converged smoothing parameter cold-starts from `β = 0`, so the shape
//! was read off a linear predictor that had not yet captured the mean structure;
//! the leftover spread of `μ` inflated the deviance term `mean[y/μ − ln(y/μ) − 1]`
//! and drove the shape estimate down (dispersion up) by `>2×` whenever the mean
//! varied appreciably. See issue #678.
//!
//! This test builds deterministic, RNG-free `Y ~ Gamma(shape = 4)` data on a
//! log-linear mean and checks the stored `dispersion_phi()` against the data's
//! true conditional dispersion computed model-independently from the *known*
//! means. It is the originally-committed failing repro from the issue.

use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

use csv::StringRecord;

/// Deterministic Gamma(shape) quantiles via the Wilson–Hilferty cube-root
/// normal approximation, evaluated on a fixed grid of standard-normal quantiles.
/// This yields RNG-free draws whose empirical shape is ≈ the requested shape, so
/// the test's "truth" is reproducible bit-for-bit across platforms.
fn deterministic_gamma_unit_mean(shape: f64, n: usize) -> Vec<f64> {
    // Evenly spaced probabilities in (0, 1), excluding the endpoints.
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let p = (i as f64 + 0.5) / n as f64;
        // Inverse standard normal (Acklam's rational approximation).
        let z = inv_std_normal(p);
        // Wilson–Hilferty: a Gamma(shape) variable g satisfies
        // (g/shape)^(1/3) ≈ Normal(1 - 1/(9 shape), 1/(9 shape)).
        let c = 1.0 / (9.0 * shape);
        let root = 1.0 - c + z * c.sqrt();
        let g_over_shape = root.max(1e-6).powi(3);
        let g = g_over_shape * shape; // Gamma(shape) draw, mean = shape
        out.push(g / shape); // normalize so the unit-mean sample has mean ≈ 1
    }
    out
}

/// Acklam's inverse-normal-CDF rational approximation (abs error < 1.15e-9).
fn inv_std_normal(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[test]
fn gamma_dispersion_is_conditional_noise_not_mean_spread() {
    init_parallelism();

    // --- deterministic Gamma(shape = 4) data on a log-linear mean -----------
    // True conditional dispersion φ = 1/shape = 0.25 (CV = 0.5). The mean
    // μ = exp(0.3 + 0.5·x) spans exp(2) ≈ 7.4× over x ∈ [−2, 2], so any
    // mean-spread contamination of the dispersion estimate is large.
    let shape_true = 4.0_f64;
    let n = 400usize;
    let unit = deterministic_gamma_unit_mean(shape_true, n);

    // x evenly spaced on [-2, 2]; pair each x with a unit-mean draw so the
    // residual structure is independent of the mean trend. Interleave the unit
    // draws (which are sorted by construction) against x so we do not induce a
    // spurious mean/variance coupling.
    let mut x = Vec::with_capacity(n);
    let mut mu = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let xi = -2.0 + 4.0 * (i as f64) / ((n - 1) as f64);
        let mui = (0.3 + 0.5 * xi).exp();
        // Decorrelate the unit draw index from x: bit-reverse-ish shuffle.
        let j = (i * 257 + 13) % n;
        let yi = mui * unit[j];
        x.push(xi);
        mu.push(mui);
        y.push(yi);
    }
    assert!(y.iter().all(|&v| v > 0.0), "Gamma outcomes must be positive");

    // --- model-independent truth: φ from the KNOWN means --------------------
    // E[(y/μ − 1)²] = Var(y)/μ² = φ. This uses the true μ, not any fitted value,
    // so it is an unbiased estimate of the conditional dispersion regardless of
    // the model.
    let phi_true: f64 =
        y.iter().zip(mu.iter()).map(|(&yi, &mui)| (yi / mui - 1.0).powi(2)).sum::<f64>()
            / (n as f64);

    // --- fit y ~ s(x, k=10) with a Gamma(log) family ------------------------
    let headers: Vec<String> = ["y", "x"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![y[i].to_string(), x[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode gamma dataset");

    let cfg = FitConfig {
        family: Some("gamma".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=10)", &ds, &cfg).expect("gam gamma fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Gamma(log)");
    };

    let phi_hat = fit.fit.dispersion_phi();
    let shape_hat = 1.0 / phi_hat;

    // Pearson φ from the model's own fitted means (sanity cross-check: the
    // estimator is consistent at the fitted means, so this should also be ≈ φ).
    eprintln!(
        "phi_true = {phi_true:.4} (shape {:.3}); stored dispersion_phi = {phi_hat:.4} (shape {shape_hat:.3})",
        1.0 / phi_true
    );

    // The stored dispersion must match the data's true conditional dispersion,
    // not be inflated by the spread of μ. A generous 25% band absorbs the finite-
    // sample noise of the deterministic draw while still catching the >2× bug.
    let ratio = phi_hat / phi_true;
    assert!(
        (0.8..=1.25).contains(&ratio),
        "stored Gamma dispersion φ̂ = {phi_hat:.4} (shape ν̂ = {shape_hat:.3}) is {ratio:.2}× the \
         data's true conditional dispersion φ_true = {phi_true:.4} (true shape ≈ {:.2}); the shape \
         must be read at the converged linear predictor, not an early warm-start η",
        1.0 / phi_true
    );
}
