//! Common synthetic-data fixtures and numerical helpers shared by
//! survival-marginal-slope tests and finite-difference crosschecks.
//!
//! # Inline PRNG
//!
//! All fixtures use the `Splitmix64` struct rather than importing an external
//! crate.  The state is a `u64`; seeding is explicit so tests are
//! deterministic across platforms.
//!
//! # Survival-marginal-slope dataset
//!
//! `build_survival_marginal_slope_synth` produces an `EncodedDataset` with the
//! biobank-shape column layout used by several integration tests:
//!
//!   columns: `entry_age, exit_age, event, prs_z, PC1 .. PC{d_pc}`
//!
//! `build_bms_marginal_slope_synth` produces the low-level `(Array2<f64>,
//! Array1<f64>)` pair `(design_matrix, y)` used by Bernoulli-marginal-slope
//! unit tests.
//!
//! # Finite-difference Jacobian
//!
//! `finite_diff_jacobian` numerically differentiates a vector-valued function
//! `η(β): R^p → R^m` via central differences and returns the `(m, p)` Jacobian
//! matrix.

use csv::StringRecord;
use gam::{encode_recordswith_inferred_schema, inference::data::EncodedDataset};
use ndarray::{Array1, Array2};

// ── Inline PRNG: splitmix64 + Box-Muller ─────────────────────────────────

/// Deterministic, no-allocation PRNG backed by splitmix64.
///
/// Construct with a seed and call `next_unit` / `next_gauss` to draw samples.
pub struct Splitmix64 {
    state: u64,
}

impl Splitmix64 {
    /// Create a new RNG with the given 64-bit seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance the state and return the raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform sample in `[0, 1)` (53-bit mantissa).
    pub fn next_unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0_f64 / ((1u64 << 53) as f64))
    }

    /// Standard normal sample via Box-Muller.
    pub fn next_gauss(&mut self) -> f64 {
        let u1 = self.next_unit().max(f64::MIN_POSITIVE);
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        r * theta.cos()
    }

    /// Uniform sample in `[lo, hi)`.
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_unit()
    }

    /// Draw two independent standard normals using one Box-Muller step.
    ///
    /// Returns `(cos_sample, sin_sample)`. Useful when callers need pairs
    /// to avoid wasting one sample; the second element is independent of
    /// the first.
    pub fn next_gauss_pair(&mut self) -> (f64, f64) {
        // Use (0,1)-open lower bound to keep ln() finite.
        let u1 = self.next_unit().max(f64::MIN_POSITIVE);
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        (r * theta.cos(), r * theta.sin())
    }
}

// ── Survival-marginal-slope synthetic dataset ─────────────────────────────

/// Build a biobank-shape survival marginal-slope `EncodedDataset`.
///
/// Columns: `entry_age, exit_age, event, prs_z, PC1 .. PC{d_pc}`.
///
/// The data is generated deterministically from `seed`.  `n` is the row
/// count; `d_pc` is the number of principal-component columns; `centers` is
/// unused here but signals the caller's intent (the formula will reference
/// `duchon(PC1,..,PC{d_pc}, centers={centers}, order=1)`).
///
/// Generation rules:
/// * PC scores: N(0, 0.5²).
/// * PRS: N(0, 1), standardised to mean 0 variance 1.
/// * Entry age: 40 + Uniform(0, 10).
/// * Follow-up: 0.5 + Uniform(0, 8).
/// * Event: Bernoulli with log-odds = 0.3·prs + 0.4·pc1 − 0.3·pc2.
pub fn build_survival_marginal_slope_synth(
    n: usize,
    d_pc: usize,
    seed: u64,
) -> EncodedDataset {
    let mut rng = Splitmix64::new(seed);

    let mut headers: Vec<String> = vec![
        "entry_age".to_string(),
        "exit_age".to_string(),
        "event".to_string(),
        "prs_z".to_string(),
    ];
    for k in 0..d_pc {
        headers.push(format!("PC{}", k + 1));
    }

    let mut pcs: Vec<Vec<f64>> = (0..d_pc).map(|_| Vec::with_capacity(n)).collect();
    let mut prs: Vec<f64> = Vec::with_capacity(n);
    let mut entry: Vec<f64> = Vec::with_capacity(n);
    let mut exit: Vec<f64> = Vec::with_capacity(n);

    for _ in 0..n {
        for col in pcs.iter_mut() {
            col.push(rng.next_gauss() * 0.5);
        }
        prs.push(rng.next_gauss());
        let age_in = 40.0 + 10.0 * rng.next_unit();
        let followup = 0.5 + 8.0 * rng.next_unit();
        entry.push(age_in);
        exit.push(age_in + followup);
    }

    // Standardise PRS.
    let prs_mean = prs.iter().copied().sum::<f64>() / n as f64;
    let prs_var = prs
        .iter()
        .map(|v| (v - prs_mean).powi(2))
        .sum::<f64>()
        / n as f64;
    let prs_sd = prs_var.sqrt().max(1e-9);
    for v in &mut prs {
        *v = (*v - prs_mean) / prs_sd;
    }

    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for i in 0..n {
        let log_odds = 0.3 * prs[i]
            + if d_pc >= 1 { 0.4 * pcs[0][i] } else { 0.0 }
            + if d_pc >= 2 { -0.3 * pcs[1][i] } else { 0.0 }
            + if d_pc >= 3 { 0.2 * pcs[2][i] } else { 0.0 };
        let event: u32 = if log_odds > 0.0 { 1 } else { 0 };
        let mut fields: Vec<String> = vec![
            entry[i].to_string(),
            exit[i].to_string(),
            event.to_string(),
            prs[i].to_string(),
        ];
        for col in pcs.iter() {
            fields.push(col[i].to_string());
        }
        rows.push(StringRecord::from(fields));
    }

    encode_recordswith_inferred_schema(headers, rows)
        .expect("encode synthetic survival marginal-slope dataset")
}

// ── Bernoulli-marginal-slope synthetic data ───────────────────────────────

/// Build a small Bernoulli-marginal-slope synthetic problem.
///
/// Returns `(x, z, y)` where:
/// * `x` is an `(n, p)` design matrix of i.i.d. N(0, 1) entries,
/// * `z` is an `(n,)` standardised latent-score vector,
/// * `y` is an `(n,)` Bernoulli response generated from the linear predictor
///   `(sin(π·x[:,0]) + 0.5·cos(2π·x[:,0])) + 0.3·z`.
pub fn build_bms_marginal_slope_synth(
    n: usize,
    p: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = Splitmix64::new(seed);

    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            x[[i, j]] = rng.next_gauss();
        }
    }

    let mut z_raw: Vec<f64> = (0..n).map(|_| rng.next_gauss()).collect();
    let z_mean = z_raw.iter().copied().sum::<f64>() / n as f64;
    let z_var = z_raw.iter().map(|v| (v - z_mean).powi(2)).sum::<f64>() / n as f64;
    let z_sd = z_var.sqrt().max(1e-9);
    for v in &mut z_raw {
        *v = (*v - z_mean) / z_sd;
    }

    let pi = std::f64::consts::PI;
    let tau = std::f64::consts::TAU;
    let y: Array1<f64> = (0..n)
        .map(|i| {
            let xi = x[[i, 0]];
            let eta = (pi * xi).sin() + 0.5 * (tau * xi).cos() + 0.3 * z_raw[i];
            let p_one = 1.0 / (1.0 + (-eta).exp());
            if rng.next_unit() < p_one { 1.0 } else { 0.0 }
        })
        .collect();

    let z = Array1::from(z_raw);
    (x, z, y)
}

// ── GAMLSS synthetic data ────────────────────────────────────────────────

/// Build a small GAMLSS (Gaussian location-scale) synthetic dataset.
///
/// Returns `(x, z, y)` where `x` is the location design `(n, p)`, `z` is the
/// log-scale design `(n, q)` (both i.i.d. N(0,1)), and `y ~ N(x β_true, exp(z
/// γ_true))` with `β_true = 1/p` and `γ_true = −0.5/q`.
pub fn build_gamlss_synth(
    n: usize,
    p: usize,
    q: usize,
    seed: u64,
) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
    let mut rng = Splitmix64::new(seed);

    let mut x = Array2::<f64>::zeros((n, p));
    let mut z = Array2::<f64>::zeros((n, q));
    for i in 0..n {
        for j in 0..p {
            x[[i, j]] = rng.next_gauss();
        }
        for j in 0..q {
            z[[i, j]] = rng.next_gauss();
        }
    }

    let beta_val = if p > 0 { 1.0 / p as f64 } else { 0.0 };
    let gamma_val = if q > 0 { -0.5 / q as f64 } else { 0.0 };
    let y: Array1<f64> = (0..n)
        .map(|i| {
            let mu: f64 = (0..p).map(|j| x[[i, j]] * beta_val).sum();
            let log_sigma: f64 = (0..q).map(|j| z[[i, j]] * gamma_val).sum();
            let sigma = log_sigma.exp();
            mu + sigma * rng.next_gauss()
        })
        .collect();

    (x, z, y)
}

// ── SAE synthetic data ────────────────────────────────────────────────────

/// Build a small SAE (sparse-autoencoder) synthetic observation matrix.
///
/// Returns an `(n, d)` matrix drawn from a mixture of `k` Gaussian components
/// with random means and unit variance, seeded by `seed`.
pub fn build_sae_synth(n: usize, d: usize, k: usize, seed: u64) -> Array2<f64> {
    let mut rng = Splitmix64::new(seed);

    // Component centres: (k, d).
    let mut centres = Array2::<f64>::zeros((k, d));
    for i in 0..k {
        for j in 0..d {
            centres[[i, j]] = rng.next_gauss() * 2.0;
        }
    }

    let mut x = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        // Pick a component.
        let bits = rng.next_u64();
        let comp = (bits as usize) % k;
        for j in 0..d {
            x[[i, j]] = centres[[comp, j]] + rng.next_gauss();
        }
    }
    x
}

// ── Finite-difference Jacobian ────────────────────────────────────────────

/// Numerically differentiate `eta_fn: R^p → R^m` at `beta` using central
/// differences with step size `eps`.
///
/// Returns the `(m, p)` Jacobian matrix `J` where
/// `J[row, col] ≈ (eta_fn(β + eps·e_col)[row] - eta_fn(β - eps·e_col)[row]) / (2·eps)`.
pub fn finite_diff_jacobian<F>(eta_fn: F, beta: &Array1<f64>, eps: f64) -> Array2<f64>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let p = beta.len();
    let eta0 = eta_fn(beta);
    let m = eta0.len();

    let mut jac = Array2::<f64>::zeros((m, p));
    let mut beta_plus = beta.clone();
    let mut beta_minus = beta.clone();

    for col in 0..p {
        beta_plus[col] = beta[col] + eps;
        beta_minus[col] = beta[col] - eps;
        let eta_plus = eta_fn(&beta_plus);
        let eta_minus = eta_fn(&beta_minus);
        for row in 0..m {
            jac[[row, col]] = (eta_plus[row] - eta_minus[row]) / (2.0 * eps);
        }
        beta_plus[col] = beta[col];
        beta_minus[col] = beta[col];
    }
    jac
}
