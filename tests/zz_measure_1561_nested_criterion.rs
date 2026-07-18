//! zz_measure DIAGNOSTIC (#1561 / #2356): does the NESTED-marginalization
//! criterion move λ̂_μ toward the oracle, while the JOINT-LAML criterion does
//! not? This is the decisive experiment for the #2356 fix design.
//!
//! Background (see #2356 and the #1561 theory memo)
//! ------------------------------------------------
//! The shipped location-scale criterion is a JOINT LAML over the flattened
//! β = [β_μ; β_σ]: it finds the joint penalized mode and prices complexity with
//! −½log|H_obs+S| on the observed joint Hessian. At that joint mode the σ-score
//! calibrates σ̂ against the REALIZED squared residuals r̂²:
//!
//!     Σ_i x_σᵢ κ_i (b_i − 1) − λ_σ S_σ β_σ = 0 ,   b_i = r̂_i²/σ̂_i² .
//!
//! But β_μ is a penalized Gaussian block, so its marginalization CONDITIONAL on
//! β_σ is available in closed form. The exact μ-marginal likelihood is
//!
//!     M(β_σ) = ℓ_p(β̂_μ(β_σ), β_σ) − ½log|H_μμ(β_σ)+S_μ| + ½log|S_μ|₊ + const,
//!
//! whose σ-stationarity adds the pointwise leverage term h_i:
//!
//!     Σ_i x_σᵢ κ_i (b_i − 1 + h_i) − λ_σ S_σ β_σ = 0 ,
//!     h_i = W_i x_{μi}ᵀ (X_μᵀ W X_μ + S_μ)⁻¹ x_{μi},   W = diag(1/σ²),  Σ_i h_i = edf_μ.
//!
//! This is the functional generalization of ML's RSS/n vs REML's RSS/(n−edf).
//! The joint mode DROPS h_i, biasing σ̂ low by the μ-leverage; every μ-complexity
//! price is then denominated in those biased weights, so λ̂_μ is under-selected.
//!
//! What this test measures
//! -----------------------
//! For each quality-loser arm, at the FIXED production λ̂_σ, it sweeps the μ
//! penalty λ_μ = λ̂_μ·e^{ρ_μ} and, at each ρ_μ, solves BOTH modes (profiling the
//! linear-Gaussian μ block, Newton on β_σ) and evaluates BOTH criteria:
//!
//!   V_joint(ρ_μ)  = −ℓ + ½‖β̂‖²_S + ½log|H_joint+S| − ½ r_μ ρ_μ
//!   V_nested(ρ_μ) = −ℓ + ½‖β̂‖²_S + ½log|H_μμ+S_μ| + ½log|A_σ| − ½ r_μ ρ_μ
//!
//! (shared ρ_μ-independent constants dropped; A_σ = −∇²_{β_σ} of the μ-marginal
//! σ-objective at the nested mode, by finite differences.) It reports the argmin
//! ρ_μ and the edf_μ there for each.
//!
//! Decisive prediction (the #2356 fix rationale): argmin V_joint ≈ ρ_μ = 0 (it
//! reproduces the production joint selection — a validity check), while
//! argmin V_nested lands at ρ_μ > 0 with a MUCH lower edf_μ, near the oracle
//! (edf_μ* ≈ 5.8 cyclic / 13.7 plain from the sibling sweep). If so, the nested
//! correction is the mechanism and the augmented-potential fix is validated.
//!
//! zz_measure discipline: numbers are eprintln'd; the only hard asserts are
//! finiteness and the ρ=0 reconstruction fidelity (β̂_μ, β̂_σ from the two solves
//! must reproduce the production fit — that validates the whole reconstruction).

use gam::estimate::BlockRole;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

const LOGB_SIGMA_FLOOR: f64 = 0.01;

// ---------------------------------------------------------------------------
// Data generators — bit-identical to the two #1561 quality losers (shared with
// tests/zz_measure_1561_mu_oracle_lambda_sweep.rs).
// ---------------------------------------------------------------------------

fn cyclic_standard_normals(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut next_unit = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = state >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = next_unit();
        let u2 = next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        out.push(r * theta.cos());
        if out.len() < n {
            out.push(r * theta.sin());
        }
    }
    out.truncate(n);
    out
}

fn next_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

struct Arm {
    label: &'static str,
    headers: Vec<String>,
    x_name: &'static str,
    mean_formula: &'static str,
    noise_formula: &'static str,
    xs: Vec<f64>,
    ys: Vec<f64>,
    grid_x: Vec<f64>,
    truth_mu_grid: Vec<f64>,
}

fn cyclic_arm() -> Arm {
    let n = 150usize;
    let period = 2.0 * PI;
    let xs: Vec<f64> = (0..n)
        .map(|i| period * (i as f64) / ((n - 1) as f64))
        .collect();
    let z = cyclic_standard_normals(n, 123);
    let ys: Vec<f64> = xs
        .iter()
        .zip(z.iter())
        .map(|(&x, &zi)| x.sin() + (0.15 + 0.1 * x.cos()) * zi)
        .collect();
    let m = 50usize;
    let grid_x: Vec<f64> = (0..m).map(|i| period * (i as f64) / (m as f64)).collect();
    let truth_mu_grid: Vec<f64> = grid_x.iter().map(|&x| x.sin()).collect();
    Arm {
        label: "cyclic",
        headers: vec!["y".to_string(), "x".to_string()],
        x_name: "x",
        mean_formula: "y ~ s(x, bs='cc', period_start=0, period_end=6.283185307179586)",
        noise_formula: "1 + s(x, bs='cc', period_start=0, period_end=6.283185307179586)",
        xs,
        ys,
        grid_x,
        truth_mu_grid,
    }
}

fn plain_arm() -> Arm {
    let n = 200usize;
    let two_pi = 2.0 * PI;
    let mut state = 42u64;
    let mut xs: Vec<f64> = (0..n).map(|_| next_unit(&mut state)).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
    let mut z = Vec::with_capacity(n);
    while z.len() < n {
        let u1 = next_unit(&mut state).max(1e-300);
        let u2 = next_unit(&mut state);
        let r = (-2.0 * u1.ln()).sqrt();
        z.push(r * (two_pi * u2).cos());
        if z.len() < n {
            z.push(r * (two_pi * u2).sin());
        }
    }
    let ys: Vec<f64> = (0..n)
        .map(|i| (two_pi * xs[i]).sin() + (0.1 + 0.2 * (two_pi * xs[i]).sin()) * z[i])
        .collect();
    let m = 100usize;
    let grid_x: Vec<f64> = (0..m).map(|i| (i as f64 + 0.5) / (m as f64)).collect();
    let truth_mu_grid: Vec<f64> = grid_x.iter().map(|&x| (two_pi * x).sin()).collect();
    Arm {
        label: "plain",
        headers: vec!["x".to_string(), "y".to_string()],
        x_name: "x",
        mean_formula: "y ~ s(x, bs='tp')",
        noise_formula: "1 + s(x, bs='tp')",
        xs,
        ys,
        grid_x,
        truth_mu_grid,
    }
}

// ---------------------------------------------------------------------------
// Dense linear algebra (small systems, p ≈ 11..26; hand-rolled, dependency-free).
// ---------------------------------------------------------------------------

fn chol_lower(a: &Array2<f64>) -> Option<Array2<f64>> {
    let p = a.nrows();
    let mut l = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    Some(l)
}

fn chol_with_jitter(a: &Array2<f64>) -> Array2<f64> {
    if let Some(l) = chol_lower(a) {
        return l;
    }
    let p = a.nrows();
    let scale: f64 = (0..p).map(|i| a[[i, i]].abs()).sum::<f64>() / (p as f64);
    let mut jitter = scale.max(1.0) * 1e-12;
    for _ in 0..50 {
        let mut aj = a.clone();
        for i in 0..p {
            aj[[i, i]] += jitter;
        }
        if let Some(l) = chol_lower(&aj) {
            return l;
        }
        jitter *= 10.0;
    }
    panic!("matrix not factorizable even with jitter");
}

fn chol_solve_vec(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let p = l.nrows();
    let mut y = Array1::<f64>::zeros(p);
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= l[[i, k]] * y[k];
        }
        y[i] = s / l[[i, i]];
    }
    let mut x = Array1::<f64>::zeros(p);
    for i in (0..p).rev() {
        let mut s = y[i];
        for k in (i + 1)..p {
            s -= l[[k, i]] * x[k];
        }
        x[i] = s / l[[i, i]];
    }
    x
}

/// log determinant of an SPD matrix from its Cholesky factor: 2·Σ log L_ii.
fn logdet_spd(a: &Array2<f64>) -> f64 {
    let l = chol_with_jitter(a);
    2.0 * (0..l.nrows()).map(|i| l[[i, i]].ln()).sum::<f64>()
}

/// tr(A⁻¹ B) via columnwise SPD solves.
fn trace_ainv_b(l: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let p = b.nrows();
    let mut tr = 0.0;
    for j in 0..p {
        let col = b.column(j).to_owned();
        let z = chol_solve_vec(l, &col);
        tr += z[j];
    }
    tr
}

/// diag(X A⁻¹ Xᵀ)_i = x_iᵀ A⁻¹ x_i for every row of X (n×p), given chol(A).
fn row_quadratic_forms(l: &Array2<f64>, x: &Array2<f64>) -> Array1<f64> {
    let n = x.nrows();
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi = x.row(i).to_owned();
        let z = chol_solve_vec(l, &xi);
        out[i] = xi.dot(&z);
    }
    out
}

/// Symmetric eigenvalues via cyclic Jacobi (small matrices only). Returns them
/// unsorted; used solely to count numerical rank of the μ penalty.
fn symmetric_eigenvalues(a: &Array2<f64>) -> Vec<f64> {
    let p = a.nrows();
    let mut m = a.clone();
    for _sweep in 0..100 {
        let mut off = 0.0;
        for i in 0..p {
            for j in (i + 1)..p {
                off += m[[i, j]].powi(2);
            }
        }
        if off < 1e-24 {
            break;
        }
        for q in 0..p {
            for r in (q + 1)..p {
                let apq = m[[q, r]];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = m[[q, q]];
                let arr = m[[r, r]];
                let phi = 0.5 * (2.0 * apq).atan2(arr - app);
                let (s, c) = phi.sin_cos();
                for k in 0..p {
                    let mqk = m[[q, k]];
                    let mrk = m[[r, k]];
                    m[[q, k]] = c * mqk - s * mrk;
                    m[[r, k]] = s * mqk + c * mrk;
                }
                for k in 0..p {
                    let mkq = m[[k, q]];
                    let mkr = m[[k, r]];
                    m[[k, q]] = c * mkq - s * mkr;
                    m[[k, r]] = s * mkq + c * mkr;
                }
            }
        }
    }
    (0..p).map(|i| m[[i, i]]).collect()
}

fn numerical_rank(a: &Array2<f64>) -> usize {
    let ev = symmetric_eigenvalues(a);
    let max = ev.iter().cloned().fold(0.0f64, |m, v| m.max(v.abs()));
    if max == 0.0 {
        return 0;
    }
    ev.iter().filter(|&&v| v > 1e-8 * max).count()
}

// ---------------------------------------------------------------------------
// Reconstruction of the production loc-scale problem in raw units.
// ---------------------------------------------------------------------------

struct Reconstructed {
    label: &'static str,
    n: usize,
    x_mu: Array2<f64>,      // n × p_mu (training)
    x_sigma: Array2<f64>,   // n × p_sigma (training)
    o_mu: Array1<f64>,      // μ affine offset (training)
    o_sigma: Array1<f64>,   // η_σ affine offset (training)
    y: Array1<f64>,
    s_mu: Array2<f64>,      // Σ_k (λ̂_μk/c²) S_μk  (raw-space base μ penalty at ρ=0)
    s_mu_blocks: Vec<Array2<f64>>, // individual λ̂_μk/c² S_μk (independent scaling)
    s_sigma: Array2<f64>,   // Σ_k λ̂_σk S_σk  (raw-space σ penalty)
    r_mu: usize,            // rank of the base μ penalty
    // production fit references
    beta_mu: Array1<f64>,
    beta_sigma: Array1<f64>,
    edf_mu: f64,
    lambda_mu: Vec<f64>,
    lambda_sigma: Vec<f64>,
    c: f64,
    // μ eval-grid design for μ-RMSE-to-truth
    x_grid: Array2<f64>,
    o_grid: Array1<f64>,
    truth_mu_grid: Vec<f64>,
}

fn reconstruct(arm: &Arm) -> Reconstructed {
    let n = arm.xs.len();
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            let vals: Vec<String> = arm
                .headers
                .iter()
                .map(|h| {
                    if h.as_str() == arm.x_name {
                        format!("{:.17e}", arm.xs[i])
                    } else {
                        format!("{:.17e}", arm.ys[i])
                    }
                })
                .collect();
            csv::StringRecord::from(vals)
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(arm.headers.clone(), rows).expect("encode arm");
    let col = ds.column_map();
    let x_idx = col[arm.x_name];
    let ncols = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some(arm.noise_formula.to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(arm.mean_formula, &ds, &cfg).expect("gam loc-scale fit");
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult {
        fit,
        response_scale,
        ..
    }) = result
    else {
        panic!("expected a GaussianLocationScale fit");
    };
    let c = response_scale;
    // Read the outer optimizer's OWN convergence certificate: does it think it
    // reached a stationary point (small |g|), or did it stall on a flat valley
    // with a large residual gradient? This distinguishes an analytic-gradient
    // desync from a premature-termination bug.
    let cert_summary = fit
        .fit
        .convergence_evidence()
        .outer_certificate()
        .map(|c| c.summary())
        .unwrap_or_else(|| "<no outer certificate>".to_string());
    eprintln!(
        "[{}] OUTER CONVERGENCE CERTIFICATE: {cert_summary}  (outer_iters={})",
        arm.label,
        fit.fit.outer_iterations
    );
    let loc = fit.fit.block_by_role(BlockRole::Location).expect("loc");
    let sca = fit.fit.block_by_role(BlockRole::Scale).expect("scale");
    let beta_mu = loc.beta.clone();
    let beta_sigma = sca.beta.clone();
    let lambda_mu = loc.lambdas.to_vec();
    let lambda_sigma = sca.lambdas.to_vec();
    let edf_mu = loc.edf;

    // Rebuild designs + penalties at training points, same basis as β̂.
    let mut train = Array2::<f64>::zeros((n, ncols));
    for (i, &x) in arm.xs.iter().enumerate() {
        train[[i, x_idx]] = x;
    }
    let md = build_term_collection_design(train.view(), &fit.meanspec_resolved).expect("μ design");
    let nd = build_term_collection_design(train.view(), &fit.noisespec_resolved).expect("σ design");
    let p_mu = md.design.ncols();
    let p_sigma = nd.design.ncols();
    assert_eq!(p_mu, beta_mu.len());
    assert_eq!(p_sigma, beta_sigma.len());

    let materialize = |dm: &gam::smooth::TermCollectionDesign, p: usize| -> Array2<f64> {
        let mut x = Array2::<f64>::zeros((dm.design.nrows(), p));
        for j in 0..p {
            let mut e = Array1::<f64>::zeros(p);
            e[j] = 1.0;
            let cj = dm.design.apply(&e);
            x.column_mut(j).assign(&cj);
        }
        x
    };
    let x_mu = materialize(&md, p_mu);
    let x_sigma = materialize(&nd, p_sigma);
    let o_mu = md.affine_offset.clone();
    let o_sigma = nd.affine_offset.clone();

    let embed = |dm: &gam::smooth::TermCollectionDesign, p: usize, lambdas: &[f64], div_c2: bool| -> Array2<f64> {
        let mut s = Array2::<f64>::zeros((p, p));
        for (k, bp) in dm.penalties.iter().enumerate() {
            let coef = if div_c2 { lambdas[k] / (c * c) } else { lambdas[k] };
            for (a, gi) in bp.col_range.clone().enumerate() {
                for (b, gj) in bp.col_range.clone().enumerate() {
                    s[[gi, gj]] += coef * bp.local[[a, b]];
                }
            }
        }
        s
    };
    // Per-penalty embedded μ blocks (each λ̂_μk/c² S_μk) so the two tp penalties
    // can be scaled INDEPENDENTLY — the production optimizer selects them apart.
    let s_mu_blocks: Vec<Array2<f64>> = md
        .penalties
        .iter()
        .enumerate()
        .map(|(k, bp)| {
            let mut s = Array2::<f64>::zeros((p_mu, p_mu));
            let coef = lambda_mu[k] / (c * c);
            for (a, gi) in bp.col_range.clone().enumerate() {
                for (b, gj) in bp.col_range.clone().enumerate() {
                    s[[gi, gj]] += coef * bp.local[[a, b]];
                }
            }
            s
        })
        .collect();
    let s_mu = embed(&md, p_mu, &lambda_mu, true);
    let s_sigma = embed(&nd, p_sigma, &lambda_sigma, false);
    let r_mu = numerical_rank(&s_mu);

    // μ eval grid.
    let mg = arm.grid_x.len();
    let mut eg = Array2::<f64>::zeros((mg, ncols));
    for (i, &gx) in arm.grid_x.iter().enumerate() {
        eg[[i, x_idx]] = gx;
    }
    let mdg = build_term_collection_design(eg.view(), &fit.meanspec_resolved).expect("μ grid");
    let x_grid = materialize(&mdg, p_mu);
    let o_grid = mdg.affine_offset.clone();

    Reconstructed {
        label: arm.label,
        n,
        x_mu,
        x_sigma,
        o_mu,
        o_sigma,
        y: Array1::from(arm.ys.clone()),
        s_mu,
        s_mu_blocks,
        s_sigma,
        r_mu,
        beta_mu,
        beta_sigma,
        edf_mu,
        lambda_mu,
        lambda_sigma,
        c,
        x_grid,
        o_grid,
        truth_mu_grid: arm.truth_mu_grid.clone(),
    }
}

// ---------------------------------------------------------------------------
// Joint / nested solve at a μ-penalty scale exp(ρ_μ), λ_σ fixed at production.
// ---------------------------------------------------------------------------

struct SolveOut {
    beta_mu: Array1<f64>,
    beta_sigma: Array1<f64>,
    edf_mu: f64,
    /// negative log-likelihood value at the mode (raw Gaussian loc-scale).
    neg_ll: f64,
    penalty_quad: f64, // β_μᵀ(e^{ρ}S_μ)β_μ + β_σᵀS_σβ_σ
    logdet_h_joint: f64,   // log|H_joint + S|  (Cholesky/jitter — PD assumption)
    logdet_h_abs: f64,     // Σ log|λ_i(H_joint+S)|  (abs-eigenvalue pseudo-logdet)
    logdet_h_floor: f64,   // Σ log(max(λ_i, floor))  (smooth positive floor)
    hj_min_eig: f64,       // min eigenvalue of H_joint+S
    hj_neg_count: usize,   // # negative eigenvalues of H_joint+S
    logdet_hmumu: f64,     // log|X_μᵀWX_μ + e^{ρ}S_μ|
    logdet_a_sigma: f64,   // log|A_σ|  (nested σ curvature)
}

/// σ from η in the raw-exp convention: σ = c·FLOOR + exp(η).
#[inline]
fn sigma_of(eta: f64, c: f64) -> f64 {
    c * LOGB_SIGMA_FLOOR + eta.exp()
}
/// κ = (dσ/dη)/σ = exp(η)/σ.
#[inline]
fn kappa_of(eta: f64, c: f64) -> f64 {
    let e = eta.exp();
    e / (c * LOGB_SIGMA_FLOOR + e)
}

/// Given β_σ, profile-solve the linear-Gaussian μ block; return
/// (β_μ, σ, κ, W, r, b, edf_μ, chol(H_μμ+λ_μS_μ), H_μμ).
struct MuProfile {
    beta_mu: Array1<f64>,
    sigma: Array1<f64>,
    kappa: Array1<f64>,
    b: Array1<f64>,     // r²/σ²
    edf_mu: f64,
    hmumu: Array2<f64>, // X_μᵀWX_μ
    h_lev: Array1<f64>, // per-row leverage h_i = W_i x_μiᵀ(H_μμ+λS_μ)⁻¹x_μi
}

fn profile_mu(rec: &Reconstructed, beta_sigma: &Array1<f64>, s_mu_scaled: &Array2<f64>) -> MuProfile {
    let n = rec.n;
    let eta_sigma = rec.x_sigma.dot(beta_sigma) + &rec.o_sigma;
    let sigma: Array1<f64> = eta_sigma.mapv(|e| sigma_of(e, rec.c));
    let kappa: Array1<f64> = eta_sigma.mapv(|e| kappa_of(e, rec.c));
    let w: Array1<f64> = sigma.mapv(|s| 1.0 / (s * s));
    // H_μμ = X_μᵀ W X_μ, and RHS X_μᵀ W (y − o_μ).
    let p = rec.x_mu.ncols();
    let mut hmumu = Array2::<f64>::zeros((p, p));
    let mut rhs = Array1::<f64>::zeros(p);
    for i in 0..n {
        let wi = w[i];
        let ri = rec.y[i] - rec.o_mu[i];
        for a in 0..p {
            let xa = rec.x_mu[[i, a]];
            rhs[a] += wi * xa * ri;
            for bb in 0..p {
                hmumu[[a, bb]] += wi * xa * rec.x_mu[[i, bb]];
            }
        }
    }
    let mut pen = hmumu.clone();
    pen += s_mu_scaled;
    let l_pen = chol_with_jitter(&pen);
    let beta_mu = chol_solve_vec(&l_pen, &rhs);
    let mu = rec.x_mu.dot(&beta_mu) + &rec.o_mu;
    let resid = &rec.y - &mu;
    let b: Array1<f64> = (0..n).map(|i| (resid[i] * resid[i]) / (sigma[i] * sigma[i])).collect();
    let edf_mu = trace_ainv_b(&l_pen, &hmumu);
    // leverage h_i = W_i x_μiᵀ (H_μμ+λS_μ)⁻¹ x_μi
    let quad = row_quadratic_forms(&l_pen, &rec.x_mu);
    let h_lev: Array1<f64> = (0..n).map(|i| w[i] * quad[i]).collect();
    MuProfile {
        beta_mu,
        sigma,
        kappa,
        b,
        edf_mu,
        hmumu,
        h_lev,
    }
}

/// σ-block gradient of the (joint or nested) objective at β_σ, profiling μ.
/// Maximization gradient: X_σᵀ (κ (b − 1 [+ h])) − λ_σ S_σ β_σ.
fn sigma_gradient(
    rec: &Reconstructed,
    prof: &MuProfile,
    beta_sigma: &Array1<f64>,
    use_h: bool,
) -> Array1<f64> {
    let n = rec.n;
    let mut per_row = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut term = prof.b[i] - 1.0;
        if use_h {
            term += prof.h_lev[i];
        }
        per_row[i] = prof.kappa[i] * term;
    }
    let mut g = rec.x_sigma.t().dot(&per_row);
    g -= &rec.s_sigma.dot(beta_sigma);
    g
}

fn solve_mode(rec: &Reconstructed, rho_mu: f64, use_h: bool, beta_sigma0: &Array1<f64>) -> SolveOut {
    let s_mu_scaled = &rec.s_mu * rho_mu.exp();
    solve_mode_with_penalty(rec, &s_mu_scaled, use_h, beta_sigma0)
}

/// Log pseudo-determinant of an SPD-ish penalty (sum of log of eigenvalues above
/// a relative floor) — the exact −2× of the REML prior-normalizer term.
fn logdet_pseudo(s: &Array2<f64>) -> f64 {
    let ev = symmetric_eigenvalues(s);
    let max = ev.iter().cloned().fold(0.0f64, |m, v| m.max(v.abs()));
    if max == 0.0 {
        return 0.0;
    }
    ev.iter().filter(|&&v| v > 1e-8 * max).map(|&v| v.ln()).sum()
}

fn solve_mode_with_penalty(
    rec: &Reconstructed,
    s_mu_scaled: &Array2<f64>,
    use_h: bool,
    beta_sigma0: &Array1<f64>,
) -> SolveOut {
    let n = rec.n;
    let s_mu_scaled = s_mu_scaled.clone();
    let mut beta_sigma = beta_sigma0.clone();
    // Damped Newton on β_σ with a Fisher-ish PD curvature (2κ²) — converges to
    // the score root regardless of the curvature used for stepping.
    for _iter in 0..200 {
        let prof = profile_mu(rec, &beta_sigma, &s_mu_scaled);
        let g = sigma_gradient(rec, &prof, &beta_sigma, use_h);
        let gnorm = g.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        if gnorm < 1e-10 {
            break;
        }
        // curvature ≈ X_σᵀ diag(2κ²) X_σ + λ_σ S_σ
        let p = rec.x_sigma.ncols();
        let mut curv = rec.s_sigma.clone();
        for i in 0..n {
            let wgt = 2.0 * prof.kappa[i] * prof.kappa[i];
            for a in 0..p {
                let xa = rec.x_sigma[[i, a]];
                for bb in 0..p {
                    curv[[a, bb]] += wgt * xa * rec.x_sigma[[i, bb]];
                }
            }
        }
        let l = chol_with_jitter(&curv);
        let step = chol_solve_vec(&l, &g);
        // backtracking on the score norm to stay robust
        let mut t = 1.0;
        let mut accepted = false;
        for _ls in 0..30 {
            let trial = &beta_sigma + &(&step * t);
            let pr = profile_mu(rec, &trial, &s_mu_scaled);
            let gt = sigma_gradient(rec, &pr, &trial, use_h);
            let gtn = gt.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            if gtn < gnorm * (1.0 - 0.1 * t) + 1e-12 {
                beta_sigma = trial;
                accepted = true;
                break;
            }
            t *= 0.5;
        }
        if !accepted {
            beta_sigma = &beta_sigma + &(&step * t);
        }
    }
    let prof = profile_mu(rec, &beta_sigma, &s_mu_scaled);

    // Value pieces.
    let mut neg_ll = 0.0;
    let ln2pi = (2.0 * PI).ln();
    for i in 0..n {
        let s = prof.sigma[i];
        neg_ll += 0.5 * ln2pi + s.ln() + 0.5 * prof.b[i];
    }
    let pen_mu = prof.beta_mu.dot(&s_mu_scaled.dot(&prof.beta_mu));
    let pen_sigma = beta_sigma.dot(&rec.s_sigma.dot(&beta_sigma));
    let penalty_quad = pen_mu + pen_sigma;

    // Joint observed Hessian H_joint + S.  Blocks (per row):
    //   mm = W = 1/σ² ; ml = 2κ m, m = r·W ; ll = κ'(a−n)+2κ²n, a=1, n=b.
    // κ' = dκ/dη = κ(1−κ) for σ = floor + exp(η).
    let p_mu = rec.x_mu.ncols();
    let p_sigma = rec.x_sigma.ncols();
    let tot = p_mu + p_sigma;
    let mut hj = Array2::<f64>::zeros((tot, tot));
    for i in 0..n {
        let sig = prof.sigma[i];
        let wi = 1.0 / (sig * sig);
        let ri = rec.y[i] - (rec.x_mu.row(i).dot(&prof.beta_mu) + rec.o_mu[i]);
        let mi = ri * wi;
        let k = prof.kappa[i];
        let kp = k * (1.0 - k);
        let ni = prof.b[i];
        let mm = wi;
        let ml = 2.0 * k * mi;
        let ll = kp * (1.0 - ni) + 2.0 * k * k * ni;
        for a in 0..p_mu {
            let xa = rec.x_mu[[i, a]];
            for bb in 0..p_mu {
                hj[[a, bb]] += mm * xa * rec.x_mu[[i, bb]];
            }
            for bb in 0..p_sigma {
                hj[[a, p_mu + bb]] += ml * xa * rec.x_sigma[[i, bb]];
                hj[[p_mu + bb, a]] += ml * xa * rec.x_sigma[[i, bb]];
            }
        }
        for a in 0..p_sigma {
            let xa = rec.x_sigma[[i, a]];
            for bb in 0..p_sigma {
                hj[[p_mu + a, p_mu + bb]] += ll * xa * rec.x_sigma[[i, bb]];
            }
        }
    }
    for a in 0..p_mu {
        for bb in 0..p_mu {
            hj[[a, bb]] += s_mu_scaled[[a, bb]];
        }
    }
    for a in 0..p_sigma {
        for bb in 0..p_sigma {
            hj[[p_mu + a, p_mu + bb]] += rec.s_sigma[[a, bb]];
        }
    }
    // The observed joint Hessian can be indefinite. Production prices the LAML
    // −½log|H| via a SPECTRAL pseudo-logdet with an eigenvalue floor
    // (DenseSpectralOperator, PseudoLogdetMode::Smooth), NOT a Cholesky. Compute
    // three variants so we can see which one reproduces the production λ̂_μ:
    //   * chol/jitter (assumes PD),
    //   * abs-eigenvalue Σ log|λ|,
    //   * positive floor Σ log(max(λ, floor)).
    let logdet_h_joint = logdet_spd(&hj);
    let hj_eigs = symmetric_eigenvalues(&hj);
    let max_eig = hj_eigs.iter().cloned().fold(0.0f64, |m, v| m.max(v.abs()));
    let floor = 1e-8 * max_eig.max(1.0);
    let logdet_h_abs: f64 = hj_eigs.iter().map(|&l| l.abs().max(floor).ln()).sum();
    let logdet_h_floor: f64 = hj_eigs.iter().map(|&l| l.max(floor).ln()).sum();
    let hj_min_eig = hj_eigs.iter().cloned().fold(f64::INFINITY, f64::min);
    let hj_neg_count = hj_eigs.iter().filter(|&&l| l < -floor).count();

    // μ-block log-det.
    let mut hmumu_pen = prof.hmumu.clone();
    hmumu_pen += &s_mu_scaled;
    let logdet_hmumu = logdet_spd(&hmumu_pen);

    // Nested σ curvature A_σ = −∂g_N/∂β_σ by central finite differences.
    let logdet_a_sigma = {
        let mut a = Array2::<f64>::zeros((p_sigma, p_sigma));
        let scl = beta_sigma.iter().map(|v| v.abs()).fold(1.0f64, f64::max);
        let eps = 1e-6 * scl;
        for j in 0..p_sigma {
            let mut bp = beta_sigma.clone();
            bp[j] += eps;
            let pr = profile_mu(rec, &bp, &s_mu_scaled);
            let gp = sigma_gradient(rec, &pr, &bp, use_h);
            let mut bm = beta_sigma.clone();
            bm[j] -= eps;
            let prm = profile_mu(rec, &bm, &s_mu_scaled);
            let gm = sigma_gradient(rec, &prm, &bm, use_h);
            for i in 0..p_sigma {
                a[[i, j]] = -(gp[i] - gm[i]) / (2.0 * eps);
            }
        }
        // symmetrize
        let mut sym = Array2::<f64>::zeros((p_sigma, p_sigma));
        for i in 0..p_sigma {
            for j in 0..p_sigma {
                sym[[i, j]] = 0.5 * (a[[i, j]] + a[[j, i]]);
            }
        }
        logdet_spd(&sym)
    };

    SolveOut {
        beta_mu: prof.beta_mu,
        beta_sigma,
        edf_mu: prof.edf_mu,
        neg_ll,
        penalty_quad,
        logdet_h_joint,
        logdet_h_abs,
        logdet_h_floor,
        hj_min_eig,
        hj_neg_count,
        logdet_hmumu,
        logdet_a_sigma,
    }
}

/// Full joint-LAML criterion V at an INDEPENDENT per-μ-penalty log-scale vector
/// `rho` (λ_μk = λ̂_μk·e^{rho_k}), λ_σ fixed. Includes the exact −½log|S_μ|₊
/// pseudo-determinant so the value is comparable across the 2-D penalty space.
/// Returns (V, edf_μ, converged β_σ).
fn full_v_joint(
    rec: &Reconstructed,
    rho: &[f64],
    bs0: &Array1<f64>,
) -> (f64, f64, Array1<f64>) {
    let p_mu = rec.x_mu.ncols();
    let mut s_mu = Array2::<f64>::zeros((p_mu, p_mu));
    for (k, blk) in rec.s_mu_blocks.iter().enumerate() {
        s_mu.scaled_add(rho[k].exp(), blk);
    }
    let out = solve_mode_with_penalty(rec, &s_mu, false, bs0);
    let logdet_s_mu = logdet_pseudo(&s_mu);
    // V = −ℓ + ½‖β‖²_S + ½log|H_joint+S| − ½log|S_μ|₊  (drop ρ-indep σ prior term)
    let v = out.neg_ll + 0.5 * out.penalty_quad + 0.5 * out.logdet_h_joint - 0.5 * logdet_s_mu;
    (v, out.edf_mu, out.beta_sigma)
}

fn grid_mu_rmse(rec: &Reconstructed, beta_mu: &Array1<f64>) -> f64 {
    let pred = rec.x_grid.dot(beta_mu) + &rec.o_grid;
    rmse(&pred.to_vec(), &rec.truth_mu_grid)
}

fn run_arm(arm: &Arm) {
    eprintln!(
        "\n============ #2356 nested-criterion test: {} arm ============",
        arm.label
    );
    let rec = reconstruct(arm);
    eprintln!(
        "[{}] n={} p_mu={} p_sigma={} c={:.5} λ̂_μ={:?} λ̂_σ={:?} edf_μ(fit)={:.3} r_μ={}",
        rec.label,
        rec.n,
        rec.x_mu.ncols(),
        rec.x_sigma.ncols(),
        rec.c,
        rec.lambda_mu,
        rec.lambda_sigma,
        rec.edf_mu,
        rec.r_mu
    );

    // ---- ρ=0 reconstruction validity: joint solve must reproduce production ----
    let j0 = solve_mode(&rec, 0.0, false, &rec.beta_sigma);
    let dmu = j0
        .beta_mu
        .iter()
        .zip(rec.beta_mu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let dsg = j0
        .beta_sigma
        .iter()
        .zip(rec.beta_sigma.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let mu_scale = rec.beta_mu.iter().map(|v| v.abs()).fold(1e-12, f64::max);
    let sg_scale = rec.beta_sigma.iter().map(|v| v.abs()).fold(1e-12, f64::max);
    eprintln!(
        "[{}] ρ=0 joint reconstruction: ‖Δβ_μ‖∞={:.2e} (rel {:.2e})  ‖Δβ_σ‖∞={:.2e} (rel {:.2e})  edf_μ={:.3}",
        rec.label,
        dmu,
        dmu / mu_scale,
        dsg,
        dsg / sg_scale,
        j0.edf_mu
    );

    // ---- report the dropped-term magnitude at the production mode -------------
    let prof0 = profile_mu(&rec, &rec.beta_sigma, &rec.s_mu);
    let sum_h: f64 = prof0.h_lev.sum();
    eprintln!(
        "[{}] Σ h_i at production σ̂ = {:.3}  (should ≈ edf_μ = {:.3})  → the dropped REML dof",
        rec.label, sum_h, prof0.edf_mu
    );

    // ---- is production a STATIONARY point of its OWN joint-LAML criterion? -----
    // Independent per-penalty central differences of V at ρ = 0 (production).
    let n_pen = rec.s_mu_blocks.len();
    let base_rho = vec![0.0f64; n_pen];
    let (v0, edf0, _) = full_v_joint(&rec, &base_rho, &rec.beta_sigma);
    let mut grad = vec![0.0f64; n_pen];
    let eps = 1e-3;
    for k in 0..n_pen {
        let mut rp = base_rho.clone();
        rp[k] += eps;
        let (vp, _, _) = full_v_joint(&rec, &rp, &rec.beta_sigma);
        let mut rm = base_rho.clone();
        rm[k] -= eps;
        let (vm, _, _) = full_v_joint(&rec, &rm, &rec.beta_sigma);
        grad[k] = (vp - vm) / (2.0 * eps);
    }
    let gnorm = grad.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    eprintln!(
        "[{}] production stationarity: V(ρ=0)={v0:.4} edf_μ={edf0:.3}  ∂V/∂ρ_μk={:?}  ‖∇‖∞={gnorm:.4}",
        rec.label,
        grad.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>()
    );
    eprintln!(
        "[{}]   → if ‖∇‖∞≈0 production IS at its criterion's stationary point (criterion prefers this edf_μ);\n[{}]     if ‖∇‖∞≫0 the OPTIMIZER stopped short of the criterion argmin.",
        rec.label, rec.label
    );

    // ---- sweep ρ_μ; evaluate both criteria ------------------------------------
    let n_grid = 33usize;
    let rho_lo = -4.0;
    let rho_hi = 4.0;
    // (V, ρ, edf_μ, rmse) argmins for: joint w/ Cholesky logdet, joint w/ ABS
    // pseudo-logdet, joint w/ positive-FLOOR pseudo-logdet, and the NESTED
    // (h-corrected) criterion w/ ABS pseudo-logdet.
    let mut best_chol = (f64::INFINITY, 0.0f64, 0.0f64, 0.0f64);
    let mut best_abs = (f64::INFINITY, 0.0f64, 0.0f64, 0.0f64);
    let mut best_floor = (f64::INFINITY, 0.0f64, 0.0f64, 0.0f64);
    let mut best_nested = (f64::INFINITY, 0.0f64, 0.0f64, 0.0f64);
    let mut bs_joint = rec.beta_sigma.clone();
    let mut bs_nested = rec.beta_sigma.clone();
    eprintln!(
        "[{}]  ρ_μ | edf_μ  V_chol   V_abs    V_floor  minEig  #neg | NEST edf_μ V_nest_abs rmse_n",
        rec.label
    );
    for g in 0..n_grid {
        let rho = rho_lo + (rho_hi - rho_lo) * (g as f64) / ((n_grid - 1) as f64);
        let j = solve_mode(&rec, rho, false, &bs_joint);
        let nst = solve_mode(&rec, rho, true, &bs_nested);
        bs_joint = j.beta_sigma.clone();
        bs_nested = nst.beta_sigma.clone();
        let base = j.neg_ll + 0.5 * j.penalty_quad - 0.5 * (rec.r_mu as f64) * rho;
        let v_chol = base + 0.5 * j.logdet_h_joint;
        let v_abs = base + 0.5 * j.logdet_h_abs;
        let v_floor = base + 0.5 * j.logdet_h_floor;
        // Nested criterion, ABS pseudo-logdet on the μ-block + σ curvature.
        let v_nested = nst.neg_ll
            + 0.5 * nst.penalty_quad
            + 0.5 * nst.logdet_hmumu
            + 0.5 * nst.logdet_a_sigma
            - 0.5 * (rec.r_mu as f64) * rho;
        let rmse_j = grid_mu_rmse(&rec, &j.beta_mu);
        let rmse_n = grid_mu_rmse(&rec, &nst.beta_mu);
        assert!(
            v_chol.is_finite() && v_abs.is_finite() && v_floor.is_finite() && v_nested.is_finite(),
            "non-finite criterion at ρ={rho}"
        );
        if v_chol < best_chol.0 {
            best_chol = (v_chol, rho, j.edf_mu, rmse_j);
        }
        if v_abs < best_abs.0 {
            best_abs = (v_abs, rho, j.edf_mu, rmse_j);
        }
        if v_floor < best_floor.0 {
            best_floor = (v_floor, rho, j.edf_mu, rmse_j);
        }
        if v_nested < best_nested.0 {
            best_nested = (v_nested, rho, nst.edf_mu, rmse_n);
        }
        if g % 2 == 0 {
            eprintln!(
                "[{}] {rho:5.2} | {:6.3} {v_chol:8.3} {v_abs:8.3} {v_floor:8.3} {:7.3} {:4} | {:6.3} {v_nested:9.3} {rmse_n:.5}",
                rec.label, j.edf_mu, j.hj_min_eig, j.hj_neg_count, nst.edf_mu
            );
        }
    }
    eprintln!(
        "[{}] ARGMIN V_chol (PD assume): ρ*={:+.2} edf_μ*={:.3} μ-RMSE*={:.5}",
        rec.label, best_chol.1, best_chol.2, best_chol.3
    );
    eprintln!(
        "[{}] ARGMIN V_abs  (|λ| floor): ρ*={:+.2} edf_μ*={:.3} μ-RMSE*={:.5}  ← closest to production LAML",
        rec.label, best_abs.1, best_abs.2, best_abs.3
    );
    eprintln!(
        "[{}] ARGMIN V_floor(max floor): ρ*={:+.2} edf_μ*={:.3} μ-RMSE*={:.5}",
        rec.label, best_floor.1, best_floor.2, best_floor.3
    );
    eprintln!(
        "[{}] ARGMIN V_nested (h-corr): ρ*={:+.2} edf_μ*={:.3} μ-RMSE*={:.5}",
        rec.label, best_nested.1, best_nested.2, best_nested.3
    );

    // validity asserts only
    assert!(
        dmu / mu_scale < 1e-2 && dsg / sg_scale < 5e-2,
        "[{}] ρ=0 joint solve did not reproduce production β̂ (relΔβ_μ={:.2e}, relΔβ_σ={:.2e}); reconstruction invalid",
        rec.label,
        dmu / mu_scale,
        dsg / sg_scale
    );
    assert!(
        best_chol.0.is_finite()
            && best_abs.0.is_finite()
            && best_floor.0.is_finite()
            && best_nested.0.is_finite(),
        "[{}] non-finite criterion argmin",
        rec.label
    );
}

#[test]
fn zz_measure_1561_nested_vs_joint_criterion() {
    init_parallelism();
    eprintln!(
        "=== #2356 DECISIVE: does the NESTED μ-marginal criterion move λ̂_μ toward the oracle? ===\n\
         Fix λ_σ at production; sweep λ_μ=λ̂_μ·e^{{ρ_μ}}. At each ρ_μ solve the JOINT mode\n\
         (σ-score drops h) and the NESTED mode (σ-score += h_i, Σh_i=edf_μ), evaluate each\n\
         criterion, and report argmin. Prediction: V_joint argmin≈0 (reproduces production),\n\
         V_nested argmin at ρ_μ>0 with edf_μ falling toward the oracle (5.8 cyclic / 13.7 plain)."
    );
    run_arm(&cyclic_arm());
    run_arm(&plain_arm());
}
