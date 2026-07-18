//! zz_measure DIAGNOSTIC (#1561): oracle λ_μ sweep for the Gaussian
//! location-scale MEAN (μ) channel — the top-ranked quality loser.
//!
//! Why this test exists
//! --------------------
//! The QualityPair meta-gate (bench/aggregate_quality_gate_1561.py, Wilcoxon+BH
//! over the retrofitted suite) ranked the two worst reference-quality losers as
//! the SAME family pair: gam over-smooths the MEAN of a Gaussian location-scale
//! fit relative to gamlss (μ-RMSE 1.67× on the plain loc-scale arm, 2.4× on the
//! cyclic arm) while matching or beating gamlss on the log-σ block. The root
//! cause was narrowed to a μ-channel λ-selection defect: gam's JOINT WPS-2016
//! observed-Hessian LAML picks (λ_μ, λ_σ) together, and its λ̂_μ is suspected to
//! land ABOVE the μ-accuracy optimum, whereas gamlss's RS/CG picks λ per
//! parameter by local ML.
//!
//! This is the DECISIVE experiment for that hypothesis. For a Gaussian
//! location-scale model the μ block is exactly linear-Gaussian GIVEN σ: at the
//! converged fit β̂_μ solves, exactly,
//!
//!     (XᵀWX + Σ_k λ̂_k S_k) β̂_μ = XᵀW(y − offset),   W = diag(1/σ̂²).
//!
//! So with the converged σ̂ FROZEN into the weights we can re-solve that same
//! penalized WLS at any μ-penalty scale exp(ρ)·λ̂ and read off the μ-RMSE-to-
//! truth at each ρ. ρ = 0 reproduces the production μ solve BY CONSTRUCTION
//! (we assert β_μ(ρ=0) reproduces the fit's β̂_μ — that single check validates
//! the whole reconstruction: design basis, penalty matrices, weights, offset,
//! AND the λ̂-to-block mapping). The argmin ρ* is then the amount of extra/less
//! μ penalty the ACCURACY optimum wants relative to REML's joint choice.
//!
//! Decision rule (see PATCH_NOTES.md for the full licence)
//! ------------------------------------------------------
//!   * ρ* ≈ 0 (the REML λ̂_μ already sits at the μ-RMSE optimum, and no ρ in the
//!     sweep buys a materially lower μ-RMSE) ⇒ the μ channel is EXONERATED; the
//!     quality loss lives elsewhere (basis, σ block, or the gamlss comparator).
//!   * ρ* ≪ 0 with a material μ-RMSE gain (REML's λ̂_μ is well above the
//!     accuracy-optimal λ*_μ) ⇒ joint-LAML μ over-smoothing CONFIRMED and
//!     quantified by (−ρ*) log-units and the RMSE gain.
//!
//! A SECOND sweep freezes the TRUE σ(x) into the weights instead of gam's σ̂.
//! Comparing ρ*(σ̂) against ρ*(σ_true) isolates hypothesis H2 (gam's more
//! flexible — hence noisier — σ̂ making the 1/σ̂² weights noisy, which the joint
//! criterion could compensate for by raising λ_μ): if ρ*(σ_true) ≈ ρ*(σ̂) the
//! weight noise is NOT the driver; if they differ, H2 is live.
//!
//! Running BOTH the cyclic and the plain loc-scale datasets lets the reader
//! compare the (λ̂ − λ*) gap cyclic-vs-plain (hypothesis H3: a cyclic-only
//! penalty rank / pseudo-logdet over-count inflating λ_μ on the cyclic arm).
//!
//! zz_measure discipline (cf. feedback_zz_measure_diagnostic_tests): the numbers
//! are eprintln'd; the ONLY hard asserts are (a) finiteness and (b) the ρ=0
//! reconstruction fidelity that makes the oracle valid. The over-smoothing
//! MAGNITUDE is reported, never gated, so this never becomes a flaky bar.

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

/// gam's location-scale noise link floor: raw σ = response_scale·FLOOR + exp(η_σ)
/// (the response-relative soft floor is part of the saved fit contract; see the
/// plain loc-scale quality test and probe_1561_locscale_lambda.rs).
const LOGB_SIGMA_FLOOR: f64 = 0.01;

// ===========================================================================
// Deterministic data generators — bit-identical to the two quality losers.
// ===========================================================================

/// Box–Muller standard normals from a 64-bit LCG. IDENTICAL to the stream in
/// tests/quality/families/quality_vs_gamlss_gaussian_location_scale_cyclic.rs.
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

/// LCG unit draw used by probe_1561_locscale_lambda.rs's plain generator.
fn next_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// One dataset + the exact formulas the quality loser uses.
struct Arm {
    label: &'static str,
    /// Column headers in dataset order (the x column is looked up by name).
    headers: Vec<String>,
    x_name: &'static str,
    mean_formula: &'static str,
    noise_formula: &'static str,
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Dense evaluation grid (where μ-RMSE-to-truth is measured, matching the
    /// quality test's off-training-point shape comparison).
    grid_x: Vec<f64>,
    /// True mean and true σ on `grid_x` (truth recovery targets), and true σ on
    /// the TRAINING `xs` (for the frozen-truth-σ weight sweep).
    truth_mu_grid: Vec<f64>,
    truth_sigma_train: Vec<f64>,
}

/// Cyclic loc-scale arm (worst loser, 2.4×): x∈[0,2π] len 150, μ*=sin x,
/// σ*=0.15+0.1cos x, seed 123. Formulas pin the period to [0, 2π].
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
    let truth_sigma_train: Vec<f64> = xs.iter().map(|&x| (0.15 + 0.1 * x.cos()).abs()).collect();
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
        truth_sigma_train,
    }
}

/// Plain loc-scale arm (2nd worst loser, 1.67×): x sorted U(0,1) n=200,
/// μ*=sin 2πx, σ*=0.1+0.2 sin 2πx, seed 42 — the probe_1561 recipe.
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
    let truth_sigma_train: Vec<f64> = xs
        .iter()
        .map(|&x| (0.1 + 0.2 * (two_pi * x).sin()).abs())
        .collect();
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
        truth_sigma_train,
    }
}

// ===========================================================================
// Small dense SPD linear algebra (systems are p ≈ 11..13; hand-rolled so the
// diagnostic carries no linalg-crate dependency, matching the probe's style).
// ===========================================================================

/// Lower-triangular Cholesky factor L with A = L Lᵀ, or None if not PD.
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

/// Solve A x = b for SPD A given its Cholesky factor L.
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

/// Cholesky factor of A with a tiny adaptive jitter fallback (kept minuscule so
/// it never perturbs the reconstruction; only rescues borderline conditioning).
fn chol_with_jitter(a: &Array2<f64>) -> Array2<f64> {
    if let Some(l) = chol_lower(a) {
        return l;
    }
    let p = a.nrows();
    let scale: f64 = (0..p).map(|i| a[[i, i]].abs()).sum::<f64>() / (p as f64);
    let mut jitter = scale.max(1.0) * 1e-12;
    for _ in 0..40 {
        let mut aj = a.clone();
        for i in 0..p {
            aj[[i, i]] += jitter;
        }
        if let Some(l) = chol_lower(&aj) {
            return l;
        }
        jitter *= 10.0;
    }
    panic!("penalized normal matrix not factorizable even with jitter");
}

/// tr(A⁻¹ B) via columnwise SPD solves (used for effective degrees of freedom).
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

// ===========================================================================
// The oracle sweep for one arm.
// ===========================================================================

fn run_arm(arm: Arm) {
    eprintln!(
        "\n================ #1561 μ-oracle sweep: {} arm ================",
        arm.label
    );

    // ---- fit exactly as the quality loser does --------------------------------
    let n = arm.xs.len();
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            // Emit columns in the header order this arm declares.
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
    let result = fit_from_formula(arm.mean_formula, &ds, &cfg).expect("gam location-scale fit");
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult {
        fit,
        response_scale,
        ..
    }) = result
    else {
        panic!("expected a GaussianLocationScale fit");
    };
    let c = response_scale;

    let loc_block = fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location block");
    let scale_block = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale block");
    let beta_mu = loc_block.beta.clone();
    let beta_sigma = scale_block.beta.clone();
    let lambda_mu: Vec<f64> = loc_block.lambdas.to_vec();
    let lambda_sigma: Vec<f64> = scale_block.lambdas.to_vec();
    let edf_mu = loc_block.edf;
    let edf_sigma = scale_block.edf;

    // ---- rebuild the μ design + penalties at the TRAINING points --------------
    // build_term_collection_design returns the design AND the penalties in the
    // SAME basis as β̂_μ (asserted below), so no basis reconstruction risk.
    let mut train_grid = Array2::<f64>::zeros((n, ncols));
    for (i, &x) in arm.xs.iter().enumerate() {
        train_grid[[i, x_idx]] = x;
    }
    let mean_design = build_term_collection_design(train_grid.view(), &fit.meanspec_resolved)
        .expect("rebuild mean design at training points");
    let noise_design = build_term_collection_design(train_grid.view(), &fit.noisespec_resolved)
        .expect("rebuild noise design at training points");
    let p = mean_design.design.ncols();
    assert_eq!(
        p,
        beta_mu.len(),
        "mean design width {} must equal μ coefficient count {}",
        p,
        beta_mu.len()
    );
    assert_eq!(
        mean_design.penalties.len(),
        lambda_mu.len(),
        "rebuilt μ penalty count {} must equal the fit's μ-block λ count {} \
         (else the λ̂→penalty mapping is wrong)",
        mean_design.penalties.len(),
        lambda_mu.len()
    );

    // Materialize X_μ (n×p) column by column via the linear operator.
    let mut x_train = Array2::<f64>::zeros((n, p));
    for j in 0..p {
        let mut e = Array1::<f64>::zeros(p);
        e[j] = 1.0;
        let col_j = mean_design.design.apply(&e);
        x_train.column_mut(j).assign(&col_j);
    }
    let mu_offset_train = mean_design.affine_offset.clone();

    // Embed each μ penalty S_k into a full p×p block (local + col_range).
    let embedded_penalties: Vec<Array2<f64>> = mean_design
        .penalties
        .iter()
        .map(|bp| {
            let mut s = Array2::<f64>::zeros((p, p));
            for (a, gi) in bp.col_range.clone().enumerate() {
                for (b, gj) in bp.col_range.clone().enumerate() {
                    s[[gi, gj]] = bp.local[[a, b]];
                }
            }
            s
        })
        .collect();

    // ---- frozen σ̂ → weights W = 1/σ̂² (raw units) -----------------------------
    let eta_sigma_train = noise_design.design.apply(&beta_sigma) + &noise_design.affine_offset;
    let sigma_hat: Vec<f64> = eta_sigma_train
        .iter()
        .map(|&e| c * LOGB_SIGMA_FLOOR + e.exp())
        .collect();
    let w_hat: Vec<f64> = sigma_hat.iter().map(|&s| 1.0 / (s * s)).collect();
    // Alternative weights from the TRUE σ(x) (isolates H2 — weight noise).
    let w_true: Vec<f64> = arm
        .truth_sigma_train
        .iter()
        .map(|&s| 1.0 / (s.max(1e-3)).powi(2))
        .collect();

    // ---- μ design on the dense evaluation grid (μ-RMSE-to-truth) --------------
    let mg = arm.grid_x.len();
    let mut eval_grid = Array2::<f64>::zeros((mg, ncols));
    for (i, &gx) in arm.grid_x.iter().enumerate() {
        eval_grid[[i, x_idx]] = gx;
    }
    let mean_design_grid = build_term_collection_design(eval_grid.view(), &fit.meanspec_resolved)
        .expect("rebuild mean design at eval grid");
    let mut x_grid = Array2::<f64>::zeros((mg, p));
    for j in 0..p {
        let mut e = Array1::<f64>::zeros(p);
        e[j] = 1.0;
        let col_j = mean_design_grid.design.apply(&e);
        x_grid.column_mut(j).assign(&col_j);
    }
    let mu_offset_grid = mean_design_grid.affine_offset.clone();

    // ---- fixed WLS pieces given a weight vector -------------------------------
    // Base μ penalty in RAW space that reproduces β̂_μ at ρ=0: Σ_k (λ̂_k / c²) S_k
    // (derivation: production selects λ̂ on the c-standardized response; the
    // returned β̂_μ solves the raw-space penalized WLS with penalty λ̂_k / c²).
    let base_penalty = |scale_by: f64| -> Array2<f64> {
        let mut s = Array2::<f64>::zeros((p, p));
        for (k, sk) in embedded_penalties.iter().enumerate() {
            let coef = (lambda_mu[k] / (c * c)) * scale_by;
            s.scaled_add(coef, sk);
        }
        s
    };

    // Weighted normal-equation pieces XᵀWX and XᵀW(y−offset) for a weight set.
    let build_xtwx_xtwr = |w: &[f64]| -> (Array2<f64>, Array1<f64>) {
        let mut xtwx = Array2::<f64>::zeros((p, p));
        let mut xtwr = Array1::<f64>::zeros(p);
        for i in 0..n {
            let wi = w[i];
            let ri = arm.ys[i] - mu_offset_train[i];
            for a in 0..p {
                let xa = x_train[[i, a]];
                xtwr[a] += wi * xa * ri;
                for b in 0..p {
                    xtwx[[a, b]] += wi * xa * x_train[[i, b]];
                }
            }
        }
        (xtwx, xtwr)
    };

    // Solve β(ρ), return (β, edf) for a weight set at μ-penalty scale exp(ρ).
    let solve_at = |xtwx: &Array2<f64>, xtwr: &Array1<f64>, rho: f64| -> (Array1<f64>, f64) {
        let mut a = xtwx.clone();
        a += &base_penalty(rho.exp());
        let l = chol_with_jitter(&a);
        let beta = chol_solve_vec(&l, xtwr);
        let edf = trace_ainv_b(&l, xtwx);
        (beta, edf)
    };

    // μ-RMSE-to-truth on the eval grid for a coefficient vector.
    let grid_rmse = |beta: &Array1<f64>| -> f64 {
        let pred = x_grid.dot(beta) + &mu_offset_grid;
        let pred_v: Vec<f64> = pred.to_vec();
        rmse(&pred_v, &arm.truth_mu_grid)
    };

    // ---- report the fit's own selection --------------------------------------
    let gam_actual_rmse = grid_rmse(&beta_mu);
    let l1_offset: f64 = mu_offset_train.iter().map(|v| v.abs()).sum::<f64>() / (n as f64);
    eprintln!(
        "[{}] fit: n={n} p_mu={p} response_scale(c)={c:.6} \
         λ̂_μ={:?} edf_μ={edf_mu:.4} λ̂_σ={:?} edf_σ={edf_sigma:.4} \
         outer_iters={} mean_affine_offset_L1={l1_offset:.2e} gam_μ_rmse={gam_actual_rmse:.5}",
        arm.label, lambda_mu, lambda_sigma, fit.fit.outer_iterations
    );

    // ---- ρ=0 reconstruction self-check (VALIDITY GUARD) -----------------------
    // Reproduces the production μ solve from the frozen-σ̂ WLS. If this matches
    // β̂_μ, every ingredient (basis, S_k, weights, offset, λ̂ mapping, the c²
    // scaling) is correct and the sweep is trustworthy.
    let (xtwx_hat, xtwr_hat) = build_xtwx_xtwr(&w_hat);
    let (beta_recon, edf_recon) = solve_at(&xtwx_hat, &xtwr_hat, 0.0);
    let recon_linf = beta_recon
        .iter()
        .zip(beta_mu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let beta_scale: f64 = beta_mu.iter().map(|v| v.abs()).fold(0.0f64, f64::max).max(1e-12);
    let recon_rel = recon_linf / beta_scale;
    eprintln!(
        "[{}] ρ=0 reconstruction: ‖β_μ(0)−β̂_μ‖∞={recon_linf:.3e} (rel {recon_rel:.3e}) \
         edf_μ(0)={edf_recon:.4} vs fit edf_μ={edf_mu:.4}  \
         rmse_μ(0)={:.5} vs gam_μ_rmse={gam_actual_rmse:.5}",
        arm.label,
        grid_rmse(&beta_recon)
    );

    // ---- sweep ρ ∈ [−8, 8] under BOTH weight sets -----------------------------
    let (xtwx_true, xtwr_true) = build_xtwx_xtwr(&w_true);
    let n_grid = 65usize; // step 0.25
    let rho_lo = -8.0;
    let rho_hi = 8.0;

    let mut best_hat = (f64::INFINITY, 0.0f64, 0.0f64); // (rmse, rho, edf)
    let mut best_true = (f64::INFINITY, 0.0f64, 0.0f64);
    eprintln!(
        "[{}]   rho   |  frozen-σ̂: edf_μ    rmse_μ  |  truth-σ: edf_μ    rmse_μ",
        arm.label
    );
    for g in 0..n_grid {
        let rho = rho_lo + (rho_hi - rho_lo) * (g as f64) / ((n_grid - 1) as f64);
        let (beta_h, edf_h) = solve_at(&xtwx_hat, &xtwr_hat, rho);
        let (beta_t, edf_t) = solve_at(&xtwx_true, &xtwr_true, rho);
        let rmse_h = grid_rmse(&beta_h);
        let rmse_t = grid_rmse(&beta_t);
        assert!(
            rmse_h.is_finite() && rmse_t.is_finite(),
            "non-finite μ-RMSE at rho={rho}"
        );
        if rmse_h < best_hat.0 {
            best_hat = (rmse_h, rho, edf_h);
        }
        if rmse_t < best_true.0 {
            best_true = (rmse_t, rho, edf_t);
        }
        // Print a readable subset (every 4th node) plus always the endpoints.
        if g % 4 == 0 || g == n_grid - 1 {
            eprintln!(
                "[{}] {rho:6.2} |            {edf_h:6.3}  {rmse_h:.5}  |          {edf_t:6.3}  {rmse_t:.5}",
                arm.label
            );
        }
    }

    // ---- decision readout -----------------------------------------------------
    let gain_hat = gam_actual_rmse - best_hat.0;
    let gain_rel = if gam_actual_rmse > 0.0 {
        gain_hat / gam_actual_rmse
    } else {
        0.0
    };
    eprintln!(
        "[{}] ARGMIN frozen-σ̂: rho*={:.2} (μ-penalty ×{:.3e} vs REML) edf_μ*={:.3} rmse_μ*={:.5} \
         | REML rmse_μ(0)={gam_actual_rmse:.5} | GAIN={gain_hat:.5} ({:.1}% of REML μ-RMSE)",
        arm.label,
        best_hat.1,
        best_hat.1.exp(),
        best_hat.2,
        best_hat.0,
        100.0 * gain_rel
    );
    eprintln!(
        "[{}] ARGMIN truth-σ : rho*={:.2} (μ-penalty ×{:.3e} vs REML) edf_μ*={:.3} rmse_μ*={:.5} \
         [ρ=0 here is NOT the production solve — weights swapped to 1/σ_true²]",
        arm.label,
        best_true.1,
        best_true.1.exp(),
        best_true.2,
        best_true.0
    );
    // ρ = 0 is the REML μ solve, so the SIGN of ρ* reads the miscalibration
    // direction: ρ* < 0 ⇒ accuracy wants LESS penalty (REML over-smooths μ);
    // ρ* > 0 ⇒ accuracy wants MORE penalty (REML under-smooths μ).
    let verdict = if best_hat.1.abs() < 0.75 || gain_rel < 0.05 {
        "μ channel EXONERATED (REML λ̂_μ ≈ μ-accuracy optimum; no material μ-RMSE on the table)"
    } else if best_hat.1 <= -0.75 && gain_rel >= 0.15 {
        "μ OVER-SMOOTHING (REML λ̂_μ ABOVE the μ-accuracy optimum — accuracy wants LESS penalty)"
    } else if best_hat.1 >= 0.75 && gain_rel >= 0.15 {
        "μ UNDER-SMOOTHING (REML λ̂_μ BELOW the μ-accuracy optimum — accuracy wants MORE penalty)"
    } else {
        "MILD μ MISCALIBRATION (ρ* nonzero, gain < 15%; direction = sign of ρ*)"
    };
    let h2 = if (best_hat.1 - best_true.1).abs() < 0.75 {
        "H2 UNLIKELY (σ̂ vs σ_true optima agree — weight noise not the driver)"
    } else {
        "H2 LIVE (σ̂ and σ_true optima disagree — weight noise contributes)"
    };
    eprintln!("[{}] VERDICT: {verdict}", arm.label);
    eprintln!("[{}] H2 read: {h2}", arm.label);

    // The ONLY validity assert: the frozen-σ̂ WLS at ρ=0 must reproduce the
    // production μ solve, otherwise the whole oracle is meaningless. Loose enough
    // to tolerate solver tolerance / floor granularity, tight enough to catch a
    // wrong basis, penalty, weight, offset, or λ̂→block mapping.
    assert!(
        recon_rel < 5e-3,
        "[{}] ρ=0 reconstruction did not reproduce β̂_μ (rel ‖·‖∞={recon_rel:.3e}) — \
         the oracle's design/penalty/weight/λ̂-mapping is wrong; sweep is INVALID",
        arm.label
    );
    assert!(
        gam_actual_rmse.is_finite() && best_hat.0.is_finite(),
        "[{}] non-finite μ-RMSE",
        arm.label
    );
}

#[test]
fn zz_measure_1561_mu_channel_oracle_lambda_sweep() {
    init_parallelism();
    eprintln!(
        "=== #1561 DECISIVE EXPERIMENT: does joint-LAML over-smooth the μ channel? ===\n\
         Freeze σ̂ into W=1/σ̂², sweep the μ penalty exp(ρ)·λ̂_μ, read μ-RMSE-to-truth.\n\
         ρ=0 ≡ the production REML μ solve (validated by reconstruction). ρ*<0 ⇒ REML\n\
         over-penalizes μ. A second sweep uses the TRUE σ(x) weights to test H2."
    );
    run_arm(cyclic_arm());
    run_arm(plain_arm());
}

// ===========================================================================
// #1561 sigma-clamp criterion probe: does the criterion's λ̂_μ move toward the
// oracle when σ is CLAMPED at the truth? — testing the σ̂-feedback hypothesis.
//
// The sibling sweep proved joint-LAML picks λ̂_μ too LOW (μ under-smoothed) with
// ~20-30% μ-RMSE recoverable, and that noisy 1/σ̂² weights don't move the ORACLE
// optimum. The open question is WHY the criterion under-selects. The leading
// hypothesis: the joint observed μ-block information H_μμ = Xᵀdiag(a/σ̂²)X is
// inflated where gam's flexible σ̂ dips below σ_true, overstating the μ-EDF in
// −½log|H+S| and dragging λ̂_μ down. This experiment clamps σ into the WEIGHTS
// the criterion actually selects against, using production fits only:
//
//   J. joint location-scale (σ̂, 2-block LAML)          — the shipped baseline
//   A. marginal Gaussian, prior weights a=1/σ̂²          — same weights, 1-block REML
//   B. marginal Gaussian, prior weights a=1/σ_true²      — σ clamped at truth
//
// (A vs J) isolates the 2-block→1-block criterion structure at fixed weights;
// (B vs A) isolates the σ̂-noise-in-weights effect — the hypothesis proper.
// If clamping σ→truth (B) makes the criterion select MORE μ smoothing (lower
// edf_μ) and lower μ-RMSE than A, the σ̂-feedback term is implicated; if B ≈ A,
// the weight noise is exonerated and the under-selection is intrinsic to the
// marginal criterion's μ-EDF accounting (next candidate). Weights enter as
// Gaussian prior weights via `FitConfig::weight_column`; a plain `s(x)` Gaussian
// stays on the dense `FitResult::Standard` path (tp/cc bases are not the
// BSpline1D single-penalty shape the O(n) scan fast-path claims).
// ===========================================================================

/// Joint location-scale fit: return (σ̂ on training rows, edf_μ, λ̂_μ, μ-RMSE).
fn fit_joint_sigma_and_mu(arm: &Arm) -> (Vec<f64>, f64, Vec<f64>, f64) {
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
    let ds = encode_recordswith_inferred_schema(arm.headers.clone(), rows).expect("encode joint");
    let col = ds.column_map();
    let x_idx = col[arm.x_name];
    let ncols = ds.headers.len();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some(arm.noise_formula.to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(arm.mean_formula, &ds, &cfg).expect("joint loc-scale fit");
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult {
        fit,
        response_scale,
        ..
    }) = result
    else {
        panic!("expected a GaussianLocationScale fit");
    };
    let c = response_scale;
    let loc = fit.fit.block_by_role(BlockRole::Location).expect("location");
    let beta_mu = loc.beta.clone();
    let edf_mu = loc.edf;
    let lambda_mu = loc.lambdas.to_vec();
    let beta_sigma = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale")
        .beta
        .clone();

    // σ̂ on the training rows (raw units).
    let mut train_grid = Array2::<f64>::zeros((n, ncols));
    for (i, &x) in arm.xs.iter().enumerate() {
        train_grid[[i, x_idx]] = x;
    }
    let noise_design = build_term_collection_design(train_grid.view(), &fit.noisespec_resolved)
        .expect("noise design");
    let eta_sigma = noise_design.design.apply(&beta_sigma) + &noise_design.affine_offset;
    let sigma_hat: Vec<f64> = eta_sigma
        .iter()
        .map(|&e| c * LOGB_SIGMA_FLOOR + e.exp())
        .collect();

    // μ-RMSE-to-truth on the eval grid.
    let mg = arm.grid_x.len();
    let mut eval_grid = Array2::<f64>::zeros((mg, ncols));
    for (i, &gx) in arm.grid_x.iter().enumerate() {
        eval_grid[[i, x_idx]] = gx;
    }
    let md = build_term_collection_design(eval_grid.view(), &fit.meanspec_resolved)
        .expect("joint eval mean design");
    let mu = md.design.apply(&beta_mu) + &md.affine_offset;
    let mu_rmse = rmse(&mu.to_vec(), &arm.truth_mu_grid);
    (sigma_hat, edf_mu, lambda_mu, mu_rmse)
}

struct MarginalFit {
    edf_mu: f64,
    lambda_mu: Vec<f64>,
    mu_rmse: f64,
    p_mu: usize,
}

/// Plain (single-parameter) Gaussian fit with the SAME μ formula but the given
/// per-row prior weights (a = 1/σ²): the criterion selects λ_μ against exactly
/// those weights. Returns edf_μ, λ̂_μ, and μ-RMSE-to-truth.
fn fit_marginal_gaussian(arm: &Arm, weights: &[f64]) -> MarginalFit {
    let n = arm.xs.len();
    let mut headers = arm.headers.clone();
    headers.push("w".to_string());
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            let mut vals: Vec<String> = arm
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
            vals.push(format!("{:.17e}", weights[i]));
            csv::StringRecord::from(vals)
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode marginal");
    let col = ds.column_map();
    let x_idx = col[arm.x_name];
    let ncols = ds.headers.len();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        weight_column: Some("w".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(arm.mean_formula, &ds, &cfg).expect("marginal gaussian fit");
    let FitResult::Standard(sr) = result else {
        panic!(
            "expected a Standard single-parameter Gaussian fit — a fast-path (SplineScan/\
             ResidualCascade) variant would mean the μ basis differs from the joint fit's"
        );
    };
    let block = sr.fit.block_by_role(BlockRole::Mean).expect("mean block");
    let beta = block.beta.clone();
    let edf_mu = block.edf;
    let lambda_mu = block.lambdas.to_vec();

    let mg = arm.grid_x.len();
    let mut eval_grid = Array2::<f64>::zeros((mg, ncols));
    for (i, &gx) in arm.grid_x.iter().enumerate() {
        eval_grid[[i, x_idx]] = gx;
    }
    let md = build_term_collection_design(eval_grid.view(), &sr.resolvedspec)
        .expect("marginal eval design");
    assert_eq!(
        md.design.ncols(),
        beta.len(),
        "marginal eval design width {} != μ coeff count {}",
        md.design.ncols(),
        beta.len()
    );
    let mu = md.design.apply(&beta) + &md.affine_offset;
    let mu_rmse = rmse(&mu.to_vec(), &arm.truth_mu_grid);
    MarginalFit {
        edf_mu,
        lambda_mu,
        mu_rmse,
        p_mu: md.design.ncols(),
    }
}

fn run_sigma_clamp_arm(arm: &Arm) {
    eprintln!(
        "\n============ #1561 σ-clamp criterion test: {} arm ============",
        arm.label
    );
    let (sigma_hat, edf_joint, lam_joint, rmse_joint) = fit_joint_sigma_and_mu(arm);
    let w_hat: Vec<f64> = sigma_hat.iter().map(|&s| 1.0 / (s * s)).collect();
    let w_true: Vec<f64> = arm
        .truth_sigma_train
        .iter()
        .map(|&s| 1.0 / (s.max(1e-3)).powi(2))
        .collect();
    // How far gam's flexible σ̂ dips below σ_true (the inflation the hypothesis
    // pins on): mean and min of σ̂/σ_true over the training rows.
    let ratios: Vec<f64> = sigma_hat
        .iter()
        .zip(arm.truth_sigma_train.iter())
        .map(|(&sh, &st)| sh / st.max(1e-6))
        .collect();
    let ratio_mean = ratios.iter().sum::<f64>() / (ratios.len() as f64);
    let ratio_min = ratios.iter().cloned().fold(f64::INFINITY, f64::min);

    let m_hat = fit_marginal_gaussian(arm, &w_hat);
    let m_true = fit_marginal_gaussian(arm, &w_true);

    eprintln!(
        "[{}] σ̂/σ_true over train rows: mean={ratio_mean:.3} min={ratio_min:.3} \
         (min<1 ⇒ σ̂ dips below truth, inflating 1/σ̂² weights)",
        arm.label
    );
    eprintln!(
        "[{}] J joint(σ̂,2-block LAML): edf_μ={edf_joint:.3} μ-RMSE={rmse_joint:.5} λ̂_μ={lam_joint:?}",
        arm.label
    );
    eprintln!(
        "[{}] A marg(σ̂ ,1-block REML): edf_μ={:.3} μ-RMSE={:.5} λ̂_μ={:?}",
        arm.label, m_hat.edf_mu, m_hat.mu_rmse, m_hat.lambda_mu
    );
    eprintln!(
        "[{}] B marg(σtrue,1-blk REML): edf_μ={:.3} μ-RMSE={:.5} λ̂_μ={:?} p_μ={}",
        arm.label, m_true.edf_mu, m_true.mu_rmse, m_true.lambda_mu, m_true.p_mu
    );
    let d_structure = edf_joint - m_hat.edf_mu; // J → A : 2-block vs 1-block
    let d_weights = m_hat.edf_mu - m_true.edf_mu; // A → B : σ̂ noise in weights
    let rmse_gain_clamp = m_hat.mu_rmse - m_true.mu_rmse; // μ-RMSE improvement from σ-clamp
    eprintln!(
        "[{}] Δedf_μ  structure(J→A)={d_structure:+.3}  weights/σ-clamp(A→B)={d_weights:+.3}  \
         | μ-RMSE clamp gain(A→B)={rmse_gain_clamp:+.5}",
        arm.label
    );

    // Decisive readout. The hypothesis predicts clamping σ→truth (B) selects
    // MORE μ smoothing (edf_μ drops) and improves μ accuracy vs σ̂ (A).
    let clamp_moves_edf = d_weights >= 0.15 * edf_joint;
    let clamp_helps_rmse = rmse_joint > 0.0 && rmse_gain_clamp >= 0.05 * rmse_joint;
    let structure_moves_edf = d_structure >= 0.15 * edf_joint;
    let verdict = if clamp_moves_edf && clamp_helps_rmse {
        "σ̂-FEEDBACK IMPLICATED: clamping σ→truth makes the criterion select MORE μ \
         smoothing AND improves μ-RMSE — the inflated 1/σ̂² weights drag λ̂_μ down"
    } else if structure_moves_edf && !clamp_moves_edf {
        "JOINT-STRUCTURE effect: the 2-block→1-block change moves μ smoothing, not the \
         σ̂ weights — look at the joint criterion's cross/logdet coupling, not σ̂ feedback"
    } else if !clamp_moves_edf && !structure_moves_edf {
        "σ̂-FEEDBACK FALSIFIED: neither σ-clamp nor 1-block structure moves λ̂_μ toward the \
         oracle — the marginal criterion under-prices μ-EDF regardless (next candidate term)"
    } else {
        "MIXED: both structure and σ-clamp contribute — see the two Δedf_μ terms"
    };
    eprintln!("[{}] VERDICT: {verdict}", arm.label);

    assert!(
        edf_joint.is_finite()
            && m_hat.edf_mu.is_finite()
            && m_true.edf_mu.is_finite()
            && rmse_joint.is_finite()
            && m_hat.mu_rmse.is_finite()
            && m_true.mu_rmse.is_finite(),
        "[{}] non-finite edf/μ-RMSE from a production fit",
        arm.label
    );
}

#[test]
fn zz_measure_1561_mu_criterion_under_sigma_clamp() {
    init_parallelism();
    eprintln!(
        "=== #1561 FOLLOW-UP: does λ̂_μ move toward the oracle when σ is clamped at truth? ===\n\
         Compare the criterion's μ smoothing across three PRODUCTION fits — joint(σ̂),\n\
         marginal(σ̂), marginal(σ_true) — to isolate whether the σ̂-feedback term\n\
         H_μμ=Xᵀdiag(a/σ̂²)X is what under-selects λ̂_μ. Oracle target (sibling test):\n\
         edf_μ* ≈ 5.8 / 13.7, μ-RMSE* ≈ 0.040 / 0.019 for cyclic / plain."
    );
    run_sigma_clamp_arm(&cyclic_arm());
    run_sigma_clamp_arm(&plain_arm());
}
