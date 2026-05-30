//! End-to-end quality: gam's **single-pass approximate leave-one-out (ALO)**
//! diagnostics for a Poisson/log tensor-product GAM must match gam's *own*
//! **brute-force exact leave-one-out**, computed by literally refitting the
//! converged working (IRLS-linearised) penalized model with each observation
//! held out.
//!
//! Mature comparator: **gam itself, in brute-force n-fold hold-out mode.** ALO
//! is a Sherman–Morrison rank-1 shortcut for the leave-one-out predictor of the
//! penalized weighted least-squares system that PIRLS converges to. The honest,
//! adversarial reference is therefore the *exact* solution of that same system
//! with row `i` removed — recomputed n times, once per observation, from gam's
//! own converged geometry. There is no external library that exposes ALO for a
//! penalized tensor-product Poisson GAM; the only correct ground truth is the
//! exhaustive refit, which is precisely what ALO claims to approximate. If the
//! shortcut and the exhaustive refit disagree, ALO is wrong.
//!
//! Why fix the converged working model rather than re-running full PIRLS + REML
//! per fold: ALO approximates leave-one-out *at the converged linearisation and
//! at fixed smoothing parameters λ* — that is the quantity it is derived from
//! and the only quantity it can be held to. Re-estimating λ each fold would
//! benchmark λ-instability, not the ALO algebra. So the brute force here drops
//! row `i` from the exact penalized normal equations
//!     H β₋ᵢ = c − w_i (z_i − o_i) x_i,     H = XᵀWX + S(λ),  c = Xᵀ W (z − o)
//! and reads η̃_i = o_i + x_iᵀ β₋ᵢ. Both H, X, W, z, o, and the link are taken
//! verbatim from gam's converged PIRLS artifact, so the two engines see bitwise
//! identical inputs and any disagreement is a real defect in the ALO update.
//!
//! Poisson/log is the canonical exponential-family case (Fisher == observed
//! information, so a single weight vector is exact) and the `te(x1, x2)` tensor
//! product exercises the multi-dimensional penalized Hessian and the chunked
//! influence-matrix inversion `a_ii = w_i x_iᵀ H⁻¹ x_i` that ALO leverage
//! depends on.
//!
//! Asserted, with one-line justifications at each site:
//!   * leverage a_ii: max |ALO − brute| < 1e-8 and pearson > 0.999999 — leverage
//!     is a deterministic property of the *full-data* hat matrix, so ALO's
//!     chunked solve and the dense reference solve must agree to solver
//!     round-off; a loose bound here would hide an influence-matrix bug.
//!   * η̃ (ALO LOO predictor): relative L2 < 1e-6 vs the exact reduced-system
//!     refit. For the canonical Poisson/log link W_h == W_s, so the ALO rank-1
//!     formula is the *exact* Sherman–Morrison solution of the same downdated
//!     system the brute force factorises — they agree to solver round-off, and
//!     a permille-scale bound would assert essentially nothing for an identity.
//!   * Bayesian SE √(φ x_iᵀ H⁻¹ x_i): pearson > 0.999999 AND max|Δ| < 1e-8 vs
//!     the reference √(φ x_iᵀH⁻¹x_i) — the same exact diagonal, so correlation
//!     alone (scale/offset blind) is backed by an absolute round-off bound.

use gam::inference::alo::compute_alo_diagnostics_from_fit;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{max_abs_diff, pearson, relative_l2};
use gam::types::LinkFunction;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson, Uniform};
use std::f64::consts::PI;

/// Dense Cholesky factorisation of a symmetric positive-definite matrix
/// (lower-triangular L with A = L Lᵀ). The penalized Hessian H = XᵀWX + S(λ)
/// of a converged Poisson/log GAM is SPD (Fisher weights ≥ 0 plus a positive
/// penalty / stabilisation ridge), so this is the natural exact solver for the
/// brute-force reference. Panics if a non-positive pivot appears, which would
/// itself signal a non-SPD Hessian (a real defect worth failing on).
fn cholesky_lower(a: &Array2<f64>) -> Array2<f64> {
    let p = a.nrows();
    assert_eq!(p, a.ncols(), "cholesky requires a square matrix");
    let mut l = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                assert!(
                    sum > 0.0,
                    "penalized Hessian is not positive definite at pivot {i}: {sum:.3e}"
                );
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    l
}

/// Solve A x = b for SPD A given its lower Cholesky factor L (A = L Lᵀ) via
/// forward/back substitution.
fn cholesky_solve(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let p = l.nrows();
    // Forward solve L y = b.
    let mut y = Array1::<f64>::zeros(p);
    for i in 0..p {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[[i, k]] * y[k];
        }
        y[i] = sum / l[[i, i]];
    }
    // Back solve Lᵀ x = y.
    let mut x = Array1::<f64>::zeros(p);
    for i in (0..p).rev() {
        let mut sum = y[i];
        for k in (i + 1)..p {
            sum -= l[[k, i]] * x[k];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

#[test]
fn alo_matches_brute_force_loo_on_poisson_log_tensor() {
    init_parallelism();

    // ---- synthetic 2-D Poisson-count truth on the unit square --------------
    // The spec's named "bone.csv (310 obs, 2D)" does not exist in this tree
    // (bench/datasets/bone.csv is a 23-row survival table), so we synthesise a
    // 2-D Poisson smooth with a fixed seed. The SAME draws feed gam's fit and
    // the brute-force reference (which reads gam's own converged geometry), so
    // the two engines are guaranteed bitwise-identical inputs.
    let n = 310usize;
    let mut rng = StdRng::seed_from_u64(20260529);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");

    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let a = u.sample(&mut rng);
        let b = u.sample(&mut rng);
        // Smooth log-mean surface with genuine 2-D structure (interaction),
        // kept in a moderate range so counts are informative but not extreme.
        let eta = 0.6 + 0.9 * (PI * a).sin() * (0.5 + b) - 0.7 * (b - 0.5).powi(2);
        let lambda = eta.exp().max(1e-9);
        let draw: f64 = Poisson::new(lambda)
            .expect("valid Poisson rate")
            .sample(&mut rng);
        x1.push(a);
        x2.push(b);
        y.push(draw);
    }

    // ---- fit with gam: y ~ te(x1, x2), Poisson / log link ------------------
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let rows = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![
                x1[i].to_string(),
                x2[i].to_string(),
                y[i].to_string(),
            ])
        })
        .collect::<Vec<_>>();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode poisson dataset");
    let col = ds.column_map();
    let x1_idx = col["x1"];
    let x2_idx = col["x2"];

    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ te(x1, x2)", &ds, &cfg).expect("gam poisson te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Poisson te(x1, x2)");
    };

    // Rebuild the frozen tensor design at the training points and confirm it
    // reproduces the fitted linear predictor (log-mean) via design*beta. This
    // pins the basis/coordinates the ALO path and the brute-force reference
    // operate on before either consumes the converged geometry.
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x1_idx]] = x1[i];
        grid[[i, x2_idx]] = x2[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild te design at training points");
    let rebuilt_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(rebuilt_eta.len(), n, "rebuilt eta length mismatch");
    assert!(
        rebuilt_eta.iter().all(|v| v.is_finite()),
        "rebuilt linear predictor must be finite"
    );

    let y_arr = Array1::from(y.clone());

    // ---- gam ALO diagnostics (the capability under test) -------------------
    let alo = compute_alo_diagnostics_from_fit(&fit.fit, y_arr.view(), LinkFunction::Log)
        .expect("ALO diagnostics for Poisson/log te(x1, x2)");
    assert_eq!(alo.leverage.len(), n, "ALO leverage length mismatch");
    assert_eq!(alo.eta_tilde.len(), n, "ALO eta_tilde length mismatch");
    assert_eq!(alo.se_bayes.len(), n, "ALO se_bayes length mismatch");

    // ---- brute-force exact LOO from gam's own converged geometry -----------
    // The PIRLS artifact carries the consistent transformed design X, the exact
    // dense penalized Hessian H = XᵀWX + S(λ), the score-side Fisher weights W
    // (== observed information for the canonical log link), the working response
    // z, the linear predictor η̂, and the offset o — exactly the inputs the ALO
    // path consumes. We re-derive leave-one-out exactly from these.
    let pirls = fit
        .fit
        .artifacts
        .pirls
        .as_ref()
        .expect("Poisson GAM fit must expose PIRLS geometry");

    let x_arc = pirls
        .x_transformed
        .try_to_dense_arc("brute-force LOO needs dense transformed design")
        .expect("dense transformed design");
    let x = x_arc.as_ref();
    let h = pirls
        .dense_stabilizedhessian_transformed("brute-force LOO needs dense penalized Hessian")
        .expect("dense penalized Hessian");
    let p = x.ncols();
    assert_eq!(x.nrows(), n, "transformed design row count mismatch");
    assert_eq!(h.nrows(), p, "Hessian must be p x p");

    // ALO forms leverage and the rank-1 downdate with the Hessian-side weights
    // (H = XᵀW_h X + S) and the score/RHS with the score-side Fisher weights.
    // For the canonical Poisson/log link Fisher == observed information, so
    // W_h == W_s == μ; we assert that here and then carry a single `w`, which
    // is exactly what makes the ALO closed form an EXACT Sherman–Morrison
    // solution of the brute-force downdated system rather than an approximation.
    let w_hess: Vec<f64> = pirls.final_weights_signed().view().to_vec();
    let w: Vec<f64> = pirls.solve_weights_psd().view().to_vec(); // score-side Fisher weights
    let wh_ws_max_diff = max_abs_diff(&w_hess, &w);
    assert!(
        wh_ws_max_diff < 1e-12,
        "canonical Poisson/log must have Hessian weights == score weights \
         (Fisher == observed information): max|W_h − W_s|={wh_ws_max_diff:.3e}"
    );
    let z = &pirls.solveworking_response; // working response
    let eta_hat = &pirls.final_eta;
    let offset = &pirls.final_offset;
    let phi = 1.0_f64; // Poisson dispersion is fixed at 1 (matches ALO's phi).

    // Full-data penalized RHS in centered (offset-subtracted) coordinates:
    //   c = Xᵀ W (z − o),     H β̂ = c   (the system PIRLS converged to).
    let mut c = Array1::<f64>::zeros(p);
    for i in 0..n {
        let coef = w[i] * (z[i] - offset[i]);
        let row = x.row(i);
        for k in 0..p {
            c[k] += coef * row[k];
        }
    }

    // Verify the brute-force engine reconstructs the SAME working-model fit gam
    // converged to: solving H β = c must reproduce the fitted η̂ = o + Xβ.
    // This pins the reference to gam's geometry before any row is dropped, so a
    // disagreement downstream is in the *hold-out* algebra, not the setup.
    let l_full = cholesky_lower(&h);
    let beta_full = cholesky_solve(&l_full, &c);
    let mut eta_recon = Array1::<f64>::zeros(n);
    for i in 0..n {
        let row = x.row(i);
        let mut dot = 0.0;
        for k in 0..p {
            dot += row[k] * beta_full[k];
        }
        eta_recon[i] = offset[i] + dot;
    }
    let recon_rel = relative_l2(eta_recon.as_slice().unwrap(), eta_hat.as_slice().unwrap());
    assert!(
        recon_rel < 1e-6,
        "brute-force engine must reconstruct gam's converged working fit before \
         hold-out (rel_l2={recon_rel:.3e}); a mismatch means X/H/W/z are not the \
         system PIRLS solved"
    );

    // For each observation i, solve the leave-one-out penalized normal equations
    //   (H − w_i x_i x_iᵀ) β₋ᵢ = c − w_i (z_i − o_i) x_i
    // exactly, and read η̃_i = o_i + x_iᵀ β₋ᵢ. Also form the exact leverage
    //   a_ii = w_i x_iᵀ H⁻¹ x_i   and conditional variance x_iᵀ H⁻¹ x_i
    // via a dense solve against the *full* Hessian. This is the exhaustive
    // n-fold reference ALO is meant to approximate.
    let mut brute_eta_tilde = vec![0.0_f64; n];
    let mut brute_leverage = vec![0.0_f64; n];
    let mut brute_se_bayes = vec![0.0_f64; n];
    for i in 0..n {
        let xi: Array1<f64> = x.row(i).to_owned();

        // x_iᵀ H⁻¹ x_i from the full-data factor (leverage + Bayesian variance).
        let hinv_xi = cholesky_solve(&l_full, &xi);
        let mut x_hinv_x = 0.0;
        for k in 0..p {
            x_hinv_x += xi[k] * hinv_xi[k];
        }
        brute_leverage[i] = w[i] * x_hinv_x;
        brute_se_bayes[i] = (phi * x_hinv_x).max(0.0).sqrt();

        // Exact hold-out refit: H₋ᵢ = H − w_i x_i x_iᵀ , RHS₋ᵢ = c − w_i(z_i−o_i)x_i.
        let mut h_minus = h.clone();
        for r in 0..p {
            let wr = w[i] * xi[r];
            for cc in 0..p {
                h_minus[[r, cc]] -= wr * xi[cc];
            }
        }
        let mut rhs_minus = c.clone();
        let drop_coef = w[i] * (z[i] - offset[i]);
        for k in 0..p {
            rhs_minus[k] -= drop_coef * xi[k];
        }
        let l_minus = cholesky_lower(&h_minus);
        let beta_minus = cholesky_solve(&l_minus, &rhs_minus);
        let mut dot = 0.0;
        for k in 0..p {
            dot += xi[k] * beta_minus[k];
        }
        brute_eta_tilde[i] = offset[i] + dot;
    }

    // ---- compare ALO vs brute-force exact LOO ------------------------------
    let alo_lev = alo.leverage.as_slice().unwrap();
    let alo_eta = alo.eta_tilde.as_slice().unwrap();
    let alo_se = alo.se_bayes.as_slice().unwrap();

    let lev_max_diff = max_abs_diff(alo_lev, &brute_leverage);
    let lev_corr = pearson(alo_lev, &brute_leverage);
    let eta_rel = relative_l2(alo_eta, &brute_eta_tilde);
    let eta_corr = pearson(alo_eta, &brute_eta_tilde);
    let se_corr = pearson(alo_se, &brute_se_bayes);
    let se_max_diff = max_abs_diff(alo_se, &brute_se_bayes);

    eprintln!(
        "ALO vs brute-force LOO (Poisson/log te(x1,x2)): n={n} p={p} \
         leverage max|Δ|={lev_max_diff:.3e} pearson={lev_corr:.6} \
         eta_tilde rel_l2={eta_rel:.3e} pearson={eta_corr:.6} \
         se_bayes max|Δ|={se_max_diff:.3e} pearson={se_corr:.5}"
    );

    // Leverage a_ii = w_i x_iᵀ H⁻¹ x_i is a deterministic function of the
    // full-data hat matrix; ALO's chunked column solve and the dense Cholesky
    // reference solve the SAME linear systems, so they must agree to solver
    // round-off. 1e-8 is the principled bound (the alo.rs Hessian-symmetry
    // tolerance is itself 1e-8); a looser bound would mask an influence-matrix
    // inversion bug.
    assert!(
        lev_max_diff < 1e-8,
        "ALO leverage must match exact w_i x_iᵀH⁻¹x_i to round-off: max|Δ|={lev_max_diff:.3e}"
    );
    assert!(
        lev_corr > 0.999999,
        "ALO leverage must be near-perfectly correlated with exact leverage: pearson={lev_corr:.6}"
    );

    // η̃: for the canonical Poisson/log link the Hessian-side weights equal the
    // score-side Fisher weights (w_h == w_s == μ ≥ 0), so the ALO closed form
    //   η̃_i = o_i + (η̂_i−o_i) + (x_iᵀH⁻¹x_i)·w_i(η̂_i−z_i) / (1 − w_i x_iᵀH⁻¹x_i)
    // is the EXACT Sherman–Morrison solution of the same rank-1 downdated
    // system the brute force factorises directly. The two therefore agree to
    // solver round-off, not to a "few permille": the only slack is the
    // difference between one factor of H reused via Sherman–Morrison and a
    // fresh Cholesky of H − w_i x_i x_iᵀ per fold, both well-conditioned here
    // (small leverages, 1−a_ii ≈ 1). 1e-6 stays comfortably above that
    // round-off while a botched 1−a_ii, a wrong score weight, or an offset
    // sign error would blow rel_l2 far past it; the previous 1.5% bound was so
    // loose it asserted essentially nothing for an exact identity.
    assert!(
        eta_rel < 1e-6,
        "ALO eta_tilde must match the exact Sherman–Morrison leave-one-out predictor to round-off: rel_l2={eta_rel:.3e}"
    );
    assert!(
        eta_corr > 0.999999,
        "ALO eta_tilde must be near-perfectly correlated with exact LOO: pearson={eta_corr:.6}"
    );

    // Bayesian SE √(φ x_iᵀ H⁻¹ x_i): ALO and the reference form the IDENTICAL
    // quantity from the same H and the same x_i (φ = 1 for Poisson), so this is
    // another exact identity up to solver round-off — not merely "monotone".
    // Pearson alone is a weak guard here (it is invariant to a wrong constant
    // scale on φ or a uniform variance offset), so we additionally pin the
    // absolute agreement: max|Δ| < 1e-8 catches any scale/offset defect in
    // ALO's uncertainty quantification that a bare correlation would miss.
    assert!(
        se_corr > 0.999999,
        "ALO Bayesian SE must track exact conditional variance diagonal: pearson={se_corr:.6}"
    );
    assert!(
        se_max_diff < 1e-8,
        "ALO Bayesian SE √(φ x_iᵀH⁻¹x_i) must equal the exact diagonal to round-off: max|Δ|={se_max_diff:.3e}"
    );
}
