//! End-to-end quality: gam's **simplex (compositional) response regression**
//! via a custom Dirichlet likelihood must **recover the known truth** that the
//! data was generated from. `DirichletReg` — the mature R reference for
//! Dirichlet-response regression — is fit on the *identical* data and demoted to
//! a match-or-beat accuracy baseline: gam's recovery error against the truth must
//! not exceed DirichletReg's by more than a small margin.
//!
//! ## Objective metric asserted (TRUTH RECOVERY)
//! The data is drawn from a **known** smooth Dirichlet truth `α_k(x) = exp(η_k(x))`
//! with `η_k = true_eta(x)`. We therefore assert that gam recovers the truth, not
//! that it reproduces a peer tool's fit:
//!   * **RMSE of the fitted closed mean composition** `μ(x) = α(x)/Σα(x)` against
//!     the *true* `μ(x)` on a dense grid is below a principled bar (a small
//!     fraction of the simplex coordinate range), AND ≤ DirichletReg's own RMSE
//!     against the same truth × 1.10 (match-or-beat on accuracy).
//!   * **RMSE of the fitted log-concentration** `log φ(x) = log Σ_k α_k(x)`
//!     against the *true* `log φ(x)` is below a principled bar, AND ≤
//!     DirichletReg's RMSE × 1.10.
//! DirichletReg's fit is still computed and its rel-L2 vs gam printed for context,
//! but "close to DirichletReg" is no longer a pass criterion.
//!
//! ## What is benchmarked
//! A composition `y_i = (y_{i1}, …, y_{iK})` on the `K`-part simplex
//! (`y_{ik} > 0`, `Σ_k y_{ik} = 1`) is modelled with a Dirichlet likelihood in
//! the *common* (log-α) parameterization, the canonical Dirichlet GLM that
//! `DirichletReg::DirichletReg()` fits by default:
//!
//!   α_{ik} = exp(η_{ik}) ,  η_{ik} = Xᵢ β_k  (k = 1..K, one block per part),
//!   ℓ = Σ_i [ lnΓ(α_{i0}) − Σ_k lnΓ(α_{ik}) + Σ_k (α_{ik} − 1) ln y_{ik} ] ,
//!       with α_{i0} = Σ_k α_{ik}.
//!
//! This single family carries **both** the Aitchison-geometry *location*
//! (the additive-log-ratio coordinates ALRₖ = ln(μ_k/μ_K) = η_k − η_K, with
//! μ_k = α_k/α_0 the closed mean composition) **and** the *concentration*
//! (precision φ = α_0 = Σ_k α_k, the Dirichlet sample size). Both vary smoothly
//! with `x` because every η_k is a penalized cubic P-spline of `x`. This is the
//! distinctive gam capability: a multi-block custom family whose likelihood
//! couples all K linear predictors per row (the Dirichlet score in block k
//! depends on the digamma of α_0, hence on every other block), fit through the
//! `fit_custom_family` / `ParameterBlockSpec` (`BlockRole` + per-block β)
//! reconstruction pattern.
//!
//! ## Comparator (best-in-class)
//! `DirichletReg::DirichletReg(DR_data(Y) ~ -1 + basis_columns)` — the mature,
//! peer-reviewed R package for Dirichlet-response regression (Maier 2014). It
//! has no penalized-smooth facility, so we give *both* engines the identical
//! cubic-B-spline basis `X = [1 | B_centered]` and let each fit the SAME
//! Dirichlet common-model likelihood on it: gam by REML-penalized smoothing,
//! DirichletReg by unpenalized maximum likelihood. It serves as a **baseline to
//! match-or-beat** on truth-recovery accuracy: gam's RMSE against the known truth
//! must not exceed DirichletReg's by more than 10%. (`-1` suppresses DirichletReg's
//! own intercept so its per-component
//! coefficient vector aligns 1:1 with our explicit-intercept design `X`, making
//! the recovered η_k = X β_k exact in either engine.)
//!
//! ## Data
//! Identical data is fed to both engines. With a fixed seed, `n = 150` rows
//! with `x ~ U(-2, 2)`; the K=3 truth is `η_k(x) = a_k + smooth_k(x)` and a
//! smooth log-concentration, drawn as a Dirichlet via normalized independent
//! Gamma(α_k, 1) variates. The exact composition columns `p1,p2,p3` and the
//! basis matrix `X` are handed verbatim to both engines.
//!
//! ## Metrics
//!   * **RMSE of the closed mean composition** `μ(x)` vs the *true* `μ(x)` over a
//!     dense grid (truth-recovery accuracy on the simplex).
//!   * **RMSE of the log-concentration** `log φ(x) = log Σ_k α_k` vs the *true*
//!     `log φ(x)` (truth recovery of the Dirichlet precision).
//! For context only (printed, not asserted as a pass gate): the relative-L2
//! agreement of gam's fit with DirichletReg's fit.

use gam::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, build_bspline_basis_1d,
};
use gam::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, FamilyEvaluation, ParameterBlockSpec,
    ParameterBlockState, PenaltyMatrix, fit_custom_family,
};
use gam::load_csvwith_inferred_schema;
use gam::matrix::DesignMatrix;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Uniform};
use statrs::function::gamma::{digamma, ln_gamma};
use std::path::Path;

const N: usize = 150;
const K: usize = 3;
const N_GRID: usize = 60;
const GRID_LO: f64 = -1.9;
const GRID_HI: f64 = 1.9;

/// Trigamma ψ'(x) = d/dx ψ(x), x > 0. Asymptotic series with the standard
/// recurrence to push the argument above 6 (mirrors the implementation gam's
/// own PIRLS path uses). Needed for the Dirichlet Fisher working weights.
fn trigamma(mut x: f64) -> f64 {
    let mut value = 0.0;
    while x < 6.0 {
        value += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    // 1/x + 1/(2x²) + 1/(6x³) − 1/(30x⁵) + 1/(42x⁷) − 1/(30x⁹)
    value
        + inv
            * (1.0
                + inv
                    * (0.5
                        + inv
                            * (1.0 / 6.0
                                + inv2 * (-1.0 / 30.0 + inv2 * (1.0 / 42.0 - inv2 / 30.0)))))
}

/// True smooth log-α surfaces η_k(x), k = 0..K-1, in the Dirichlet common
/// parameterization. Distinct shapes so the location (ALR) and concentration
/// (Σα) both genuinely vary with x.
fn true_eta(x: f64) -> [f64; K] {
    [
        0.6 + 0.9 * x.sin(),                     // part 1
        0.2 - 0.5 * x + 0.4 * (1.3 * x).cos(),   // part 2
        -0.3 + 0.7 * (0.8 * x).sin() + 0.15 * x, // part 3
    ]
}

/// Build a shared P-spline block design `[1 | B_centered]` (explicit intercept
/// + sum-to-zero–centered cubic B-spline) with the second-difference penalty
/// zero-padded over the unpenalized intercept column. The basis is built over
/// `x_all = [train ; grid]` so the columns and the data-dependent centering are
/// identical for fitting and prediction; rows are then split. Returns the
/// `(train_design, penalties, nullspace_dims, grid_design)` parts shared by
/// every Dirichlet block.
fn pspline_parts(x_all: &[f64]) -> (Array2<f64>, Vec<PenaltyMatrix>, Vec<usize>, Array2<f64>) {
    let x_arr = Array1::from_vec(x_all.to_vec());
    let lo = x_all.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = x_all.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Generate {
            data_range: (lo, hi),
            num_internal_knots: 8,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::WeightedSumToZero { weights: None },
        boundary: Default::default(),
        boundary_conditions: Default::default(),
    };
    let basis = build_bspline_basis_1d(x_arr.view(), &spec).expect("build P-spline basis");
    let b_all = basis.design.to_dense();
    let p_s = b_all.ncols();
    let p = p_s + 1; // + explicit intercept

    let n_all = x_all.len();
    let mut full = Array2::<f64>::zeros((n_all, p));
    for i in 0..n_all {
        full[[i, 0]] = 1.0;
        for j in 0..p_s {
            full[[i, j + 1]] = b_all[[i, j]];
        }
    }

    let mut penalties = Vec::with_capacity(basis.penalties.len());
    let mut nullspace_dims = Vec::with_capacity(basis.penalties.len());
    for (k, s_basis) in basis.penalties.iter().enumerate() {
        assert!(
            s_basis.nrows() == p_s && s_basis.ncols() == p_s,
            "penalty {k} shape {:?} != {p_s}x{p_s}",
            s_basis.shape()
        );
        let mut s = Array2::<f64>::zeros((p, p));
        for r in 0..p_s {
            for c in 0..p_s {
                s[[r + 1, c + 1]] = s_basis[[r, c]];
            }
        }
        penalties.push(PenaltyMatrix::from(s));
        // The padded penalty gains one extra null direction (the unpenalized
        // intercept column) on top of the basis penalty's own structural null
        // space (linear trend for a 2nd-order difference).
        let base_null = basis.nullspace_dims.get(k).copied().unwrap_or(0);
        nullspace_dims.push(base_null + 1);
    }

    let train = full.slice(ndarray::s![0..N, ..]).to_owned();
    let grid = full.slice(ndarray::s![N.., ..]).to_owned();
    (train, penalties, nullspace_dims, grid)
}

/// Dirichlet likelihood in the common (log-α) parameterization: one block per
/// simplex part, `α_k = exp(η_k)`. The row likelihood couples every block
/// through `α_0 = Σ_k α_k`, so this exercises gam's coupled multi-block path.
///
/// Per-block IRLS (Fisher-scoring) working set in η-space, derived from
///   s_k = ∂ℓ/∂η_k = α_k (ψ(α_0) − ψ(α_k) + ln y_k) ,
///   E[−∂²ℓ/∂η_k²] = α_k² (ψ'(α_k) − ψ'(α_0))  (> 0; ψ' decreasing, α_k ≤ α_0),
/// giving working response z_k = η_k + s_k / w_k and weight w_k. Block-diagonal
/// Fisher scoring converges to the joint Dirichlet MLE by block coordinate
/// iteration (the standard backfitting used across distributional parameters).
#[derive(Clone)]
struct DirichletCommonFamily {
    /// `log_y[k]` is the column of ln y_{·k} for part k (length N each).
    log_y: Vec<Array1<f64>>,
}

impl CustomFamily for DirichletCommonFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != K {
            return Err(format!(
                "DirichletCommonFamily expects {K} blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.log_y[0].len();
        for st in block_states {
            if st.eta.len() != n {
                return Err("DirichletCommonFamily block eta length mismatch".to_string());
            }
        }

        // α_{ik} = exp(η_{ik}) and α_{i0} = Σ_k α_{ik}.
        let mut alpha: Vec<Array1<f64>> = Vec::with_capacity(K);
        for st in block_states {
            alpha.push(st.eta.mapv(f64::exp));
        }
        let mut alpha0 = Array1::<f64>::zeros(n);
        for a in &alpha {
            alpha0 += a;
        }

        let mut ll = 0.0;
        for i in 0..n {
            ll += ln_gamma(alpha0[i]);
            for k in 0..K {
                ll += -ln_gamma(alpha[k][i]) + (alpha[k][i] - 1.0) * self.log_y[k][i];
            }
        }

        let mut working_sets = Vec::with_capacity(K);
        for k in 0..K {
            let mut z = Array1::<f64>::zeros(n);
            let mut w = Array1::<f64>::zeros(n);
            for i in 0..n {
                let a_k = alpha[k][i];
                let a0 = alpha0[i];
                let dig0 = digamma(a0);
                let dig_k = digamma(a_k);
                let score = a_k * (dig0 - dig_k + self.log_y[k][i]);
                // Fisher information in η-space; strictly positive because the
                // trigamma is strictly decreasing and α_k ≤ α_0, with a tiny
                // floor so a (degenerate) α_k == α_0 row never divides by zero.
                let weight = (a_k * a_k * (trigamma(a_k) - trigamma(a0))).max(1e-10);
                w[i] = weight;
                z[i] = block_states[k].eta[i] + score / weight;
            }
            working_sets.push(BlockWorkingSet::diagonal_checked(z, w)?);
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: working_sets,
        })
    }
}

/// Closed mean composition μ_k = α_k / Σα and log-concentration log Σα from a
/// stack of K linear predictors `eta[k]` (each length `m`).
fn closed_means_and_logconc(eta: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let m = eta[0].len();
    let mut mu = vec![vec![0.0; m]; K];
    let mut log_conc = vec![0.0; m];
    for i in 0..m {
        let mut a0 = 0.0;
        let mut a = [0.0; K];
        for k in 0..K {
            a[k] = eta[k][i].exp();
            a0 += a[k];
        }
        log_conc[i] = a0.ln();
        for k in 0..K {
            mu[k][i] = a[k] / a0;
        }
    }
    (mu, log_conc)
}

#[test]
fn gam_dirichlet_regression_recovers_truth() {
    gam::init_parallelism();

    // ---- synthetic data: x ~ U(-2,2), Dirichlet(α(x)) draws ---------------
    let mut rng = StdRng::seed_from_u64(0x0D15_C0FF_EE05_2926);
    let unit = Uniform::new(-2.0, 2.0).expect("uniform[-2,2]");
    let mut x: Vec<f64> = (0..N).map(|_| unit.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));

    // Dirichlet(α) = normalized independent Gamma(α_k, 1) variates.
    let mut p: Vec<Vec<f64>> = vec![vec![0.0; N]; K];
    for i in 0..N {
        let eta = true_eta(x[i]);
        let mut draws = [0.0f64; K];
        let mut s = 0.0;
        for k in 0..K {
            let alpha_k = eta[k].exp();
            let g = Gamma::<f64>::new(alpha_k, 1.0).expect("Gamma(α_k,1)");
            let d = g.sample(&mut rng).max(1e-12);
            draws[k] = d;
            s += d;
        }
        for k in 0..K {
            p[k][i] = (draws[k] / s).max(1e-9);
        }
        // Re-close after the positivity floor so each row sums to exactly 1.
        let renorm: f64 = (0..K).map(|k| p[k][i]).sum();
        for k in 0..K {
            p[k][i] /= renorm;
        }
    }

    let grid: Vec<f64> = (0..N_GRID)
        .map(|i| GRID_LO + (GRID_HI - GRID_LO) * (i as f64) / ((N_GRID - 1) as f64))
        .collect();

    // x_all = [train ; grid] so the basis columns + centering are shared.
    let mut x_all = x.clone();
    x_all.extend_from_slice(&grid);
    let (train_design, penalties, nullspace_dims, grid_design) = pspline_parts(&x_all);
    let p_cols = train_design.ncols();

    // ---- gam: K-block Dirichlet common-parameterization custom family ------
    let log_y: Vec<Array1<f64>> = (0..K)
        .map(|k| Array1::from_iter(p[k].iter().map(|&v| v.ln())))
        .collect();
    let family = DirichletCommonFamily {
        log_y: log_y.clone(),
    };

    let n_pen = penalties.len();
    let specs: Vec<ParameterBlockSpec> = (0..K)
        .map(|k| ParameterBlockSpec {
            name: format!("alpha{k}"),
            design: DesignMatrix::from(train_design.clone()),
            offset: Array1::zeros(N),
            penalties: penalties.clone(),
            nullspace_dims: nullspace_dims.clone(),
            initial_log_lambdas: Array1::zeros(n_pen),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        })
        .collect();

    let options = BlockwiseFitOptions {
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let fit = fit_custom_family(&family, &specs, &options).expect("gam Dirichlet fit");
    assert!(
        fit.outer_converged,
        "gam Dirichlet outer optimization did not converge"
    );

    // gam fitted η_k on the dense grid (reconstruct from per-block β̂).
    let gam_eta_grid: Vec<Vec<f64>> = (0..K)
        .map(|k| grid_design.dot(&fit.block_states[k].beta).to_vec())
        .collect();
    let (gam_mu_grid, gam_logconc_grid) = closed_means_and_logconc(&gam_eta_grid);

    // ---- DirichletReg: the mature Dirichlet-regression reference -----------
    // Hand both engines the SAME design X = [1 | B_centered]; `-1` suppresses
    // DirichletReg's own intercept so its per-component coefficients align 1:1
    // with our X columns. We pass the training design columns, fit the common
    // model, and emit the K·p coefficient matrix; we then reconstruct the grid
    // η_k = X_grid β_k in Rust from the identical grid design.
    let col_storage: Vec<Vec<f64>> = (0..p_cols)
        .map(|j| (0..N).map(|i| train_design[[i, j]]).collect())
        .collect();
    let basis_names: Vec<String> = (0..p_cols).map(basis_col_name).collect();
    let mut columns: Vec<Column<'_>> = Vec::with_capacity(K + p_cols);
    for k in 0..K {
        columns.push(Column::new(part_name(k), &p[k]));
    }
    for (name, c) in basis_names.iter().zip(col_storage.iter()) {
        columns.push(Column::new(name.as_str(), c));
    }

    let rhs = basis_names.join(" + ");
    let body = format!(
        r#"
        suppressPackageStartupMessages(library(DirichletReg))
        Y <- DR_data(cbind(df${p0}, df${p1}, df${p2}))
        m <- DirichletReg(Y ~ -1 + {rhs}, data = df)
        # Common model: one coefficient vector per simplex part, in part order.
        co <- coef(m)
        flat <- as.numeric(unlist(co))
        emit("coef", flat)
        emit("ncoef", length(flat))
        "#,
        p0 = part_name(0),
        p1 = part_name(1),
        p2 = part_name(2),
        rhs = rhs,
    );
    let r = run_r(&columns, &body);
    let coef_flat = r.vector("coef");
    let ncoef = r.scalar("ncoef") as usize;
    assert_eq!(
        ncoef,
        K * p_cols,
        "DirichletReg returned {ncoef} coefficients, expected K*p = {}",
        K * p_cols
    );

    // Reconstruct DirichletReg η_k on the grid from the emitted coefficients
    // (block k occupies coef_flat[k*p .. (k+1)*p], aligned to our X columns).
    let dr_eta_grid: Vec<Vec<f64>> = (0..K)
        .map(|k| {
            let beta = Array1::from_iter(coef_flat[k * p_cols..(k + 1) * p_cols].iter().copied());
            grid_design.dot(&beta).to_vec()
        })
        .collect();
    let (dr_mu_grid, dr_logconc_grid) = closed_means_and_logconc(&dr_eta_grid);

    // ---- TRUTH on the grid -------------------------------------------------
    // The data was generated from the known smooth surfaces η_k = true_eta(x);
    // the objective target is the resulting closed mean composition and
    // log-concentration evaluated on the identical dense grid.
    let true_eta_grid: Vec<Vec<f64>> = {
        let mut stacks: Vec<Vec<f64>> = (0..K).map(|_| Vec::with_capacity(N_GRID)).collect();
        for &xg in &grid {
            let e = true_eta(xg);
            for k in 0..K {
                stacks[k].push(e[k]);
            }
        }
        stacks
    };
    let (true_mu_grid, true_logconc_grid) = closed_means_and_logconc(&true_eta_grid);

    // Axis-stacked closed mean composition for grid-wide RMSE.
    let mut gam_mu_flat = Vec::with_capacity(K * N_GRID);
    let mut dr_mu_flat = Vec::with_capacity(K * N_GRID);
    let mut true_mu_flat = Vec::with_capacity(K * N_GRID);
    for k in 0..K {
        gam_mu_flat.extend_from_slice(&gam_mu_grid[k]);
        dr_mu_flat.extend_from_slice(&dr_mu_grid[k]);
        true_mu_flat.extend_from_slice(&true_mu_grid[k]);
    }

    // ---- objective metric: recovery error vs the KNOWN truth ----------------
    let gam_rmse_mu = rmse(&gam_mu_flat, &true_mu_flat);
    let dr_rmse_mu = rmse(&dr_mu_flat, &true_mu_flat);
    let gam_rmse_logconc = rmse(&gam_logconc_grid, &true_logconc_grid);
    let dr_rmse_logconc = rmse(&dr_logconc_grid, &true_logconc_grid);

    // Context only (NOT a pass gate): how closely gam's fit tracks DirichletReg's.
    let rel_mu_vs_dr = relative_l2(&gam_mu_flat, &dr_mu_flat);
    let rel_logconc_vs_dr = relative_l2(&gam_logconc_grid, &dr_logconc_grid);

    eprintln!(
        "dirichlet truth recovery: n={N} K={K} grid={N_GRID} p={p_cols} | \
         RMSE(mu) gam={gam_rmse_mu:.4} dr={dr_rmse_mu:.4} | \
         RMSE(log_phi) gam={gam_rmse_logconc:.4} dr={dr_rmse_logconc:.4} | \
         context rel_l2_vs_dr mu={rel_mu_vs_dr:.4} log_phi={rel_logconc_vs_dr:.4}"
    );

    // PRIMARY claim: gam recovers the true compositional surface. The simplex
    // coordinates μ_k ∈ (0,1); recovering the smooth truth from n=150 noisy
    // Dirichlet draws to within 0.05 RMSE (5% of the unit coordinate range) is
    // a genuine accuracy bar that an actual defect in the coupled multi-block
    // Dirichlet path would fail.
    assert!(
        gam_rmse_mu < 0.05,
        "gam did not recover the true mean composition: RMSE(mu)={gam_rmse_mu:.4} (bar 0.05)"
    );
    // The log-concentration log φ ranges over roughly [0, 2] across the grid;
    // 0.30 RMSE recovers its smooth shape well.
    assert!(
        gam_rmse_logconc < 0.30,
        "gam did not recover the true log-concentration: RMSE(log_phi)={gam_rmse_logconc:.4} (bar 0.30)"
    );

    // MATCH-OR-BEAT: gam's penalized REML fit must be at least as accurate as the
    // mature unpenalized DirichletReg MLE on the same data, up to a 10% margin.
    assert!(
        gam_rmse_mu <= dr_rmse_mu * 1.10,
        "gam mean-composition recovery worse than DirichletReg baseline: \
         gam={gam_rmse_mu:.4} > 1.10*dr={:.4}",
        dr_rmse_mu * 1.10
    );
    assert!(
        gam_rmse_logconc <= dr_rmse_logconc * 1.10,
        "gam log-concentration recovery worse than DirichletReg baseline: \
         gam={gam_rmse_logconc:.4} > 1.10*dr={:.4}",
        dr_rmse_logconc * 1.10
    );
}

/// Stable column header for simplex part `k` (`p1`, `p2`, `p3`).
fn part_name(k: usize) -> &'static str {
    match k {
        0 => "p1",
        1 => "p2",
        2 => "p3",
        _ => panic!("only K={K} parts are defined"),
    }
}

/// Stable basis-column header `bNN`, used identically when handing columns to R
/// and when building the R formula RHS.
fn basis_col_name(j: usize) -> String {
    format!("b{j:02}")
}

// =============================================================================
// REAL-DATA ARM
// =============================================================================
//
// Source: Aitchison, J. (1986), *The Statistical Analysis of Compositional
// Data*, Chapman & Hall — the "Skye AFM lavas" dataset (also shipped as
// `compositions::SkyeAFM` in R). 23 lava specimens from the Isle of Skye, each a
// 3-part composition `(A, F, M)` (alkali / iron-oxide / magnesia weight %, each
// row summing to 100) ordered along the magmatic differentiation series. This is
// genuine compositional regression with NO known generating truth, so the
// objective quality of a fit is its OUT-OF-SAMPLE distributional fit, scored by
// held-out Dirichlet negative log-likelihood (deviance per the Dirichlet
// likelihood). The smooth covariate is the exogenous normalized differentiation
// index `t ∈ [0,1]` (sampling order along the series — metadata, NOT a simplex
// part), so predictor and response never share information. The SAME shared
// cubic-P-spline basis `X = [1 | B_centered(t)]` is handed to BOTH engines; gam
// fits the coupled K-block Dirichlet common model by REML, DirichletReg fits the
// identical likelihood by unpenalized ML. gam must clear an ABSOLUTE held-out
// NLL bar AND match-or-beat DirichletReg's held-out NLL within a small margin.

/// Per-composition Dirichlet negative log-density in the common (α) param.
/// `−ln p(y | α) = −lnΓ(α₀) + Σ_k lnΓ(α_k) − Σ_k (α_k − 1) ln y_k`,
/// with `α₀ = Σ_k α_k`. `eta[k][i]` is the linear predictor `η_k = ln α_k` of
/// row `i`; `y[k][i]` is the observed proportion of part `k`. Returns the mean
/// NLL across the supplied rows (a proper, strictly-Dirichlet held-out score).
fn mean_dirichlet_nll(eta: &[Vec<f64>], y: &[Vec<f64>]) -> f64 {
    let m = eta[0].len();
    let mut total = 0.0;
    for i in 0..m {
        let mut a = [0.0f64; K];
        let mut a0 = 0.0;
        for k in 0..K {
            a[k] = eta[k][i].exp();
            a0 += a[k];
        }
        let mut nll = -ln_gamma(a0);
        for k in 0..K {
            nll += ln_gamma(a[k]) - (a[k] - 1.0) * y[k][i].ln();
        }
        total += nll;
    }
    total / m as f64
}

#[test]
fn gam_dirichlet_regression_recovers_truth_on_real_data() {
    gam::init_parallelism();

    // ---- load the Skye AFM lavas composition (A,F,M weight %) --------------
    let ds = load_csvwith_inferred_schema(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/datasets/skye_afm_lavas.csv"
    )))
    .expect("load skye_afm_lavas.csv");
    let col = ds.column_map();
    let a_idx = col["A"];
    let f_idx = col["F"];
    let m_idx = col["M"];
    let a_raw: Vec<f64> = ds.values.column(a_idx).to_vec();
    let f_raw: Vec<f64> = ds.values.column(f_idx).to_vec();
    let m_raw: Vec<f64> = ds.values.column(m_idx).to_vec();
    let n = a_raw.len();
    assert!(n == 23, "skye_afm_lavas should have 23 rows, got {n}");

    // Close each row to the unit simplex (parts are weight %, summing to ~100);
    // floor positivity so every part is strictly inside the open simplex.
    let raw = [a_raw, f_raw, m_raw];
    let mut comp: Vec<Vec<f64>> = vec![vec![0.0; n]; K];
    for i in 0..n {
        let s: f64 = (0..K).map(|k| raw[k][i]).sum();
        let mut row_sum = 0.0;
        for k in 0..K {
            comp[k][i] = (raw[k][i] / s).max(1e-9);
            row_sum += comp[k][i];
        }
        for k in 0..K {
            comp[k][i] /= row_sum;
        }
    }

    // Exogenous differentiation index t ∈ [0,1] = normalized sampling order
    // along the magmatic series (the natural AFM ordinate). This is metadata,
    // never a simplex part, so predictor and response are independent.
    let t_all: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();

    // ---- deterministic train/test split: every 4th row held out -----------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    let n_train = train_rows.len();
    let n_test = test_rows.len();
    assert!(
        n_train >= 15 && n_test >= 5,
        "split sizes: train={n_train} test={n_test}"
    );

    let train_comp: Vec<Vec<f64>> = (0..K)
        .map(|k| train_rows.iter().map(|&i| comp[k][i]).collect())
        .collect();
    let test_comp: Vec<Vec<f64>> = (0..K)
        .map(|k| test_rows.iter().map(|&i| comp[k][i]).collect())
        .collect();
    let t_train: Vec<f64> = train_rows.iter().map(|&i| t_all[i]).collect();
    let t_test: Vec<f64> = test_rows.iter().map(|&i| t_all[i]).collect();

    // ---- shared basis over [train ; test] so columns + centering match -----
    // Reuse the exact P-spline construction the synthetic arm uses; the basis is
    // built on the concatenation so fitting (train rows) and prediction (test
    // rows) read identical, data-shared columns. With only 15 train rows we use
    // far fewer internal knots — that lives inside `pspline_real_parts` below.
    let mut t_concat = t_train.clone();
    t_concat.extend_from_slice(&t_test);
    let (train_design, penalties, nullspace_dims, test_design) =
        pspline_real_parts(&t_concat, n_train);
    let p_cols = train_design.ncols();

    // ---- gam: K-block Dirichlet common-parameterization custom family ------
    let log_y: Vec<Array1<f64>> = (0..K)
        .map(|k| Array1::from_iter(train_comp[k].iter().map(|&v| v.ln())))
        .collect();
    let family = DirichletCommonFamily { log_y };

    let n_pen = penalties.len();
    let specs: Vec<ParameterBlockSpec> = (0..K)
        .map(|k| ParameterBlockSpec {
            name: format!("alpha{k}"),
            design: DesignMatrix::from(train_design.clone()),
            offset: Array1::zeros(n_train),
            penalties: penalties.clone(),
            nullspace_dims: nullspace_dims.clone(),
            initial_log_lambdas: Array1::zeros(n_pen),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        })
        .collect();

    let options = BlockwiseFitOptions {
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let fit = fit_custom_family(&family, &specs, &options).expect("gam Dirichlet fit (skye)");
    assert!(
        fit.outer_converged,
        "gam Dirichlet outer optimization did not converge on skye data"
    );

    // gam fitted η_k on the HELD-OUT rows (reconstruct from per-block β̂).
    let gam_eta_test: Vec<Vec<f64>> = (0..K)
        .map(|k| test_design.dot(&fit.block_states[k].beta).to_vec())
        .collect();

    // ---- DirichletReg: mature Dirichlet-regression reference, SAME data ----
    // Both engines get X = [1 | B_centered(t)]; `-1` drops DirichletReg's own
    // intercept so its per-part coefficients align 1:1 with our X columns. We
    // pass the TRAIN design + train composition, fit the common model, emit the
    // K·p coefficient matrix, then reconstruct η_k = X_test β_k on the held-out
    // rows in Rust from the identical test design.
    let col_storage: Vec<Vec<f64>> = (0..p_cols)
        .map(|j| (0..n_train).map(|i| train_design[[i, j]]).collect())
        .collect();
    let basis_names: Vec<String> = (0..p_cols).map(basis_col_name).collect();
    let mut columns: Vec<Column<'_>> = Vec::with_capacity(K + p_cols);
    for k in 0..K {
        columns.push(Column::new(part_name(k), &train_comp[k]));
    }
    for (name, c) in basis_names.iter().zip(col_storage.iter()) {
        columns.push(Column::new(name.as_str(), c));
    }

    let rhs = basis_names.join(" + ");
    let body = format!(
        r#"
        suppressPackageStartupMessages(library(DirichletReg))
        Y <- DR_data(cbind(df${p0}, df${p1}, df${p2}))
        m <- DirichletReg(Y ~ -1 + {rhs}, data = df)
        co <- coef(m)
        flat <- as.numeric(unlist(co))
        emit("coef", flat)
        emit("ncoef", length(flat))
        "#,
        p0 = part_name(0),
        p1 = part_name(1),
        p2 = part_name(2),
        rhs = rhs,
    );
    let r = run_r(&columns, &body);
    let coef_flat = r.vector("coef");
    let ncoef = r.scalar("ncoef") as usize;
    assert_eq!(
        ncoef,
        K * p_cols,
        "DirichletReg returned {ncoef} coefficients, expected K*p = {}",
        K * p_cols
    );
    let dr_eta_test: Vec<Vec<f64>> = (0..K)
        .map(|k| {
            let beta = Array1::from_iter(coef_flat[k * p_cols..(k + 1) * p_cols].iter().copied());
            test_design.dot(&beta).to_vec()
        })
        .collect();

    // ---- objective metric: held-out Dirichlet NLL (truth unknown) ----------
    let gam_test_nll = mean_dirichlet_nll(&gam_eta_test, &test_comp);
    let dr_test_nll = mean_dirichlet_nll(&dr_eta_test, &test_comp);

    // Context only (NOT a pass gate): closeness of gam's vs DirichletReg's
    // held-out linear predictors, axis-stacked over the K parts.
    let mut gam_eta_flat = Vec::with_capacity(K * n_test);
    let mut dr_eta_flat = Vec::with_capacity(K * n_test);
    for k in 0..K {
        gam_eta_flat.extend_from_slice(&gam_eta_test[k]);
        dr_eta_flat.extend_from_slice(&dr_eta_test[k]);
    }
    let rel_eta_vs_dr = relative_l2(&gam_eta_flat, &dr_eta_flat);

    eprintln!(
        "skye AFM held-out Dirichlet NLL: n_train={n_train} n_test={n_test} p={p_cols} | \
         gam_nll={gam_test_nll:.4} dr_nll={dr_test_nll:.4} | \
         context rel_l2(eta) vs dr={rel_eta_vs_dr:.4}"
    );

    // ---- PRIMARY objective assertion: absolute held-out NLL bar ------------
    // The Skye AFM composition is moderately concentrated (parts away from the
    // simplex edges), so a competent Dirichlet fit attains a clearly negative
    // mean held-out NLL (the log-density is positive once φ = Σα is appreciable).
    // A flat, mis-concentrated, or broken coupled-block fit would push the NLL
    // up toward / above zero. A mean held-out NLL below -3.0 is a genuine
    // distributional-fit bar that a defect in the coupled K-block Dirichlet path
    // would fail.
    assert!(
        gam_test_nll < -3.0,
        "gam held-out Dirichlet NLL too high (poor distributional fit): \
         {gam_test_nll:.4} (bar -3.0)"
    );

    // ---- BASELINE (match-or-beat): no worse than DirichletReg held-out NLL --
    // gam's penalized REML fit must be at least as good as the mature unpenalized
    // DirichletReg ML fit on the same held-out rows, up to a 10% margin. NLL is
    // negative here, so "no more than 10% worse" means not exceeding
    // dr_nll + 0.10*|dr_nll|.
    let dr_bar = dr_test_nll + 0.10 * dr_test_nll.abs();
    assert!(
        gam_test_nll <= dr_bar,
        "gam held-out Dirichlet NLL worse than DirichletReg baseline: \
         gam={gam_test_nll:.4} > dr+10%={dr_bar:.4} (dr={dr_test_nll:.4})"
    );
}

/// Real-data variant of `pspline_parts`: builds the shared `[1 | B_centered]`
/// P-spline design over `t_all = [train ; test]` and splits at `n_train`. The
/// Skye series has only ~15 training rows, so the basis uses a small number of
/// internal knots to stay well-posed; everything else (sum-to-zero centering,
/// zero-padded 2nd-difference penalty, +1 null dim for the intercept) mirrors
/// the synthetic arm so both engines see identical columns.
fn pspline_real_parts(
    t_all: &[f64],
    n_train: usize,
) -> (Array2<f64>, Vec<PenaltyMatrix>, Vec<usize>, Array2<f64>) {
    let t_arr = Array1::from_vec(t_all.to_vec());
    let lo = t_all.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = t_all.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Generate {
            data_range: (lo, hi),
            num_internal_knots: 3,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::WeightedSumToZero { weights: None },
        boundary: Default::default(),
        boundary_conditions: Default::default(),
    };
    let basis = build_bspline_basis_1d(t_arr.view(), &spec).expect("build skye P-spline basis");
    let b_all = basis.design.to_dense();
    let p_s = b_all.ncols();
    let p = p_s + 1;

    let n_all = t_all.len();
    let mut full = Array2::<f64>::zeros((n_all, p));
    for i in 0..n_all {
        full[[i, 0]] = 1.0;
        for j in 0..p_s {
            full[[i, j + 1]] = b_all[[i, j]];
        }
    }

    let mut penalties = Vec::with_capacity(basis.penalties.len());
    let mut nullspace_dims = Vec::with_capacity(basis.penalties.len());
    for (k, s_basis) in basis.penalties.iter().enumerate() {
        assert!(
            s_basis.nrows() == p_s && s_basis.ncols() == p_s,
            "skye penalty {k} shape {:?} != {p_s}x{p_s}",
            s_basis.shape()
        );
        let mut s = Array2::<f64>::zeros((p, p));
        for r in 0..p_s {
            for c in 0..p_s {
                s[[r + 1, c + 1]] = s_basis[[r, c]];
            }
        }
        penalties.push(PenaltyMatrix::from(s));
        let base_null = basis.nullspace_dims.get(k).copied().unwrap_or(0);
        nullspace_dims.push(base_null + 1);
    }

    let train = full.slice(ndarray::s![0..n_train, ..]).to_owned();
    let test = full.slice(ndarray::s![n_train.., ..]).to_owned();
    (train, penalties, nullspace_dims, test)
}
