//! End-to-end quality: gam's **simplex (compositional) response regression**
//! via a custom Dirichlet likelihood must match `DirichletReg` — the mature R
//! reference for Dirichlet-response regression — on identical data.
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
//! DirichletReg by unpenalized maximum likelihood. With a modest basis (8
//! interior knots) and a strong signal at n=150, the penalized and unpenalized
//! fits must produce essentially the same fitted composition surface; a real
//! divergence is a real bug in gam's coupled multi-block custom-family path.
//! (`-1` suppresses DirichletReg's own intercept so its per-component
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
//!   * **Relative L2** of the fitted *closed* mean composition `μ(x)` over a
//!     dense grid (the natural scale-free compositional accuracy measure).
//!   * **Pearson correlation** of the recovered ALR coordinates between the two
//!     engines, per ALR axis (the Aitchison-coordinate agreement).
//!   * **Relative L2** of the fitted log-concentration `log φ(x) = log Σ_k α_k`.

use gam::basis::{BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, build_bspline_basis_1d};
use gam::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, FamilyEvaluation, ParameterBlockSpec,
    ParameterBlockState, PenaltyMatrix, fit_custom_family,
};
use gam::matrix::DesignMatrix;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Uniform};
use statrs::function::gamma::{digamma, ln_gamma};

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
        + inv * (1.0
            + inv * (0.5
                + inv * (1.0 / 6.0
                    + inv2 * (-1.0 / 30.0 + inv2 * (1.0 / 42.0 - inv2 / 30.0)))))
}

/// True smooth log-α surfaces η_k(x), k = 0..K-1, in the Dirichlet common
/// parameterization. Distinct shapes so the location (ALR) and concentration
/// (Σα) both genuinely vary with x.
fn true_eta(x: f64) -> [f64; K] {
    [
        0.6 + 0.9 * x.sin(),                      // part 1
        0.2 - 0.5 * x + 0.4 * (1.3 * x).cos(),    // part 2
        -0.3 + 0.7 * (0.8 * x).sin() + 0.15 * x,  // part 3
    ]
}

/// Build a shared P-spline block design `[1 | B_centered]` (explicit intercept
/// + sum-to-zero–centered cubic B-spline) with the second-difference penalty
/// zero-padded over the unpenalized intercept column. The basis is built over
/// `x_all = [train ; grid]` so the columns and the data-dependent centering are
/// identical for fitting and prediction; rows are then split. Returns the
/// `(train_design, penalties, nullspace_dims, grid_design)` parts shared by
/// every Dirichlet block.
fn pspline_parts(
    x_all: &[f64],
) -> (Array2<f64>, Vec<PenaltyMatrix>, Vec<usize>, Array2<f64>) {
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

/// ALR coordinates ALRₖ = ln(μ_k/μ_K) = η_k − η_{K-1} for k = 0..K-2, flattened
/// in axis-major order over a grid of length `m`.
fn alr_flat(eta: &[Vec<f64>]) -> Vec<f64> {
    let m = eta[0].len();
    let mut out = Vec::with_capacity((K - 1) * m);
    for k in 0..(K - 1) {
        for i in 0..m {
            out.push(eta[k][i] - eta[K - 1][i]);
        }
    }
    out
}

#[test]
fn gam_dirichlet_regression_matches_dirichletreg() {
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
    let gam_alr_grid = alr_flat(&gam_eta_grid);

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
    let dr_alr_grid = alr_flat(&dr_eta_grid);

    // ---- compare -----------------------------------------------------------
    // Relative L2 of the closed mean composition over the grid (axis-stacked).
    let mut gam_mu_flat = Vec::with_capacity(K * N_GRID);
    let mut dr_mu_flat = Vec::with_capacity(K * N_GRID);
    for k in 0..K {
        gam_mu_flat.extend_from_slice(&gam_mu_grid[k]);
        dr_mu_flat.extend_from_slice(&dr_mu_grid[k]);
    }
    let rel_mu = relative_l2(&gam_mu_flat, &dr_mu_flat);
    let rel_logconc = relative_l2(&gam_logconc_grid, &dr_logconc_grid);
    let corr_alr = pearson(&gam_alr_grid, &dr_alr_grid);

    eprintln!(
        "dirichlet regression vs DirichletReg: n={N} K={K} grid={N_GRID} p={p_cols} \
         rel_l2_mean={rel_mu:.4} rel_l2_logconc={rel_logconc:.4} pearson_alr={corr_alr:.5}"
    );

    // Both engines fit the SAME Dirichlet common-model likelihood on the SAME
    // cubic-B-spline basis; gam adds only a REML smoothness penalty that, with
    // 8 interior knots and a strong signal at n=150, barely shrinks the fit.
    // The fitted closed mean composition must therefore essentially coincide
    // and the Aitchison (ALR) coordinates must be near-perfectly correlated.
    // 3% relative L2 on the mean composition and on log-concentration, and
    // 0.99 ALR correlation, are tight enough that any real defect in the
    // coupled multi-block Dirichlet path would trip them, while leaving margin
    // for the penalized-vs-unpenalized difference between the two engines.
    assert!(
        corr_alr > 0.99,
        "ALR coordinates should be near-identical to DirichletReg: pearson={corr_alr:.5}"
    );
    assert!(
        rel_mu < 0.03,
        "fitted mean composition diverges from DirichletReg: rel_l2={rel_mu:.4}"
    );
    assert!(
        rel_logconc < 0.03,
        "fitted log-concentration diverges from DirichletReg: rel_l2={rel_logconc:.4}"
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
