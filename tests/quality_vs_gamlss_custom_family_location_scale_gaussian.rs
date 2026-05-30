//! End-to-end quality: gam's multi-block **custom family** (GAMLSS-style
//! Gaussian location-scale) must RECOVER the data-generating truth on identical
//! data, matching-or-beating `gamlss` (the mature distributional-regression
//! reference) on recovery accuracy.
//!
//! A Gaussian location-scale model fits two *independent* coefficient blocks,
//! `β_μ` and `β_σ`, each driving its own smooth predictor through its own
//! design matrix:
//!
//!   μ(x) = X_μ β_μ ,   log σ(x) = X_σ β_σ ,
//!   ℓ = Σ_i [ −log σ_i − ½ (y_i − μ_i)² / σ_i² ]   (constants omitted).
//!
//! Unlike a single-response GAM, the joint Hessian couples the two blocks
//! off-diagonally (∂²ℓ/∂β_μ∂β_σ ≠ 0), and the penalized outer optimization
//! must select two smoothing parameters jointly. This is the canonical test
//! of gam's `fit_custom_family` multi-block path: off-diagonal Hessian
//! assembly + penalized outer selection on a true two-parameter family.
//!
//! OBJECTIVE METRIC (truth recovery — the data are simulated from a KNOWN
//! μ(x)=1+sin(x), log σ(x)=0.2−0.15x): on a 50-point grid we assert that gam's
//! fitted mean tracks the TRUE μ(x) to within a small fraction of the signal
//! range, and gam's fitted log-σ tracks the TRUE log σ(x) to within a tight
//! log-unit bound. Neither bound references gamlss. The PRIMARY claim is that
//! gam recovers the generating functions, not that it mimics another tool.
//!
//! gamlss is fit on the IDENTICAL data and demoted to a BASELINE-TO-MATCH-OR-
//! BEAT: we additionally require gam's recovery error (RMSE-to-truth, for both
//! μ and log σ) to be no worse than gamlss's by more than 10%. That is an
//! accuracy claim against ground truth, not a "reproduce gamlss's noisy fit"
//! claim — the reference's rel-L2 is still printed for context but never gates.

use gam::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, build_bspline_basis_1d,
};
use gam::custom_family::{
    BlockwiseFitOptions, ParameterBlockSpec, PenaltyMatrix, fit_custom_family,
};
use gam::families::gamlss::GaussianLocationScaleFamily;
use gam::matrix::DesignMatrix;
use gam::resource::ResourcePolicy;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use ndarray::{Array1, Array2};

const N: usize = 180;
const N_GRID: usize = 50;
const GRID_LO: f64 = 0.1;
const GRID_HI: f64 = 4.9;

/// True mean function μ(x) = 1 + sin(x).
fn true_mu(x: f64) -> f64 {
    1.0 + x.sin()
}

/// True scale function σ(x) = exp(0.2 − 0.15 x).
fn true_sigma(x: f64) -> f64 {
    (0.2 - 0.15 * x).exp()
}

/// Deterministic standard-normal draws via Box–Muller on a small LCG, so the
/// exact same Y vector is generated once and handed verbatim to both engines.
struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        // Numerical Recipes LCG constants.
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    fn next_unit(&mut self) -> f64 {
        // 53-bit mantissa in (0, 1).
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// 50-point evaluation grid, identical in Rust and R (`seq(GRID_LO, GRID_HI,
/// length.out = N_GRID)`).
fn eval_grid() -> Vec<f64> {
    (0..N_GRID)
        .map(|i| GRID_LO + (GRID_HI - GRID_LO) * (i as f64) / ((N_GRID - 1) as f64))
        .collect()
}

/// Build a P-spline block: design `[1 | B_centered]` (explicit intercept +
/// sum-to-zero–centered cubic B-spline) with a single difference penalty
/// zero-padded over the unpenalized intercept column. The basis is evaluated
/// over `x_all` (train rows followed by grid rows) so the columns — and the
/// data-dependent centering — are identical for fitting and for prediction;
/// we then split the rows. Returns `(train_spec_design, grid_design)`.
fn pspline_block(name: &str, x_all: &[f64]) -> (ParameterBlockSpec, Array2<f64>) {
    let x_arr = Array1::from_vec(x_all.to_vec());
    let lo = x_all.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = x_all.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Generate {
            data_range: (lo, hi),
            num_internal_knots: 12,
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

    // Full design over all rows: column 0 = intercept, columns 1.. = centered basis.
    let n_all = x_all.len();
    let mut full = Array2::<f64>::zeros((n_all, p));
    for i in 0..n_all {
        full[[i, 0]] = 1.0;
        for j in 0..p_s {
            full[[i, j + 1]] = b_all[[i, j]];
        }
    }

    // Zero-pad each basis penalty over the intercept column.
    let mut penalties = Vec::with_capacity(basis.penalties.len());
    let mut nullspace_dims = Vec::with_capacity(basis.penalties.len());
    for (k, s_basis) in basis.penalties.iter().enumerate() {
        assert!(
            s_basis.nrows() == p_s && s_basis.ncols() == p_s,
            "{name} penalty {k} shape {:?} != {p_s}x{p_s}",
            s_basis.shape()
        );
        let mut s = Array2::<f64>::zeros((p, p));
        for r in 0..p_s {
            for c in 0..p_s {
                s[[r + 1, c + 1]] = s_basis[[r, c]];
            }
        }
        penalties.push(PenaltyMatrix::from(s));
        // The padded penalty gains exactly one extra null direction (the
        // unpenalized intercept column) on top of the basis penalty's own
        // structural null space (linear trend for a 2nd-order difference).
        let base_null = basis.nullspace_dims.get(k).copied().unwrap_or(0);
        nullspace_dims.push(base_null + 1);
    }
    let n_pen = penalties.len();

    let train = full.slice(ndarray::s![0..N, ..]).to_owned();
    let grid = full.slice(ndarray::s![N.., ..]).to_owned();

    let block = ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::from(train),
        offset: Array1::zeros(N),
        penalties,
        nullspace_dims,
        initial_log_lambdas: Array1::zeros(n_pen),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    (block, grid)
}

/// Joint Gaussian location-scale log-likelihood (constants omitted), the exact
/// objective both engines maximize.
fn gaussian_locscale_ll(y: &[f64], mu: &[f64], sigma: &[f64]) -> f64 {
    y.iter()
        .zip(mu)
        .zip(sigma)
        .map(|((&yi, &mi), &si)| {
            let r = yi - mi;
            -si.ln() - 0.5 * r * r / (si * si)
        })
        .sum()
}

#[test]
fn gam_custom_family_location_scale_matches_gamlss() {
    gam::init_parallelism();

    // ---- synthetic data: X ~ Unif(0,5), Y ~ N(1+sin X, exp(0.2-0.15X)²) ----
    let mut rng = Lcg::new(0x5A1E_C0FF_EE12_3456);
    let mut x: Vec<f64> = (0..N).map(|_| 5.0 * rng.next_unit()).collect();
    // Sort so the smooth's knot span is well covered and R/Rust see the same rows.
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite X"));
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| true_mu(xi) + true_sigma(xi) * rng.next_normal())
        .collect();

    let grid = eval_grid();

    // x_all = [train ; grid] so basis columns + centering are shared.
    let mut x_all = x.clone();
    x_all.extend_from_slice(&grid);

    // ---- gam: two-block Gaussian location-scale custom family --------------
    let (spec_mu, grid_mu) = pspline_block("mu", &x_all);
    let (spec_sigma, grid_sigma) = pspline_block("log_sigma", &x_all);
    let specs = vec![spec_mu, spec_sigma];

    let y_arr = Array1::from_vec(y.clone());
    let family = GaussianLocationScaleFamily {
        y: y_arr.clone(),
        weights: Array1::ones(N),
        mu_design: Some(specs[GaussianLocationScaleFamily::BLOCK_MU].design.clone()),
        log_sigma_design: Some(
            specs[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA]
                .design
                .clone(),
        ),
        policy: ResourcePolicy::default_library(),
        cached_row_scalars: std::sync::RwLock::new(None),
    };

    let options = BlockwiseFitOptions {
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let fit = fit_custom_family(&family, &specs, &options).expect("gam location-scale fit");
    assert!(
        fit.outer_converged,
        "gam location-scale outer optimization did not converge"
    );

    let beta_mu = &fit.block_states[GaussianLocationScaleFamily::BLOCK_MU].beta;
    let beta_ls = &fit.block_states[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA].beta;

    // Predictions on the 50-point grid (identity link for μ, log link for σ).
    // For σ we keep the predictor on the LOG scale: log σ(x) = X_σ β_σ. Comparing
    // on the log scale is the well-conditioned choice (see the bounds below).
    let gam_mu_grid: Vec<f64> = grid_mu.dot(beta_mu).to_vec();
    let gam_log_sigma_grid: Vec<f64> = grid_sigma.dot(beta_ls).to_vec();

    // Fitted (μ, σ) at the *training* points, for the LL comparison.
    let gam_mu_train: Vec<f64> = fit.block_states[GaussianLocationScaleFamily::BLOCK_MU]
        .eta
        .to_vec();
    let gam_sigma_train: Vec<f64> = fit.block_states[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA]
        .eta
        .mapv(f64::exp)
        .to_vec();
    let gam_ll = gaussian_locscale_ll(&y, &gam_mu_train, &gam_sigma_train);

    // ---- gamlss: the mature distributional-regression reference ------------
    let r = run_r(
        &[Column::new("X", &x), Column::new("Y", &y)],
        r#"
        suppressPackageStartupMessages(library(gamlss))
        ctrl <- gamlss.control(trace = FALSE)
        m <- gamlss(Y ~ pb(X), sigma.fo = ~ pb(X), family = NO(), data = df, control = ctrl)

        # Fitted (mu, sigma) on the training rows (response scale: sigma uses log link).
        emit("mu_train", as.numeric(fitted(m, "mu")))
        emit("sigma_train", as.numeric(fitted(m, "sigma")))

        # Predictions on the identical 50-point evaluation grid.
        grid <- seq(0.1, 4.9, length.out = 50)
        nd <- data.frame(X = grid)
        mu_g <- as.numeric(predict(m, what = "mu", newdata = nd, type = "response", data = df))
        sig_g <- as.numeric(predict(m, what = "sigma", newdata = nd, type = "response", data = df))
        emit("mu_grid", mu_g)
        emit("sigma_grid", sig_g)
        "#,
    );
    let gamlss_mu_grid = r.vector("mu_grid");
    let gamlss_sigma_grid = r.vector("sigma_grid");
    let gamlss_mu_train = r.vector("mu_train");
    let gamlss_sigma_train = r.vector("sigma_train");
    assert_eq!(gamlss_mu_grid.len(), N_GRID, "gamlss mu grid length");
    assert_eq!(gamlss_sigma_grid.len(), N_GRID, "gamlss sigma grid length");
    assert_eq!(gamlss_mu_train.len(), N, "gamlss mu train length");

    let gamlss_ll = gaussian_locscale_ll(&y, gamlss_mu_train, gamlss_sigma_train);

    // ---- ground truth on the evaluation grid -------------------------------
    // The data were generated from KNOWN functions, so the objective claim is
    // recovery of those functions. We compare in the natural link space both
    // engines optimize: μ on the identity scale, σ on the log scale (a constant
    // offset in log σ is a multiplicative factor in σ, so log-unit error is the
    // well-conditioned, scale-free measure of scale recovery).
    let truth_mu_grid: Vec<f64> = grid.iter().map(|&xi| true_mu(xi)).collect();
    let truth_log_sigma_grid: Vec<f64> = grid.iter().map(|&xi| true_sigma(xi).ln()).collect();
    // gamlss emits σ on the response scale; lift it to the log scale.
    let gamlss_log_sigma_grid: Vec<f64> = gamlss_sigma_grid.iter().map(|&s| s.ln()).collect();

    // ---- OBJECTIVE metric: recovery error against the generating truth ------
    let gam_mu_err = rmse(&gam_mu_grid, &truth_mu_grid);
    let gam_log_sigma_err = rmse(&gam_log_sigma_grid, &truth_log_sigma_grid);
    let gamlss_mu_err = rmse(gamlss_mu_grid, &truth_mu_grid);
    let gamlss_log_sigma_err = rmse(&gamlss_log_sigma_grid, &truth_log_sigma_grid);

    // Context only (never gates): how close the two fitted functions sit.
    let rel_mu_vs_gamlss = relative_l2(&gam_mu_grid, gamlss_mu_grid);
    let ll_abs_rel = (gam_ll - gamlss_ll).abs() / gamlss_ll.abs().max(1.0);

    eprintln!(
        "gaussian location-scale truth recovery: n={N} grid={N_GRID} \
         gam_mu_rmse_to_truth={gam_mu_err:.4} gamlss_mu_rmse_to_truth={gamlss_mu_err:.4} \
         gam_logsig_rmse_to_truth={gam_log_sigma_err:.4} gamlss_logsig_rmse_to_truth={gamlss_log_sigma_err:.4} \
         (ctx: rel_l2_mu_vs_gamlss={rel_mu_vs_gamlss:.4} gam_ll={gam_ll:.3} gamlss_ll={gamlss_ll:.3} ll_rel={ll_abs_rel:.4})"
    );

    // (1) ABSOLUTE truth recovery — the primary claim.
    // μ. The mean signal μ(x)=1+sin(x) spans ≈2 units (range of sin over
    // [0.1,4.9] is from −1 to +1). At the smaller true σ (σ≈exp(0.2)≈1.22 down
    // to exp(0.2−0.15·4.9)≈0.59) and n=180, a correctly penalized smooth must
    // pin μ to well under a fifth of the signal amplitude. 0.20 RMSE is ~10% of
    // the 2-unit signal range and comfortably below the per-point noise σ; a
    // location block that mis-assembles the off-diagonal Hessian cannot hit it.
    assert!(
        gam_mu_err < 0.20,
        "gam failed to recover true μ(x): rmse_to_truth={gam_mu_err:.4} (signal range ≈2)"
    );
    // log σ. The TRUE log σ(x)=0.2−0.15x ranges over ≈0.74 log units across the
    // grid. log σ is a second-moment quantity estimated from n=180 squared
    // residuals, so it is intrinsically noisier than the mean; recovering it to
    // within 0.25 log units RMSE (< a third of the signal swing, i.e. better
    // than a ~28% multiplicative σ error on average) demonstrates the scale
    // block genuinely tracks the heteroscedastic envelope rather than collapsing
    // to a homoscedastic constant.
    assert!(
        gam_log_sigma_err < 0.25,
        "gam failed to recover true log σ(x): rmse_to_truth={gam_log_sigma_err:.4} (signal swing ≈0.74)"
    );

    // (2) MATCH-OR-BEAT the mature reference ON ACCURACY (not on output).
    // gamlss is a trusted distributional-regression engine fit on the identical
    // data; gam's recovery error must be no worse than gamlss's by more than
    // 10%. This is an accuracy comparison against ground truth — beating gamlss
    // means gam's fit is genuinely closer to the truth, which is a real quality
    // win, not mimicry of gamlss's (itself noisy) fitted curve.
    assert!(
        gam_mu_err <= gamlss_mu_err * 1.10,
        "gam's μ recovery worse than gamlss: gam={gam_mu_err:.4} gamlss={gamlss_mu_err:.4}"
    );
    assert!(
        gam_log_sigma_err <= gamlss_log_sigma_err * 1.10,
        "gam's log σ recovery worse than gamlss: gam={gam_log_sigma_err:.4} gamlss={gamlss_log_sigma_err:.4}"
    );
}
