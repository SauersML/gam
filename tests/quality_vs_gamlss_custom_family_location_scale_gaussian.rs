//! End-to-end quality: gam's multi-block **custom family** (GAMLSS-style
//! Gaussian location-scale) must match `gamlss` — the mature reference for
//! distributional regression — on identical data.
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
//! We fit the SAME synthetic data with both engines:
//!   * gam : `fit_custom_family(&GaussianLocationScaleFamily{..}, [spec_μ, spec_σ])`
//!   * R   : `gamlss(Y ~ pb(X), sigma.fo = ~ pb(X), family = NO)`
//!
//! and assert that the fitted mean function μ(x) and scale function σ(x) agree
//! pointwise over a 50-point grid, and that the achieved Gaussian
//! log-likelihood (recomputed from each engine's fitted (μ, σ) with the
//! identical formula above) agrees. Both engines penalize a P-spline by a
//! REML/GAIC-selected smoothing parameter, so close agreement is the correct
//! expectation; a real divergence is a real bug in gam's location-scale path.

use gam::basis::{BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, build_bspline_basis_1d};
use gam::custom_family::{
    BlockwiseFitOptions, ParameterBlockSpec, PenaltyMatrix, fit_custom_family,
};
use gam::families::gamlss::GaussianLocationScaleFamily;
use gam::matrix::DesignMatrix;
use gam::resource::ResourcePolicy;
use gam::test_support::reference::{Column, relative_l2, run_r};
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
    let gam_mu_grid: Vec<f64> = grid_mu.dot(beta_mu).to_vec();
    let gam_sigma_grid: Vec<f64> = grid_sigma.dot(beta_ls).mapv(f64::exp).to_vec();

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

    // ---- compare -----------------------------------------------------------
    let rel_mu = relative_l2(&gam_mu_grid, gamlss_mu_grid);
    let rel_sigma = relative_l2(&gam_sigma_grid, gamlss_sigma_grid);
    let ll_abs_rel = (gam_ll - gamlss_ll).abs() / gamlss_ll.abs().max(1.0);

    eprintln!(
        "gaussian location-scale vs gamlss: n={N} grid={N_GRID} \
         rel_l2_mu={rel_mu:.4} rel_l2_sigma={rel_sigma:.4} \
         gam_ll={gam_ll:.3} gamlss_ll={gamlss_ll:.3} ll_rel={ll_abs_rel:.4}"
    );

    // Both engines fit a penalized cubic P-spline for μ and for log σ on the
    // SAME data by a smoothing-parameter–selected objective, so the fitted
    // functions must essentially coincide. The bounds (1.5% relative L2 on μ
    // and σ over the grid; 1% relative LL) are the SPEC tolerances: tight
    // enough that any real defect in the off-diagonal Hessian assembly or the
    // joint penalized outer selection would trip them, while still allowing
    // for the two engines' differing basis/knot/selection conventions.
    assert!(
        rel_mu <= 0.015,
        "fitted μ diverges from gamlss: rel_l2={rel_mu:.4}"
    );
    assert!(
        rel_sigma <= 0.015,
        "fitted σ diverges from gamlss: rel_l2={rel_sigma:.4}"
    );
    assert!(
        ll_abs_rel <= 0.01,
        "log-likelihood disagrees with gamlss: gam={gam_ll:.3} gamlss={gamlss_ll:.3} (rel={ll_abs_rel:.4})"
    );
}
