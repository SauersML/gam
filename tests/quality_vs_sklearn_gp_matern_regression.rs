//! End-to-end OBJECTIVE quality: gam's Gaussian-process Matérn smooth
//! (`matern(x, nu=2.5)`) must RECOVER the known deterministic function the data
//! were generated from — and do so at least as accurately as the Python
//! reference exact-GP regressor (`sklearn.gaussian_process.
//! GaussianProcessRegressor` with a `Matern(nu=2.5)` kernel).
//!
//! OBJECTIVE METRIC ASSERTED (truth recovery, case 1):
//!   * PRIMARY: RMSE(gam_fit, TRUTH) on a dense interior grid <= the noise
//!     standard deviation (sigma = 0.1). The data were generated as
//!     y = truth(x) + N(0, sigma^2); a faithful ν = 5/2 GP must denoise back to
//!     within the noise level, i.e. its bias+resolution error must not exceed a
//!     single noise sd. This is a pure ground-truth accuracy claim — it does not
//!     reference any peer tool.
//!   * MATCH-OR-BEAT (accuracy baseline, NOT a "same-as" claim): gam's RMSE to
//!     truth <= 1.10 × sklearn's RMSE to truth. sklearn's GaussianProcessRegressor
//!     is the mature Python exact-GP regressor; we require gam to be at least as
//!     accurate at recovering the truth, never that gam reproduces sklearn's
//!     (itself noisy, possibly mis-tuned) fitted output.
//!
//! Deliberately NOT asserted: closeness of gam's fit to sklearn's fit, and EDF
//! agreement. Matching a peer tool's noisy posterior mean, or its effective
//! complexity, proves nothing about correctness — both could overfit or
//! mis-select length-scale alike. The rel-L2 / Pearson / EDF agreement between
//! the two engines is still COMPUTED and printed for context, but it is not a
//! pass/fail criterion.
//!
//! Why sklearn as the baseline: it is the reference exact-GP regressor in the
//! Python ecosystem, fitting precisely the Matérn ν = 5/2 kernel gam's
//! `matern(x, nu=2.5)` basis targets, selecting its length-scale by maximizing
//! the log marginal likelihood. That makes it a fair, strong accuracy bar to
//! match-or-beat on truth recovery.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, rmse, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

#[test]
fn gam_matern_gp_recovers_truth_and_beats_sklearn() {
    init_parallelism();

    // ---- fixed-seed synthetic data, fed IDENTICALLY to gam and sklearn ----
    // x ~ U[0,1] (n=150, sorted); truth(x) = 2 + 1.5·sin(6π·x) + 0.5·cos(4π·x);
    // y = truth(x) + N(0, 0.1²). The deterministic truth is known exactly.
    let n = 150usize;
    let noise_sd = 0.1_f64;
    let mut rng = StdRng::seed_from_u64(20260529);
    let ux = Uniform::new(0.0, 1.0).expect("uniform [0,1]");
    let noise = Normal::new(0.0, noise_sd).expect("gaussian noise sd=0.1");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
    let truth = |t: f64| {
        2.0 + 1.5 * (6.0 * std::f64::consts::PI * t).sin()
            + 0.5 * (4.0 * std::f64::consts::PI * t).cos()
    };
    let y: Vec<f64> = x
        .iter()
        .map(|&t| truth(t) + noise.sample(&mut rng))
        .collect();

    // ---- fit with gam: y ~ matern(x, nu=2.5, k=25), REML ------------------
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode matern GP dataset");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ matern(x, nu=2.5, k=25)", &ds, &cfg).expect("gam matern GP fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Gaussian GAM fit for matern() GP smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // ---- shared dense evaluation grid: 200 pts in (0.01, 0.99) ------------
    // Interior of [0,1]: keeps the comparison off the GP-kernel boundary, where
    // extrapolation conventions differ between a penalized basis and an exact GP.
    let grid_n = 200usize;
    let x_grid: Vec<f64> = (0..grid_n)
        .map(|i| 0.01 + (0.99 - 0.01) * i as f64 / (grid_n - 1) as f64)
        .collect();

    // gam fitted function at the grid: rebuild the design from the frozen spec
    // (identity link ⇒ design·beta = posterior mean). Headers are [x, y]: x@0.
    let mut g = Array2::<f64>::zeros((grid_n, 2));
    for (i, &t) in x_grid.iter().enumerate() {
        g[[i, 0]] = t;
        g[[i, 1]] = 0.0;
    }
    let grid_design = build_term_collection_design(g.view(), &fit.resolvedspec)
        .expect("rebuild matern GP design at grid points");
    let gam_grid: Vec<f64> = grid_design.design.apply(&fit.fit.beta).to_vec();

    // truth on the grid (known deterministic function, no noise) -----------
    let truth_grid: Vec<f64> = x_grid.iter().map(|&t| truth(t)).collect();

    // ---- fit the SAME data with sklearn GaussianProcessRegressor ----------
    // Matern(nu=2.5) kernel × a fitted constant amplitude + WhiteKernel noise;
    // sklearn maximizes the log marginal likelihood (its default) to pick the
    // length-scale, amplitude, and noise. We predict its posterior mean on
    // x_grid: this is the BASELINE-TO-BEAT for truth-recovery accuracy. We also
    // report sklearn's effective DoF, trace(K (K+σ²I)⁻¹), purely for context.
    let r = run_python(
        &[
            Column::new("x", &x),
            Column::new("y", &y),
            Column::new("xg", &x_grid),
        ],
        r#"
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

x = np.asarray(df["x"], dtype=float).reshape(-1, 1)
y = np.asarray(df["y"], dtype=float).reshape(-1)
xg = np.asarray(df["xg"], dtype=float).reshape(-1, 1)

kernel = (ConstantKernel(1.0, (1e-3, 1e3))
          * Matern(length_scale=0.1, length_scale_bounds=(1e-3, 1e1), nu=2.5)
          + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-6, 1e1)))
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=8,
                              random_state=0)
gp.fit(x, y)

mu = gp.predict(xg)
emit("grid_fit", mu)

# GP effective degrees of freedom: trace(K (K + sigma^2 I)^-1) at training pts.
# Reconstruct from the optimized kernel: split signal kernel from white noise.
k = gp.kernel_
# k = Product(Constant*Matern) + WhiteKernel ; the noise variance lives in the
# WhiteKernel term, the signal kernel is everything else.
white = k.k2
signal = k.k1
Ksig = signal(x)
sigma2 = white.noise_level
n = x.shape[0]
A = Ksig + sigma2 * np.eye(n)
S = Ksig @ np.linalg.solve(A, np.eye(n))   # smoother (hat) matrix
emit("edf", np.trace(S))
"#,
    );
    let sk_grid = r.vector("grid_fit");
    let sk_edf = r.scalar("edf");
    assert_eq!(
        sk_grid.len(),
        grid_n,
        "sklearn grid prediction length mismatch"
    );

    // ---- OBJECTIVE metric: accuracy vs the KNOWN truth --------------------
    let gam_rmse_truth = rmse(&gam_grid, &truth_grid);
    let sk_rmse_truth = rmse(sk_grid, &truth_grid);

    // ---- context only (NOT pass/fail): agreement between the two engines --
    let rel = relative_l2(&gam_grid, sk_grid);
    let corr = pearson(&gam_grid, sk_grid);
    let edf_rel = (gam_edf - sk_edf).abs() / sk_edf.abs().max(1.0);

    eprintln!(
        "matern(x,nu=2.5,k=25) truth-recovery: n={n} grid={grid_n} noise_sd={noise_sd:.3} \
         gam_rmse_truth={gam_rmse_truth:.4} sklearn_rmse_truth={sk_rmse_truth:.4} \
         (context: gam_edf={gam_edf:.3} sk_edf={sk_edf:.3} edf_rel={edf_rel:.3} \
         gam_vs_sklearn_rel_l2={rel:.4} pearson={corr:.5})"
    );

    // PRIMARY claim — gam recovers the deterministic truth to within one noise
    // sd. The data carry N(0, 0.1²) noise; a faithful ν = 5/2 GP denoises the
    // n=150 sample back to at worst the noise level (SNR ≈ 15, signal sd ≈ 1.1).
    // A real kernel-basis or hyperparameter-selection bug would blow the bias or
    // leave residual wiggle above the noise floor and trip this. Not weakened.
    assert!(
        gam_rmse_truth <= noise_sd,
        "gam matern GP fails to recover the known truth within the noise level: \
         rmse(gam, truth)={gam_rmse_truth:.4} > noise_sd={noise_sd:.3}"
    );

    // MATCH-OR-BEAT — gam is at least as accurate at recovering the truth as the
    // mature Python exact-GP reference (within a 10% tolerance). This is an
    // ACCURACY comparison on ground truth, NOT a "gam reproduces sklearn's fit"
    // claim: we never assert the two fitted functions are close.
    assert!(
        gam_rmse_truth <= sk_rmse_truth * 1.10,
        "gam matern GP is less accurate than sklearn GPR at recovering the truth: \
         rmse(gam, truth)={gam_rmse_truth:.4} > 1.10 × rmse(sklearn, truth)={sk_rmse_truth:.4}"
    );
}
