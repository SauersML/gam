//! End-to-end quality: gam's Gaussian-process Matérn smooth
//! (`matern(x, nu=2.5)`) must agree with `sklearn.gaussian_process.
//! GaussianProcessRegressor` — the Python standard for vanilla GP regression —
//! when both fit the same Matérn ν = 5/2 kernel to the same data.
//!
//! Why this comparator: scikit-learn's `GaussianProcessRegressor` is the
//! reference exact-GP regressor in the Python ecosystem. With a
//! `Matern(nu=2.5)` kernel it implements precisely the kernel gam's
//! `matern(x, nu=2.5)` basis targets, and it selects its length-scale by
//! maximizing the log marginal likelihood (sklearn's default
//! `fmin_l_bfgs_b` optimizer) — the same marginal-likelihood / REML criterion
//! gam uses to choose its smoothing. Two exact GP methods fitting the identical
//! ν = 5/2 kernel to identical fixed-seed data must produce nearly the same
//! posterior-mean function; a real divergence is a real bug in gam's GP-kernel
//! basis or hyperparameter selection, and we do NOT weaken the bound to hide it.
//!
//! Note on length-scale: gam's Matérn basis is REML-penalized (a smoothing
//! parameter on a finite kernel basis) while sklearn fits a continuous-kernel
//! length-scale plus a `WhiteKernel` noise term. These parametrizations are not
//! interchangeable, and gam exposes no scalar "length-scale" on its fit, so a
//! numeric length-scale equality would be unprincipled. We therefore compare the
//! quantity that is genuinely shared and observable — the fitted (posterior
//! mean) function on a dense common grid — plus effective complexity (EDF), and
//! additionally anchor both engines to the KNOWN deterministic truth they were
//! generated from. That keeps every assertion meaningful.
//!
//! We compare on a dense, identical evaluation grid:
//!   * relative L2 of the two fitted functions   (scale-free trajectory match),
//!   * Pearson correlation of the two functions   (shape match),
//!   * effective degrees of freedom               (complexity match),
//!   * relative L2 of each engine vs the known truth (both must recover it).

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

#[test]
fn gam_matern_gp_matches_sklearn_gpr() {
    init_parallelism();

    // ---- fixed-seed synthetic data, fed IDENTICALLY to gam and sklearn ----
    // x ~ U[0,1] (n=150, sorted); truth(x) = 2 + 1.5·sin(6π·x) + 0.5·cos(4π·x);
    // y = truth(x) + N(0, 0.1²). The deterministic truth is known exactly.
    let n = 150usize;
    let mut rng = StdRng::seed_from_u64(20260529);
    let ux = Uniform::new(0.0, 1.0).expect("uniform [0,1]");
    let noise = Normal::new(0.0, 0.1).expect("gaussian noise sd=0.1");
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
    // length-scale, amplitude, and noise — the marginal-likelihood criterion
    // gam's REML matches. We predict the posterior mean on x_grid and also
    // report sklearn's effective DoF, trace(K (K+σ²I)⁻¹), the GP analogue of EDF.
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

    // ---- compare ----------------------------------------------------------
    let rel = relative_l2(&gam_grid, sk_grid);
    let corr = pearson(&gam_grid, sk_grid);
    let edf_rel = (gam_edf - sk_edf).abs() / sk_edf.abs().max(1.0);
    let gam_vs_truth = relative_l2(&gam_grid, &truth_grid);
    let sk_vs_truth = relative_l2(sk_grid, &truth_grid);

    eprintln!(
        "matern(x,nu=2.5,k=25) vs sklearn GPR Matern(nu=2.5): n={n} grid={grid_n} \
         gam_edf={gam_edf:.3} sk_edf={sk_edf:.3} edf_rel={edf_rel:.3} \
         rel_l2={rel:.4} pearson={corr:.5} \
         gam_vs_truth={gam_vs_truth:.4} sklearn_vs_truth={sk_vs_truth:.4}"
    );

    // Both engines are exact GP methods fitting the identical Matérn ν = 5/2
    // kernel by marginal likelihood to identical fixed-seed data, so their
    // posterior-mean functions must nearly coincide. Bounds (per the spec):
    //  - Pearson > 0.998: trajectories must share shape; a kernel or
    //    hyperparameter-selection divergence would drop this well below it.
    //  - relative L2 < 0.05: tight pointwise agreement on the dense grid — both
    //    are exact GP methods, so a loose bound here would assert nothing.
    //  - each engine within rel-L2 0.05 of the KNOWN truth: anchors the
    //    comparison to ground truth, so agreeing on a *wrong* function cannot
    //    pass. SNR here is ~15 (signal sd ≈ 1.1, noise sd = 0.1), so a faithful
    //    ν = 5/2 GP recovers the smooth truth to a few percent.
    //  - EDF within 30%: same effective model complexity. gam's penalized
    //    finite-basis EDF and sklearn's trace(K(K+σ²I)⁻¹) use different
    //    conventions, so same-ballpark wiggliness is the right expectation, not
    //    a bit-identical match.
    assert!(
        corr > 0.998,
        "matern GP shape diverges from sklearn GPR: pearson={corr:.5}"
    );
    assert!(
        rel < 0.05,
        "matern GP fitted function diverges from sklearn GPR: rel_l2={rel:.4}"
    );
    assert!(
        gam_vs_truth < 0.05,
        "gam matern GP fails to recover the known truth: rel_l2={gam_vs_truth:.4}"
    );
    assert!(
        sk_vs_truth < 0.05,
        "sklearn GPR fails to recover the known truth (sanity on the comparator): rel_l2={sk_vs_truth:.4}"
    );
    assert!(
        edf_rel < 0.30,
        "matern GP effective degrees of freedom disagree: gam={gam_edf:.3} sklearn={sk_edf:.3} (rel={edf_rel:.3})"
    );
}
