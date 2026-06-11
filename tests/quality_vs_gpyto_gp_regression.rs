//! End-to-end OBJECTIVE quality: gam's Matérn-kernel GP smooth
//! (`matern(x, nu=2.5)`) must RECOVER THE TRUE GENERATING FUNCTION from noisy
//! data, and do so at least as accurately as **GPyTorch** (the modern PyTorch
//! foundation for exact Gaussian-process inference) fit on the identical data.
//!
//! The response is simulated from a KNOWN smooth truth
//! `f(x) = x·sin x + x/5` with i.i.d. Gaussian noise `σ = 0.12`, so we have
//! ground truth on a dense grid and can measure objective quality directly
//! rather than asking whether gam reproduces another tool's (itself noisy) fit.
//!
//! OBJECTIVE METRICS ASSERTED (none is "close to the reference's output"):
//!
//!   1. **Truth recovery (PRIMARY).** RMSE of gam's fitted posterior mean against
//!      the noise-free truth on a dense interior grid must be a small fraction of
//!      the signal: `rmse(gam, truth) <= 0.35·σ`. A good smoother removes most of
//!      the noise, so its error against the truth is well below the per-point
//!      noise level — this is the real claim ("gam recovers `x·sin x + x/5`"),
//!      and it is independent of any other tool.
//!
//!   2. **Match-or-beat GPyTorch on ACCURACY (baseline).** Fitting the same data
//!      with an exact Matérn-5/2 GP, gam's truth-recovery RMSE must be no worse
//!      than `1.10×` GPyTorch's truth-recovery RMSE. GPyTorch is demoted from
//!      "the answer gam must reproduce" to "a strong baseline gam must match or
//!      beat" on the objective accuracy metric — both errors are measured against
//!      the SAME known truth, so a tie/beat is a genuine quality statement.
//!
//!   3. **Uncertainty calibration (objective).** gam's 95% credible band
//!      `1.96·sqrt(diag(X Vb Xᵀ))` for the smooth FUNCTION must actually cover the
//!      truth: empirical coverage of `f(x)` by `[mean ± half-width]` across the
//!      grid lands within `±0.10` of the nominal `0.95`. This asserts the band is
//!      correctly sized against ground truth — not that it matches GPyTorch's std.
//!
//! For CONTEXT only (printed, never asserted) we also compute the relative L2
//! between gam's and GPyTorch's fitted means and the Pearson shape correlation of
//! the two uncertainty profiles. Those "closeness to the reference" numbers are
//! diagnostics, not pass/fail criteria: matching a peer tool's noisy fit proves
//! nothing about correctness.
//!
//! We deliberately do NOT cross-assert a raw log-marginal-likelihood number:
//! gam's `reml_score` is the restricted (REML) criterion while GPyTorch's
//! `ExactMarginalLogLikelihood` is the full ML log-evidence; they differ by an
//! O(½n) constant and a nullspace log-determinant.
//!
//! A genuine truth-recovery or calibration shortfall failing here is a real bug
//! in gam's Matérn kernel or its REML smoothing selection; we do NOT weaken the
//! bounds to hide it.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pad_to, pearson, r2, relative_l2, rmse, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::path::Path;

/// Canonical lidar benchmark (range -> logratio). Source: Sigrist (1994) lidar
/// experiment, distributed with R packages `SemiPar`/`gamair` and reused across
/// the smoothing literature; vendored here at `bench/datasets/lidar.csv`.
const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

/// Fraction of grid points at which the noise-free truth `f(x)` lies inside the
/// symmetric band `[mean(x) − half(x), mean(x) + half(x)]`. Objective empirical
/// coverage of a 95% credible band against ground truth.
fn empirical_coverage(mean: &[f64], half_width: &[f64], truth: &[f64]) -> f64 {
    assert_eq!(
        mean.len(),
        half_width.len(),
        "mean/half-width length mismatch"
    );
    assert_eq!(mean.len(), truth.len(), "mean/truth length mismatch");
    let n = mean.len();
    assert!(n > 0, "need a non-empty grid for coverage");
    let inside = (0..n)
        .filter(|&i| (truth[i] - mean[i]).abs() <= half_width[i])
        .count();
    inside as f64 / n as f64
}

#[test]
fn gam_gp_regression_recovers_truth() {
    init_parallelism();

    // ---- fixed-seed synthetic 1-D data, fed IDENTICALLY to gam and GPyTorch --
    // x ~ U[-2, 2] (n=200, sorted); y = x·sin x + x/5 + 0.12·N(0,1).
    let n = 200usize;
    let noise_sigma = 0.12_f64;
    let mut rng = StdRng::seed_from_u64(20260529);
    let ux = Uniform::new(-2.0, 2.0).expect("uniform [-2,2]");
    let noise = Normal::new(0.0, noise_sigma).expect("gaussian noise");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
    let truth = |t: f64| t * t.sin() + t / 5.0;
    let y: Vec<f64> = x
        .iter()
        .map(|&t| truth(t) + noise.sample(&mut rng))
        .collect();

    // ---- fit with gam: y ~ matern(x, nu=2.5, k=30), REML ---------------------
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode GP dataset");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ matern(x, nu=2.5, k=30)", &ds, &cfg).expect("gam matern fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Gaussian GAM fit for matern() GP smooth");
    };

    // ---- shared dense 250-point evaluation grid (interior of [-2, 2]) --------
    let grid_n = 250usize;
    let (lo, hi) = (-1.95_f64, 1.95_f64);
    let x_grid: Vec<f64> = (0..grid_n)
        .map(|i| lo + (hi - lo) * i as f64 / (grid_n - 1) as f64)
        .collect();
    // Noise-free ground truth on the same grid: the objective target.
    let truth_grid: Vec<f64> = x_grid.iter().map(|&t| truth(t)).collect();

    // gam fitted posterior mean + 95% credible-interval half-width on the grid.
    // Identity link ⇒ design·beta = mean. The Bayesian band is
    // 1.96·sqrt(diag(X Vb Xᵀ)) with Vb the conditional posterior covariance.
    let mut g = Array2::<f64>::zeros((grid_n, 2));
    for (i, &t) in x_grid.iter().enumerate() {
        g[[i, 0]] = t;
        g[[i, 1]] = 0.0;
    }
    let grid_design = build_term_collection_design(g.view(), &fit.resolvedspec)
        .expect("rebuild matern design at grid points");
    let gam_grid: Vec<f64> = grid_design.design.apply(&fit.fit.beta).to_vec();

    let vb = fit
        .fit
        .covariance_conditional
        .as_ref()
        .expect("gam reports the conditional posterior covariance Vb");
    let xmat = grid_design.design.to_dense();
    assert_eq!(xmat.ncols(), vb.nrows(), "design/Vb dimension mismatch");
    // Bandwidth = 1.96·sqrt(xᵢ Vb xᵢᵀ) row by row.
    let gam_band: Vec<f64> = (0..grid_n)
        .map(|i| {
            let xi = xmat.row(i);
            let vx = vb.dot(&xi);
            let var: f64 = xi.dot(&vx);
            1.96 * var.max(0.0).sqrt()
        })
        .collect();

    // ---- fit the SAME data with GPyTorch (exact GP, Matérn-5/2 kernel) -------
    // Maximize the exact log marginal likelihood over length-scale, output scale,
    // and Gaussian noise via Adam, then evaluate the latent-function posterior mean
    // on the identical grid. GPyTorch is the strong BASELINE to match-or-beat on
    // truth-recovery accuracy, NOT an answer gam must reproduce.
    // The harness writes one rectangular CSV, so every column must share a row
    // count. The dense grid (`grid_n`) is LONGER than the training set (`n`), so
    // the longest column sets the row count: ride the n training rows along
    // right-padded up to `grid_n` and slice them back to the first `n` inside
    // the body (the padded tail is never read for training).
    let py = run_python(
        &[
            Column::new("x", &pad_to(&x, grid_n)),
            Column::new("y", &pad_to(&y, grid_n)),
            Column::new("xg", &x_grid),
        ],
        &format!(
            r#"
import torch, gpytorch
torch.manual_seed(0)

xt = torch.as_tensor(np.asarray(df["x"], dtype=float)[:{n}], dtype=torch.float64)
yt = torch.as_tensor(np.asarray(df["y"], dtype=float)[:{n}], dtype=torch.float64)
xg = torch.as_tensor(np.asarray(df["xg"], dtype=float), dtype=torch.float64)

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, tx, ty, lik):
        super().__init__(tx, ty, lik)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5))
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))

lik = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGP(xt, yt, lik)
model.double(); lik.double()

model.train(); lik.train()
opt = torch.optim.Adam(model.parameters(), lr=0.05)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
for _ in range(400):
    opt.zero_grad()
    out = model(xt)
    loss = -mll(out, yt)
    loss.backward()
    opt.step()

# Latent-function posterior MEAN on the shared grid at the fitted optimum. We use
# model(xg) (the latent GP f), NOT lik(model(xg)) (a new noisy observation):
# the smooth/mean function is what we score against the known truth, and its
# uncertainty band excludes the observation noise variance sigma^2.
model.eval(); lik.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
    f = model(xg)
    mean = f.mean.numpy()
    std = f.stddev.numpy()

emit("grid_fit", mean)
emit("grid_std", std)
"#
        ),
    );
    let gpt_grid = py.vector("grid_fit");
    let gpt_std = py.vector("grid_std");
    assert_eq!(gpt_grid.len(), grid_n, "GPyTorch grid mean length mismatch");
    assert_eq!(gpt_std.len(), grid_n, "GPyTorch grid std length mismatch");

    // ---- OBJECTIVE quality metrics, all measured against the KNOWN truth -----
    let gam_truth_rmse = rmse(&gam_grid, &truth_grid);
    let gpt_truth_rmse = rmse(gpt_grid, &truth_grid);
    let coverage = empirical_coverage(&gam_grid, &gam_band, &truth_grid);

    // Context-only diagnostics (NOT asserted): closeness to the peer tool.
    let rel_to_ref = relative_l2(&gam_grid, gpt_grid);
    let band_corr = pearson(&gam_band, gpt_std);

    eprintln!(
        "gp matern(x,nu=2.5,k=30) TRUTH RECOVERY: n={n} grid={grid_n} sigma={noise_sigma:.3} \
         gam_rmse_vs_truth={gam_truth_rmse:.4} gpt_rmse_vs_truth={gpt_truth_rmse:.4} \
         coverage95={coverage:.3} | context-only rel_l2_to_gpt={rel_to_ref:.4} \
         band_pearson={band_corr:.4}"
    );

    // 1. PRIMARY — truth recovery: a good Matérn-5/2 smoother removes most of the
    //    i.i.d. noise, so its RMSE against the noise-free truth is a small
    //    fraction of the per-point noise level sigma. 0.35*sigma is a principled,
    //    un-weakened bar for n=200 on a smooth signal.
    let truth_bar = 0.35 * noise_sigma;
    assert!(
        gam_truth_rmse <= truth_bar,
        "gam GP failed to recover the truth: rmse_vs_truth={gam_truth_rmse:.4} > bar={truth_bar:.4} (sigma={noise_sigma:.3})"
    );

    // 2. BASELINE — match or beat GPyTorch on the SAME objective accuracy metric
    //    (RMSE against the known truth). gam must be no worse than 1.10x the exact
    //    GP's truth-recovery error.
    assert!(
        gam_truth_rmse <= gpt_truth_rmse * 1.10,
        "gam GP truth-recovery worse than GPyTorch baseline: \
         gam={gam_truth_rmse:.4} > 1.10*gpt={:.4}",
        gpt_truth_rmse * 1.10
    );

    // 3. CALIBRATION — the 95% credible band for the smooth function must actually
    //    cover the truth at close to its nominal rate. Empirical coverage within
    //    +/-0.10 of 0.95 asserts the uncertainty is correctly sized against ground
    //    truth (not that it matches GPyTorch's std).
    assert!(
        (coverage - 0.95).abs() <= 0.10,
        "gam 95% credible band miscalibrated against truth: empirical coverage={coverage:.3} (nominal 0.95)"
    );
}

#[test]
fn gam_gp_regression_recovers_truth_on_real_data() {
    // REAL-DATA ARM of the Matérn-5/2 GP regression capability. On the lidar
    // benchmark the generating function is UNKNOWN, so objective quality is
    // measured as OUT-OF-SAMPLE predictive accuracy, not recovery of a synthetic
    // truth: we make a deterministic train/test split (every 4th row held out),
    // fit gam's `matern(range, nu=2.5)` GP smooth on TRAIN only, predict the SAME
    // held-out rows that an exact GPyTorch Matérn-5/2 GP predicts from the SAME
    // training rows, and assert:
    //
    //   PRIMARY (objective, tool-free): held-out coefficient of determination
    //     `test_R2 >= 0.55` — the GP smooth genuinely explains held-out variance,
    //     well above the constant test-mean predictor (R2 = 0).
    //
    //   BASELINE (match-or-beat): gam's held-out RMSE must be no worse than
    //     `gpytorch_test_rmse * 1.10`. GPyTorch is a strong baseline to match or
    //     beat on the SAME accuracy metric, never an output to replicate.
    init_parallelism();

    // ---- load the canonical lidar dataset (range -> logratio) ----------------
    let ds = load_csvwith_inferred_schema(Path::new(LIDAR_CSV)).expect("load lidar.csv");
    let col = ds.column_map();
    let range_idx = col["range"];
    let logratio_idx = col["logratio"];
    let range: Vec<f64> = ds.values.column(range_idx).to_vec();
    let logratio: Vec<f64> = ds.values.column(logratio_idx).to_vec();
    let n = range.len();
    assert!(n > 100, "lidar should have ~221 rows, got {n}");

    // ---- deterministic train/test split: every 4th row held out -------------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 100 && test_rows.len() > 30,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_range: Vec<f64> = train_rows.iter().map(|&i| range[i]).collect();
    let train_logratio: Vec<f64> = train_rows.iter().map(|&i| logratio[i]).collect();
    let test_range: Vec<f64> = test_rows.iter().map(|&i| range[i]).collect();
    let test_logratio: Vec<f64> = test_rows.iter().map(|&i| logratio[i]).collect();

    // Build a training-only encoded dataset by sub-setting the encoded rows;
    // headers, schema, and column kinds are unchanged, so the formula resolves
    // identically to the full-data fit.
    let p = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: logratio ~ matern(range, nu=2.5, k=30), REML ------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("logratio ~ matern(range, nu=2.5, k=30)", &train_ds, &cfg)
        .expect("gam matern fit on lidar train");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Gaussian GAM fit for matern() GP smooth on lidar");
    };

    // gam predictions at the held-out `range` points: rebuild the frozen-spec
    // design (identity link => design*beta = predicted mean).
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (i, &r) in test_range.iter().enumerate() {
        test_grid[[i, range_idx]] = r;
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild matern design at held-out lidar points");
    let gam_test_pred: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(
        gam_test_pred.len(),
        test_rows.len(),
        "gam held-out prediction length mismatch"
    );

    // ---- fit the SAME train rows with GPyTorch, predict the SAME test rows ---
    // Exact Matérn-5/2 GP, hyperparameters by Adam on the exact log marginal
    // likelihood; predict the LATENT mean at the held-out range points. Every
    // Column handed to one run_python call is train-length: the test range rides
    // along right-padded, and only its first `test_n` entries are read back.
    let py = run_python(
        &[
            Column::new("range", &train_range),
            Column::new("logratio", &train_logratio),
            Column::new("test_range", &pad_to(&test_range, train_range.len())),
            Column::new("test_n", &vec![test_range.len() as f64; train_range.len()]),
        ],
        r#"
import torch, gpytorch
torch.manual_seed(0)

k = int(round(float(np.asarray(df["test_n"])[0])))
xt = torch.as_tensor(np.asarray(df["range"], dtype=float), dtype=torch.float64)
yt = torch.as_tensor(np.asarray(df["logratio"], dtype=float), dtype=torch.float64)
xtest = torch.as_tensor(np.asarray(df["test_range"], dtype=float)[:k], dtype=torch.float64)

# Standardize the input for numerically well-conditioned length-scale learning;
# the GP posterior mean is invariant to this affine reparameterization.
xm, xs = xt.mean(), xt.std()
xt_s = (xt - xm) / xs
xtest_s = (xtest - xm) / xs

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, tx, ty, lik):
        super().__init__(tx, ty, lik)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5))
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))

lik = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGP(xt_s, yt, lik)
model.double(); lik.double()

model.train(); lik.train()
opt = torch.optim.Adam(model.parameters(), lr=0.05)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
for _ in range(500):
    opt.zero_grad()
    out = model(xt_s)
    loss = -mll(out, yt)
    loss.backward()
    opt.step()

model.eval(); lik.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
    f = model(xtest_s)
    pred = f.mean.numpy()

emit("test_pred", pred)
"#,
    );
    let gpt_test_pred = py.vector("test_pred");
    assert_eq!(
        gpt_test_pred.len(),
        test_rows.len(),
        "GPyTorch held-out prediction length mismatch"
    );

    // ---- objective held-out metrics on gam's OWN predictions -----------------
    let gam_test_r2 = r2(&gam_test_pred, &test_logratio);
    let gam_test_rmse = rmse(&gam_test_pred, &test_logratio);
    let gpt_test_rmse = rmse(gpt_test_pred, &test_logratio);

    // Context-only diagnostic (NOT asserted): closeness of the two held-out
    // prediction vectors. Matching a peer tool's noisy fit proves nothing.
    let rel_to_ref = relative_l2(&gam_test_pred, gpt_test_pred);

    eprintln!(
        "lidar matern(range,nu=2.5,k=30) held-out: n_train={} n_test={} \
         gam_test_R2={gam_test_r2:.4} gam_test_rmse={gam_test_rmse:.4} \
         gpt_test_rmse={gpt_test_rmse:.4} | context-only rel_l2_to_gpt={rel_to_ref:.4}",
        train_rows.len(),
        test_rows.len(),
    );

    // ---- PRIMARY objective assertion: gam predicts the held-out signal -------
    // The lidar mean function is strongly nonlinear with a clear signal; a
    // competent Matérn GP explains well over half the held-out variance. R2 >=
    // 0.55 is far above the constant-mean baseline (0) and catches under/over-
    // smoothing of the GP length-scale.
    assert!(
        gam_test_r2 >= 0.55,
        "gam GP held-out predictive R2 too low: {gam_test_r2:.4} (< 0.55)"
    );

    // ---- BASELINE (match-or-beat): no worse than GPyTorch on held-out RMSE ---
    assert!(
        gam_test_rmse <= gpt_test_rmse * 1.10,
        "gam held-out RMSE {gam_test_rmse:.4} exceeds GPyTorch baseline {gpt_test_rmse:.4} * 1.10"
    );
}
