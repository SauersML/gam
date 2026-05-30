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
use gam::test_support::reference::{Column, pearson, relative_l2, rmse, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Fraction of grid points at which the noise-free truth `f(x)` lies inside the
/// symmetric band `[mean(x) − half(x), mean(x) + half(x)]`. Objective empirical
/// coverage of a 95% credible band against ground truth.
fn empirical_coverage(mean: &[f64], half_width: &[f64], truth: &[f64]) -> f64 {
    assert_eq!(mean.len(), half_width.len(), "mean/half-width length mismatch");
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
    let py = run_python(
        &[
            Column::new("x", &x),
            Column::new("y", &y),
            Column::new("xg", &x_grid),
        ],
        r#"
import torch, gpytorch
torch.manual_seed(0)

xt = torch.as_tensor(np.asarray(df["x"], dtype=float), dtype=torch.float64)
yt = torch.as_tensor(np.asarray(df["y"], dtype=float), dtype=torch.float64)
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
"#,
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
