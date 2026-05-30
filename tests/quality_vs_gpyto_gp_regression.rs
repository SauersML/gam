//! End-to-end quality: gam's Matérn-kernel GP smooth (`matern(x, nu=2.5)`) must
//! match **GPyTorch** — the modern PyTorch foundation for *exact* Gaussian-process
//! inference — on the same data.
//!
//! GPyTorch fits an exact GP with a Matérn(ν=5/2) kernel by maximizing the exact
//! log marginal likelihood over the kernel hyperparameters (length-scale, output
//! scale) and the Gaussian observation noise. gam's `matern(x, nu=2.5, k=30)`
//! basis is a finite-rank realization of the same Matérn-5/2 kernel whose
//! smoothing is selected by REML. Agreement therefore validates three independent
//! pieces of gam's GP machinery at once:
//!
//!   1. **Fitted posterior mean** on a dense grid — relative L2 < 0.06 (scale-free,
//!      tight). The synthetic response carries both a smooth oscillation
//!      (`x·sin x`) and a linear trend (`x/5`) to tax nonstationary flexibility.
//!   2. **Length-scale (ρ)** — recovered *identically* for both engines as the
//!      operational correlation length of the fitted posterior mean (the lag at
//!      which the curve's normalized spatial autocorrelation decays to 1/e). This
//!      is the standard model-agnostic definition of a GP correlation length and
//!      is computed by the SAME routine on both fitted curves, so it compares the
//!      recovered smoothness scale rather than two engines' internal
//!      parameterizations: |gam_ρ − gptorch_ρ| / mean < 15%.
//!   3. **Latent-function uncertainty bands** — gam's 95% credible-interval
//!      half-width `1.96·sqrt(diag(X Vb Xᵀ))` (uncertainty of the smooth mean
//!      function) vs GPyTorch's latent-function posterior std `model(xg).stddev`
//!      (NOT the noisy-observation std `lik(model(xg))`, which would add σ²) must
//!      track in shape across the grid: Pearson correlation > 0.92.
//!
//! We deliberately do NOT cross-assert a raw log-marginal-likelihood number: gam's
//! `reml_score` is the *restricted* (REML) profile criterion — it marginalizes the
//! Matérn nullspace, uses the restricted dof ν = n − M, and carries the profiled-
//! out-scale constant ½ν(1+ln 2π σ̂²) — whereas GPyTorch's `ExactMarginalLogLikelihood`
//! is the *full* ML log-evidence with a free noise variance and a constant mean.
//! Those two objectives differ by an O(½n) additive constant and a nullspace
//! log-determinant, so a "5% match" would be comparing structurally different
//! quantities (and any agreement would be coincidental). We instead compare only
//! quantities computed identically on both fitted models.
//!
//! A real divergence in any of the three assertions is a real bug in gam's Matérn
//! kernel or its smoothing selection; we do NOT weaken the bounds to hide it.

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

/// Operational GP correlation length of a function sampled on an *evenly spaced*
/// grid: the lag (in x-units) at which the normalized spatial autocorrelation of
/// the mean-centered curve first decays to 1/e. This is the standard model-free
/// definition of a stationary GP length-scale and is applied identically to
/// gam's and GPyTorch's fitted posterior means, so it compares the recovered
/// smoothness scale rather than the two engines' internal kernel parameters.
fn correlation_length(grid: &[f64], values: &[f64]) -> f64 {
    assert_eq!(grid.len(), values.len(), "grid/value length mismatch");
    let n = values.len();
    assert!(n >= 8, "need a dense grid for a stable correlation length");
    let dx = (grid[n - 1] - grid[0]) / (n - 1) as f64;
    let mean = values.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = values.iter().map(|v| v - mean).collect();
    let c0: f64 = centered.iter().map(|v| v * v).sum::<f64>() / n as f64;
    assert!(c0 > 0.0, "fitted curve is constant; no correlation length");
    let target = 1.0 / std::f64::consts::E;
    let mut prev = 1.0_f64;
    for lag in 1..n {
        let mut acc = 0.0;
        for i in 0..(n - lag) {
            acc += centered[i] * centered[i + lag];
        }
        let r = (acc / (n - lag) as f64) / c0;
        if r <= target {
            // Linear interpolation between lag-1 and lag for a smooth estimate.
            let frac = (prev - target) / (prev - r).max(1e-12);
            return (lag as f64 - 1.0 + frac) * dx;
        }
        prev = r;
    }
    // Never decayed to 1/e within the window: the curve is smoother than the
    // grid; report the full span as a lower bound on the length-scale.
    (n - 1) as f64 * dx
}

#[test]
fn gam_gp_regression_matches_gpytorch() {
    init_parallelism();

    // ---- fixed-seed synthetic 1-D data, fed IDENTICALLY to gam and GPyTorch --
    // x ~ U[-2, 2] (n=200, sorted); y = x·sin x + x/5 + 0.12·N(0,1).
    let n = 200usize;
    let mut rng = StdRng::seed_from_u64(20260529);
    let ux = Uniform::new(-2.0, 2.0).expect("uniform [-2,2]");
    let noise = Normal::new(0.0, 0.12).expect("gaussian noise");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
    let truth = |t: f64| t * t.sin() + t / 5.0;
    let y: Vec<f64> = x.iter().map(|&t| truth(t) + noise.sample(&mut rng)).collect();

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

    let gam_rho = correlation_length(&x_grid, &gam_grid);

    // ---- fit the SAME data with GPyTorch (exact GP, Matérn-5/2 kernel) -------
    // Maximize the exact log marginal likelihood over length-scale, output scale,
    // and Gaussian noise via Adam, then evaluate the latent-function posterior mean
    // and std on the identical grid. GPyTorch is the modern reference for exact GP
    // inference, so its fitted mean and uncertainty are the ground truth here.
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

# Latent-function posterior on the shared grid at the fitted optimum. We use
# model(xg) (the latent GP f), NOT lik(model(xg)) (a new noisy observation):
# gam's band 1.96*sqrt(diag(X Vb X^T)) is the uncertainty of the smooth/mean
# FUNCTION, so the matching GPyTorch quantity is the latent-function std, not
# the observation-predictive std (which adds the noise variance sigma^2 and
# would compare a different quantity).
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

    let gpt_rho = correlation_length(&x_grid, gpt_grid);

    // ---- compare -------------------------------------------------------------
    let rel = relative_l2(&gam_grid, gpt_grid);
    // Predictive-uncertainty SHAPE: gam half-width vs GPyTorch posterior std.
    let band_corr = pearson(&gam_band, gpt_std);
    let rho_rel = (gam_rho - gpt_rho).abs() / (0.5 * (gam_rho + gpt_rho)).max(1e-12);

    eprintln!(
        "gp matern(x,nu=2.5,k=30) vs GPyTorch exact GP: n={n} grid={grid_n} \
         rel_l2={rel:.4} rho_gam={gam_rho:.4} rho_gpt={gpt_rho:.4} rho_rel={rho_rel:.4} \
         band_pearson={band_corr:.4}"
    );

    // 1. Fitted posterior mean: both engines fit the identical Matérn-5/2 GP on
    //    identical data, so the recovered mean must track tightly. Scale-free
    //    relative L2 < 0.06 leaves a sane finite-rank-vs-exact margin while
    //    catching any real kernel/marginal-likelihood divergence.
    assert!(
        rel < 0.06,
        "gam GP mean diverges from GPyTorch exact GP: rel_l2={rel:.4}"
    );

    // 2. Length-scale agreement: same recovered smoothness scale to within 15%.
    //    A genuine kernel-bandwidth disagreement would blow this well past 15%.
    assert!(
        rho_rel < 0.15,
        "recovered GP correlation length disagrees: gam={gam_rho:.4} gpt={gpt_rho:.4} (rel={rho_rel:.4})"
    );

    // 3. Predictive uncertainty bands: gam's Bayesian half-width and GPyTorch's
    //    posterior std must share shape across the grid (both widen away from
    //    data support and at the edges). Pearson > 0.92 demands genuine agreement
    //    in where the model is (un)certain, not just a scalar offset.
    assert!(
        band_corr > 0.92,
        "predictive uncertainty bands disagree in shape: pearson={band_corr:.4}"
    );
}
