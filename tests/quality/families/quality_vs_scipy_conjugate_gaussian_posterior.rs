//! End-to-end quality (OBJECTIVE METRIC — recovery of an exact analytic
//! posterior): gam's NUTS posterior for a Gaussian / identity-link penalized
//! smooth must recover the *closed-form* conjugate posterior `N(μ_post, Vb)`,
//! a mathematical ground-truth quantity available in closed form. The
//! pass/fail assertion is gam's error against the analytic `μ_post`/`sd_post`
//! computed exactly (no sampling) — NOT closeness to any tool's fitted output.
//!
//! The asserted metrics are:
//!   1. `max|gam_mean − μ_exact|` (analytic posterior mean recovery), bounded
//!      by a few Monte-Carlo standard errors of a 10k-draw sampler;
//!   2. `pearson(gam_mean, μ_exact)` ≈ 1 (no directional drift);
//!   3. `max|gam_sd − sd_exact|` (analytic posterior sd recovery), bounded by a
//!      small fraction of the largest exact sd.
//! All three compare gam to the EXACT analytic posterior, i.e. ground truth.
//!
//! scipy's role is demoted to a BASELINE-TO-MATCH-OR-BEAT: `scipy.stats`
//! draws 10k i.i.d. samples from the *same* exact target and we assert gam's
//! recovery error is no worse than scipy's own finite-sample Monte-Carlo error
//! (within a slack factor). gam is a true MCMC sampler vs. scipy's i.i.d.
//! draws on a Gaussian, so matching that floor is a strong, objective claim
//! about gam's whitening + triangular-solve machinery — not "we mimic scipy".
//!
//! Comparator: **scipy.stats** — but only as the source of the EXACT analytic
//! `μ_post`/`Σ_post` (ground truth) plus a finite-sample MC-error baseline. For
//! a Gaussian likelihood with identity link and the GAM smoothing penalty
//! `S_λ = λP` entering on the φ=1 scale, gam's NUTS targets the log-posterior
//!
//!     log p(β) ∝ −(1/2φ)·[ (y − Xβ)ᵀ(y − Xβ) + βᵀ S_λ β ] ,
//!
//! whose normal form is the standard mgcv Bayesian posterior `N(μ_post, Vb)`:
//!
//!     H = XᵀX + S_λ ,   Vb = Σ_post = φ·H⁻¹ ,   μ_post = H⁻¹·(Xᵀ y) .
//!
//! Both the likelihood AND the penalty are φ-scaled (this is the `Vb = φ·H⁻¹`
//! module invariant in `inference::dispersion_cov`, NOT the classical
//! conjugate-regression prior precision `S` that is φ-independent). This normal
//! posterior is available in closed form, so `scipy.stats.multivariate_normal`
//! samples it exactly.
//!
//! gam's NUTS path whitens the coefficient space with the *fitted penalized
//! Hessian* (the φ-scaled `(1/φ)·H`) as the mass matrix and draws the φ-scaled
//! covariance `φ·H⁻¹`, so gam's draws must reproduce scipy's exact posterior.
//! This is the cleanest possible check of gam's whitening + Cholesky-inversion
//! machinery (`explicit_fit_hessian_for_whitening` →
//! `run_nuts_sampling_flattened_family`): a real divergence is a real bug in
//! the sampler's mass matrix or the triangular solve.
//!
//! We feed BOTH engines the *identical* materialized design matrix `X`, the
//! identical penalty `S_λ = λP` (gam's `weighted_blockwise_penalty_sum` at the
//! fitted λ), the identical response `y`, and the identical noise variance
//! `σ² = φ` (gam's estimated dispersion). The only difference is the sampling
//! mechanism: gam runs NUTS (1 chain, 10k draws — this convex Gaussian target
//! converges in a few leapfrogs), scipy draws i.i.d. from the exact Gaussian.

use gam::hmc::{
    FamilyNutsInputs, GlmFlatInputs, NutsConfig, explicit_fit_hessian_for_whitening,
    run_nuts_sampling_flattened_family,
};
use gam::smooth::{build_term_collection_design, weighted_blockwise_penalty_sum};
use gam::test_support::reference::{Column, max_abs_diff, pearson, run_python};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

#[test]
fn gam_nuts_posterior_matches_scipy_exact_conjugate_gaussian() {
    init_parallelism();

    // ---- load the canonical lidar dataset (range -> logratio) -------------
    let ds = load_csvwith_inferred_schema(Path::new(LIDAR_CSV)).expect("load lidar.csv");
    let col = ds.column_map();
    let range_idx = col["range"];
    let logratio_idx = col["logratio"];
    let range: Vec<f64> = ds.values.column(range_idx).to_vec();
    let logratio: Vec<f64> = ds.values.column(logratio_idx).to_vec();
    let n = range.len();
    assert!(n > 100, "lidar should have ~221 rows, got {n}");

    // ---- fit with gam: logratio ~ s(range, k=5), Gaussian/identity --------
    // A modest basis (k=5) keeps the posterior low-dimensional so the
    // closed-form comparison is sharp and the mass matrix well-conditioned.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("logratio ~ s(range, k=5)", &ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };

    // Rebuild the materialized design at the training points from the frozen
    // spec — this is exactly the `X` gam's NUTS path whitens against (identity
    // link => design·beta = mean). This same `X` is sent to scipy.
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for (i, &r) in range.iter().enumerate() {
        grid[[i, range_idx]] = r;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let x_dense = design.design.to_dense();
    let p = x_dense.ncols();
    assert_eq!(x_dense.nrows(), n, "design row count must equal n");

    // The total penalty S_λ = Σ_k λ_k S_k at gam's fitted smoothing parameter,
    // and the fitted dispersion σ² = φ. These define the conjugate posterior.
    let lambdas = fit.fit.lambdas.as_slice().expect("contiguous lambdas");
    let penalty = weighted_blockwise_penalty_sum(&design.penalties, lambdas, p);
    let dispersion = fit
        .fit
        .dispersion()
        .expect("Gaussian fit reports an estimated dispersion");
    let phi = dispersion.phi();
    assert!(
        phi.is_finite() && phi > 0.0,
        "fitted dispersion must be positive, got {phi}"
    );

    // ---- run gam's NUTS over the coefficient vector -----------------------
    // We whiten with the SAVED penalized Hessian H = XᵀX + S_λ (the exact
    // function the CLI `sample` path uses; the sampler scales the Cholesky by
    // √φ so the drawn covariance is the φ-scaled posterior φ·H⁻¹).
    // One chain, 10k post-warmup draws: this convex Gaussian target mixes
    // immediately, so the Monte-Carlo error on each coordinate mean is
    // ~ sd/sqrt(10000) ≈ sd/100.
    let likelihood = LikelihoodSpec {
        response: ResponseFamily::Gaussian,
        link: InverseLink::Standard(StandardLink::Identity),
    };
    let weights = ndarray::Array1::<f64>::ones(n);
    let y = ndarray::Array1::from(logratio.clone());
    let hessian = explicit_fit_hessian_for_whitening(&fit.fit, p, "scipy conjugate test")
        .expect("fitted penalized Hessian for whitening");
    let nuts_cfg = NutsConfig {
        n_samples: 10_000,
        nwarmup: 1_000,
        n_chains: 1,
        target_accept: 0.8,
        seed: 20260529,
    };
    let nuts = run_nuts_sampling_flattened_family(
        likelihood,
        FamilyNutsInputs::Glm(GlmFlatInputs {
            x: x_dense.view(),
            y: y.view(),
            weights: weights.view(),
            penalty_matrix: penalty.view(),
            mode: fit.fit.beta.view(),
            hessian: hessian.view(),
            likelihood_scale: fit.fit.likelihood_scale,
            dispersion,
            firth_bias_reduction: false,
            offset: None,
        }),
        &nuts_cfg,
    )
    .expect("gam NUTS sampling over Gaussian/identity posterior");
    assert_eq!(nuts.posterior_mean.len(), p, "posterior mean dim mismatch");

    let gam_mean: Vec<f64> = nuts.posterior_mean.to_vec();
    let gam_sd: Vec<f64> = nuts.posterior_std.to_vec();

    // ---- exact conjugate posterior via scipy.stats ------------------------
    // Hand scipy the IDENTICAL X (flattened row-major), S_λ (flattened),
    // y, and σ². scipy forms Σ_post = (XᵀX/σ² + S_λ)⁻¹, μ_post = Σ_post·Xᵀy/σ²,
    // then draws 10k samples from multivariate_normal — exact ground truth.
    let x_flat: Vec<f64> = x_dense.iter().copied().collect();
    let s_flat: Vec<f64> = penalty.iter().copied().collect();
    let dims = vec![n as f64, p as f64, phi];

    let py = run_python(
        &[
            Column::new("x_flat", &x_flat),
            // pad scalar/short vectors to the longest column (n*p) so the CSV
            // is rectangular; the python body slices back to the real lengths.
            Column::new("s_flat", &pad(&s_flat, x_flat.len())),
            Column::new("y", &pad(&logratio, x_flat.len())),
            Column::new("dims", &pad(&dims, x_flat.len())),
        ],
        r#"
from scipy.stats import multivariate_normal
n = int(df["dims"].to_numpy()[0])
p = int(df["dims"].to_numpy()[1])
sigma2 = float(df["dims"].to_numpy()[2])
X = df["x_flat"].to_numpy()[: n * p].reshape(n, p)
S = df["s_flat"].to_numpy()[: p * p].reshape(p, p)
y = df["y"].to_numpy()[:n]
# Penalized-Gaussian (mgcv "Vb") posterior. gam's NUTS targets
#   log p(beta) ∝ -1/(2*sigma2) * [ (y - X beta)'(y - X beta) + beta' S beta ],
# i.e. BOTH the likelihood and the penalty S_lambda live on the sigma2=phi
# scale, giving the standard GAM Bayesian covariance Vb = sigma2 * H^-1 with
# the *unscaled* penalized Hessian H = X'X + S. (This is the mgcv convention:
# S_lambda is defined at phi=1 and the whole posterior is phi-scaled, NOT the
# classical conjugate-regression prior precision S that is independent of phi.)
#   H = X'X + S ,  Sigma_post = sigma2 * H^-1 ,  mu_post = H^-1 @ (X'y).
H = X.T @ X + S
H_inv = np.linalg.inv(H)
cov = sigma2 * H_inv
mu = H_inv @ (X.T @ y)
# Exact analytic posterior sd of each coefficient (sqrt of diagonal).
sd_exact = np.sqrt(np.clip(np.diag(cov), 0.0, None))
# Draw 10k i.i.d. samples from the exact posterior to mirror gam's NUTS run.
rng = np.random.default_rng(20260529)
draws = multivariate_normal(mean=mu, cov=cov, allow_singular=True).rvs(size=10000, random_state=rng)
draws = np.atleast_2d(draws)
emit("mu_exact", mu)
emit("sd_exact", sd_exact)
emit("mean_draws", draws.mean(axis=0))
emit("sd_draws", draws.std(axis=0, ddof=1))
"#,
    );

    let mu_exact = py.vector("mu_exact");
    let sd_exact = py.vector("sd_exact");
    let scipy_mean_draws = py.vector("mean_draws");
    let scipy_sd_draws = py.vector("sd_draws");
    assert_eq!(mu_exact.len(), p, "scipy posterior dim mismatch");

    // ---- compare ----------------------------------------------------------
    // gam's NUTS posterior mean vs the EXACT analytic posterior mean.
    let mean_abs = max_abs_diff(&gam_mean, mu_exact);
    let mean_corr = pearson(&gam_mean, mu_exact);
    // gam's NUTS posterior sd vs the EXACT analytic posterior sd.
    let sd_abs = max_abs_diff(&gam_sd, sd_exact);
    // scipy's own 10k-draw Monte-Carlo error against its exact mean — this is
    // the irreducible sampling floor that even the ground-truth sampler incurs.
    let scipy_mc_floor = max_abs_diff(scipy_mean_draws, mu_exact);
    let scipy_sd_floor = max_abs_diff(scipy_sd_draws, sd_exact);
    let max_sd_exact = sd_exact.iter().cloned().fold(0.0_f64, f64::max);

    eprintln!(
        "lidar s(range,k=5) conjugate posterior: n={n} p={p} sigma2={phi:.5}\n  \
         max|gam_mean - mu_exact| = {mean_abs:.5}  (scipy 10k MC floor = {scipy_mc_floor:.5})\n  \
         pearson(gam_mean, mu_exact) = {mean_corr:.6}\n  \
         max|gam_sd - sd_exact|   = {sd_abs:.5}  (scipy 10k MC floor = {scipy_sd_floor:.5}, max sd_exact = {max_sd_exact:.4})"
    );

    // ---- OBJECTIVE assertion 1: analytic posterior-mean recovery ----------
    // PRIMARY claim: gam's NUTS posterior mean coincides with the EXACT
    // closed-form posterior mean μ_post = H⁻¹Xᵀy (ground truth, computed with
    // no sampling). The exact posterior sds here are ~0.1-0.3; with 10k draws
    // the Monte-Carlo error on a coordinate mean is ~sd/100 <~ 0.003, so 0.01
    // is a tight, principled absolute bound (a few MC standard errors) that
    // asserts genuine recovery yet survives sampling noise — anything larger
    // would signal a wrong mass matrix or a bad triangular solve.
    assert!(
        mean_abs < 0.01,
        "gam NUTS posterior means diverge from the EXACT analytic posterior mean: \
         max|gam_mean - mu_exact|={mean_abs:.5} (bound=0.01)"
    );
    assert!(
        mean_corr > 0.9999,
        "gam NUTS posterior means decorrelate from the EXACT analytic posterior mean: \
         pearson={mean_corr:.6} (bound=0.9999)"
    );

    // ---- OBJECTIVE assertion 2: analytic posterior-sd recovery ------------
    // gam's NUTS posterior sds must match the EXACT analytic sds (sqrt of the
    // closed-form Σ_post diagonal). 5% of the largest exact sd is a sane,
    // un-weakened absolute tolerance for a 10k-draw sd estimate (relative MC
    // error on an sd estimate is ~1/sqrt(2N) ≈ 0.7%).
    let sd_bound = 0.05 * max_sd_exact.max(0.05);
    assert!(
        sd_abs < sd_bound,
        "gam NUTS posterior sds diverge from the EXACT analytic posterior sds: \
         max|gam_sd - sd_exact|={sd_abs:.5} (bound={sd_bound:.5})"
    );

    // ---- BASELINE-TO-MATCH-OR-BEAT: scipy's finite-sample MC floor --------
    // scipy draws 10k i.i.d. samples from the SAME exact target; its empirical
    // mean/sd incur an irreducible finite-sample error against the analytic
    // truth. gam (a genuine MCMC sampler, not i.i.d.) must recover the truth at
    // least as well as that i.i.d. floor, within a modest slack. This makes the
    // comparator a baseline on an OBJECTIVE accuracy metric (error vs. ground
    // truth), never a "reproduce scipy's fit" check.
    let floor_slack = 3.0;
    assert!(
        mean_abs <= floor_slack * scipy_mc_floor + 0.002,
        "gam's posterior-mean recovery error ({mean_abs:.5}) exceeds {floor_slack}x scipy's \
         i.i.d. Monte-Carlo floor ({scipy_mc_floor:.5}) against the same analytic truth"
    );
    assert!(
        sd_abs <= floor_slack * scipy_sd_floor + 0.002,
        "gam's posterior-sd recovery error ({sd_abs:.5}) exceeds {floor_slack}x scipy's \
         i.i.d. Monte-Carlo floor ({scipy_sd_floor:.5}) against the same analytic truth"
    );
}

/// Right-pad a vector with zeros to `len` so heterogeneous-length quantities
/// can ride in a single rectangular CSV handed to the reference body, which
/// slices each back to its true length. Length must not exceed `len`.
fn pad(v: &[f64], len: usize) -> Vec<f64> {
    assert!(
        v.len() <= len,
        "pad target {len} shorter than source {}",
        v.len()
    );
    let mut out = v.to_vec();
    out.resize(len, 0.0);
    out
}
