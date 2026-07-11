//! End-to-end behaviour of the penalized-complexity (PC) smoothing prior
//! (`RhoPrior::PenalizedComplexity`, issue #463).
//!
//! The model is a balanced one-way Gaussian random effect: `p` orthogonal group
//! indicators with an identity (ridge) penalty, so the single smoothing
//! parameter `λ = exp(ρ)` is the random-effect precision and REML/LAML selects
//! it. The PC prior measures complexity by the marginal coefficient standard
//! deviation `d = exp(-ρ/2)` and places an exponential prior on it with rate
//! `θ = -ln(tail_prob)/upper`. Two end-to-end properties are pinned:
//!
//!  * **Monotone informativeness.** The combined REML+prior objective is
//!    `F(ρ) = R(ρ) + ρ/2 + θ·exp(-ρ/2)`. Since `∂F'/∂θ = -½·exp(-ρ/2) < 0`,
//!    the implicit-function theorem gives `dρ̂/dθ = (½·exp(-ρ/2))/F''(ρ̂) > 0`
//!    at any interior minimum: a *tighter* PC prior (smaller `upper` ⇒ larger
//!    `θ`) selects strictly more smoothing (larger `λ`, smaller fitted
//!    structure) on *any* data. This is the direction-certain behavioural
//!    signature — comparing against a flat prior is not, because on degenerate
//!    (pure-noise) data flat REML already drives `λ` past the PC mode.
//!  * **No over-shrinkage of real signal.** When the group means are genuinely
//!    distinct and the likelihood is informative, the PC prior recovers them
//!    about as well as a flat prior.

use gam::estimate::{FitOptions, PenaltySpec, fit_gam_with_penalty_specs};
use gam::matrix::{DenseDesignMatrix, DesignMatrix};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, RhoPrior, StandardLink};
use ndarray::{Array1, Array2};

/// Tiny deterministic standard-normal stream (LCG + Box–Muller) so the
/// simulation is fully reproducible without a dependency on `rand`.
struct Rng(u64);
impl Rng {
    fn next_u(&mut self) -> f64 {
        // Numerical Recipes LCG; take the top 53 bits as a uniform in (0, 1).
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.0 >> 11) as f64;
        (bits + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.next_u().max(1e-300);
        let u2 = self.next_u();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn fit_options(rho_prior: RhoPrior) -> FitOptions {
    FitOptions {
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: false,
        max_iter: 100,
        tol: 1e-8,
        nullspace_dims: Vec::new(),
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior,
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    }
}

fn gaussian_identity() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    )
}

/// Balanced group-indicator design: `n_per` rows per group, `p` groups,
/// columns mutually orthogonal so `XᵀX = n_per · I`.
fn group_design(n_per: usize, p: usize) -> Array2<f64> {
    let n = n_per * p;
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, i % p]] = 1.0;
    }
    x
}

fn fit(x: &Array2<f64>, y: &Array1<f64>, prior: RhoPrior) -> (f64, Array1<f64>) {
    let p = x.ncols();
    let n = x.nrows();
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let design = DesignMatrix::Dense(DenseDesignMatrix::from(x.clone()));
    let result = fit_gam_with_penalty_specs(
        design,
        y.view(),
        weights.view(),
        offset.view(),
        vec![PenaltySpec::Dense(Array2::<f64>::eye(p))],
        vec![0],
        gaussian_identity(),
        &fit_options(prior),
    )
    .expect("PC-prior fit must succeed");
    (result.lambdas[0], result.beta.clone())
}

fn l2(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

/// Monotone informativeness: on the *same* low-signal data, a tighter PC prior
/// (smaller `upper` ⇒ larger rate `θ`) must select strictly more smoothing
/// (larger `λ`) and a smaller fitted structure than a vague one. This is the
/// `dρ̂/dθ > 0` property derived in the module docs, and it holds on any data
/// with a unique interior minimum — no dependence on where flat REML happens to
/// land.
#[test]
fn tighter_pc_prior_selects_more_smoothing() {
    let (n_per, p) = (24usize, 10usize);
    let x = group_design(n_per, p);
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut y = Array1::<f64>::zeros(n_per * p);
    for yi in y.iter_mut() {
        *yi = rng.normal(); // truth is exactly zero; only noise.
    }

    // Same tail probability, different upper bounds: `upper = 0.1` demands the
    // smooth stay within a marginal SD of 0.1 (θ ≈ 29.96), `upper = 5.0` is
    // nearly vague (θ ≈ 0.599).
    let (lam_tight, beta_tight) = fit(
        &x,
        &y,
        RhoPrior::PenalizedComplexity {
            upper: 0.1,
            tail_prob: 0.05,
        },
    );
    let (lam_loose, beta_loose) = fit(
        &x,
        &y,
        RhoPrior::PenalizedComplexity {
            upper: 5.0,
            tail_prob: 0.05,
        },
    );

    // More smoothing: larger selected precision λ.
    assert!(
        lam_tight > lam_loose,
        "tighter PC must select more smoothing: λ_tight={lam_tight} λ_loose={lam_loose}"
    );
    // Less fitted structure: stronger ridge shrinkage of the group means.
    let struct_tight = l2(&x.dot(&beta_tight));
    let struct_loose = l2(&x.dot(&beta_loose));
    assert!(
        struct_tight < struct_loose,
        "tighter PC must shrink structure more: ||Xβ_tight||={struct_tight} ||Xβ_loose||={struct_loose}"
    );
}

/// Signal present: distinct, mean-zero group means with a small noise floor and
/// an informative likelihood. The PC prior must not over-shrink — it recovers
/// the truth about as well as the flat prior (and far better than the flat
/// prior recovers nothing on the low-signal case).
#[test]
fn pc_prior_recovers_real_signal() {
    let (n_per, p) = (24usize, 10usize);
    let x = group_design(n_per, p);
    let mut beta_true = Array1::<f64>::zeros(p);
    for j in 0..p {
        beta_true[j] = j as f64 - (p as f64 - 1.0) / 2.0; // spread, mean zero.
    }
    let mut rng = Rng(0xfeed_face_cafe_d00d);
    let mut y = Array1::<f64>::zeros(n_per * p);
    for i in 0..y.len() {
        y[i] = beta_true[i % p] + 0.15 * rng.normal();
    }

    let (_lam_flat, beta_flat) = fit(&x, &y, RhoPrior::Flat);
    let (_lam_pc, beta_pc) = fit(
        &x,
        &y,
        RhoPrior::PenalizedComplexity {
            upper: 0.5,
            tail_prob: 0.05,
        },
    );

    let rmse = |b: &Array1<f64>| ((b - &beta_true).mapv(|v| v * v).sum() / p as f64).sqrt();
    let rmse_pc = rmse(&beta_pc);
    let rmse_flat = rmse(&beta_flat);
    // PC recovers the genuine signal: small absolute error and no material
    // degradation versus the flat prior despite its shrinkage pull.
    assert!(
        rmse_pc < 0.2,
        "PC must recover real signal: rmse_pc={rmse_pc}"
    );
    assert!(
        rmse_pc < 1.5 * rmse_flat + 0.05,
        "PC must not over-shrink real signal: rmse_pc={rmse_pc} rmse_flat={rmse_flat}"
    );
}
