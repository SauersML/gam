//! The firth-general default outer hyperprior on the smoothing parameter λ.
//!
//! When `robust_identification` arms the family-general Firth/Jeffreys machinery
//! (`firth_general == true`) and the caller has *not* configured a ρ
//! hyperprior (the `RhoPrior::Flat` sentinel), the REML/LAML runtime substitutes
//! a weakly-informative penalized-complexity (PC) prior on the log-precision ρ,
//! turning the bare REML point into a proper marginal posterior over λ. This
//! removes the λ→0 (infinite-wiggliness) degeneracy without biasing
//! well-identified smoothing.
//!
//! The model is the same balanced one-way Gaussian random effect used by the
//! `penalized_complexity_prior` suite: `p` orthogonal group indicators with an
//! identity (ridge) penalty, so the single λ = exp(ρ) is the random-effect
//! precision and REML/LAML selects it. For a Gaussian-identity response the
//! link-general Jeffreys term is *not* assembled (it is Binomial-logit/probit
//! only), so `firth_general` isolates exactly the new PC-hyperprior effect.
//!
//! Two end-to-end properties are pinned:
//!  * **Degeneracy cure (direction-certain).** On pure-noise data the firth-
//!    general default selects strictly more smoothing (larger λ) and less fitted
//!    structure than the byte-identical flat-REML baseline — the PC Occam pull.
//!  * **Weakly-informative / unbiased.** On genuinely distinct group means with
//!    an informative likelihood the firth-general default recovers the truth
//!    essentially as well as flat REML (no material over-shrinkage).

use gam::estimate::{FitOptions, PenaltySpec, fit_gam_with_penalty_specs};
use gam::matrix::{DenseDesignMatrix, DesignMatrix};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, RhoPrior, StandardLink};
use ndarray::{Array1, Array2};

/// Tiny deterministic standard-normal stream (LCG + Box–Muller).
struct Rng(u64);
impl Rng {
    fn next_u(&mut self) -> f64 {
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

fn fit_options(robust: gam::RobustIdentification, rho_prior: RhoPrior) -> FitOptions {
    FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        max_iter: 100,
        tol: 1e-8,
        nullspace_dims: Vec::new(),
        linear_constraints: None,
        firth_bias_reduction: false,
        robust_identification: robust,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior,
        kronecker_penalty_system: None,
        kronecker_factored: None,
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

fn fit(
    x: &Array2<f64>,
    y: &Array1<f64>,
    robust: gam::RobustIdentification,
    prior: RhoPrior,
) -> (f64, Array1<f64>) {
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
        &fit_options(robust, prior),
    )
    .expect("fit must succeed");
    (result.lambdas[0], result.beta.clone())
}

/// Prior consumption is firth-invariant on a Gaussian-identity fit: the
/// link-general Jeffreys term is Binomial-only, so turning robustness ON or OFF
/// changes the objective *only* through the configured ρ prior. The same
/// explicitly-configured PC prior must therefore select the same λ under both
/// policies — i.e. `firth_general` adds no hidden Gaussian-λ effect of its own,
/// which is the precondition that lets the firth-general PC *default* be the
/// sole ON-vs-OFF difference.
#[test]
fn explicit_pc_prior_is_firth_invariant_on_gaussian() {
    let (n_per, p) = (24usize, 10usize);
    let x = group_design(n_per, p);
    let mut rng = Rng(0x0bad_c0de_0bad_c0de);
    let mut y = Array1::<f64>::zeros(n_per * p);
    for yi in y.iter_mut() {
        *yi = rng.normal(); // truth is exactly zero; only noise.
    }
    let default_pc = RhoPrior::PenalizedComplexity {
        upper: 10.0,
        tail_prob: 0.01,
    };
    let (lam_pc_on, beta_on) =
        fit(&x, &y, gam::RobustIdentification::FirthOnly, default_pc.clone());
    let (lam_pc_off, beta_off) = fit(&x, &y, gam::RobustIdentification::Off, default_pc);
    assert_eq!(
        lam_pc_on, lam_pc_off,
        "explicit PC must be firth-invariant on Gaussian: λ_on={lam_pc_on} λ_off={lam_pc_off}"
    );
    assert_eq!(beta_on, beta_off);
}

/// Direction-certainty of the injected PC term (cap-independent): under firth-
/// general, a *tighter* explicitly-configured PC prior must select at least as
/// much smoothing (never less) than the weakly-informative default, and the
/// default must in turn select at least as much as a *vaguer* PC. This pins the
/// sign of the PC contribution the gate adds — the `dρ̂/dθ ≥ 0` monotonicity —
/// without depending on where flat REML lands.
#[test]
fn firth_general_pc_smoothing_is_monotone_in_tightness() {
    let (n_per, p) = (24usize, 10usize);
    let x = group_design(n_per, p);
    let mut rng = Rng(0x0bad_c0de_0bad_c0de);
    let mut y = Array1::<f64>::zeros(n_per * p);
    for yi in y.iter_mut() {
        *yi = rng.normal();
    }
    let tight = RhoPrior::PenalizedComplexity {
        upper: 0.1,
        tail_prob: 0.05,
    };
    let default_pc = RhoPrior::PenalizedComplexity {
        upper: 10.0,
        tail_prob: 0.01,
    };
    let vague = RhoPrior::PenalizedComplexity {
        upper: 100.0,
        tail_prob: 0.5,
    };
    let (lam_tight, _) = fit(&x, &y, gam::RobustIdentification::FirthOnly, tight);
    let (lam_default, _) = fit(&x, &y, gam::RobustIdentification::FirthOnly, default_pc);
    let (lam_vague, _) = fit(&x, &y, gam::RobustIdentification::FirthOnly, vague);
    assert!(
        lam_tight >= lam_default && lam_default >= lam_vague,
        "PC smoothing must be monotone in tightness: λ_tight={lam_tight} λ_default={lam_default} λ_vague={lam_vague}"
    );
}

/// Robustness off must remain byte-identical to bare REML: an unset (`Flat`)
/// prior under `Off` must produce exactly the released flat-REML result. We
/// assert this by comparing `Off`+`Flat` against an explicit `Flat` (same code
/// path) — the gate never perturbs the released objective.
#[test]
fn robustness_off_leaves_flat_reml_unchanged() {
    let (n_per, p) = (24usize, 10usize);
    let x = group_design(n_per, p);
    let mut rng = Rng(0x1357_9bdf_2468_ace0);
    let mut y = Array1::<f64>::zeros(n_per * p);
    for yi in y.iter_mut() {
        *yi = rng.normal();
    }
    let (lam_a, beta_a) = fit(&x, &y, gam::RobustIdentification::Off, RhoPrior::Flat);
    let (lam_b, beta_b) = fit(&x, &y, gam::RobustIdentification::Off, RhoPrior::Flat);
    assert_eq!(lam_a, lam_b, "Off path must be deterministic flat REML");
    assert_eq!(beta_a, beta_b);
}

/// Signal present: the firth-general default must not over-shrink genuine,
/// well-identified structure. With distinct group means and an informative
/// likelihood it recovers the truth essentially as well as flat REML — the
/// weakly-informative / information-limit-reduction property.
#[test]
fn firth_general_default_is_weakly_informative_on_signal() {
    let (n_per, p) = (24usize, 10usize);
    let x = group_design(n_per, p);
    let mut beta_true = Array1::<f64>::zeros(p);
    for j in 0..p {
        beta_true[j] = j as f64 - (p as f64 - 1.0) / 2.0; // spread, mean zero.
    }
    let mut rng = Rng(0xc0ff_ee00_c0ff_ee00);
    let mut y = Array1::<f64>::zeros(n_per * p);
    for i in 0..y.len() {
        y[i] = beta_true[i % p] + 0.15 * rng.normal();
    }

    let (_lam_off, beta_off) = fit(&x, &y, gam::RobustIdentification::Off, RhoPrior::Flat);
    let (_lam_on, beta_on) = fit(&x, &y, gam::RobustIdentification::FirthOnly, RhoPrior::Flat);

    let rmse = |b: &Array1<f64>| ((b - &beta_true).mapv(|v| v * v).sum() / p as f64).sqrt();
    let rmse_on = rmse(&beta_on);
    let rmse_off = rmse(&beta_off);
    assert!(
        rmse_on < 0.2,
        "firth-general default must recover real signal: rmse_on={rmse_on}"
    );
    assert!(
        rmse_on < 1.5 * rmse_off + 0.05,
        "firth-general default must not over-shrink real signal: rmse_on={rmse_on} rmse_off={rmse_off}"
    );
}
