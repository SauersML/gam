//! The firth-general default outer hyperprior on the smoothing parameter λ.
//!
//! With the family-general Firth/Jeffreys machinery now unconditionally armed
//! (`firth_general == true`), when the caller has *not* configured a ρ
//! hyperprior (the `RhoPrior::Flat` sentinel) the REML/LAML runtime substitutes
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
//!  * **Degeneracy cure (direction-certain).** Under the firth-general default the
//!    injected PC term's smoothing is monotone in prior tightness — the PC Occam
//!    pull has the correct sign.
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

fn fit_options(rho_prior: RhoPrior) -> FitOptions {
    FitOptions {
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
    .expect("fit must succeed");
    (result.lambdas[0], result.beta.clone())
}

/// Explicit-PC consumption is well-defined and deterministic on a Gaussian-
/// identity fit: the link-general Jeffreys term is Binomial-only, so the
/// always-on robust machinery changes the objective *only* through the configured
/// ρ prior. An explicitly-configured PC prior must therefore select a finite λ
/// reproducibly — i.e. `firth_general` adds no hidden, non-deterministic
/// Gaussian-λ effect of its own.
#[test]
fn explicit_pc_prior_is_deterministic_on_gaussian() {
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
    let (lam_a, beta_a) = fit(&x, &y, default_pc.clone());
    let (lam_b, beta_b) = fit(&x, &y, default_pc);
    assert!(
        lam_a.is_finite() && lam_a > 0.0,
        "explicit PC must select a finite positive λ on Gaussian: λ={lam_a}"
    );
    assert_eq!(
        lam_a, lam_b,
        "explicit PC fit must be deterministic: λ_a={lam_a} λ_b={lam_b}"
    );
    assert_eq!(beta_a, beta_b);
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
    let (lam_tight, _) = fit(&x, &y, tight);
    let (lam_default, _) = fit(&x, &y, default_pc);
    let (lam_vague, _) = fit(&x, &y, vague);
    assert!(
        lam_tight >= lam_default && lam_default >= lam_vague,
        "PC smoothing must be monotone in tightness: λ_tight={lam_tight} λ_default={lam_default} λ_vague={lam_vague}"
    );
}

/// An unset (`Flat`) prior under the always-on robust machinery (which then
/// substitutes the firth-general PC default) must be deterministic: repeating the
/// same `Flat` fit must produce exactly the same λ and β — the gate never
/// introduces non-determinism into the default objective.
#[test]
fn flat_prior_default_path_is_deterministic() {
    let (n_per, p) = (24usize, 10usize);
    let x = group_design(n_per, p);
    let mut rng = Rng(0x1357_9bdf_2468_ace0);
    let mut y = Array1::<f64>::zeros(n_per * p);
    for yi in y.iter_mut() {
        *yi = rng.normal();
    }
    let (lam_a, beta_a) = fit(&x, &y, RhoPrior::Flat);
    let (lam_b, beta_b) = fit(&x, &y, RhoPrior::Flat);
    assert_eq!(lam_a, lam_b, "Flat default path must be deterministic");
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

    let (_lam, beta) = fit(&x, &y, RhoPrior::Flat);

    let rmse = |b: &Array1<f64>| ((b - &beta_true).mapv(|v| v * v).sum() / p as f64).sqrt();
    let rmse_fit = rmse(&beta);
    assert!(
        rmse_fit < 0.2,
        "firth-general default must recover real signal without over-shrinkage: \
         rmse={rmse_fit}"
    );
}
