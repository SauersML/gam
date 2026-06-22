//! Regression: the shared-precision latent identifiability prior
//! (`AuxPrior` / `AuxPriorDimSelection` / `IsometryToReference`) must normalize
//! its REML log-precision term by the FULL number of governed scalar
//! coordinates, `K = n_obs * latent_dim`, not by `n_obs` alone.
//!
//! A single shared precision `mu` is the isotropic precision of a Gaussian over
//! every one of the `K = n_obs * latent_dim` flattened latent coordinates. The
//! prior log-determinant that enters the marginal likelihood is therefore
//! `log det_+(mu * I_K) = K * ln(mu)`, so the auto REML optimum is the plain
//! Gaussian precision MLE `mu_hat = K / Σr²` and the stationary score satisfies
//! `0.5 * mu * Σr² - 0.5 * K = 0`. Counting only `n_obs` undercounts the
//! log-determinant by exactly `latent_dim`, biasing `mu_hat` low by that factor.

use gam::terms::latent::aux_prior_reml_stats;
use ndarray::Array2;

/// With every residual entry equal, `Σr² = n_obs * latent_dim * cell²`, so the
/// auto precision is `mu = K / Σr² = 1 / cell²` independent of the shape — but
/// the buggy `n_obs`-only normalizer would report `mu = n_obs / Σr²`, smaller by
/// `latent_dim`. We pin both the value and the score's stationarity.
fn check(n_obs: usize, latent_dim: usize, cell: f64) {
    let t = Array2::<f64>::from_elem((n_obs, latent_dim), cell);
    let targets = Array2::<f64>::zeros((n_obs, latent_dim));
    let stats = aux_prior_reml_stats(t.view(), targets.view(), None)
        .expect("auto aux-prior stats must resolve for a finite non-zero residual");

    let residual_sq = (n_obs * latent_dim) as f64 * cell * cell;
    assert!(
        (stats.residual_sq - residual_sq).abs() < 1e-9,
        "residual_sq: got {}, expected {}",
        stats.residual_sq,
        residual_sq
    );

    // Correct count: K = n_obs * latent_dim.
    let k = (n_obs * latent_dim) as f64;
    let expected_mu = k / residual_sq;
    assert!(
        (stats.mu - expected_mu).abs() < 1e-9,
        "mu: got {}, expected K/Σr² = {} (n_obs={}, latent_dim={})",
        stats.mu,
        expected_mu,
        n_obs,
        latent_dim
    );

    // The buggy n_obs-only count would be smaller by exactly latent_dim.
    let buggy_mu = n_obs as f64 / residual_sq;
    assert!(
        (stats.mu / buggy_mu - latent_dim as f64).abs() < 1e-9,
        "the corrected precision must exceed the n_obs-only value by latent_dim={}; got ratio {}",
        latent_dim,
        stats.mu / buggy_mu
    );

    // Score and its stationarity in log_mu: d(score)/d(log_mu) = 0.5*mu*Σr² - 0.5*K.
    let expected_score = 0.5 * stats.mu * residual_sq - 0.5 * k * stats.log_mu;
    assert!(
        (stats.score - expected_score).abs() < 1e-9,
        "score: got {}, expected 0.5*mu*Σr² - 0.5*K*ln(mu) = {}",
        stats.score,
        expected_score
    );
    let dscore_dlogmu = 0.5 * stats.mu * residual_sq - 0.5 * k;
    assert!(
        dscore_dlogmu.abs() < 1e-9,
        "auto mu must be the score's stationary point: d(score)/d(log_mu) = {} (expected 0)",
        dscore_dlogmu
    );
}

#[test]
fn shared_precision_counts_all_latent_coordinates() {
    // Multi-dimensional latent: the regime the bug corrupted.
    check(4, 2, 2.0_f64.sqrt());
    check(7, 3, 0.5);
    check(10, 5, 1.25);
}

#[test]
fn scalar_latent_is_unchanged() {
    // latent_dim == 1 ⇒ K == n_obs, so the corrected normalizer reproduces the
    // historical scalar-latent behavior exactly (no regression there).
    let n_obs = 6;
    let cell = 0.75;
    let t = Array2::<f64>::from_elem((n_obs, 1), cell);
    let targets = Array2::<f64>::zeros((n_obs, 1));
    let stats = aux_prior_reml_stats(t.view(), targets.view(), None).expect("auto stats");
    let residual_sq = n_obs as f64 * cell * cell;
    assert!((stats.mu - n_obs as f64 / residual_sq).abs() < 1e-9);
}

#[test]
fn fixed_precision_score_uses_full_count() {
    // With a user-fixed precision the value of `mu` is not re-selected, but the
    // reported score's normalizer must still count K = n_obs * latent_dim.
    let (n_obs, latent_dim, mu) = (5usize, 4usize, 0.3_f64);
    let t = Array2::<f64>::from_elem((n_obs, latent_dim), 1.0);
    let targets = Array2::<f64>::zeros((n_obs, latent_dim));
    let stats = aux_prior_reml_stats(t.view(), targets.view(), Some(mu)).expect("fixed stats");
    let residual_sq = (n_obs * latent_dim) as f64;
    let k = (n_obs * latent_dim) as f64;
    let expected_score = 0.5 * mu * residual_sq - 0.5 * k * mu.ln();
    assert!(
        (stats.score - expected_score).abs() < 1e-9,
        "fixed-strength score: got {}, expected {}",
        stats.score,
        expected_score
    );
}
