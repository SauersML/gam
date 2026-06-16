//! Regression for #819: a Gaussian `group()` random-intercept fit must not
//! abort inside the Rust boundary once the coefficient count crosses the
//! sparse-exact REML threshold (`p >= 256` / `n_obs * p` above the small-problem
//! budget).
//!
//! Root cause (the angle this Rust test pins down): the sparse-exact REML
//! geometry carried a *placeholder* empty (0x0) dense inner Hessian in the
//! eval bundle, and `RemlState::objective_innerhessian` returned it verbatim.
//! The post-fit first-order smoothing correction
//! (`compute_smoothing_correction`) then Cholesky-factored that 0x0 matrix
//! (faer happily produces an `n = 0` factor, so the existing `Err` guard never
//! fired) and solved it against a length-`p` right-hand side, tripping faer's
//! `rhs.nrows() == n` assertion and aborting the whole fit.
//!
//! The fix materializes the real penalized inner Hessian for the sparse-exact
//! geometry from the simplicial Cholesky factor the inner loop already built.
//! The sharpest observable consequence — stronger than "it merely fits" — is
//! that the smoothing-parameter uncertainty correction now *succeeds*: the
//! corrected covariance `Vp` (`covariance_corrected`) is materialized rather
//! than the process aborting. We assert exactly that, plus a finite REML score
//! and recovery of the per-group structure the data were generated from.

use csv::StringRecord;
use gam::pirls::PirlsStatus;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Build a balanced random-intercept panel: `n_groups` sites, `per_group`
/// observations each, `y = intercept + b_g + noise`. Returns the encoded
/// dataset and the true per-group mean `intercept + b_g`.
fn make_panel(
    n_groups: usize,
    per_group: usize,
    seed: u64,
) -> (gam::inference::data::EncodedDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let between = Normal::new(0.0_f64, 2.0_f64).expect("between-group");
    let within = Normal::new(0.0_f64, 1.0_f64).expect("within-group");
    let intercept = 5.0_f64;

    let true_effect: Vec<f64> = (0..n_groups).map(|_| between.sample(&mut rng)).collect();
    let true_group_mean: Vec<f64> = true_effect.iter().map(|b| intercept + b).collect();

    let headers = ["site", "y"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n_groups * per_group);
    for (g, b) in true_effect.iter().enumerate() {
        for _ in 0..per_group {
            let y = intercept + b + within.sample(&mut rng);
            rows.push(StringRecord::from(vec![format!("g{g}"), y.to_string()]));
        }
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode panel");
    (data, true_group_mean)
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        cov += (x - ma) * (y - mb);
        va += (x - ma) * (x - ma);
        vb += (y - mb) * (y - mb);
    }
    cov / (va.sqrt() * vb.sqrt())
}

#[test]
fn group_random_intercept_fits_above_sparse_exact_threshold() {
    init_parallelism();

    // 300 groups => p = 301 >= 256: the geometry selector routes to
    // RemlGeometry::SparseExactSpd (random-intercept Gram is extremely sparse).
    // Pre-fix this aborts the process at the empty (0x0) Hessian solve.
    let n_groups = 300;
    let per_group = 20;
    let (data, true_group_mean) = make_panel(n_groups, per_group, 0);

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let result =
        fit_from_formula("y ~ group(site)", &data, &cfg).expect("group(site) panel must fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard fit for y ~ group(site)");
    };

    // The fit must converge with a finite marginal-likelihood objective.
    assert_eq!(fit.fit.pirls_status, PirlsStatus::Converged);
    assert!(
        fit.fit.reml_score.is_finite(),
        "reml_score must be finite, got {}",
        fit.fit.reml_score
    );

    // The exact root cause: the smoothing-parameter uncertainty correction must
    // actually run on the sparse-exact geometry. `covariance_corrected` (Vp) is
    // the product of `compute_smoothing_correction` — the function that aborted
    // pre-fix. Its presence (finite, symmetric, p x p) proves the real inner
    // Hessian was materialized for the sparse-exact path.
    let vp = fit
        .fit
        .covariance_corrected
        .as_ref()
        .expect("corrected covariance Vp must be materialized (smoothing correction must run)");
    let p = fit.fit.beta.len();
    assert_eq!(vp.shape(), [p, p], "Vp must be p x p");
    assert!(
        vp.iter().all(|v| v.is_finite()),
        "Vp entries must all be finite"
    );
    // Vp must be a valid covariance contribution: non-negative diagonal.
    for i in 0..p {
        assert!(
            vp[[i, i]] >= -1e-9,
            "Vp diagonal entry {i} is materially negative: {}",
            vp[[i, i]]
        );
    }

    // And the fit must recover the group structure it was generated from. The
    // first coefficient is the global intercept; the remaining `n_groups`
    // coefficients are the per-group deviations (shrunken BLUPs). Their sum with
    // the intercept must track the true per-group means.
    assert_eq!(
        p,
        n_groups + 1,
        "expected intercept + one coefficient per group, got p={p}"
    );
    let intercept = fit.fit.beta[0];
    let blup_means: Vec<f64> = (0..n_groups)
        .map(|g| intercept + fit.fit.beta[g + 1])
        .collect();
    let corr = pearson(&blup_means, &true_group_mean);
    assert!(
        corr > 0.9,
        "random-intercept BLUPs failed to recover the group means: corr={corr}"
    );
}

#[test]
fn group_random_intercept_just_below_threshold_still_fits() {
    init_parallelism();
    // 254 groups => p = 255 < 256: routes to the dense geometry and already
    // worked. Kept as the control so the test pins the *boundary*, not just one
    // size — both sides of the geometry split must produce a usable corrected
    // covariance.
    let n_groups = 254;
    let per_group = 8;
    let (data, true_group_mean) = make_panel(n_groups, per_group, 1);

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let result = fit_from_formula("y ~ group(site)", &data, &cfg)
        .expect("254-group panel must fit (control)");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard fit");
    };
    assert_eq!(fit.fit.pirls_status, PirlsStatus::Converged);
    assert!(fit.fit.reml_score.is_finite());
    assert!(
        fit.fit.covariance_corrected.is_some(),
        "control fit must also materialize Vp"
    );
    let intercept = fit.fit.beta[0];
    let blup_means: Vec<f64> = (0..n_groups)
        .map(|g| intercept + fit.fit.beta[g + 1])
        .collect();
    assert!(pearson(&blup_means, &true_group_mean) > 0.9);
}
