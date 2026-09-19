//! gam#2960: when the resource governor refuses the dense covariance bundle,
//! the standard optimizer takes its factorized inference branch and publishes
//! standard errors with no covariance. On a design whose unpenalized
//! non-intercept column is conditioned (centred and scaled, its mean folded into
//! the intercept), the published coordinates' variance is `diag(M·Σ·Mᵀ)`, which
//! reads the intercept cross-covariances, so a diagonal solved in the internal
//! coordinates cannot be carried back and those standard errors used to be
//! dropped. The branch now solves the published diagonal directly.
//!
//! This test lives in its own binary because it reserves the process-wide
//! memory governor down to the factorized regime, which would starve any test
//! running beside it in the same process. The governor-forcing arm is adapted
//! from gam-2929's `factorized_fit_publishes_standard_errors_without_a_covariance_2955`.

use gam_problem::LikelihoodSpec;
use gam_solve::estimate::{
    ExternalOptimOptions, ExternalOptimResult, optimize_external_designwith_heuristic_log_lambdas,
};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};

const ROWS: usize = 240;
const COLUMNS: usize = 24;

/// Intercept, one unpenalized covariate with a non-zero mean and a non-unit
/// spread (so the column conditioning is active), and penalized harmonics.
fn conditioned_design() -> (Array1<f64>, Array2<f64>) {
    let grid: Vec<f64> = (0..ROWS).map(|i| (i as f64 + 0.5) / ROWS as f64).collect();
    let x = Array2::from_shape_fn((ROWS, COLUMNS), |(i, j)| match j {
        0 => 1.0,
        1 => 3.0 + 2.0 * grid[i],
        _ => {
            let harmonic = (j / 2) as f64;
            let angle = 2.0 * std::f64::consts::PI * harmonic * grid[i];
            if j % 2 == 0 { angle.sin() } else { angle.cos() }
        }
    });
    let y = Array1::from_iter(grid.iter().enumerate().map(|(i, t)| {
        1.5 * (3.0 + 2.0 * t)
            + (2.0 * std::f64::consts::PI * t).sin()
            + if i % 3 == 0 { 0.2 } else { -0.1 }
    }));
    (y, x)
}

fn fit(y: &Array1<f64>, x: &Array2<f64>) -> ExternalOptimResult {
    let weights = Array1::<f64>::ones(ROWS);
    let offset = Array1::<f64>::zeros(ROWS);
    let penalties = vec![BlockwisePenalty::new(
        2..COLUMNS,
        Array2::<f64>::eye(COLUMNS - 2),
    )];
    let options = ExternalOptimOptions {
        family: LikelihoodSpec::gaussian_identity(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: true,
        max_iter: 80,
        tol: 1e-7,
        nullspace_dims: vec![0],
        linear_constraints: None,
        firth_bias_reduction: None,
        rho_prior: Default::default(),
        persistent_warm_start_store: None,
    };
    optimize_external_designwith_heuristic_log_lambdas(
        y.view(),
        weights.view(),
        x.clone(),
        offset.view(),
        penalties,
        None,
        &options,
    )
    .unwrap_or_else(|error| panic!("the gam#2960 fit must succeed: {error:?}"))
}

#[test]
fn factorized_standard_errors_agree_with_the_dense_fit_on_a_conditioned_design_2960() {
    let (y, x) = conditioned_design();

    let dense = fit(&y, &x);
    let covariance = dense
        .covariance_conditional
        .as_ref()
        .expect("at full budget the fit publishes its dense conditional covariance");
    assert!(
        dense
            .inference
            .as_ref()
            .expect("the fit publishes inference")
            .factorized_standard_errors
            .is_none(),
        "a fit that published its covariance carries no factorized standard errors"
    );
    let dense_standard_errors =
        gam_problem::se_from_covariance(covariance).expect("the dense diagonal is valid");

    // Leave 26 p×p f64 matrices on the ledger: more than the factorized
    // inference state reserves (seven), fewer than the dense covariance bundle
    // (fifty-two), so the fit takes the factorized branch.
    let governor = gam_runtime::resource::MemoryGovernor::global();
    let leave = 26 * COLUMNS * COLUMNS * std::mem::size_of::<f64>();
    let hold = governor
        .try_reserve(
            governor.remaining_bytes().saturating_sub(leave),
            "gam#2960 test: refuse the dense covariance bundle",
        )
        .expect("the test reserves the governor down to the factorized regime");
    let factorized = fit(&y, &x);
    drop(hold);

    assert!(
        factorized.covariance_conditional.is_none(),
        "with the dense bundle refused the fit publishes no dense conditional covariance"
    );
    let standard_errors = factorized
        .inference
        .as_ref()
        .expect("the factorized fit publishes inference")
        .factorized_standard_errors
        .as_ref()
        .expect("on a conditioned design the factorized branch publishes standard errors");
    assert_eq!(standard_errors.len(), COLUMNS);
    let mut worst = 0.0_f64;
    for (index, (&got, &want)) in standard_errors
        .iter()
        .zip(dense_standard_errors.iter())
        .enumerate()
    {
        let gap = (got - want).abs() / want.max(f64::MIN_POSITIVE);
        worst = worst.max(gap);
        assert!(
            gap <= 1e-8,
            "standard error {index}: factorized {got:.17e} vs dense back-transformed {want:.17e}"
        );
    }
    eprintln!("[2960 factorized SE] p={COLUMNS} worst relative gap to the dense fit={worst:.3e}");
}
