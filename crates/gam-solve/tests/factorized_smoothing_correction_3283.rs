//! gam#3283: when the resource governor refuses the dense covariance bundle,
//! the standard optimizer takes its factorized inference branch (#2960). That
//! branch used to publish the conditional standard errors and nothing else: no
//! smoothing-parameter correction and no typed reason for its absence, so every
//! default uncertainty surface fell back to the conditional law. It now
//! assembles the same first-order correction the dense branch does, keeps it as
//! its square-root factor `B` (`C = B·Bᵀ`, `p × r`), and publishes the
//! corrected standard errors of `Vp = Vb + B·Bᵀ` beside the conditional ones.
//!
//! This test lives in its own binary because it reserves the process-wide
//! memory governor down to the factorized regime, which would starve any test
//! running beside it in the same process (the #2960 test's arrangement).

use gam_problem::LikelihoodSpec;
use gam_solve::estimate::{
    ExternalOptimOptions, ExternalOptimResult, optimize_external_designwith_heuristic_log_lambdas,
};
use gam_solve::model_types::SmoothingCorrectionMethod;
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};

const ROWS: usize = 96;
const COLUMNS: usize = 21;

/// An intercept and penalized harmonics. No unpenalized non-intercept column,
/// so the parametric column conditioning is inactive and the published frame is
/// the internal one: the dense branch's correction diagonal and the factorized
/// branch's `‖b_i‖²` are then the same sums of squares of the same factor.
fn harmonic_design() -> (Array1<f64>, Array2<f64>) {
    let grid: Vec<f64> = (0..ROWS).map(|i| (i as f64 + 0.5) / ROWS as f64).collect();
    let x = Array2::from_shape_fn((ROWS, COLUMNS), |(i, j)| match j {
        0 => 1.0,
        _ => {
            let harmonic = j.div_ceil(2) as f64;
            let angle = 2.0 * std::f64::consts::PI * harmonic * grid[i];
            if j % 2 == 1 { angle.sin() } else { angle.cos() }
        }
    });
    let y = Array1::from_iter(grid.iter().enumerate().map(|(i, t)| {
        0.5 + (2.0 * std::f64::consts::PI * t).sin()
            + 0.3 * (6.0 * std::f64::consts::PI * t).cos()
            + if i % 3 == 0 { 0.2 } else { -0.1 }
    }));
    (y, x)
}

fn fit(y: &Array1<f64>, x: &Array2<f64>) -> ExternalOptimResult {
    let weights = Array1::<f64>::ones(ROWS);
    let offset = Array1::<f64>::zeros(ROWS);
    let penalties = vec![BlockwisePenalty::new(
        1..COLUMNS,
        Array2::<f64>::eye(COLUMNS - 1),
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
    .unwrap_or_else(|error| panic!("the gam#3283 fit must succeed: {error:?}"))
}

#[test]
fn factorized_fit_publishes_the_smoothing_correction_the_dense_fit_publishes_3283() {
    let (y, x) = harmonic_design();

    let dense = fit(&y, &x);
    let conditional = dense
        .covariance_conditional
        .as_ref()
        .expect("at full budget the fit publishes its dense conditional covariance");
    let corrected = dense
        .covariance_corrected
        .as_ref()
        .expect("at full budget the fit publishes its dense corrected covariance");
    let dense_inference = dense.inference.as_ref().expect("the dense fit publishes inference");
    let dense_correction = dense_inference
        .smoothing_correction
        .as_ref()
        .expect("the dense fit publishes its smoothing correction");
    assert!(
        dense_inference.smoothing_correction_factorized.is_none(),
        "a fit that published its covariance carries no factorized correction"
    );

    // Leave 26 p×p f64 matrices on the ledger, the #2960 test's regime: more
    // than the factorized inference state (seven) plus the correction's
    // workspace (eight), fewer than the dense covariance bundle.
    let governor = gam_runtime::resource::MemoryGovernor::global();
    let leave = 26 * COLUMNS * COLUMNS * std::mem::size_of::<f64>();
    let hold = governor
        .try_reserve(
            governor.remaining_bytes().saturating_sub(leave),
            "gam#3283 test: refuse the dense covariance bundle",
        )
        .expect("the test reserves the governor down to the factorized regime");
    let factorized = fit(&y, &x);
    drop(hold);

    assert!(
        factorized.covariance_conditional.is_none() && factorized.covariance_corrected.is_none(),
        "with the dense bundle refused the fit publishes no dense covariance"
    );
    let inference = factorized
        .inference
        .as_ref()
        .expect("the factorized fit publishes inference");
    assert_eq!(
        inference.smoothing_correction_absence, None,
        "the factorized fit carries its correction, so it records no absence"
    );
    let published = inference.smoothing_correction_factorized.as_ref().unwrap_or_else(|| {
        panic!(
            "the factorized fit publishes its smoothing correction; it recorded {:?}",
            inference.smoothing_correction_absence
        )
    });
    assert!(
        matches!(
            inference.smoothing_correction_method,
            Some(SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace { .. })
        ),
        "the factorized correction carries its method: {:?}",
        inference.smoothing_correction_method
    );
    let conditional_errors = inference
        .factorized_standard_errors
        .as_ref()
        .expect("the factorized fit publishes its conditional standard errors");
    assert_eq!(published.factor.nrows(), COLUMNS);
    assert_eq!(published.standard_errors.len(), COLUMNS);

    // The comparison below reads the inference branch alone only if both fits
    // reached the same optimum.
    assert_eq!(
        factorized.log_lambdas.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        dense.log_lambdas.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "both fits must reach the same smoothing parameters: dense {:?}, factorized {:?}",
        dense.log_lambdas,
        factorized.log_lambdas
    );

    let mut worst = 0.0_f64;
    for index in 0..COLUMNS {
        // The same factor from the same assembly: the dense correction's
        // diagonal is written as the factor row's sum of squares.
        let row = published.factor.row(index);
        let factor_variance = row.dot(&row);
        assert_eq!(
            factor_variance.to_bits(),
            dense_correction[[index, index]].to_bits(),
            "coefficient {index}: factorized correction variance {factor_variance:.17e}, dense \
             {:.17e}",
            dense_correction[[index, index]]
        );

        // Each branch's conditional and corrected variances share one solved
        // diagonal `d_i`: `s·d_i` and `s·d_i + ‖b_i‖²`. So the cross-branch gap
        // in the corrected variance is the cross-branch gap in the conditional
        // one, whatever the two solves' rounding. What separates the gaps is
        // rounding in reading them back: each factorized variance is a
        // correctly rounded square root then a square (≤ 1.5ε relative), its
        // corrected sum one more addition (ε/2), and the dense `Vp_ii` one
        // addition (ε/2): at most 4ε of the larger corrected variance. Forming
        // the two gaps and their difference here rounds once each, ε/2 of each
        // gap and of the result.
        let factorized_corrected = published.standard_errors[index].powi(2);
        let factorized_conditional = conditional_errors[index].powi(2);
        let corrected_gap = factorized_corrected - corrected[[index, index]];
        let conditional_gap = factorized_conditional - conditional[[index, index]];
        let scale = factorized_corrected.max(corrected[[index, index]]);
        let residue = (corrected_gap - conditional_gap).abs();
        let rounding = 4.0 * f64::EPSILON * scale
            + f64::EPSILON * (corrected_gap.abs() + conditional_gap.abs());
        worst = worst.max(residue / scale);
        assert!(
            residue <= rounding,
            "coefficient {index}: corrected gap {corrected_gap:.6e} against conditional gap \
             {conditional_gap:.6e} (factorized corrected {factorized_corrected:.17e}, dense \
             {:.17e})",
            corrected[[index, index]]
        );
    }
    eprintln!(
        "[3283 factorized correction] p={COLUMNS} rank={} worst relative residue={worst:.3e}",
        published.factor.ncols()
    );
}
