//! Focused production-path proof for three-class posterior rule reuse.
//!
//! A prediction design commonly contains many rows that request the same
//! Gauss-Hermite orders.  The rules depend only on those orders, so the batched
//! API owns one rule ladder for the full design evaluation.  This example
//! checks repeated batched rows against independently integrated single rows
//! through the public API and prints wall-clock measurements for both routes.

use gam_models::multinomial_posterior::{
    MultinomialPosteriorIntegrationControl, integrate_multinomial_design_moments,
};
use ndarray::Array2;
use std::error::Error;
use std::time::Instant;

const ROWS: usize = 16;

fn mismatch(message: String) -> Box<dyn Error> {
    Box::new(std::io::Error::other(message))
}

fn main() -> Result<(), Box<dyn Error>> {
    // P=1 and M=2 make every all-ones design row induce the same full-rank
    // two-logit Gaussian.  The covariance is wide enough to require a genuine
    // conditioned rule ladder rather than the point-mass or rank-one path.
    let coefficients = Array2::from_shape_vec((1, 2), vec![1.1, -0.6])
        .map_err(|error| mismatch(error.to_string()))?;
    let coefficient_covariance = Array2::from_shape_vec((2, 2), vec![3.0, 0.7, 0.7, 1.8])
        .map_err(|error| mismatch(error.to_string()))?;
    let design = Array2::<f64>::ones((ROWS, 1));
    let control = MultinomialPosteriorIntegrationControl {
        absolute_tolerance: 2.0e-9,
        relative_tolerance: 2.0e-9,
        minimum_sparse_level: 2,
        maximum_sparse_level: 16,
        maximum_function_evaluations: 2_000_000,
    };

    let batched_start = Instant::now();
    let batched = integrate_multinomial_design_moments(
        coefficients.view(),
        coefficient_covariance.view(),
        design.view(),
        &control,
    )?;
    let batched_wall = batched_start.elapsed();

    for row in 1..ROWS {
        for class in 0..3 {
            if batched.class_mean[[row, class]].to_bits()
                != batched.class_mean[[0, class]].to_bits()
            {
                return Err(mismatch(format!(
                    "batched class mean differs at row {row}, class {class}"
                )));
            }
            if batched.class_standard_deviation[[row, class]].to_bits()
                != batched.class_standard_deviation[[0, class]].to_bits()
            {
                return Err(mismatch(format!(
                    "batched class standard deviation differs at row {row}, class {class}"
                )));
            }
        }
    }

    let single_row = Array2::<f64>::ones((1, 1));
    let independent_start = Instant::now();
    for row in 0..ROWS {
        // A one-row design owns its own rule ladder, so each of these rows is
        // integrated independently of every other.
        let independent = integrate_multinomial_design_moments(
            coefficients.view(),
            coefficient_covariance.view(),
            single_row.view(),
            &control,
        )?;
        for class in 0..3 {
            let mean_difference =
                (independent.class_mean[[0, class]] - batched.class_mean[[row, class]]).abs();
            let deviation_difference = (independent.class_standard_deviation[[0, class]]
                - batched.class_standard_deviation[[row, class]])
            .abs();
            if mean_difference > 2.0e-14 || deviation_difference > 2.0e-14 {
                return Err(mismatch(format!(
                    "independent row {row}, class {class} disagrees with the batched posterior: \
                     mean difference {mean_difference:.6e}, standard-deviation difference \
                     {deviation_difference:.6e}"
                )));
            }
        }
    }
    let independent_wall = independent_start.elapsed();

    println!(
        "POSTERIOR_RULE_REUSE_2612 rows={ROWS} batched_wall_ms={:.3} independent_wall_ms={:.3}",
        batched_wall.as_secs_f64() * 1.0e3,
        independent_wall.as_secs_f64() * 1.0e3,
    );
    Ok(())
}
