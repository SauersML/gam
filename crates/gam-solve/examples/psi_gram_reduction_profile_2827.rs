//! Time one production node reduction, excluding design construction.
//!
//! Usage: psi_gram_reduction_profile_2827 ROWS COLUMNS
//! The realizer deliberately refuses its second callback after the first
//! sufficient statistic has been computed. This is a reduction measurement,
//! not a completed tensor certificate or fit.

use gam_solve::psi_gram_tensor::PsiGramTensor;
use ndarray::{Array1, Array2};
use std::time::Instant;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: psi_gram_reduction_profile_2827 ROWS COLUMNS".into());
    }
    let rows = args[1]
        .parse::<usize>()
        .map_err(|error| error.to_string())?;
    let columns = args[2]
        .parse::<usize>()
        .map_err(|error| error.to_string())?;
    if rows == 0 || columns == 0 {
        return Err("ROWS and COLUMNS must be positive".into());
    }
    let mut design = Some(Array2::from_shape_fn((rows, columns), |(i, j)| {
        ((i * 17 + j * 31) % 101) as f64 / 101.0 - 0.5
    }));
    let weights = Array1::from_shape_fn(rows, |i| 0.5 + (i % 7) as f64 / 7.0);
    let response = Array1::from_shape_fn(rows, |i| (i % 13) as f64 / 13.0 - 0.5);
    let mut started = None;
    let mut elapsed = None;
    let outcome = PsiGramTensor::build(
        |_| {
            if let Some(design) = design.take() {
                started = Some(Instant::now());
                Ok(design)
            } else {
                elapsed = Some(started.expect("first node began").elapsed());
                Err("intentional measurement stop after one completed reduction".to_string())
            }
        },
        weights.view(),
        response.view(),
        -1.0,
        1.0,
    );
    let why = outcome
        .err()
        .ok_or("measurement unexpectedly completed a tensor")?;
    if !why.contains("intentional measurement stop") {
        return Err(why);
    }
    println!(
        "rows={rows} columns={columns} completed_node_reductions=1 reduction_seconds={:.9}",
        elapsed
            .ok_or("first reduction did not complete")?
            .as_secs_f64(),
    );
    Ok(())
}
