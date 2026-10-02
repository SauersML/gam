#![cfg(test)]
//! Per-input pieces on inputs that each lie along one of a few directions: the fitted library is
//! the map exactly, and each input lists about one piece.

use super::pieces::{Site, fit, sets};
use ndarray::{Array1, Array2};

fn noise(seed: usize) -> f64 {
    let mut x = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

const D: usize = 8;
const DIRECTIONS: usize = 4;

/// `count` inputs, input `t` along direction `t mod 4` with a scale of size in `[1, 2]` and either
/// sign, so every direction is a line through the mean.
fn inputs(count: usize, salt: usize) -> Array2<f64> {
    let directions = Array2::from_shape_fn((DIRECTIONS, D), |(k, i)| noise(10 + 7 * k + i));
    Array2::from_shape_fn((count, D), |(t, i)| {
        let size = 1.5 + 0.5 * noise(salt + 3 * t);
        let sign = if (t / DIRECTIONS) % 2 == 0 { 1.0 } else { -1.0 };
        sign * size * directions[[t % DIRECTIONS, i]]
    })
}

#[test]
fn inputs_along_few_directions_each_list_about_one_piece() {
    let x = inputs(400, 1000);
    let w = Array2::from_shape_fn((D, D), |(i, j)| noise(500 + D * i + j));
    let mean = x.mean_axis(ndarray::Axis(0)).expect("rows");
    let second_moment = x.t().dot(&x) / x.nrows() as f64;
    let fisher = Array2::eye(D);
    let site = Site { w: w.clone(), second_moment, mean: Array1::from(mean), fisher };
    let (library, whitened) = fit(&site, &x, 1.0e4).expect("fits");
    assert!(library.exactness(&w) < 1e-9, "{}", library.exactness(&w));
    let fresh = inputs(200, 9000);
    let chosen = sets(&library, &whitened, &fresh, 1.0e4);
    let mean_active = chosen.iter().map(Vec::len).sum::<usize>() as f64 / chosen.len() as f64;
    assert!(mean_active <= 1.5, "{mean_active} pieces per input; history {:?}", library.history);
}
