#![cfg(test)]
//! The Fisher-whitened singular pieces of a map are the map exactly, and are orthogonal in the
//! whitened metric.

use super::pieces::{Narrow, Site, fisher_svd, fisher_svd_narrow};
use ndarray::{Array1, Array2, Axis};

fn noise(seed: usize) -> f64 {
    let mut x = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

const D_IN: usize = 8;
const D_OUT: usize = 6;

#[test]
fn the_fisher_svd_library_is_the_map_and_whitened_orthogonal() {
    let x = Array2::from_shape_fn((200, D_IN), |(t, i)| noise(10 + D_IN * t + i) + 0.3);
    let w = Array2::from_shape_fn((D_OUT, D_IN), |(i, j)| noise(500 + D_IN * i + j));
    let g = Array2::from_shape_fn((300, D_OUT), |(t, i)| noise(9000 + D_OUT * t + i));
    let mean: Array1<f64> = x.mean_axis(Axis(0)).expect("rows");
    let site = Site { w: w.clone(), second_moment: x.t().dot(&x) / x.nrows() as f64, mean: mean.clone(), fisher: g.t().dot(&g) / g.nrows() as f64 };
    let library = fisher_svd(&site).expect("library");
    assert!(library.exactness(&w) < 1e-9, "{}", library.exactness(&w));
    // Pieces' contributions are orthogonal in the Fisher metric over the centred inputs: the
    // cost of dropping a set is the sum of the pieces' own costs.
    let a = (&x - &mean).dot(&library.v);
    let fu = library.u.dot(&site.fisher).dot(&library.u.t());
    let gram = a.t().dot(&a) / x.nrows() as f64 * &fu;
    let largest = gram.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for i in 0..gram.nrows() {
        for j in 0..gram.ncols() {
            if i != j {
                assert!(gram[[i, j]].abs() <= 1e-8 * largest, "({i}, {j}): {} against {largest}", gram[[i, j]]);
            }
        }
    }
}

/// The narrow-side statistics give the same pieces as the full ones, on either side.
#[test]
fn the_narrow_fisher_svd_is_the_fisher_svd() {
    for (d_in, d_out) in [(D_IN, D_OUT), (D_OUT, D_IN)] {
        let x = Array2::from_shape_fn((200, d_in), |(t, i)| noise(10 + d_in * t + i) + 0.3);
        let w = Array2::from_shape_fn((d_out, d_in), |(i, j)| noise(500 + d_in * i + j));
        let g = Array2::from_shape_fn((300, d_out), |(t, i)| noise(9000 + d_out * t + i));
        let mean: Array1<f64> = x.mean_axis(Axis(0)).expect("rows");
        let centred = &x - &mean;
        let covariance = centred.t().dot(&centred) / x.nrows() as f64;
        let fisher = g.t().dot(&g) / g.nrows() as f64;
        let site = Site { w: w.clone(), second_moment: x.t().dot(&x) / x.nrows() as f64, mean, fisher: fisher.clone() };
        let full = fisher_svd(&site).expect("library");
        let narrow = if d_in <= d_out {
            Narrow::Reads { covariance: covariance.clone(), pulled_fisher: w.t().dot(&fisher).dot(&w) }
        } else {
            Narrow::Writes { fisher: fisher.clone(), written_covariance: w.dot(&covariance).dot(&w.t()) }
        };
        let (library, weights) = fisher_svd_narrow(&w, &narrow).expect("narrow library");
        assert!(library.exactness(&w) < 1e-9, "{}", library.exactness(&w));
        // Each piece's weight is its own u B u, and each piece is a full piece up to sign.
        for c in 0..weights.len().min(full.u.nrows()) {
            let own = library.u.row(c).dot(&fisher.dot(&library.u.row(c)));
            assert!((own - weights[c]).abs() <= 1e-8 * weights[0], "piece {c}: {own} against {}", weights[c]);
            let matched = (0..full.u.nrows()).any(|k| {
                let same = |sign: f64| {
                    (&library.u.row(c) - &(&full.u.row(k) * sign)).iter().all(|e| e.abs() <= 1e-6)
                        && (&library.v.column(c) - &(&full.v.column(k) * sign)).iter().all(|e| e.abs() <= 1e-6)
                };
                same(1.0) || same(-1.0)
            });
            assert!(matched, "piece {c} of the narrow library is not a Fisher-SVD piece");
        }
    }
}
