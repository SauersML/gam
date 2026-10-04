#![cfg(test)]
//! Combined products are the plain ones: many threads' products with the same large matrices,
//! on either side, some threads asking more often than others and some asking none.

use super::combine::{map, product};
use gam_linalg::faer_ndarray::fast_ab;
use ndarray::Array2;

fn matrix(rows: usize, cols: usize, salt: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| (((i * 31 + j * 17 + salt * 7) % 97) as f64 - 48.0) / 13.0)
}

#[test]
fn combined_products_are_the_plain_products() {
    let large = [matrix(300, 300, 1), matrix(300, 300, 2), matrix(400, 300, 3)];
    let transposed = large[2].t();
    let items: Vec<usize> = (0..97).collect();
    let results = map(&items, |&i| {
        // A varying number of products per item, of every kind.
        let mut out = Vec::new();
        for step in 0..(i % 5) {
            let thin = matrix(300, 1 + (i + step) % 3, i + step);
            out.push((product(&large[step % 3], &thin), fast_ab(&large[step % 3], &thin)));
            let rows = matrix(1 + step % 2, 300, i * 3 + step);
            out.push((product(&rows, &large[(step + 1) % 2]), fast_ab(&rows, &large[(step + 1) % 2])));
            let wide = matrix(300, 2, i);
            out.push((product(&transposed, &matrix(400, 2, step)), fast_ab(&transposed, &matrix(400, 2, step))));
            // Below the size that is combined.
            out.push((product(&wide.t(), &matrix(300, 4, 9)), fast_ab(&wide.t(), &matrix(300, 4, 9))));
        }
        out
    });
    assert_eq!(results.len(), items.len());
    for (i, pairs) in results.iter().enumerate() {
        assert_eq!(pairs.len(), 4 * (i % 5));
        for (combined, plain) in pairs {
            assert_eq!(combined.dim(), plain.dim());
            let worst = combined.iter().zip(plain.iter()).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(worst <= 1e-9, "item {i}: off by {worst}");
        }
    }
}
