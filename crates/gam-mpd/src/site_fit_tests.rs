#![cfg(test)]
//! A site's library gated in blocks by its own code.

use super::blocks::Generic;
use super::masked::Library;
use super::site_fit::{Samples, blocks, measure};
use ndarray::{Array1, Array2};

fn noise(seed: usize) -> f64 {
    let mut x = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// A 3 → 2 site on 64 inputs, unit Fisher, and its generic description.
fn site() -> (Array2<f64>, Samples, Generic) {
    let rows = 64;
    let x = Array2::from_shape_fn((rows, 3), |(t, i)| noise(10 + 3 * t + i));
    let w = Array2::from_shape_fn((2, 3), |(i, j)| 2.0 * noise(500 + 3 * i + j));
    let moment = x.t().dot(&x) / rows as f64;
    let samples = Samples {
        reads: x.mapv(|v| v as f32),
        sensitivity: Array1::ones(rows),
        fisher: Array2::eye(2),
        second_moment: moment.clone(),
        gradients: Vec::new(),
    };
    let statistics = vec![super::pieces::Site { w: w.clone(), second_moment: moment, mean: Array1::zeros(3), fisher: Array2::eye(2) }];
    (w, samples, Generic::new(&statistics, 1e4))
}

#[test]
fn a_subcomponent_split_in_two_halves_is_merged_back() {
    let (w, samples, describe) = site();
    // The map's two singular pieces, the first one split into two identical halves that always
    // run together.
    let d = super::dense::svd(w.view(), false).expect("svd");
    let mut u_rows = Vec::new();
    let mut v_rows = Vec::new();
    for j in 0..d.singular_values.len() {
        let root = d.singular_values[j].sqrt();
        let (u, v) = (d.u.column(j).to_owned() * root, d.vt.row(j).to_owned() * root);
        if j == 0 {
            u_rows.extend([u.clone(), u]);
            v_rows.extend([&v * 0.5, &v * 0.5]);
        } else {
            u_rows.push(u);
            v_rows.push(v);
        }
    }
    let stack = |rows: &[Array1<f64>]| Array2::from_shape_fn((rows.len(), rows[0].len()), |(i, j)| rows[i][j]);
    let library = Library { v: stack(&v_rows), u: stack(&u_rows), mean: Array1::zeros(3) };
    let (rank_one, _) = measure(0, &w, &samples, &describe, 1e4, &library).expect("measures");
    let (blocked, round) = blocks(0, &w, &samples, &describe, 1e4, &library, &[1, 1, 1]).expect("blocks");
    // The halves always run together: they end in one block (alone, or with the rest of the map).
    assert!(blocked.ranks.len() < 3 && blocked.ranks[0] >= 2, "the halves stayed apart: {:?}", blocked.ranks);
    assert!(round.code < rank_one.code, "{} against {}", round.code, rank_one.code);
    let map = blocked.library.u.t().dot(&blocked.library.v);
    let error = (&map - &w).iter().fold(0.0_f64, |m, e| m.max(e.abs()));
    assert!(error <= 1e-12, "merging moved the map by {error}");
}
