#![cfg(test)]
//! A site's library gated in blocks by its own code.

use super::blocks::Generic;
use super::masked::Library;
use super::site_fit::{Samples, ard, blocks, code_of, measure};
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

#[test]
fn given_sets_are_priced_as_the_selection_prices_its_own() {
    let (w, samples, describe) = site();
    let d = super::dense::svd(w.view(), false).expect("svd");
    let roots: Vec<f64> = d.singular_values.iter().map(|s| s.sqrt()).collect();
    let k = roots.len();
    let library = Library {
        v: Array2::from_shape_fn((k, 3), |(j, i)| d.vt[[j, i]] * roots[j]),
        u: Array2::from_shape_fn((k, 2), |(j, i)| d.u[[i, j]] * roots[j]),
        mean: Array1::zeros(3),
    };
    let (selected, sets) = measure(0, &w, &samples, &describe, 1e4, &library).expect("measures");
    let again = code_of(0, &w, &samples, &describe, 1e4, &library, &sets).expect("prices");
    assert!((again.code - selected.code).abs() <= 1e-9 * selected.code.abs(), "{} against {}", again.code, selected.code);
    // Every subcomponent off is a valid explanation too, never cheaper than the selection.
    let none = code_of(0, &w, &samples, &describe, 1e4, &library, &vec![Vec::new(); sets.len()]).expect("prices");
    assert!(none.code >= selected.code, "{} against {}", none.code, selected.code);
    assert_eq!(none.l0, 0.0);
}

#[test]
fn groups_fitted_by_evidence_partition_their_columns_and_code_no_worse_than_rank_one() {
    let (w, samples, describe) = site();
    let d = super::dense::svd(w.view(), false).expect("svd");
    let roots = d.singular_values.mapv(f64::sqrt);
    let u = (&d.u * &roots).t().to_owned();
    let v = (d.vt.t().to_owned() * &roots).t().to_owned();
    let library = Library { v, u, mean: Array1::zeros(3) };
    let (rank_one, _) = measure(0, &w, &samples, &describe, 1e4, &library).expect("measures");
    let (fitted, round) = ard(0, &w, &samples, &describe, 1e4, (&library, &[1, 1]), 20).expect("fits");
    assert_eq!(fitted.ranks.iter().sum::<usize>(), fitted.library.v.nrows());
    assert!(!fitted.ranks.is_empty() && fitted.ranks.iter().all(|r| *r >= 1));
    assert!(round.code <= rank_one.code + 1.0, "{} against {}", round.code, rank_one.code);
}
