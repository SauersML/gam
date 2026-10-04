#![cfg(test)]
//! The attention rules nominate every head that realizes them, whatever its gain and whatever
//! invertible change of its content coordinates, and bind each with its own scale.

use super::rules::{content_rows, copy_alignment, copy_prediction, copy_scale, match_alignment, match_prediction, match_reading, match_scale};
use ndarray::{Array1, Array2, Axis};

fn noise(seed: usize) -> f64 {
    let mut x = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

fn random(rows: usize, cols: usize, salt: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| noise(salt + 1000 * i + j))
}

fn gain(d: usize, salt: usize) -> Array1<f64> {
    Array1::from_shape_fn(d, |i| 1.0 + 0.5 * noise(salt + i))
}

#[test]
fn heads_that_match_at_different_gains_and_coordinates_bind_with_their_own_scales() {
    let (d, width, first) = (24, 8, 2);
    let rows = content_rows(width, first);
    let (g, gs) = (gain(d, 1), gain(d, 2));
    let (source_output, source_value) = (random(d, width, 3), random(width, d, 4));
    let key = random(width, d, 5);
    let reading = match_reading(&key, &source_output, &source_value, &g, &gs, &rows);
    let prediction = match_prediction(&reading, &g, None).expect("prediction");
    // Head one: the rule at scale 0.7 on its content rows, anything elsewhere.
    let mut query = random(width, d, 6);
    for (i, r) in rows.iter().enumerate() {
        query.row_mut(*r).assign(&(&prediction.row(i) * 0.7));
    }
    // Head two: its content coordinates changed by an invertible S (keys S K, queries S⁻ᵀ Q), at
    // scale 2.5 instead.
    let s = &random(rows.len(), rows.len(), 7) + &(Array2::<f64>::eye(rows.len()) * 3.0);
    let s_inverse_t = gam_linalg::decompose::solve(s.view(), Array2::<f64>::eye(rows.len()).view()).expect("invertible").reversed_axes();
    let mut key_two = key.clone();
    let mut query_two = random(width, d, 8);
    let (key_rows, query_rows) = (key.select(Axis(0), &rows), query.select(Axis(0), &rows));
    let (moved_keys, moved_queries) = (s.dot(&key_rows), s_inverse_t.dot(&query_rows) * (2.5 / 0.7));
    for (i, r) in rows.iter().enumerate() {
        key_two.row_mut(*r).assign(&moved_keys.row(i));
        query_two.row_mut(*r).assign(&moved_queries.row(i));
    }
    let reading_two = match_reading(&key_two, &source_output, &source_value, &g, &gs, &rows);
    let prediction_two = match_prediction(&reading_two, &g, None).expect("prediction");
    for (q, r, p, scale) in [(&query, &reading, &prediction, 0.7), (&query_two, &reading_two, &prediction_two, 2.5)] {
        let alignment = match_alignment(q, r, &g, &rows).expect("alignment");
        assert!(alignment > 0.999, "alignment {alignment}");
        let bound = match_scale(q, r, &g, &rows, p);
        assert!((bound - scale).abs() < 1e-6 * scale, "scale {bound} against {scale}");
    }
    // An unrelated head is not nominated.
    let alignment = match_alignment(&random(width, d, 9), &reading, &g, &rows).expect("alignment");
    assert!(alignment.abs() < 0.5, "unrelated alignment {alignment}");
}

#[test]
fn copying_and_anti_copying_heads_bind_the_copy_body_with_their_signs() {
    let (d, width) = (24, 8);
    let (g, gf) = (gain(d, 11), gain(d, 12));
    let value = random(width, d, 13);
    let prediction = copy_prediction(&value, &g, &gf).expect("prediction");
    for scale in [1.3, -0.4] {
        let output = &prediction * scale;
        let alignment = copy_alignment(&output, &value, &prediction);
        assert!((alignment - scale.signum()).abs() < 1e-9, "alignment {alignment}");
        assert!((copy_scale(&output, &value, &prediction) - scale).abs() < 1e-9);
    }
    // Writing what was read: through the gains, the value head's reads come back out.
    let written = prediction.dot(&value);
    let reads = value.t().dot(&gam_linalg::decompose::solve(value.dot(&value.t()).view(), value.view()).expect("solve"));
    let through = &reads * &(&g / &gf).insert_axis(Axis(1));
    assert!((&written - &through).mapv(f64::abs).iter().fold(0.0_f64, |m, x| m.max(*x)) < 1e-8);
}
