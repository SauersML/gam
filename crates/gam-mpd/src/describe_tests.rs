#![cfg(test)]
//! Structured descriptions: a generic block costs its rank's reals, a block that rotates one
//! character into another costs two, and a block inside one declared group costs that group.

use super::describe::{Chart, Core, Metric, describe};
use ndarray::{Array2, s};

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

fn metric(d_out: usize, d_in: usize) -> Metric {
    Metric { moment: Array2::eye(d_in), fisher: Array2::eye(d_out), observations: 1e4 }
}

fn decoded(u: &Array2<f64>, v: &Array2<f64>) -> Array2<f64> {
    u.t().dot(v)
}

fn relative(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let e = a - b;
    (&e * &e).sum().sqrt() / (b * b).sum().sqrt()
}

#[test]
fn a_generic_block_costs_its_rank() {
    let (d_out, d_in, k) = (7, 6, 3);
    let w = random(d_out, k, 1).dot(&random(k, d_in, 2));
    let d = describe(&w, k, &metric(d_out, d_in), &[], &[]).expect("a description");
    assert_eq!(d.core, Core::Generic { rank: k });
    assert_eq!(d.reals, k * (d_out + d_in - k));
    // The precision is the one whose bits and error balance at n = 10⁴: a percent or so of the map.
    assert!(relative(&decoded(&d.u, &d.v), &w) < 5e-2, "decoded error {}", relative(&decoded(&d.u, &d.v), &w));
    assert!(d.real_bits >= d.reals as f64);
}

/// A token map `p × d` on labels `0..p`.
fn token_labels(p: usize) -> Array2<usize> {
    Array2::from_shape_fn((p, 1), |(t, _)| t)
}

#[test]
fn a_rotation_between_characters_costs_two_reals() {
    let (p, d) = (13, 9);
    let labels = token_labels(p);
    let reader = Chart::harmonic("reads", random(p, d, 3).view(), labels.view(), p).expect("chart");
    let writer = Chart::harmonic("writes", random(p, d, 4).view(), labels.view(), p).expect("chart");
    let f = 3;
    let at = |chart: &Chart| chart.groups.iter().find(|g| g.mode == vec![f]).expect("frequency").start;
    let (pw, pr) = (writer.basis.slice(s![.., at(&writer)..at(&writer) + 2]).to_owned(), reader.basis.slice(s![.., at(&reader)..at(&reader) + 2]).to_owned());
    let (a, b) = (1.7 * 0.6_f64.cos(), 1.7 * 0.6_f64.sin());
    let core = ndarray::arr2(&[[a, -b], [b, a]]);
    let w = pw.dot(&core).dot(&pr.t());
    let m = metric(d, d);
    let structured = describe(&w, 2, &m, std::slice::from_ref(&writer), std::slice::from_ref(&reader)).expect("a description");
    let generic = describe(&w, 2, &m, &[], &[]).expect("a description");
    assert_eq!(structured.core, Core::Rotation { reflections: vec![false] });
    assert_eq!(structured.reals, 2);
    assert_eq!(structured.writer.1, vec![vec![f]]);
    assert_eq!(structured.reader.1, vec![vec![f]]);
    assert_eq!(generic.reals, 2 * (d + d - 2));
    assert!(structured.total() < 0.5 * generic.total(), "structured {} against generic {}", structured.total(), generic.total());
    assert!(relative(&decoded(&structured.u, &structured.v), &w) < 5e-2);
}

#[test]
fn a_harmonic_chart_on_labelled_rows_reads_one_character() {
    // Rows labelled by two operands; each operand embedded in its own coordinates.
    let (p, d) = (7, 8);
    let rows = p * p;
    let labels = Array2::from_shape_fn((rows, 2), |(r, i)| if i == 0 { r / p } else { r % p });
    let (first, second) = (random(p, d, 5), random(p, d, 11));
    let values = Array2::from_shape_fn((rows, 2 * d), |(r, j)| if j < d { first[[r / p, j]] } else { second[[r % p, j - d]] });
    let chart = Chart::harmonic("reads", values.view(), labels.view(), p).expect("chart");
    assert_eq!(chart.groups.len(), (p * p - 1) / 2 + 1);
    // The direction reading cos(2π·2a/p) has exactly that profile wherever the values resolve it.
    // ... up to the unit mean-square scale (the cosine's own is ½).
    let g = chart.groups.iter().find(|g| g.mode == vec![2, 0]).expect("mode");
    let profile = values.dot(&chart.basis.column(g.start));
    for r in 0..rows {
        let want = std::f64::consts::SQRT_2 * (2.0 * std::f64::consts::PI * (2 * (r / p)) as f64 / p as f64).cos();
        assert!((profile[r] - want).abs() < 1e-8, "row {r}: {} against {want}", profile[r]);
    }
}

#[test]
fn a_block_inside_one_declared_group_costs_that_group() {
    // Writes three heads of four coordinates each; the block writes head 1 only.
    let (heads, width, d_in, k) = (3, 4, 7, 2);
    let d_out = heads * width;
    let groups: Vec<Vec<usize>> = (0..heads).map(|h| (h * width..(h + 1) * width).collect()).collect();
    let chart = Chart::coordinates("heads", d_out, &groups).expect("chart");
    let mut writer = Array2::<f64>::zeros((d_out, k));
    writer.slice_mut(s![width..2 * width, ..]).assign(&random(width, k, 9));
    let w = writer.dot(&random(k, d_in, 10));
    let d = describe(&w, k, &metric(d_out, d_in), std::slice::from_ref(&chart), &[]).expect("a description");
    assert_eq!(d.writer.1, vec![vec![1]]);
    assert_eq!(d.reals, k * (width + d_in - k));
    assert!(relative(&decoded(&d.u, &d.v), &w) < 5e-2);
}
