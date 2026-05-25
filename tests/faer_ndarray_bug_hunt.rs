use faer::Side;
use gam::faer_ndarray::{FaerEigh, fast_ab, fast_ata, fast_atv, fast_av, fast_xt_diag_y};
use ndarray::{Array1, Array2, Axis, s};

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim(), "shape mismatch in max_abs_diff");
    (a - b).iter().fold(0.0_f64, |m, &v| m.max(v.abs()))
}

fn max_abs_diff_vec(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch in max_abs_diff_vec");
    (a - b).iter().fold(0.0_f64, |m, &v| m.max(v.abs()))
}

fn synth_matrix(n: usize, p: usize, phase: f64) -> Array2<f64> {
    Array2::from_shape_fn((n, p), |(i, j)| {
        let x = i as f64;
        let y = j as f64;
        (0.13 * x + 0.27 * y + phase).sin() + (0.07 * x - 0.11 * y + phase).cos()
    })
}

#[test]
fn bug_fast_ab_matches_ndarray_dot_random_like_inputs() {
    let a = synth_matrix(137, 53, 0.3);
    let b = synth_matrix(53, 41, 0.9);
    let got = fast_ab(&a, &b);
    let expected = a.dot(&b);
    let err = max_abs_diff(&got, &expected);
    assert!(
        err <= 1e-9,
        "fast_ab should match ndarray dot within 1e-9, max error was {err:e}"
    );
}

#[test]
fn bug_fast_ata_matches_transpose_product_random_like_input() {
    let a = synth_matrix(211, 47, 1.1);
    let got = fast_ata(&a);
    let expected = a.t().dot(&a);
    let err = max_abs_diff(&got, &expected);
    assert!(
        err <= 1e-9,
        "fast_ata should match A^T A within 1e-9, max error was {err:e}"
    );
}

#[test]
fn bug_fast_av_matches_matrix_vector_product() {
    let a = synth_matrix(149, 39, 0.5);
    let v = synth_matrix(39, 1, 1.7).index_axis(Axis(1), 0).to_owned();
    let got = fast_av(&a, &v);
    let expected = a.dot(&v);
    let err = max_abs_diff_vec(&got, &expected);
    assert!(
        err <= 1e-9,
        "fast_av should match A v within 1e-9, max error was {err:e}"
    );
}

#[test]
fn bug_fast_atv_matches_transpose_vector_product() {
    let a = synth_matrix(149, 39, 0.2);
    let v = synth_matrix(149, 1, 1.4).index_axis(Axis(1), 0).to_owned();
    let got = fast_atv(&a, &v);
    let expected = a.t().dot(&v);
    let err = max_abs_diff_vec(&got, &expected);
    assert!(
        err <= 1e-9,
        "fast_atv should match A^T v within 1e-9, max error was {err:e}"
    );
}

#[test]
fn bug_fast_xt_diag_y_matches_naive_weighted_crossprod() {
    let x = synth_matrix(333, 21, 0.4);
    let y = synth_matrix(333, 17, 1.3);
    let w = Array1::from_shape_fn(333, |i| (i as f64 * 0.17).sin());
    let wy = Array2::from_shape_fn((333, 17), |(i, j)| w[i] * y[[i, j]]);
    let expected = x.t().dot(&wy);
    let got = fast_xt_diag_y(&x, &w, &y);
    let err = max_abs_diff(&got, &expected);
    assert!(
        err <= 1e-9,
        "fast_xt_diag_y should match X^T diag(w) Y within 1e-9, max error was {err:e}"
    );
}

#[test]
fn bug_chunked_gram_matches_sum_of_per_chunk_grams() {
    let x = synth_matrix(4097, 19, 0.8);
    let y = synth_matrix(4097, 13, 1.8);
    let w = Array1::from_shape_fn(4097, |i| (i as f64 * 0.03).cos());

    let full = fast_xt_diag_y(&x, &w, &y);

    let chunk_rows = 257;
    let mut sum = Array2::<f64>::zeros((x.ncols(), y.ncols()));
    for start in (0..x.nrows()).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(x.nrows());
        let xc = x.slice(s![start..end, ..]).to_owned();
        let yc = y.slice(s![start..end, ..]).to_owned();
        let wc = w.slice(s![start..end]).to_owned();
        sum += &fast_xt_diag_y(&xc, &wc, &yc);
    }

    let err = max_abs_diff(&full, &sum);
    assert!(
        err <= 1e-9,
        "chunked Gram aggregation should equal dense one-block result within 1e-9, max error was {err:e}"
    );
}

#[test]
fn bug_faer_eigh_spd_positive_and_reconstructs_matrix() {
    let m = synth_matrix(25, 25, 0.6);
    let mut a = m.t().dot(&m);
    for i in 0..25 {
        a[[i, i]] += 1e-2;
    }

    let (evals, evecs) = a
        .eigh(Side::Lower)
        .expect("eigh should succeed on SPD input");
    assert!(
        evals.iter().all(|&v| v > 0.0),
        "all eigenvalues should be positive for SPD matrix"
    );

    let d = Array2::from_diag(&evals);
    let recon = evecs.dot(&d).dot(&evecs.t());
    let err = max_abs_diff(&recon, &a);
    assert!(
        err <= 1e-9,
        "U Λ U^T should reconstruct SPD input within 1e-9, max error was {err:e}"
    );
}
