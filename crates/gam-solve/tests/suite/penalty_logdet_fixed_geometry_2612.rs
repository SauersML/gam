use gam_solve::pirls::dense_block_xtwx;
use ndarray::{Array2, Array3, arr1, arr2, s};

fn assert_close(actual: f64, expected: f64, tolerance: f64, context: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{context}: actual={actual:.16e}, expected={expected:.16e}, error={:.3e}",
        (actual - expected).abs()
    );
}

#[test]
fn block_fisher_gram_decomposition_matches_the_symmetrized_tensor_definition() {
    // Strided design input pins the API-boundary layout normalization while a
    // deliberately tiny Fisher asymmetry pins the canonical average performed
    // before the tuned weighted cross-products.
    let design_storage = arr2(&[
        [1.0, 9.0, -0.5, 8.0, 0.25, 7.0],
        [0.3, 6.0, 1.2, 5.0, -0.7, 4.0],
        [-0.8, 3.0, 0.4, 2.0, 1.5, 1.0],
        [1.1, 0.0, -0.2, -1.0, 0.6, -2.0],
    ]);
    let design = design_storage.slice(s![.., ..;2]);
    let mut fisher = Array3::<f64>::zeros((4, 2, 2));
    for row in 0..4 {
        fisher[[row, 0, 0]] = 0.8 + 0.1 * row as f64;
        fisher[[row, 1, 1]] = 1.3 - 0.05 * row as f64;
        fisher[[row, 0, 1]] = -0.2 + 0.03 * row as f64;
        fisher[[row, 1, 0]] = fisher[[row, 0, 1]] + 1.0e-12;
    }
    let row_weights = arr1(&[1.0, 0.5, 1.7, 0.9]);
    let actual = dense_block_xtwx(design, fisher.view(), Some(row_weights.view()))
        .expect("finite block Fisher Gram");

    let (n, p, outputs) = (design.nrows(), design.ncols(), 2);
    let mut expected = Array2::<f64>::zeros((outputs * p, outputs * p));
    for a in 0..outputs {
        for b in 0..outputs {
            for i in 0..p {
                for j in 0..p {
                    let mut value = 0.0;
                    for row in 0..n {
                        let symmetric_fisher = 0.5 * (fisher[[row, a, b]] + fisher[[row, b, a]]);
                        value += row_weights[row]
                            * design[[row, i]]
                            * symmetric_fisher
                            * design[[row, j]];
                    }
                    expected[[a * p + i, b * p + j]] = value;
                }
            }
        }
    }
    for ((row, col), &value) in actual.indexed_iter() {
        assert_close(value, expected[[row, col]], 2.0e-12, "block Fisher Gram");
    }
}
