use gam_solve::estimate::reml::reml_outer_engine::compute_block_penalty_logdet_derivs_with_prior_factors;
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
fn fixed_penalty_geometry_preserves_root_scale_modes_and_exact_rho_derivatives() {
    // Complementary rank-one ranges with a 1e-32 lambda ratio are the regime
    // where an eigendecomposition of the assembled weighted penalty loses the
    // small but structurally real mode. The fixed range plus scaled-root QR
    // must retain both coercivity contributions.
    let components = vec![
        arr2(&[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        arr2(&[[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]]),
    ];
    let rho = arr1(&[0.25, 0.25 - 32.0 * 10.0_f64.ln()]);
    let blocks: Vec<&[Array2<f64>]> = vec![components.as_slice()];
    let result = compute_block_penalty_logdet_derivs(&[rho.clone()], &blocks, 0.0)
        .expect("fixed structural range remains factorable at an extreme lambda ratio");

    let expected_value = 2.0_f64.ln() + rho[0] + 3.0_f64.ln() + rho[1];
    assert_close(result.value, expected_value, 2.0e-12, "pseudo-logdet");
    assert_close(result.first[0], 1.0, 1.0e-12, "first derivative 0");
    assert_close(result.first[1], 1.0, 1.0e-12, "first derivative 1");
    let second = result.second.expect("exact Hessian is populated");
    assert_close(second[[0, 0]], 0.0, 1.0e-12, "second derivative 00");
    assert_close(second[[0, 1]], 0.0, 1.0e-12, "second derivative 01");
    assert_close(second[[1, 0]], 0.0, 1.0e-12, "second derivative 10");
    assert_close(second[[1, 1]], 0.0, 1.0e-12, "second derivative 11");
}

#[test]
fn fixed_penalty_geometry_cache_is_exact_and_prior_factor_multiplicity_is_retained() {
    let rho = arr1(&[0.4, -0.2]);
    let first_layout = vec![
        arr2(&[[1.0, 0.0], [0.0, 0.0]]),
        arr2(&[[0.0, 0.0], [0.0, 1.0]]),
    ];
    let second_layout = vec![
        arr2(&[[4.0, 0.0], [0.0, 0.0]]),
        arr2(&[[0.0, 0.0], [0.0, 1.0]]),
    ];
    let first_blocks: Vec<&[Array2<f64>]> = vec![first_layout.as_slice()];
    let second_blocks: Vec<&[Array2<f64>]> = vec![second_layout.as_slice()];
    let first = compute_block_penalty_logdet_derivs(&[rho.clone()], &first_blocks, 0.0)
        .expect("first exact layout");
    let second = compute_block_penalty_logdet_derivs(&[rho.clone()], &second_blocks, 0.0)
        .expect("same-shape changed layout");
    assert_close(
        second.value - first.value,
        4.0_f64.ln(),
        1.0e-12,
        "content-addressed geometry invalidation",
    );

    let overlapping_factors = vec![arr2(&[[1.0]]), arr2(&[[1.0]])];
    let factor_blocks: Vec<&[Array2<f64>]> = vec![overlapping_factors.as_slice()];
    let masks = vec![vec![true, true]];
    let factored = compute_block_penalty_logdet_derivs_with_prior_factors(
        &[rho.clone()],
        &factor_blocks,
        Some(&masks),
        0.0,
    )
    .expect("independent prior factors");
    assert_close(
        factored.value,
        rho.sum(),
        1.0e-12,
        "factor normalizer multiplicity",
    );
    assert_close(factored.first[0], 1.0, 1.0e-12, "factor derivative 0");
    assert_close(factored.first[1], 1.0, 1.0e-12, "factor derivative 1");
    assert!(
        factored
            .second
            .expect("factor Hessian")
            .iter()
            .all(|value| *value == 0.0),
        "independent rank-one factor normalizers are exactly affine in rho"
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
