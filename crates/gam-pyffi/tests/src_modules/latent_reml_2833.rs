use super::*;

fn fixture(n: usize, dim: usize, k: usize, outputs: usize) -> (Array1<f64>, LatentOuterProblem) {
    let t = Array1::from_shape_fn(n * dim, |i| {
        0.5 + 0.38 * ((i + 1) as f64 * 1.731).sin()
    });
    let centers = Array2::from_shape_fn((k, dim), |(i, a)| {
        0.5 + 0.44 * ((i * dim + a + 1) as f64 * 2.317).sin()
    });
    let y = Array2::from_shape_fn((n, outputs), |(i, a)| {
        (3.0 * t[i * dim]).sin() + 0.3 * ((i + 1) as f64 * (a + 2) as f64 * 2.13).sin()
    });
    let problem = LatentOuterProblem {
        y,
        centers,
        penalty: Array2::eye(k),
        weights: Some(Array1::from_shape_fn(n, |i| 1.0 + 0.4 * (i as f64).cos())),
        aux_u: None,
        dim_selection: None,
        family: AuxPriorFamily::Ridge,
        aux_strength: None,
        init_lambda: None,
        n_obs: n,
        latent_dim: dim,
        m: 2,
        basis_kind: "duchon".to_string(),
        tensor_knots: None,
        tensor_knot_offsets: None,
        tensor_degrees: None,
        periodic: None,
    };
    (t, problem)
}

#[test]
fn latent_reml_2833_design_jets_and_fixed_coefficient_frame() {
    for dim in [1, 2, 4] {
        let (t, problem) = fixture(16, dim, 9, 1);
        let evaluate = |point: ArrayView1<'_, f64>| {
            build_latent_forward_design(
                "duchon", point, 16, dim, problem.centers.view(), 2,
                None, None, None, None,
            ).unwrap()
        };
        let (design, _, jet) = evaluate(t.view());
        let h = 1e-6;
        for coordinate in 0..t.len() {
            let mut plus = t.clone();
            let mut minus = t.clone();
            plus[coordinate] += h;
            minus[coordinate] -= h;
            let (xp, _, _) = evaluate(plus.view());
            let (xm, _, _) = evaluate(minus.view());
            for row in 0..design.nrows() {
                for col in 0..design.ncols() {
                    let expected = if row == coordinate / dim {
                        jet[[row, col, coordinate % dim]]
                    } else {
                        0.0
                    };
                    let observed = (xp[[row, col]] - xm[[row, col]]) / (2.0 * h);
                    assert!(
                        (observed - expected).abs() < 2e-6 * (1.0 + expected.abs()),
                        "dim={dim}, coordinate={coordinate}, row={row}, col={col}: \
                         derivative={expected}, finite difference={observed}"
                    );
                }
            }
        }
    }
}

#[test]
fn latent_reml_2833_reported_score_gradient_is_its_derivative() {
    for (n, dim, k, outputs) in [(12, 1, 5, 1), (20, 1, 8, 2), (24, 2, 9, 1)] {
        let (t, problem) = fixture(n, dim, k, outputs);
        let (value, gradient) = problem.try_value_and_grad(t.view(), true).unwrap();
        let gradient = gradient.unwrap();
        let h = 1e-5;
        let mut squared_error = 0.0;
        let mut squared_fd = 0.0;
        for coordinate in 0..t.len() {
            let mut plus = t.clone();
            let mut minus = t.clone();
            plus[coordinate] += h;
            minus[coordinate] -= h;
            let vp = problem.try_value_and_grad(plus.view(), false).unwrap().0;
            let vm = problem.try_value_and_grad(minus.view(), false).unwrap().0;
            let central = (vp - vm) / (2.0 * h);
            squared_error += (central - gradient[coordinate]).powi(2);
            squared_fd += central * central;
        }
        assert!(
            squared_error.sqrt() <= 1e-4 * (1.0 + squared_fd.sqrt()),
            "n={n}, dim={dim}, outputs={outputs}: error={}, fd norm={}",
            squared_error.sqrt(), squared_fd.sqrt()
        );
        let norm = gradient.dot(&gradient).sqrt();
        assert!(norm > 0.0);
        let next = &t - &gradient.mapv(|g| 1e-6 * g / norm);
        let next_value = problem.try_value_and_grad(next.view(), false).unwrap().0;
        assert!(next_value < value - 0.5e-6 * norm);
    }
}
