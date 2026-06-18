use gam::solver::psi_gram_tensor::PsiGramTensor;
use ndarray::{Array1, Array2};
use std::cell::Cell;

fn adversarial_design(psi: f64, n: usize, k: usize) -> Result<Array2<f64>, String> {
    let mut x = Array2::<f64>::zeros((n, k));
    let kappa = psi.exp();
    for i in 0..n {
        let row = i as f64 + 1.0;
        for j in 0..k {
            let col = j as f64 + 1.0;
            let r = 0.03 + row * col / (n as f64 * k as f64) * 4.5;
            x[[i, j]] = if j + 1 == k {
                r * r
            } else {
                let s = kappa * r;
                (1.0 + s) * (-s).exp()
            };
        }
    }
    Ok(x)
}

fn dense_stats(
    psi: f64,
    n: usize,
    k: usize,
    weights: &Array1<f64>,
    z: &Array1<f64>,
) -> (Array2<f64>, Array1<f64>, f64) {
    let design = adversarial_design(psi, n, k).expect("dense design");
    let mut weighted_design = design.clone();
    for (mut row, &w) in weighted_design.outer_iter_mut().zip(weights.iter()) {
        row.mapv_inplace(|v| v * w);
    }
    let mut wz = z.clone();
    let mut zt_w_z = 0.0;
    for ((slot, &w), &zi) in wz.iter_mut().zip(weights.iter()).zip(z.iter()) {
        *slot = w * zi;
        zt_w_z += w * zi * zi;
    }
    (
        design.t().dot(&weighted_design),
        design.t().dot(&wz),
        zt_w_z,
    )
}

#[test]
fn psi_gram_tensor_cache_matches_dense_xtwx_bit_identically_and_is_n_free() {
    let (n, k) = (192usize, 8usize);
    let weights = Array1::from_iter((0..n).map(|i| 0.75 + ((i % 7) as f64) * 0.08));
    let z = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.19).cos() + 0.1));
    let (psi_lo, psi_hi) = (-1.25, 1.15);
    let calls = Cell::new(0usize);

    let tensor = PsiGramTensor::build(
        |psi| {
            calls.set(calls.get() + 1);
            adversarial_design(psi, n, k)
        },
        weights.view(),
        z.view(),
        psi_lo,
        psi_hi,
    )
    .expect("analytic design should certify");
    let build_calls = calls.get();

    for &psi in &[-0.91, -0.17, 0.23, 0.79] {
        assert!(
            tensor.contains(psi),
            "psi sample must be in the certified window"
        );
        let cache = tensor.gaussian_fixed_cache_at(psi);
        assert!(
            cache.row_prediction_is_stale,
            "psi tensor caches must tell Gaussian consumers not to apply stale rows"
        );
        assert_eq!(
            calls.get(),
            build_calls,
            "trial accessor re-entered the n-row design realizer at psi={psi}"
        );
        let (dense_gram, dense_rhs, dense_ztwz) = dense_stats(psi, n, k, &weights, &z);
        assert_eq!(
            cache.centered_weighted_y_sq.to_bits(),
            dense_ztwz.to_bits(),
            "z'Wz changed bits at psi={psi}"
        );
        for ((r, c), &dense) in dense_gram.indexed_iter() {
            let hoisted = cache.xtwx_orig[[r, c]];
            assert_eq!(
                hoisted.to_bits(),
                dense.to_bits(),
                "hoisted X'WX differs from dense path bits at psi={psi}, entry=({r},{c}); hoisted={hoisted:.17e}, dense={dense:.17e}"
            );
        }
        for (j, &dense) in dense_rhs.iter().enumerate() {
            let hoisted = cache.xtwy_orig[j];
            assert_eq!(
                hoisted.to_bits(),
                dense.to_bits(),
                "hoisted X'Wz differs from dense path bits at psi={psi}, entry={j}; hoisted={hoisted:.17e}, dense={dense:.17e}"
            );
        }
    }
}

fn ridge_profile_deviance(gram: &Array2<f64>, rhs: &Array1<f64>, ywy: f64, lambda: f64) -> f64 {
    let k = rhs.len();
    let mut aug = Array2::<f64>::zeros((k, k + 1));
    aug.slice_mut(ndarray::s![.., ..k]).assign(gram);
    for i in 0..k {
        aug[[i, i]] += lambda;
    }
    aug.slice_mut(ndarray::s![.., k]).assign(rhs);
    for col in 0..k {
        let piv = (col..k)
            .max_by(|&p, &q| aug[[p, col]].abs().total_cmp(&aug[[q, col]].abs()))
            .unwrap();
        if piv != col {
            for j in 0..=k {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[piv, j]];
                aug[[piv, j]] = tmp;
            }
        }
        let pivot = aug[[col, col]];
        for row in 0..k {
            if row == col {
                continue;
            }
            let factor = aug[[row, col]] / pivot;
            for j in col..=k {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }
    let beta = Array1::from_iter((0..k).map(|i| aug[[i, k]] / aug[[i, i]]));
    ywy - beta.dot(rhs)
}

#[test]
fn reduced_basis_skip_witness_does_not_certify_bit_identical_stats() {
    let (n, k) = (192usize, 8usize);
    let weights = Array1::from_iter((0..n).map(|i| 0.75 + ((i % 7) as f64) * 0.08));
    let z = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.19).cos() + 0.1));
    let (psi_lo, psi_hi) = (-1.25, 1.15);
    let calls = Cell::new(0usize);

    let tensor = PsiGramTensor::build(
        |psi| {
            calls.set(calls.get() + 1);
            adversarial_design(psi, n, k)
        },
        weights.view(),
        z.view(),
        psi_lo,
        psi_hi,
    )
    .expect("analytic design should certify");
    let build_calls = calls.get();
    let psi_ref = -0.91;
    let psi_trial = 0.79;

    assert!(
        tensor.reduced_basis_equal(psi_ref, psi_trial),
        "full-rank projector witness should accept this pair; if it refuses, this \
         test no longer exercises the production skip gate"
    );
    let cache = tensor.gaussian_fixed_cache_at(psi_trial);
    assert_eq!(
        calls.get(),
        build_calls,
        "trial accessor re-entered the n-row design realizer"
    );
    let (dense_gram, dense_rhs, _) = dense_stats(psi_trial, n, k, &weights, &z);
    let dense_dev =
        ridge_profile_deviance(&dense_gram, &dense_rhs, cache.centered_weighted_y_sq, 0.7);
    let hoisted_dev = ridge_profile_deviance(
        &cache.xtwx_orig,
        &cache.xtwy_orig,
        cache.centered_weighted_y_sq,
        0.7,
    );
    let rel = (dense_dev - hoisted_dev).abs() / dense_dev.abs().max(1e-300);
    assert!(
        rel <= 1e-8,
        "reduced-basis witness accepted psi_ref={psi_ref}, psi_trial={psi_trial}, \
         but hoisted profile objective drifted by rel={rel:.3e}"
    );

    for ((r, c), &dense) in dense_gram.indexed_iter() {
        let hoisted = cache.xtwx_orig[[r, c]];
        assert_eq!(
            hoisted.to_bits(),
            dense.to_bits(),
            "reduced-basis witness accepted psi_ref={psi_ref}, psi_trial={psi_trial}, \
             but hoisted X'WX is not bit-identical at entry=({r},{c}); \
             hoisted={hoisted:.17e}, dense={dense:.17e}"
        );
    }
    for (j, &dense) in dense_rhs.iter().enumerate() {
        let hoisted = cache.xtwy_orig[j];
        assert_eq!(
            hoisted.to_bits(),
            dense.to_bits(),
            "reduced-basis witness accepted psi_ref={psi_ref}, psi_trial={psi_trial}, \
             but hoisted X'Wz is not bit-identical at entry={j}; \
             hoisted={hoisted:.17e}, dense={dense:.17e}"
        );
    }
}
