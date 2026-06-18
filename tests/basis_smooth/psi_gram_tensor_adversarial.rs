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
