use gam::estimate::{ExternalOptimOptions, evaluate_externalgradients};
use gam::smooth::BlockwisePenalty;
use gam::types::LikelihoodFamily;
use ndarray::{Array1, Array2};

fn make_problem(
    signal_scale: f64,
) -> (
    Array2<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Vec<BlockwisePenalty>,
) {
    let n = 400usize;
    let p = 6usize;
    let mut x = Array2::<f64>::zeros((n, p));
    let mut y = Array1::<f64>::zeros(n);
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);

    for i in 0..n {
        let t = (i as f64) / (n as f64);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = (12.0 * t - 6.0).sin();
        x[[i, 2]] = (10.0 * t - 5.0).cos();
        x[[i, 3]] = (24.0 * t - 12.0).sin();
        x[[i, 4]] = (18.0 * t - 9.0).cos();
        x[[i, 5]] = 2.0 * t - 1.0;

        // Purposefully aggressive signal to push some eta into saturation regime.
        let eta = -1.8
            + signal_scale
                * (10.0 * x[[i, 1]] - 8.0 * x[[i, 2]] + 5.0 * x[[i, 3]] - 4.0 * x[[i, 4]]
                    + 2.5 * x[[i, 5]]);
        let p_i = 1.0 / (1.0 + (-eta).exp());
        y[i] = if p_i > 0.5 { 1.0 } else { 0.0 };
    }

    let mut s1 = Array2::<f64>::zeros((p, p));
    let mut s2 = Array2::<f64>::zeros((p, p));
    // Do not penalize intercept.
    for j in [1usize, 2usize, 3usize] {
        s1[[j, j]] = 1.0;
    }
    for j in [4usize, 5usize] {
        s2[[j, j]] = 1.0;
    }

    (
        x,
        y,
        w,
        offset,
        vec![
            BlockwisePenalty::new(0..p, s1),
            BlockwisePenalty::new(0..p, s2),
        ],
    )
}

#[test]
fn tracethird_auto_correction_againstfd() {
    let opts = ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodFamily::BinomialLogit,
        compute_inference: true,
        max_iter: 80,
        tol: 1e-6,
        nullspace_dims: vec![0, 0],
        linear_constraints: None,
        firth_bias_reduction: Some(false),
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };
    let rho = Array1::from_vec(vec![1.0, 1.5]);

    unsafe {
        std::env::set_var("GAM_DISABLE_GRAD_GATE", "1");
    }

    for scale in [4.0_f64, 1.0_f64, 0.25_f64] {
        let (x, y, w, offset, s_list) = make_problem(scale);
        let (analytic, fd) = evaluate_externalgradients(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            &s_list,
            &opts,
            &rho,
        )
        .expect("gradient eval should succeed");
        for i in 0..analytic.len() {
            assert_eq!(
                analytic[i].signum(),
                fd[i].signum(),
                "trace-third auto-correct sign mismatch at scale={scale:.2} i={i}: analytic={} fd={}",
                analytic[i],
                fd[i]
            );
        }
        let num = (&analytic - &fd).mapv(|v| v * v).sum().sqrt();
        let den = fd.mapv(|v| v * v).sum().sqrt().max(1e-12);
        let rel = num / den;

        eprintln!(
            "scale={:.2} trace-third auto-correct rel error: {:.3e}",
            scale, rel
        );
        assert!(rel.is_finite(), "relative error must be finite");
    }
}
