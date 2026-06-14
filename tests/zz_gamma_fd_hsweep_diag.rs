//! TEMPORARY diagnostic (delete after use): classify the interior gamma-log
//! ρ-gradient discrepancy as FD truncation/inner-noise vs a true analytic
//! desync by sweeping the central-difference step h. A real objective↔gradient
//! desync gives an h-independent error floor; FD truncation scales ~h².
use gam::estimate::{
    ExternalOptimOptions, evaluate_externalcost_andridge, evaluate_externalgradient,
};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn identity_ridge(p: usize) -> Array2<f64> {
    Array2::<f64>::eye(p)
}

fn positive_response_single_block(seed: u64, intercept: f64) -> (Array2<f64>, Array1<f64>) {
    let n = 180usize;
    let p = 7usize;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        for j in 1..p {
            x[[i, j]] = rng.random_range(-0.8..0.8);
        }
    }
    let mut beta = Array1::<f64>::zeros(p);
    beta[0] = intercept;
    for j in 1..p {
        beta[j] = 0.2 / (j as f64).sqrt();
    }
    let eta = x.dot(&beta);
    let y = Array1::from_iter(
        eta.iter()
            .map(|e| (e.exp() * (1.0 + 0.2 * rng.random_range(-1.0..1.0))).max(1e-3)),
    );
    (x, y)
}

fn opts() -> ExternalOptimOptions {
    ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodSpec::new(
            ResponseFamily::Gamma,
            InverseLink::Standard(StandardLink::Log),
        ),
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 600,
        tol: 1e-11,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    }
}

#[test]
fn gamma_log_rho_gradient_h_sweep() {
    let (x, y) = positive_response_single_block(404, 0.4);
    let n = y.len();
    let p = x.ncols();
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let s_list = vec![BlockwisePenalty::new(0..p, identity_ridge(p))];
    let o = opts();
    let rho = Array1::from(vec![0.0_f64]);

    let analytic = evaluate_externalgradient(
        y.view(),
        w.view(),
        x.clone(),
        offset.view(),
        &s_list,
        &o,
        &rho,
    )
    .expect("analytic gradient")[0];
    assert!(
        analytic.is_finite(),
        "analytic gamma-log rho gradient must be finite"
    );

    let cost = |r: &Array1<f64>| -> f64 {
        evaluate_externalcost_andridge(y.view(), w.view(), x.clone(), offset.view(), &s_list, &o, r)
            .expect("cost")
            .0
    };

    println!("ANALYTIC gamma-log rho-grad @ rho=0 : {analytic:.12e}");
    for h in [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6] {
        let mut rp = rho.clone();
        rp[0] += h;
        let mut rm = rho.clone();
        rm[0] -= h;
        let fd = (cost(&rp) - cost(&rm)) / (2.0 * h);
        assert!(
            fd.is_finite(),
            "finite-difference gradient must be finite at h={h:.1e}"
        );
        let rel = (analytic - fd).abs() / analytic.abs().max(1e-12);
        println!("h={h:.1e}  fd={fd:.12e}  rel_err={rel:.3e}");
    }
}
