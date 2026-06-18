//! TEMP diagnostic for #1255: localize the binomial/probit REML outer-gradient
//! vs finite-difference mismatch. Prints analytic gradient, central FD at three
//! step sizes, and the cost at base/+/- for each rho component. Delete after the
//! root cause is pinned.

use gam::estimate::{
    ExternalOptimOptions, evaluate_externalcost_andridge, evaluate_externalgradient,
};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2, array};

fn second_difference_penalty(k: usize) -> Array2<f64> {
    let mut d = Array2::<f64>::zeros((k - 2, k));
    for i in 0..(k - 2) {
        d[[i, i]] = 1.0;
        d[[i, i + 1]] = -2.0;
        d[[i, i + 2]] = 1.0;
    }
    d.t().dot(&d)
}

fn build(link: InverseLink, family: ResponseFamily, binary_y: bool) -> (
    Array1<f64>,
    Array1<f64>,
    Array2<f64>,
    Array1<f64>,
    Vec<BlockwisePenalty>,
    ExternalOptimOptions,
) {
    let n = 96usize;
    let k = 6usize;
    let p = 1 + 2 * k;
    let mut x = Array2::<f64>::zeros((n, p));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        x[[i, 0]] = 1.0;
        let z = -1.0 + 2.0 * i as f64 / (n as f64 - 1.0);
        let mut acc = 1.0;
        for j in 0..k {
            acc *= z;
            x[[i, 1 + j]] = acc;
            x[[i, 1 + k + j]] = acc + 1.0e-3 * ((i + j) as f64).sin();
        }
        if binary_y {
            y[i] = if (std::f64::consts::PI * z).sin() + 0.3 * (3.0 * z).cos() > 0.0 {
                1.0
            } else {
                0.0
            };
        } else {
            y[i] = 0.4 + (std::f64::consts::PI * z).sin() + 0.05 * (7.0 * z).cos();
        }
    }
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let penalties = vec![
        BlockwisePenalty::new(1..(1 + k), second_difference_penalty(k)),
        BlockwisePenalty::new((1 + k)..p, second_difference_penalty(k)),
    ];
    let opts = ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodSpec::new(family, link),
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 300,
        tol: 1.0e-12,
        nullspace_dims: vec![2, 2],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    };
    (y, weights, x, offset, penalties, opts)
}

fn report(tag: &str, opts: ExternalOptimOptions, y: Array1<f64>, w: Array1<f64>, x: Array2<f64>, offset: Array1<f64>, penalties: Vec<BlockwisePenalty>, rho: Array1<f64>) {
    let analytic = evaluate_externalgradient(
        y.view(), w.view(), x.clone(), offset.view(), &penalties, &opts, &rho,
    )
    .expect("analytic gradient");
    println!("=== {tag} | rho={:?} | max_iter={} tol={:.1e} ===", rho.to_vec(), opts.max_iter, opts.tol);
    let cost_base = evaluate_externalcost_andridge(
        y.view(), w.view(), x.clone(), offset.view(), &penalties, &opts, &rho,
    ).expect("cost base").0;
    println!("cost(base) = {:.12e}", cost_base);
    for j in 0..rho.len() {
        print!("  comp[{j}] analytic={:.9e}", analytic[j]);
        for step in [5.0e-6_f64, 1.0e-4, 1.0e-3, 1.0e-2] {
            let mut plus = rho.clone();
            plus[j] += step;
            let mut minus = rho.clone();
            minus[j] -= step;
            let fp = evaluate_externalcost_andridge(
                y.view(), w.view(), x.clone(), offset.view(), &penalties, &opts, &plus,
            ).expect("f+").0;
            let fm = evaluate_externalcost_andridge(
                y.view(), w.view(), x.clone(), offset.view(), &penalties, &opts, &minus,
            ).expect("f-").0;
            let fd = (fp - fm) / (2.0 * step);
            print!(" | fd@{:.0e}={:.9e}", step, fd);
        }
        println!();
    }
}

#[test]
#[ignore = "temp diagnostic for #1255"]
fn diag_1255_binomial_probit_outer() {
    // The failing fixture: binomial / probit (non-canonical).
    let (y, w, x, offset, pen, opts) =
        build(InverseLink::Standard(StandardLink::Probit), ResponseFamily::Binomial, true);
    report("binomial/probit", opts, y, w, x, offset, pen, array![0.2, 0.25]);

    // Canonical-link control: binomial / logit, SAME binary data + fixture.
    let (y, w, x, offset, pen, opts) =
        build(InverseLink::Standard(StandardLink::Logit), ResponseFamily::Binomial, true);
    report("binomial/logit(canonical)", opts, y, w, x, offset, pen, array![0.2, 0.25]);

    // Gaussian/identity control (the passing row).
    let (y, w, x, offset, pen, opts) =
        build(InverseLink::Standard(StandardLink::Identity), ResponseFamily::Gaussian, false);
    report("gaussian/identity", opts, y, w, x, offset, pen, array![0.2, 0.25]);
}
