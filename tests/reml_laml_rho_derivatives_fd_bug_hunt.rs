use gam::estimate::{
    ExternalOptimOptions, evaluate_externalcost_andridge, evaluate_externalgradient,
};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn opts(nullspace_dims: Vec<usize>) -> ExternalOptimOptions {
    ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 500,
        tol: 1e-12,
        nullspace_dims,
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    }
}

fn second_difference_penalty(k: usize) -> Array2<f64> {
    let mut d = Array2::<f64>::zeros((k - 2, k));
    for i in 0..(k - 2) {
        d[[i, i]] = 1.0;
        d[[i, i + 1]] = -2.0;
        d[[i, i + 2]] = 1.0;
    }
    d.t().dot(&d)
}

fn build_problem(
    seed: u64,
    blocks: usize,
) -> (
    Array2<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Vec<BlockwisePenalty>,
    Vec<usize>,
) {
    let n = 180usize;
    let block_k = 5usize;
    let p = 1 + blocks * block_k;
    let mut rng = StdRng::seed_from_u64(seed);

    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        for b in 0..blocks {
            let z = rng.random_range(-1.0..1.0);
            let mut acc = 1.0;
            for j in 0..block_k {
                acc *= z;
                x[[i, 1 + b * block_k + j]] = acc;
            }
        }
    }

    let mut beta = Array1::<f64>::zeros(p);
    beta[0] = 0.2;
    for j in 1..p {
        beta[j] = 0.15 / (j as f64).sqrt();
    }
    let eta = x.dot(&beta);
    let y = Array1::from_iter(eta.iter().map(|e| e + rng.random_range(-0.3..0.3)));
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);

    let mut penalties = Vec::new();
    let mut nullspace_dims = Vec::new();
    for b in 0..blocks {
        let start = 1 + b * block_k;
        let end = start + block_k;
        penalties.push(BlockwisePenalty::new(
            start..end,
            second_difference_penalty(block_k),
        ));
        nullspace_dims.push(2usize);
    }

    (x, y, w, offset, penalties, nullspace_dims)
}

fn cost(
    y: &Array1<f64>,
    w: &Array1<f64>,
    x: &Array2<f64>,
    offset: &Array1<f64>,
    s_list: &[BlockwisePenalty],
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
) -> f64 {
    evaluate_externalcost_andridge(
        y.view(),
        w.view(),
        x.clone(),
        offset.view(),
        s_list,
        opts,
        rho,
    )
    .expect("cost evaluation should succeed")
    .0
}

#[test]
fn reml_laml_rho_gradient_and_hessian_match_finite_difference_on_random_small_multiblock_problems()
{
    let h = 1e-5;
    let mut rng = StdRng::seed_from_u64(0xBAD5EED_u64);

    for blocks in 1..=3usize {
        let (x, y, w, offset, s_list, nullspace_dims) =
            build_problem(10_000 + blocks as u64, blocks);
        let opts = opts(nullspace_dims);
        for _ in 0..8 {
            let rho = Array1::from_iter((0..blocks).map(|_| rng.random_range(-2.5..2.5)));
            let g = evaluate_externalgradient(
                y.view(),
                w.view(),
                x.clone(),
                offset.view(),
                &s_list,
                &opts,
                &rho,
            )
            .expect("analytic gradient evaluation should succeed");

            for i in 0..blocks {
                let mut rp = rho.clone();
                rp[i] += h;
                let mut rm = rho.clone();
                rm[i] -= h;
                let fd = (cost(&y, &w, &x, &offset, &s_list, &opts, &rp)
                    - cost(&y, &w, &x, &offset, &s_list, &opts, &rm))
                    / (2.0 * h);
                assert!(
                    (g[i] - fd).abs() <= 1e-4,
                    "Expected analytic REML/LAML score derivative to match finite difference within 1e-4 but at blocks={blocks}, rho={:?}, coord={i} analytic={} fd={} diff={}",
                    rho,
                    g[i],
                    fd,
                    (g[i] - fd).abs()
                );
            }

            for i in 0..blocks {
                let mut rp = rho.clone();
                rp[i] += h;
                let gp = evaluate_externalgradient(
                    y.view(),
                    w.view(),
                    x.clone(),
                    offset.view(),
                    &s_list,
                    &opts,
                    &rp,
                )
                .expect("analytic gradient plus should succeed");
                let mut rm = rho.clone();
                rm[i] -= h;
                let gm = evaluate_externalgradient(
                    y.view(),
                    w.view(),
                    x.clone(),
                    offset.view(),
                    &s_list,
                    &opts,
                    &rm,
                )
                .expect("analytic gradient minus should succeed");
                for j in 0..blocks {
                    let analytic_second = (gp[j] - gm[j]) / (2.0 * h);

                    let mut rpp = rho.clone();
                    rpp[i] += h;
                    rpp[j] += h;
                    let mut rpm = rho.clone();
                    rpm[i] += h;
                    rpm[j] -= h;
                    let mut rmp = rho.clone();
                    rmp[i] -= h;
                    rmp[j] += h;
                    let mut rmm = rho.clone();
                    rmm[i] -= h;
                    rmm[j] -= h;
                    let fd_second = (cost(&y, &w, &x, &offset, &s_list, &opts, &rpp)
                        - cost(&y, &w, &x, &offset, &s_list, &opts, &rpm)
                        - cost(&y, &w, &x, &offset, &s_list, &opts, &rmp)
                        + cost(&y, &w, &x, &offset, &s_list, &opts, &rmm))
                        / (4.0 * h * h);

                    assert!(
                        (analytic_second - fd_second).abs() <= 1e-3,
                        "Expected analytic REML/LAML second derivative to match finite difference within 1e-3 but at blocks={blocks}, rho={:?}, entry=({i},{j}) analytic={} fd={} diff={}",
                        rho,
                        analytic_second,
                        fd_second,
                        (analytic_second - fd_second).abs()
                    );
                }
            }
        }
    }
}
