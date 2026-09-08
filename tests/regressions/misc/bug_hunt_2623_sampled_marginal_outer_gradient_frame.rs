//! #2623 GATE: the #784 block-local sampled-marginalization splice must hand the
//! outer optimizer the derivative of the criterion it also hands it.
//!
//! This is the finite-difference row the #2623 thread kept naming and could not
//! land, because until the penalty-frame defect was found its splice channel
//! failed and landing it meant landing a red gate.
//!
//! Why no existing guard could see this. The one outer-gradient FD guard in the
//! tree, `dense_spectral_large_p_outer_gradient_matches_finite_difference`,
//! exercises a GAUSSIAN dense-spectral fit — a regime in which Laplace is exact,
//! the `#784` correction is declined unconditionally, and the channels asserted
//! here are never formed at all. A comfortable fixture is not a gate: this one
//! is chosen so the splice ENGAGES, and the test asserts that it did before it
//! compares anything, so a future change that silently declines the correction
//! fails here instead of passing vacuously.
//!
//! What it pins. Contracting the ORIGINAL-frame canonical penalties against the
//! TRANSFORMED-frame Hessian, mode and design made the splice's ρ-gradient wrong
//! by 4.6e-2 to 1.0 relative on the cells below — the worst of them reporting
//! `+1.5e9` where the criterion's own slope is `-27`, i.e. not merely inaccurate
//! but sign-inverted with nine orders of magnitude of error. On the fold that
//! opened #2623 that inversion is what made `‖g‖ = 9.45` AT the cost minimum,
//! and the resulting Wolfe failure is what turned an expensive evaluation into
//! 178 of them.
//!
//! The bound is expressed against the FD's own truncation floor rather than as a
//! round number: at `h = 1e-3` the central difference on this cell resolves to
//! ~6e-8 relative, and the defect this pins was 4.6e-2. `1e-4` therefore sits
//! three orders above the floor and two orders below the defect, so neither a
//! tighter FD step nor a noisier box can flip the verdict.

use gam::estimate::outer_eval_capture::{enable_rho_outer_audit, take_rho_outer_audit};
use gam::estimate::{
    ExternalOptimOptions, evaluate_externalcost_andridge, evaluate_externalgradient,
};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2};

fn second_difference_penalty(k: usize) -> Array2<f64> {
    let mut d = Array2::<f64>::zeros((k - 2, k));
    for i in 0..(k - 2) {
        d[[i, i]] = 1.0;
        d[[i, i + 1]] = -2.0;
        d[[i, i + 2]] = 1.0;
    }
    d.t().dot(&d)
}

struct Fixture {
    y: Array1<f64>,
    w: Array1<f64>,
    x: Array2<f64>,
    offset: Array1<f64>,
    penalties: Vec<BlockwisePenalty>,
    opts: ExternalOptimOptions,
}

/// Two DISTINCT harmonic blocks over two DISTINCT covariates, binomial-logit.
/// Well conditioned by construction, so an engagement here cannot be blamed on a
/// near-singular `H_pen` or on runaway whitened draws — the splice engages
/// because the standardized cubic skewness genuinely exceeds `τ(n_eff)`, which
/// is the regime the correction exists for.
fn fixture(n: usize, k: usize, amp: f64) -> Fixture {
    let p = 1 + 2 * k;
    let mut x = Array2::<f64>::zeros((n, p));
    let mut y = Array1::<f64>::zeros(n);
    let inv_phi = 2.0 / (1.0 + 5.0_f64.sqrt());
    let half_pi = 0.5 * std::f64::consts::PI;
    for i in 0..n {
        x[[i, 0]] = 1.0;
        let z = -1.0 + 2.0 * i as f64 / (n as f64 - 1.0);
        let z2 = -1.0 + 2.0 * (0.25 + (i as f64) * inv_phi).fract();
        for j in 0..k {
            let order = (j + 1) as f64;
            x[[i, 1 + j]] = (order * half_pi * (z + 1.0)).sin();
            x[[i, 1 + k + j]] = (order * half_pi * (z2 + 1.0)).cos();
        }
        let signal = 0.7 * (std::f64::consts::PI * z).sin() + 0.3 * (2.0 * half_pi * z2).cos();
        let prob = 1.0 / (1.0 + (-amp * signal).exp());
        let u = (0.5 + (i as f64) * inv_phi).fract();
        y[i] = if u < prob { 1.0 } else { 0.0 };
    }
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
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 300,
        tol: 1.0e-12,
        nullspace_dims: vec![2, 2],
        linear_constraints: None,
        firth_bias_reduction: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    };
    Fixture {
        y,
        w: Array1::ones(n),
        x,
        offset: Array1::zeros(n),
        penalties,
        opts,
    }
}

#[test]
fn sampled_marginal_splice_outer_gradient_matches_finite_difference_2623() {
    gam::init_parallelism();

    let fix = fixture(240, 6, 3.0);
    let rho = Array1::from(vec![-1.0_f64, -0.95]);

    enable_rho_outer_audit();
    let analytic = evaluate_externalgradient(
        fix.y.view(),
        fix.w.view(),
        fix.x.clone(),
        fix.offset.view(),
        &fix.penalties,
        &fix.opts,
        &rho,
    )
    .expect("gradient evaluation");
    let audit = take_rho_outer_audit().expect("rho audit armed");

    // ARM THE GATE FIRST. Without this the comparison below is vacuous on any
    // build where the splice declines: the criterion and the gradient would then
    // agree because neither carries the channels this test exists to grade.
    assert!(
        audit.quadrature_marginal_engaged,
        "#2623 gate is unarmed: the #784 splice DECLINED on this fixture, so the \
         channels this test grades were never formed. Fix the fixture, do not \
         relax the bound."
    );
    let sampled = audit
        .quadrature_marginal
        .as_ref()
        .expect("an engaged splice publishes its channel split");

    // The FD reference is a difference of deterministic `Δ_b` quadratures.
    // The engaged gate already certifies the fine/coarse rule difference; keep
    // the node-count assertion here so a vacuous zero-node result cannot pass.
    assert!(
        sampled.node_count > 0 && sampled.quadrature_error.is_finite(),
        "#2623 gate: invalid quadrature certificate (nodes={}, error={})",
        sampled.node_count,
        sampled.quadrature_error,
    );

    let cost_at = |theta: &Array1<f64>| -> (f64, Vec<usize>) {
        enable_rho_outer_audit();
        let cost = evaluate_externalcost_andridge(
            fix.y.view(),
            fix.w.view(),
            fix.x.clone(),
            fix.offset.view(),
            &fix.penalties,
            &fix.opts,
            theta,
        )
        .expect("cost evaluation")
        .0;
        let audit = take_rho_outer_audit().expect("rho audit armed");
        let block = audit
            .quadrature_marginal
            .as_ref()
            .map(|record| record.block_cols.clone())
            .unwrap_or_default();
        (cost, block)
    };

    // Three orders above the central difference's own truncation floor on this
    // cell (~6e-8 relative at this step) and two below the defect it pins
    // (4.6e-2 on this coordinate, 1.0 on the worst cell measured).
    const RELATIVE_BOUND: f64 = 1.0e-4;
    let step = 1.0e-3_f64;

    for j in 0..rho.len() {
        let mut plus = rho.clone();
        plus[j] += step;
        let mut minus = rho.clone();
        minus[j] -= step;
        let (cost_plus, block_plus) = cost_at(&plus);
        let (cost_minus, block_minus) = cost_at(&minus);

        // Block membership is chosen by a skewness THRESHOLD, so it can move
        // across a stencil. When it does the quotient is a difference of two
        // different functions and is not a derivative of either — that is a
        // fixture defect, not a gradient defect, and it must fail loudly rather
        // than be silently compared.
        assert_eq!(
            block_minus, sampled.block_cols,
            "#2623 gate: the sampled block moved between rho-h and rho \
             ({block_minus:?} vs {:?}); the finite difference is not a derivative",
            sampled.block_cols
        );
        assert_eq!(
            block_plus, sampled.block_cols,
            "#2623 gate: the sampled block moved between rho and rho+h \
             ({block_plus:?} vs {:?}); the finite difference is not a derivative",
            sampled.block_cols
        );

        let fd = (cost_plus - cost_minus) / (2.0 * step);
        let scale = analytic[j].abs().max(fd.abs()).max(1.0);
        let rel = (analytic[j] - fd).abs() / scale;
        assert!(
            rel < RELATIVE_BOUND,
            "#2623: the spliced outer gradient is not the derivative of the \
             spliced criterion at rho coordinate {j}: analytic={:+.10e} \
             fd={:+.10e} rel={rel:.3e} (bound {RELATIVE_BOUND:.1e}). Channels: \
             a={:+.6e} trace={:+.6e} mode={:+.6e}.",
            analytic[j],
            fd,
            sampled.explicit_a[j],
            sampled.trace_bc[j],
            sampled.mode_d[j],
        );
    }
}
