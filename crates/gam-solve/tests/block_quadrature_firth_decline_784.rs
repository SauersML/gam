//! #784: the block-local quadrature declines under Firth/Jeffreys bias reduction.
//!
//! The correction integrates the remainder of the plain penalized likelihood
//! about its mode. Under Firth the inner mode is the mode of the
//! Jeffreys-penalized objective, so that remainder keeps a linear term, omits
//! the Jeffreys change, and is measured against a Hessian that carries the
//! Jeffreys curvature. On perfectly separated binomial data the mis-targeted
//! correction was orders of magnitude above `1/n_eff` and left the outer search
//! unable to certify, so every separated fit the Jeffreys prior is armed for
//! was refused. The correction must not be consulted at all for a
//! Firth fit, and must still be consulted for the same fit without Firth.
//!
//! Only this binary's process registers the corrector double below, so no other
//! test sees it.

use std::sync::atomic::{AtomicUsize, Ordering};

use gam_linalg::matrix::DesignMatrix;
use gam_problem::laplace_sampler_contract::{
    AxisBreakpoint, BlockExcessTarget, BlockQuadratureMarginal, BlockQuadratureOrderRefusal,
    BlockQuadratureOrderStep, BlockQuadratureRefusal, CompositeAxisMarginal,
    LaplaceMarginalCorrector, set_laplace_marginal_corrector,
};
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_solve::estimate::{ExternalOptimOptions, evaluate_externalgradient};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};

static DIAGNOSTIC_CALLS: AtomicUsize = AtomicUsize::new(0);

/// A corrector double that counts how often the correction asks for its
/// skewness diagnostic and reports a Gaussian posterior (zero skewness), so a
/// consulted correction splices nothing and the fit is otherwise unchanged.
struct DiagnosticCounter;

impl LaplaceMarginalCorrector for DiagnosticCounter {
    fn directional_cubic_diagnostic(
        &self,
        eigenvalues: &Array1<f64>,
        eigenvectors: &Array2<f64>,
        design: &DesignMatrix,
        c_weights: &Array1<f64>,
    ) -> Result<(f64, Array1<f64>), String> {
        let p = eigenvalues.len();
        if eigenvectors.dim() != (p, p) || design.ncols() != p || c_weights.len() != design.nrows()
        {
            return Err(format!(
                "the counter's diagnostic got {p} \
                 eigenvalues, {}x{} eigenvectors, a {}x{} design and {} curvature weights",
                eigenvectors.nrows(),
                eigenvectors.ncols(),
                design.nrows(),
                design.ncols(),
                c_weights.len()
            ));
        }
        DIAGNOSTIC_CALLS.fetch_add(1, Ordering::SeqCst);
        Ok((0.0, Array1::zeros(p)))
    }

    fn block_quadrature_marginal_correction(
        &self,
        target: &dyn BlockExcessTarget,
        axis_orders: &[usize],
    ) -> Result<BlockQuadratureMarginal, BlockQuadratureRefusal> {
        panic!(
            "a zero-skewness diagnostic never reaches the block rule ({}-axis block, orders \
             {axis_orders:?})",
            target.block_dim()
        )
    }

    fn composite_axis_marginal_correction(
        &self,
        target: &dyn BlockExcessTarget,
        next_order_remainder: f64,
    ) -> Result<CompositeAxisMarginal, BlockQuadratureOrderRefusal> {
        panic!(
            "a zero-skewness diagnostic never reaches the composite rule ({}-axis block, \
             next_order_remainder={next_order_remainder:e})",
            target.block_dim()
        )
    }

    fn composite_axis_marginal_correction_on_partition(
        &self,
        target: &dyn BlockExcessTarget,
        breakpoints: &[AxisBreakpoint],
    ) -> Result<CompositeAxisMarginal, BlockQuadratureOrderRefusal> {
        panic!(
            "a zero-skewness diagnostic never reaches the composite rule ({}-axis block, \
             {} breakpoints)",
            target.block_dim(),
            breakpoints.len()
        )
    }

    fn publish_order_search_step(&self, step: &BlockQuadratureOrderStep) {
        panic!("a zero-skewness diagnostic never runs the order search: {step}")
    }

    fn is_representable_order(&self, order: usize) -> bool {
        gam_math::quadrature::standard_normal_gauss_hermite_order_is_representable(order)
    }
}

/// An intercept and two linear columns, each column under its own 1x1 ridge
/// (the design `y ~ x0 + x1` builds), with Bernoulli responses about
/// `logit μ = 0.8·x0 − 0.5·x1` drawn from deterministic low-discrepancy
/// sequences.
fn binomial_fixture(n: usize) -> (Array1<f64>, Array2<f64>, Vec<BlockwisePenalty>) {
    let inv_phi = 2.0 / (1.0 + 5.0_f64.sqrt());
    let mut x = Array2::<f64>::zeros((n, 3));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x0 = -2.0 + 4.0 * i as f64 / (n as f64 - 1.0);
        let x1 = -2.0 + 4.0 * (0.25 + (i as f64) * inv_phi).fract();
        x[[i, 0]] = 1.0;
        x[[i, 1]] = x0;
        x[[i, 2]] = x1;
        let mu = 1.0 / (1.0 + (-(0.8 * x0 - 0.5 * x1)).exp());
        let u = (0.37 + (i as f64) * 2.0_f64.sqrt()).fract();
        y[i] = if u < mu { 1.0 } else { 0.0 };
    }
    let ridge = || Array2::from_elem((1, 1), 1.0);
    let penalties = vec![
        BlockwisePenalty::new(1..2, ridge()),
        BlockwisePenalty::new(2..3, ridge()),
    ];
    (y, x, penalties)
}

fn diagnostic_calls_for(firth_bias_reduction: Option<bool>) -> usize {
    let (y, x, penalties) = binomial_fixture(120);
    let n = y.len();
    let opts = ExternalOptimOptions {
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 300,
        tol: 1.0e-12,
        nullspace_dims: vec![0, 0],
        linear_constraints: None,
        firth_bias_reduction,
        rho_prior: Default::default(),
        persistent_warm_start_store: None,
    };
    let rho = Array1::from(vec![0.0_f64, 0.5]);
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let before = DIAGNOSTIC_CALLS.load(Ordering::SeqCst);
    let gradient = evaluate_externalgradient(
        y.view(),
        weights.view(),
        x,
        offset.view(),
        &penalties,
        &opts,
        &rho,
    )
    .expect("the binomial-logit outer gradient evaluates");
    assert!(
        gradient.iter().all(|value| value.is_finite()),
        "outer gradient must be finite (firth={firth_bias_reduction:?}): {gradient}"
    );
    DIAGNOSTIC_CALLS.load(Ordering::SeqCst) - before
}

#[test]
fn block_quadrature_is_declined_under_firth_and_consulted_without_it_784() {
    drop(set_laplace_marginal_corrector(Box::new(DiagnosticCounter)));
    let plain = diagnostic_calls_for(None);
    assert!(
        plain > 0,
        "#784 decline pin is unarmed: the plain binomial fit never consulted the block \
         correction, so a Firth decline could not be observed"
    );
    let firth = diagnostic_calls_for(Some(true));
    assert_eq!(
        firth, 0,
        "the Firth fit consulted the block correction {firth} time(s); its integrand is the \
         plain penalized likelihood, not the Jeffreys-penalized objective the Firth mode solves"
    );
}
