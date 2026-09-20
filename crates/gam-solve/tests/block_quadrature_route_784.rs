//! #784: a one-axis block piece is integrated by the composite Gauss–Kronrod rule
//! unless its axis is truncated.
//!
//! On a reciprocal-power link the row surface is defined only on `η > 0`, so a
//! one-axis target reports the feasible interval through
//! [`BlockExcessTarget::axis_truncation`]. The Gauss–Hermite rule integrates that
//! interval exactly through its truncated-normal transport; the composite rule
//! partitions the whole axis and would meet the cut as infeasible nodes. A
//! Gamma-inverse fit must therefore reach the Gauss–Hermite rule and never the
//! composite one, and the same fit on the log link (no cut) must reach the
//! composite rule and never the Gauss–Hermite one.
//!
//! Only this binary's process registers the corrector double below, so no other
//! test sees it.

use std::sync::{Mutex, PoisonError};

use gam_linalg::matrix::DesignMatrix;
use gam_problem::laplace_sampler_contract::{
    AxisBreakpoint, BlockExcessTarget, BlockQuadratureMarginal, BlockQuadratureMoments,
    BlockQuadratureOrderRefusal, BlockQuadratureOrderStep, BlockQuadratureRefusal,
    CompositeAxisMarginal, LaplaceMarginalCorrector, set_laplace_marginal_corrector,
};
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_solve::estimate::{ExternalOptimOptions, evaluate_externalgradient};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};

/// Which rule a piece was integrated by, and whether its axis was truncated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuleCall {
    GaussHermite { truncated: bool },
    Composite { truncated: bool },
}

static CALLS: Mutex<Vec<RuleCall>> = Mutex::new(Vec::new());

fn record(call: RuleCall) {
    CALLS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push(call);
}

/// A corrector double. Its diagnostic flags only the stiffest direction, so the
/// block has one axis. Every rule it is asked for records itself and splices zero,
/// resolved at the first rule, so the fit is otherwise unchanged.
struct RuleRecorder;

impl RuleRecorder {
    fn zero_marginal(
        target: &dyn BlockExcessTarget,
        axis_orders: &[usize],
    ) -> Result<BlockQuadratureMarginal, BlockQuadratureRefusal> {
        let m = target.block_dim();
        let rows = target
            .base_neg_score()
            .map_err(BlockQuadratureRefusal::Integration)?
            .len();
        Ok(BlockQuadratureMarginal {
            value: 0.0,
            rho_gradient: Array1::zeros(target.rho_dim()),
            axis_orders: axis_orders.to_vec(),
            axis_quadrature_errors: vec![0.0; m],
            quadrature_error: 0.0,
            node_count: 1,
            chunk_nodes: 1,
            reservation_bytes: 0,
            moments: Some(BlockQuadratureMoments {
                e_t: Array1::zeros(m),
                e_tt: Array2::zeros((m, m)),
                e_neg_score: Array1::zeros(rows),
                e_t_neg_score: Array2::zeros((rows, m)),
            }),
        })
    }
}

impl LaplaceMarginalCorrector for RuleRecorder {
    fn directional_cubic_diagnostic(
        &self,
        eigenvalues: &Array1<f64>,
        eigenvectors: &Array2<f64>,
        design: &DesignMatrix,
        c_weights: &Array1<f64>,
        refine_supremum: bool,
    ) -> Result<(f64, Array1<f64>), String> {
        let p = eigenvalues.len();
        if eigenvectors.dim() != (p, p) || design.ncols() != p || c_weights.len() != design.nrows()
        {
            return Err(format!(
                "the recorder's diagnostic (refine_supremum={refine_supremum}) got {p} \
                 eigenvalues, {}x{} eigenvectors, a {}x{} design and {} curvature weights",
                eigenvectors.nrows(),
                eigenvectors.ncols(),
                design.nrows(),
                design.ncols(),
                c_weights.len()
            ));
        }
        let stiffest = (0..p)
            .max_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]))
            .ok_or_else(|| "the recorder's diagnostic got an empty spectrum".to_string())?;
        let mut directional = Array1::<f64>::zeros(p);
        directional[stiffest] = 1.0;
        Ok((1.0, directional))
    }

    fn block_quadrature_marginal_correction(
        &self,
        target: &dyn BlockExcessTarget,
        axis_orders: &[usize],
    ) -> Result<BlockQuadratureMarginal, BlockQuadratureRefusal> {
        record(RuleCall::GaussHermite {
            truncated: target.axis_truncation().is_some(),
        });
        Self::zero_marginal(target, axis_orders)
    }

    fn composite_axis_marginal_correction(
        &self,
        target: &dyn BlockExcessTarget,
        next_order_remainder: f64,
    ) -> Result<CompositeAxisMarginal, BlockQuadratureOrderRefusal> {
        record(RuleCall::Composite {
            truncated: target.axis_truncation().is_some(),
        });
        Self::zero_marginal(target, &[4])
            .map(|marginal| CompositeAxisMarginal {
                marginal,
                breakpoints: Vec::new(),
            })
            .map_err(|cause| BlockQuadratureOrderRefusal {
                axis: 0,
                axis_orders: vec![4],
                paired_error: f64::INFINITY,
                resolution_target: next_order_remainder,
                cause,
            })
    }

    fn composite_axis_marginal_correction_on_partition(
        &self,
        target: &dyn BlockExcessTarget,
        breakpoints: &[AxisBreakpoint],
    ) -> Result<CompositeAxisMarginal, BlockQuadratureOrderRefusal> {
        let mut out = self.composite_axis_marginal_correction(target, f64::INFINITY)?;
        out.breakpoints = breakpoints.to_vec();
        Ok(out)
    }

    /// Every rule resolves at the first order, so the order search raises no axis.
    fn publish_order_search_step(&self, step: &BlockQuadratureOrderStep) {
        assert!(
            step.axis_quadrature_errors.iter().all(|&error| error == 0.0),
            "the recorder resolves every axis at its first rule, so no axis is raised: {step}"
        );
    }

    fn is_representable_order(&self, order: usize) -> bool {
        gam_math::quadrature::standard_normal_gauss_hermite_order_is_representable(order)
    }
}

/// An intercept and two linear columns, each under its own 1x1 ridge, with
/// Gamma(shape 20) responses about `η = 2 + 0.3·x0 − 0.2·x1` on the requested link
/// (`η ∈ [1, 3]`, so the inverse link's mode is inside `η > 0`), drawn
/// deterministically through the Wilson–Hilferty cube of a golden-ratio normal
/// sequence.
fn gamma_fixture(
    n: usize,
    link: StandardLink,
) -> (Array1<f64>, Array2<f64>, Vec<BlockwisePenalty>) {
    let shape = 20.0_f64;
    let inv_phi = 2.0 / (1.0 + 5.0_f64.sqrt());
    let mut x = Array2::<f64>::zeros((n, 3));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x0 = -2.0 + 4.0 * i as f64 / (n as f64 - 1.0);
        let x1 = -2.0 + 4.0 * (0.25 + (i as f64) * inv_phi).fract();
        x[[i, 0]] = 1.0;
        x[[i, 1]] = x0;
        x[[i, 2]] = x1;
        let eta = 2.0 + 0.3 * x0 - 0.2 * x1;
        let mu = match link {
            StandardLink::Inverse => eta.recip(),
            _ => (eta - 2.0).exp(),
        };
        let u1 = (0.37 + (i as f64) * inv_phi).fract();
        let u2 = (0.11 + (i as f64) * 2.0_f64.sqrt()).fract();
        let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let cube_root = 1.0 - 1.0 / (9.0 * shape) + normal / (3.0 * shape.sqrt());
        y[i] = mu * cube_root.powi(3);
    }
    let ridge = || Array2::from_elem((1, 1), 1.0);
    let penalties = vec![
        BlockwisePenalty::new(1..2, ridge()),
        BlockwisePenalty::new(2..3, ridge()),
    ];
    (y, x, penalties)
}

fn rule_calls_for(link: StandardLink) -> Vec<RuleCall> {
    let (y, x, penalties) = gamma_fixture(160, link);
    let n = y.len();
    let opts = ExternalOptimOptions {
        family: LikelihoodSpec::new(ResponseFamily::Gamma, InverseLink::Standard(link)),
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
        firth_bias_reduction: None,
        rho_prior: Default::default(),
        persistent_warm_start_store: None,
    };
    let rho = Array1::from(vec![0.0_f64, 0.5]);
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    CALLS.lock().unwrap_or_else(PoisonError::into_inner).clear();
    let gradient = evaluate_externalgradient(
        y.view(),
        weights.view(),
        x,
        offset.view(),
        &penalties,
        &opts,
        &rho,
    )
    .unwrap_or_else(|error| panic!("the Gamma-{link:?} outer gradient evaluates: {error}"));
    assert!(
        gradient.iter().all(|value| value.is_finite()),
        "outer gradient must be finite (link={link:?}): {gradient}"
    );
    std::mem::take(&mut *CALLS.lock().unwrap_or_else(PoisonError::into_inner))
}

#[test]
fn a_truncated_axis_keeps_the_gauss_hermite_rule_and_an_open_axis_takes_the_composite_784() {
    drop(set_laplace_marginal_corrector(Box::new(RuleRecorder)));

    let inverse = rule_calls_for(StandardLink::Inverse);
    assert!(
        !inverse.is_empty(),
        "#784 route pin is unarmed: the Gamma-inverse fit never asked for a block rule"
    );
    assert!(
        inverse
            .iter()
            .all(|call| *call == RuleCall::GaussHermite { truncated: true }),
        "a one-axis piece on the inverse link is truncated at η = 0 and must be integrated by \
         the Gauss–Hermite transport, never the composite rule: {inverse:?}"
    );

    let log = rule_calls_for(StandardLink::Log);
    assert!(
        !log.is_empty(),
        "#784 route pin is unarmed: the Gamma-log fit never asked for a block rule"
    );
    assert!(
        log.iter()
            .all(|call| *call == RuleCall::Composite { truncated: false }),
        "a one-axis piece on the log link has no cut and must be integrated by the composite \
         rule: {log:?}"
    );
}
