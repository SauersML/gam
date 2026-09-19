//! #784: the block-local quadrature scores its non-Gaussian remainder at the
//! likelihood the inner solve converged under.
//!
//! The correction integrates `ΔF(t) = F(β̂ + V_b t) − F(β̂) + (Sβ̂)·δ − ½ Σ_i W_i s_i²`
//! against the Laplace Gaussian. `W_i` are the solve's observed weights, so the
//! quadratic part of `F` cancels exactly when `F` is scored at the solve's own
//! likelihood scale: `ΔF` has no quadratic term at the mode. PIRLS estimates the
//! Gamma shape `k̂` and locks it for the solve. Scoring `F` at the
//! configuration's shape instead (φ = 1 against the solve's 1/k̂ ≈ 1/20 on this
//! cell) leaves a quadratic coefficient of order one, and the lane sweep 1147402
//! measured the spliced correction there at +2.05e-2, against +3.10e-4 at the
//! solve's own shape.
//!
//! Only this binary's process registers the corrector double below, so no other
//! test sees it.

use std::sync::{Mutex, PoisonError};

use gam_linalg::matrix::DesignMatrix;
use gam_problem::laplace_sampler_contract::{
    BlockExcessTarget, BlockQuadratureMarginal, BlockQuadratureMoments, BlockQuadratureOrderStep,
    BlockQuadratureRefusal, LaplaceMarginalCorrector, set_laplace_marginal_corrector,
};
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_solve::estimate::{ExternalOptimOptions, evaluate_externalgradient};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};

/// One block axis's estimate of `ΔF`'s quadratic coefficient in whitened units,
/// with the two parts of its band.
struct AxisQuadratic {
    axis: usize,
    /// The Gauss–Hermite order the order search asked the probe for along this axis.
    axis_order: usize,
    curvature: f64,
    estimate: f64,
    rounding: f64,
    truncation: f64,
    half_step_rounding: f64,
}

static QUADRATICS: Mutex<Vec<AxisQuadratic>> = Mutex::new(Vec::new());

/// The whitened step: one posterior standard deviation along the axis, the scale
/// the correction integrates over.
const STEP: f64 = 1.0;

/// A corrector double. It admits every positive-curvature direction. On each rule
/// it is asked for, it Richardson-estimates `ΔF`'s quadratic coefficient along
/// every block axis instead of integrating, then splices zero.
///
/// With `q(z) = [ΔF(z) + ΔF(−z)]/z² = 2(c₂ + c₄z² + c₆z⁴ + …)` in whitened units:
/// - `ĉ₂ = (4q(h/2) − q(h))/6 = c₂ − c₆h⁴/4` removes the quartic term.
/// - The same combination at `(h/2, h/4)` gives `c₂ − c₆h⁴/64`, so the remaining
///   truncation `c₆h⁴/4` is `(16/15)·|ĉ₂(h/2) − ĉ₂(h)|`.
/// - `q(z)` rounds within `(b(z) + b(−z))/z²` for the target's own
///   [`BlockExcessTarget::excess_rounding_band`] `b`, so `ĉ₂` rounds within
///   `(4·δq(h/2) + δq(h))/6`.
struct QuadraticCoefficientProbe;

impl LaplaceMarginalCorrector for QuadraticCoefficientProbe {
    fn directional_cubic_diagnostic(
        &self,
        hessian: &Array2<f64>,
        design: &DesignMatrix,
        c_weights: &Array1<f64>,
        refine_supremum: bool,
    ) -> Result<(f64, Array1<f64>), String> {
        if design.ncols() != hessian.nrows() || c_weights.len() != design.nrows() {
            return Err(format!(
                "the probe's diagnostic (refine_supremum={refine_supremum}) got a {}x{} Hessian, a \
                 {}x{} design and {} curvature weights",
                hessian.nrows(),
                hessian.ncols(),
                design.nrows(),
                design.ncols(),
                c_weights.len()
            ));
        }
        Ok((1.0, Array1::from_elem(hessian.nrows(), 1.0)))
    }

    fn block_quadrature_marginal_correction(
        &self,
        target: &dyn BlockExcessTarget,
        axis_orders: &[usize],
    ) -> Result<BlockQuadratureMarginal, BlockQuadratureRefusal> {
        let m = target.block_dim();
        let curvatures = target.block_curvatures();
        let mut records = Vec::with_capacity(m);
        for axis in 0..m {
            let sample = |z: f64| {
                let mut plus = Array1::<f64>::zeros(m);
                plus[axis] = z / curvatures[axis].sqrt();
                let minus = plus.mapv(|value| -value);
                let q = (target.excess(&plus) + target.excess(&minus)) / (z * z);
                let band = (target.excess_rounding_band(&plus) + target.excess_rounding_band(&minus))
                    / (z * z);
                (q, band)
            };
            let (q_full, band_full) = sample(STEP);
            let (q_half, band_half) = sample(STEP / 2.0);
            let (q_quarter, band_quarter) = sample(STEP / 4.0);
            let estimate = (4.0 * q_half - q_full) / 6.0;
            let finer = (4.0 * q_quarter - q_half) / 6.0;
            records.push(AxisQuadratic {
                axis,
                axis_order: axis_orders[axis],
                curvature: curvatures[axis],
                estimate,
                rounding: (4.0 * band_half + band_full) / 6.0
                    + (4.0 * band_quarter + band_half) / 6.0 * 16.0 / 15.0,
                truncation: (finer - estimate).abs() * 16.0 / 15.0,
                half_step_rounding: band_half,
            });
        }
        QUADRATICS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend(records);
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

    /// The probe reports every axis resolved at the first rule it is asked for, so the
    /// order search never raises one, and a published step means it did.
    fn publish_order_search_step(&self, step: &BlockQuadratureOrderStep) {
        assert!(
            step.axis_quadrature_errors.iter().all(|&error| error == 0.0),
            "the probe resolves every axis at its first rule, so no axis is raised: {step}"
        );
    }

    /// The production ceiling, measured by the rule builder the standard corrector integrates
    /// with. The probe reports every axis resolved at the first rule it is asked for (order
    /// four), so the order search's stop predicate never judges an axis and the ceiling is
    /// only compared against. The pin asserts order four.
    fn max_representable_order(&self) -> usize {
        gam_math::quadrature::max_representable_standard_normal_gauss_hermite_order()
    }
}

fn second_difference_penalty(k: usize) -> Array2<f64> {
    let mut root = Array2::<f64>::zeros((k - 2, k));
    for row in 0..(k - 2) {
        root[[row, row]] = 1.0;
        root[[row, row + 1]] = -2.0;
        root[[row, row + 2]] = 1.0;
    }
    root.t().dot(&root)
}

/// Two harmonic blocks over two covariates with Gamma(shape 20) responses about
/// `μ = exp(2·signal)`, drawn deterministically through the Wilson–Hilferty cube
/// of a golden-ratio normal sequence: the n=240, k=8 cell of sweep 1147402.
fn gamma_fixture(n: usize, k: usize) -> (Array1<f64>, Array2<f64>, Vec<BlockwisePenalty>) {
    let shape = 20.0_f64;
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
        let u1 = (0.37 + (i as f64) * inv_phi).fract();
        let u2 = (0.11 + (i as f64) * 2.0_f64.sqrt()).fract();
        let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let cube_root = 1.0 - 1.0 / (9.0 * shape) + normal / (3.0 * shape.sqrt());
        y[i] = (2.0 * signal).exp() * cube_root.powi(3);
    }
    let penalties = vec![
        BlockwisePenalty::new(1..(1 + k), second_difference_penalty(k)),
        BlockwisePenalty::new((1 + k)..p, second_difference_penalty(k)),
    ];
    (y, x, penalties)
}

#[test]
fn block_quadrature_remainder_has_no_quadratic_term_at_the_mode_784() {
    drop(set_laplace_marginal_corrector(Box::new(QuadraticCoefficientProbe)));
    let (y, x, penalties) = gamma_fixture(240, 8);
    let n = y.len();
    let opts = ExternalOptimOptions {
        family: LikelihoodSpec::new(ResponseFamily::Gamma, InverseLink::Standard(StandardLink::Log)),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 300,
        tol: 1.0e-12,
        nullspace_dims: vec![2, 2],
        linear_constraints: None,
        firth_bias_reduction: None,
        rho_prior: Default::default(),
        persistent_warm_start_store: None,
    };
    let rho = Array1::from(vec![-1.0_f64, -0.95]);
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let gradient = evaluate_externalgradient(
        y.view(),
        weights.view(),
        x,
        offset.view(),
        &penalties,
        &opts,
        &rho,
    )
    .expect("the Gamma-log outer gradient evaluates");
    let records = QUADRATICS.lock().unwrap_or_else(PoisonError::into_inner);
    assert!(
        !records.is_empty(),
        "#784 units pin is unarmed: the block quadrature was never asked for a rule (gradient \
         {gradient})"
    );
    for record in records.iter() {
        eprintln!(
            "[784units] axis {} curvature {:.4e}: c2 {:+.6e}, rounding {:.3e}, truncation {:.3e}, \
             q(h/2) rounding {:.3e} against an O(1) coefficient",
            record.axis,
            record.curvature,
            record.estimate,
            record.rounding,
            record.truncation,
            record.half_step_rounding
        );
    }
    for record in records.iter() {
        assert_eq!(
            record.axis_order, 4,
            "the probe resolves every axis at the first rule, so the order search must ask for \
             order four along axis {} and never reach the representable ceiling",
            record.axis
        );
        assert!(
            record.estimate.abs() <= record.rounding + record.truncation,
            "the remainder has a quadratic term at the mode along axis {} (curvature {:.4e}): \
             c2 = {:+.6e} against rounding {:.3e} and truncation {:.3e}, so ΔF was scored at a \
             likelihood scale other than the solve's",
            record.axis,
            record.curvature,
            record.estimate,
            record.rounding,
            record.truncation
        );
    }
}
