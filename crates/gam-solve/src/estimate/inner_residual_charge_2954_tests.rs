//! #2954: the certificate's band charges the error `V` carries because the inner
//! mode stops at a residual, `E_r = ½·rᵀH_β⁻¹r`, on routes whose assembly
//! presents exact KKT and so never corrects for it. Arming the capture hands the
//! residual over for the band alone: the criterion must not move.

use super::*;
use crate::estimate::outer_eval_capture::{
    InnerResidualSource, begin_certificate_parts_capture, take_certificate_evidence,
};
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};

const ROWS: usize = 400;
const BASIS: usize = 12;

/// Degree-one B-splines (hat functions) on `[0, 1]` with `BASIS` uniform knots,
/// a partition of unity, so the second-difference penalty's null space carries
/// the level and the slope and no separate intercept column is needed.
fn fixture(family: &ResponseFamily) -> (Array2<f64>, Array1<f64>) {
    let mut state = 0x2954_u64;
    let mut unit = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut x = Array2::<f64>::zeros((ROWS, BASIS));
    let mut y = Array1::<f64>::zeros(ROWS);
    for i in 0..ROWS {
        let t = unit();
        for j in 0..BASIS {
            x[[i, j]] = (1.0 - (t * (BASIS - 1) as f64 - j as f64).abs()).max(0.0);
        }
        let f = (2.0 * std::f64::consts::PI * t).sin();
        y[i] = match family {
            ResponseFamily::Binomial => {
                if unit() < 1.0 / (1.0 + (-2.0 * f).exp()) {
                    1.0
                } else {
                    0.0
                }
            }
            _ => f + 0.3 * (unit() - 0.5),
        };
    }
    (x, y)
}

fn second_difference_penalty(k: usize) -> Array2<f64> {
    let mut d = Array2::<f64>::zeros((k - 2, k));
    for r in 0..k - 2 {
        d[[r, r]] = 1.0;
        d[[r, r + 1]] = -2.0;
        d[[r, r + 2]] = 1.0;
    }
    d.t().dot(&d)
}

/// What one evaluation at `rho` on a freshly built REML state returns: the value
/// and gradient bits, the armed capture's evidence, and the inner solve's
/// residual, penalized Hessian and coefficients in its own frame.
struct Evaluated {
    cost: f64,
    gradient: Array1<f64>,
    evidence: Option<crate::estimate::outer_eval_capture::CertificateEvidence>,
    residual: Array1<f64>,
    hessian: Array2<f64>,
    beta: Array1<f64>,
}

fn evaluate(family: ResponseFamily, link: StandardLink, rho: f64, armed: bool) -> Evaluated {
    let (x, y) = fixture(&family);
    let weights = Array1::<f64>::ones(ROWS);
    let offset = Array1::<f64>::zeros(ROWS);
    let specs = vec![PenaltySpec::from_blockwise_ref(&BlockwisePenalty::new(
        0..BASIS,
        second_difference_penalty(BASIS),
    ))];
    let ext = ExternalOptimOptions {
        family: LikelihoodSpec::new(family, InverseLink::Standard(link)),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 300,
        tol: 1e-7,
        nullspace_dims: vec![2],
        linear_constraints: None,
        firth_bias_reduction: Some(false),
        rho_prior: Default::default(),
        persistent_warm_start_store: None,
    };
    let cfg = super::external_options::resolved_external_config(&ext)
        .expect("the external config resolves")
        .0;
    let x_dm: DesignMatrix = x.into();
    let (canonical, active_nullspace_dims) = gam_terms::construction::canonicalize_penalty_specs(
        &specs,
        &ext.nullspace_dims,
        BASIS,
        "inner_residual_charge_2954",
    )
    .expect("the penalty canonicalizes");
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x_dm, &specs);
    let x_fit = conditioning.apply_to_design(&x_dm);
    let mut state = RemlState::newwith_offset(
        y.view(),
        x_fit,
        weights.view(),
        offset.view(),
        canonical,
        BASIS,
        &cfg,
        Some(active_nullspace_dims),
        None,
        None,
    )
    .expect("the REML state builds");
    state.set_rho_prior(ext.rho_prior.clone());
    let rho = Array1::from(vec![rho]);
    if armed {
        begin_certificate_parts_capture();
    }
    let eval = state
        .compute_outer_eval_with_order(
            &rho,
            crate::rho_optimizer::OuterEvalOrder::ValueGradientHessian,
        )
        .expect("the point evaluates");
    let evidence = armed.then(take_certificate_evidence);
    let bundle = state
        .obtain_eval_bundle(&rho)
        .expect("the evaluated bundle is cached");
    let pirls = bundle.pirls_result.as_ref();
    Evaluated {
        cost: eval.cost,
        gradient: eval.gradient.clone(),
        evidence,
        residual: pirls.penalized_gradient_transformed.clone(),
        hessian: pirls.penalized_hessian_transformed.to_dense(),
        beta: pirls.beta_transformed.as_ref().clone(),
    }
}

/// Arming the certificate capture hands the inner residual to the band and
/// nothing else: `V` and its gradient are bit-identical armed and unarmed, on an
/// iterative P-IRLS route (binomial logit, charged from the inner Newton's own
/// final gradient) and a direct one (Gaussian identity, charged from the
/// normal-equation residual), each evaluated on its own fresh state so no cache
/// can serve the second evaluation.
#[test]
fn arming_the_capture_leaves_the_criterion_bit_identical_2954() {
    for (family, link, source) in [
        (
            ResponseFamily::Binomial,
            StandardLink::Logit,
            InnerResidualSource::InnerGradient,
        ),
        (
            ResponseFamily::Gaussian,
            StandardLink::Identity,
            InnerResidualSource::NormalEquations,
        ),
    ] {
        let unarmed = evaluate(family.clone(), link, 1.0, false);
        let armed = evaluate(family.clone(), link, 1.0, true);
        assert_eq!(
            armed.cost.to_bits(),
            unarmed.cost.to_bits(),
            "{family:?}: V armed {:.17e} unarmed {:.17e}",
            armed.cost,
            unarmed.cost,
        );
        assert_eq!(
            armed.gradient.mapv(f64::to_bits),
            unarmed.gradient.mapv(f64::to_bits),
            "{family:?}: the gradient moves when the capture is armed",
        );
        let charge = armed
            .evidence
            .and_then(|evidence| evidence.inner_residual)
            .unwrap_or_else(|| panic!("{family:?}: the armed evaluation charges its inner mode"));
        assert_eq!(charge.source, source, "{family:?}");
        assert!(
            charge.energy.is_finite() && charge.energy >= 0.0,
            "{family:?}: {charge:?}"
        );
    }
}

/// A direct Gaussian-identity solve returns `β̂` from a backward-stable
/// factorization of `H = XᵀWX + S_λ`, so its normal-equation residual is
/// rounding: the solve's backward error `(H + ΔH)β̂ = b`, `|ΔH| ≤
/// γ_(3p+1)·|R̂ᵀ||R̂|` (Higham Thm 10.4) with `‖|R̂ᵀ||R̂|‖ ≤ ‖R̂‖_F² = tr H ≤
/// √p·‖H‖_F ≤ p·‖H‖_F`, leaves `‖Hβ̂ − b‖ ≤ p·γ_(3p+1)·‖H‖_F·‖β̂‖`, and forming `Hβ̂ − b` in
/// floating point and carrying it through the orthogonal reparameterization adds
/// `γ_(p+1)·(‖H‖_F‖β̂‖ + ‖b‖) + γ_p·‖r‖`, at most `3·γ_(p+1)·‖H‖_F·‖β̂‖` more.
#[test]
fn a_direct_gaussian_solve_leaves_a_rounding_level_residual_2954() {
    let evaluated = evaluate(ResponseFamily::Gaussian, StandardLink::Identity, 1.0, true);
    let p = evaluated.beta.len();
    let frobenius = evaluated.hessian.iter().map(|v| v * v).sum::<f64>().sqrt();
    let beta_norm = evaluated.beta.dot(&evaluated.beta).sqrt();
    let residual_norm = evaluated.residual.dot(&evaluated.residual).sqrt();
    let gamma = gam_linalg::roundoff::accumulation_growth;
    let bound = (p as f64 * gamma(3 * p + 1) + 3.0 * gamma(p + 1)) * frobenius * beta_norm;
    assert!(
        residual_norm <= bound,
        "‖r‖ = {residual_norm:.3e} exceeds the direct solve's rounding bound {bound:.3e} \
         (p = {p}, ‖H‖_F = {frobenius:.3e}, ‖β̂‖ = {beta_norm:.3e})",
    );
}
