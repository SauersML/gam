//! Shared exact-Newton primary directional sweep for the marginal-slope
//! families' Gaussian-frailty scale (BMS, survival, latent survival).
//!
//! This used to be live production infrastructure in
//! `gam_models::marginal_slope_shared`; it was superseded there by faster,
//! per-family specializations, but is kept here — as a genuine, drift-free
//! single source shared by more than one family's test suite — rather than
//! duplicated per call site. Centralizing the sweep removes the per-family
//! duplication of the obj/grad/hess loop nest, which is the single most
//! drift-prone piece of the exact-Newton stack: a stray index or a missing
//! symmetric assignment in one copy silently destabilizes only that family's
//! optimizer.

use gam_math::jet_partitions::MultiDirJet;
use gam_models::survival::lognormal_kernel::ProbitFrailtyScaleJet;
use ndarray::{Array1, Array2};

/// One auxiliary-parameter channel's objective, full primary gradient, and
/// symmetric primary Hessian.
pub struct DirectionalPrimaryTerms {
    pub objective: f64,
    pub grad: Array1<f64>,
    pub hess: Array2<f64>,
}

pub fn probit_frailty_scale_multi_dir_jet(
    gaussian_frailty_sd: Option<f64>,
    missing_sigma_message: &str,
    n_dirs: usize,
    first_masks: &[usize],
    second_masks: &[usize],
) -> Result<MultiDirJet, String> {
    let sigma = gaussian_frailty_sd.ok_or_else(|| missing_sigma_message.to_string())?;
    let jet = ProbitFrailtyScaleJet::from_log_sigma(sigma.ln());
    let mut coeffs = Vec::with_capacity(1 + first_masks.len() + second_masks.len());
    coeffs.push((0usize, jet.s));
    coeffs.extend(first_masks.iter().copied().map(|mask| (mask, jet.ds)));
    coeffs.extend(second_masks.iter().copied().map(|mask| (mask, jet.d2s)));
    Ok(MultiDirJet::with_coeffs(n_dirs, &coeffs))
}

/// Per-sweep scale jets for the shared directional obj/grad/hess kernel.
///
/// Every marginal-slope family forms its exact-Newton primary terms by
/// differentiating the same row negative-log directional jet
/// (`row_neglog_directional_with_scale_jet`) along unit primary directions.
/// The sweep appends one unit direction for the gradient pass and two for the
/// Hessian pass on top of a fixed *leading* prefix of directions, scaling the
/// frailty kernel with an order-matched [`MultiDirJet`] each time. The `obj`
/// slot is `Some` only when the caller also wants the zeroth-order objective
/// (the prefix-only evaluation); psi-Hessian directional sweeps leave it `None`.
#[derive(Clone)]
pub struct DirectionalScaleJets {
    pub obj: Option<MultiDirJet>,
    pub grad: MultiDirJet,
    pub hess: MultiDirJet,
}

/// Given a fixed `leading` prefix of directions and a family-specific row jet
/// evaluator `eval`, this builds the objective (when requested), the gradient
/// `g_a = D[leading, e_a] φ`, and the symmetric Hessian
/// `H_ab = D[leading, e_a, e_b] φ`, where `e_a` is the `a`-th unit primary
/// direction (length `primary_dim`) and `D[..]` is the mixed directional
/// derivative the row jet returns. `eval(dirs, scale)` must return the highest
/// mixed-partial coefficient of the row negative-log jet for the supplied
/// directions and scale jet — exactly what each family's
/// `row_neglog_directional_with_scale_jet` produces.
pub fn directional_obj_grad_hess<Eval>(
    primary_dim: usize,
    leading: &[&Array1<f64>],
    scales: &DirectionalScaleJets,
    eval: Eval,
) -> Result<DirectionalPrimaryTerms, String>
where
    Eval: Fn(&[&Array1<f64>], &MultiDirJet) -> Result<f64, String>,
{
    let objective = if let Some(scale_obj) = scales.obj.as_ref() {
        eval(leading, scale_obj)?
    } else {
        0.0
    };

    let unit = |a: usize| -> Array1<f64> {
        let mut da = Array1::<f64>::zeros(primary_dim);
        da[a] = 1.0;
        da
    };

    let units: Vec<Array1<f64>> = (0..primary_dim).map(unit).collect();

    let mut grad = Array1::<f64>::zeros(primary_dim);
    let mut dirs: Vec<&Array1<f64>> = Vec::with_capacity(leading.len() + 2);
    for a in 0..primary_dim {
        dirs.clear();
        dirs.extend_from_slice(leading);
        dirs.push(&units[a]);
        grad[a] = eval(&dirs, &scales.grad)?;
    }

    let mut hess = Array2::<f64>::zeros((primary_dim, primary_dim));
    for a in 0..primary_dim {
        for b in a..primary_dim {
            dirs.clear();
            dirs.extend_from_slice(leading);
            dirs.push(&units[a]);
            dirs.push(&units[b]);
            let value = eval(&dirs, &scales.hess)?;
            hess[[a, b]] = value;
            hess[[b, a]] = value;
        }
    }

    Ok(DirectionalPrimaryTerms {
        objective,
        grad,
        hess,
    })
}
