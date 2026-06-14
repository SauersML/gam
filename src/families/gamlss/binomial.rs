// Real concern-organized submodule of the gamlss family stack.
// Cross-module items are re-exported flat through the parent (`gamlss.rs`),
// so `use super::*;` makes the sibling-concern symbols this module references
// resolve through the parent namespace.
use super::*;



pub(crate) struct BinomialLocationScaleCore {
    sigma: Array1<f64>,
    dsigma_deta: Array1<f64>,
    q0: Array1<f64>,
    mu: Array1<f64>,
    dmu_dq: Array1<f64>,
    d2mu_dq2: Array1<f64>,
    d3mu_dq3: Array1<f64>,
    log_likelihood: f64,
}


#[derive(Clone, Copy)]
pub(crate) struct NonWiggleQDerivs {
    q_t: f64,
    q_ls: f64,
    q_tl: f64,
    q_ll: f64,
    q_tl_ls: f64,
    q_ll_ls: f64,
}


#[derive(Clone, Copy)]
pub(crate) struct NonWiggleQDirectional {
    delta_q: f64,
    delta_q_t: f64,
    delta_q_ls: f64,
    delta_q_tl: f64,
    delta_q_ll: f64,
}


#[derive(Clone, Copy)]
pub(crate) struct BinomialLocationScaleRow {
    sigma: f64,
    dsigma_deta: f64,
    q0: f64,
    inverse_link: crate::mixture_link::InverseLinkJet,
    ll: f64,
}


/// Non-wiggle location-scale map derivatives via shared scalar core.
pub(crate) fn nonwiggle_q_derivs(eta_t: f64, sigma: f64) -> NonWiggleQDerivs {
    let inv_sigma = sigma.recip();
    let q_t = -inv_sigma;
    let q_ls = eta_t * inv_sigma;
    let q_tl = inv_sigma;
    let q_ll = -eta_t * inv_sigma;
    let q_tl_ls = -inv_sigma;
    let q_ll_ls = eta_t * inv_sigma;
    NonWiggleQDerivs {
        q_t,
        q_ls,
        q_tl,
        q_ll,
        q_tl_ls,
        q_ll_ls,
    }
}


/// Directional derivatives along (d_eta_t, d_eta_ls):
/// delta_q = q_t d_eta_t + q_ls d_eta_ls
/// delta_q_t = q_tl d_eta_ls
/// delta_q_ls = q_tl d_eta_t + q_ll d_eta_ls
/// delta_q_tt = 0
/// delta_q_tl = q_tl_ls d_eta_ls
/// delta_q_ll = q_tl_ls d_eta_t + q_ll_ls d_eta_ls
pub(crate) fn nonwiggle_q_directional(
    q: NonWiggleQDerivs,
    d_eta_t: f64,
    d_eta_ls: f64,
) -> NonWiggleQDirectional {
    // Directional-chain derivation:
    //
    // For any scalar f(eta_t,eta_ls), directional derivative along
    // d eta = (d_eta_t, d_eta_ls) is
    //   dot{f} = f_t d_eta_t + f_ls d_eta_ls.
    //
    // Apply to q and its eta-partials:
    //   dot{q}      = q_t d_eta_t + q_ls d_eta_ls.
    //   dot{q_t}    = q_tt d_eta_t + q_tl d_eta_ls = q_tl d_eta_ls (q_tt=0).
    //   dot{q_ls}   = q_tl d_eta_t + q_ll d_eta_ls.
    //   dot{q_tt}   = 0.
    //   dot{q_tl}   = q_tl_ls d_eta_ls.
    //   dot{q_ll}   = q_tl_ls d_eta_t + q_ll_ls d_eta_ls.
    NonWiggleQDirectional {
        delta_q: q.q_t * d_eta_t + q.q_ls * d_eta_ls,
        delta_q_t: q.q_tl * d_eta_ls,
        delta_q_ls: q.q_tl * d_eta_t + q.q_ll * d_eta_ls,
        delta_q_tl: q.q_tl_ls * d_eta_ls,
        delta_q_ll: q.q_tl_ls * d_eta_t + q.q_ll_ls * d_eta_ls,
    }
}


#[inline]
pub(crate) fn log1mexp_neg_positive(z: f64) -> f64 {
    assert!(z >= 0.0);
    if z == 0.0 {
        f64::NEG_INFINITY
    } else if z <= std::f64::consts::LN_2 {
        (-(-z).exp_m1()).ln()
    } else {
        (1.0 - (-z).exp()).ln()
    }
}


#[inline]
pub(crate) fn bernoulli_log_likelihood_from_probability(y: f64, weight: f64, mu: f64) -> Result<f64, String> {
    if weight == 0.0 {
        return Ok(0.0);
    }
    if !mu.is_finite() || !(0.0..=1.0).contains(&mu) {
        return Err(GamlssError::NumericalFailure {
            reason: format!(
                "binomial location-scale inverse link returned invalid probability {mu}"
            ),
        }
        .into());
    }
    let log_mu = if mu == 0.0 {
        if y == 0.0 { 0.0 } else { f64::NEG_INFINITY }
    } else {
        mu.ln()
    };
    let log_one_minus = if mu == 1.0 {
        if y == 1.0 { 0.0 } else { f64::NEG_INFINITY }
    } else {
        (1.0 - mu).ln()
    };
    let ll = weight * (y * log_mu + (1.0 - y) * log_one_minus);
    if ll.is_finite() {
        Ok(ll)
    } else {
        Err(GamlssError::NonFinite {
            reason: format!(
                "binomial location-scale log likelihood is non-finite at y={y}, mu={mu}"
            ),
        }
        .into())
    }
}


#[inline]
pub(crate) fn binomial_location_scale_q0(eta_t: f64, sigma: f64) -> f64 {
    -eta_t / sigma
}


#[inline]
pub(crate) fn binomial_location_scale_log_likelihood(
    y: f64,
    weight: f64,
    q: f64,
    link_kind: &InverseLink,
    mu: f64,
) -> Result<f64, String> {
    if weight == 0.0 {
        return Ok(0.0);
    }
    match link_kind {
        InverseLink::Standard(StandardLink::Probit) => {
            Ok(weight * (y * normal_logcdf(q) + (1.0_f64 - y) * normal_logsf(q)))
        }
        InverseLink::Standard(StandardLink::Logit) => Ok(weight
            * (-y * crate::linalg::utils::stable_softplus(-q)
                - (1.0_f64 - y) * crate::linalg::utils::stable_softplus(q))),
        InverseLink::Standard(StandardLink::CLogLog) => {
            let z = q.exp();
            let log_p = if z == 0.0 {
                q
            } else if z.is_infinite() {
                0.0
            } else {
                log1mexp_neg_positive(z)
            };
            let log_survival = -z;
            let ll = weight * (y * log_p + (1.0_f64 - y) * log_survival);
            if ll.is_finite() {
                Ok(ll)
            } else {
                Err(GamlssError::NonFinite { reason: format!(
                    "binomial cloglog location-scale log likelihood is non-finite at y={y}, q={q}"
                ) }.into())
            }
        }
        _ => bernoulli_log_likelihood_from_probability(y, weight, mu),
    }
}


#[inline]
pub(crate) fn binomial_expected_q_information_derivatives(
    weight: f64,
    mu: f64,
    d1: f64,
    d2: f64,
    d3: f64,
) -> (f64, f64, f64) {
    if weight == 0.0
        || !mu.is_finite()
        || !d1.is_finite()
        || !d2.is_finite()
        || !d3.is_finite()
        || mu <= 0.0
        || mu >= 1.0
        || d1 == 0.0
    {
        return (0.0, 0.0, 0.0);
    }
    let var = mu * (1.0 - mu);
    if !var.is_finite() || var <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let var1 = d1 * (1.0 - 2.0 * mu);
    let var2 = d2 * (1.0 - 2.0 * mu) - 2.0 * d1 * d1;

    let f = weight * d1 * d1 / var;
    let num1 = 2.0 * d1 * d2 * var - d1 * d1 * var1;
    let f1 = weight * num1 / (var * var);
    let num1_prime = 2.0 * (d2 * d2 + d1 * d3) * var - d1 * d1 * var2;
    let f2 = weight * (num1_prime / (var * var) - 2.0 * num1 * var1 / (var * var * var));
    if f.is_finite() && f1.is_finite() && f2.is_finite() {
        (f, f1, f2)
    } else {
        (0.0, 0.0, 0.0)
    }
}


pub(crate) fn binomial_expected_location_scale_second_coefficients(
    q: NonWiggleQDerivs,
    f: f64,
    f1: f64,
    f2: f64,
    d_eta_t_u: f64,
    d_eta_ls_u: f64,
    d_eta_t_v: f64,
    d_eta_ls_v: f64,
) -> (f64, f64, f64) {
    let u = nonwiggle_q_directional(q, d_eta_t_u, d_eta_ls_u);
    let v = nonwiggle_q_directional(q, d_eta_t_v, d_eta_ls_v);
    let q_uv = q.q_tl * (d_eta_t_u * d_eta_ls_v + d_eta_t_v * d_eta_ls_u)
        + q.q_ll * d_eta_ls_u * d_eta_ls_v;
    let q_t_uv = q.q_tl_ls * d_eta_ls_u * d_eta_ls_v;
    let q_ls_uv = q.q_tl_ls * (d_eta_ls_u * d_eta_t_v + d_eta_ls_v * d_eta_t_u)
        + q.q_ll_ls * d_eta_ls_u * d_eta_ls_v;
    let scalar = f2 * u.delta_q * v.delta_q + f1 * q_uv;
    let tt = scalar * q.q_t * q.q_t
        + 2.0 * f1 * u.delta_q * q.q_t * v.delta_q_t
        + 2.0 * f1 * v.delta_q * q.q_t * u.delta_q_t
        + 2.0 * f * (q.q_t * q_t_uv + u.delta_q_t * v.delta_q_t);
    let tl = scalar * q.q_t * q.q_ls
        + f1 * u.delta_q * (v.delta_q_t * q.q_ls + q.q_t * v.delta_q_ls)
        + f1 * v.delta_q * (u.delta_q_t * q.q_ls + q.q_t * u.delta_q_ls)
        + f * (q_t_uv * q.q_ls
            + q.q_t * q_ls_uv
            + u.delta_q_t * v.delta_q_ls
            + v.delta_q_t * u.delta_q_ls);
    let ll = scalar * q.q_ls * q.q_ls
        + 2.0 * f1 * u.delta_q * q.q_ls * v.delta_q_ls
        + 2.0 * f1 * v.delta_q * q.q_ls * u.delta_q_ls
        + 2.0 * f * (q.q_ls * q_ls_uv + u.delta_q_ls * v.delta_q_ls);
    (tt, tl, ll)
}


pub(crate) fn binomial_location_scalerow(
    y: f64,
    weight: f64,
    eta_t: f64,
    eta_ls: f64,
    etawiggle: f64,
    link_kind: &InverseLink,
) -> Result<BinomialLocationScaleRow, String> {
    let SigmaJet1 {
        sigma,
        d1: dsigma_deta,
    } = exp_sigma_jet1_scalar(eta_ls);
    let q0 = binomial_location_scale_q0(eta_t, sigma);
    let q = q0 + etawiggle;
    let jet = inverse_link_jet_for_inverse_link(link_kind, q)
        .map_err(|e| format!("location-scale inverse-link evaluation failed: {e}"))?;
    let raw_mu = jet.mu;
    // μ is stored RAW (unclamped). The q-derivative tower built downstream
    // (binomial_neglog_q_derivatives_dispatch et al.) is the EXACT derivative
    // of the loss evaluated here, computed via the per-branch reciprocals in
    // `binomial_loglik_mu_derivatives` plus the saturation guard in the
    // `*_from_jet` consumers. Flooring μ at MIN_PROB here would replace every
    // representable sub-MIN_PROB tail probability with a 1e-10 surrogate,
    // corrupting the Fisher curvature throughout the saturated tail (#948).
    // The inverse-link derivatives d1/d2/d3 carry the legitimate gradient
    // signal and are likewise preserved.
    let inverse_link = jet;
    let ll = binomial_location_scale_log_likelihood(y, weight, q, link_kind, raw_mu)?;
    Ok(BinomialLocationScaleRow {
        sigma,
        dsigma_deta,
        q0,
        inverse_link,
        ll,
    })
}


/// Compute only the log-likelihood scalar for the binomial location-scale model.
/// This avoids allocating 7 n-vectors that `binomial_location_scale_core` would produce,
/// making backtracking line searches much cheaper at large scale.
pub(crate) fn binomial_location_scale_ll_only(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    eta_t: &Array1<f64>,
    eta_ls: &Array1<f64>,
    etawiggle: Option<&Array1<f64>>,
    link_kind: &InverseLink,
) -> Result<f64, String> {
    let n = y.len();
    let y_slice = y.as_slice().expect("y must be contiguous");
    let w_slice = weights.as_slice().expect("weights must be contiguous");
    let et_slice = eta_t.as_slice().expect("eta_t must be contiguous");
    let el_slice = eta_ls.as_slice().expect("eta_ls must be contiguous");
    let ew_slice = etawiggle.map(|w| w.as_slice().expect("etawiggle must be contiguous"));
    (0..n)
        .into_par_iter()
        .try_fold(
            || 0.0_f64,
            |acc, i| -> Result<f64, String> {
                let SigmaJet1 { sigma, .. } = exp_sigma_jet1_scalar(el_slice[i]);
                let q0 = binomial_location_scale_q0(et_slice[i], sigma);
                let q = q0 + ew_slice.map_or(0.0, |w| w[i]);
                if matches!(link_kind, InverseLink::Standard(StandardLink::Probit)) {
                    return Ok(acc
                        + binomial_location_scale_log_likelihood(
                            y_slice[i], w_slice[i], q, link_kind, 0.5,
                        )?);
                }
                let jet = inverse_link_jet_for_inverse_link(link_kind, q)
                    .map_err(|e| format!("location-scale inverse-link evaluation failed: {e}"))?;
                Ok(acc
                    + binomial_location_scale_log_likelihood(
                        y_slice[i], w_slice[i], q, link_kind, jet.mu,
                    )?)
            },
        )
        .try_reduce(|| 0.0_f64, |a, b| Ok(a + b))
}


pub(crate) fn binomial_location_scale_core(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    eta_t: &Array1<f64>,
    eta_ls: &Array1<f64>,
    etawiggle: Option<&Array1<f64>>,
    link_kind: &InverseLink,
) -> Result<BinomialLocationScaleCore, String> {
    let n = y.len();
    if weights.len() != n || eta_t.len() != n || eta_ls.len() != n {
        return Err(GamlssError::DimensionMismatch {
            reason: "binomial location-scale core size mismatch".to_string(),
        }
        .into());
    }
    if let Some(w) = etawiggle
        && w.len() != n
    {
        return Err(GamlssError::DimensionMismatch {
            reason: "binomial location-scale core wiggle size mismatch".to_string(),
        }
        .into());
    }

    // Parallel per-row probit/inverse-link evaluation. At large scale
    // (n = 320K) the sequential probit erfc loop was a major single-thread
    // hotspot called dozens of times per outer REML gradient evaluation.
    let y_slice = y.as_slice().expect("y must be contiguous");
    let w_slice = weights.as_slice().expect("weights must be contiguous");
    let et_slice = eta_t.as_slice().expect("eta_t must be contiguous");
    let el_slice = eta_ls.as_slice().expect("eta_ls must be contiguous");
    let ew_slice = etawiggle.map(|w| w.as_slice().expect("etawiggle must be contiguous"));

    // Write each row's seven scalar derivatives directly into preallocated
    // output buffers in parallel, reducing the per-row log-likelihood
    // alongside. The previous path collected a `Vec<BinomialLocationScaleRow>`
    // (8 scalar fields plus alignment) and then serially scattered into the
    // seven `Array1`s, which at large scale n=3e5 cost ~50 MB of transient
    // allocation and a single-threaded post-pass.
    let mut sigma = vec![0.0_f64; n];
    let mut dsigma_deta = vec![0.0_f64; n];
    let mut q0 = vec![0.0_f64; n];
    let mut mu = vec![0.0_f64; n];
    let mut dmu_dq = vec![0.0_f64; n];
    let mut d2mu_dq2 = vec![0.0_f64; n];
    let mut d3mu_dq3 = vec![0.0_f64; n];

    /// Wrapper to send raw pointers across threads for disjoint per-row writes.
    /// Each parallel iteration writes to a unique index `i`, and the caller
    /// ensures the pointers outlive the parallel region (see SAFETY: notes
    /// on each `unsafe` site below).
    #[derive(Clone, Copy)]
    struct SendPtr(*mut f64);
    // SAFETY: pointers are constructed from live writable buffers and used
    // only for disjoint per-row writes inside a bounded parallel region; the
    // owning `Vec`s outlive the region.
    unsafe impl Send for SendPtr {}
    // SAFETY: same live-buffer and disjoint-index invariants as `Send`; no
    // two threads write the same offset through any shared `SendPtr` value.
    unsafe impl Sync for SendPtr {}
    impl SendPtr {
        #[inline(always)]
        // SAFETY: `self.0` points to a live writable allocation with length
        // greater than `i`, and `i` is exclusively owned by the calling
        // parallel iteration.
        unsafe fn write(self, i: usize, v: f64) {
            // SAFETY: see `write`'s function-level note: `i` is in-bounds
            // and exclusively owned by this iteration.
            unsafe { *self.0.add(i) = v };
        }
    }

    let sigma_p = SendPtr(sigma.as_mut_ptr());
    let dsigma_p = SendPtr(dsigma_deta.as_mut_ptr());
    let q0_p = SendPtr(q0.as_mut_ptr());
    let mu_p = SendPtr(mu.as_mut_ptr());
    let dmu_p = SendPtr(dmu_dq.as_mut_ptr());
    let d2mu_p = SendPtr(d2mu_dq2.as_mut_ptr());
    let d3mu_p = SendPtr(d3mu_dq3.as_mut_ptr());

    let ll = (0..n)
        .into_par_iter()
        .map(move |i| {
            let row = binomial_location_scalerow(
                y_slice[i],
                w_slice[i],
                et_slice[i],
                el_slice[i],
                ew_slice.map_or(0.0, |w| w[i]),
                link_kind,
            )?;
            // SAFETY: `i` comes from `0..n`, so it is in-bounds for each
            // preallocated length-`n` buffer, and every index is produced once;
            // each pointer targets a distinct output buffer.
            unsafe {
                sigma_p.write(i, row.sigma);
                dsigma_p.write(i, row.dsigma_deta);
                q0_p.write(i, row.q0);
                mu_p.write(i, row.inverse_link.mu);
                dmu_p.write(i, row.inverse_link.d1);
                d2mu_p.write(i, row.inverse_link.d2);
                d3mu_p.write(i, row.inverse_link.d3);
            }
            Ok::<f64, String>(row.ll)
        })
        .try_reduce(|| 0.0_f64, |a, b| Ok(a + b))?;

    Ok(BinomialLocationScaleCore {
        sigma: Array1::from_vec(sigma),
        dsigma_deta: Array1::from_vec(dsigma_deta),
        q0: Array1::from_vec(q0),
        mu: Array1::from_vec(mu),
        dmu_dq: Array1::from_vec(dmu_dq),
        d2mu_dq2: Array1::from_vec(d2mu_dq2),
        d3mu_dq3: Array1::from_vec(d3mu_dq3),
        log_likelihood: ll,
    })
}


#[inline]
pub(crate) fn binomial_location_scale_nll_tower(
    y: f64,
    weight: f64,
    eta_t: f64,
    eta_ls: f64,
    q_value: f64,
    mu: f64,
    dmu_dq: f64,
    d2mu_dq2: f64,
    d3mu_dq3: f64,
    link_kind: &InverseLink,
    include_fourth: bool,
) -> Result<crate::families::jet_tower::Tower4<2>, String> {
    use crate::families::jet_tower::Tower4;
    let eta_t_tower = Tower4::<2>::variable(eta_t, 0);
    let eta_ls_tower = Tower4::<2>::variable(eta_ls, 1);
    let inv_sigma = (eta_ls_tower * -1.0).exp();
    let q = -eta_t_tower * inv_sigma;
    let ll = binomial_location_scale_log_likelihood(y, weight, q_value, link_kind, mu)?;
    let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
        y, weight, q_value, mu, dmu_dq, d2mu_dq2, d3mu_dq3, link_kind,
    );
    let m4 = if include_fourth {
        binomial_neglog_q_fourth_derivative_dispatch(
            y, weight, q_value, mu, dmu_dq, d2mu_dq2, d3mu_dq3, link_kind,
        )?
    } else {
        0.0
    };
    Ok(q.compose_unary([-ll, m1, m2, m3, m4]))
}


#[inline]
pub(crate) fn binomial_location_scale_nll_tower_from_core_row(
    y: f64,
    weight: f64,
    core: &BinomialLocationScaleCore,
    row: usize,
    link_kind: &InverseLink,
    include_fourth: bool,
) -> Result<crate::families::jet_tower::Tower4<2>, String> {
    let sigma = core.sigma[row];
    let eta_t = -core.q0[row] * sigma;
    let eta_ls = sigma.ln();
    binomial_location_scale_nll_tower(
        y,
        weight,
        eta_t,
        eta_ls,
        core.q0[row],
        core.mu[row],
        core.dmu_dq[row],
        core.d2mu_dq2[row],
        core.d3mu_dq3[row],
        link_kind,
        include_fourth,
    )
}


/// Pure row-coefficient builder for the binomial location-scale joint
/// directional derivative `D_β H_L[u]`. Returns `(c_tt, c_tl, c_ll)` such
/// that the resulting matrix is
///
///   X_t^T diag(c_tt) X_t + X_t^T diag(c_tl) X_ls (+ symmetric)
///   + X_ls^T diag(c_ll) X_ls.
///
/// Inputs `d_eta_t = X_t · u_t`, `d_eta_ls = X_ls · u_ls` are the linear
/// predictor perturbations along the joint direction `u = (u_t, u_ls)`.
pub(crate) fn binomial_location_scale_first_directional_coefficients(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    core: &BinomialLocationScaleCore,
    d_eta_t: &Array1<f64>,
    d_eta_ls: &Array1<f64>,
    link_kind: &InverseLink,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    let n = y.len();
    let triples: Result<Vec<(f64, f64, f64)>, String> = (0..n)
        .into_par_iter()
        .map(|i| {
            let tower = binomial_location_scale_nll_tower_from_core_row(
                y[i], weights[i], core, i, link_kind, false,
            )?;
            let dir = [d_eta_t[i], d_eta_ls[i]];
            let contracted = tower.third_contracted(&dir);
            Ok((contracted[0][0], contracted[0][1], contracted[1][1]))
        })
        .collect();
    let mut coeff_tt = Array1::<f64>::zeros(n);
    let mut coeff_tl = Array1::<f64>::zeros(n);
    let mut coeff_ll = Array1::<f64>::zeros(n);
    for (i, (tt, tl, ll)) in triples?.into_iter().enumerate() {
        coeff_tt[i] = tt;
        coeff_tl[i] = tl;
        coeff_ll[i] = ll;
    }
    Ok((coeff_tt, coeff_tl, coeff_ll))
}


/// Pure row-coefficient builder for the binomial location-scale joint
/// second directional derivative `D²_β H_L[u, v]`. Returns
/// `(c_tt, c_tl, c_ll)` analogous to the first-order helper but built from
/// the four predictor perturbations `(d_eta_t_u, d_eta_ls_u, d_eta_t_v,
/// d_eta_ls_v)`.
pub(crate) fn binomial_location_scalesecond_directional_coefficients(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    core: &BinomialLocationScaleCore,
    d_eta_t_u: &Array1<f64>,
    d_eta_ls_u: &Array1<f64>,
    d_eta_t_v: &Array1<f64>,
    d_eta_ls_v: &Array1<f64>,
    link_kind: &InverseLink,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let n = y.len();
    // Per-row second-directional coefficient computation. m4 dispatch
    // can fail (Result), so collect a Result<Vec<(tt, tl, ll)>>.
    let triples: Result<Vec<(f64, f64, f64)>, String> = (0..n)
        .into_par_iter()
        .map(|i| -> Result<(f64, f64, f64), String> {
            let tower = binomial_location_scale_nll_tower_from_core_row(
                y[i], weights[i], core, i, link_kind, true,
            )?;
            let dir_u = [d_eta_t_u[i], d_eta_ls_u[i]];
            let dir_v = [d_eta_t_v[i], d_eta_ls_v[i]];
            let contracted = tower.fourth_contracted(&dir_u, &dir_v);
            Ok((contracted[0][0], contracted[0][1], contracted[1][1]))
        })
        .collect();
    let triples = triples?;
    let mut coeff_tt = Array1::<f64>::zeros(n);
    let mut coeff_tl = Array1::<f64>::zeros(n);
    let mut coeff_ll = Array1::<f64>::zeros(n);
    for (i, (tt, tl, ll)) in triples.into_iter().enumerate() {
        coeff_tt[i] = tt;
        coeff_tl[i] = tl;
        coeff_ll[i] = ll;
    }
    Ok((coeff_tt, coeff_tl, coeff_ll))
}


/// Built-in Gaussian location-scale family:
/// - Block 0: location μ(·) with identity link
/// - Block 1: log-scale log σ(·) with log link



impl BinomialLocationScaleFamily {
    pub const BLOCK_T: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;

    pub fn parameternames() -> &'static [&'static str] {
        &["threshold", "log_sigma"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::InverseLink, ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "binomial_location_scale",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }

    fn exact_joint_supported(&self) -> bool {
        self.threshold_design.is_some() && self.log_sigma_design.is_some()
    }

    fn dense_block_designs(&self) -> Result<(Cow<'_, Array2<f64>>, Cow<'_, Array2<f64>>), String> {
        dense_locscale_block_designs_cached(
            self.threshold_design.as_ref(),
            self.log_sigma_design.as_ref(),
            "BinomialLocationScaleFamily",
            "BinomialLocationScale",
            "threshold",
            &self.policy.material_policy(),
        )
    }

    fn dense_block_designs_fromspecs<'a>(
        &self,
        specs: &'a [ParameterBlockSpec],
    ) -> Result<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>), String> {
        dense_locscale_block_designs_fromspecs(
            specs,
            2,
            "BinomialLocationScaleFamily",
            "BinomialLocationScale",
            Self::BLOCK_T,
            Self::BLOCK_LOG_SIGMA,
            "threshold",
            &self.policy.material_policy(),
        )
    }

    fn exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        // The non-wiggle family is structurally capable of exact joint outer
        // rho-derivatives whenever the realized threshold and log-sigma
        // designs are available somewhere. Prefer cached family designs when
        // present, but allow the outer hyper code to recover the exact same
        // joint path from the realized `specs`.
        //
        // This is not a convenience fallback. The coupled profiled derivative
        // is defined in terms of the joint mode system
        //
        //   H u_k = -A_k beta,
        //
        // so if the block specs already determine the realized joint
        // curvature, forcing the code back onto a blockwise surrogate just
        // because the family did not cache duplicate dense designs would be
        // mathematically wrong.
        if self.threshold_design.is_some() && self.log_sigma_design.is_some() {
            return self.dense_block_designs().map(Some);
        }
        if let Some(specs) = specs {
            return self.dense_block_designs_fromspecs(specs).map(Some);
        }
        Ok(None)
    }

    fn exact_joint_block_designs_owned(
        &self,
        specs: Option<&[ParameterBlockSpec]>,
    ) -> Result<Option<(DesignMatrix, DesignMatrix)>, String> {
        let designs = if let (Some(x_t), Some(x_ls)) = (
            self.threshold_design.as_ref(),
            self.log_sigma_design.as_ref(),
        ) {
            Some((x_t.clone(), x_ls.clone()))
        } else if let Some(specs) = specs {
            if specs.len() != 2 {
                return Err(GamlssError::DimensionMismatch { reason: format!(
                    "BinomialLocationScaleFamily spec-aware operator path expects 2 specs, got {}",
                    specs.len()
                ) }.into());
            }
            Some((
                specs[Self::BLOCK_T].design.clone(),
                specs[Self::BLOCK_LOG_SIGMA].design.clone(),
            ))
        } else {
            None
        };
        let Some((x_t, x_ls)) = designs else {
            return Ok(None);
        };
        let n = self.y.len();
        if x_t.nrows() != n || x_ls.nrows() != n {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialLocationScaleFamily operator designs have row mismatch: y={}, threshold={}, log_sigma={}",
                n,
                x_t.nrows(),
                x_ls.nrows()
            ) }.into());
        }
        Ok(Some((x_t, x_ls)))
    }

    fn exact_newton_joint_gradient_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &DesignMatrix,
        x_ls: &DesignMatrix,
    ) -> Result<ExactNewtonJointGradientEvaluation, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n
            || eta_ls.len() != n
            || self.weights.len() != n
            || x_t.nrows() != n
            || x_ls.nrows() != n
        {
            return Err(
                "BinomialLocationScaleFamily joint gradient input size mismatch".to_string(),
            );
        }

        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let mut grad_eta_t_v = vec![0.0_f64; n];
        let mut grad_eta_ls_v = vec![0.0_f64; n];
        let y_slice = self.y.as_slice().expect("y must be contiguous");
        let w_slice = self.weights.as_slice().expect("weights must be contiguous");
        let q0_slice = core.q0.as_slice().expect("q0 must be contiguous");
        let eta_t_slice = eta_t.as_slice().expect("eta_t must be contiguous");
        let eta_ls_slice = eta_ls.as_slice().expect("eta_ls must be contiguous");
        let link_kind = &self.link_kind;
        let gradient_pairs: Result<Vec<(f64, f64)>, String> = (0..n)
            .into_par_iter()
            .map(|i| {
                let tower = binomial_location_scale_nll_tower(
                    y_slice[i],
                    w_slice[i],
                    eta_t_slice[i],
                    eta_ls_slice[i],
                    q0_slice[i],
                    core.mu[i],
                    core.dmu_dq[i],
                    core.d2mu_dq2[i],
                    core.d3mu_dq3[i],
                    link_kind,
                    false,
                )?;
                Ok((-tower.g[0], -tower.g[1]))
            })
            .collect();
        for (i, (g_t, g_ls)) in gradient_pairs?.into_iter().enumerate() {
            grad_eta_t_v[i] = g_t;
            grad_eta_ls_v[i] = g_ls;
        }
        let grad_eta_t = Array1::from_vec(grad_eta_t_v);
        let grad_eta_ls = Array1::from_vec(grad_eta_ls_v);
        let grad_t = x_t.transpose_vector_multiply(&grad_eta_t);
        let grad_ls = x_ls.transpose_vector_multiply(&grad_eta_ls);
        let total = grad_t.len() + grad_ls.len();
        let mut gradient = Array1::<f64>::zeros(total);
        gradient.slice_mut(s![0..grad_t.len()]).assign(&grad_t);
        gradient.slice_mut(s![grad_t.len()..total]).assign(&grad_ls);
        Ok(ExactNewtonJointGradientEvaluation {
            log_likelihood: core.log_likelihood,
            gradient,
        })
    }

    fn exact_newton_joint_hessian_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_block_designs_owned(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_from_design_matrices(block_states, &x_t, &x_ls)
    }

    fn exact_newton_joint_hessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessian_directional_derivative_from_designs(
            block_states,
            &x_t,
            &x_ls,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessian_second_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.exact_newton_joint_hessiansecond_directional_derivative_from_designs(
            block_states,
            &x_t,
            &x_ls,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn expected_joint_information_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n
            || eta_ls.len() != n
            || self.weights.len() != n
            || x_t.nrows() != n
            || x_ls.nrows() != n
        {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily expected information input size mismatch"
                    .to_string(),
            }
            .into());
        }
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let rows: Vec<(f64, f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let q = nonwiggle_q_derivs(eta_t[i], core.sigma[i]);
                let (f, _, _) = binomial_expected_q_information_derivatives(
                    self.weights[i],
                    core.mu[i],
                    core.dmu_dq[i],
                    core.d2mu_dq2[i],
                    core.d3mu_dq3[i],
                );
                (f * q.q_t * q.q_t, f * q.q_t * q.q_ls, f * q.q_ls * q.q_ls)
            })
            .collect();
        let mut coeff_tt = Array1::<f64>::zeros(n);
        let mut coeff_tl = Array1::<f64>::zeros(n);
        let mut coeff_ll = Array1::<f64>::zeros(n);
        for (i, (tt, tl, ll)) in rows.into_iter().enumerate() {
            coeff_tt[i] = tt;
            coeff_tl[i] = tl;
            coeff_ll[i] = ll;
        }
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        let h_tt = xt_diag_x_dense(x_t, &coeff_tt)?;
        let h_tl = xt_diag_y_dense(x_t, &coeff_tl, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &coeff_ll)?;
        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..pt, 0..pt]).assign(&h_tt);
        h.slice_mut(s![0..pt, pt..total]).assign(&h_tl);
        h.slice_mut(s![pt..total, pt..total]).assign(&h_ll);
        mirror_upper_to_lower(&mut h);
        Ok(Some(h))
    }

    fn expected_joint_information_directional_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n
            || eta_ls.len() != n
            || self.weights.len() != n
            || x_t.nrows() != n
            || x_ls.nrows() != n
        {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily expected dI input size mismatch".to_string(),
            }
            .into());
        }
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        if d_beta_flat.len() != total {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expected dI direction length mismatch: got {}, expected {}",
                    d_beta_flat.len(),
                    total
                ),
            }
            .into());
        }
        let d_eta_t = fast_av(x_t, &d_beta_flat.slice(s![0..pt]));
        let d_eta_ls = fast_av(x_ls, &d_beta_flat.slice(s![pt..total]));
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let rows: Vec<(f64, f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let q = nonwiggle_q_derivs(eta_t[i], core.sigma[i]);
                let u = nonwiggle_q_directional(q, d_eta_t[i], d_eta_ls[i]);
                let (f, f1, _) = binomial_expected_q_information_derivatives(
                    self.weights[i],
                    core.mu[i],
                    core.dmu_dq[i],
                    core.d2mu_dq2[i],
                    core.d3mu_dq3[i],
                );
                let tt = f1 * u.delta_q * q.q_t * q.q_t + 2.0 * f * q.q_t * u.delta_q_t;
                let tl = f1 * u.delta_q * q.q_t * q.q_ls
                    + f * (u.delta_q_t * q.q_ls + q.q_t * u.delta_q_ls);
                let ll = f1 * u.delta_q * q.q_ls * q.q_ls + 2.0 * f * q.q_ls * u.delta_q_ls;
                (tt, tl, ll)
            })
            .collect();
        let mut coeff_tt = Array1::<f64>::zeros(n);
        let mut coeff_tl = Array1::<f64>::zeros(n);
        let mut coeff_ll = Array1::<f64>::zeros(n);
        for (i, (tt, tl, ll)) in rows.into_iter().enumerate() {
            coeff_tt[i] = tt;
            coeff_tl[i] = tl;
            coeff_ll[i] = ll;
        }
        let d_h_tt = xt_diag_x_dense(x_t, &coeff_tt)?;
        let d_h_tl = xt_diag_y_dense(x_t, &coeff_tl, x_ls)?;
        let d_h_ll = xt_diag_x_dense(x_ls, &coeff_ll)?;
        let mut d_h = Array2::<f64>::zeros((total, total));
        d_h.slice_mut(s![0..pt, 0..pt]).assign(&d_h_tt);
        d_h.slice_mut(s![0..pt, pt..total]).assign(&d_h_tl);
        d_h.slice_mut(s![pt..total, pt..total]).assign(&d_h_ll);
        mirror_upper_to_lower(&mut d_h);
        Ok(Some(d_h))
    }

    fn expected_joint_information_second_directional_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n
            || eta_ls.len() != n
            || self.weights.len() != n
            || x_t.nrows() != n
            || x_ls.nrows() != n
        {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily expected d2I input size mismatch".to_string(),
            }
            .into());
        }
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        if d_beta_u_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialLocationScaleFamily expected d2I u direction length mismatch: got {}, expected {}",
                d_beta_u_flat.len(),
                total
            ) }.into());
        }
        if d_betav_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialLocationScaleFamily expected d2I v direction length mismatch: got {}, expected {}",
                d_betav_flat.len(),
                total
            ) }.into());
        }
        let d_eta_t_u = fast_av(x_t, &d_beta_u_flat.slice(s![0..pt]));
        let d_eta_ls_u = fast_av(x_ls, &d_beta_u_flat.slice(s![pt..total]));
        let d_eta_t_v = fast_av(x_t, &d_betav_flat.slice(s![0..pt]));
        let d_eta_ls_v = fast_av(x_ls, &d_betav_flat.slice(s![pt..total]));
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let rows: Vec<(f64, f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let q = nonwiggle_q_derivs(eta_t[i], core.sigma[i]);
                let (f, f1, f2) = binomial_expected_q_information_derivatives(
                    self.weights[i],
                    core.mu[i],
                    core.dmu_dq[i],
                    core.d2mu_dq2[i],
                    core.d3mu_dq3[i],
                );
                binomial_expected_location_scale_second_coefficients(
                    q,
                    f,
                    f1,
                    f2,
                    d_eta_t_u[i],
                    d_eta_ls_u[i],
                    d_eta_t_v[i],
                    d_eta_ls_v[i],
                )
            })
            .collect();
        let mut coeff_tt = Array1::<f64>::zeros(n);
        let mut coeff_tl = Array1::<f64>::zeros(n);
        let mut coeff_ll = Array1::<f64>::zeros(n);
        for (i, (tt, tl, ll)) in rows.into_iter().enumerate() {
            coeff_tt[i] = tt;
            coeff_tl[i] = tl;
            coeff_ll[i] = ll;
        }
        let d2_h_tt = xt_diag_x_dense(x_t, &coeff_tt)?;
        let d2_h_tl = xt_diag_y_dense(x_t, &coeff_tl, x_ls)?;
        let d2_h_ll = xt_diag_x_dense(x_ls, &coeff_ll)?;
        let mut d2_h = Array2::<f64>::zeros((total, total));
        d2_h.slice_mut(s![0..pt, 0..pt]).assign(&d2_h_tt);
        d2_h.slice_mut(s![0..pt, pt..total]).assign(&d2_h_tl);
        d2_h.slice_mut(s![pt..total, pt..total]).assign(&d2_h_ll);
        mirror_upper_to_lower(&mut d2_h);
        Ok(Some(d2_h))
    }

    fn expected_joint_contracted_trace_hessian_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
        trace_weight: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n
            || eta_ls.len() != n
            || self.weights.len() != n
            || x_t.nrows() != n
            || x_ls.nrows() != n
        {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily expected contracted trace input size mismatch"
                    .to_string(),
            }
            .into());
        }
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        if trace_weight.dim() != (total, total) {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expected contracted trace weight shape {:?} == ({total}, {total})",
                    trace_weight.dim()
                ),
            }
            .into());
        }
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let rows: Vec<(f64, f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut trace_tt = 0.0;
                for a in 0..pt {
                    for b in 0..pt {
                        trace_tt += x_t[[i, a]] * trace_weight[[a, b]] * x_t[[i, b]];
                    }
                }
                let mut trace_tl = 0.0;
                for a in 0..pt {
                    for b in 0..pls {
                        trace_tl += x_t[[i, a]]
                            * (trace_weight[[a, pt + b]] + trace_weight[[pt + b, a]])
                            * x_ls[[i, b]];
                    }
                }
                let mut trace_ll = 0.0;
                for a in 0..pls {
                    for b in 0..pls {
                        trace_ll += x_ls[[i, a]] * trace_weight[[pt + a, pt + b]] * x_ls[[i, b]];
                    }
                }
                let q = nonwiggle_q_derivs(eta_t[i], core.sigma[i]);
                let (f, f1, f2) = binomial_expected_q_information_derivatives(
                    self.weights[i],
                    core.mu[i],
                    core.dmu_dq[i],
                    core.d2mu_dq2[i],
                    core.d3mu_dq3[i],
                );
                let (tt_tt, tt_tl, tt_ll) = binomial_expected_location_scale_second_coefficients(
                    q, f, f1, f2, 1.0, 0.0, 1.0, 0.0,
                );
                let (tl_tt, tl_tl, tl_ll) = binomial_expected_location_scale_second_coefficients(
                    q, f, f1, f2, 1.0, 0.0, 0.0, 1.0,
                );
                let (ll_tt, ll_tl, ll_ll) = binomial_expected_location_scale_second_coefficients(
                    q, f, f1, f2, 0.0, 1.0, 0.0, 1.0,
                );
                (
                    trace_tt * tt_tt + trace_tl * tt_tl + trace_ll * tt_ll,
                    trace_tt * tl_tt + trace_tl * tl_tl + trace_ll * tl_ll,
                    trace_tt * ll_tt + trace_tl * ll_tl + trace_ll * ll_ll,
                )
            })
            .collect();
        let mut coeff_tt = Array1::<f64>::zeros(n);
        let mut coeff_tl = Array1::<f64>::zeros(n);
        let mut coeff_ll = Array1::<f64>::zeros(n);
        for (i, (tt, tl, ll)) in rows.into_iter().enumerate() {
            coeff_tt[i] = tt;
            coeff_tl[i] = tl;
            coeff_ll[i] = ll;
        }
        let h_tt = xt_diag_x_dense(x_t, &coeff_tt)?;
        let h_tl = xt_diag_y_dense(x_t, &coeff_tl, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &coeff_ll)?;
        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..pt, 0..pt]).assign(&h_tt);
        h.slice_mut(s![0..pt, pt..total]).assign(&h_tl);
        h.slice_mut(s![pt..total, pt..total]).assign(&h_ll);
        mirror_upper_to_lower(&mut h);
        Ok(Some(h))
    }

    fn expected_joint_information_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.expected_joint_information_from_designs(block_states, &x_t, &x_ls)
    }

    fn expected_joint_information_directional_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.expected_joint_information_directional_from_designs(
            block_states,
            &x_t,
            &x_ls,
            d_beta_flat,
        )
    }

    fn expected_joint_information_second_directional_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.expected_joint_information_second_directional_from_designs(
            block_states,
            &x_t,
            &x_ls,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn expected_joint_contracted_trace_hessian_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: Option<&[ParameterBlockSpec]>,
        trace_weight: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        self.expected_joint_contracted_trace_hessian_from_designs(
            block_states,
            &x_t,
            &x_ls,
            trace_weight,
        )
    }

    fn exact_newton_joint_psi_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psi_terms_from_designs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            &x_t,
            &x_ls,
        )
    }

    fn exact_newton_joint_psisecond_order_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psisecond_order_terms_from_designs(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            &x_t,
            &x_ls,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psihessian_directional_derivative_from_designs(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            &x_t,
            &x_ls,
        )
    }

    /// Compute the rowwise joint curvature coefficients (D_tt, D_tl, D_ll)
    /// shared by the dense joint Hessian path and the matrix-free workspace.
    fn exact_newton_joint_hessian_row_coefficients(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let mut coeff_tt = vec![0.0_f64; n];
        let mut coeff_tl = vec![0.0_f64; n];
        let mut coeff_ll = vec![0.0_f64; n];
        let y_slice = self.y.as_slice().expect("y must be contiguous");
        let w_slice = self.weights.as_slice().expect("weights must be contiguous");
        let q0_slice = core.q0.as_slice().expect("q0 must be contiguous");
        let sigma_slice = core.sigma.as_slice().expect("sigma must be contiguous");
        let dsigma_slice = core
            .dsigma_deta
            .as_slice()
            .expect("dsigma_deta must be contiguous");
        let mu_slice = core.mu.as_slice().expect("mu must be contiguous");
        let dmu_slice = core.dmu_dq.as_slice().expect("dmu_dq must be contiguous");
        let d2mu_slice = core
            .d2mu_dq2
            .as_slice()
            .expect("d2mu_dq2 must be contiguous");
        let d3mu_slice = core
            .d3mu_dq3
            .as_slice()
            .expect("d3mu_dq3 must be contiguous");
        let link_kind = &self.link_kind;
        coeff_tt
            .par_iter_mut()
            .zip(coeff_tl.par_iter_mut())
            .zip(coeff_ll.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((c_tt, c_tl), c_ll))| {
                let q = q0_slice[i];
                let r = 1.0 / sigma_slice[i];
                let kappa = dsigma_slice[i] / sigma_slice[i];
                let (m1, m2, _) = binomial_neglog_q_derivatives_dispatch(
                    y_slice[i],
                    w_slice[i],
                    q,
                    mu_slice[i],
                    dmu_slice[i],
                    d2mu_slice[i],
                    d3mu_slice[i],
                    link_kind,
                );
                *c_tt = m2 * r * r;
                *c_tl = kappa * r * (m1 + q * m2);
                *c_ll = kappa * kappa * q * (m1 + q * m2);
            });
        Ok((
            Array1::from_vec(coeff_tt),
            Array1::from_vec(coeff_tl),
            Array1::from_vec(coeff_ll),
        ))
    }

    /// Exact diagonal-block-only Hessians (h_tt, h_ll) used by `evaluate()`
    /// to populate per-block working sets without ever materializing the
    /// dense p×p joint matrix.
    fn exact_newton_block_diagonal_hessians_from_design_matrices(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &DesignMatrix,
        x_ls: &DesignMatrix,
    ) -> Result<(Array2<f64>, Array2<f64>), String> {
        let (coeff_tt, _coeff_tl, coeff_ll) =
            self.exact_newton_joint_hessian_row_coefficients(block_states)?;
        let h_tt = xt_diag_x_design(x_t, &coeff_tt)?;
        let h_ll = xt_diag_x_design(x_ls, &coeff_ll)?;
        Ok((h_tt, h_ll))
    }

    fn exact_newton_joint_hessian_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Exact joint coefficient-space Hessian for the probit, non-wiggle
        // location-scale family.
        //
        // At the fitted mode, the correct joint outer smoothing sensitivity is
        //
        //   H u_k = -g_k,
        //   g_k = A_k beta,
        //
        // so the solve must use the full joint working-curvature matrix `H`.
        // For this family the likelihood is coupled through
        //
        //   q = -eta_t * exp(-eta_ls),
        //
        // so the threshold and log-sigma blocks are not independent even if
        // the penalties are block-diagonal.
        //
        // Write for row i
        //
        //   t_i = x_i^T beta_t,
        //   s_i = z_i^T beta_ls,
        //   r_i = exp(-s_i),
        //   q_i = -t_i r_i,
        //   F_i(q) = -w_i [ y_i log Phi(q) + (1-y_i) log(1-Phi(q)) ].
        //
        // Let
        //
        //   m1_i = F_i'(q_i),
        //   m2_i = F_i''(q_i).
        //
        // The q-derivatives with respect to the two predictors are
        //
        //   q_t  = -r,
        //   q_ls = -q,
        //   q_tt = 0,
        //   q_t,ls = r,
        //   q_ls,ls = q.
        //
        // For any scalar-composition objective G(t,s)=F(q(t,s)), the Hessian
        // coefficients are
        //
        //   G_ab = m2 q_a q_b + m1 q_ab.
        //
        // Therefore the exact rowwise joint curvature in (eta_t, eta_ls) is
        //
        //   coeff_tt = m2 r^2,
        //   coeff_t,ls = r (m1 + q m2),
        //   coeff_ls,ls = q (m1 + q m2),
        //
        // and the full joint coefficient-space Hessian is assembled as
        //
        //   H_tt    = X_t^T diag(coeff_tt)    X_t,
        //   H_t,ls  = X_t^T diag(coeff_t,ls)  X_ls,
        //   H_ls,ls = X_ls^T diag(coeff_ls,ls) X_ls.
        //
        // The off-diagonal block is generally nonzero. That is exactly the
        // coupling term the broken blockwise outer-gradient path was dropping.
        let (coeff_tt, coeff_tl, coeff_ll) =
            self.exact_newton_joint_hessian_row_coefficients(block_states)?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();

        let h_tt = xt_diag_x_dense(x_t, &coeff_tt)?;
        let h_tl = xt_diag_y_dense(x_t, &coeff_tl, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &coeff_ll)?;
        let total = pt + pls;
        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..pt, 0..pt]).assign(&h_tt);
        h.slice_mut(s![0..pt, pt..total]).assign(&h_tl);
        h.slice_mut(s![pt..total, pt..total]).assign(&h_ll);
        mirror_upper_to_lower(&mut h);
        Ok(Some(h))
    }

    fn exact_newton_joint_hessian_from_design_matrices(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &DesignMatrix,
        x_ls: &DesignMatrix,
    ) -> Result<Option<Array2<f64>>, String> {
        if let (Some(x_t_dense), Some(x_ls_dense)) = (x_t.as_dense_ref(), x_ls.as_dense_ref()) {
            return self.exact_newton_joint_hessian_from_designs(
                block_states,
                x_t_dense,
                x_ls_dense,
            );
        }
        let (coeff_tt, coeff_tl, coeff_ll) =
            self.exact_newton_joint_hessian_row_coefficients(block_states)?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();

        let h_tt = xt_diag_x_design(x_t, &coeff_tt)?;
        let h_tl = xt_diag_y_design(x_t, &coeff_tl, x_ls)?;
        let h_ll = xt_diag_x_design(x_ls, &coeff_ll)?;
        let total = pt + pls;
        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..pt, 0..pt]).assign(&h_tt);
        h.slice_mut(s![0..pt, pt..total]).assign(&h_tl);
        h.slice_mut(s![pt..total, pt..total]).assign(&h_ll);
        mirror_upper_to_lower(&mut h);
        Ok(Some(h))
    }

    fn exact_newton_joint_hessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Exact first directional derivative D_beta H_L[u] of the joint
        // likelihood curvature.
        //
        // Write
        //
        //   t  = X_t beta_t,
        //   ls = X_ls beta_ls,
        //   s  = exp(-ls),
        //   q  = -t .* s.
        //
        // For a full coefficient-space direction
        //
        //   u = (u_t, u_ls),
        //   xi_t  = X_t u_t,
        //   xi_ls = X_ls u_ls,
        //
        // the induced q-direction is
        //
        //   alpha = D q[u] = -s .* xi_t - q .* xi_ls.
        //
        // The joint diagonal-working-curvature likelihood matrix is
        //
        //   H_L = J^T W J,
        //   J_t  = -diag(s) X_t,
        //   J_ls = -diag(q) X_ls.
        //
        // Differentiating once gives
        //
        //   D_beta H_L[u]
        //   = K[u]^T W J
        //     + J^T W K[u]
        //     + J^T diag(nu .* alpha) J,
        //
        // where
        //
        //   K_t[u]  = diag(s .* xi_ls) X_t,
        //   K_ls[u] = diag(s .* xi_t + q .* xi_ls) X_ls,
        //
        // and `nu = d'''(q)` is the third derivative of the scalar row loss.
        // This is exactly the joint curvature drift that enters the profiled
        // derivative through
        //
        //   dot H_k = A_k + D_beta H_L[u_k],
        //   dJ/drho_k
        //   = 0.5 beta^T A_k beta
        //     + 0.5 tr(H^{-1} dot H_k)
        //     - 0.5 tr(S^+ A_k).
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        if d_beta_flat.len() != pt + pls {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily joint d_beta length mismatch: got {}, expected {}",
                    d_beta_flat.len(),
                    pt + pls
                ),
            }
            .into());
        }
        let d_eta_t = fast_av(x_t, &d_beta_flat.slice(s![0..pt]));
        let d_eta_ls = fast_av(x_ls, &d_beta_flat.slice(s![pt..pt + pls]));
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let (coeff_tt, coeff_tl, coeff_ll) =
            binomial_location_scale_first_directional_coefficients(
                &self.y,
                &self.weights,
                &core,
                &d_eta_t,
                &d_eta_ls,
                &self.link_kind,
            )?;

        let d_h_tt = xt_diag_x_dense(x_t, &coeff_tt)?;
        let d_h_tl = xt_diag_y_dense(x_t, &coeff_tl, x_ls)?;
        let d_h_ll = xt_diag_x_dense(x_ls, &coeff_ll)?;
        let total = pt + pls;
        let mut d_h = Array2::<f64>::zeros((total, total));
        d_h.slice_mut(s![0..pt, 0..pt]).assign(&d_h_tt);
        d_h.slice_mut(s![0..pt, pt..total]).assign(&d_h_tl);
        d_h.slice_mut(s![pt..total, pt..total]).assign(&d_h_ll);
        mirror_upper_to_lower(&mut d_h);
        Ok(Some(d_h))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Exact mixed second directional derivative D_beta^2 H_L[u, v].
        //
        // This is the family-specific part of the total second curvature drift
        //
        //   ddot H_{k,l}
        //   = B_{k,l}
        //     + D_beta H_L[u_{k,l}]
        //     + D_beta^2 H_L[u_l, u_k],
        //
        // used in the profiled outer Hessian
        //
        //   d^2J/(drho_k drho_l)
        //   = u_l^T A_k beta
        //     + 0.5 beta^T B_{k,l} beta
        //     + 0.5 tr(H^{-1} ddot H_{k,l})
        //     - 0.5 tr(H^{-1} dot H_l H^{-1} dot H_k)
        //     - 0.5 d^2/drho_k drho_l log|S|_+.
        //
        // For directions
        //
        //   u = (u_t, u_ls),  v = (v_t, v_ls),
        //
        // define the rowwise predictor perturbations
        //
        //   xi_t^(u)  = X_t u_t,    xi_ls^(u)  = X_ls u_ls,
        //   xi_t^(v)  = X_t v_t,    xi_ls^(v)  = X_ls v_ls.
        //
        // With the exact exp sigma link,
        //
        //   s = exp(-eta_ls),
        //   q = -eta_t .* s,
        //
        // the first and second q-drifts are
        //
        //   alpha(u)   = D q[u]   = -s .* xi_t^(u) - q .* xi_ls^(u),
        //   alpha(v)   = D q[v]   = -s .* xi_t^(v) - q .* xi_ls^(v),
        //   alpha(u,v) = D^2 q[u,v]
        //              = s .* (xi_t^(u) .* xi_ls^(v) + xi_t^(v) .* xi_ls^(u))
        //                + q .* xi_ls^(u) .* xi_ls^(v).
        //
        // Differentiating the scalar-composition Hessian coefficients twice
        // yields the rowwise formulas below. Those formulas are exactly the
        // fourth-order beta-curvature contraction needed to make the joint
        // rho-Hessian path consistent with the first-order joint solve.
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        if d_beta_u_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialLocationScaleFamily joint d_beta_u length mismatch: got {}, expected {}",
                d_beta_u_flat.len(),
                total
            ) }.into());
        }
        if d_betav_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialLocationScaleFamily joint d_betav length mismatch: got {}, expected {}",
                d_betav_flat.len(),
                total
            ) }.into());
        }
        let d_eta_t_u = fast_av(x_t, &d_beta_u_flat.slice(s![0..pt]));
        let d_eta_ls_u = fast_av(x_ls, &d_beta_u_flat.slice(s![pt..total]));
        let d_eta_tv = fast_av(x_t, &d_betav_flat.slice(s![0..pt]));
        let d_eta_lsv = fast_av(x_ls, &d_betav_flat.slice(s![pt..total]));
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let (coeff_tt, coeff_tl, coeff_ll) =
            binomial_location_scalesecond_directional_coefficients(
                &self.y,
                &self.weights,
                &core,
                &d_eta_t_u,
                &d_eta_ls_u,
                &d_eta_tv,
                &d_eta_lsv,
                &self.link_kind,
            )?;

        let d2_h_tt = xt_diag_x_dense(x_t, &coeff_tt)?;
        let d2_h_tl = xt_diag_y_dense(x_t, &coeff_tl, x_ls)?;
        let d2_h_ll = xt_diag_x_dense(x_ls, &coeff_ll)?;
        let mut d2_h = Array2::<f64>::zeros((total, total));
        d2_h.slice_mut(s![0..pt, 0..pt]).assign(&d2_h_tt);
        d2_h.slice_mut(s![0..pt, pt..total]).assign(&d2_h_tl);
        d2_h.slice_mut(s![pt..total, pt..total]).assign(&d2_h_ll);
        mirror_upper_to_lower(&mut d2_h);
        Ok(Some(d2_h))
    }

    fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
        policy: &crate::resource::ResourcePolicy,
    ) -> Result<Option<LocationScaleJointPsiDirection>, String> {
        let Some(parts) = locscale_joint_psi_direction_parts(
            block_states,
            derivative_blocks,
            psi_index,
            self.y.len(),
            x_t.ncols(),
            x_ls.ncols(),
            Self::BLOCK_T,
            Self::BLOCK_LOG_SIGMA,
            2,
            "BinomialLocationScaleFamily",
            "threshold",
            policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(LocationScaleJointPsiDirection {
            block_idx: parts.block_idx,
            local_idx: parts.local_idx,
            x_primary_psi: parts.primary_psi,
            x_ls_psi: parts.log_sigma_psi,
            z_primary_psi: parts.primary_z,
            z_ls_psi: parts.log_sigma_z,
        }))
    }

    fn exact_newton_joint_psisecond_design_drifts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &LocationScaleJointPsiDirection,
        psi_b: &LocationScaleJointPsiDirection,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<LocationScaleJointPsiSecondDrifts, String> {
        locscale_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            psi_a,
            psi_b,
            LocScalePsiDriftConfig {
                n: self.y.len(),
                p_primary: x_t.ncols(),
                p_log_sigma: x_ls.ncols(),
                primary_block_idx: Self::BLOCK_T,
                log_sigma_block_idx: Self::BLOCK_LOG_SIGMA,
                family_name: "BinomialLocationScaleFamily",
                primary_label: "threshold",
                policy: &self.policy,
            },
        )
    }

    fn exact_newton_joint_psi_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        if specs.len() != 2 || derivative_blocks.len() != 2 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialLocationScaleFamily joint psi terms expect 2 specs and 2 derivative blocks, got {} and {}",
                specs.len(),
                derivative_blocks.len()
            ) }.into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        // Joint fixed-beta psi terms for the coupled 2-block probit model.
        //
        // We work over the flattened coefficient vector beta = [beta_t; beta_ls]
        // and one realized spatial coordinate psi_a. The exact profiled/Laplace
        // outer calculus needs the family-side explicit objects
        //
        //   V_psi^explicit,  g_psi^explicit,  H_psi^explicit,
        //
        // all in this flattened coefficient space. These are likelihood-only
        // objects:
        //
        //   D_psi, D_{beta psi}, D_{beta beta psi}
        //
        // Generic exact-joint code adds the realized penalty motion
        //
        //   0.5 beta^T S_psi beta,  S_psi beta,  S_psi
        //
        // when forming V_i, g_i, H_i. Keeping the family hook likelihood-only
        // is what makes the unified S(theta) outer calculus correct for both
        // psi-moving designs and psi-moving penalties.
        //
        // Model:
        //   eta_t  = X_t beta_t,
        //   eta_ls = X_ls beta_ls,
        //   r      = exp(-eta_ls),
        //   q      = -eta_t .* r.
        //
        // A single realized psi_a may move either block design, so define the
        // fixed-beta predictor drifts
        //
        //   z_t  = X_{t,psi}  beta_t   (zero if psi_a is not a threshold psi)
        //   z_ls = X_{ls,psi} beta_ls  (zero if psi_a is not a log-sigma psi).
        //
        // Then the explicit q-drift is
        //
        //   q_psi = -r .* z_t - q .* z_ls.
        //
        // Rowwise scalar derivatives of the negative Bernoulli-probit loss are
        //
        //   a = dF/dq,
        //   b = d²F/dq²,
        //   c = d³F/dq³.
        //
        // Predictor-space score pieces:
        //
        //   r_t  = dF/deta_t  = -a r,
        //   r_ls = dF/deta_ls = -a q.
        //
        // Their explicit psi derivatives at fixed beta are
        //
        //   d_psi r_t  = -b q_psi r + a r z_ls,
        //   d_psi r_ls = -(a + q b) q_psi.
        //
        // Hence the exact joint score derivative is
        //
        //   g_psi
        //   = [ X_{t,psi}^T r_t  + X_t^T d_psi r_t,
        //       X_{ls,psi}^T r_ls + X_ls^T d_psi r_ls ].
        //
        // The exact envelope term is
        //
        //   V_psi^explicit = r_t^T z_t + r_ls^T z_ls.
        //
        // For the Laplace trace we also need the explicit Hessian drift. The
        // joint exact Hessian has block coefficients
        //
        //   h_tt = b r²,
        //   h_tl = r (a + q b),
        //   h_ll = q (a + q b),
        //
        // so differentiating those coefficients at fixed beta gives
        //
        //   d_psi h_tt = r² (c q_psi - 2 b z_ls),
        //   d_psi h_tl = r [ (2 b + c q) q_psi - (a + q b) z_ls ],
        //   d_psi h_ll = (a + 3 q b + q² c) q_psi.
        //
        // The full joint explicit Hessian drift is then
        //
        //   H_tt,psi
        //   = X_{t,psi}^T diag(h_tt) X_t
        //     + X_t^T diag(h_tt) X_{t,psi}
        //     + X_t^T diag(d_psi h_tt) X_t,
        //
        //   H_tl,psi
        //   = X_{t,psi}^T diag(h_tl) X_ls
        //     + X_t^T diag(h_tl) X_{ls,psi}
        //     + X_t^T diag(d_psi h_tl) X_ls,
        //
        //   H_ll,psi
        //   = X_{ls,psi}^T diag(h_ll) X_ls
        //     + X_ls^T diag(h_ll) X_{ls,psi}
        //     + X_ls^T diag(d_psi h_ll) X_ls.
        //
        // Even when only one block moves explicitly, the resulting score and
        // Hessian objects are joint because q couples eta_t and eta_ls.
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let (z_t, z_ls) = (&dir_a.z_primary_psi, &dir_a.z_ls_psi);

        // Per-row scalars assembled in parallel. The probit/inverse-link
        // derivatives are O(n) at large scale and are called O(K) times per
        // outer REML gradient (K = number of psi coords), so a parallel pass is
        // worthwhile here.
        struct PsiTermsRow {
            r_t: f64,
            r_ls: f64,
            dr_t: f64,
            dr_ls: f64,
            h_tt: f64,
            h_tl: f64,
            h_ll: f64,
            dh_tt: f64,
            dh_tl: f64,
            dh_ll: f64,
            obj: f64,
        }
        let y_p = self.y.as_slice().expect("y must be contiguous");
        let w_p = self.weights.as_slice().expect("weights must be contiguous");
        let q0_p = core.q0.as_slice().expect("q0 must be contiguous");
        let sigma_p = core.sigma.as_slice().expect("sigma must be contiguous");
        let dsigma_p = core
            .dsigma_deta
            .as_slice()
            .expect("dsigma_deta must be contiguous");
        let mu_p = core.mu.as_slice().expect("mu must be contiguous");
        let dmu_p = core.dmu_dq.as_slice().expect("dmu_dq must be contiguous");
        let d2mu_p = core
            .d2mu_dq2
            .as_slice()
            .expect("d2mu_dq2 must be contiguous");
        let d3mu_p = core
            .d3mu_dq3
            .as_slice()
            .expect("d3mu_dq3 must be contiguous");
        let z_t_p = z_t.as_slice().expect("z_t must be contiguous");
        let z_ls_p = z_ls.as_slice().expect("z_ls must be contiguous");
        let link_kind_p = &self.link_kind;
        let rows: Vec<PsiTermsRow> = (0..n)
            .into_par_iter()
            .map(|i| {
                let q = q0_p[i];
                let r = 1.0 / sigma_p[i];
                let s = dsigma_p[i] / sigma_p[i];
                let sz = s * z_ls_p[i];
                let q_psi = -r * z_t_p[i] - q * sz;
                let (a, b, c) = binomial_neglog_q_derivatives_dispatch(
                    y_p[i],
                    w_p[i],
                    q,
                    mu_p[i],
                    dmu_p[i],
                    d2mu_p[i],
                    d3mu_p[i],
                    link_kind_p,
                );
                let r_t = -a * r;
                let r_ls = -a * q * s;
                PsiTermsRow {
                    r_t,
                    r_ls,
                    dr_t: -b * q_psi * r + a * r * sz,
                    dr_ls: -(a + q * b) * q_psi,
                    h_tt: b * r * r,
                    h_tl: r * (a + q * b),
                    h_ll: q * (a + q * b),
                    dh_tt: r * r * (c * q_psi - 2.0 * b * sz),
                    dh_tl: r * ((2.0 * b + c * q) * q_psi - (a + q * b) * sz),
                    dh_ll: (a + 3.0 * q * b + q * q * c) * q_psi,
                    obj: r_t * z_t_p[i] + r_ls * z_ls_p[i],
                }
            })
            .collect();
        let mut r_t = Array1::<f64>::zeros(n);
        let mut r_ls = Array1::<f64>::zeros(n);
        let mut dr_t = Array1::<f64>::zeros(n);
        let mut dr_ls = Array1::<f64>::zeros(n);
        let mut h_tt = Array1::<f64>::zeros(n);
        let mut h_tl = Array1::<f64>::zeros(n);
        let mut h_ll = Array1::<f64>::zeros(n);
        let mut dh_tt = Array1::<f64>::zeros(n);
        let mut dh_tl = Array1::<f64>::zeros(n);
        let mut dh_ll = Array1::<f64>::zeros(n);
        let mut objective_psi = 0.0_f64;
        for (i, row) in rows.into_iter().enumerate() {
            r_t[i] = row.r_t;
            r_ls[i] = row.r_ls;
            dr_t[i] = row.dr_t;
            dr_ls[i] = row.dr_ls;
            h_tt[i] = row.h_tt;
            h_tl[i] = row.h_tl;
            h_ll[i] = row.h_ll;
            dh_tt[i] = row.dh_tt;
            dh_tl[i] = row.dh_tl;
            dh_ll[i] = row.dh_ll;
            objective_psi += row.obj;
        }

        let hessian_psi_operator = build_two_block_custom_family_joint_psi_operator_from_actions(
            dir_a.x_primary_psi.cloned_first_action(),
            dir_a.x_ls_psi.cloned_first_action(),
            0..pt,
            pt..pt + pls,
            x_t,
            x_ls,
            &h_tt,
            &h_tl,
            &h_ll,
            &dh_tt,
            &dh_tl,
            &dh_ll,
        )?;
        let x_t_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let score_t = x_t_map.transpose_mul(r_t.view()) + fast_atv(x_t, &dr_t);
        let score_ls = x_ls_map.transpose_mul(r_ls.view()) + fast_atv(x_ls, &dr_ls);
        let mut score_psi = Array1::<f64>::zeros(total);
        score_psi.slice_mut(s![0..pt]).assign(&score_t);
        score_psi.slice_mut(s![pt..pt + pls]).assign(&score_ls);
        let hessian_psi = if hessian_psi_operator.is_some() {
            Array2::zeros((0, 0))
        } else {
            let h_tt_block = weighted_crossprod_psi_maps(
                x_t_map,
                h_tt.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_t),
            )? + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                h_tt.view(),
                x_t_map,
            )? + &xt_diag_x_dense(x_t, &dh_tt)?;
            let h_tl_block = weighted_crossprod_psi_maps(
                x_t_map,
                h_tl.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )? + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                h_tl.view(),
                x_ls_map,
            )? + &xt_diag_y_dense(x_t, &dh_tl, x_ls)?;
            let h_ll_block = weighted_crossprod_psi_maps(
                x_ls_map,
                h_ll.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )? + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
                h_ll.view(),
                x_ls_map,
            )? + &xt_diag_x_dense(x_ls, &dh_ll)?;

            let mut hessian_psi = Array2::<f64>::zeros((total, total));
            hessian_psi.slice_mut(s![0..pt, 0..pt]).assign(&h_tt_block);
            hessian_psi
                .slice_mut(s![0..pt, pt..pt + pls])
                .assign(&h_tl_block);
            hessian_psi
                .slice_mut(s![pt..pt + pls, pt..pt + pls])
                .assign(&h_ll_block);
            mirror_upper_to_lower(&mut hessian_psi);
            hessian_psi
        };

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator,
        }))
    }

    fn exact_newton_joint_psisecond_order_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some(dir_i) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_i,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let Some(dir_j) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_j,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psisecond_order_terms_from_parts(
                block_states,
                derivative_blocks,
                &dir_i,
                &dir_j,
                x_t,
                x_ls,
            )?,
        ))
    }

    fn exact_newton_joint_psisecond_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        dir_i: &LocationScaleJointPsiDirection,
        dir_j: &LocationScaleJointPsiDirection,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms, String> {
        let second_drifts = self.exact_newton_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            dir_i,
            dir_j,
            x_t,
            x_ls,
        )?;
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        let x_t_i_map = dir_i.x_primary_psi.as_linear_map_ref();
        let x_t_j_map = dir_j.x_primary_psi.as_linear_map_ref();
        let x_ls_i_map = dir_i.x_ls_psi.as_linear_map_ref();
        let x_ls_j_map = dir_j.x_ls_psi.as_linear_map_ref();
        let x_t_ab_map = second_psi_linear_map(
            second_drifts.x_primary_ab_action.as_ref(),
            second_drifts.x_primary_ab.as_ref(),
            n,
            pt,
        );
        let x_ls_ab_map = second_psi_linear_map(
            second_drifts.x_ls_ab_action.as_ref(),
            second_drifts.x_ls_ab.as_ref(),
            n,
            pls,
        );

        // Exact fixed-beta psi/psi terms for the coupled non-wiggle probit
        // family.
        //
        // For two realized spatial coordinates psi_a, psi_b define
        //
        //   z_t,a  = X_{t,a} beta_t,    z_ls,a  = X_{ls,a} beta_ls,
        //   z_t,b  = X_{t,b} beta_t,    z_ls,b  = X_{ls,b} beta_ls,
        //   z_t,ab = X_{t,ab} beta_t,   z_ls,ab = X_{ls,ab} beta_ls.
        //
        // On the smooth interior branch, with r = exp(-eta_ls) and q = -eta_t r,
        //
        //   q_a  = -r z_t,a - q z_ls,a,
        //   q_b  = -r z_t,b - q z_ls,b,
        //   q_ab = -r z_t,ab
        //          + r(z_t,a z_ls,b + z_t,b z_ls,a)
        //          + q(z_ls,a z_ls,b - z_ls,ab).
        //
        // For scalar row loss derivatives
        //
        //   a = dF/dq,  b = d²F/dq²,  c = d³F/dq³,  d = d⁴F/dq⁴,
        //
        // the exact fixed-beta psi/psi objects are
        //
        //   V_ab = sum [ a q_ab + b q_a q_b ],
        //
        //   g_ab = [ X_{t,ab}^T r_t + X_{t,a}^T d_b r_t + X_{t,b}^T d_a r_t + X_t^T d_ab r_t,
        //            X_{ls,ab}^T r_ls + X_{ls,a}^T d_b r_ls + X_{ls,b}^T d_a r_ls + X_ls^T d_ab r_ls ],
        //
        // where
        //
        //   r_t  = -a r,
        //   r_ls = -a q,
        //
        //   d_a r_t  = -b q_a r + a r z_ls,a,
        //   d_a r_ls = -(a + q b) q_a,
        //
        //   d_ab r_t
        //   = r[
        //       -c q_a q_b - b q_ab
        //       + b(q_a z_ls,b + q_b z_ls,a)
        //       - a z_ls,a z_ls,b
        //       + a z_ls,ab
        //     ],
        //
        //   d_ab r_ls
        //   = -[(2b + q c) q_a q_b + (a + q b) q_ab].
        //
        // The exact Hessian psi/psi drift comes from the second derivatives of
        // the joint Hessian coefficients. In the notation of the unified outer
        // calculus, these rowwise coefficient drifts are precisely the
        // likelihood-side pieces of
        //
        //   D_{beta beta psi_a psi_b},
        //
        // before the generic assembler adds any realized-penalty contribution
        //
        //   S_ab = partial_{psi_a psi_b} S(theta).
        //
        // So this helper returns likelihood-only
        //
        //   D_ab, D_{beta ab}, D_{beta beta ab},
        //
        // and the unified exact assembler in custom_family.rs forms
        //
        //   V_ab = D_ab + 0.5 beta^T S_ab beta,
        //   g_ab = D_{beta ab} + S_ab beta,
        //   H_ab = D_{beta beta ab} + S_ab.
        //
        // Once H_ab is known, the outer assembler combines it with the joint
        // mode responses beta_a, beta_b, beta_ab and the contractions
        //
        //   T_a[beta_b], T_b[beta_a], D_beta H[beta_ab], D_beta^2 H[beta_a, beta_b]
        //
        // to form
        //
        //   ddot H_ab
        //   = H_ab + T_a[beta_b] + T_b[beta_a]
        //     + D_beta H[beta_ab] + D_beta^2 H[beta_a, beta_b].
        //
        // That is why this helper computes only the fixed-beta psi/psi object:
        // the total profiled/Laplace Hessian drift is assembled generically in
        // custom_family.rs after the joint solves.
        //
        // Concretely, the rowwise coefficient identities below are
        //
        //   h_tt = b r²,
        //   h_tl = r(a + q b),
        //   h_ll = q(a + q b),
        //
        // namely
        //
        //   d_ab h_tt
        //   = r²[
        //       d q_a q_b + c q_ab
        //       - 2c(q_b z_ls,a + q_a z_ls,b)
        //       + 4b z_ls,a z_ls,b
        //       - 2b z_ls,ab
        //     ],
        //
        //   d_ab h_tl
        //   = r[
        //       ((3c + q d) q_b) q_a
        //       + (2b + q c) q_ab
        //       - (2b + q c)(q_b z_ls,a + q_a z_ls,b)
        //       + (a + q b)(z_ls,a z_ls,b - z_ls,ab)
        //     ],
        //
        //   d_ab h_ll
        //   = (4b + 5q c + q² d) q_a q_b
        //     + (a + 3q b + q² c) q_ab.
        //
        // Differentiating X^T diag(h) X twice then gives the explicit joint
        // psi/psi Hessian blocks.
        let mut r_t = Array1::<f64>::zeros(n);
        let mut r_ls = Array1::<f64>::zeros(n);
        let mut dr_t_i = Array1::<f64>::zeros(n);
        let mut dr_t_j = Array1::<f64>::zeros(n);
        let mut dr_ls_i = Array1::<f64>::zeros(n);
        let mut dr_ls_j = Array1::<f64>::zeros(n);
        let mut d2r_t = Array1::<f64>::zeros(n);
        let mut d2r_ls = Array1::<f64>::zeros(n);
        let mut h_tt = Array1::<f64>::zeros(n);
        let mut h_tl = Array1::<f64>::zeros(n);
        let mut h_ll = Array1::<f64>::zeros(n);
        let mut dh_tt_i = Array1::<f64>::zeros(n);
        let mut dh_tt_j = Array1::<f64>::zeros(n);
        let mut dh_tl_i = Array1::<f64>::zeros(n);
        let mut dh_tl_j = Array1::<f64>::zeros(n);
        let mut dh_ll_i = Array1::<f64>::zeros(n);
        let mut dh_ll_j = Array1::<f64>::zeros(n);
        let mut d2h_tt = Array1::<f64>::zeros(n);
        let mut d2h_tl = Array1::<f64>::zeros(n);
        let mut d2h_ll = Array1::<f64>::zeros(n);
        let mut objective_psi_psi = 0.0;
        struct PsiSecondRow {
            r_t: f64,
            r_ls: f64,
            dr_t_i: f64,
            dr_t_j: f64,
            dr_ls_i: f64,
            dr_ls_j: f64,
            d2r_t: f64,
            d2r_ls: f64,
            h_tt: f64,
            h_tl: f64,
            h_ll: f64,
            dh_tt_i: f64,
            dh_tt_j: f64,
            dh_tl_i: f64,
            dh_tl_j: f64,
            dh_ll_i: f64,
            dh_ll_j: f64,
            d2h_tt: f64,
            d2h_tl: f64,
            d2h_ll: f64,
            objective: f64,
        }
        let y_p = self.y.as_slice().expect("y must be contiguous");
        let w_p = self.weights.as_slice().expect("weights must be contiguous");
        let q_p = core.q0.as_slice().expect("q0 must be contiguous");
        let sigma_p = core.sigma.as_slice().expect("sigma must be contiguous");
        let mu_p = core.mu.as_slice().expect("mu must be contiguous");
        let dmu_p = core.dmu_dq.as_slice().expect("dmu_dq must be contiguous");
        let d2mu_p = core
            .d2mu_dq2
            .as_slice()
            .expect("d2mu_dq2 must be contiguous");
        let d3mu_p = core
            .d3mu_dq3
            .as_slice()
            .expect("d3mu_dq3 must be contiguous");
        let z_t_i = dir_i
            .z_primary_psi
            .as_slice()
            .expect("z_t_psi_i must be contiguous");
        let z_t_j = dir_j
            .z_primary_psi
            .as_slice()
            .expect("z_t_psi_j must be contiguous");
        let z_ls_i = dir_i
            .z_ls_psi
            .as_slice()
            .expect("z_ls_psi_i must be contiguous");
        let z_ls_j = dir_j
            .z_ls_psi
            .as_slice()
            .expect("z_ls_psi_j must be contiguous");
        let z_t_ab = second_drifts
            .z_primary_ab
            .as_slice()
            .expect("z_t_ab must be contiguous");
        let z_ls_ab = second_drifts
            .z_ls_ab
            .as_slice()
            .expect("z_ls_ab must be contiguous");
        let link_kind_p = &self.link_kind;
        let rows: Result<Vec<PsiSecondRow>, String> = (0..n)
            .into_par_iter()
            .map(|row| {
                let q = q_p[row];
                let r = 1.0 / sigma_p[row];
                let q_i = -r * z_t_i[row] - q * z_ls_i[row];
                let q_j = -r * z_t_j[row] - q * z_ls_j[row];
                let q_ij = -r * z_t_ab[row]
                    + r * (z_t_i[row] * z_ls_j[row] + z_t_j[row] * z_ls_i[row])
                    + q * (z_ls_i[row] * z_ls_j[row] - z_ls_ab[row]);
                let (a, b, c) = binomial_neglog_q_derivatives_dispatch(
                    y_p[row],
                    w_p[row],
                    q,
                    mu_p[row],
                    dmu_p[row],
                    d2mu_p[row],
                    d3mu_p[row],
                    link_kind_p,
                );
                let d = binomial_neglog_q_fourth_derivative_dispatch(
                    y_p[row],
                    w_p[row],
                    q,
                    mu_p[row],
                    dmu_p[row],
                    d2mu_p[row],
                    d3mu_p[row],
                    link_kind_p,
                )?;
                let u = a + q * b;
                let u_i = (2.0 * b + q * c) * q_i;
                let u_j = (2.0 * b + q * c) * q_j;
                Ok(PsiSecondRow {
                    r_t: -a * r,
                    r_ls: -a * q,
                    dr_t_i: -b * q_i * r + a * r * z_ls_i[row],
                    dr_t_j: -b * q_j * r + a * r * z_ls_j[row],
                    dr_ls_i: -u * q_i,
                    dr_ls_j: -u * q_j,
                    d2r_t: r
                        * (-c * q_i * q_j - b * q_ij + b * (q_i * z_ls_j[row] + q_j * z_ls_i[row])
                            - a * z_ls_i[row] * z_ls_j[row]
                            + a * z_ls_ab[row]),
                    d2r_ls: -((2.0 * b + q * c) * q_i * q_j + u * q_ij),
                    h_tt: b * r * r,
                    h_tl: r * u,
                    h_ll: q * u,
                    dh_tt_i: r * r * (c * q_i - 2.0 * b * z_ls_i[row]),
                    dh_tt_j: r * r * (c * q_j - 2.0 * b * z_ls_j[row]),
                    dh_tl_i: r * (u_i - u * z_ls_i[row]),
                    dh_tl_j: r * (u_j - u * z_ls_j[row]),
                    dh_ll_i: (a + 3.0 * q * b + q * q * c) * q_i,
                    dh_ll_j: (a + 3.0 * q * b + q * q * c) * q_j,
                    d2h_tt: r
                        * r
                        * (d * q_i * q_j + c * q_ij
                            - 2.0 * c * (q_j * z_ls_i[row] + q_i * z_ls_j[row])
                            + 4.0 * b * z_ls_i[row] * z_ls_j[row]
                            - 2.0 * b * z_ls_ab[row]),
                    d2h_tl: r
                        * (((3.0 * c + q * d) * q_j) * q_i + (2.0 * b + q * c) * q_ij
                            - (2.0 * b + q * c) * (q_j * z_ls_i[row] + q_i * z_ls_j[row])
                            + u * (z_ls_i[row] * z_ls_j[row] - z_ls_ab[row])),
                    d2h_ll: (4.0 * b + 5.0 * q * c + q * q * d) * q_i * q_j
                        + (a + 3.0 * q * b + q * q * c) * q_ij,
                    objective: a * q_ij + b * q_i * q_j,
                })
            })
            .collect();
        for (row, vals) in rows?.into_iter().enumerate() {
            r_t[row] = vals.r_t;
            r_ls[row] = vals.r_ls;
            dr_t_i[row] = vals.dr_t_i;
            dr_t_j[row] = vals.dr_t_j;
            dr_ls_i[row] = vals.dr_ls_i;
            dr_ls_j[row] = vals.dr_ls_j;
            d2r_t[row] = vals.d2r_t;
            d2r_ls[row] = vals.d2r_ls;
            h_tt[row] = vals.h_tt;
            h_tl[row] = vals.h_tl;
            h_ll[row] = vals.h_ll;
            dh_tt_i[row] = vals.dh_tt_i;
            dh_tt_j[row] = vals.dh_tt_j;
            dh_tl_i[row] = vals.dh_tl_i;
            dh_tl_j[row] = vals.dh_tl_j;
            dh_ll_i[row] = vals.dh_ll_i;
            dh_ll_j[row] = vals.dh_ll_j;
            d2h_tt[row] = vals.d2h_tt;
            d2h_tl[row] = vals.d2h_tl;
            d2h_ll[row] = vals.d2h_ll;
            objective_psi_psi += vals.objective;
        }
        let mut score_psi_psi = Array1::<f64>::zeros(total);
        score_psi_psi.slice_mut(s![0..pt]).assign(
            &(x_t_ab_map.transpose_mul(r_t.view())
                + x_t_i_map.transpose_mul(dr_t_j.view())
                + x_t_j_map.transpose_mul(dr_t_i.view())
                + fast_atv(x_t, &d2r_t)),
        );
        score_psi_psi.slice_mut(s![pt..pt + pls]).assign(
            &(x_ls_ab_map.transpose_mul(r_ls.view())
                + x_ls_i_map.transpose_mul(dr_ls_j.view())
                + x_ls_j_map.transpose_mul(dr_ls_i.view())
                + fast_atv(x_ls, &d2r_ls)),
        );

        let h_tt_block = weighted_crossprod_psi_maps(
            x_t_ab_map,
            h_tt.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_t),
        )? + &weighted_crossprod_psi_maps(x_t_i_map, h_tt.view(), x_t_j_map)?
            + &weighted_crossprod_psi_maps(x_t_j_map, h_tt.view(), x_t_i_map)?
            + &weighted_crossprod_psi_maps(
                x_t_i_map,
                dh_tt_j.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_t),
            )?
            + &weighted_crossprod_psi_maps(
                x_t_j_map,
                dh_tt_i.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_t),
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                dh_tt_i.view(),
                x_t_j_map,
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                dh_tt_j.view(),
                x_t_i_map,
            )?
            + &xt_diag_x_dense(x_t, &d2h_tt)?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                h_tt.view(),
                x_t_ab_map,
            )?;
        let h_tl_block = weighted_crossprod_psi_maps(
            x_t_ab_map,
            h_tl.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(x_t_i_map, h_tl.view(), x_ls_j_map)?
            + &weighted_crossprod_psi_maps(x_t_j_map, h_tl.view(), x_ls_i_map)?
            + &weighted_crossprod_psi_maps(
                x_t_i_map,
                dh_tl_j.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                x_t_j_map,
                dh_tl_i.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                dh_tl_i.view(),
                x_ls_j_map,
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                dh_tl_j.view(),
                x_ls_i_map,
            )?
            + &xt_diag_y_dense(x_t, &d2h_tl, x_ls)?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_t),
                h_tl.view(),
                x_ls_ab_map,
            )?;
        let h_ll_block = weighted_crossprod_psi_maps(
            x_ls_ab_map,
            h_ll.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(x_ls_i_map, h_ll.view(), x_ls_j_map)?
            + &weighted_crossprod_psi_maps(x_ls_j_map, h_ll.view(), x_ls_i_map)?
            + &weighted_crossprod_psi_maps(
                x_ls_i_map,
                dh_ll_j.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                x_ls_j_map,
                dh_ll_i.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
                dh_ll_i.view(),
                x_ls_j_map,
            )?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
                dh_ll_j.view(),
                x_ls_i_map,
            )?
            + &xt_diag_x_dense(x_ls, &d2h_ll)?
            + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_ls),
                h_ll.view(),
                x_ls_ab_map,
            )?;

        let mut hessian_psi_psi = Array2::<f64>::zeros((total, total));
        hessian_psi_psi
            .slice_mut(s![0..pt, 0..pt])
            .assign(&h_tt_block);
        hessian_psi_psi
            .slice_mut(s![0..pt, pt..pt + pls])
            .assign(&h_tl_block);
        hessian_psi_psi
            .slice_mut(s![pt..pt + pls, pt..pt + pls])
            .assign(&h_ll_block);
        mirror_upper_to_lower(&mut hessian_psi_psi);

        Ok(crate::custom_family::ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi,
            hessian_psi_psi_operator: None,
        })
    }

    fn exact_newton_joint_psihessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psihessian_directional_derivative_from_parts(
                block_states,
                &dir_a,
                d_beta_flat,
                x_t,
                x_ls,
            )?,
        ))
    }

    fn exact_newton_joint_psihessian_directional_derivative_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        dir_a: &LocationScaleJointPsiDirection,
        d_beta_flat: &Array1<f64>,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let total = pt + pls;
        if d_beta_flat.len() != total {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialLocationScaleFamily joint psi hessian directional derivative length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                total
            ) }.into());
        }
        let xi_t = fast_av(x_t, &d_beta_flat.slice(s![0..pt]));
        let xi_ls = fast_av(x_ls, &d_beta_flat.slice(s![pt..pt + pls]));
        let x_t_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();

        // Mixed contraction T_a[u] = D_beta H_{psi_a}[u].
        //
        // In the non-wiggle family the realized design derivatives X_{psi_a}
        // are fixed with respect to beta, so differentiating the explicit
        // Hessian drift H_{psi_a} only moves the rowwise coefficient arrays.
        // This helper therefore returns exactly the likelihood-side mixed drift
        // required by the unified outer Hessian formula
        //
        //   ddot H_{ij}
        //   = H_{ij}
        //     + T_i[beta_j]
        //     + T_j[beta_i]
        //     + D_beta H[beta_ij]
        //     + D_beta^2 H[beta_i, beta_j].
        //
        // For i = psi_a, the generic assembler supplies beta_j and any
        // realized-penalty piece S_{psi_a} itself; this family hook contributes
        // only the exact likelihood-side T_a[beta_j].
        //
        // With
        //   du   = D_beta q[u]   = -r xi_t - q xi_ls,
        //   q_a  = q_{psi_a}     = -r z_t,a - q z_ls,a,
        //   q_au = D_beta q_a[u] = r z_t,a xi_ls - du z_ls,a,
        //
        // the directional derivatives of the first-order Hessian-drift
        // coefficients are the mixed specializations of the exact psi/psi
        // formulas with z_ls,ab = 0 and q_ab = q_au:
        //
        //   D_u(d_a h_tt)
        //   = r²[
        //       d du q_a + c q_au
        //       - 2c(q_a xi_ls + du z_ls,a)
        //       + 4b xi_ls z_ls,a
        //     ],
        //
        //   D_u(d_a h_tl)
        //   = r[
        //       ((3c + q d) q_a) du
        //       + (2b + q c) q_au
        //       - (2b + q c)(q_a xi_ls + du z_ls,a)
        //       + (a + q b) xi_ls z_ls,a
        //     ],
        //
        //   D_u(d_a h_ll)
        //   = (4b + 5q c + q² d) du q_a
        //     + (a + 3q b + q² c) q_au.
        //
        // Since X_t, X_ls, X_{t,psi_a}, X_{ls,psi_a} are all beta-independent
        // here, the full matrix contraction is obtained by replacing the row
        // coefficient arrays in H_{psi_a} by their directional derivatives.
        let mut dh_tt_u = Array1::<f64>::zeros(n);
        let mut dh_tl_u = Array1::<f64>::zeros(n);
        let mut dh_ll_u = Array1::<f64>::zeros(n);
        let mut h_tt_u = Array1::<f64>::zeros(n);
        let mut h_tl_u = Array1::<f64>::zeros(n);
        let mut h_ll_u = Array1::<f64>::zeros(n);
        for row in 0..n {
            let q = core.q0[row];
            let r = 1.0 / core.sigma[row];
            let s = core.dsigma_deta[row] / core.sigma[row];
            let xi_ls_s = s * xi_ls[row];
            let z_ls_psi_s = s * dir_a.z_ls_psi[row];
            let du = -r * xi_t[row] - q * xi_ls_s;
            let q_a = -r * dir_a.z_primary_psi[row] - q * z_ls_psi_s;
            let q_au = r * dir_a.z_primary_psi[row] * xi_ls_s - du * z_ls_psi_s;
            let (a, b, c) = binomial_neglog_q_derivatives_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            );
            let d = binomial_neglog_q_fourth_derivative_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            )?;
            let u = a + q * b;
            h_tt_u[row] = r * r * (c * du - 2.0 * b * xi_ls_s);
            h_tl_u[row] = r * ((2.0 * b + q * c) * du - u * xi_ls_s);
            h_ll_u[row] = (a + 3.0 * q * b + q * q * c) * du;
            dh_tt_u[row] = r
                * r
                * (d * du * q_a + c * q_au - 2.0 * c * (q_a * xi_ls_s + du * z_ls_psi_s)
                    + 4.0 * b * xi_ls_s * z_ls_psi_s);
            dh_tl_u[row] = r
                * (((3.0 * c + q * d) * q_a) * du + (2.0 * b + q * c) * q_au
                    - (2.0 * b + q * c) * (q_a * xi_ls_s + du * z_ls_psi_s)
                    + u * xi_ls_s * z_ls_psi_s);
            dh_ll_u[row] = (4.0 * b + 5.0 * q * c + q * q * d) * du * q_a
                + (a + 3.0 * q * b + q * q * c) * q_au;
        }

        let tt_block = weighted_crossprod_psi_maps(
            x_t_map,
            h_tt_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_t),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_t),
            h_tt_u.view(),
            x_t_map,
        )? + &xt_diag_x_dense(x_t, &dh_tt_u)?;
        let tl_block = weighted_crossprod_psi_maps(
            x_t_map,
            h_tl_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_t),
            h_tl_u.view(),
            x_ls_map,
        )? + &xt_diag_y_dense(x_t, &dh_tl_u, x_ls)?;
        let ll_block = weighted_crossprod_psi_maps(
            x_ls_map,
            h_ll_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
            h_ll_u.view(),
            x_ls_map,
        )? + &xt_diag_x_dense(x_ls, &dh_ll_u)?;
        let mut out = Array2::<f64>::zeros((total, total));
        out.slice_mut(s![0..pt, 0..pt]).assign(&tt_block);
        out.slice_mut(s![0..pt, pt..pt + pls]).assign(&tl_block);
        out.slice_mut(s![pt..pt + pls, pt..pt + pls])
            .assign(&ll_block);
        mirror_upper_to_lower(&mut out);
        Ok(out)
    }

    /// Build the [`BlockEffectiveJacobian`] for block `block_idx`.
    ///
    /// The two-output map is (η_threshold, η_log_sigma):
    /// - block 0 (threshold):  output 0 = design rows, output 1 = zeros
    /// - block 1 (log_sigma):  output 0 = zeros, output 1 = design rows
    pub fn block_effective_jacobian(
        specs: &[ParameterBlockSpec],
        block_idx: usize,
    ) -> Result<Box<dyn BlockEffectiveJacobian>, String> {
        crate::util::block_jacobian::AdditiveWiggleBlockLayout {
            family: "BinomialLocationScaleFamily",
            n_outputs: 2,
            additive_blocks: &[Self::BLOCK_T, Self::BLOCK_LOG_SIGMA],
            wiggle_block: None,
        }
        .block_effective_jacobian(specs, block_idx)
    }
}


impl CustomFamily for BinomialLocationScaleFamily {
    /// The Binomial location-scale joint Hessian depends on β because the
    /// Hessian blocks are functions of q = -t/σ and the link derivatives,
    /// all of which change when β_t or β_{log σ} move.
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Operator-aware: matrix-free workspace applies joint Hv at
        // O(n · (p_t + p_ℓ)); only fall back to the dense build cost when
        // `use_joint_matrix_free_path` declines the operator path.
        crate::families::location_scale_engine::location_scale_coefficient_hessian_cost(
            self.y.len() as u64,
            specs,
        )
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        if !self.exact_joint_supported() {
            return Err(
                "BinomialLocationScaleFamily requires exact curvature designs; diagonal fallback has been removed"
                    .to_string(),
            );
        }
        let threshold_design = self.threshold_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleFamily exact path is missing threshold design".to_string()
        })?;
        let log_sigma_design = self.log_sigma_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleFamily exact path is missing log-sigma design".to_string()
        })?;

        // Per-block gradients from the eta-space score.
        //
        //   score_q = -m1   (m1 = dF/dq, F = -ℓ)
        //   grad_eta_t[i]  = score_q * q_t
        //   grad_eta_ls[i] = score_q * q_ls
        let mut grad_eta_t_v = vec![0.0_f64; n];
        let mut grad_eta_ls_v = vec![0.0_f64; n];
        let y_slice_e = self.y.as_slice().expect("y must be contiguous");
        let w_slice_e = self.weights.as_slice().expect("weights must be contiguous");
        let q0_slice_e = core.q0.as_slice().expect("q0 must be contiguous");
        let eta_t_slice_e = eta_t.as_slice().expect("eta_t must be contiguous");
        let eta_ls_slice_e = eta_ls.as_slice().expect("eta_ls must be contiguous");
        let link_kind_e = &self.link_kind;
        let gradient_pairs: Result<Vec<(f64, f64)>, String> = (0..n)
            .into_par_iter()
            .map(|i| {
                let tower = binomial_location_scale_nll_tower(
                    y_slice_e[i],
                    w_slice_e[i],
                    eta_t_slice_e[i],
                    eta_ls_slice_e[i],
                    q0_slice_e[i],
                    core.mu[i],
                    core.dmu_dq[i],
                    core.d2mu_dq2[i],
                    core.d3mu_dq3[i],
                    link_kind_e,
                    false,
                )?;
                Ok((-tower.g[0], -tower.g[1]))
            })
            .collect();
        for (i, (g_t, g_ls)) in gradient_pairs?.into_iter().enumerate() {
            grad_eta_t_v[i] = g_t;
            grad_eta_ls_v[i] = g_ls;
        }
        let grad_eta_t = Array1::from_vec(grad_eta_t_v);
        let grad_eta_ls = Array1::from_vec(grad_eta_ls_v);
        let grad_t = threshold_design.transpose_vector_multiply(&grad_eta_t);
        let grad_ls = log_sigma_design.transpose_vector_multiply(&grad_eta_ls);

        // Per-block Hessians without ever materializing the full p×p joint
        // matrix — the off-diagonal cross block is unused for IRLS-style block
        // working sets and would cost O(p_t * p_ls * n) to form. The diagonal
        // blocks are computed from the same row coefficients as the joint.
        let (h_tt, h_ll) = self.exact_newton_block_diagonal_hessians_from_design_matrices(
            block_states,
            threshold_design,
            log_sigma_design,
        )?;
        Ok(FamilyEvaluation {
            log_likelihood: core.log_likelihood,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_t,
                    hessian: SymmetricMatrix::Dense(h_tt),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_ls,
                    hessian: SymmetricMatrix::Dense(h_ll),
                },
            ],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }
        // Zero-allocation O(n) scalar loop — no working sets, no n-vector intermediates.
        binomial_location_scale_ll_only(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )
    }

    /// Outer-only log-likelihood with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `Some`, only the sampled rows
    /// contribute; each row's per-row log-likelihood term is multiplied by
    /// `WeightedOuterRow.weight`, the Horvitz–Thompson inverse-inclusion
    /// factor 1/π_i (uniform or stratified sampling both supported), so the
    /// partial sum is an unbiased estimator of the full-data log-likelihood.
    /// When `None`, this returns the full-data `log_likelihood_only`. Inner
    /// PIRLS line searches never install the subsample option, so they
    /// continue to score the exact full-data log-likelihood.
    fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        let Some(subsample) = options.outer_score_subsample.as_ref() else {
            return self.log_likelihood_only(block_states);
        };
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != n || eta_ls.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily input size mismatch".to_string(),
            }
            .into());
        }
        use rayon::iter::ParallelIterator;
        let link_kind = &self.link_kind;
        let ll: Result<f64, String> = subsample
            .rows
            .par_iter()
            .try_fold(
                || 0.0_f64,
                |acc, row| -> Result<f64, String> {
                    let i = row.index;
                    let wi = self.weights[i];
                    if wi == 0.0 {
                        return Ok(acc);
                    }
                    let SigmaJet1 { sigma, .. } = exp_sigma_jet1_scalar(eta_ls[i]);
                    let q = binomial_location_scale_q0(eta_t[i], sigma);
                    let mu = if matches!(link_kind, InverseLink::Standard(StandardLink::Probit)) {
                        0.5
                    } else {
                        let jet = inverse_link_jet_for_inverse_link(link_kind, q).map_err(|e| {
                            format!("location-scale inverse-link evaluation failed: {e}")
                        })?;
                        jet.mu
                    };
                    let term =
                        binomial_location_scale_log_likelihood(self.y[i], wi, q, link_kind, mu)?;
                    Ok(acc + row.weight * term)
                },
            )
            .try_reduce(|| 0.0_f64, |a, b| Ok(a + b));
        ll
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn diagonalworking_weights_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        idx: usize,
        arr: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(arr.iter().all(|v| !v.is_nan()));
        Err(
            "BinomialLocationScaleFamily no longer supports diagonal working weights; exact curvature is required"
                .to_string(),
        )
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        self.exact_newton_joint_psi_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
        )
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.exact_newton_joint_psisecond_order_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_i,
            psi_j,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_psihessian_directional_derivative_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if !self.exact_joint_supported() {
            return Ok(None);
        }
        Ok(Some(Arc::new(
            BinomialLocationScaleExactNewtonJointPsiWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
            )?,
        )))
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if !self.exact_joint_supported() {
            return Ok(None);
        }
        let pt = self
            .threshold_design
            .as_ref()
            .ok_or_else(|| {
                "BinomialLocationScaleFamily exact path is missing threshold design".to_string()
            })?
            .ncols();
        let pls = self
            .log_sigma_design
            .as_ref()
            .ok_or_else(|| {
                "BinomialLocationScaleFamily exact path is missing log-sigma design".to_string()
            })?
            .ncols();
        let total = pt + pls;
        let (start, end, joint_direction) = match block_idx {
            Self::BLOCK_T => {
                if d_beta.len() != pt {
                    return Err(GamlssError::DimensionMismatch { reason: format!(
                        "BinomialLocationScaleFamily threshold d_beta length mismatch: got {}, expected {}",
                        d_beta.len(),
                        pt
                    ) }.into());
                }
                let mut dir = Array1::<f64>::zeros(total);
                dir.slice_mut(s![0..pt]).assign(d_beta);
                (0usize, pt, dir)
            }
            Self::BLOCK_LOG_SIGMA => {
                if d_beta.len() != pls {
                    return Err(GamlssError::DimensionMismatch { reason: format!(
                        "BinomialLocationScaleFamily log-sigma d_beta length mismatch: got {}, expected {}",
                        d_beta.len(),
                        pls
                    ) }.into());
                }
                let mut dir = Array1::<f64>::zeros(total);
                dir.slice_mut(s![pt..pt + pls]).assign(d_beta);
                (pt, pt + pls, dir)
            }
            _ => return Ok(None),
        };
        let joint = self
            .exact_newton_joint_hessian_directional_derivative(block_states, &joint_direction)?
            .ok_or_else(|| {
                format!("missing joint exact-newton directional Hessian for block {block_idx}")
            })?;
        Ok(Some(joint.slice(s![start..end, start..end]).to_owned()))
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, None)
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            None,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_for_specs(block_states, Some(specs))
    }

    fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_flat,
        )
    }

    fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_for_specs(
            block_states,
            Some(specs),
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn joint_jeffreys_information_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.expected_joint_information_for_specs(block_states, Some(specs))
    }

    fn joint_jeffreys_information_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.expected_joint_information_directional_for_specs(
            block_states,
            Some(specs),
            d_beta_flat,
        )
    }

    fn joint_jeffreys_information_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.expected_joint_information_second_directional_for_specs(
            block_states,
            Some(specs),
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn joint_jeffreys_information_contracted_trace_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        weight: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.expected_joint_contracted_trace_hessian_for_specs(block_states, Some(specs), weight)
    }

    fn joint_jeffreys_information_contracted_trace_hessian_available(&self) -> bool {
        true
    }

    fn joint_jeffreys_information_matches_observed_hessian(&self) -> bool {
        // The Jeffreys information above is the EXPECTED Fisher information,
        // not the observed Hessian: observed-Hessian conditioning certificates
        // ("Jeffreys provably skippable" matvec pre-checks) must not gate the
        // expected-information term off — for probit-class likelihoods the
        // observed information grows on saturated misclassified rows exactly
        // where the expected information collapses and the gate must arm
        // (gam#1020).
        false
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_block_designs_owned(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_gradient_from_designs(block_states, &x_t, &x_ls)
            .map(Some)
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_block_designs_owned(Some(specs))? else {
            return Ok(None);
        };
        let workspace = BinomialLocationScaleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            x_t,
            x_ls,
        )?;
        Ok(Some(Arc::new(workspace)))
    }

    /// Outer-aware joint-Hessian workspace with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `None`, this is byte-identical
    /// to `exact_newton_joint_hessian_workspace`. When `Some`, the precomputed
    /// per-row coefficient arrays (`coeff_tt`, `coeff_tl`, `coeff_ll`) — which
    /// every downstream assembly (`hessian_dense`, `hessian_matvec`,
    /// `hessian_diagonal`) consumes row-linearly via `Xᵀ diag(W) X` — are
    /// replaced by a Horvitz–Thompson mask: each sampled row's coefficient is
    /// multiplied by `WeightedOuterRow.weight` (the inverse-inclusion factor
    /// 1/π_i; uniform or stratified sampling both supported), and non-sampled
    /// rows are zeroed. The resulting joint Hessian is an unbiased estimator
    /// of the full-data joint Hessian. Inner PIRLS never installs the option,
    /// so the inner solve continues to consume the exact full-data Hessian.
    fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_block_designs_owned(Some(specs))? else {
            return Ok(None);
        };
        let mut workspace = BinomialLocationScaleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            x_t,
            x_ls,
        )?;
        if let Some(subsample) = options.outer_score_subsample.as_ref() {
            workspace.apply_outer_subsample(subsample.rows.as_ref());
        }
        Ok(Some(Arc::new(workspace)))
    }

    /// Outer-derivative policy: declare HT-subsample capability.
    ///
    /// BinomialLocationScaleFamily overrides
    /// `log_likelihood_only_with_options` and
    /// `exact_newton_joint_hessian_workspace_with_options` to consume
    /// `options.outer_score_subsample` with per-row Horvitz–Thompson weights
    /// (each sampled row's contribution is multiplied by
    /// `WeightedOuterRow.weight = 1/π_i`; non-sampled rows are zeroed),
    /// yielding unbiased estimators of the full-data log-likelihood and
    /// joint Hessian. The ψ-workspace path is not yet subsample-aware: it
    /// builds the exact full-data ψ Hessian blocks, which are trivially
    /// unbiased; so the outer-score components are a sum of HT-unbiased and
    /// exact-unbiased pieces and the total remains an unbiased estimator of
    /// the full-data outer score. Inner-PIRLS and final-covariance paths
    /// never install the option, so they continue to consume the exact
    /// full-data quantities.
    fn outer_derivative_subsample_capable(&self) -> bool {
        true
    }

    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        // Representation support means the realized two-block designs can be
        // applied as β-space operators. It does not imply that exact
        // second-order outer θ work is cheap.
        if specs.len() != 2 {
            return false;
        }
        let n = self.y.len();
        specs[Self::BLOCK_T].design.nrows() == n && specs[Self::BLOCK_LOG_SIGMA].design.nrows() == n
    }
}


impl CustomFamilyGenerative for BinomialLocationScaleFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 2 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleFamily expects 2 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != self.y.len() || eta_ls.len() != self.y.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleFamily generative size mismatch".to_string(),
            }
            .into());
        }
        let mean = gamlss_rowwise_map_result(self.y.len(), |i| {
            let sigma = exp_sigma_from_eta_scalar(eta_ls[i]);
            let q = binomial_location_scale_q0(eta_t[i], sigma);
            let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q)
                .map_err(|e| format!("location-scale inverse-link evaluation failed: {e}"))?;
            Ok(jet.mu)
        })?;
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Bernoulli,
        })
    }
}


/// Matrix-free joint-Hessian operator for the two-block binomial
/// location-scale family.
///
/// The dense joint Hessian is `H = [[X_t^T D_tt X_t, X_t^T D_tl X_ls],
///                                  [X_ls^T D_tl X_t, X_ls^T D_ll X_ls]]`
/// where `D_tt`, `D_tl`, `D_ll` are diagonal weight vectors derived from the
/// rowwise scalar-composition Hessian. For a flattened direction
/// `v = (v_t, v_ls)`, `H v` is computed as
///
///   u_t = X_t v_t,  u_ls = X_ls v_ls,
///   r_t = D_tt .* u_t + D_tl .* u_ls,
///   r_ls = D_tl .* u_t + D_ll .* u_ls,
///   H v = (X_t^T r_t, X_ls^T r_ls).
///
/// Cost is Θ(n (p_t + p_ls)) per matvec versus Θ(n (p_t + p_ls)^2) to form
/// the dense matrix. The same block-operator structure is used for first and
/// second directional derivatives.
struct BinomialLocationScaleHessianWorkspace {
    family: BinomialLocationScaleFamily,
    x_t: DesignMatrix,
    x_ls: DesignMatrix,
    core: BinomialLocationScaleCore,
    coeff_tt: Array1<f64>,
    coeff_tl: Array1<f64>,
    coeff_ll: Array1<f64>,
    direction_eta_cache: Mutex<HashMap<BinomialDirectionKey, Arc<BinomialDirectionEta>>>,
    first_coeff_cache: Mutex<HashMap<BinomialDirectionKey, Arc<BinomialRowCoeffTriple>>>,
    // No `second_coeff_cache` deliberately: see `second_coefficients` for why
    // the per-pair cache was a memory-only loss at large-scale shape.
}


#[derive(Clone, Eq, Hash, PartialEq)]
struct BinomialDirectionKey {
    bits: Vec<u64>,
}


impl BinomialDirectionKey {
    fn from_array(v: &Array1<f64>) -> Self {
        Self {
            bits: v.iter().map(|value| value.to_bits()).collect(),
        }
    }
}


struct BinomialDirectionEta {
    t: Array1<f64>,
    ls: Array1<f64>,
}


struct BinomialRowCoeffTriple {
    tt: Arc<Array1<f64>>,
    tl: Arc<Array1<f64>>,
    ll: Arc<Array1<f64>>,
}


impl BinomialLocationScaleHessianWorkspace {
    fn new(
        family: BinomialLocationScaleFamily,
        block_states: Vec<ParameterBlockState>,
        x_t: DesignMatrix,
        x_ls: DesignMatrix,
    ) -> Result<Self, String> {
        let eta_t = &block_states[BinomialLocationScaleFamily::BLOCK_T].eta;
        let eta_ls = &block_states[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].eta;
        let core = binomial_location_scale_core(
            &family.y,
            &family.weights,
            eta_t,
            eta_ls,
            None,
            &family.link_kind,
        )?;
        let (coeff_tt, coeff_tl, coeff_ll) =
            family.exact_newton_joint_hessian_row_coefficients(&block_states)?;
        Ok(Self {
            family,
            x_t,
            x_ls,
            core,
            coeff_tt,
            coeff_tl,
            coeff_ll,
            direction_eta_cache: Mutex::new(HashMap::new()),
            first_coeff_cache: Mutex::new(HashMap::new()),
        })
    }

    fn direction_eta(
        &self,
        key: &BinomialDirectionKey,
        d_beta: &Array1<f64>,
        pt: usize,
        total: usize,
    ) -> Arc<BinomialDirectionEta> {
        if let Some(value) = self
            .direction_eta_cache
            .lock()
            .expect("binomial direction eta cache lock poisoned")
            .get(key)
            .cloned()
        {
            return value;
        }
        let value = Arc::new(BinomialDirectionEta {
            t: self
                .x_t
                .matrixvectormultiply(&d_beta.slice(s![0..pt]).to_owned()),
            ls: self
                .x_ls
                .matrixvectormultiply(&d_beta.slice(s![pt..total]).to_owned()),
        });
        let mut cache = self
            .direction_eta_cache
            .lock()
            .expect("binomial direction eta cache lock poisoned");
        cache
            .entry(key.clone())
            .or_insert_with(|| value.clone())
            .clone()
    }

    fn first_coefficients(
        &self,
        key: &BinomialDirectionKey,
        eta: &BinomialDirectionEta,
    ) -> Result<Arc<BinomialRowCoeffTriple>, String> {
        if let Some(value) = self
            .first_coeff_cache
            .lock()
            .expect("binomial first coefficient cache lock poisoned")
            .get(key)
            .cloned()
        {
            return Ok(value);
        }
        let (tt, tl, ll) = binomial_location_scale_first_directional_coefficients(
            &self.family.y,
            &self.family.weights,
            &self.core,
            &eta.t,
            &eta.ls,
            &self.family.link_kind,
        )?;
        let value = Arc::new(BinomialRowCoeffTriple {
            tt: Arc::new(tt),
            tl: Arc::new(tl),
            ll: Arc::new(ll),
        });
        let mut cache = self
            .first_coeff_cache
            .lock()
            .expect("binomial first coefficient cache lock poisoned");
        Ok(cache
            .entry(key.clone())
            .or_insert_with(|| value.clone())
            .clone())
    }

    /// No caching here, deliberately: at large-scale shape (n=320k, K=14 outer
    /// coords) the K² ≈ 196 unique direction-pairs are queried exactly once
    /// per outer Hessian eval, and each cached entry stored 3·n f64s
    /// = ~7.7 MB → ~1.5 GB peak per eval with zero practical hit-rate.
    /// Across outer evals the directions shift with ρ/ψ so cross-eval hits
    /// are nil. Computing on demand is O(n) — under 10 ms at this scale,
    /// dwarfed by the (n × p²) trace work that consumes the result.
    fn second_coefficients(
        &self,
        eta_u: &BinomialDirectionEta,
        eta_v: &BinomialDirectionEta,
    ) -> Result<Arc<BinomialRowCoeffTriple>, String> {
        let (tt, tl, ll) = binomial_location_scalesecond_directional_coefficients(
            &self.family.y,
            &self.family.weights,
            &self.core,
            &eta_u.t,
            &eta_u.ls,
            &eta_v.t,
            &eta_v.ls,
            &self.family.link_kind,
        )?;
        Ok(Arc::new(BinomialRowCoeffTriple {
            tt: Arc::new(tt),
            tl: Arc::new(tl),
            ll: Arc::new(ll),
        }))
    }

    /// Apply a Horvitz–Thompson outer-row subsample mask to the precomputed
    /// per-row coefficient arrays in place. Each sampled row's `coeff_*[i]`
    /// is multiplied by its `WeightedOuterRow.weight` (the HT inverse-
    /// inclusion factor 1/π_i); non-sampled rows are zeroed. Because every
    /// downstream assembly (`hessian_dense`, `hessian_matvec`,
    /// `hessian_diagonal`) is row-linear in these arrays via `Xᵀ diag(W) X`,
    /// the resulting joint-Hessian is an unbiased estimator of the full-data
    /// joint Hessian.
    fn apply_outer_subsample(
        &mut self,
        rows: &[crate::families::marginal_slope_shared::WeightedOuterRow],
    ) {
        let n = self.coeff_tt.len();
        let mut mask_tt = Array1::<f64>::zeros(n);
        let mut mask_tl = Array1::<f64>::zeros(n);
        let mut mask_ll = Array1::<f64>::zeros(n);
        for r in rows {
            let i = r.index;
            mask_tt[i] = self.coeff_tt[i] * r.weight;
            mask_tl[i] = self.coeff_tl[i] * r.weight;
            mask_ll[i] = self.coeff_ll[i] * r.weight;
        }
        self.coeff_tt = mask_tt;
        self.coeff_tl = mask_tl;
        self.coeff_ll = mask_ll;
    }
}


impl ExactNewtonJointHessianWorkspace for BinomialLocationScaleHessianWorkspace {
    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        // Same Hv structure as `hessian_matvec`, built once via 3 GEMMs:
        //   H_tt = X_tᵀ diag(coeff_tt) X_t,
        //   H_tl = X_tᵀ diag(coeff_tl) X_ls,
        //   H_ll = X_lsᵀ diag(coeff_ll) X_ls,
        // versus letting `MatrixFreeSpdOperator::materialize_dense_operator`
        // reconstruct the dense Hessian via `total` canonical-basis HVPs. At
        // large scale, canonical-basis materialization costs p_total full
        // Hessian-vector products. The design helpers below stream row chunks,
        // so the only dense object retained here is the small p_total×p_total
        // coefficient Hessian.
        let pt = self.x_t.ncols();
        let pls = self.x_ls.ncols();
        let total = pt + pls;
        let h_tt = xt_diag_x_design(&self.x_t, &self.coeff_tt)?;
        let h_tl = xt_diag_y_design(&self.x_t, &self.coeff_tl, &self.x_ls)?;
        let h_ll = xt_diag_x_design(&self.x_ls, &self.coeff_ll)?;
        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..pt, 0..pt]).assign(&h_tt);
        h.slice_mut(s![0..pt, pt..total]).assign(&h_tl);
        h.slice_mut(s![pt..total, pt..total]).assign(&h_ll);
        mirror_upper_to_lower(&mut h);
        Ok(Some(h))
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let pt = self.x_t.ncols();
        let pls = self.x_ls.ncols();
        let total = pt + pls;
        if v.len() != total {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScale matvec dimension mismatch: got {}, expected {}",
                    v.len(),
                    total
                ),
            }
            .into());
        }
        // u_t = X_t v_t, u_ls = X_ls v_ls
        let u_t = self
            .x_t
            .matrixvectormultiply(&v.slice(s![0..pt]).to_owned());
        let u_ls = self
            .x_ls
            .matrixvectormultiply(&v.slice(s![pt..total]).to_owned());
        // r_t = D_tt .* u_t + D_tl .* u_ls; r_ls = D_tl .* u_t + D_ll .* u_ls
        let r_t = &self.coeff_tt * &u_t + &self.coeff_tl * &u_ls;
        let r_ls = &self.coeff_tl * &u_t + &self.coeff_ll * &u_ls;
        // (X_t^T r_t, X_ls^T r_ls)
        let out_t = self.x_t.transpose_vector_multiply(&r_t);
        let out_ls = self.x_ls.transpose_vector_multiply(&r_ls);
        let mut out = Array1::<f64>::zeros(total);
        out.slice_mut(s![0..pt]).assign(&out_t);
        out.slice_mut(s![pt..total]).assign(&out_ls);
        Ok(Some(out))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        let pt = self.x_t.ncols();
        let pls = self.x_ls.ncols();
        let total = pt + pls;
        let mut diag = Array1::<f64>::zeros(total);
        let diag_t = design_weighted_column_squares(&self.x_t, &self.coeff_tt)?;
        let diag_ls = design_weighted_column_squares(&self.x_ls, &self.coeff_ll)?;
        diag.slice_mut(s![0..pt]).assign(&diag_t);
        diag.slice_mut(s![pt..total]).assign(&diag_ls);
        Ok(Some(diag))
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(self
            .directional_derivative_operator(d_beta_flat)?
            .map(|operator| operator.to_dense()))
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        let pt = self.x_t.ncols();
        let pls = self.x_ls.ncols();
        let total = pt + pls;
        if d_beta_flat.len() != total {
            return Err(GamlssError::InvalidInput {
                reason: format!(
                    "BinomialLocationScale dH operator: d_beta length {} != {}",
                    d_beta_flat.len(),
                    total
                ),
            }
            .into());
        }
        let key = BinomialDirectionKey::from_array(d_beta_flat);
        let eta = self.direction_eta(&key, d_beta_flat, pt, total);
        let coeffs = self.first_coefficients(&key, &eta)?;
        Ok(Some(Arc::new(make_two_block_design_row_coeff_operator(
            self.x_t.clone(),
            self.x_ls.clone(),
            coeffs.tt.clone(),
            coeffs.tl.clone(),
            coeffs.ll.clone(),
        )?)))
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(self
            .second_directional_derivative_operator(d_beta_u_flat, d_beta_v_flat)?
            .map(|operator| operator.to_dense()))
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        let pt = self.x_t.ncols();
        let pls = self.x_ls.ncols();
        let total = pt + pls;
        if d_beta_u.len() != total || d_beta_v.len() != total {
            return Err(GamlssError::InvalidInput {
                reason: format!(
                    "BinomialLocationScale d2H operator: d_beta_{{u,v}} length {}/{} != {}",
                    d_beta_u.len(),
                    d_beta_v.len(),
                    total
                ),
            }
            .into());
        }
        let key_u = BinomialDirectionKey::from_array(d_beta_u);
        let key_v = BinomialDirectionKey::from_array(d_beta_v);
        let eta_u = self.direction_eta(&key_u, d_beta_u, pt, total);
        let eta_v = self.direction_eta(&key_v, d_beta_v, pt, total);
        let coeffs = self.second_coefficients(&eta_u, &eta_v)?;
        Ok(Some(Arc::new(make_two_block_design_row_coeff_operator(
            self.x_t.clone(),
            self.x_ls.clone(),
            coeffs.tt.clone(),
            coeffs.tl.clone(),
            coeffs.ll.clone(),
        )?)))
    }
}


/// Built-in binomial location-scale family with a configurable inverse link and learnable wiggle on q.
///
/// Block structure:
/// - Block 0: threshold T(covariates)
/// - Block 1: log sigma(covariates)
/// - Block 2: wiggle(q) represented by B-spline coefficients on q
#[derive(Clone)]
pub struct BinomialLocationScaleWiggleFamily {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub link_kind: InverseLink,
    pub threshold_design: Option<DesignMatrix>,
    pub log_sigma_design: Option<DesignMatrix>,
    pub wiggle_knots: Array1<f64>,
    pub wiggle_degree: usize,
    /// Resource policy threaded into PsiDesignMap construction (and any other
    /// per-call materialization decision) made during exact-Newton joint psi
    /// derivative evaluation. Defaults to `ResourcePolicy::default_library()`
    /// when the family is built without an explicit policy.
    pub policy: crate::resource::ResourcePolicy,
}


impl BinomialLocationScaleWiggleFamily {
    pub const BLOCK_T: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;
    pub const BLOCK_WIGGLE: usize = 2;

    pub fn parameternames() -> &'static [&'static str] {
        &["threshold", "log_sigma", "wiggle"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[
            ParameterLink::InverseLink,
            ParameterLink::Log,
            ParameterLink::Wiggle,
        ]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "binomial_location_scalewiggle",
            parameternames: Self::parameternames(),
            parameter_links: Self::parameter_links(),
        }
    }

    fn exact_joint_supported(&self) -> bool {
        self.threshold_design.is_some() && self.log_sigma_design.is_some()
    }

    pub fn initializewiggle_knots_from_q(
        q_seed: ArrayView1<'_, f64>,
        degree: usize,
        num_internal_knots: usize,
    ) -> Result<Array1<f64>, String> {
        initializewiggle_knots_from_seed(q_seed, degree, num_internal_knots)
    }

    fn wiggle_basiswith_options(
        &self,
        q0: ArrayView1<'_, f64>,
        basis_options: BasisOptions,
    ) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(
            q0,
            &self.wiggle_knots,
            self.wiggle_degree,
            basis_options.derivative_order,
        )
    }

    fn wiggle_design(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        self.wiggle_basiswith_options(q0, BasisOptions::value())
    }

    fn wiggle_dq_dq0(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d_constrained = self.wiggle_basiswith_options(q0, BasisOptions::first_derivative())?;
        if d_constrained.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "wiggle derivative col mismatch: got {}, expected {}",
                    d_constrained.ncols(),
                    beta_link_wiggle.len()
                ),
            }
            .into());
        }
        Ok(d_constrained.dot(&beta_link_wiggle) + 1.0)
    }

    fn wiggle_d2q_dq02(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d2_constrained =
            self.wiggle_basiswith_options(q0, BasisOptions::second_derivative())?;
        if d2_constrained.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "wiggle second-derivative col mismatch: got {}, expected {}",
                    d2_constrained.ncols(),
                    beta_link_wiggle.len()
                ),
            }
            .into());
        }
        Ok(d2_constrained.dot(&beta_link_wiggle))
    }

    fn wiggle_d3q_dq03(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d3_constrained = self.wiggle_d3basis_constrained(q0)?;
        if d3_constrained.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "wiggle third-derivative col mismatch: got {}, expected {}",
                    d3_constrained.ncols(),
                    beta_link_wiggle.len()
                ),
            }
            .into());
        }
        Ok(d3_constrained.dot(&beta_link_wiggle))
    }

    fn wiggle_d3basis_constrained(&self, q0: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(q0, &self.wiggle_knots, self.wiggle_degree, 3)
    }

    fn wiggle_d4q_dq04(
        &self,
        q0: ArrayView1<'_, f64>,
        beta_link_wiggle: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let d4 = monotone_wiggle_basis_with_derivative_order(
            q0,
            &self.wiggle_knots,
            self.wiggle_degree,
            4,
        )?;
        if d4.ncols() != beta_link_wiggle.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "wiggle fourth-derivative col mismatch: got {}, expected {}",
                    d4.ncols(),
                    beta_link_wiggle.len()
                ),
            }
            .into());
        }
        Ok(d4.dot(&beta_link_wiggle))
    }

    fn dense_block_designs(&self) -> Result<(Cow<'_, Array2<f64>>, Cow<'_, Array2<f64>>), String> {
        dense_locscale_block_designs_cached(
            self.threshold_design.as_ref(),
            self.log_sigma_design.as_ref(),
            "BinomialLocationScaleWiggleFamily",
            "BinomialLocationScaleWiggle",
            "threshold",
            &self.policy.material_policy(),
        )
    }

    fn dense_block_designs_fromspecs<'a>(
        &self,
        specs: &'a [ParameterBlockSpec],
    ) -> Result<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>), String> {
        dense_locscale_block_designs_fromspecs(
            specs,
            3,
            "BinomialLocationScaleWiggleFamily",
            "BinomialLocationScaleWiggle",
            Self::BLOCK_T,
            Self::BLOCK_LOG_SIGMA,
            "threshold",
            &self.policy.material_policy(),
        )
    }

    fn exact_joint_dense_block_designs<'a>(
        &'a self,
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<(Cow<'a, Array2<f64>>, Cow<'a, Array2<f64>>)>, String> {
        if self.threshold_design.is_some() && self.log_sigma_design.is_some() {
            return self.dense_block_designs().map(Some);
        }
        if let Some(specs) = specs {
            return self.dense_block_designs_fromspecs(specs).map(Some);
        }
        Ok(None)
    }

    fn shadow_with_exact_joint_designs(
        &self,
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Self>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        Ok(Some(Self {
            y: self.y.clone(),
            weights: self.weights.clone(),
            link_kind: self.link_kind.clone(),
            threshold_design: Some(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                x_t.into_owned(),
            ))),
            log_sigma_design: Some(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                x_ls.into_owned(),
            ))),
            wiggle_knots: self.wiggle_knots.clone(),
            wiggle_degree: self.wiggle_degree,
            policy: self.policy.clone(),
        }))
    }

    fn exact_newton_joint_psi_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psi_terms_from_designs(
            block_states,
            derivative_blocks,
            psi_index,
            &x_t,
            &x_ls,
        )
    }

    fn exact_newton_joint_psisecond_order_terms_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psisecond_order_terms_from_designs(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            &x_t,
            &x_ls,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative_for_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        self.exact_newton_joint_psihessian_directional_derivative_from_designs(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            &x_t,
            &x_ls,
        )
    }

    fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
        policy: &crate::resource::ResourcePolicy,
    ) -> Result<Option<LocationScaleJointPsiDirection>, String> {
        let Some(parts) = locscale_joint_psi_direction_parts(
            block_states,
            derivative_blocks,
            psi_index,
            self.y.len(),
            x_t.ncols(),
            x_ls.ncols(),
            Self::BLOCK_T,
            Self::BLOCK_LOG_SIGMA,
            3,
            "BinomialLocationScaleWiggleFamily",
            "threshold",
            policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(LocationScaleJointPsiDirection {
            block_idx: parts.block_idx,
            local_idx: parts.local_idx,
            z_primary_psi: parts.primary_z,
            z_ls_psi: parts.log_sigma_z,
            x_primary_psi: parts.primary_psi,
            x_ls_psi: parts.log_sigma_psi,
        }))
    }

    fn exact_newton_joint_psisecond_design_drifts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_a: &LocationScaleJointPsiDirection,
        psi_b: &LocationScaleJointPsiDirection,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<LocationScaleJointPsiSecondDrifts, String> {
        locscale_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            psi_a,
            psi_b,
            LocScalePsiDriftConfig {
                n: self.y.len(),
                p_primary: x_t.ncols(),
                p_log_sigma: x_ls.ncols(),
                primary_block_idx: Self::BLOCK_T,
                log_sigma_block_idx: Self::BLOCK_LOG_SIGMA,
                family_name: "BinomialLocationScaleWiggleFamily",
                primary_label: "threshold",
                policy: &self.policy,
            },
        )
    }

    fn exact_newton_joint_psi_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        if self
            .exact_newton_joint_psi_direction(
                block_states,
                derivative_blocks,
                psi_index,
                x_t,
                x_ls,
                &self.policy,
            )?
            .is_none()
        {
            return Ok(None);
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let base_core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(base_core.q0.view())?;
        let d0 =
            self.wiggle_basiswith_options(base_core.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(base_core.q0.view(), BasisOptions::second_derivative())?;
        let d3q = self.wiggle_d3q_dq03(base_core.q0.view(), betaw.view())?;
        let m = d0.dot(betaw) + 1.0;
        let g2 = self.wiggle_d2q_dq02(base_core.q0.view(), betaw.view())?;
        let g3 = d3q;
        let (sigma, ..) = exp_sigma_derivs_up_to_third(eta_ls.view());

        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let pw = b0.ncols();
        let total = pt + pls + pw;
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let (z_t_psi, z_ls_psi) = (&dir_a.z_primary_psi, &dir_a.z_ls_psi);
        let mut objective_psi = 0.0;

        let mut score_t_xa = Array1::<f64>::zeros(n);
        let mut score_t_x = Array1::<f64>::zeros(n);
        let mut score_ls_xa = Array1::<f64>::zeros(n);
        let mut score_ls_x = Array1::<f64>::zeros(n);
        let mut score_w_b = Array1::<f64>::zeros(n);
        let mut score_w_d1 = Array1::<f64>::zeros(n);

        let mut coeff_tt_w = Array1::<f64>::zeros(n);
        let mut coeff_tt_d = Array1::<f64>::zeros(n);
        let mut coeff_tl_w = Array1::<f64>::zeros(n);
        let mut coeff_tl_d = Array1::<f64>::zeros(n);
        let mut coeff_ll_w = Array1::<f64>::zeros(n);
        let mut coeff_ll_d = Array1::<f64>::zeros(n);
        let mut coeff_tw_b_w = Array1::<f64>::zeros(n);
        let mut coeff_tw_b_d = Array1::<f64>::zeros(n);
        let mut coeff_tw_d1_w = Array1::<f64>::zeros(n);
        let mut coeff_tw_d1_d = Array1::<f64>::zeros(n);
        let mut coeff_tw_d2_d = Array1::<f64>::zeros(n);
        let mut coeff_lw_b_w = Array1::<f64>::zeros(n);
        let mut coeff_lw_b_d = Array1::<f64>::zeros(n);
        let mut coeff_lw_d1_w = Array1::<f64>::zeros(n);
        let mut coeff_lw_d1_d = Array1::<f64>::zeros(n);
        let mut coeff_lw_d2_d = Array1::<f64>::zeros(n);
        let mut coeff_ww_bb = Array1::<f64>::zeros(n);
        let mut coeff_ww_db = Array1::<f64>::zeros(n);

        // Exact likelihood-only joint psi terms for the probit wiggle family.
        //
        // This helper is intentionally the same generic rowwise kernel as the
        // non-wiggle family. The only difference is the location-side row:
        //
        //   gamma = [beta_t; betaw],
        //   delta = beta_ls,
        //   z_r   = [x_{t,r}; B_r(q0)],
        //   x_r   = x_{ls,r},
        //   a_r   = z_r^T gamma,
        //   ell_r = x_r^T delta,
        //   q_r   = -a_r * exp(-ell_r).
        //
        // In this wiggle family we realize the same kernel through the chain
        //
        //   q = q0 + betaw^T B(q0),
        //   q0 = -eta_t * exp(-eta_ls),
        //   m  = dq/dq0   = 1 + betaw^T B'(q0),
        //   g2 = d²q/dq0² = betaw^T B''(q0),
        //   g3 = d³q/dq0³ = betaw^T B'''(q0).
        //
        // For a realized hyperdirection psi_a:
        //
        //   h_a     = q_{psi_a},
        //   c_a     = q_{beta psi_a},
        //   R_a     = q_{beta beta psi_a},
        //
        // and the generic scalar-loss identities are
        //
        //   D_a            = sum_r r_r h_{r,a},
        //   D_{beta a}     = sum_r [ w_r h_{r,a} b_r + r_r c_{r,a} ],
        //   D_{beta beta a}
        //                  = sum_r [ nu_r h_{r,a} b_r b_r^T
        //                              + w_r(c_{r,a} b_r^T + b_r c_{r,a}^T + h_{r,a} Q_r)
        //                              + r_r R_{r,a} ].
        //
        // Generic exact-joint code adds all realized penalty motion S_a after
        // the fact, so this family hook must stay likelihood-only.
        //
        // The rowwise objects below are the wiggle specialization of the same
        // q_r = -a_r exp(-ell_r) kernel. All wiggle-specific complexity is
        // localized to the realized row B_r(q0) and its q0-derivatives.
        for row in 0..n {
            let q0 = base_core.q0[row];
            let q = q0 + etaw[row];
            let q0_geom = nonwiggle_q_derivs(eta_t[row], sigma[row]);
            let r_sigma = 1.0 / sigma[row];
            let q0_a = -r_sigma * z_t_psi[row] - q0 * z_ls_psi[row];
            let q0_t_a = q0_geom.q_tl * z_ls_psi[row];
            let q0_ls_a = q0_geom.q_tl * z_t_psi[row] + q0_geom.q_ll * z_ls_psi[row];
            let q0_tl_a = q0_geom.q_tl_ls * z_ls_psi[row];
            let q0_ll_a = q0_geom.q_tl_ls * z_t_psi[row] + q0_geom.q_ll_ls * z_ls_psi[row];

            let q_t = m[row] * q0_geom.q_t;
            let q_ls = m[row] * q0_geom.q_ls;
            let q_tt = g2[row] * q0_geom.q_t * q0_geom.q_t;
            let q_tl = g2[row] * q0_geom.q_t * q0_geom.q_ls + m[row] * q0_geom.q_tl;
            let q_ll = g2[row] * q0_geom.q_ls * q0_geom.q_ls + m[row] * q0_geom.q_ll;
            let q_t_a = g2[row] * q0_a * q0_geom.q_t + m[row] * q0_t_a;
            let q_ls_a = g2[row] * q0_a * q0_geom.q_ls + m[row] * q0_ls_a;
            let q_tt_a =
                g3[row] * q0_a * q0_geom.q_t * q0_geom.q_t + g2[row] * (2.0 * q0_geom.q_t * q0_t_a);
            let q_tl_a = g3[row] * q0_a * q0_geom.q_t * q0_geom.q_ls
                + g2[row] * (q0_t_a * q0_geom.q_ls + q0_geom.q_t * q0_ls_a + q0_a * q0_geom.q_tl)
                + m[row] * q0_tl_a;
            let q_ll_a = g3[row] * q0_a * q0_geom.q_ls * q0_geom.q_ls
                + g2[row] * (2.0 * q0_geom.q_ls * q0_ls_a + q0_a * q0_geom.q_ll)
                + m[row] * q0_ll_a;

            let (loss_1, loss_2, loss_3) = binomial_neglog_q_derivatives_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            );
            let alpha = m[row] * q0_a;
            objective_psi += loss_1 * alpha;

            score_t_xa[row] = loss_1 * q_t;
            score_t_x[row] = loss_2 * alpha * q_t + loss_1 * q_t_a;
            score_ls_xa[row] = loss_1 * q_ls;
            score_ls_x[row] = loss_2 * alpha * q_ls + loss_1 * q_ls_a;
            score_w_b[row] = loss_2 * alpha;
            score_w_d1[row] = loss_1 * q0_a;

            coeff_tt_w[row] = loss_2 * q_t * q_t + loss_1 * q_tt;
            coeff_tt_d[row] = loss_3 * alpha * q_t * q_t
                + 2.0 * loss_2 * q_t * q_t_a
                + loss_2 * alpha * q_tt
                + loss_1 * q_tt_a;
            coeff_tl_w[row] = loss_2 * q_t * q_ls + loss_1 * q_tl;
            coeff_tl_d[row] = loss_3 * alpha * q_t * q_ls
                + loss_2 * (q_t_a * q_ls + q_t * q_ls_a)
                + loss_2 * alpha * q_tl
                + loss_1 * q_tl_a;
            coeff_ll_w[row] = loss_2 * q_ls * q_ls + loss_1 * q_ll;
            coeff_ll_d[row] = loss_3 * alpha * q_ls * q_ls
                + 2.0 * loss_2 * q_ls * q_ls_a
                + loss_2 * alpha * q_ll
                + loss_1 * q_ll_a;

            coeff_tw_b_w[row] = loss_2 * q_t;
            coeff_tw_b_d[row] = loss_3 * alpha * q_t + loss_2 * q_t_a;
            coeff_tw_d1_w[row] = loss_1 * q0_geom.q_t;
            coeff_tw_d1_d[row] = loss_2 * (q_t * q0_a + alpha * q0_geom.q_t) + loss_1 * q0_t_a;
            coeff_tw_d2_d[row] = loss_1 * q0_a * q0_geom.q_t;

            coeff_lw_b_w[row] = loss_2 * q_ls;
            coeff_lw_b_d[row] = loss_3 * alpha * q_ls + loss_2 * q_ls_a;
            coeff_lw_d1_w[row] = loss_1 * q0_geom.q_ls;
            coeff_lw_d1_d[row] = loss_2 * (q_ls * q0_a + alpha * q0_geom.q_ls) + loss_1 * q0_ls_a;
            coeff_lw_d2_d[row] = loss_1 * q0_a * q0_geom.q_ls;

            coeff_ww_bb[row] = loss_3 * alpha;
            coeff_ww_db[row] = loss_2 * q0_a;
        }
        let x_t_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let score_t = x_t_map.transpose_mul(score_t_xa.view()) + fast_atv(x_t, &score_t_x);
        let score_ls = x_ls_map.transpose_mul(score_ls_xa.view()) + fast_atv(x_ls, &score_ls_x);
        let score_w = fast_atv(&b0, &score_w_b) + fast_atv(&d0, &score_w_d1);
        let mut score_psi = Array1::<f64>::zeros(total);
        score_psi.slice_mut(s![0..pt]).assign(&score_t);
        score_psi.slice_mut(s![pt..pt + pls]).assign(&score_ls);
        score_psi.slice_mut(s![pt + pls..total]).assign(&score_w);

        let x_t_action_opt = dir_a.x_primary_psi.cloned_first_action();
        let x_ls_action_opt = dir_a.x_ls_psi.cloned_first_action();
        if x_t_action_opt.is_some() || x_ls_action_opt.is_some() {
            let basis_arc = Arc::new(b0.clone());
            let basis_d1_arc = Arc::new(d0.clone());
            let basis_d2_arc = Arc::new(dd0.clone());
            let zeros = Array1::<f64>::zeros(n);
            let operator = CustomFamilyJointPsiOperator::new(
                total,
                vec![
                    CustomFamilyJointDesignChannel::new(
                        0..pt,
                        shared_dense_arc(x_t),
                        x_t_action_opt,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        pt..pt + pls,
                        shared_dense_arc(x_ls),
                        x_ls_action_opt,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        pt + pls..total,
                        Arc::clone(&basis_arc),
                        None,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        pt + pls..total,
                        Arc::clone(&basis_d1_arc),
                        None,
                    ),
                    CustomFamilyJointDesignChannel::new(
                        pt + pls..total,
                        Arc::clone(&basis_d2_arc),
                        None,
                    ),
                ],
                vec![
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        0,
                        coeff_tt_w.clone(),
                        coeff_tt_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        1,
                        coeff_tl_w.clone(),
                        coeff_tl_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        0,
                        coeff_tl_w.clone(),
                        coeff_tl_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        1,
                        coeff_ll_w.clone(),
                        coeff_ll_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        2,
                        coeff_tw_b_w.clone(),
                        coeff_tw_b_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        2,
                        0,
                        coeff_tw_b_w.clone(),
                        coeff_tw_b_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        3,
                        coeff_tw_d1_w.clone(),
                        coeff_tw_d1_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        3,
                        0,
                        coeff_tw_d1_w.clone(),
                        coeff_tw_d1_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        0,
                        4,
                        zeros.clone(),
                        coeff_tw_d2_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        4,
                        0,
                        zeros.clone(),
                        coeff_tw_d2_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        2,
                        coeff_lw_b_w.clone(),
                        coeff_lw_b_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        2,
                        1,
                        coeff_lw_b_w.clone(),
                        coeff_lw_b_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        3,
                        coeff_lw_d1_w.clone(),
                        coeff_lw_d1_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        3,
                        1,
                        coeff_lw_d1_w.clone(),
                        coeff_lw_d1_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        1,
                        4,
                        zeros.clone(),
                        coeff_lw_d2_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        4,
                        1,
                        zeros.clone(),
                        coeff_lw_d2_d.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        2,
                        2,
                        zeros.clone(),
                        coeff_ww_bb.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(
                        3,
                        2,
                        zeros.clone(),
                        coeff_ww_db.clone(),
                    ),
                    CustomFamilyJointDesignPairContribution::new(2, 3, zeros, coeff_ww_db.clone()),
                ],
            );
            return Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
                objective_psi,
                score_psi,
                hessian_psi: Array2::zeros((0, 0)),
                hessian_psi_operator: Some(std::sync::Arc::new(operator)),
            }));
        }
        let h_tt_block = weighted_crossprod_psi_maps(
            x_t_map,
            coeff_tt_w.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_t),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_t),
            coeff_tt_w.view(),
            x_t_map,
        )? + &xt_diag_x_dense(x_t, &coeff_tt_d)?;
        let h_tl_block = weighted_crossprod_psi_maps(
            x_t_map,
            coeff_tl_w.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_t),
            coeff_tl_w.view(),
            x_ls_map,
        )? + &xt_diag_y_dense(x_t, &coeff_tl_d, x_ls)?;
        let h_ll_block = weighted_crossprod_psi_maps(
            x_ls_map,
            coeff_ll_w.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_ls),
            coeff_ll_w.view(),
            x_ls_map,
        )? + &xt_diag_x_dense(x_ls, &coeff_ll_d)?;
        let h_tw = weighted_crossprod_psi_maps(
            x_t_map,
            coeff_tw_b_w.view(),
            CustomFamilyPsiLinearMapRef::Dense(&b0),
        )? + &xt_diag_y_dense(x_t, &coeff_tw_b_d, &b0)?
            + &weighted_crossprod_psi_maps(
                x_t_map,
                coeff_tw_d1_w.view(),
                CustomFamilyPsiLinearMapRef::Dense(&d0),
            )?
            + &xt_diag_y_dense(x_t, &coeff_tw_d1_d, &d0)?
            + &xt_diag_y_dense(x_t, &coeff_tw_d2_d, &dd0)?;
        let h_lw = weighted_crossprod_psi_maps(
            x_ls_map,
            coeff_lw_b_w.view(),
            CustomFamilyPsiLinearMapRef::Dense(&b0),
        )? + &xt_diag_y_dense(x_ls, &coeff_lw_b_d, &b0)?
            + &weighted_crossprod_psi_maps(
                x_ls_map,
                coeff_lw_d1_w.view(),
                CustomFamilyPsiLinearMapRef::Dense(&d0),
            )?
            + &xt_diag_y_dense(x_ls, &coeff_lw_d1_d, &d0)?
            + &xt_diag_y_dense(x_ls, &coeff_lw_d2_d, &dd0)?;
        let a_ww = xt_diag_y_dense(&d0, &coeff_ww_db, &b0)?;
        let h_ww = xt_diag_x_dense(&b0, &coeff_ww_bb)? + &a_ww + a_ww.t();

        let mut hessian_psi = Array2::<f64>::zeros((total, total));
        hessian_psi.slice_mut(s![0..pt, 0..pt]).assign(&h_tt_block);
        hessian_psi
            .slice_mut(s![0..pt, pt..pt + pls])
            .assign(&h_tl_block);
        hessian_psi
            .slice_mut(s![pt..pt + pls, pt..pt + pls])
            .assign(&h_ll_block);
        hessian_psi
            .slice_mut(s![0..pt, pt + pls..total])
            .assign(&h_tw);
        hessian_psi
            .slice_mut(s![pt..pt + pls, pt + pls..total])
            .assign(&h_lw);
        hessian_psi
            .slice_mut(s![pt + pls..total, pt + pls..total])
            .assign(&h_ww);
        mirror_upper_to_lower(&mut hessian_psi);

        Ok(Some(crate::custom_family::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator: None,
        }))
    }

    fn exact_newton_joint_psisecond_order_terms_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        if block_states.len() != 3 || derivative_blocks.len() != 3 {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "BinomialLocationScaleWiggleFamily joint psi second-order terms expect 3 blocks and 3 derivative block lists, got {} and {}",
                block_states.len(),
                derivative_blocks.len()
            ) }.into());
        }
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_i,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        let Some(dir_b) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_j,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psisecond_order_terms_from_parts(
                block_states,
                derivative_blocks,
                &dir_a,
                &dir_b,
                x_t,
                x_ls,
            )?,
        ))
    }

    fn exact_newton_joint_psisecond_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        dir_a: &LocationScaleJointPsiDirection,
        dir_b: &LocationScaleJointPsiDirection,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms, String> {
        let second_drifts = self.exact_newton_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            dir_a,
            dir_b,
            x_t,
            x_ls,
        )?;
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let base_core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(base_core.q0.view())?;
        let d0 =
            self.wiggle_basiswith_options(base_core.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(base_core.q0.view(), BasisOptions::second_derivative())?;
        let d3_basis = self.wiggle_d3basis_constrained(base_core.q0.view())?;
        let d3q = self.wiggle_d3q_dq03(base_core.q0.view(), betaw.view())?;
        let d4q = self.wiggle_d4q_dq04(base_core.q0.view(), betaw.view())?;
        if b0.ncols() != betaw.len()
            || d0.ncols() != betaw.len()
            || dd0.ncols() != betaw.len()
            || d3_basis.ncols() != betaw.len()
        {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle derivative/beta mismatch in joint psi psi terms: B={} B'={} B''={} B'''={} betaw={}",
                b0.ncols(),
                d0.ncols(),
                dd0.ncols(),
                d3_basis.ncols(),
                betaw.len()
            ) }.into());
        }
        let m = d0.dot(betaw) + 1.0;
        let g2 = dd0.dot(betaw);
        let g3 = d3q;
        let g4 = d4q;
        let (sigma, ds, d2s, d3s) = exp_sigma_derivs_up_to_third(eta_ls.view());

        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let pw = b0.ncols();
        let total = pt + pls + pw;
        let x_t_a_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_t_b_map = dir_b.x_primary_psi.as_linear_map_ref();
        let x_ls_a_map = dir_a.x_ls_psi.as_linear_map_ref();
        let x_ls_b_map = dir_b.x_ls_psi.as_linear_map_ref();
        let x_t_ab_map = second_psi_linear_map(
            second_drifts.x_primary_ab_action.as_ref(),
            second_drifts.x_primary_ab.as_ref(),
            n,
            pt,
        );
        let x_ls_ab_map = second_psi_linear_map(
            second_drifts.x_ls_ab_action.as_ref(),
            second_drifts.x_ls_ab.as_ref(),
            n,
            pls,
        );
        let mut objective_psi_psi = 0.0;
        let mut score_psi_psi = Array1::<f64>::zeros(total);
        let mut hessian_psi_psi = Array2::<f64>::zeros((total, total));

        // Likelihood-only exact psi/psi terms for the wiggle family.
        //
        // This is the same generic second-order kernel as the non-wiggle path,
        // still over the flattened coefficients beta = [beta_t; beta_ls; betaw].
        // The family provides only the likelihood-side fixed-beta objects
        //
        //   D_ab, D_{beta ab}, D_{beta beta ab},
        //
        // while generic exact-joint code in custom_family.rs adds all realized
        // penalty motion S_ab.
        //
        // Using the generic rowwise notation
        //
        //   h_a   = q_{psi_a},      h_b   = q_{psi_b},
        //   h_ab  = q_{psi_a psi_b},
        //   c_a   = q_{beta psi_a}, c_b   = q_{beta psi_b},
        //   c_ab  = q_{beta psi_a psi_b},
        //   R_a   = q_{beta beta psi_a},
        //   R_b   = q_{beta beta psi_b},
        //   R_ab  = q_{beta beta psi_a psi_b},
        //
        // the exact scalar-loss kernel is
        //
        //   D_ab
        //   = sum_r [ w_r h_{r,a} h_{r,b} + r_r h_{r,ab} ],
        //
        //   D_{beta ab}
        //   = sum_r [
        //       r_r c_{r,ab}
        //       + w_r h_{r,b} c_{r,a}
        //       + w_r h_{r,a} c_{r,b}
        //       + (w_r h_{r,ab} + nu_r h_{r,a} h_{r,b}) b_r
        //     ],
        //
        //   D_{beta beta ab}
        //   = sum_r [
        //       r_r R_{r,ab}
        //       + w_r h_{r,b} R_{r,a}
        //       + w_r h_{r,a} R_{r,b}
        //       + w_r(c_{r,ab} b_r^T + b_r c_{r,ab}^T
        //             + c_{r,a} c_{r,b}^T + c_{r,b} c_{r,a}^T
        //             + h_{r,ab} Q_r)
        //       + nu_r h_{r,b}(c_{r,a} b_r^T + b_r c_{r,a}^T)
        //       + nu_r h_{r,a}(c_{r,b} b_r^T + b_r c_{r,b}^T)
        //       + nu_r h_{r,a} h_{r,b} Q_r
        //       + (tau_r h_{r,a} h_{r,b} + nu_r h_{r,ab}) b_r b_r^T
        //     ].
        //
        // The wiggle specialization enters only through the rowwise q-objects
        // built below from the combined location-side row z_r = [x_{t,r}; B_r(q0)].
        let mut b = Array1::<f64>::zeros(total);
        let mut c_a = Array1::<f64>::zeros(total);
        let mut c_b = Array1::<f64>::zeros(total);
        let mut c_ab = Array1::<f64>::zeros(total);
        let mut q_mat = Array2::<f64>::zeros((total, total));
        let mut r_a = Array2::<f64>::zeros((total, total));
        let mut r_b = Array2::<f64>::zeros((total, total));
        let mut r_ab = Array2::<f64>::zeros((total, total));
        let mut qw_a = Array1::<f64>::zeros(pw);
        let mut qw_b = Array1::<f64>::zeros(pw);
        let mut qw_ab = Array1::<f64>::zeros(pw);
        let mut q_tw_a = Array1::<f64>::zeros(pw);
        let mut q_tw_b = Array1::<f64>::zeros(pw);
        let mut q_lw_a = Array1::<f64>::zeros(pw);
        let mut q_lw_b = Array1::<f64>::zeros(pw);
        let mut d0_ab = Array1::<f64>::zeros(pw);
        let mut q_tw_ab = Array1::<f64>::zeros(pw);
        let mut q_lw_ab = Array1::<f64>::zeros(pw);
        for row in 0..n {
            let q0 = base_core.q0[row];
            let q = q0 + etaw[row];
            let q0_geom = nonwiggle_q_derivs(eta_t[row], sigma[row]);
            let s_safe = sigma[row];
            let s2 = s_safe * s_safe;
            let s3 = s2 * s_safe;
            let s4 = s3 * s_safe;
            let q0_tl_ls_ls =
                d3s[row] / s2 - 6.0 * ds[row] * d2s[row] / s3 + 6.0 * ds[row].powi(3) / s4;
            let r_sigma = 1.0 / s_safe;

            let q0_a = -r_sigma * dir_a.z_primary_psi[row] - q0 * dir_a.z_ls_psi[row];
            let q0_b = -r_sigma * dir_b.z_primary_psi[row] - q0 * dir_b.z_ls_psi[row];
            let q0_ab = -r_sigma * second_drifts.z_primary_ab[row]
                + r_sigma
                    * (dir_a.z_primary_psi[row] * dir_b.z_ls_psi[row]
                        + dir_b.z_primary_psi[row] * dir_a.z_ls_psi[row])
                + q0 * (dir_a.z_ls_psi[row] * dir_b.z_ls_psi[row] - second_drifts.z_ls_ab[row]);

            let q0_t_a = q0_geom.q_tl * dir_a.z_ls_psi[row];
            let q0_t_b = q0_geom.q_tl * dir_b.z_ls_psi[row];
            let q0_t_ab = q0_geom.q_tl_ls * dir_a.z_ls_psi[row] * dir_b.z_ls_psi[row]
                + q0_geom.q_tl * second_drifts.z_ls_ab[row];
            let q0_ls_a =
                q0_geom.q_tl * dir_a.z_primary_psi[row] + q0_geom.q_ll * dir_a.z_ls_psi[row];
            let q0_ls_b =
                q0_geom.q_tl * dir_b.z_primary_psi[row] + q0_geom.q_ll * dir_b.z_ls_psi[row];
            let q0_ls_ab = -q0_ab;
            let q0_tl_a = q0_geom.q_tl_ls * dir_a.z_ls_psi[row];
            let q0_tl_b = q0_geom.q_tl_ls * dir_b.z_ls_psi[row];
            let q0_tl_ab = q0_tl_ls_ls * dir_a.z_ls_psi[row] * dir_b.z_ls_psi[row]
                + q0_geom.q_tl_ls * second_drifts.z_ls_ab[row];
            let q0_ll_a =
                q0_geom.q_tl_ls * dir_a.z_primary_psi[row] + q0_geom.q_ll_ls * dir_a.z_ls_psi[row];
            let q0_ll_b =
                q0_geom.q_tl_ls * dir_b.z_primary_psi[row] + q0_geom.q_ll_ls * dir_b.z_ls_psi[row];
            let q0_ll_ab = q0_ab;

            let m_a = g2[row] * q0_a;
            let m_b = g2[row] * q0_b;
            let m_ab = g3[row] * q0_a * q0_b + g2[row] * q0_ab;
            let g2_a = g3[row] * q0_a;
            let g2_b = g3[row] * q0_b;
            let g2_ab = g4[row] * q0_a * q0_b + g3[row] * q0_ab;

            let q_a = m[row] * q0_a;
            let q_b = m[row] * q0_b;
            let q_ab = m[row] * q0_ab + g2[row] * q0_a * q0_b;
            let q_t = m[row] * q0_geom.q_t;
            let q_ls = m[row] * q0_geom.q_ls;
            let q_tt = g2[row] * q0_geom.q_t * q0_geom.q_t;
            let q_tl = g2[row] * q0_geom.q_t * q0_geom.q_ls + m[row] * q0_geom.q_tl;
            let q_ll = g2[row] * q0_geom.q_ls * q0_geom.q_ls + m[row] * q0_geom.q_ll;
            let q_t_a = m_a * q0_geom.q_t + m[row] * q0_t_a;
            let q_t_b = m_b * q0_geom.q_t + m[row] * q0_t_b;
            let q_ls_a = m_a * q0_geom.q_ls + m[row] * q0_ls_a;
            let q_ls_b = m_b * q0_geom.q_ls + m[row] * q0_ls_b;
            let q_t_ab = m_ab * q0_geom.q_t + m_a * q0_t_b + m_b * q0_t_a + m[row] * q0_t_ab;
            let q_ls_ab = m_ab * q0_geom.q_ls + m_a * q0_ls_b + m_b * q0_ls_a + m[row] * q0_ls_ab;
            let q_tt_a = g2_a * q0_geom.q_t * q0_geom.q_t + g2[row] * 2.0 * q0_geom.q_t * q0_t_a;
            let q_tt_b = g2_b * q0_geom.q_t * q0_geom.q_t + g2[row] * 2.0 * q0_geom.q_t * q0_t_b;
            let q_tt_ab = g2_ab * q0_geom.q_t * q0_geom.q_t
                + g2_a * 2.0 * q0_geom.q_t * q0_t_b
                + g2_b * 2.0 * q0_geom.q_t * q0_t_a
                + g2[row] * (2.0 * q0_t_a * q0_t_b + 2.0 * q0_geom.q_t * q0_t_ab);
            let q_tl_a = g2_a * q0_geom.q_t * q0_geom.q_ls
                + g2[row] * (q0_t_a * q0_geom.q_ls + q0_geom.q_t * q0_ls_a)
                + m_a * q0_geom.q_tl
                + m[row] * q0_tl_a;
            let q_tl_b = g2_b * q0_geom.q_t * q0_geom.q_ls
                + g2[row] * (q0_t_b * q0_geom.q_ls + q0_geom.q_t * q0_ls_b)
                + m_b * q0_geom.q_tl
                + m[row] * q0_tl_b;
            let q_tl_ab = g2_ab * q0_geom.q_t * q0_geom.q_ls
                + g2_a * (q0_t_b * q0_geom.q_ls + q0_geom.q_t * q0_ls_b)
                + g2_b * (q0_t_a * q0_geom.q_ls + q0_geom.q_t * q0_ls_a)
                + g2[row]
                    * (q0_t_ab * q0_geom.q_ls
                        + q0_t_a * q0_ls_b
                        + q0_t_b * q0_ls_a
                        + q0_geom.q_t * q0_ls_ab)
                + m_ab * q0_geom.q_tl
                + m_a * q0_tl_b
                + m_b * q0_tl_a
                + m[row] * q0_tl_ab;
            let q_ll_a = g2_a * q0_geom.q_ls * q0_geom.q_ls
                + g2[row] * 2.0 * q0_geom.q_ls * q0_ls_a
                + m_a * q0_geom.q_ll
                + m[row] * q0_ll_a;
            let q_ll_b = g2_b * q0_geom.q_ls * q0_geom.q_ls
                + g2[row] * 2.0 * q0_geom.q_ls * q0_ls_b
                + m_b * q0_geom.q_ll
                + m[row] * q0_ll_b;
            let q_ll_ab = g2_ab * q0_geom.q_ls * q0_geom.q_ls
                + g2_a * 2.0 * q0_geom.q_ls * q0_ls_b
                + g2_b * 2.0 * q0_geom.q_ls * q0_ls_a
                + g2[row] * (2.0 * q0_ls_a * q0_ls_b + 2.0 * q0_geom.q_ls * q0_ls_ab)
                + m_ab * q0_geom.q_ll
                + m_a * q0_ll_b
                + m_b * q0_ll_a
                + m[row] * q0_ll_ab;

            let brow = b0.row(row);
            let drow = d0.row(row);
            let ddrow = dd0.row(row);
            let d3row = d3_basis.row(row);
            qw_a.fill(0.0);
            qw_a.scaled_add(q0_a, &drow);
            qw_b.fill(0.0);
            qw_b.scaled_add(q0_b, &drow);
            qw_ab.fill(0.0);
            qw_ab.scaled_add(q0_a * q0_b, &ddrow);
            qw_ab.scaled_add(q0_ab, &drow);
            q_tw_a.fill(0.0);
            q_tw_a.scaled_add(q0_a * q0_geom.q_t, &ddrow);
            q_tw_a.scaled_add(q0_t_a, &drow);
            q_tw_b.fill(0.0);
            q_tw_b.scaled_add(q0_b * q0_geom.q_t, &ddrow);
            q_tw_b.scaled_add(q0_t_b, &drow);
            q_lw_a.fill(0.0);
            q_lw_a.scaled_add(q0_a * q0_geom.q_ls, &ddrow);
            q_lw_a.scaled_add(q0_ls_a, &drow);
            q_lw_b.fill(0.0);
            q_lw_b.scaled_add(q0_b * q0_geom.q_ls, &ddrow);
            q_lw_b.scaled_add(q0_ls_b, &drow);
            d0_ab.fill(0.0);
            d0_ab.scaled_add(q0_a * q0_b, &d3row);
            d0_ab.scaled_add(q0_ab, &ddrow);
            q_tw_ab.fill(0.0);
            q_tw_ab.scaled_add(q0_geom.q_t, &d0_ab);
            q_tw_ab.scaled_add(q0_b * q0_t_a, &ddrow);
            q_tw_ab.scaled_add(q0_a * q0_t_b, &ddrow);
            q_tw_ab.scaled_add(q0_t_ab, &drow);
            q_lw_ab.fill(0.0);
            q_lw_ab.scaled_add(q0_geom.q_ls, &d0_ab);
            q_lw_ab.scaled_add(q0_b * q0_ls_a, &ddrow);
            q_lw_ab.scaled_add(q0_a * q0_ls_b, &ddrow);
            q_lw_ab.scaled_add(q0_ls_ab, &drow);

            let (loss_1, loss_2, loss_3) = binomial_neglog_q_derivatives_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            );
            let loss_4 = binomial_neglog_q_fourth_derivative_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            )?;
            objective_psi_psi += loss_2 * q_a * q_b + loss_1 * q_ab;

            let xtr = x_t.row(row);
            let xlsr = x_ls.row(row);
            let xta = x_t_a_map.row_vector(row)?;
            let xtb = x_t_b_map.row_vector(row)?;
            let xlsa = x_ls_a_map.row_vector(row)?;
            let xlsb = x_ls_b_map.row_vector(row)?;
            let xtab = x_t_ab_map.row_vector(row)?;
            let xlsab = x_ls_ab_map.row_vector(row)?;

            b.fill(0.0);
            b.slice_mut(s![0..pt]).scaled_add(q_t, &xtr);
            b.slice_mut(s![pt..pt + pls]).scaled_add(q_ls, &xlsr);
            b.slice_mut(s![pt + pls..]).assign(&brow);
            c_a.fill(0.0);
            c_a.slice_mut(s![0..pt]).scaled_add(q_t_a, &xtr);
            c_a.slice_mut(s![0..pt]).scaled_add(q_t, &xta.view());
            c_a.slice_mut(s![pt..pt + pls]).scaled_add(q_ls_a, &xlsr);
            c_a.slice_mut(s![pt..pt + pls])
                .scaled_add(q_ls, &xlsa.view());
            c_a.slice_mut(s![pt + pls..]).assign(&qw_a);
            c_b.fill(0.0);
            c_b.slice_mut(s![0..pt]).scaled_add(q_t_b, &xtr);
            c_b.slice_mut(s![0..pt]).scaled_add(q_t, &xtb.view());
            c_b.slice_mut(s![pt..pt + pls]).scaled_add(q_ls_b, &xlsr);
            c_b.slice_mut(s![pt..pt + pls])
                .scaled_add(q_ls, &xlsb.view());
            c_b.slice_mut(s![pt + pls..]).assign(&qw_b);
            c_ab.fill(0.0);
            c_ab.slice_mut(s![0..pt]).scaled_add(q_t_ab, &xtr);
            c_ab.slice_mut(s![0..pt]).scaled_add(q_t_b, &xta.view());
            c_ab.slice_mut(s![0..pt]).scaled_add(q_t_a, &xtb.view());
            c_ab.slice_mut(s![0..pt]).scaled_add(q_t, &xtab.view());
            c_ab.slice_mut(s![pt..pt + pls]).scaled_add(q_ls_ab, &xlsr);
            c_ab.slice_mut(s![pt..pt + pls])
                .scaled_add(q_ls_b, &xlsa.view());
            c_ab.slice_mut(s![pt..pt + pls])
                .scaled_add(q_ls_a, &xlsb.view());
            c_ab.slice_mut(s![pt..pt + pls])
                .scaled_add(q_ls, &xlsab.view());
            c_ab.slice_mut(s![pt + pls..]).assign(&qw_ab);

            score_psi_psi.scaled_add(loss_1, &c_ab);
            score_psi_psi.scaled_add(loss_2 * q_b, &c_a);
            score_psi_psi.scaled_add(loss_2 * q_a, &c_b);
            score_psi_psi.scaled_add(loss_2 * q_ab + loss_3 * q_a * q_b, &b);

            q_mat.fill(0.0);
            r_a.fill(0.0);
            r_b.fill(0.0);
            r_ab.fill(0.0);
            scaled_outer_add(q_mat.slice_mut(s![0..pt, 0..pt]), q_tt, xtr, xtr);
            scaled_outer_add(q_mat.slice_mut(s![0..pt, pt..pt + pls]), q_tl, xtr, xlsr);
            scaled_outer_add(
                q_mat.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsr,
                xlsr,
            );
            scaled_outer_add(
                q_mat.slice_mut(s![0..pt, pt + pls..]),
                q0_geom.q_t,
                xtr,
                drow,
            );
            scaled_outer_add(
                q_mat.slice_mut(s![pt..pt + pls, pt + pls..]),
                q0_geom.q_ls,
                xlsr,
                drow,
            );
            mirror_upper_to_lower(&mut q_mat);

            scaled_outer_add(r_a.slice_mut(s![0..pt, 0..pt]), q_tt_a, xtr, xtr);
            scaled_outer_add(r_a.slice_mut(s![0..pt, 0..pt]), q_tt, xta.view(), xtr);
            scaled_outer_add(r_a.slice_mut(s![0..pt, 0..pt]), q_tt, xtr, xta.view());
            scaled_outer_add(r_a.slice_mut(s![0..pt, pt..pt + pls]), q_tl_a, xtr, xlsr);
            scaled_outer_add(
                r_a.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xta.view(),
                xlsr,
            );
            scaled_outer_add(
                r_a.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xtr,
                xlsa.view(),
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll_a,
                xlsr,
                xlsr,
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsa.view(),
                xlsr,
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsr,
                xlsa.view(),
            );
            scaled_outer_add(
                r_a.slice_mut(s![0..pt, pt + pls..]),
                q0_geom.q_t,
                xta.view(),
                drow,
            );
            scaled_outer_add(
                r_a.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xtr,
                q_tw_a.view(),
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt + pls..]),
                q0_geom.q_ls,
                xlsa.view(),
                drow,
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsr,
                q_lw_a.view(),
            );
            mirror_upper_to_lower(&mut r_a);

            scaled_outer_add(r_b.slice_mut(s![0..pt, 0..pt]), q_tt_b, xtr, xtr);
            scaled_outer_add(r_b.slice_mut(s![0..pt, 0..pt]), q_tt, xtb.view(), xtr);
            scaled_outer_add(r_b.slice_mut(s![0..pt, 0..pt]), q_tt, xtr, xtb.view());
            scaled_outer_add(r_b.slice_mut(s![0..pt, pt..pt + pls]), q_tl_b, xtr, xlsr);
            scaled_outer_add(
                r_b.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xtb.view(),
                xlsr,
            );
            scaled_outer_add(
                r_b.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xtr,
                xlsb.view(),
            );
            scaled_outer_add(
                r_b.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll_b,
                xlsr,
                xlsr,
            );
            scaled_outer_add(
                r_b.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsb.view(),
                xlsr,
            );
            scaled_outer_add(
                r_b.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsr,
                xlsb.view(),
            );
            scaled_outer_add(
                r_b.slice_mut(s![0..pt, pt + pls..]),
                q0_geom.q_t,
                xtb.view(),
                drow,
            );
            scaled_outer_add(
                r_b.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xtr,
                q_tw_b.view(),
            );
            scaled_outer_add(
                r_b.slice_mut(s![pt..pt + pls, pt + pls..]),
                q0_geom.q_ls,
                xlsb.view(),
                drow,
            );
            scaled_outer_add(
                r_b.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsr,
                q_lw_b.view(),
            );
            mirror_upper_to_lower(&mut r_b);

            scaled_outer_add(r_ab.slice_mut(s![0..pt, 0..pt]), q_tt_ab, xtr, xtr);
            scaled_outer_add(r_ab.slice_mut(s![0..pt, 0..pt]), q_tt_b, xta.view(), xtr);
            scaled_outer_add(r_ab.slice_mut(s![0..pt, 0..pt]), q_tt_b, xtr, xta.view());
            scaled_outer_add(r_ab.slice_mut(s![0..pt, 0..pt]), q_tt_a, xtb.view(), xtr);
            scaled_outer_add(r_ab.slice_mut(s![0..pt, 0..pt]), q_tt_a, xtr, xtb.view());
            scaled_outer_add(r_ab.slice_mut(s![0..pt, 0..pt]), q_tt, xtab.view(), xtr);
            scaled_outer_add(r_ab.slice_mut(s![0..pt, 0..pt]), q_tt, xtr, xtab.view());
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, 0..pt]),
                q_tt,
                xta.view(),
                xtb.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, 0..pt]),
                q_tt,
                xtb.view(),
                xta.view(),
            );

            scaled_outer_add(r_ab.slice_mut(s![0..pt, pt..pt + pls]), q_tl_ab, xtr, xlsr);
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl_b,
                xta.view(),
                xlsr,
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl_b,
                xtr,
                xlsa.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl_a,
                xtb.view(),
                xlsr,
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl_a,
                xtr,
                xlsb.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xtab.view(),
                xlsr,
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xtr,
                xlsab.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xta.view(),
                xlsb.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xtb.view(),
                xlsa.view(),
            );

            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll_ab,
                xlsr,
                xlsr,
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll_b,
                xlsa.view(),
                xlsr,
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll_b,
                xlsr,
                xlsa.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll_a,
                xlsb.view(),
                xlsr,
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll_a,
                xlsr,
                xlsb.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsab.view(),
                xlsr,
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsr,
                xlsab.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsa.view(),
                xlsb.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsb.view(),
                xlsa.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt + pls..]),
                q0_geom.q_t,
                xtab.view(),
                drow,
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xta.view(),
                q_tw_b.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xtb.view(),
                q_tw_a.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xtr,
                q_tw_ab.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt + pls..]),
                q0_geom.q_ls,
                xlsab.view(),
                drow,
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsa.view(),
                q_lw_b.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsb.view(),
                q_lw_a.view(),
            );
            scaled_outer_add(
                r_ab.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsr,
                q_lw_ab.view(),
            );
            mirror_upper_to_lower(&mut r_ab);

            hessian_psi_psi.scaled_add(loss_1, &r_ab);
            hessian_psi_psi.scaled_add(loss_2 * q_b, &r_a);
            hessian_psi_psi.scaled_add(loss_2 * q_a, &r_b);
            scaled_outer_add(hessian_psi_psi.view_mut(), loss_2, c_ab.view(), b.view());
            scaled_outer_add(hessian_psi_psi.view_mut(), loss_2, b.view(), c_ab.view());
            scaled_outer_add(hessian_psi_psi.view_mut(), loss_2, c_a.view(), c_b.view());
            scaled_outer_add(hessian_psi_psi.view_mut(), loss_2, c_b.view(), c_a.view());
            hessian_psi_psi.scaled_add(loss_2 * q_ab, &q_mat);
            scaled_outer_add(
                hessian_psi_psi.view_mut(),
                loss_3 * q_b,
                c_a.view(),
                b.view(),
            );
            scaled_outer_add(
                hessian_psi_psi.view_mut(),
                loss_3 * q_b,
                b.view(),
                c_a.view(),
            );
            scaled_outer_add(
                hessian_psi_psi.view_mut(),
                loss_3 * q_a,
                c_b.view(),
                b.view(),
            );
            scaled_outer_add(
                hessian_psi_psi.view_mut(),
                loss_3 * q_a,
                b.view(),
                c_b.view(),
            );
            hessian_psi_psi.scaled_add(loss_3 * q_a * q_b, &q_mat);
            scaled_outer_add(
                hessian_psi_psi.view_mut(),
                loss_4 * q_a * q_b + loss_3 * q_ab,
                b.view(),
                b.view(),
            );
        }

        Ok(crate::custom_family::ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi,
            hessian_psi_psi_operator: None,
        })
    }

    fn exact_newton_joint_psihessian_directional_derivative_from_designs(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(dir_a) = self.exact_newton_joint_psi_direction(
            block_states,
            derivative_blocks,
            psi_index,
            x_t,
            x_ls,
            &self.policy,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(
            self.exact_newton_joint_psihessian_directional_derivative_from_parts(
                block_states,
                &dir_a,
                d_beta_flat,
                x_t,
                x_ls,
            )?,
        ))
    }

    fn exact_newton_joint_psihessian_directional_derivative_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        dir_a: &LocationScaleJointPsiDirection,
        d_beta_flat: &Array1<f64>,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let base_core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(base_core.q0.view())?;
        let d0 =
            self.wiggle_basiswith_options(base_core.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(base_core.q0.view(), BasisOptions::second_derivative())?;
        let d3_basis = self.wiggle_d3basis_constrained(base_core.q0.view())?;
        let d4q = self.wiggle_d4q_dq04(base_core.q0.view(), betaw.view())?;
        let pw = b0.ncols();
        let layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
        let (u_t, u_ls, uw) = layout.split_three(
            d_beta_flat,
            "wiggle joint psi hessian directional derivative",
        )?;
        let total = pt + pls + pw;
        if d0.ncols() != betaw.len()
            || dd0.ncols() != betaw.len()
            || d3_basis.ncols() != betaw.len()
        {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle derivative/beta mismatch in joint psi mixed drift: B'={} B''={} B'''={} betaw={}",
                d0.ncols(),
                dd0.ncols(),
                d3_basis.ncols(),
                betaw.len()
            ) }.into());
        }
        let xi_t = x_t.dot(&u_t);
        let xi_ls = x_ls.dot(&u_ls);
        let x_t_map = dir_a.x_primary_psi.as_linear_map_ref();
        let x_ls_map = dir_a.x_ls_psi.as_linear_map_ref();
        let m = d0.dot(betaw) + 1.0;
        let g2 = dd0.dot(betaw);
        let g3 = self.wiggle_d3q_dq03(base_core.q0.view(), betaw.view())?;
        let g4 = d4q;
        let (sigma, ds, d2s, d3s, d4s) = exp_sigma_derivs_up_to_fourth_array(eta_ls.view());

        // Exact likelihood-side mixed drift T_a[u] = D_beta H_{psi_a}^{(D)}[u].
        //
        // The unified outer Hessian in custom_family.rs uses
        //   ddot H_ij = H_ij + T_i[beta_j] + T_j[beta_i]
        //             + D_beta H[beta_ij] + D_beta^2 H[beta_i, beta_j].
        //
        // For wiggle we still use the same scalar-loss row kernel as non-wiggle;
        // only the location-side row changes to z_r = [x_{t,r}; B_r(q0)] with
        // q = q0 + betaw^T B(q0), q0 = -eta_t * exp(-eta_ls).
        let mut out = Array2::<f64>::zeros((total, total));
        let mut b = Array1::<f64>::zeros(total);
        let mut c_a = Array1::<f64>::zeros(total);
        let mut gamma = Array1::<f64>::zeros(total);
        let mut gamma_a = Array1::<f64>::zeros(total);
        let mut q_mat = Array2::<f64>::zeros((total, total));
        let mut r_a = Array2::<f64>::zeros((total, total));
        let mut c_u = Array2::<f64>::zeros((total, total));
        let mut delta_a = Array2::<f64>::zeros((total, total));
        let mut q_tw = Array1::<f64>::zeros(pw);
        let mut q_lw = Array1::<f64>::zeros(pw);
        let mut qw_a = Array1::<f64>::zeros(pw);
        let mut q_tw_a = Array1::<f64>::zeros(pw);
        let mut q_lw_a = Array1::<f64>::zeros(pw);
        let mut dq_tw_u = Array1::<f64>::zeros(pw);
        let mut dq_lw_u = Array1::<f64>::zeros(pw);
        let mut dq_tw_a_u = Array1::<f64>::zeros(pw);
        let mut dq_lw_a_u = Array1::<f64>::zeros(pw);
        for row in 0..n {
            let q = core.q0[row] + etaw[row];
            let (loss_1, loss_2, loss_3) = binomial_neglog_q_derivatives_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            );
            let loss_4 = binomial_neglog_q_fourth_derivative_dispatch(
                self.y[row],
                self.weights[row],
                q,
                core.mu[row],
                core.dmu_dq[row],
                core.d2mu_dq2[row],
                core.d3mu_dq3[row],
                &self.link_kind,
            )?;
            let q0 = nonwiggle_q_derivs(eta_t[row], sigma[row]);
            let s_safe = sigma[row];
            let s2 = s_safe * s_safe;
            let s3 = s2 * s_safe;
            let s4 = s3 * s_safe;
            let s5 = s4 * s_safe;
            let q0_tl_ls_ls = d3s[row] / s2 - 6.0 * ds[row] * d2s[row] / s3
                + 6.0 * ds[row] * ds[row] * ds[row] / s4;
            let q0_tl_ls_ls_ls =
                d4s[row] / s2 - 8.0 * ds[row] * d3s[row] / s3 - 6.0 * d2s[row] * d2s[row] / s3
                    + 36.0 * ds[row] * ds[row] * d2s[row] / s4
                    - 24.0 * ds[row] * ds[row] * ds[row] * ds[row] / s5;
            let q0_ll_ls_ls = eta_t[row] * q0_tl_ls_ls_ls;

            let xtr = x_t.row(row);
            let xlsr = x_ls.row(row);
            let xta = x_t_map.row_vector(row)?;
            let xlsa = x_ls_map.row_vector(row)?;
            let br = b0.row(row);
            let dr = d0.row(row);
            let ddr = dd0.row(row);
            let d3r = d3_basis.row(row);

            let xi_t_i = xi_t[row];
            let xi_ls_i = xi_ls[row];
            let xi_ta_i = xta.dot(&u_t);
            let xi_lsa_i = xlsa.dot(&u_ls);
            let d_dot_u = dr.dot(&uw);
            let dd_dot_u = ddr.dot(&uw);
            let d3_dot_u = d3r.dot(&uw);

            let dq0_u = q0.q_t * xi_t_i + q0.q_ls * xi_ls_i;
            let dq0_t_u = q0.q_tl * xi_ls_i;
            let dq0_ls_u = q0.q_tl * xi_t_i + q0.q_ll * xi_ls_i;
            let dq0_tl_u = q0.q_tl_ls * xi_ls_i;
            let dq0_ll_u = q0.q_tl_ls * xi_t_i + q0.q_ll_ls * xi_ls_i;
            let dq0_tl_ls_u = q0_tl_ls_ls * xi_ls_i;
            let dq0_ll_ls_u = q0_tl_ls_ls * xi_t_i + q0_ll_ls_ls * xi_ls_i;

            let q0_a = -q0.q_t * dir_a.z_primary_psi[row] - q0.q_ls * dir_a.z_ls_psi[row];
            let q0_t_a = q0.q_tl_ls * dir_a.z_ls_psi[row];
            let q0_ls_a = q0.q_tl_ls * dir_a.z_primary_psi[row] + q0.q_ll_ls * dir_a.z_ls_psi[row];
            let q0_tl_a = q0.q_tl_ls * dir_a.z_ls_psi[row];
            let q0_ll_a = q0.q_tl_ls * dir_a.z_primary_psi[row] + q0.q_ll_ls * dir_a.z_ls_psi[row];
            let dq0_a_u = q0_t_a * xi_t_i + q0_ls_a * xi_ls_i;
            let dq0_t_a_u = dq0_tl_ls_u * dir_a.z_ls_psi[row];
            let dq0_ls_a_u =
                dq0_tl_ls_u * dir_a.z_primary_psi[row] + dq0_ll_ls_u * dir_a.z_ls_psi[row];
            let dq0_tl_a_u = dq0_tl_ls_u * dir_a.z_ls_psi[row];
            let dq0_ll_a_u =
                dq0_tl_ls_u * dir_a.z_primary_psi[row] + dq0_ll_ls_u * dir_a.z_ls_psi[row];

            let q_t = m[row] * q0.q_t;
            let q_ls = m[row] * q0.q_ls;
            let q_tt = g2[row] * q0.q_t * q0.q_t;
            let q_tl = g2[row] * q0.q_t * q0.q_ls + m[row] * q0.q_tl;
            let q_ll = g2[row] * q0.q_ls * q0.q_ls + m[row] * q0.q_ll;
            q_tw.fill(0.0);
            q_tw.scaled_add(q0.q_t, &dr);
            q_lw.fill(0.0);
            q_lw.scaled_add(q0.q_ls, &dr);

            let dm_u = g2[row] * dq0_u + d_dot_u;
            let dg2_u = g3[row] * dq0_u + dd_dot_u;
            let dg3_u = g4[row] * dq0_u + d3_dot_u;

            let q_a = m[row] * q0_a;
            let q_t_a = g2[row] * q0_a * q0.q_t + m[row] * q0_t_a;
            let q_ls_a = g2[row] * q0_a * q0.q_ls + m[row] * q0_ls_a;
            let q_tt_a = g3[row] * q0_a * q0.q_t * q0.q_t + g2[row] * (2.0 * q0.q_t * q0_t_a);
            let q_tl_a = g3[row] * q0_a * q0.q_t * q0.q_ls
                + g2[row] * (q0_t_a * q0.q_ls + q0.q_t * q0_ls_a + q0_a * q0.q_tl)
                + m[row] * q0_tl_a;
            let q_ll_a = g3[row] * q0_a * q0.q_ls * q0.q_ls
                + g2[row] * (2.0 * q0.q_ls * q0_ls_a + q0_a * q0.q_ll)
                + m[row] * q0_ll_a;
            qw_a.fill(0.0);
            qw_a.scaled_add(q0_a, &dr);
            q_tw_a.fill(0.0);
            q_tw_a.scaled_add(q0_a * q0.q_t, &ddr);
            q_tw_a.scaled_add(q0_t_a, &dr);
            q_lw_a.fill(0.0);
            q_lw_a.scaled_add(q0_a * q0.q_ls, &ddr);
            q_lw_a.scaled_add(q0_ls_a, &dr);

            let dq_tt_u = dg2_u * q0.q_t * q0.q_t + g2[row] * (2.0 * q0.q_t * dq0_t_u);
            let dq_tl_u = dg2_u * q0.q_t * q0.q_ls
                + g2[row] * (dq0_t_u * q0.q_ls + q0.q_t * dq0_ls_u)
                + dm_u * q0.q_tl
                + m[row] * dq0_tl_u;
            let dq_ll_u = dg2_u * q0.q_ls * q0.q_ls
                + g2[row] * (2.0 * q0.q_ls * dq0_ls_u)
                + dm_u * q0.q_ll
                + m[row] * dq0_ll_u;
            dq_tw_u.fill(0.0);
            dq_tw_u.scaled_add(dq0_u * q0.q_t, &ddr);
            dq_tw_u.scaled_add(dq0_t_u, &dr);
            dq_lw_u.fill(0.0);
            dq_lw_u.scaled_add(dq0_u * q0.q_ls, &ddr);
            dq_lw_u.scaled_add(dq0_ls_u, &dr);

            let dq_tt_a_u = dg3_u * q0_a * q0.q_t * q0.q_t
                + g3[row] * (dq0_a_u * q0.q_t * q0.q_t + 2.0 * q0_a * q0.q_t * dq0_t_u)
                + dg2_u * (2.0 * q0.q_t * q0_t_a)
                + g2[row] * (2.0 * dq0_t_u * q0_t_a + 2.0 * q0.q_t * dq0_t_a_u);
            let dq_tl_a_u = dg3_u * q0_a * q0.q_t * q0.q_ls
                + g3[row]
                    * (dq0_a_u * q0.q_t * q0.q_ls
                        + q0_a * dq0_t_u * q0.q_ls
                        + q0_a * q0.q_t * dq0_ls_u)
                + dg2_u * (q0_t_a * q0.q_ls + q0.q_t * q0_ls_a + q0_a * q0.q_tl)
                + g2[row]
                    * (dq0_t_a_u * q0.q_ls
                        + q0_t_a * dq0_ls_u
                        + dq0_t_u * q0_ls_a
                        + q0.q_t * dq0_ls_a_u
                        + dq0_a_u * q0.q_tl
                        + q0_a * dq0_tl_u)
                + dm_u * q0_tl_a
                + m[row] * dq0_tl_a_u;
            let dq_ll_a_u = dg3_u * q0_a * q0.q_ls * q0.q_ls
                + g3[row] * (dq0_a_u * q0.q_ls * q0.q_ls + 2.0 * q0_a * q0.q_ls * dq0_ls_u)
                + dg2_u * (2.0 * q0.q_ls * q0_ls_a + q0_a * q0.q_ll)
                + g2[row]
                    * (2.0 * dq0_ls_u * q0_ls_a
                        + 2.0 * q0.q_ls * dq0_ls_a_u
                        + dq0_a_u * q0.q_ll
                        + q0_a * dq0_ll_u)
                + dm_u * q0_ll_a
                + m[row] * dq0_ll_a_u;
            dq_tw_a_u.fill(0.0);
            dq_tw_a_u.scaled_add(dq0_u * q0_a * q0.q_t, &d3r);
            dq_tw_a_u.scaled_add(dq0_a_u * q0.q_t + q0_a * dq0_t_u + dq0_u * q0_t_a, &ddr);
            dq_tw_a_u.scaled_add(dq0_t_a_u, &dr);
            dq_lw_a_u.fill(0.0);
            dq_lw_a_u.scaled_add(dq0_u * q0_a * q0.q_ls, &d3r);
            dq_lw_a_u.scaled_add(dq0_a_u * q0.q_ls + q0_a * dq0_ls_u + dq0_u * q0_ls_a, &ddr);
            dq_lw_a_u.scaled_add(dq0_ls_a_u, &dr);

            b.fill(0.0);
            b.slice_mut(s![0..pt]).scaled_add(q_t, &xtr);
            b.slice_mut(s![pt..pt + pls]).scaled_add(q_ls, &xlsr);
            b.slice_mut(s![pt + pls..]).assign(&br);

            c_a.fill(0.0);
            c_a.slice_mut(s![0..pt]).scaled_add(q_t_a, &xtr);
            c_a.slice_mut(s![0..pt]).scaled_add(q_t, &xta.view());
            c_a.slice_mut(s![pt..pt + pls]).scaled_add(q_ls_a, &xlsr);
            c_a.slice_mut(s![pt..pt + pls])
                .scaled_add(q_ls, &xlsa.view());
            c_a.slice_mut(s![pt + pls..]).assign(&qw_a);

            gamma.fill(0.0);
            gamma
                .slice_mut(s![0..pt])
                .scaled_add(q_tt * xi_t_i + q_tl * xi_ls_i + q0.q_t * d_dot_u, &xtr);
            gamma
                .slice_mut(s![pt..pt + pls])
                .scaled_add(q_tl * xi_t_i + q_ll * xi_ls_i + q0.q_ls * d_dot_u, &xlsr);
            gamma.slice_mut(s![pt + pls..]).scaled_add(dq0_u, &dr);

            let q_tw_a_dot_u = q_tw_a.dot(&uw);
            let q_lw_a_dot_u = q_lw_a.dot(&uw);
            gamma_a.fill(0.0);
            gamma_a.slice_mut(s![0..pt]).scaled_add(
                q_tt_a * xi_t_i
                    + q_tt * xi_ta_i
                    + q_tl_a * xi_ls_i
                    + q_tl * xi_lsa_i
                    + q_tw_a_dot_u,
                &xtr,
            );
            gamma_a.slice_mut(s![0..pt]).scaled_add(
                q_tt * xi_t_i + q_tl * xi_ls_i + q0.q_t * d_dot_u,
                &xta.view(),
            );
            gamma_a.slice_mut(s![pt..pt + pls]).scaled_add(
                q_tl_a * xi_t_i
                    + q_tl * xi_ta_i
                    + q_ll_a * xi_ls_i
                    + q_ll * xi_lsa_i
                    + q_lw_a_dot_u,
                &xlsr,
            );
            gamma_a.slice_mut(s![pt..pt + pls]).scaled_add(
                q_tl * xi_t_i + q_ll * xi_ls_i + q0.q_ls * d_dot_u,
                &xlsa.view(),
            );
            gamma_a
                .slice_mut(s![pt + pls..])
                .scaled_add(xi_t_i, &q_tw_a);
            gamma_a.slice_mut(s![pt + pls..]).scaled_add(xi_ta_i, &q_tw);
            gamma_a
                .slice_mut(s![pt + pls..])
                .scaled_add(xi_ls_i, &q_lw_a);
            gamma_a
                .slice_mut(s![pt + pls..])
                .scaled_add(xi_lsa_i, &q_lw);

            let alpha = b.dot(d_beta_flat);
            let alpha_a = c_a.dot(d_beta_flat);

            q_mat.fill(0.0);
            scaled_outer_add(q_mat.slice_mut(s![0..pt, 0..pt]), q_tt, xtr, xtr);
            scaled_outer_add(q_mat.slice_mut(s![0..pt, pt..pt + pls]), q_tl, xtr, xlsr);
            scaled_outer_add(
                q_mat.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsr,
                xlsr,
            );
            scaled_outer_add(
                q_mat.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xtr,
                q_tw.view(),
            );
            scaled_outer_add(
                q_mat.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsr,
                q_lw.view(),
            );
            mirror_upper_to_lower(&mut q_mat);

            r_a.fill(0.0);
            scaled_outer_add(r_a.slice_mut(s![0..pt, 0..pt]), q_tt_a, xtr, xtr);
            scaled_outer_add(r_a.slice_mut(s![0..pt, 0..pt]), q_tt, xta.view(), xtr);
            scaled_outer_add(r_a.slice_mut(s![0..pt, 0..pt]), q_tt, xtr, xta.view());
            scaled_outer_add(r_a.slice_mut(s![0..pt, pt..pt + pls]), q_tl_a, xtr, xlsr);
            scaled_outer_add(
                r_a.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xta.view(),
                xlsr,
            );
            scaled_outer_add(
                r_a.slice_mut(s![0..pt, pt..pt + pls]),
                q_tl,
                xtr,
                xlsa.view(),
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll_a,
                xlsr,
                xlsr,
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsa.view(),
                xlsr,
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                q_ll,
                xlsr,
                xlsa.view(),
            );
            scaled_outer_add(
                r_a.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xta.view(),
                q_tw.view(),
            );
            scaled_outer_add(
                r_a.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xtr,
                q_tw_a.view(),
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsa.view(),
                q_lw.view(),
            );
            scaled_outer_add(
                r_a.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsr,
                q_lw_a.view(),
            );
            mirror_upper_to_lower(&mut r_a);

            c_u.fill(0.0);
            scaled_outer_add(c_u.slice_mut(s![0..pt, 0..pt]), dq_tt_u, xtr, xtr);
            scaled_outer_add(c_u.slice_mut(s![0..pt, pt..pt + pls]), dq_tl_u, xtr, xlsr);
            scaled_outer_add(
                c_u.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                dq_ll_u,
                xlsr,
                xlsr,
            );
            scaled_outer_add(
                c_u.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xtr,
                dq_tw_u.view(),
            );
            scaled_outer_add(
                c_u.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsr,
                dq_lw_u.view(),
            );
            mirror_upper_to_lower(&mut c_u);

            delta_a.fill(0.0);
            scaled_outer_add(delta_a.slice_mut(s![0..pt, 0..pt]), dq_tt_a_u, xtr, xtr);
            scaled_outer_add(
                delta_a.slice_mut(s![0..pt, 0..pt]),
                dq_tt_u,
                xta.view(),
                xtr,
            );
            scaled_outer_add(
                delta_a.slice_mut(s![0..pt, 0..pt]),
                dq_tt_u,
                xtr,
                xta.view(),
            );
            scaled_outer_add(
                delta_a.slice_mut(s![0..pt, pt..pt + pls]),
                dq_tl_a_u,
                xtr,
                xlsr,
            );
            scaled_outer_add(
                delta_a.slice_mut(s![0..pt, pt..pt + pls]),
                dq_tl_u,
                xta.view(),
                xlsr,
            );
            scaled_outer_add(
                delta_a.slice_mut(s![0..pt, pt..pt + pls]),
                dq_tl_u,
                xtr,
                xlsa.view(),
            );
            scaled_outer_add(
                delta_a.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                dq_ll_a_u,
                xlsr,
                xlsr,
            );
            scaled_outer_add(
                delta_a.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                dq_ll_u,
                xlsa.view(),
                xlsr,
            );
            scaled_outer_add(
                delta_a.slice_mut(s![pt..pt + pls, pt..pt + pls]),
                dq_ll_u,
                xlsr,
                xlsa.view(),
            );
            scaled_outer_add(
                delta_a.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xta.view(),
                dq_tw_u.view(),
            );
            scaled_outer_add(
                delta_a.slice_mut(s![0..pt, pt + pls..]),
                1.0,
                xtr,
                dq_tw_a_u.view(),
            );
            scaled_outer_add(
                delta_a.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsa.view(),
                dq_lw_u.view(),
            );
            scaled_outer_add(
                delta_a.slice_mut(s![pt..pt + pls, pt + pls..]),
                1.0,
                xlsr,
                dq_lw_a_u.view(),
            );
            mirror_upper_to_lower(&mut delta_a);

            out.scaled_add(loss_1, &delta_a);
            out.scaled_add(loss_2 * alpha, &r_a);
            out.scaled_add(loss_2 * q_a, &c_u);
            scaled_outer_add(out.view_mut(), loss_2, gamma_a.view(), b.view());
            scaled_outer_add(out.view_mut(), loss_2, b.view(), gamma_a.view());
            scaled_outer_add(out.view_mut(), loss_2, gamma.view(), c_a.view());
            scaled_outer_add(out.view_mut(), loss_2, c_a.view(), gamma.view());
            out.scaled_add(loss_2 * alpha_a, &q_mat);
            scaled_outer_add(out.view_mut(), loss_3 * alpha * q_a, b.view(), b.view());
            scaled_outer_add(out.view_mut(), loss_3 * q_a, gamma.view(), b.view());
            scaled_outer_add(out.view_mut(), loss_3 * q_a, b.view(), gamma.view());
            scaled_outer_add(out.view_mut(), loss_3 * alpha, c_a.view(), b.view());
            scaled_outer_add(out.view_mut(), loss_3 * alpha, b.view(), c_a.view());
            out.scaled_add(loss_3 * alpha * q_a, &q_mat);
            scaled_outer_add(
                out.view_mut(),
                loss_4 * alpha * q_a + loss_3 * alpha_a,
                b.view(),
                b.view(),
            );
        }
        mirror_upper_to_lower(&mut out);
        Ok(out)
    }

    /// Build a turnkey wiggle block from a q-seed vector and knot settings.
    /// Returns both the block input and the generated knot vector.
    pub fn buildwiggle_block_input(
        q_seed: ArrayView1<'_, f64>,
        degree: usize,
        num_internal_knots: usize,
        penalty_order: usize,
        double_penalty: bool,
    ) -> Result<(ParameterBlockInput, Array1<f64>), String> {
        let knots = Self::initializewiggle_knots_from_q(q_seed, degree, num_internal_knots)?;
        let block = buildwiggle_block_input_from_knots(
            q_seed,
            &knots,
            degree,
            penalty_order,
            double_penalty,
        )?;
        Ok((block, knots))
    }

    /// Compute the rowwise pieces (diagonal weights + B/B' basis arrays) used
    /// to assemble the joint Hessian for the 3-block wiggle family. Both the
    /// dense Hessian path and the matrix-free workspace consume these pieces
    /// without recomputing the per-row scalar derivatives.
    fn wiggle_hessian_row_pieces(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<BinomialLocationScaleWiggleHessianRowPieces, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let betaw0 = block_states[Self::BLOCK_WIGGLE].beta.clone();
        let core0 = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core0.q0.view())?;
        let d0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::second_derivative())?;
        if b0.ncols() != betaw0.len() || d0.ncols() != betaw0.len() || dd0.ncols() != betaw0.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "wiggle basis/beta mismatch in exact joint Hessian: B={} B'={} B''={} betaw={}",
                    b0.ncols(),
                    d0.ncols(),
                    dd0.ncols(),
                    betaw0.len()
                ),
            }
            .into());
        }
        let m = d0.dot(&betaw0) + 1.0;
        let g2 = dd0.dot(&betaw0);
        let (sigma, ..) = exp_sigma_derivs_up_to_third(eta_ls.view());
        let mut coeff_tt = Array1::<f64>::zeros(n);
        let mut coeff_tl = Array1::<f64>::zeros(n);
        let mut coeff_ll = Array1::<f64>::zeros(n);
        let mut coeff_tw_b = Array1::<f64>::zeros(n);
        let mut coeff_tw_d = Array1::<f64>::zeros(n);
        let mut coeff_lw_b = Array1::<f64>::zeros(n);
        let mut coeff_lw_d = Array1::<f64>::zeros(n);
        let mut coeffww = Array1::<f64>::zeros(n);
        for i in 0..n {
            let q_i = core0.q0[i] + etaw[i];
            let (m1, m2, _) = binomial_neglog_q_derivatives_dispatch(
                self.y[i],
                self.weights[i],
                q_i,
                core0.mu[i],
                core0.dmu_dq[i],
                core0.d2mu_dq2[i],
                core0.d3mu_dq3[i],
                &self.link_kind,
            );
            let q0 = nonwiggle_q_derivs(eta_t[i], sigma[i]);

            let q_t = m[i] * q0.q_t;
            let q_ls = m[i] * q0.q_ls;
            let q_tt = g2[i] * q0.q_t * q0.q_t;
            let q_tl = g2[i] * q0.q_t * q0.q_ls + m[i] * q0.q_tl;
            let q_ll = g2[i] * q0.q_ls * q0.q_ls + m[i] * q0.q_ll;

            coeff_tt[i] = hessian_coeff_fromobjective_q_terms(m1, m2, q_t, q_t, q_tt);
            coeff_tl[i] = hessian_coeff_fromobjective_q_terms(m1, m2, q_t, q_ls, q_tl);
            coeff_ll[i] = hessian_coeff_fromobjective_q_terms(m1, m2, q_ls, q_ls, q_ll);
            coeff_tw_b[i] = m2 * q_t;
            coeff_tw_d[i] = m1 * q0.q_t;
            coeff_lw_b[i] = m2 * q_ls;
            coeff_lw_d[i] = m1 * q0.q_ls;
            coeffww[i] = m2;
        }
        Ok(BinomialLocationScaleWiggleHessianRowPieces {
            coeff_tt,
            coeff_tl,
            coeff_ll,
            coeff_tw_b,
            coeff_tw_d,
            coeff_lw_b,
            coeff_lw_d,
            coeffww,
            b0,
            d0,
        })
    }

    fn expected_wiggle_geometry_inputs<'a>(
        &'a self,
        block_states: &'a [ParameterBlockState],
        specs: Option<&'a [ParameterBlockSpec]>,
    ) -> Result<Option<ExpectedWiggleGeometryInputs<'a>>, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(specs)? else {
            return Ok(None);
        };
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason:
                    "BinomialLocationScaleWiggleFamily expected-information input size mismatch"
                        .to_string(),
            }
            .into());
        }
        Ok(Some(ExpectedWiggleGeometryInputs {
            x_t,
            x_ls,
            eta_t,
            eta_ls,
            etaw,
        }))
    }

    fn expected_wiggle_information_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(inputs) = self.expected_wiggle_geometry_inputs(block_states, Some(specs))? else {
            return Ok(None);
        };
        let ExpectedWiggleGeometryInputs {
            x_t,
            x_ls,
            eta_t,
            eta_ls,
            etaw,
        } = inputs;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core.q0.view())?;
        let d0 = self.wiggle_basiswith_options(core.q0.view(), BasisOptions::first_derivative())?;
        let m = d0.dot(betaw) + 1.0;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let pw = b0.ncols();
        let total = pt + pls + pw;
        let mut out = Array2::<f64>::zeros((total, total));
        let mut b = Array1::<f64>::zeros(total);
        for i in 0..self.y.len() {
            let q = core.q0[i] + etaw[i];
            let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q).map_err(|e| {
                format!("BinomialLocationScaleWiggle expected information link jet failed: {e}")
            })?;
            let (f, _, _) = binomial_expected_q_information_derivatives(
                self.weights[i],
                jet.mu,
                jet.d1,
                jet.d2,
                jet.d3,
            );
            if f == 0.0 {
                continue;
            }
            let q0 = nonwiggle_q_derivs(eta_t[i], core.sigma[i]);
            b.fill(0.0);
            b.slice_mut(s![0..pt])
                .scaled_add(m[i] * q0.q_t, &x_t.row(i));
            b.slice_mut(s![pt..pt + pls])
                .scaled_add(m[i] * q0.q_ls, &x_ls.row(i));
            b.slice_mut(s![pt + pls..]).assign(&b0.row(i));
            scaled_outer_add(out.view_mut(), f, b.view(), b.view());
        }
        mirror_upper_to_lower(&mut out);
        Ok(Some(out))
    }

    fn expected_wiggle_information_directional_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(inputs) = self.expected_wiggle_geometry_inputs(block_states, Some(specs))? else {
            return Ok(None);
        };
        let ExpectedWiggleGeometryInputs {
            x_t,
            x_ls,
            eta_t,
            eta_ls,
            etaw,
        } = inputs;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core.q0.view())?;
        let d0 = self.wiggle_basiswith_options(core.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(core.q0.view(), BasisOptions::second_derivative())?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let pw = b0.ncols();
        let layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
        let (u_t, u_ls, uw) = layout.split_three(d_beta_flat, "expected wiggle d_beta")?;
        let d_eta_t = fast_av(&x_t, &u_t);
        let d_eta_ls = fast_av(&x_ls, &u_ls);
        let m = d0.dot(betaw) + 1.0;
        let g2 = dd0.dot(betaw);
        let total = layout.total();
        let mut out = Array2::<f64>::zeros((total, total));
        let mut b = Array1::<f64>::zeros(total);
        let mut bu = Array1::<f64>::zeros(total);
        for i in 0..self.y.len() {
            let q = core.q0[i] + etaw[i];
            let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q).map_err(|e| {
                format!("BinomialLocationScaleWiggle expected dI link jet failed: {e}")
            })?;
            let (f, f1, _) = binomial_expected_q_information_derivatives(
                self.weights[i],
                jet.mu,
                jet.d1,
                jet.d2,
                jet.d3,
            );
            if f == 0.0 && f1 == 0.0 {
                continue;
            }
            let q0 = nonwiggle_q_derivs(eta_t[i], core.sigma[i]);
            let dq0_u = q0.q_t * d_eta_t[i] + q0.q_ls * d_eta_ls[i];
            let dq0_t_u = q0.q_tl * d_eta_ls[i];
            let dq0_ls_u = q0.q_tl * d_eta_t[i] + q0.q_ll * d_eta_ls[i];
            let bu_w = b0.row(i).dot(&uw);
            let b1u = d0.row(i).dot(&uw);
            let dm_u = g2[i] * dq0_u + b1u;
            let alpha_u = m[i] * dq0_u + bu_w;
            let q_t = m[i] * q0.q_t;
            let q_ls = m[i] * q0.q_ls;
            let dq_t_u = dm_u * q0.q_t + m[i] * dq0_t_u;
            let dq_ls_u = dm_u * q0.q_ls + m[i] * dq0_ls_u;
            b.fill(0.0);
            bu.fill(0.0);
            b.slice_mut(s![0..pt]).scaled_add(q_t, &x_t.row(i));
            b.slice_mut(s![pt..pt + pls]).scaled_add(q_ls, &x_ls.row(i));
            b.slice_mut(s![pt + pls..]).assign(&b0.row(i));
            bu.slice_mut(s![0..pt]).scaled_add(dq_t_u, &x_t.row(i));
            bu.slice_mut(s![pt..pt + pls])
                .scaled_add(dq_ls_u, &x_ls.row(i));
            bu.slice_mut(s![pt + pls..]).scaled_add(dq0_u, &d0.row(i));
            scaled_outer_add(out.view_mut(), f1 * alpha_u, b.view(), b.view());
            scaled_outer_add(out.view_mut(), f, bu.view(), b.view());
            scaled_outer_add(out.view_mut(), f, b.view(), bu.view());
        }
        mirror_upper_to_lower(&mut out);
        Ok(Some(out))
    }

    fn expected_wiggle_information_second_directional_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(inputs) = self.expected_wiggle_geometry_inputs(block_states, Some(specs))? else {
            return Ok(None);
        };
        let ExpectedWiggleGeometryInputs {
            x_t,
            x_ls,
            eta_t,
            eta_ls,
            etaw,
        } = inputs;
        let betaw = &block_states[Self::BLOCK_WIGGLE].beta;
        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core.q0.view())?;
        let d0 = self.wiggle_basiswith_options(core.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(core.q0.view(), BasisOptions::second_derivative())?;
        let d3q = self.wiggle_d3q_dq03(core.q0.view(), betaw.view())?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let pw = b0.ncols();
        let layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
        let (u_t, u_ls, uw) = layout.split_three(d_beta_u_flat, "expected wiggle d_beta_u")?;
        let (v_t, v_ls, vw) = layout.split_three(d_betav_flat, "expected wiggle d_beta_v")?;
        let d_eta_t_u = fast_av(&x_t, &u_t);
        let d_eta_ls_u = fast_av(&x_ls, &u_ls);
        let d_eta_t_v = fast_av(&x_t, &v_t);
        let d_eta_ls_v = fast_av(&x_ls, &v_ls);
        let m = d0.dot(betaw) + 1.0;
        let g2 = dd0.dot(betaw);
        let total = layout.total();
        let mut out = Array2::<f64>::zeros((total, total));
        let mut b = Array1::<f64>::zeros(total);
        let mut bu = Array1::<f64>::zeros(total);
        let mut bv = Array1::<f64>::zeros(total);
        let mut buv = Array1::<f64>::zeros(total);
        for i in 0..self.y.len() {
            let q = core.q0[i] + etaw[i];
            let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q).map_err(|e| {
                format!("BinomialLocationScaleWiggle expected d2I link jet failed: {e}")
            })?;
            let (f, f1, f2) = binomial_expected_q_information_derivatives(
                self.weights[i],
                jet.mu,
                jet.d1,
                jet.d2,
                jet.d3,
            );
            if f == 0.0 && f1 == 0.0 && f2 == 0.0 {
                continue;
            }
            let q0 = nonwiggle_q_derivs(eta_t[i], core.sigma[i]);
            let dq0_u = q0.q_t * d_eta_t_u[i] + q0.q_ls * d_eta_ls_u[i];
            let dq0_v = q0.q_t * d_eta_t_v[i] + q0.q_ls * d_eta_ls_v[i];
            let d2q0_uv = q0.q_tl * (d_eta_t_u[i] * d_eta_ls_v[i] + d_eta_t_v[i] * d_eta_ls_u[i])
                + q0.q_ll * d_eta_ls_u[i] * d_eta_ls_v[i];
            let dq0_t_u = q0.q_tl * d_eta_ls_u[i];
            let dq0_t_v = q0.q_tl * d_eta_ls_v[i];
            let dq0_ls_u = q0.q_tl * d_eta_t_u[i] + q0.q_ll * d_eta_ls_u[i];
            let dq0_ls_v = q0.q_tl * d_eta_t_v[i] + q0.q_ll * d_eta_ls_v[i];
            let d2q0_t_uv = q0.q_tl_ls * d_eta_ls_u[i] * d_eta_ls_v[i];
            let d2q0_ls_uv = q0.q_tl_ls
                * (d_eta_ls_u[i] * d_eta_t_v[i] + d_eta_ls_v[i] * d_eta_t_u[i])
                + q0.q_ll_ls * d_eta_ls_u[i] * d_eta_ls_v[i];

            let br = b0.row(i);
            let dr = d0.row(i);
            let ddr = dd0.row(i);
            let b_u = br.dot(&uw);
            let b_v = br.dot(&vw);
            let b1_u = dr.dot(&uw);
            let b1_v = dr.dot(&vw);
            let b2_u = ddr.dot(&uw);
            let b2_v = ddr.dot(&vw);
            let dm_u = g2[i] * dq0_u + b1_u;
            let dm_v = g2[i] * dq0_v + b1_v;
            let d2m_uv = d3q[i] * dq0_u * dq0_v + g2[i] * d2q0_uv + b2_v * dq0_u + b2_u * dq0_v;
            let alpha_u = m[i] * dq0_u + b_u;
            let alpha_v = m[i] * dq0_v + b_v;
            let alpha_uv = m[i] * d2q0_uv + g2[i] * dq0_u * dq0_v + b1_u * dq0_v + b1_v * dq0_u;

            let q_t = m[i] * q0.q_t;
            let q_ls = m[i] * q0.q_ls;
            let dq_t_u = dm_u * q0.q_t + m[i] * dq0_t_u;
            let dq_t_v = dm_v * q0.q_t + m[i] * dq0_t_v;
            let dq_ls_u = dm_u * q0.q_ls + m[i] * dq0_ls_u;
            let dq_ls_v = dm_v * q0.q_ls + m[i] * dq0_ls_v;
            let d2q_t_uv = d2m_uv * q0.q_t + dm_u * dq0_t_v + dm_v * dq0_t_u + m[i] * d2q0_t_uv;
            let d2q_ls_uv =
                d2m_uv * q0.q_ls + dm_u * dq0_ls_v + dm_v * dq0_ls_u + m[i] * d2q0_ls_uv;

            b.fill(0.0);
            bu.fill(0.0);
            bv.fill(0.0);
            buv.fill(0.0);
            b.slice_mut(s![0..pt]).scaled_add(q_t, &x_t.row(i));
            b.slice_mut(s![pt..pt + pls]).scaled_add(q_ls, &x_ls.row(i));
            b.slice_mut(s![pt + pls..]).assign(&br);
            bu.slice_mut(s![0..pt]).scaled_add(dq_t_u, &x_t.row(i));
            bu.slice_mut(s![pt..pt + pls])
                .scaled_add(dq_ls_u, &x_ls.row(i));
            bu.slice_mut(s![pt + pls..]).scaled_add(dq0_u, &dr);
            bv.slice_mut(s![0..pt]).scaled_add(dq_t_v, &x_t.row(i));
            bv.slice_mut(s![pt..pt + pls])
                .scaled_add(dq_ls_v, &x_ls.row(i));
            bv.slice_mut(s![pt + pls..]).scaled_add(dq0_v, &dr);
            buv.slice_mut(s![0..pt]).scaled_add(d2q_t_uv, &x_t.row(i));
            buv.slice_mut(s![pt..pt + pls])
                .scaled_add(d2q_ls_uv, &x_ls.row(i));
            buv.slice_mut(s![pt + pls..])
                .scaled_add(dq0_u * dq0_v, &ddr);
            buv.slice_mut(s![pt + pls..]).scaled_add(d2q0_uv, &dr);

            scaled_outer_add(
                out.view_mut(),
                f2 * alpha_u * alpha_v + f1 * alpha_uv,
                b.view(),
                b.view(),
            );
            scaled_outer_add(out.view_mut(), f1 * alpha_u, bv.view(), b.view());
            scaled_outer_add(out.view_mut(), f1 * alpha_u, b.view(), bv.view());
            scaled_outer_add(out.view_mut(), f1 * alpha_v, bu.view(), b.view());
            scaled_outer_add(out.view_mut(), f1 * alpha_v, b.view(), bu.view());
            scaled_outer_add(out.view_mut(), f, buv.view(), b.view());
            scaled_outer_add(out.view_mut(), f, b.view(), buv.view());
            scaled_outer_add(out.view_mut(), f, bu.view(), bv.view());
            scaled_outer_add(out.view_mut(), f, bv.view(), bu.view());
        }
        mirror_upper_to_lower(&mut out);
        Ok(Some(out))
    }
}


/// Per-row pieces of the 3-block wiggle joint Hessian.
///
/// `coeff_*` are diagonal weights (length n). `b0` and `d0` are the realized
/// wiggle basis values and first-derivative values at the current q0
/// (n × p_w). The dense Hessian path assembles these into a (p_t+p_ls+p_w)²
/// matrix; the matrix-free workspace applies the operator
///
///   r_t = D_tt u_t + D_tl u_ls + D_tw_b (B v_w) + D_tw_d (B' v_w),
///   r_ls = D_tl u_t + D_ll u_ls + D_lw_b (B v_w) + D_lw_d (B' v_w),
///   r_b = D_tw_b u_t + D_lw_b u_ls + D_ww (B v_w),
///   r_d = D_tw_d u_t + D_lw_d u_ls,
///
/// and combines `out_w = B^T r_b + (B')^T r_d` to form `H v` directly.
struct BinomialLocationScaleWiggleHessianRowPieces {
    coeff_tt: Array1<f64>,
    coeff_tl: Array1<f64>,
    coeff_ll: Array1<f64>,
    coeff_tw_b: Array1<f64>,
    coeff_tw_d: Array1<f64>,
    coeff_lw_b: Array1<f64>,
    coeff_lw_d: Array1<f64>,
    coeffww: Array1<f64>,
    b0: Array2<f64>,
    d0: Array2<f64>,
}


struct ExpectedWiggleGeometryInputs<'a> {
    x_t: Cow<'a, Array2<f64>>,
    x_ls: Cow<'a, Array2<f64>>,
    eta_t: &'a Array1<f64>,
    eta_ls: &'a Array1<f64>,
    etaw: &'a Array1<f64>,
}


impl BinomialLocationScaleWiggleHessianRowPieces {
    fn assemble_dense(&self, x_t: &Array2<f64>, x_ls: &Array2<f64>) -> Result<Array2<f64>, String> {
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let pw = self.b0.ncols();
        let total = pt + pls + pw;
        let h_tt = xt_diag_x_dense(x_t, &self.coeff_tt)?;
        let h_tl = xt_diag_y_dense(x_t, &self.coeff_tl, x_ls)?;
        let h_ll = xt_diag_x_dense(x_ls, &self.coeff_ll)?;
        let h_tw = xt_diag_y_dense(x_t, &self.coeff_tw_b, &self.b0)?
            + &xt_diag_y_dense(x_t, &self.coeff_tw_d, &self.d0)?;
        let h_lw = xt_diag_y_dense(x_ls, &self.coeff_lw_b, &self.b0)?
            + &xt_diag_y_dense(x_ls, &self.coeff_lw_d, &self.d0)?;
        let hww = xt_diag_x_dense(&self.b0, &self.coeffww)?;

        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..pt, 0..pt]).assign(&h_tt);
        h.slice_mut(s![0..pt, pt..pt + pls]).assign(&h_tl);
        h.slice_mut(s![pt..pt + pls, pt..pt + pls]).assign(&h_ll);
        h.slice_mut(s![0..pt, pt + pls..total]).assign(&h_tw);
        h.slice_mut(s![pt..pt + pls, pt + pls..total]).assign(&h_lw);
        h.slice_mut(s![pt + pls..total, pt + pls..total])
            .assign(&hww);
        mirror_upper_to_lower(&mut h);
        Ok(h)
    }

    /// Block-diagonal Hessians (h_tt, h_ll, h_ww) without ever materializing
    /// the cross blocks. Used by `evaluate()` to populate per-block working
    /// sets.
    fn assemble_block_diagonals(
        &self,
        x_t: &Array2<f64>,
        x_ls: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), String> {
        let h_tt = xt_diag_x_dense(x_t, &self.coeff_tt)?;
        let h_ll = xt_diag_x_dense(x_ls, &self.coeff_ll)?;
        let h_ww = xt_diag_x_dense(&self.b0, &self.coeffww)?;
        Ok((h_tt, h_ll, h_ww))
    }
}


/// Per-row coefficient arrays for the BLS Wiggle joint first-directional
/// Hessian derivative `D_β H_L[u]`, shared by the dense `_directional_derivative`
/// assembly and the matrix-free `bls_wiggle_directional_operator`.
struct BinomialWiggleDhRowCoeffs {
    coeff_tt: Array1<f64>,
    coeff_tl: Array1<f64>,
    coeff_ll: Array1<f64>,
    coeff_tw_b: Array1<f64>,
    coeff_tw_d: Array1<f64>,
    coeff_tw_dd: Array1<f64>,
    coeff_lw_b: Array1<f64>,
    coeff_lw_d: Array1<f64>,
    coeff_lw_dd: Array1<f64>,
    coeffww_bb: Array1<f64>,
    coeffww_db: Array1<f64>,
}


/// All references needed to evaluate [`BinomialWiggleDhRowCoeffs`].
struct BinomialWiggleDhRowInputs<'a> {
    core0: &'a BinomialLocationScaleCore,
    eta_t: &'a Array1<f64>,
    etaw: &'a Array1<f64>,
    sigma: &'a Array1<f64>,
    m: &'a Array1<f64>,
    g2: &'a Array1<f64>,
    g3: &'a Array1<f64>,
    b0: &'a Array2<f64>,
    d0: &'a Array2<f64>,
    dd0: &'a Array2<f64>,
    uw: &'a Array1<f64>,
    d_eta_t: &'a Array1<f64>,
    d_eta_ls: &'a Array1<f64>,
}


impl BinomialLocationScaleWiggleFamily {
    /// Per-row coefficient loop for the joint first-directional Hessian
    /// derivative. The dense and operator paths build the identical 11
    /// coefficient arrays from the same canonical directional-q formulas.
    fn binomial_wiggle_dh_row_coeffs(
        &self,
        n: usize,
        inputs: &BinomialWiggleDhRowInputs<'_>,
    ) -> BinomialWiggleDhRowCoeffs {
        let BinomialWiggleDhRowInputs {
            core0,
            eta_t,
            etaw,
            sigma,
            m,
            g2,
            g3,
            b0,
            d0,
            dd0,
            uw,
            d_eta_t,
            d_eta_ls,
        } = *inputs;

        let mut coeff_tt = Array1::<f64>::zeros(n);
        let mut coeff_tl = Array1::<f64>::zeros(n);
        let mut coeff_ll = Array1::<f64>::zeros(n);
        let mut coeff_tw_b = Array1::<f64>::zeros(n);
        let mut coeff_tw_d = Array1::<f64>::zeros(n);
        let mut coeff_tw_dd = Array1::<f64>::zeros(n);
        let mut coeff_lw_b = Array1::<f64>::zeros(n);
        let mut coeff_lw_d = Array1::<f64>::zeros(n);
        let mut coeff_lw_dd = Array1::<f64>::zeros(n);
        let mut coeffww_bb = Array1::<f64>::zeros(n);
        let mut coeffww_db = Array1::<f64>::zeros(n);
        for i in 0..n {
            let q_i = core0.q0[i] + etaw[i];
            let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
                self.y[i],
                self.weights[i],
                q_i,
                core0.mu[i],
                core0.dmu_dq[i],
                core0.d2mu_dq2[i],
                core0.d3mu_dq3[i],
                &self.link_kind,
            );
            let q0 = nonwiggle_q_derivs(eta_t[i], sigma[i]);
            let dq0 = nonwiggle_q_directional(q0, d_eta_t[i], d_eta_ls[i]);

            let br = b0.row(i);
            let dr = d0.row(i);
            let ddr = dd0.row(i);
            let duw_i = dr.dot(uw);
            let dduw_i = ddr.dot(uw);

            let delta_m = g2[i] * dq0.delta_q + duw_i;
            let delta_g2 = g3[i] * dq0.delta_q + dduw_i;

            let q_t = m[i] * q0.q_t;
            let q_ls = m[i] * q0.q_ls;
            let q_tt = g2[i] * q0.q_t * q0.q_t;
            let q_tl = g2[i] * q0.q_t * q0.q_ls + m[i] * q0.q_tl;
            let q_ll = g2[i] * q0.q_ls * q0.q_ls + m[i] * q0.q_ll;

            let delta_q_t = delta_m * q0.q_t + m[i] * dq0.delta_q_t;
            let delta_q_ls = delta_m * q0.q_ls + m[i] * dq0.delta_q_ls;
            let delta_q_tt = delta_g2 * q0.q_t * q0.q_t + g2[i] * 2.0 * q0.q_t * dq0.delta_q_t;
            let delta_q_tl = delta_g2 * q0.q_t * q0.q_ls
                + g2[i] * (dq0.delta_q_t * q0.q_ls + q0.q_t * dq0.delta_q_ls)
                + delta_m * q0.q_tl
                + m[i] * dq0.delta_q_tl;
            let delta_q_ll = delta_g2 * q0.q_ls * q0.q_ls
                + g2[i] * 2.0 * q0.q_ls * dq0.delta_q_ls
                + delta_m * q0.q_ll
                + m[i] * dq0.delta_q_ll;

            let delta_q = m[i] * dq0.delta_q + br.dot(uw);

            coeff_tt[i] = directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, delta_q, q_t, q_t, q_tt, delta_q_t, delta_q_t, delta_q_tt,
            );
            coeff_tl[i] = directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, delta_q, q_t, q_ls, q_tl, delta_q_t, delta_q_ls, delta_q_tl,
            );
            coeff_ll[i] = directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, delta_q, q_ls, q_ls, q_ll, delta_q_ls, delta_q_ls, delta_q_ll,
            );
            coeff_tw_b[i] = m3 * delta_q * q_t + m2 * delta_q_t;
            coeff_tw_d[i] = m2 * (q_t * dq0.delta_q + delta_q * q0.q_t) + m1 * dq0.delta_q_t;
            coeff_tw_dd[i] = m1 * dq0.delta_q * q0.q_t;
            coeff_lw_b[i] = m3 * delta_q * q_ls + m2 * delta_q_ls;
            coeff_lw_d[i] = m2 * (q_ls * dq0.delta_q + delta_q * q0.q_ls) + m1 * dq0.delta_q_ls;
            coeff_lw_dd[i] = m1 * dq0.delta_q * q0.q_ls;
            coeffww_bb[i] = m3 * delta_q;
            coeffww_db[i] = m2 * dq0.delta_q;
        }

        BinomialWiggleDhRowCoeffs {
            coeff_tt,
            coeff_tl,
            coeff_ll,
            coeff_tw_b,
            coeff_tw_d,
            coeff_tw_dd,
            coeff_lw_b,
            coeff_lw_d,
            coeff_lw_dd,
            coeffww_bb,
            coeffww_db,
        }
    }

    /// Build the [`BlockEffectiveJacobian`] for block `block_idx`.
    ///
    /// The two-output map is (η_threshold, η_log_sigma).
    /// The wiggle block operates on the combined linear predictor through the
    /// nonlinear inverse link and has a zero effective linear Jacobian.
    ///
    /// - block 0 (threshold):  output 0 = design rows, output 1 = zeros
    /// - block 1 (log_sigma):  output 0 = zeros, output 1 = design rows
    /// - block 2 (wiggle):     all zeros (nonlinear link modulation)
    pub fn block_effective_jacobian(
        specs: &[ParameterBlockSpec],
        block_idx: usize,
    ) -> Result<Box<dyn BlockEffectiveJacobian>, String> {
        crate::util::block_jacobian::AdditiveWiggleBlockLayout {
            family: "BinomialLocationScaleWiggleFamily",
            n_outputs: 2,
            additive_blocks: &[Self::BLOCK_T, Self::BLOCK_LOG_SIGMA],
            wiggle_block: Some(Self::BLOCK_WIGGLE),
        }
        .block_effective_jacobian(specs, block_idx)
    }
}


impl CustomFamily for BinomialLocationScaleWiggleFamily {
    /// The Binomial location-scale-wiggle joint Hessian depends on β because
    /// it involves the nonlinear link function evaluated at the combined
    /// predictor, which changes with all three coefficient blocks.
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Operator-aware: matrix-free workspace applies joint Hv at
        // O(n · (p_t + p_ℓ + p_w)); only fall back to the dense build cost when
        // `use_joint_matrix_free_path` declines the operator path.
        crate::families::location_scale_engine::location_scale_coefficient_hessian_cost(
            self.y.len() as u64,
            specs,
        )
    }

    /// The wiggle family carries a structural null-space direction: the
    /// threshold β_t and the overall wiggle-intercept combination
    /// `β_w^⊤ B(q₀)` both shift q = q₀ + B^⊤ β_w additively, which makes the
    /// penalized joint Hessian H = H_L + S near-singular along that
    /// direction (σ_min ≈ ridge_floor ≈ 1e-10).  Under the default `Smooth`
    /// regularization this null direction contributes a first-order
    /// component to `d log|H|/dρ` via `φ'(σ_min) · dσ_min/dρ` that cannot
    /// be matched by the analytic `u^⊤ (dH/dρ) u` formula — the
    /// eigenvector `u` for a near-zero σ is numerically arbitrary inside
    /// the null space, so first-order perturbation theory breaks down.
    /// `HardPseudo` excludes σ ≤ ε from BOTH log|H| and its gradient
    /// consistently, so the null direction drops out of the analytic geometry.
    fn pseudo_logdet_mode(&self) -> crate::custom_family::PseudoLogdetMode {
        crate::custom_family::PseudoLogdetMode::HardPseudo
    }

    fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(None);
        }
        Ok(monotone_wiggle_nonnegative_constraints(spec.design.ncols()))
    }

    fn post_update_block_beta(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(!block_spec.name.is_empty());
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(beta);
        }
        validate_monotone_wiggle_beta_nonnegative(
            &beta,
            "BinomialLocationScaleWiggleFamily post-update",
        )?;
        Ok(beta)
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let wiggle_design = self.wiggle_design(core.q0.view())?;
        let dq_dq0 =
            self.wiggle_dq_dq0(core.q0.view(), block_states[Self::BLOCK_WIGGLE].beta.view())?;
        let threshold_design = self.threshold_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleWiggleFamily exact-newton path is missing threshold design"
                .to_string()
        })?;
        let log_sigma_design = self.log_sigma_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleWiggleFamily exact-newton path is missing log-sigma design"
                .to_string()
        })?;

        // Per-block gradients from the eta-space score.
        //
        //   q = q0 + w(q0), a = dq/dq0
        //   score_q = -m1   (m1 = dF/dq, F = -ℓ)
        //   grad_eta_t[i]  = score_q * a * q0_t
        //   grad_eta_ls[i] = score_q * a * q0_ls
        //   grad_q[i]      = score_q          (wiggle basis acts on q)
        let mut grad_eta_t = Array1::<f64>::zeros(n);
        let mut grad_eta_ls = Array1::<f64>::zeros(n);
        let mut grad_q = Array1::<f64>::zeros(n);
        for i in 0..n {
            let q_i = core.q0[i] + etaw[i];
            let (m1, _, _) = binomial_neglog_q_derivatives_dispatch(
                self.y[i],
                self.weights[i],
                q_i,
                core.mu[i],
                core.dmu_dq[i],
                core.d2mu_dq2[i],
                core.d3mu_dq3[i],
                &self.link_kind,
            );
            let score_q = -m1;
            let q0d = nonwiggle_q_derivs(eta_t[i], core.sigma[i]);
            grad_eta_t[i] = score_q * dq_dq0[i] * q0d.q_t;
            grad_eta_ls[i] = score_q * dq_dq0[i] * q0d.q_ls;
            grad_q[i] = score_q;
        }
        let grad_t = threshold_design.transpose_vector_multiply(&grad_eta_t);
        let grad_ls = log_sigma_design.transpose_vector_multiply(&grad_eta_ls);
        let grad_w = fast_atv(&wiggle_design, &grad_q);

        // Per-block diagonal Hessians without ever materializing the full p×p
        // joint matrix. The shared row-pieces struct exposes block diagonals
        // directly, so the cross blocks (h_tl, h_tw, h_lw) are not formed.
        let (x_t, x_ls) = self
            .exact_joint_dense_block_designs(None)?
            .ok_or("BinomialLocationScaleWiggleFamily: joint block designs unavailable")?;
        let pieces = self.wiggle_hessian_row_pieces(block_states)?;
        let (h_tt, h_ll, h_ww) = pieces.assemble_block_diagonals(&x_t, &x_ls)?;
        Ok(FamilyEvaluation {
            log_likelihood: core.log_likelihood,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_t,
                    hessian: SymmetricMatrix::Dense(h_tt),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_ls,
                    hessian: SymmetricMatrix::Dense(h_ll),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_w,
                    hessian: SymmetricMatrix::Dense(h_ww),
                },
            ],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        binomial_location_scale_ll_only(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )
    }

    /// Outer-only log-likelihood with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `Some`, only the sampled rows
    /// contribute; each row's per-row log-likelihood term is multiplied by
    /// `WeightedOuterRow.weight`, the Horvitz–Thompson inverse-inclusion
    /// factor 1/π_i (uniform or stratified sampling both supported), so the
    /// partial sum is an unbiased estimator of the full-data log-likelihood.
    /// When `None`, this returns the full-data `log_likelihood_only`. Inner
    /// PIRLS line searches never install the subsample option, so they
    /// continue to score the exact full-data log-likelihood.
    fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        let Some(subsample) = options.outer_score_subsample.as_ref() else {
            return self.log_likelihood_only(block_states);
        };
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        use rayon::iter::ParallelIterator;
        let link_kind = &self.link_kind;
        let ll: Result<f64, String> = subsample
            .rows
            .par_iter()
            .try_fold(
                || 0.0_f64,
                |acc, row| -> Result<f64, String> {
                    let i = row.index;
                    let wi = self.weights[i];
                    if wi == 0.0 {
                        return Ok(acc);
                    }
                    let SigmaJet1 { sigma, .. } = exp_sigma_jet1_scalar(eta_ls[i]);
                    let q0 = binomial_location_scale_q0(eta_t[i], sigma);
                    let q = q0 + etaw[i];
                    let mu = if matches!(link_kind, InverseLink::Standard(StandardLink::Probit)) {
                        0.5
                    } else {
                        let jet = inverse_link_jet_for_inverse_link(link_kind, q).map_err(|e| {
                            format!("location-scale inverse-link evaluation failed: {e}")
                        })?;
                        jet.mu
                    };
                    let term =
                        binomial_location_scale_log_likelihood(self.y[i], wi, q, link_kind, mu)?;
                    Ok(acc + row.weight * term)
                },
            )
            .try_reduce(|| 0.0_f64, |a, b| Ok(a + b));
        ll
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let (x_t, x_ls) = self.dense_block_designs()?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let core0 = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core0.q0.view())?;
        let pw = b0.ncols();
        let total = pt + pls + pw;

        let (range_start, range_end) = match block_idx {
            Self::BLOCK_T => (0usize, pt),
            Self::BLOCK_LOG_SIGMA => (pt, pt + pls),
            Self::BLOCK_WIGGLE => (pt + pls, total),
            _ => return Ok(None),
        };
        if d_beta.len() != (range_end - range_start) {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "block {block_idx} d_beta length mismatch: got {}, expected {}",
                    d_beta.len(),
                    range_end - range_start
                ),
            }
            .into());
        }

        // Block-local exact Newton directional derivative is extracted from the
        // full joint directional Hessian.
        //
        // For the 3-block wiggle model with beta=(beta_t,beta_ls,betaw),
        // define the full negative-loglik Hessian H(beta) in flattened block
        // coordinates. For a direction that moves only one block,
        //
        //   u = [u_t, 0,   0]   or
        //   u = [0,   u_ls,0]   or
        //   u = [0,   0,   uw],
        //
        // the exact blockwise directional Hessian required by the trait is just
        // the corresponding principal block of D H[u]:
        //
        //   D H_block[u_block]
        //   = (D H_joint[u])_{block,block}.
        //
        // This avoids maintaining a second, partially duplicated derivation for
        // the block-local case and keeps the exact-newton block callback aligned
        // with the already-validated joint formulas.
        let mut d_beta_flat = Array1::<f64>::zeros(total);
        match block_idx {
            Self::BLOCK_T => {
                d_beta_flat.slice_mut(s![0..pt]).assign(d_beta);
            }
            Self::BLOCK_LOG_SIGMA => {
                d_beta_flat.slice_mut(s![pt..pt + pls]).assign(d_beta);
            }
            Self::BLOCK_WIGGLE => {
                d_beta_flat.slice_mut(s![pt + pls..]).assign(d_beta);
            }
            _ => {}
        }
        let d_joint = self
            .exact_newton_joint_hessian_directional_derivative(block_states, &d_beta_flat)?
            .ok_or_else(|| "missing exact wiggle joint dH".to_string())?;
        let out = d_joint
            .slice(s![range_start..range_end, range_start..range_end])
            .to_owned();
        Ok(Some(out))
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        // Exact joint Hessian for the 3-block binomial location-scale wiggle family.
        //
        // Model:
        //   q0 = -eta_t / sigma(eta_ls),
        //   q  = q0 + betaw^T B(q0),
        //   mu = Phi(q),
        //   F  = -sum_i ell_i(mu_i).
        //
        // The shared rowwise weights (coeff_tt, coeff_tl, coeff_ll, coeff_tw_b,
        // coeff_tw_d, coeff_lw_b, coeff_lw_d, coeffww) plus the realized B/B'
        // basis arrays are computed once by `wiggle_hessian_row_pieces` and
        // assembled here into the dense p×p matrix. The matrix-free workspace
        // path reuses the exact same row pieces to apply H to a vector
        // without ever forming the dense matrix.
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(None)? else {
            return Ok(None);
        };
        let pieces = self.wiggle_hessian_row_pieces(block_states)?;
        Ok(Some(pieces.assemble_dense(&x_t, &x_ls)?))
    }

    fn joint_jeffreys_information_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.expected_wiggle_information_with_specs(block_states, specs)
    }

    fn joint_jeffreys_information_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.expected_wiggle_information_directional_with_specs(block_states, specs, d_beta_flat)
    }

    fn joint_jeffreys_information_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.expected_wiggle_information_second_directional_with_specs(
            block_states,
            specs,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn joint_jeffreys_information_matches_observed_hessian(&self) -> bool {
        // Expected Fisher information override (gam#1020): observed-Hessian
        // conditioning pre-checks must not skip the expected-information gate.
        false
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Exact directional derivative dH[u] for the same 3-block model.
        //
        // Direction:
        //   u = (u_t, u_l, uw),
        //   d_eta_t = X_t u_t, d_eta_l = X_l u_l.
        //
        // Canonical objective identity for scalar-q composition:
        //   dH_ab[u] =
        //      m3 * dq * q_a q_b
        //    + m2 * (dq_a q_b + q_a dq_b + dq q_ab)
        //    + m1 * dq_ab
        // where (m1,m2,m3) are derivatives of F wrt q.
        //
        // Log-likelihood derivative relation used in code:
        //   s = d ell/dq, c = d² ell/dq², t = d³ ell/dq³
        //   m1 = -s, m2 = -c, m3 = -t.
        //
        // Required analytic chain terms:
        //
        // 1) Wiggle scalars:
        //   m  = 1 + betaw^T B'(q0)
        //   g2 = betaw^T B''(q0)
        //   g3 = betaw^T B'''(q0)
        //
        // 2) Directional wiggle scalars:
        //   dm  = (B'·uw)  + g2*dq0
        //   dg2 = (B''·uw) + g3*dq0
        //
        // 3) Directional q pieces:
        //   dq   = m*dq0 + B·uw
        //   dq_t = dm*q0_t + m*dq0_t
        //   dq_l = dm*q0_l + m*dq0_l
        //
        // 4) Directional second q pieces:
        //   dq_tt = dg2*q0_t*q0_t + g2*(2*q0_t*dq0_t)
        //   dq_tl = dg2*q0_t*q0_l + g2*(dq0_t*q0_l + q0_t*dq0_l)
        //           + dm*q0_tl + m*dq0_tl
        //   dq_ll = dg2*q0_l*q0_l + g2*(2*q0_l*dq0_l)
        //           + dm*q0_ll + m*dq0_ll
        //
        // 5) Mixed w-block directional terms:
        //   qw   = B,         dqw   = B' dq0
        //   q_tw  = q0_t B',   dq_tw  = dq0_t B' + dq0 q0_t B''
        //   q_lw  = q0_l B',   dq_lw  = dq0_l B' + dq0 q0_l B''
        //   qww  = 0,         dqww  = 0
        //
        // Implementation below follows these formulas exactly block-by-block.
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(None)? else {
            return Ok(None);
        };
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let betaw0 = block_states[Self::BLOCK_WIGGLE].beta.clone();
        let core0 = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core0.q0.view())?;
        let pw = b0.ncols();
        let beta_layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
        let total = beta_layout.total();
        let (u_t, u_ls, uw) = beta_layout.split_three(d_beta_flat, "wiggle joint d_beta")?;
        let d_eta_t = fast_av(&x_t, &u_t);
        let d_eta_ls = fast_av(&x_ls, &u_ls);

        let d0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::second_derivative())?;
        let d3q = self.wiggle_d3q_dq03(core0.q0.view(), betaw0.view())?;
        if d0.ncols() != betaw0.len() || dd0.ncols() != betaw0.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "wiggle derivative/beta mismatch in exact joint dH: B'={} B''={} betaw={}",
                    d0.ncols(),
                    dd0.ncols(),
                    betaw0.len()
                ),
            }
            .into());
        }
        let m = d0.dot(&betaw0) + 1.0;
        let g2 = dd0.dot(&betaw0);
        let g3 = d3q;
        let (sigma, ..) = exp_sigma_derivs_up_to_third(eta_ls.view());

        let BinomialWiggleDhRowCoeffs {
            coeff_tt,
            coeff_tl,
            coeff_ll,
            coeff_tw_b,
            coeff_tw_d,
            coeff_tw_dd,
            coeff_lw_b,
            coeff_lw_d,
            coeff_lw_dd,
            coeffww_bb,
            coeffww_db,
        } = self.binomial_wiggle_dh_row_coeffs(
            n,
            &BinomialWiggleDhRowInputs {
                core0: &core0,
                eta_t,
                etaw,
                sigma: &sigma,
                m: &m,
                g2: &g2,
                g3: &g3,
                b0: &b0,
                d0: &d0,
                dd0: &dd0,
                uw: &uw,
                d_eta_t: &d_eta_t,
                d_eta_ls: &d_eta_ls,
            },
        );
        let d_h_tt = xt_diag_x_dense(&x_t, &coeff_tt)?;
        let d_h_tl = xt_diag_y_dense(&x_t, &coeff_tl, &x_ls)?;
        let d_h_ll = xt_diag_x_dense(&x_ls, &coeff_ll)?;
        let d_h_tw = xt_diag_y_dense(&x_t, &coeff_tw_b, &b0)?
            + &xt_diag_y_dense(&x_t, &coeff_tw_d, &d0)?
            + &xt_diag_y_dense(&x_t, &coeff_tw_dd, &dd0)?;
        let d_h_lw = xt_diag_y_dense(&x_ls, &coeff_lw_b, &b0)?
            + &xt_diag_y_dense(&x_ls, &coeff_lw_d, &d0)?
            + &xt_diag_y_dense(&x_ls, &coeff_lw_dd, &dd0)?;
        let mut d_hww = xt_diag_x_dense(&b0, &coeffww_bb)?;
        d_hww += &xt_diag_y_dense(&d0, &coeffww_db, &b0)?;
        d_hww += &xt_diag_y_dense(&b0, &coeffww_db, &d0)?;

        let mut d_h = Array2::<f64>::zeros((total, total));
        d_h.slice_mut(s![0..pt, 0..pt]).assign(&d_h_tt);
        d_h.slice_mut(s![0..pt, pt..pt + pls]).assign(&d_h_tl);
        d_h.slice_mut(s![pt..pt + pls, pt..pt + pls])
            .assign(&d_h_ll);
        d_h.slice_mut(s![0..pt, pt + pls..total]).assign(&d_h_tw);
        d_h.slice_mut(s![pt..pt + pls, pt + pls..total])
            .assign(&d_h_lw);
        d_h.slice_mut(s![pt + pls..total, pt + pls..total])
            .assign(&d_hww);
        mirror_upper_to_lower(&mut d_h);
        Ok(Some(d_h))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }

        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(None)? else {
            return Ok(None);
        };
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let betaw0 = block_states[Self::BLOCK_WIGGLE].beta.clone();
        let core0 = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core0.q0.view())?;
        let d0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::second_derivative())?;
        let d3_basis = self.wiggle_d3basis_constrained(core0.q0.view())?;
        let d3q = self.wiggle_d3q_dq03(core0.q0.view(), betaw0.view())?;
        let d4q = self.wiggle_d4q_dq04(core0.q0.view(), betaw0.view())?;
        let pw = b0.ncols();
        let beta_layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
        let total = beta_layout.total();
        if d0.ncols() != betaw0.len()
            || dd0.ncols() != betaw0.len()
            || d3_basis.ncols() != betaw0.len()
        {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle derivative/beta mismatch in exact joint d2H: B'={} B''={} B'''={} betaw={}",
                d0.ncols(),
                dd0.ncols(),
                d3_basis.ncols(),
                betaw0.len()
            ) }.into());
        }

        let (u_t, u_ls, uw) = beta_layout.split_three(d_beta_u_flat, "wiggle joint d_beta_u")?;
        let (v_t, v_ls, vw) = beta_layout.split_three(d_betav_flat, "wiggle joint d_betav")?;
        let d_eta_t_u = fast_av(&x_t, &u_t);
        let d_eta_ls_u = fast_av(&x_ls, &u_ls);
        let d_eta_tv = fast_av(&x_t, &v_t);
        let d_eta_lsv = fast_av(&x_ls, &v_ls);

        let m = d0.dot(&betaw0) + 1.0;
        let g2 = dd0.dot(&betaw0);
        let g3 = d3q;
        let g4 = d4q;
        let (sigma, ds, d2s, d3s, d4s) = exp_sigma_derivs_up_to_fourth_array(eta_ls.view());

        let mut d2_h: Array2<f64> = (0..n)
            .into_par_iter()
            .map(|i| -> Result<Array2<f64>, String> {
                let mut row_h = Array2::<f64>::zeros((total, total));
                // Per-row scalar objective derivatives for F_i(q).
                let q_i = core0.q0[i] + etaw[i];
                let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
                    self.y[i],
                    self.weights[i],
                    q_i,
                    core0.mu[i],
                    core0.dmu_dq[i],
                    core0.d2mu_dq2[i],
                    core0.d3mu_dq3[i],
                    &self.link_kind,
                );
                let m4 = binomial_neglog_q_fourth_derivative_dispatch(
                    self.y[i],
                    self.weights[i],
                    q_i,
                    core0.mu[i],
                    core0.dmu_dq[i],
                    core0.d2mu_dq2[i],
                    core0.d3mu_dq3[i],
                    &self.link_kind,
                )?;

                // Non-wiggle q0(eta_t, eta_ls) derivatives and sigma-ratio helpers.
                let q0 = nonwiggle_q_derivs(eta_t[i], sigma[i]);
                let s_safe = sigma[i];
                let s2 = s_safe * s_safe;
                let s3 = s2 * s_safe;
                let s4 = s3 * s_safe;
                let s5 = s4 * s_safe;
                let q0_tl_ls_ls =
                    d3s[i] / s2 - 6.0 * ds[i] * d2s[i] / s3 + 6.0 * ds[i] * ds[i] * ds[i] / s4;
                let q0_tl_ls_ls_ls =
                    d4s[i] / s2 - 8.0 * ds[i] * d3s[i] / s3 - 6.0 * d2s[i] * d2s[i] / s3
                        + 36.0 * ds[i] * ds[i] * d2s[i] / s4
                        - 24.0 * ds[i] * ds[i] * ds[i] * ds[i] / s5;
                let q0_ll_ls_ls = eta_t[i] * q0_tl_ls_ls_ls;

                let u_t_i = d_eta_t_u[i];
                let u_ls_i = d_eta_ls_u[i];
                let v_t_i = d_eta_tv[i];
                let v_ls_i = d_eta_lsv[i];

                // Directional z=q0 primitives for u and v.
                let dq0_u = q0.q_t * u_t_i + q0.q_ls * u_ls_i;
                let dq0v = q0.q_t * v_t_i + q0.q_ls * v_ls_i;
                let d2q0_uv =
                    q0.q_tl * (u_t_i * v_ls_i + v_t_i * u_ls_i) + q0.q_ll * u_ls_i * v_ls_i;

                let dq0_t_u = q0.q_tl * u_ls_i;
                let dq0_tv = q0.q_tl * v_ls_i;
                let dq0_ls_u = q0.q_tl * u_t_i + q0.q_ll * u_ls_i;
                let dq0_lsv = q0.q_tl * v_t_i + q0.q_ll * v_ls_i;
                let dq0_tl_u = q0.q_tl_ls * u_ls_i;
                let dq0_tlv = q0.q_tl_ls * v_ls_i;
                let dq0_ll_u = q0.q_tl_ls * u_t_i + q0.q_ll_ls * u_ls_i;
                let dq0_llv = q0.q_tl_ls * v_t_i + q0.q_ll_ls * v_ls_i;

                let d2q0_t_uv = q0.q_tl_ls * u_ls_i * v_ls_i;
                let d2q0_ls_uv =
                    q0.q_tl_ls * (u_ls_i * v_t_i + v_ls_i * u_t_i) + q0.q_ll_ls * u_ls_i * v_ls_i;
                let d2q0_tl_uv = q0_tl_ls_ls * u_ls_i * v_ls_i;
                let d2q0_ll_uv =
                    q0_tl_ls_ls * (u_t_i * v_ls_i + v_t_i * u_ls_i) + q0_ll_ls_ls * u_ls_i * v_ls_i;

                let br = b0.row(i);
                let dr = d0.row(i);
                let ddr = dd0.row(i);
                let d3r = d3_basis.row(i);
                let b_u = br.dot(&uw);
                let bv = br.dot(&vw);
                let b1_u = dr.dot(&uw);
                let b1v = dr.dot(&vw);
                let b2_u = ddr.dot(&uw);
                let b2v = ddr.dot(&vw);
                let b3_u = d3r.dot(&uw);
                let b3v = d3r.dot(&vw);

                // Wiggle scalar chain terms:
                //   m = 1 + g1,     g2 = betaw^T B''(q0),
                //   dm[u]   = B'·uw + g2*dq0[u],
                //   d2m[u,v]= g3*dq0[u]dq0[v] + g2*d2q0[u,v] + (B''·vw)dq0[u] + (B''·uw)dq0[v],
                //   dg2[u]  = B''·uw + g3*dq0[u],
                //   d2g2[u,v]=g4*dq0[u]dq0[v] + g3*d2q0[u,v] + (B'''·vw)dq0[u] + (B'''·uw)dq0[v].
                let dm_u = b1_u + g2[i] * dq0_u;
                let dmv = b1v + g2[i] * dq0v;
                let d2m_uv = g3[i] * dq0_u * dq0v + g2[i] * d2q0_uv + b2v * dq0_u + b2_u * dq0v;
                let dg2_u = b2_u + g3[i] * dq0_u;
                let dg2v = b2v + g3[i] * dq0v;
                let d2g2_uv = g4[i] * dq0_u * dq0v + g3[i] * d2q0_uv + b3v * dq0_u + b3_u * dq0v;

                // First/second directional terms for total q.
                let dq_u = m[i] * dq0_u + b_u;
                let dqv = m[i] * dq0v + bv;
                // Simplify exact formula for q = q0 + betaw^T B(q0):
                //   D²q[u,v] = m*d²q0 + g2*dq0[u]dq0[v] + (B'·uw)dq0[v] + (B'·vw)dq0[u].
                let d2q_uv = m[i] * d2q0_uv + g2[i] * dq0_u * dq0v + b1_u * dq0v + b1v * dq0_u;

                // q partials by block and their first/second directional derivatives.
                let q_t = m[i] * q0.q_t;
                let q_ls = m[i] * q0.q_ls;
                let q_tt = g2[i] * q0.q_t * q0.q_t;
                let q_tl = g2[i] * q0.q_t * q0.q_ls + m[i] * q0.q_tl;
                let q_ll = g2[i] * q0.q_ls * q0.q_ls + m[i] * q0.q_ll;

                let dq_t_u = dm_u * q0.q_t + m[i] * dq0_t_u;
                let dq_tv = dmv * q0.q_t + m[i] * dq0_tv;
                let dq_ls_u = dm_u * q0.q_ls + m[i] * dq0_ls_u;
                let dq_lsv = dmv * q0.q_ls + m[i] * dq0_lsv;

                let d2q_t_uv = d2m_uv * q0.q_t + dm_u * dq0_tv + dmv * dq0_t_u + m[i] * d2q0_t_uv;
                let d2q_ls_uv =
                    d2m_uv * q0.q_ls + dm_u * dq0_lsv + dmv * dq0_ls_u + m[i] * d2q0_ls_uv;

                let dq_tt_u = dg2_u * q0.q_t * q0.q_t + g2[i] * (2.0 * q0.q_t * dq0_t_u);
                let dq_ttv = dg2v * q0.q_t * q0.q_t + g2[i] * (2.0 * q0.q_t * dq0_tv);
                let d2q_tt_uv = d2g2_uv * q0.q_t * q0.q_t
                    + dg2_u * (2.0 * q0.q_t * dq0_tv)
                    + dg2v * (2.0 * q0.q_t * dq0_t_u)
                    + g2[i] * (2.0 * dq0_t_u * dq0_tv + 2.0 * q0.q_t * d2q0_t_uv);

                let dq_tl_u = dg2_u * q0.q_t * q0.q_ls
                    + g2[i] * (dq0_t_u * q0.q_ls + q0.q_t * dq0_ls_u)
                    + dm_u * q0.q_tl
                    + m[i] * dq0_tl_u;
                let dq_tlv = dg2v * q0.q_t * q0.q_ls
                    + g2[i] * (dq0_tv * q0.q_ls + q0.q_t * dq0_lsv)
                    + dmv * q0.q_tl
                    + m[i] * dq0_tlv;
                let d2q_tl_uv = d2g2_uv * q0.q_t * q0.q_ls
                    + dg2_u * (dq0_tv * q0.q_ls + q0.q_t * dq0_lsv)
                    + dg2v * (dq0_t_u * q0.q_ls + q0.q_t * dq0_ls_u)
                    + g2[i]
                        * (d2q0_t_uv * q0.q_ls
                            + dq0_t_u * dq0_lsv
                            + dq0_tv * dq0_ls_u
                            + q0.q_t * d2q0_ls_uv)
                    + d2m_uv * q0.q_tl
                    + dm_u * dq0_tlv
                    + dmv * dq0_tl_u
                    + m[i] * d2q0_tl_uv;

                let dq_ll_u = dg2_u * q0.q_ls * q0.q_ls
                    + g2[i] * (2.0 * q0.q_ls * dq0_ls_u)
                    + dm_u * q0.q_ll
                    + m[i] * dq0_ll_u;
                let dq_llv = dg2v * q0.q_ls * q0.q_ls
                    + g2[i] * (2.0 * q0.q_ls * dq0_lsv)
                    + dmv * q0.q_ll
                    + m[i] * dq0_llv;
                let d2q_ll_uv = d2g2_uv * q0.q_ls * q0.q_ls
                    + dg2_u * (2.0 * q0.q_ls * dq0_lsv)
                    + dg2v * (2.0 * q0.q_ls * dq0_ls_u)
                    + g2[i] * (2.0 * dq0_ls_u * dq0_lsv + 2.0 * q0.q_ls * d2q0_ls_uv)
                    + d2m_uv * q0.q_ll
                    + dm_u * dq0_llv
                    + dmv * dq0_ll_u
                    + m[i] * d2q0_ll_uv;

                // Exact second directional coefficients for the scalar block weights.
                let coeff_tt = second_directionalhessian_coeff_fromobjective_q_terms(
                    m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_t, q_t, q_tt, dq_t_u, dq_tv, dq_t_u,
                    dq_tv, d2q_t_uv, d2q_t_uv, dq_tt_u, dq_ttv, d2q_tt_uv,
                );
                let coeff_tl = second_directionalhessian_coeff_fromobjective_q_terms(
                    m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_t, q_ls, q_tl, dq_t_u, dq_tv, dq_ls_u,
                    dq_lsv, d2q_t_uv, d2q_ls_uv, dq_tl_u, dq_tlv, d2q_tl_uv,
                );
                let coeff_ll = second_directionalhessian_coeff_fromobjective_q_terms(
                    m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_ls, q_ls, q_ll, dq_ls_u, dq_lsv, dq_ls_u,
                    dq_lsv, d2q_ls_uv, d2q_ls_uv, dq_ll_u, dq_llv, d2q_ll_uv,
                );

                let xtr = x_t.row(i);
                let xlsr = x_ls.row(i);
                for a_idx in 0..pt {
                    for b_idx in a_idx..pt {
                        row_h[[a_idx, b_idx]] += coeff_tt * xtr[a_idx] * xtr[b_idx];
                    }
                }
                for a_idx in 0..pt {
                    for b_idx in 0..pls {
                        row_h[[a_idx, pt + b_idx]] += coeff_tl * xtr[a_idx] * xlsr[b_idx];
                    }
                }
                for a_idx in 0..pls {
                    for b_idx in a_idx..pls {
                        row_h[[pt + a_idx, pt + b_idx]] += coeff_ll * xlsr[a_idx] * xlsr[b_idx];
                    }
                }

                for j in 0..pw {
                    let qw = br[j];
                    let dqw_u = dr[j] * dq0_u;
                    let dqwv = dr[j] * dq0v;
                    let d2qw_uv = ddr[j] * dq0_u * dq0v + dr[j] * d2q0_uv;
                    let q_tw = dr[j] * q0.q_t;
                    let q_lw = dr[j] * q0.q_ls;
                    let dq_tw_u = ddr[j] * dq0_u * q0.q_t + dr[j] * dq0_t_u;
                    let dq_twv = ddr[j] * dq0v * q0.q_t + dr[j] * dq0_tv;
                    let d2q_tw_uv = d3r[j] * dq0_u * dq0v * q0.q_t
                        + ddr[j] * (d2q0_uv * q0.q_t + dq0_u * dq0_tv + dq0v * dq0_t_u)
                        + dr[j] * d2q0_t_uv;
                    let dq_lw_u = ddr[j] * dq0_u * q0.q_ls + dr[j] * dq0_ls_u;
                    let dq_lwv = ddr[j] * dq0v * q0.q_ls + dr[j] * dq0_lsv;
                    let d2q_lw_uv = d3r[j] * dq0_u * dq0v * q0.q_ls
                        + ddr[j] * (d2q0_uv * q0.q_ls + dq0_u * dq0_lsv + dq0v * dq0_ls_u)
                        + dr[j] * d2q0_ls_uv;

                    let coeff_tw = second_directionalhessian_coeff_fromobjective_q_terms(
                        m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_t, qw, q_tw, dq_t_u, dq_tv, dqw_u,
                        dqwv, d2q_t_uv, d2qw_uv, dq_tw_u, dq_twv, d2q_tw_uv,
                    );
                    let coeff_lw = second_directionalhessian_coeff_fromobjective_q_terms(
                        m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_ls, qw, q_lw, dq_ls_u, dq_lsv, dqw_u,
                        dqwv, d2q_ls_uv, d2qw_uv, dq_lw_u, dq_lwv, d2q_lw_uv,
                    );

                    for a_idx in 0..pt {
                        row_h[[a_idx, pt + pls + j]] += coeff_tw * xtr[a_idx];
                    }
                    for a_idx in 0..pls {
                        row_h[[pt + a_idx, pt + pls + j]] += coeff_lw * xlsr[a_idx];
                    }
                }

                for j in 0..pw {
                    let qwj = br[j];
                    let dqwj_u = dr[j] * dq0_u;
                    let dqwjv = dr[j] * dq0v;
                    let d2qwj_uv = ddr[j] * dq0_u * dq0v + dr[j] * d2q0_uv;
                    for k in j..pw {
                        let qwk = br[k];
                        let dqwk_u = dr[k] * dq0_u;
                        let dqwkv = dr[k] * dq0v;
                        let d2qwk_uv = ddr[k] * dq0_u * dq0v + dr[k] * d2q0_uv;
                        let coeffww = second_directionalhessian_coeff_fromobjective_q_terms(
                            m1, m2, m3, m4, dq_u, dqv, d2q_uv, qwj, qwk, 0.0, dqwj_u, dqwjv,
                            dqwk_u, dqwkv, d2qwj_uv, d2qwk_uv, 0.0, 0.0, 0.0,
                        );
                        row_h[[pt + pls + j, pt + pls + k]] += coeffww;
                    }
                }

                Ok(row_h)
            })
            .try_reduce(
                || Array2::<f64>::zeros((total, total)),
                |mut acc, row_h| {
                    acc += &row_h;
                    Ok(acc)
                },
            )?;

        mirror_upper_to_lower(&mut d2_h);
        Ok(Some(d2_h))
    }

    fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(shadow) = self.shadow_with_exact_joint_designs(specs)? else {
            return Ok(None);
        };
        shadow.exact_newton_joint_hessian(block_states)
    }

    fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(shadow) = self.shadow_with_exact_joint_designs(specs)? else {
            return Ok(None);
        };
        shadow.exact_newton_joint_hessian_directional_derivative(block_states, d_beta_flat)
    }

    fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(shadow) = self.shadow_with_exact_joint_designs(specs)? else {
            return Ok(None);
        };
        shadow.exact_newton_joint_hessiansecond_directional_derivative(
            block_states,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiTerms>, String> {
        // These three joint psi hooks are the wiggle family's exact
        // likelihood-side contribution to the unified full [rho, psi] outer
        // Hessian:
        //
        //   exact_newton_joint_psi_terms(...)                    -> D_a, D_{beta a}, D_{beta beta a}
        //   exact_newton_joint_psisecond_order_terms(...)       -> D_ab, D_{beta ab}, D_{beta beta ab}
        //   exact_newton_joint_psihessian_directional_derivative(...) -> T_a[u]
        //
        // Generic exact-joint code in custom_family.rs adds all realized
        // penalty motion S_a / S_ab and combines these likelihood-only objects
        // with the joint mode solves beta_i, beta_ij and the total Hessian
        // drifts dot H_i, ddot H_ij. Keeping this contract explicit is what
        // makes the wiggle family's full [rho, psi] Hessian real rather than a
        // gradient-only or block-local surrogate.
        self.exact_newton_joint_psi_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
        )
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.exact_newton_joint_psisecond_order_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_i,
            psi_j,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_psihessian_directional_derivative_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if !self.exact_joint_supported() {
            return Ok(None);
        }
        Ok(Some(Arc::new(
            BinomialLocationScaleWiggleExactNewtonJointPsiWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
            )?,
        )))
    }

    fn block_geometry(
        &self,
        block_states: &[ParameterBlockState],
        spec: &crate::custom_family::ParameterBlockSpec,
    ) -> Result<(DesignMatrix, Array1<f64>), String> {
        if spec.name != "wiggle" {
            return Ok((spec.design.clone(), spec.offset.clone()));
        }
        if block_states.len() < 2 {
            return Err(GamlssError::UnsupportedConfiguration {
                reason: "wiggle geometry requires threshold and log-sigma blocks".to_string(),
            }
            .into());
        }
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != self.y.len() || eta_ls.len() != self.y.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: "wiggle geometry input size mismatch".to_string(),
            }
            .into());
        }
        let mut q0 = Array1::<f64>::zeros(eta_t.len());
        for i in 0..q0.len() {
            let sigma = exp_sigma_from_eta_scalar(eta_ls[i]);
            q0[i] = binomial_location_scale_q0(eta_t[i], sigma);
        }
        let x = self.wiggle_design(q0.view())?;
        if x.ncols() != spec.design.ncols() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "dynamic wiggle design col mismatch: got {}, expected {}",
                    x.ncols(),
                    spec.design.ncols()
                ),
            }
            .into());
        }
        let nrows = x.nrows();
        Ok((
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
            Array1::zeros(nrows),
        ))
    }

    fn block_geometry_is_dynamic(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let workspace = BinomialLocationScaleWiggleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            x_t.into_owned(),
            x_ls.into_owned(),
        )?;
        Ok(Some(Arc::new(workspace)))
    }

    /// Outer-aware joint-Hessian workspace with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `None`, this is byte-identical
    /// to `exact_newton_joint_hessian_workspace`. When `Some`, the precomputed
    /// per-row coefficient arrays in `pieces` (`coeff_tt`, `coeff_tl`,
    /// `coeff_ll`, `coeff_tw_b`, `coeff_tw_d`, `coeff_lw_b`, `coeff_lw_d`,
    /// `coeffww`) — which every downstream assembly (`hessian_dense`,
    /// `hessian_matvec`, `hessian_diagonal`) consumes row-linearly via
    /// `Xᵀ diag(W) Y` — are replaced by a Horvitz–Thompson mask: each sampled
    /// row's coefficient is multiplied by `WeightedOuterRow.weight` (the
    /// inverse-inclusion factor 1/π_i; uniform or stratified sampling both
    /// supported), and non-sampled rows are zeroed. The resulting joint
    /// Hessian is an unbiased estimator of the full-data joint Hessian.
    /// Inner PIRLS never installs the option, so the inner solve continues
    /// to consume the exact full-data Hessian.
    fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let mut workspace = BinomialLocationScaleWiggleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            x_t.into_owned(),
            x_ls.into_owned(),
        )?;
        if let Some(subsample) = options.outer_score_subsample.as_ref() {
            workspace.apply_outer_subsample(subsample.rows.as_ref());
        }
        Ok(Some(Arc::new(workspace)))
    }

    /// Outer-derivative policy: declare HT-subsample capability.
    ///
    /// BinomialLocationScaleWiggleFamily overrides
    /// `log_likelihood_only_with_options` and
    /// `exact_newton_joint_hessian_workspace_with_options` to consume
    /// `options.outer_score_subsample` with per-row Horvitz–Thompson weights
    /// (each sampled row's contribution is multiplied by
    /// `WeightedOuterRow.weight = 1/π_i`; non-sampled rows are zeroed),
    /// yielding unbiased estimators of the full-data log-likelihood and
    /// joint Hessian. The ψ-workspace path is not yet subsample-aware: it
    /// builds the exact full-data ψ Hessian blocks, which are trivially
    /// unbiased; so the outer-score components are a sum of HT-unbiased and
    /// exact-unbiased pieces and the total remains an unbiased estimator of
    /// the full-data outer score. Inner-PIRLS and final-covariance paths
    /// never install the option, so they continue to consume the exact
    /// full-data quantities.
    fn outer_derivative_subsample_capable(&self) -> bool {
        true
    }

    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        // Same gating as the workspace impl: matrix-free path is available
        // when both threshold and log-σ block designs are present (the
        // wiggle block is folded into the per-row pieces inside
        // `BinomialLocationScaleWiggleHessianWorkspace`). This advertises
        // β-space representation support only.
        self.exact_joint_supported()
            && matches!(
                self.exact_joint_dense_block_designs(Some(specs)),
                Ok(Some(_))
            )
    }
}


impl BinomialLocationScaleWiggleFamily {
    /// Build a matrix-free `RowCoeffOperator` for the BLS Wiggle joint
    /// directional derivative `D_β H_L[u]`. Channels (in order):
    /// X_t, X_ls, B (b0), B' (d0), B'' (dd0). The operator acts on the
    /// joint coefficient vector `(β_t, β_ls, β_w)`.
    fn bls_wiggle_directional_operator(
        &self,
        block_states: &[ParameterBlockState],
        x_t_arc: Arc<Array2<f64>>,
        x_ls_arc: Arc<Array2<f64>>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let pt = x_t_arc.ncols();
        let pls = x_ls_arc.ncols();
        let betaw0 = block_states[Self::BLOCK_WIGGLE].beta.clone();
        let core0 = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core0.q0.view())?;
        let pw = b0.ncols();
        let beta_layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
        let total = beta_layout.total();
        if d_beta_flat.len() != total {
            return Err(GamlssError::InvalidInput {
                reason: format!(
                    "BLS wiggle dH operator: d_beta length {} != {}",
                    d_beta_flat.len(),
                    total
                ),
            }
            .into());
        }
        let (u_t, u_ls, uw) =
            beta_layout.split_three(d_beta_flat, "wiggle joint dH operator d_beta")?;
        let d_eta_t = fast_av(x_t_arc.as_ref(), &u_t);
        let d_eta_ls = fast_av(x_ls_arc.as_ref(), &u_ls);

        let d0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::second_derivative())?;
        let d3q = self.wiggle_d3q_dq03(core0.q0.view(), betaw0.view())?;
        if d0.ncols() != betaw0.len() || dd0.ncols() != betaw0.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "wiggle derivative/beta mismatch in dH operator: B'={} B''={} betaw={}",
                    d0.ncols(),
                    dd0.ncols(),
                    betaw0.len()
                ),
            }
            .into());
        }
        let m = d0.dot(&betaw0) + 1.0;
        let g2 = dd0.dot(&betaw0);
        let g3 = d3q;
        let (sigma, ..) = exp_sigma_derivs_up_to_third(eta_ls.view());

        let BinomialWiggleDhRowCoeffs {
            coeff_tt,
            coeff_tl,
            coeff_ll,
            coeff_tw_b,
            coeff_tw_d,
            coeff_tw_dd,
            coeff_lw_b,
            coeff_lw_d,
            coeff_lw_dd,
            coeffww_bb,
            coeffww_db,
        } = self.binomial_wiggle_dh_row_coeffs(
            n,
            &BinomialWiggleDhRowInputs {
                core0: &core0,
                eta_t,
                etaw,
                sigma: &sigma,
                m: &m,
                g2: &g2,
                g3: &g3,
                b0: &b0,
                d0: &d0,
                dd0: &dd0,
                uw: &uw,
                d_eta_t: &d_eta_t,
                d_eta_ls: &d_eta_ls,
            },
        );

        let basis: Arc<Array2<f64>> = Arc::new(b0);
        let basis_d1: Arc<Array2<f64>> = Arc::new(d0);
        let basis_d2: Arc<Array2<f64>> = Arc::new(dd0);

        Ok(Some(Arc::new(RowCoeffOperator::from_directions(
            vec![pt, pls, pw],
            vec![
                (0, x_t_arc),
                (1, x_ls_arc),
                (2, basis),
                (2, basis_d1),
                (2, basis_d2),
            ],
            vec![
                // (X_t, X_t)  ← `xt_diag_x_dense(&x_t, &coeff_tt)`
                (0, 0, coeff_tt),
                // (X_t, X_ls) ← `xt_diag_y_dense(&x_t, &coeff_tl, &x_ls)`
                (0, 1, coeff_tl),
                // (X_ls, X_ls) ← `xt_diag_x_dense(&x_ls, &coeff_ll)`
                (1, 1, coeff_ll),
                // (X_t, B / B' / B'') ← three sub-blocks of d_h_tw =
                // `xt_diag_y_dense(x_t, coeff_tw_b, b0) + xt_diag_y_dense(
                //  x_t, coeff_tw_d, d0) + xt_diag_y_dense(x_t, coeff_tw_dd, dd0)`
                (0, 2, coeff_tw_b),
                (0, 3, coeff_tw_d),
                (0, 4, coeff_tw_dd),
                // (X_ls, B / B' / B'') ← analogous d_h_lw triple
                (1, 2, coeff_lw_b),
                (1, 3, coeff_lw_d),
                (1, 4, coeff_lw_dd),
                // (B, B) ← `xt_diag_x_dense(&b0, &coeffww_bb)`
                (2, 2, coeffww_bb),
                // (B, B') ← `xt_diag_y_dense(&d0, &coeffww_db, &b0) +
                // xt_diag_y_dense(&b0, &coeffww_db, &d0)` =
                // d0^T diag(c) b0 + b0^T diag(c) d0 (symmetric pair)
                (2, 3, coeffww_db),
            ],
            n,
        ))))
    }

    /// Build a matrix-free `RowCoeffOperator` for the BLS Wiggle joint
    /// second directional derivative `D²_β H_L[u, v]`. Channels: X_t,
    /// X_ls, B, B', B'', B'''.
    ///
    /// The dense path computes a per-row scalar `coeff_*(i, j[, k])` via
    /// `second_directionalhessian_coeff_fromobjective_q_terms` and outer-
    /// products it into the (t,t) / (t,ls) / (ls,ls) / (t,w) / (ls,w) /
    /// (w,w) blocks. Each `coeff_tw(i, j)` is *linear* in the basis
    /// derivatives at column j (`br[j], dr[j], ddr[j], d3r[j]` — they
    /// only ever appear once in the q-Hessian directional polynomial),
    /// so each per-(i,j) contribution decomposes into 4 channel-pair
    /// row coefficients (X_t, B/B'/B''/B'''). The wiggle-wiggle term
    /// `coeff_ww(i, j, k)` is *bilinear* in (br[j], dr[j], ddr[j]) ⊗
    /// (br[k], dr[k], ddr[k]), giving 4 symmetric pair coefficients on
    /// (B, B), (B, B'), (B, B''), (B', B'). No (B'', B'') term — the
    /// formula is at most degree 2 in any single basis derivative.
    fn bls_wiggle_second_directional_operator(
        &self,
        block_states: &[ParameterBlockState],
        x_t_arc: Arc<Array2<f64>>,
        x_ls_arc: Arc<Array2<f64>>,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.y.len();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != n || eta_ls.len() != n || etaw.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleWiggleFamily input size mismatch".to_string(),
            }
            .into());
        }
        let pt = x_t_arc.ncols();
        let pls = x_ls_arc.ncols();
        let betaw0 = block_states[Self::BLOCK_WIGGLE].beta.clone();
        let core0 = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core0.q0.view())?;
        let d0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::second_derivative())?;
        let d3_basis = self.wiggle_d3basis_constrained(core0.q0.view())?;
        let d3q = self.wiggle_d3q_dq03(core0.q0.view(), betaw0.view())?;
        let d4q = self.wiggle_d4q_dq04(core0.q0.view(), betaw0.view())?;
        let pw = b0.ncols();
        let beta_layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
        let total = beta_layout.total();
        if d_beta_u.len() != total || d_beta_v.len() != total {
            return Err(GamlssError::InvalidInput {
                reason: format!(
                    "BLS wiggle d2H operator: d_beta_{{u,v}} length {}/{} != {}",
                    d_beta_u.len(),
                    d_beta_v.len(),
                    total
                ),
            }
            .into());
        }
        if d0.ncols() != betaw0.len()
            || dd0.ncols() != betaw0.len()
            || d3_basis.ncols() != betaw0.len()
        {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle derivative/beta mismatch in d2H operator: B'={} B''={} B'''={} betaw={}",
                d0.ncols(),
                dd0.ncols(),
                d3_basis.ncols(),
                betaw0.len()
            ) }.into());
        }

        let (u_t, u_ls, uw) = beta_layout.split_three(d_beta_u, "wiggle d2H op u")?;
        let (v_t, v_ls, vw) = beta_layout.split_three(d_beta_v, "wiggle d2H op v")?;
        let d_eta_t_u = fast_av(x_t_arc.as_ref(), &u_t);
        let d_eta_ls_u = fast_av(x_ls_arc.as_ref(), &u_ls);
        let d_eta_t_v = fast_av(x_t_arc.as_ref(), &v_t);
        let d_eta_ls_v = fast_av(x_ls_arc.as_ref(), &v_ls);

        let m = d0.dot(&betaw0) + 1.0;
        let g2 = dd0.dot(&betaw0);
        let g3 = d3q;
        let g4 = d4q;
        let (sigma, ds, d2s, d3s, d4s) = exp_sigma_derivs_up_to_fourth_array(eta_ls.view());

        // Per-row scalar pair coefficients.
        let mut coeff_tt = Array1::<f64>::zeros(n);
        let mut coeff_tl = Array1::<f64>::zeros(n);
        let mut coeff_ll = Array1::<f64>::zeros(n);
        // Per-row coefficients for the t↔wiggle decomposition into
        // (X_t, B), (X_t, B'), (X_t, B''), (X_t, B''') pair entries.
        let mut alpha_tw_b = Array1::<f64>::zeros(n);
        let mut alpha_tw_d = Array1::<f64>::zeros(n);
        let mut alpha_tw_dd = Array1::<f64>::zeros(n);
        let mut alpha_tw_d3 = Array1::<f64>::zeros(n);
        let mut alpha_lw_b = Array1::<f64>::zeros(n);
        let mut alpha_lw_d = Array1::<f64>::zeros(n);
        let mut alpha_lw_dd = Array1::<f64>::zeros(n);
        let mut alpha_lw_d3 = Array1::<f64>::zeros(n);
        // Wiggle-wiggle bilinear pair entries on (B,B), (B,B'), (B,B''), (B',B').
        let mut c_ww_bb = Array1::<f64>::zeros(n);
        let mut c_ww_bd = Array1::<f64>::zeros(n);
        let mut c_ww_bdd = Array1::<f64>::zeros(n);
        let mut c_ww_dd_pair = Array1::<f64>::zeros(n);

        for i in 0..n {
            let q_i = core0.q0[i] + etaw[i];
            let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
                self.y[i],
                self.weights[i],
                q_i,
                core0.mu[i],
                core0.dmu_dq[i],
                core0.d2mu_dq2[i],
                core0.d3mu_dq3[i],
                &self.link_kind,
            );
            let m4 = binomial_neglog_q_fourth_derivative_dispatch(
                self.y[i],
                self.weights[i],
                q_i,
                core0.mu[i],
                core0.dmu_dq[i],
                core0.d2mu_dq2[i],
                core0.d3mu_dq3[i],
                &self.link_kind,
            )?;

            let q0_d = nonwiggle_q_derivs(eta_t[i], sigma[i]);
            let s_safe = sigma[i];
            let s2 = s_safe * s_safe;
            let s3 = s2 * s_safe;
            let s4 = s3 * s_safe;
            let s5 = s4 * s_safe;
            let q0_tl_ls_ls =
                d3s[i] / s2 - 6.0 * ds[i] * d2s[i] / s3 + 6.0 * ds[i] * ds[i] * ds[i] / s4;
            let q0_tl_ls_ls_ls =
                d4s[i] / s2 - 8.0 * ds[i] * d3s[i] / s3 - 6.0 * d2s[i] * d2s[i] / s3
                    + 36.0 * ds[i] * ds[i] * d2s[i] / s4
                    - 24.0 * ds[i] * ds[i] * ds[i] * ds[i] / s5;
            let q0_ll_ls_ls = eta_t[i] * q0_tl_ls_ls_ls;

            let u_t_i = d_eta_t_u[i];
            let u_ls_i = d_eta_ls_u[i];
            let v_t_i = d_eta_t_v[i];
            let v_ls_i = d_eta_ls_v[i];

            let dq0_u = q0_d.q_t * u_t_i + q0_d.q_ls * u_ls_i;
            let dq0v = q0_d.q_t * v_t_i + q0_d.q_ls * v_ls_i;
            let d2q0_uv =
                q0_d.q_tl * (u_t_i * v_ls_i + v_t_i * u_ls_i) + q0_d.q_ll * u_ls_i * v_ls_i;

            let dq0_t_u = q0_d.q_tl * u_ls_i;
            let dq0_tv = q0_d.q_tl * v_ls_i;
            let dq0_ls_u = q0_d.q_tl * u_t_i + q0_d.q_ll * u_ls_i;
            let dq0_lsv = q0_d.q_tl * v_t_i + q0_d.q_ll * v_ls_i;
            let dq0_tl_u = q0_d.q_tl_ls * u_ls_i;
            let dq0_tlv = q0_d.q_tl_ls * v_ls_i;
            let dq0_ll_u = q0_d.q_tl_ls * u_t_i + q0_d.q_ll_ls * u_ls_i;
            let dq0_llv = q0_d.q_tl_ls * v_t_i + q0_d.q_ll_ls * v_ls_i;

            let d2q0_t_uv = q0_d.q_tl_ls * u_ls_i * v_ls_i;
            let d2q0_ls_uv =
                q0_d.q_tl_ls * (u_ls_i * v_t_i + v_ls_i * u_t_i) + q0_d.q_ll_ls * u_ls_i * v_ls_i;
            let d2q0_tl_uv = q0_tl_ls_ls * u_ls_i * v_ls_i;
            let d2q0_ll_uv =
                q0_tl_ls_ls * (u_t_i * v_ls_i + v_t_i * u_ls_i) + q0_ll_ls_ls * u_ls_i * v_ls_i;

            let br = b0.row(i);
            let dr = d0.row(i);
            let ddr = dd0.row(i);
            let d3r = d3_basis.row(i);
            let b_u = br.dot(&uw);
            let bv = br.dot(&vw);
            let b1_u = dr.dot(&uw);
            let b1v = dr.dot(&vw);
            let b2_u = ddr.dot(&uw);
            let b2v = ddr.dot(&vw);
            let b3_u = d3r.dot(&uw);
            let b3v = d3r.dot(&vw);

            let dm_u = b1_u + g2[i] * dq0_u;
            let dmv = b1v + g2[i] * dq0v;
            let d2m_uv = g3[i] * dq0_u * dq0v + g2[i] * d2q0_uv + b2v * dq0_u + b2_u * dq0v;
            let dg2_u = b2_u + g3[i] * dq0_u;
            let dg2v = b2v + g3[i] * dq0v;
            let d2g2_uv = g4[i] * dq0_u * dq0v + g3[i] * d2q0_uv + b3v * dq0_u + b3_u * dq0v;

            let dq_u = m[i] * dq0_u + b_u;
            let dqv = m[i] * dq0v + bv;
            let d2q_uv = m[i] * d2q0_uv + g2[i] * dq0_u * dq0v + b1_u * dq0v + b1v * dq0_u;

            let q_t = m[i] * q0_d.q_t;
            let q_ls = m[i] * q0_d.q_ls;
            let q_tt = g2[i] * q0_d.q_t * q0_d.q_t;
            let q_tl = g2[i] * q0_d.q_t * q0_d.q_ls + m[i] * q0_d.q_tl;
            let q_ll = g2[i] * q0_d.q_ls * q0_d.q_ls + m[i] * q0_d.q_ll;

            let dq_t_u = dm_u * q0_d.q_t + m[i] * dq0_t_u;
            let dq_tv = dmv * q0_d.q_t + m[i] * dq0_tv;
            let dq_ls_u = dm_u * q0_d.q_ls + m[i] * dq0_ls_u;
            let dq_lsv = dmv * q0_d.q_ls + m[i] * dq0_lsv;

            let d2q_t_uv = d2m_uv * q0_d.q_t + dm_u * dq0_tv + dmv * dq0_t_u + m[i] * d2q0_t_uv;
            let d2q_ls_uv =
                d2m_uv * q0_d.q_ls + dm_u * dq0_lsv + dmv * dq0_ls_u + m[i] * d2q0_ls_uv;

            let dq_tt_u = dg2_u * q0_d.q_t * q0_d.q_t + g2[i] * (2.0 * q0_d.q_t * dq0_t_u);
            let dq_ttv = dg2v * q0_d.q_t * q0_d.q_t + g2[i] * (2.0 * q0_d.q_t * dq0_tv);
            let d2q_tt_uv = d2g2_uv * q0_d.q_t * q0_d.q_t
                + dg2_u * (2.0 * q0_d.q_t * dq0_tv)
                + dg2v * (2.0 * q0_d.q_t * dq0_t_u)
                + g2[i] * (2.0 * dq0_t_u * dq0_tv + 2.0 * q0_d.q_t * d2q0_t_uv);

            let dq_tl_u = dg2_u * q0_d.q_t * q0_d.q_ls
                + g2[i] * (dq0_t_u * q0_d.q_ls + q0_d.q_t * dq0_ls_u)
                + dm_u * q0_d.q_tl
                + m[i] * dq0_tl_u;
            let dq_tlv = dg2v * q0_d.q_t * q0_d.q_ls
                + g2[i] * (dq0_tv * q0_d.q_ls + q0_d.q_t * dq0_lsv)
                + dmv * q0_d.q_tl
                + m[i] * dq0_tlv;
            let d2q_tl_uv = d2g2_uv * q0_d.q_t * q0_d.q_ls
                + dg2_u * (dq0_tv * q0_d.q_ls + q0_d.q_t * dq0_lsv)
                + dg2v * (dq0_t_u * q0_d.q_ls + q0_d.q_t * dq0_ls_u)
                + g2[i]
                    * (d2q0_t_uv * q0_d.q_ls
                        + dq0_t_u * dq0_lsv
                        + dq0_tv * dq0_ls_u
                        + q0_d.q_t * d2q0_ls_uv)
                + d2m_uv * q0_d.q_tl
                + dm_u * dq0_tlv
                + dmv * dq0_tl_u
                + m[i] * d2q0_tl_uv;

            let dq_ll_u = dg2_u * q0_d.q_ls * q0_d.q_ls
                + g2[i] * (2.0 * q0_d.q_ls * dq0_ls_u)
                + dm_u * q0_d.q_ll
                + m[i] * dq0_ll_u;
            let dq_llv = dg2v * q0_d.q_ls * q0_d.q_ls
                + g2[i] * (2.0 * q0_d.q_ls * dq0_lsv)
                + dmv * q0_d.q_ll
                + m[i] * dq0_llv;
            let d2q_ll_uv = d2g2_uv * q0_d.q_ls * q0_d.q_ls
                + dg2_u * (2.0 * q0_d.q_ls * dq0_lsv)
                + dg2v * (2.0 * q0_d.q_ls * dq0_ls_u)
                + g2[i] * (2.0 * dq0_ls_u * dq0_lsv + 2.0 * q0_d.q_ls * d2q0_ls_uv)
                + d2m_uv * q0_d.q_ll
                + dm_u * dq0_llv
                + dmv * dq0_ll_u
                + m[i] * d2q0_ll_uv;

            // Scalar pair coefficients on (X_t, X_t), (X_t, X_ls), (X_ls, X_ls).
            coeff_tt[i] = second_directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_t, q_t, q_tt, dq_t_u, dq_tv, dq_t_u, dq_tv,
                d2q_t_uv, d2q_t_uv, dq_tt_u, dq_ttv, d2q_tt_uv,
            );
            coeff_tl[i] = second_directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_t, q_ls, q_tl, dq_t_u, dq_tv, dq_ls_u, dq_lsv,
                d2q_t_uv, d2q_ls_uv, dq_tl_u, dq_tlv, d2q_tl_uv,
            );
            coeff_ll[i] = second_directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_ls, q_ls, q_ll, dq_ls_u, dq_lsv, dq_ls_u,
                dq_lsv, d2q_ls_uv, d2q_ls_uv, dq_ll_u, dq_llv, d2q_ll_uv,
            );

            // Cross block (X_a, B/B'/B''/B''') with X_a ∈ {X_t, X_ls}. Each
            // `coeff_xw(i, j)` is linear in (br[j], dr[j], ddr[j], d3r[j])
            // because each q-Hessian variable carrying `j` (q_xw, dq_xw_u,
            // dq_xwv, d2q_xw_uv, qw, dqw_u, dqwv, d2qw_uv) is linear in those
            // four. We expand `second_directionalhessian_coeff_fromobjective_q_terms`
            // by collecting like basis-derivative powers; the coefficients are
            // the four α_xw_{b,d,dd,d3} arrays.
            //
            // qw=br, dqw_u=dr·dq0u, dqwv=dr·dq0v, d2qw_uv=ddr·dq0u·dq0v + dr·d2q0_uv
            // q_xw=dr·q0_x, dq_xw_u=ddr·dq0u·q0_x + dr·dq0_x_u, dq_xwv=ddr·dq0v·q0_x + dr·dq0_xv
            // d2q_xw_uv=d3r·dq0u·dq0v·q0_x + ddr·(d2q0_uv·q0_x + dq0u·dq0_xv + dq0v·dq0_x_u) + dr·d2q0_x_uv
            //
            // d_qaqb_u = dq_x_u·qw + q_x·dqw_u  →  dq_x_u·br + q_x·dr·dq0u
            // d_qaqbv  = dq_xv·qw + q_x·dqwv    →  dq_xv·br + q_x·dr·dq0v
            // d2_qaqb_uv = d2q_x_uv·br + dq_x_u·dr·dq0v + dq_xv·dr·dq0u + q_x·d2qw_uv
            //
            // The full formula (expanded for "tw"; "lw" identical with x→ls):
            //
            //   m4·dq_u·dqv·q_x·br
            // + m3·(d2q_uv·q_x·br + dq_u·(dq_xv·br + q_x·dr·dq0v) + dqv·(dq_x_u·br + q_x·dr·dq0u) + dq_u·dqv·dr·q0_x)
            // + m2·(d2q_x_uv·br + dq_x_u·dr·dq0v + dq_xv·dr·dq0u + q_x·(ddr·dq0u·dq0v + dr·d2q0_uv)
            //       + d2q_uv·dr·q0_x + dq_u·(ddr·dq0v·q0_x + dr·dq0_xv) + dqv·(ddr·dq0u·q0_x + dr·dq0_x_u))
            // + m1·(d3r·dq0u·dq0v·q0_x + ddr·(d2q0_uv·q0_x + dq0u·dq0_xv + dq0v·dq0_x_u) + dr·d2q0_x_uv)
            //
            // Collecting like basis-derivative terms produces the closed-form
            // expressions below.

            // X_t ↔ wiggle channels.
            alpha_tw_b[i] = m4 * dq_u * dqv * q_t
                + m3 * (d2q_uv * q_t + dq_u * dq_tv + dqv * dq_t_u)
                + m2 * d2q_t_uv;
            alpha_tw_d[i] = m3 * (dq_u * q_t * dq0v + dqv * q_t * dq0_u + dq_u * dqv * q0_d.q_t)
                + m2 * (dq_t_u * dq0v
                    + dq_tv * dq0_u
                    + q_t * d2q0_uv
                    + d2q_uv * q0_d.q_t
                    + dq_u * dq0_tv
                    + dqv * dq0_t_u)
                + m1 * d2q0_t_uv;
            alpha_tw_dd[i] = m2
                * (q_t * dq0_u * dq0v + dq_u * dq0v * q0_d.q_t + dqv * dq0_u * q0_d.q_t)
                + m1 * (d2q0_uv * q0_d.q_t + dq0_u * dq0_tv + dq0v * dq0_t_u);
            alpha_tw_d3[i] = m1 * dq0_u * dq0v * q0_d.q_t;

            // X_ls ↔ wiggle channels (same formulas, swap t→ls).
            alpha_lw_b[i] = m4 * dq_u * dqv * q_ls
                + m3 * (d2q_uv * q_ls + dq_u * dq_lsv + dqv * dq_ls_u)
                + m2 * d2q_ls_uv;
            alpha_lw_d[i] = m3 * (dq_u * q_ls * dq0v + dqv * q_ls * dq0_u + dq_u * dqv * q0_d.q_ls)
                + m2 * (dq_ls_u * dq0v
                    + dq_lsv * dq0_u
                    + q_ls * d2q0_uv
                    + d2q_uv * q0_d.q_ls
                    + dq_u * dq0_lsv
                    + dqv * dq0_ls_u)
                + m1 * d2q0_ls_uv;
            alpha_lw_dd[i] = m2
                * (q_ls * dq0_u * dq0v + dq_u * dq0v * q0_d.q_ls + dqv * dq0_u * q0_d.q_ls)
                + m1 * (d2q0_uv * q0_d.q_ls + dq0_u * dq0_lsv + dq0v * dq0_ls_u);
            alpha_lw_d3[i] = m1 * dq0_u * dq0v * q0_d.q_ls;

            // Wiggle ↔ wiggle (bilinear in (br, dr, ddr) ⊗ (br, dr, ddr); no d3r).
            //
            // qa=brj, qb=brk, qab=0, dqa_u=drj·dq0u, dqav=drj·dq0v,
            // dqb_u=drk·dq0u, dqbv=drk·dq0v, d2qa_uv=ddrj·dq0u·dq0v+drj·d2q0_uv,
            // d2qb_uv=ddrk·dq0u·dq0v+drk·d2q0_uv, dqab_u=0, dqabv=0, d2qab_uv=0.
            //
            //   m4·dq_u·dqv·brj·brk
            // + m3·(d2q_uv·brj·brk + dq_u·(drj·dq0v·brk + brj·drk·dq0v)
            //                       + dqv·(drj·dq0u·brk + brj·drk·dq0u))
            // + m2·d2_qaqb_uv
            // where d2_qaqb_uv = (ddrj·dq0u·dq0v+drj·d2q0_uv)·brk
            //                  + drj·dq0u·drk·dq0v + drj·dq0v·drk·dq0u
            //                  + brj·(ddrk·dq0u·dq0v+drk·d2q0_uv).
            //
            // Pair (B, B): m4·dq_u·dqv + m3·d2q_uv  → coefficient of br[j]·br[k].
            // Pair (B, B'): m3·(dq_u·dq0v + dqv·dq0u) + m2·d2q0_uv → br·dr + dr·br.
            // Pair (B, B''): m2·dq0u·dq0v → br·ddr + ddr·br.
            // Pair (B', B'): 2·m2·dq0u·dq0v → dr·dr (the diagonal pair only
            //   accumulates once in `RowCoeffOperator`, so we double-count
            //   here to match the symmetric `dr[j]·dq0u·dr[k]·dq0v +
            //   dr[j]·dq0v·dr[k]·dq0u` cross product).
            c_ww_bb[i] = m4 * dq_u * dqv + m3 * d2q_uv;
            c_ww_bd[i] = m3 * (dq_u * dq0v + dqv * dq0_u) + m2 * d2q0_uv;
            c_ww_bdd[i] = m2 * dq0_u * dq0v;
            c_ww_dd_pair[i] = 2.0 * m2 * dq0_u * dq0v;
        }

        let basis: Arc<Array2<f64>> = Arc::new(b0);
        let basis_d1: Arc<Array2<f64>> = Arc::new(d0);
        let basis_d2: Arc<Array2<f64>> = Arc::new(dd0);
        let basis_d3: Arc<Array2<f64>> = Arc::new(d3_basis);

        Ok(Some(Arc::new(RowCoeffOperator::from_directions(
            vec![pt, pls, pw],
            vec![
                (0, x_t_arc),
                (1, x_ls_arc),
                (2, basis),
                (2, basis_d1),
                (2, basis_d2),
                (2, basis_d3),
            ],
            vec![
                // (X_t, X_t)   ← `d2_h[a, b] += coeff_tt · xtr[a] · xtr[b]`
                (0, 0, coeff_tt),
                // (X_t, X_ls)  ← `d2_h[a, pt+b] += coeff_tl · xtr[a] · xlsr[b]`
                (0, 1, coeff_tl),
                // (X_ls, X_ls) ← `d2_h[pt+a, pt+b] += coeff_ll · xlsr[a] · xlsr[b]`
                (1, 1, coeff_ll),
                // (X_t, B/B'/B''/B''') ← per-row α_tw_{b,d,dd,d3} decomposition of
                // `d2_h[a, pt+pls+j] += coeff_tw(i,j) · xtr[a]` (coeff_tw is
                // linear in br[j], dr[j], ddr[j], d3r[j])
                (0, 2, alpha_tw_b),
                (0, 3, alpha_tw_d),
                (0, 4, alpha_tw_dd),
                (0, 5, alpha_tw_d3),
                // (X_ls, B/B'/B''/B''') ← analogous α_lw_{b,d,dd,d3} decomposition
                // of `d2_h[pt+a, pt+pls+j] += coeff_lw(i,j) · xlsr[a]`
                (1, 2, alpha_lw_b),
                (1, 3, alpha_lw_d),
                (1, 4, alpha_lw_dd),
                (1, 5, alpha_lw_d3),
                // (B, B/B'/B'') ← bilinear decomposition of
                // `d2_h[pt+pls+j, pt+pls+k] += coeff_ww(i,j,k)` in
                // (br, dr, ddr) ⊗ (br, dr, ddr); no d3r entry — coeff_ww is
                // at most degree 2 in any single basis derivative.
                (2, 2, c_ww_bb),
                (2, 3, c_ww_bd),
                (2, 4, c_ww_bdd),
                // (B', B') diagonal — coefficient absorbs a factor of 2 to
                // match the symmetric `dr[j]·dq0u·dr[k]·dq0v + dr[j]·dq0v·
                // dr[k]·dq0u` cross product (the diagonal pair only
                // accumulates once in `RowCoeffOperator::mul_vec`).
                (3, 3, c_ww_dd_pair),
            ],
            n,
        ))))
    }
}


/// Matrix-free joint-Hessian operator for the 3-block binomial
/// location-scale wiggle family. See `BinomialLocationScaleWiggleHessianRowPieces`
/// for the per-row weight structure.
struct BinomialLocationScaleWiggleHessianWorkspace {
    family: BinomialLocationScaleWiggleFamily,
    block_states: Vec<ParameterBlockState>,
    x_t: Arc<Array2<f64>>,
    x_ls: Arc<Array2<f64>>,
    pieces: BinomialLocationScaleWiggleHessianRowPieces,
}



impl BinomialLocationScaleWiggleHessianWorkspace {
    fn new(
        family: BinomialLocationScaleWiggleFamily,
        block_states: Vec<ParameterBlockState>,
        x_t: Array2<f64>,
        x_ls: Array2<f64>,
    ) -> Result<Self, String> {
        let pieces = family.wiggle_hessian_row_pieces(&block_states)?;
        Ok(Self {
            family,
            block_states,
            x_t: Arc::new(x_t),
            x_ls: Arc::new(x_ls),
            pieces,
        })
    }

    /// Apply a Horvitz–Thompson outer-row subsample mask to the precomputed
    /// per-row coefficient arrays in place.
    ///
    /// Each sampled row's `coeff_*[i]` is multiplied by its
    /// `WeightedOuterRow.weight` (the HT inverse-inclusion factor 1/π_i —
    /// uniform or stratified sampling both supported). All non-sampled rows
    /// are zeroed. Because every downstream assembly (`hessian_dense`,
    /// `hessian_matvec`, `hessian_diagonal`) is row-linear in these arrays
    /// via `Xᵀ diag(W) Y`, the resulting joint-Hessian is an unbiased
    /// estimator of the full-data joint Hessian. The `b0`/`d0` basis matrices
    /// are independent of the per-row weights and remain unchanged.
    fn apply_outer_subsample(
        &mut self,
        rows: &[crate::families::marginal_slope_shared::WeightedOuterRow],
    ) {
        let n = self.pieces.coeff_tt.len();
        let mut mask_tt = Array1::<f64>::zeros(n);
        let mut mask_tl = Array1::<f64>::zeros(n);
        let mut mask_ll = Array1::<f64>::zeros(n);
        let mut mask_tw_b = Array1::<f64>::zeros(n);
        let mut mask_tw_d = Array1::<f64>::zeros(n);
        let mut mask_lw_b = Array1::<f64>::zeros(n);
        let mut mask_lw_d = Array1::<f64>::zeros(n);
        let mut maskww = Array1::<f64>::zeros(n);
        for r in rows {
            let i = r.index;
            let w = r.weight;
            mask_tt[i] = self.pieces.coeff_tt[i] * w;
            mask_tl[i] = self.pieces.coeff_tl[i] * w;
            mask_ll[i] = self.pieces.coeff_ll[i] * w;
            mask_tw_b[i] = self.pieces.coeff_tw_b[i] * w;
            mask_tw_d[i] = self.pieces.coeff_tw_d[i] * w;
            mask_lw_b[i] = self.pieces.coeff_lw_b[i] * w;
            mask_lw_d[i] = self.pieces.coeff_lw_d[i] * w;
            maskww[i] = self.pieces.coeffww[i] * w;
        }
        self.pieces.coeff_tt = mask_tt;
        self.pieces.coeff_tl = mask_tl;
        self.pieces.coeff_ll = mask_ll;
        self.pieces.coeff_tw_b = mask_tw_b;
        self.pieces.coeff_tw_d = mask_tw_d;
        self.pieces.coeff_lw_b = mask_lw_b;
        self.pieces.coeff_lw_d = mask_lw_d;
        self.pieces.coeffww = maskww;
    }
}


impl ExactNewtonJointHessianWorkspace for BinomialLocationScaleWiggleHessianWorkspace {
    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        // Same Hv structure as `hessian_matvec`, but routed through the
        // already-existing `assemble_dense` row-pieces helper (eight GEMMs
        // covering h_tt, h_tl, h_ll, h_tw_b, h_tw_d, h_lw_b, h_lw_d, h_ww).
        // Avoids `total` canonical-basis HVPs in
        // `MatrixFreeSpdOperator::materialize_dense_operator`, which at
        // large scale (n≈320k, p_total≈82) costs ~568s per κ-iter versus
        // ~1s for the dense build.
        let dense = self
            .pieces
            .assemble_dense(self.x_t.as_ref(), self.x_ls.as_ref())?;
        Ok(Some(dense))
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let pt = self.x_t.ncols();
        let pls = self.x_ls.ncols();
        let pw = self.pieces.b0.ncols();
        let total = pt + pls + pw;
        if v.len() != total {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleWiggle matvec dimension mismatch: got {}, expected {}",
                    v.len(),
                    total
                ),
            }
            .into());
        }
        let v_t = v.slice(s![0..pt]);
        let v_ls = v.slice(s![pt..pt + pls]);
        let v_w = v.slice(s![pt + pls..total]);

        let u_t = self.x_t.dot(&v_t);
        let u_ls = self.x_ls.dot(&v_ls);
        let u_b = self.pieces.b0.dot(&v_w);
        let u_d = self.pieces.d0.dot(&v_w);

        let r_t = &self.pieces.coeff_tt * &u_t
            + &self.pieces.coeff_tl * &u_ls
            + &self.pieces.coeff_tw_b * &u_b
            + &self.pieces.coeff_tw_d * &u_d;
        let r_ls = &self.pieces.coeff_tl * &u_t
            + &self.pieces.coeff_ll * &u_ls
            + &self.pieces.coeff_lw_b * &u_b
            + &self.pieces.coeff_lw_d * &u_d;
        let r_b = &self.pieces.coeff_tw_b * &u_t
            + &self.pieces.coeff_lw_b * &u_ls
            + &self.pieces.coeffww * &u_b;
        let r_d = &self.pieces.coeff_tw_d * &u_t + &self.pieces.coeff_lw_d * &u_ls;

        let out_t = fast_atv(self.x_t.as_ref(), &r_t);
        let out_ls = fast_atv(self.x_ls.as_ref(), &r_ls);
        let out_w = fast_atv(&self.pieces.b0, &r_b) + &fast_atv(&self.pieces.d0, &r_d);

        let mut out = Array1::<f64>::zeros(total);
        out.slice_mut(s![0..pt]).assign(&out_t);
        out.slice_mut(s![pt..pt + pls]).assign(&out_ls);
        out.slice_mut(s![pt + pls..total]).assign(&out_w);
        Ok(Some(out))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        let pt = self.x_t.ncols();
        let pls = self.x_ls.ncols();
        let pw = self.pieces.b0.ncols();
        let total = pt + pls + pw;
        let mut diag = Array1::<f64>::zeros(total);
        let n = self.pieces.coeff_tt.len();
        for j in 0..pt {
            let col = self.x_t.column(j);
            let mut acc = 0.0;
            for i in 0..n {
                let v = col[i];
                acc += self.pieces.coeff_tt[i] * v * v;
            }
            diag[j] = acc;
        }
        for j in 0..pls {
            let col = self.x_ls.column(j);
            let mut acc = 0.0;
            for i in 0..n {
                let v = col[i];
                acc += self.pieces.coeff_ll[i] * v * v;
            }
            diag[pt + j] = acc;
        }
        for j in 0..pw {
            let col = self.pieces.b0.column(j);
            let mut acc = 0.0;
            for i in 0..n {
                let v = col[i];
                acc += self.pieces.coeffww[i] * v * v;
            }
            diag[pt + pls + j] = acc;
        }
        Ok(Some(diag))
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative(&self.block_states, d_beta_flat)
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        self.family.bls_wiggle_directional_operator(
            &self.block_states,
            self.x_t.clone(),
            self.x_ls.clone(),
            d_beta_flat,
        )
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &self.block_states,
                d_beta_u_flat,
                d_beta_v_flat,
            )
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn crate::solver::estimate::reml::unified::HyperOperator>>, String>
    {
        self.family.bls_wiggle_second_directional_operator(
            &self.block_states,
            self.x_t.clone(),
            self.x_ls.clone(),
            d_beta_u,
            d_beta_v,
        )
    }
}


impl CustomFamilyGenerative for BinomialLocationScaleWiggleFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String> {
        if block_states.len() != 3 {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "BinomialLocationScaleWiggleFamily expects 3 blocks, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = &block_states[Self::BLOCK_WIGGLE].eta;
        if eta_t.len() != self.y.len() || eta_ls.len() != self.y.len() || etaw.len() != self.y.len()
        {
            return Err(GamlssError::DimensionMismatch {
                reason: "BinomialLocationScaleWiggleFamily generative size mismatch".to_string(),
            }
            .into());
        }
        let mean = gamlss_rowwise_map_result(self.y.len(), |i| {
            let sigma = exp_sigma_from_eta_scalar(eta_ls[i]);
            let q0 = binomial_location_scale_q0(eta_t[i], sigma);
            let jet = inverse_link_jet_for_inverse_link(&self.link_kind, q0 + etaw[i])
                .map_err(|e| format!("location-scale inverse-link evaluation failed: {e}"))?;
            Ok(jet.mu)
        })?;
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Bernoulli,
        })
    }
}
