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
pub(crate) fn bernoulli_log_likelihood_from_probability(
    y: f64,
    weight: f64,
    mu: f64,
) -> Result<f64, String> {
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
    pub(crate) struct SendPtr(*mut f64);
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

