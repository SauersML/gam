// Real concern-organized submodule of the gamlss family stack.
// Cross-module items are re-exported flat through the parent (`gamlss.rs`),
// so `use super::*;` makes the sibling-concern symbols this module references
// resolve through the parent namespace.
use super::*;

use gam_math::jet_scalar::{OneSeedBatch, TwoSeedBatch};
use wide::f64x4;

pub(crate) struct BinomialLocationScaleCore {
    pub(crate) sigma: Array1<f64>,
    pub(crate) dsigma_deta: Array1<f64>,
    pub(crate) q0: Array1<f64>,
    pub(crate) mu: Array1<f64>,
    pub(crate) dmu_dq: Array1<f64>,
    pub(crate) d2mu_dq2: Array1<f64>,
    pub(crate) d3mu_dq3: Array1<f64>,
    pub(crate) log_likelihood: f64,
}

#[derive(Clone, Copy)]
pub(crate) struct NonWiggleQDerivs {
    pub(crate) q_t: f64,
    pub(crate) q_ls: f64,
    pub(crate) q_tl: f64,
    pub(crate) q_ll: f64,
    pub(crate) q_tl_ls: f64,
    pub(crate) q_ll_ls: f64,
}

#[derive(Clone, Copy)]
pub(crate) struct NonWiggleQDirectional {
    pub(crate) delta_q: f64,
    pub(crate) delta_q_t: f64,
    pub(crate) delta_q_ls: f64,
    pub(crate) delta_q_tl: f64,
    pub(crate) delta_q_ll: f64,
}

#[derive(Clone, Copy)]
pub(crate) struct BinomialLocationScaleRow {
    pub(crate) sigma: f64,
    pub(crate) dsigma_deta: f64,
    pub(crate) q0: f64,
    pub(crate) inverse_link: gam_solve::mixture_link::InverseLinkJet,
    pub(crate) ll: f64,
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
            * (-y * gam_linalg::utils::stable_softplus(-q)
                - (1.0_f64 - y) * gam_linalg::utils::stable_softplus(q))),
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
    Ok(
        gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            n,
            |range| -> Result<f64, String> {
                let mut acc = 0.0_f64;
                for i in range {
                    let SigmaJet1 { sigma, .. } = exp_sigma_jet1_scalar(el_slice[i]);
                    let q0 = binomial_location_scale_q0(et_slice[i], sigma);
                    let q = q0 + ew_slice.map_or(0.0, |w| w[i]);
                    if matches!(link_kind, InverseLink::Standard(StandardLink::Probit)) {
                        acc += binomial_location_scale_log_likelihood(
                            y_slice[i], w_slice[i], q, link_kind, 0.5,
                        )?;
                        continue;
                    }
                    let jet = inverse_link_jet_for_inverse_link(link_kind, q).map_err(|e| {
                        format!("location-scale inverse-link evaluation failed: {e}")
                    })?;
                    acc += binomial_location_scale_log_likelihood(
                        y_slice[i], w_slice[i], q, link_kind, jet.mu,
                    )?;
                }
                Ok(acc)
            },
            |a, b| Ok(a + b),
        )?
        .unwrap_or(0.0),
    )
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

    let ll = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
        n,
        move |range| -> Result<f64, String> {
            let mut acc = 0.0_f64;
            for i in range {
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
                acc += row.ll;
            }
            Ok(acc)
        },
        |a, b| Ok(a + b),
    )?
    .unwrap_or(0.0);

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

/// The binomial location-scale row NLL written ONCE over a generic
/// [`JetScalar<2>`] (#932). The map `q = −η_t·exp(−η_ls)` is built in the
/// scalar algebra `S` (the primaries seeded by the caller via `seed`), then the
/// per-row negative log-likelihood follows from the hand-certified `q`-space
/// derivative stack `[−ll, m1, m2, m3, m4]` through one
/// [`JetScalar::compose_unary`].
///
/// Instantiating `S` selects the channel a consumer needs without ever
/// materialising the dense `Tower4<2>` `t3`/`t4`:
/// * `S = Order2<2>` → value/grad/Hessian (the joint-Hessian path),
/// * `S = OneSeed<2>` → the contracted third `Σ_c ℓ_{abc} dir_c`,
/// * `S = TwoSeed<2>` → the contracted fourth `Σ_{cd} ℓ_{abcd} u_c v_d`.
///
/// `seed(value, axis)` produces the primary jet for axis `axis` (0 = η_t,
/// 1 = η_ls); the directional scalars fold their contraction direction in
/// through this closure (mirrors `survival::location_scale::sls_row_nll`).
///
/// `need_value` gates the `q`-space VALUE channel `−ll`. The composed value
/// `d[0] = −ll` flows ONLY into the result's value channel: the gradient and
/// Hessian read `d[1..=2]`, the contracted third reads the ε-Hessian, the
/// contracted fourth reads the εδ-Hessian — NONE touch `d[0]`. So the
/// directional consumers (`OneSeed`/`TwoSeed`), which discard the value
/// channel, are byte-for-byte invariant under `d[0]`; they pass
/// `need_value = false` and skip computing `ll` altogether. That elides one
/// per-row log-likelihood — whose Probit/Logit/CLogLog branches each evaluate
/// an `erfc`/`softplus`/`log1mexp`-class special function — that the `?` error
/// propagation otherwise keeps alive (blocking the compiler from dead-code
/// eliminating it). This is EXACT, not an approximation: the contracted output
/// tensors are unchanged. Value-channel consumers (`Order2` joint-Hessian, the
/// dense `Tower4` oracle / gradient path) pass `true` and get the exact `−ll`.
#[inline]
pub(crate) fn binomial_location_scale_nll_generic<S: gam_math::jet_scalar::JetScalar<2>>(
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
    need_value: bool,
    seed: impl Fn(f64, usize) -> S,
) -> Result<S, String> {
    let eta_t_jet = seed(eta_t, 0);
    let eta_ls_jet = seed(eta_ls, 1);
    let inv_sigma = eta_ls_jet.scale(-1.0).exp();
    let q = eta_t_jet.neg().mul(&inv_sigma);
    // The value channel `−ll` is dead for every derivative/contraction channel;
    // only pay the special-function-bearing log-likelihood when a value-channel
    // consumer actually reads it (proven bit-identical for the contracted
    // paths: their output is independent of `d[0]`).
    let neg_ll = if need_value {
        -binomial_location_scale_log_likelihood(y, weight, q_value, link_kind, mu)?
    } else {
        0.0
    };
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
    Ok(q.compose_unary([neg_ll, m1, m2, m3, m4]))
}

/// Gradient-only instantiation of the single-source location-scale row NLL.
/// `Order1<2>` deletes the unread Hessian/t3/t4 state, and `need_value=false`
/// also skips the special-function-bearing log-likelihood value already
/// supplied by the row core.
#[inline]
pub(crate) fn binomial_location_scale_nll_gradient(
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
) -> Result<[f64; 2], String> {
    use gam_math::jet_scalar::{JetScalar, Order1};

    let out = binomial_location_scale_nll_generic::<Order1<2>>(
        y,
        weight,
        eta_t,
        eta_ls,
        q_value,
        mu,
        dmu_dq,
        d2mu_dq2,
        d3mu_dq3,
        link_kind,
        false,
        false,
        |x, axis| Order1::<2>::variable(x, axis),
    )?;
    let (_, gradient) = out.into_channels();
    Ok(gradient)
}

/// SIMD 4-rows-per-pass evaluation of the binomial location-scale row NLL at the
/// packed [`OneSeedBatch`] directional scalar, returning the contracted-third
/// tensor `Σ_c ℓ_{xyc} dir_c` for FOUR rows at once. The op graph mirrors
/// [`binomial_location_scale_nll_generic`] term-for-term over `OneSeedBatch<2>`
/// (`q = −η_t·e^{−η_ls}`, then `q.compose_unary([0, m1, m2, m3, 0])`).
///
/// Unlike the survival-LS row NLL, the binomial row NLL composes its q-space
/// derivative stack UNCONDITIONALLY — there is no per-term `0·∞` gating — so
/// every row is homogeneous and no signature grouping is needed: the same op
/// graph applies to all four lanes. By the jet engine's lane identity
/// (`OneSeedBatch` lane `i` `to_bits`== `OneSeed` row `i`; the transcendental
/// `exp` and the rational tensor composition are evaluated per lane through the
/// identical scalar code), lane `i` of the returned tensor is
/// `to_bits`-identical to the scalar
/// `binomial_location_scale_nll_generic::<OneSeed<2>>(..).contracted_third()`.
/// The per-row `(m1, m2, m3)` q-derivative stack is the SAME
/// [`binomial_neglog_q_derivatives_dispatch`] scalar the per-row path computed,
/// packed lane-wise.
#[inline]
fn binomial_ls_directional_third_batch(
    eta_t: f64x4,
    eta_ls: f64x4,
    dir0: f64x4,
    dir1: f64x4,
    m1: f64x4,
    m2: f64x4,
    m3: f64x4,
) -> [[f64x4; 2]; 2] {
    let eta_t_jet = OneSeedBatch::<2>::seed_direction(eta_t, 0, dir0);
    let eta_ls_jet = OneSeedBatch::<2>::seed_direction(eta_ls, 1, dir1);
    let inv_sigma = eta_ls_jet.scale(-1.0).exp();
    let q = eta_t_jet.neg().mul(&inv_sigma);
    let zero = f64x4::splat(0.0);
    q.compose_unary([zero, m1, m2, m3, zero]).contracted_third()
}

/// SIMD 4-rows-per-pass evaluation of the binomial location-scale row NLL at the
/// packed [`TwoSeedBatch`] bidirectional scalar, returning the contracted-fourth
/// tensor `Σ_cd ℓ_{xycd} u_c v_d` for FOUR rows at once. Mirrors
/// [`binomial_ls_directional_third_batch`] but with the fourth q-derivative `m4`
/// (so `q.compose_unary([0, m1, m2, m3, m4])`). Same lane-identity bit-identity
/// to the scalar `..::<TwoSeed<2>>(..).contracted_fourth()`.
#[inline]
fn binomial_ls_directional_fourth_batch(
    eta_t: f64x4,
    eta_ls: f64x4,
    du0: f64x4,
    du1: f64x4,
    dv0: f64x4,
    dv1: f64x4,
    m1: f64x4,
    m2: f64x4,
    m3: f64x4,
    m4: f64x4,
) -> [[f64x4; 2]; 2] {
    let eta_t_jet = TwoSeedBatch::<2>::seed(eta_t, 0, du0, dv0);
    let eta_ls_jet = TwoSeedBatch::<2>::seed(eta_ls, 1, du1, dv1);
    let inv_sigma = eta_ls_jet.scale(-1.0).exp();
    let q = eta_t_jet.neg().mul(&inv_sigma);
    let zero = f64x4::splat(0.0);
    q.compose_unary([zero, m1, m2, m3, m4]).contracted_fourth()
}

/// Reconstruct the per-row `(eta_t, eta_ls)` the row NLL seeds from the core's
/// `(σ, q0)` — identical to
/// `binomial_location_scale_nll_generic_from_core_row`'s reconstruction, so
/// the lane-batched path seeds the SAME primary values as the scalar path.
#[inline]
fn core_row_eta(core: &BinomialLocationScaleCore, row: usize) -> (f64, f64) {
    let sigma = core.sigma[row];
    (-core.q0[row] * sigma, sigma.ln())
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
///
/// Evaluated 4 rows per SIMD pass through [`binomial_ls_directional_third_batch`]
/// (bit-identical to the prior scalar per-row `OneSeed<2>` contraction; see
/// `simd_directional_coefficients_match_scalar_per_row_to_bits`).
pub(crate) fn binomial_location_scale_first_directional_coefficients(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    core: &BinomialLocationScaleCore,
    d_eta_t: &Array1<f64>,
    d_eta_ls: &Array1<f64>,
    link_kind: &InverseLink,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    let n = y.len();
    let nchunks = n.div_ceil(4);
    // SIMD 4-rows-per-pass over the contracted-third directional contraction.
    // Each chunk's trailing lanes (when `n % 4 != 0`) are padded with the
    // chunk's first row — a valid, finite row whose result is simply discarded —
    // so the unconditional binomial compose never sees a non-finite lane.
    let chunked: Result<Vec<Vec<(f64, f64, f64)>>, String> = (0..nchunks)
        .into_par_iter()
        .map(|chunk| -> Result<Vec<(f64, f64, f64)>, String> {
            let start = chunk * 4;
            let cnt = (n - start).min(4);
            let rows: [usize; 4] = std::array::from_fn(|l| start + if l < cnt { l } else { 0 });
            let eta: [(f64, f64); 4] = std::array::from_fn(|l| core_row_eta(core, rows[l]));
            let eta_t = f64x4::new(std::array::from_fn(|l| eta[l].0));
            let eta_ls = f64x4::new(std::array::from_fn(|l| eta[l].1));
            let dir0 = f64x4::new(std::array::from_fn(|l| d_eta_t[rows[l]]));
            let dir1 = f64x4::new(std::array::from_fn(|l| d_eta_ls[rows[l]]));
            // Per-lane q-derivative stack: the SAME scalar dispatch the per-row
            // path ran, packed lane-wise.
            let mut m1a = [0.0_f64; 4];
            let mut m2a = [0.0_f64; 4];
            let mut m3a = [0.0_f64; 4];
            for l in 0..4 {
                let r = rows[l];
                let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
                    y[r],
                    weights[r],
                    core.q0[r],
                    core.mu[r],
                    core.dmu_dq[r],
                    core.d2mu_dq2[r],
                    core.d3mu_dq3[r],
                    link_kind,
                );
                m1a[l] = m1;
                m2a[l] = m2;
                m3a[l] = m3;
            }
            let third = binomial_ls_directional_third_batch(
                eta_t,
                eta_ls,
                dir0,
                dir1,
                f64x4::new(m1a),
                f64x4::new(m2a),
                f64x4::new(m3a),
            );
            let tt = third[0][0].to_array();
            let tl = third[0][1].to_array();
            let ll = third[1][1].to_array();
            Ok((0..cnt).map(|l| (tt[l], tl[l], ll[l])).collect())
        })
        .collect();
    let chunked = chunked?;
    let mut coeff_tt = Array1::<f64>::zeros(n);
    let mut coeff_tl = Array1::<f64>::zeros(n);
    let mut coeff_ll = Array1::<f64>::zeros(n);
    let mut idx = 0usize;
    for chunk in chunked {
        for (tt, tl, ll) in chunk {
            coeff_tt[idx] = tt;
            coeff_tl[idx] = tl;
            coeff_ll[idx] = ll;
            idx += 1;
        }
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
    let nchunks = n.div_ceil(4);
    // SIMD 4-rows-per-pass over the contracted-fourth bidirectional contraction.
    // m4 dispatch can fail (Result); trailing lanes are padded with the chunk's
    // first row (a row already evaluated within `cnt`, so it adds no new error
    // source), and discarded.
    let chunked: Result<Vec<Vec<(f64, f64, f64)>>, String> = (0..nchunks)
        .into_par_iter()
        .map(|chunk| -> Result<Vec<(f64, f64, f64)>, String> {
            let start = chunk * 4;
            let cnt = (n - start).min(4);
            let rows: [usize; 4] = std::array::from_fn(|l| start + if l < cnt { l } else { 0 });
            let eta: [(f64, f64); 4] = std::array::from_fn(|l| core_row_eta(core, rows[l]));
            let eta_t = f64x4::new(std::array::from_fn(|l| eta[l].0));
            let eta_ls = f64x4::new(std::array::from_fn(|l| eta[l].1));
            let du0 = f64x4::new(std::array::from_fn(|l| d_eta_t_u[rows[l]]));
            let du1 = f64x4::new(std::array::from_fn(|l| d_eta_ls_u[rows[l]]));
            let dv0 = f64x4::new(std::array::from_fn(|l| d_eta_t_v[rows[l]]));
            let dv1 = f64x4::new(std::array::from_fn(|l| d_eta_ls_v[rows[l]]));
            let mut m1a = [0.0_f64; 4];
            let mut m2a = [0.0_f64; 4];
            let mut m3a = [0.0_f64; 4];
            let mut m4a = [0.0_f64; 4];
            for l in 0..4 {
                let r = rows[l];
                let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
                    y[r],
                    weights[r],
                    core.q0[r],
                    core.mu[r],
                    core.dmu_dq[r],
                    core.d2mu_dq2[r],
                    core.d3mu_dq3[r],
                    link_kind,
                );
                let m4 = binomial_neglog_q_fourth_derivative_dispatch(
                    y[r],
                    weights[r],
                    core.q0[r],
                    core.mu[r],
                    core.dmu_dq[r],
                    core.d2mu_dq2[r],
                    core.d3mu_dq3[r],
                    link_kind,
                )?;
                m1a[l] = m1;
                m2a[l] = m2;
                m3a[l] = m3;
                m4a[l] = m4;
            }
            let fourth = binomial_ls_directional_fourth_batch(
                eta_t,
                eta_ls,
                du0,
                du1,
                dv0,
                dv1,
                f64x4::new(m1a),
                f64x4::new(m2a),
                f64x4::new(m3a),
                f64x4::new(m4a),
            );
            let tt = fourth[0][0].to_array();
            let tl = fourth[0][1].to_array();
            let ll = fourth[1][1].to_array();
            Ok((0..cnt).map(|l| (tt[l], tl[l], ll[l])).collect())
        })
        .collect();
    let chunked = chunked?;
    let mut coeff_tt = Array1::<f64>::zeros(n);
    let mut coeff_tl = Array1::<f64>::zeros(n);
    let mut coeff_ll = Array1::<f64>::zeros(n);
    let mut idx = 0usize;
    for chunk in chunked {
        for (tt, tl, ll) in chunk {
            coeff_tt[idx] = tt;
            coeff_tl[idx] = tl;
            coeff_ll[idx] = ll;
            idx += 1;
        }
    }
    Ok((coeff_tt, coeff_tl, coeff_ll))
}

#[cfg(test)]
mod packed_scalar_oracle_tests {
    //! #932 oracle: the packed `Order2`/`OneSeed`/`TwoSeed` evaluations of the
    //! single-source [`binomial_location_scale_nll_generic`] must reproduce,
    //! channel-for-channel, the dense `Tower4<2>` builder
    //! ([`binomial_location_scale_nll_tower`]) the contracted/Hessian hot paths
    //! replaced — value/grad/Hessian for `Order2`, the contracted third for
    //! `OneSeed`, the contracted fourth for `TwoSeed`.
    use super::*;
    use crate::gamlss::test_support::binomial_location_scale_nll_tower;
    use gam_math::jet_scalar::{JetScalar, OneSeed, Order2, TwoSeed};
    use gam_math::nested_dual::JetField;
    use gam_problem::{InverseLink, StandardLink};

    fn rel_close(a: f64, b: f64, label: &str) {
        let band = 1e-9 + 1e-9 * a.abs().max(b.abs());
        assert!(
            (a - b).abs() <= band,
            "{label}: {a:+.15e} vs {b:+.15e} (band {band:.3e})"
        );
    }

    /// Evaluate the dense tower and the three packed scalars over a grid of
    /// (y, eta_t, eta_ls) for each link, pinning every channel a consumer reads.
    #[test]
    fn packed_scalars_match_dense_tower_all_channels() {
        let links = [
            InverseLink::Standard(StandardLink::Logit),
            InverseLink::Standard(StandardLink::Probit),
            InverseLink::Standard(StandardLink::CLogLog),
        ];
        let grid = [
            (0.0_f64, 0.4_f64, -0.3_f64),
            (1.0, -0.7, 0.5),
            (0.0, 1.2, 0.2),
            (1.0, 0.1, -0.8),
        ];
        let dir_u = [0.6_f64, -0.2_f64];
        let dir_v = [-0.4_f64, 1.1_f64];
        for link in &links {
            for &(y, eta_t, eta_ls) in &grid {
                let weight = 1.3_f64;
                let SigmaJet1 { sigma, .. } = exp_sigma_jet1_scalar(eta_ls);
                let q0 = binomial_location_scale_q0(eta_t, sigma);
                let jet = match inverse_link_jet_for_inverse_link(link, q0) {
                    Ok(j) => j,
                    Err(_) => continue,
                };
                let args = |include_fourth: bool| {
                    (
                        y,
                        weight,
                        eta_t,
                        eta_ls,
                        q0,
                        jet.mu,
                        jet.d1,
                        jet.d2,
                        jet.d3,
                        link,
                        include_fourth,
                    )
                };

                // Dense Tower4 oracle.
                let (y_, w_, et_, el_, q_, mu_, d1_, d2_, d3_, lk_, _f) = args(true);
                let tower = binomial_location_scale_nll_tower(
                    y_, w_, et_, el_, q_, mu_, d1_, d2_, d3_, lk_, true,
                )
                .expect("tower");

                let gradient = binomial_location_scale_nll_gradient(
                    y_, w_, et_, el_, q_, mu_, d1_, d2_, d3_, lk_,
                )
                .expect("order1 gradient");
                for a in 0..2 {
                    rel_close(gradient[a], tower.g[a], "order1 grad");
                }

                // Order2 (v, g, H).
                let (y2, w2, et2, el2, q2, mu2, d12, d22, d32, lk2, f2) = args(false);
                let o2 = binomial_location_scale_nll_generic::<Order2<2>>(
                    y2,
                    w2,
                    et2,
                    el2,
                    q2,
                    mu2,
                    d12,
                    d22,
                    d32,
                    lk2,
                    f2,
                    true, // need_value: Order2 reads the value channel
                    |x, axis| Order2::variable(x, axis),
                )
                .expect("order2");
                rel_close(o2.value(), tower.v, "order2 value");
                for a in 0..2 {
                    rel_close(o2.g()[a], tower.g[a], "order2 grad");
                    for b in 0..2 {
                        rel_close(o2.h()[a][b], tower.h[a][b], "order2 hess");
                    }
                }

                // OneSeed contracted third Σ_c ℓ_{abc} dir_u_c.
                let truth3 = tower.third_contracted(&dir_u);
                let os = binomial_location_scale_nll_generic::<OneSeed<2>>(
                    y2,
                    w2,
                    et2,
                    el2,
                    q2,
                    mu2,
                    d12,
                    d22,
                    d32,
                    lk2,
                    false, // include_fourth
                    false, // need_value: contracted third is independent of d[0]
                    |x, axis| OneSeed::seed_direction(x, axis, dir_u[axis]),
                )
                .expect("oneseed");
                let third = os.contracted_third();
                for a in 0..2 {
                    for b in 0..2 {
                        rel_close(third[a][b], truth3[a][b], "oneseed third");
                    }
                }

                // TwoSeed contracted fourth Σ_{cd} ℓ_{abcd} u_c v_d.
                let truth4 = tower.fourth_contracted(&dir_u, &dir_v);
                let ts = binomial_location_scale_nll_generic::<TwoSeed<2>>(
                    y2,
                    w2,
                    et2,
                    el2,
                    q2,
                    mu2,
                    d12,
                    d22,
                    d32,
                    lk2,
                    true,  // include_fourth
                    false, // need_value: contracted fourth is independent of d[0]
                    |x, axis| TwoSeed::seed(x, axis, dir_u[axis], dir_v[axis]),
                )
                .expect("twoseed");
                let fourth = ts.contracted_fourth();
                for a in 0..2 {
                    for b in 0..2 {
                        rel_close(fourth[a][b], truth4[a][b], "twoseed fourth");
                    }
                }
            }
        }
    }

    #[test]
    fn measure_gradient_order1_vs_tower4_932() {
        use std::hint::black_box;
        use std::time::Instant;

        let links = [
            InverseLink::Standard(StandardLink::Logit),
            InverseLink::Standard(StandardLink::Probit),
            InverseLink::Standard(StandardLink::CLogLog),
        ];
        let y = 1.0;
        let weight = 1.3;
        let eta_t = -0.7;
        let eta_ls = 0.5;
        let SigmaJet1 { sigma, .. } = exp_sigma_jet1_scalar(eta_ls);
        let q = binomial_location_scale_q0(eta_t, sigma);
        let iterations = if cfg!(debug_assertions) { 4 } else { 250_000 };
        for link in &links {
            let jet = inverse_link_jet_for_inverse_link(link, q).expect("inverse-link jet");
            let gradient = binomial_location_scale_nll_gradient(
                y, weight, eta_t, eta_ls, q, jet.mu, jet.d1, jet.d2, jet.d3, link,
            )
            .expect("order1 gradient");
            let tower = binomial_location_scale_nll_tower(
                y, weight, eta_t, eta_ls, q, jet.mu, jet.d1, jet.d2, jet.d3, link, false,
            )
            .expect("tower gradient baseline");
            assert_eq!(
                gradient, tower.g,
                "Order1 must be bit-identical to Tower4 gradient"
            );

            let mut order1_best = f64::INFINITY;
            let mut tower_best = f64::INFINITY;
            for _ in 0..5 {
                let start = Instant::now();
                for _ in 0..iterations {
                    black_box(
                        binomial_location_scale_nll_gradient(
                            black_box(y),
                            black_box(weight),
                            black_box(eta_t),
                            black_box(eta_ls),
                            black_box(q),
                            black_box(jet.mu),
                            black_box(jet.d1),
                            black_box(jet.d2),
                            black_box(jet.d3),
                            black_box(link),
                        )
                        .expect("order1 timing"),
                    );
                }
                order1_best = order1_best.min(start.elapsed().as_secs_f64());

                let start = Instant::now();
                for _ in 0..iterations {
                    black_box(
                        binomial_location_scale_nll_tower(
                            black_box(y),
                            black_box(weight),
                            black_box(eta_t),
                            black_box(eta_ls),
                            black_box(q),
                            black_box(jet.mu),
                            black_box(jet.d1),
                            black_box(jet.d2),
                            black_box(jet.d3),
                            black_box(link),
                            false,
                        )
                        .expect("tower timing"),
                    );
                }
                tower_best = tower_best.min(start.elapsed().as_secs_f64());
            }
            let order1_ns = order1_best * 1e9 / iterations as f64;
            let tower_ns = tower_best * 1e9 / iterations as f64;
            eprintln!(
                "BINOMIAL-LS-GRAD-932 link={link:?} tower4={tower_ns:.2} ns/row order1={order1_ns:.2} ns/row speedup={:.3}x",
                tower_ns / order1_ns,
            );
            if !cfg!(debug_assertions) {
                assert!(
                    order1_ns < tower_ns,
                    "Order1 gradient must beat Tower4 for {link:?}: {order1_ns} vs {tower_ns} ns/row"
                );
            }
        }
    }
}

#[cfg(test)]
mod simd_directional_bit_identity_tests {
    //! The SIMD 4-rows-per-pass directional/bidirectional coefficient builders
    //! ([`binomial_location_scale_first_directional_coefficients`] /
    //! [`binomial_location_scalesecond_directional_coefficients`], now lane
    //! batched) must be `f64::to_bits`-identical, for EVERY row, to the scalar
    //! per-row `OneSeed<2>`/`TwoSeed<2>` contraction they replaced — including the
    //! `n % 4 != 0` trailing-batch tail.
    use super::*;
    use gam_math::jet_scalar::{OneSeed, TwoSeed};
    use gam_problem::{InverseLink, StandardLink};

    /// Generic per-row NLL from the precomputed core, parameterised on the
    /// `JetScalar<2>` the consumer needs (the packed `Order2`/`OneSeed`/`TwoSeed`
    /// scalars for the Hessian / contracted-third / contracted-fourth hot paths,
    /// without the dense `Tower4<2>` `t3`/`t4`). Reconstructs `(η_t, η_ls)` from the
    /// core's `(σ, q0)` and forwards the per-row stack to
    /// `binomial_location_scale_nll_generic`.
    ///
    /// Test-only to-bits reference oracle for
    /// `simd_directional_coefficients_match_scalar_per_row_to_bits`; the production
    /// hot paths now evaluate the directional contractions through the SIMD
    /// lane-batched kernels.
    #[inline]
    fn binomial_location_scale_nll_generic_from_core_row<S: gam_math::jet_scalar::JetScalar<2>>(
        y: f64,
        weight: f64,
        core: &BinomialLocationScaleCore,
        row: usize,
        link_kind: &InverseLink,
        include_fourth: bool,
        need_value: bool,
        seed: impl Fn(f64, usize) -> S,
    ) -> Result<S, String> {
        let sigma = core.sigma[row];
        let eta_t = -core.q0[row] * sigma;
        let eta_ls = sigma.ln();
        binomial_location_scale_nll_generic::<S>(
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
            need_value,
            seed,
        )
    }

    /// Tiny deterministic LCG (no external rng dep in the test).
    struct Lcg(u64);
    impl Lcg {
        fn step(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        /// Finite value in roughly `[-1.5, 1.5]`, occasionally exact `0.0`.
        fn val(&mut self) -> f64 {
            let u = self.step();
            if u & 0x1F == 0 {
                return 0.0;
            }
            ((u >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 3.0
        }
    }

    fn assert_bits(a: f64, b: f64, label: &str) {
        if a.is_nan() {
            assert!(b.is_nan(), "{label}: scalar NaN but SIMD finite ({b:e})");
        } else {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label}: SIMD {a:+.17e} != scalar {b:+.17e}"
            );
        }
    }

    #[test]
    fn simd_directional_coefficients_match_scalar_per_row_to_bits() {
        let links = [
            InverseLink::Standard(StandardLink::Logit),
            InverseLink::Standard(StandardLink::Probit),
            InverseLink::Standard(StandardLink::CLogLog),
        ];
        let mut rng = Lcg(0xD1B54A32D192ED03);
        let mut compared = 0usize;
        let mut tail_seen = false;
        // Deliberately n % 4 != 0 to exercise the trailing partial batch.
        for &n in &[13usize, 17, 23, 30, 1, 2, 3] {
            if n % 4 != 0 {
                tail_seen = true;
            }
            for link in &links {
                let y = Array1::from_iter((0..n).map(|_| (rng.step() & 1) as f64));
                let weights = Array1::from_iter((0..n).map(|_| rng.val().abs() + 0.3));
                let eta_t = Array1::from_iter((0..n).map(|_| rng.val()));
                let eta_ls = Array1::from_iter((0..n).map(|_| rng.val() * 0.5));
                let core = binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, link)
                    .expect("core");

                let d_eta_t = Array1::from_iter((0..n).map(|_| rng.val()));
                let d_eta_ls = Array1::from_iter((0..n).map(|_| rng.val()));
                let d_eta_t_v = Array1::from_iter((0..n).map(|_| rng.val()));
                let d_eta_ls_v = Array1::from_iter((0..n).map(|_| rng.val()));

                // ---- First directional (OneSeed contracted third) ----
                let (tt, tl, ll) = binomial_location_scale_first_directional_coefficients(
                    &y, &weights, &core, &d_eta_t, &d_eta_ls, link,
                )
                .expect("first directional");
                for i in 0..n {
                    let dir = [d_eta_t[i], d_eta_ls[i]];
                    let s = binomial_location_scale_nll_generic_from_core_row::<OneSeed<2>>(
                        y[i],
                        weights[i],
                        &core,
                        i,
                        link,
                        false,
                        false,
                        |x, axis| OneSeed::seed_direction(x, axis, dir[axis]),
                    )
                    .expect("scalar oneseed");
                    let c = s.contracted_third();
                    assert_bits(tt[i], c[0][0], "first tt");
                    assert_bits(tl[i], c[0][1], "first tl");
                    assert_bits(ll[i], c[1][1], "first ll");
                    compared += 3;
                }

                // ---- Second directional (TwoSeed contracted fourth) ----
                let (tt2, tl2, ll2) = binomial_location_scalesecond_directional_coefficients(
                    &y,
                    &weights,
                    &core,
                    &d_eta_t,
                    &d_eta_ls,
                    &d_eta_t_v,
                    &d_eta_ls_v,
                    link,
                )
                .expect("second directional");
                for i in 0..n {
                    let dir_u = [d_eta_t[i], d_eta_ls[i]];
                    let dir_v = [d_eta_t_v[i], d_eta_ls_v[i]];
                    let s = binomial_location_scale_nll_generic_from_core_row::<TwoSeed<2>>(
                        y[i],
                        weights[i],
                        &core,
                        i,
                        link,
                        true,
                        false,
                        |x, axis| TwoSeed::seed(x, axis, dir_u[axis], dir_v[axis]),
                    )
                    .expect("scalar twoseed");
                    let c = s.contracted_fourth();
                    assert_bits(tt2[i], c[0][0], "second tt");
                    assert_bits(tl2[i], c[0][1], "second tl");
                    assert_bits(ll2[i], c[1][1], "second ll");
                    compared += 3;
                }
            }
        }
        assert!(tail_seen, "tail (n % 4 != 0) never exercised");
        assert!(
            compared >= 1000,
            "expected >=1000 comparisons, got {compared}"
        );
    }
}
