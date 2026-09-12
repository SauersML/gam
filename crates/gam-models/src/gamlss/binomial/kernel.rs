// Real concern-organized submodule of the gamlss family stack.
// Cross-module items are re-exported flat through the parent (`gamlss.rs`),
// so `use super::*;` makes the sibling-concern symbols this module references
// resolve through the parent namespace.
use super::*;

use gam_row_macros::row_program;

pub(crate) struct BinomialLocationScaleCore {
    pub(crate) sigma: Array1<f64>,
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
}

#[derive(Clone, Copy)]
pub(crate) struct BinomialLocationScaleRow {
    pub(crate) sigma: f64,
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
    }
}

/// Classical binomial deviance `2·Σ wᵢ [y ln(y/μ) + (1−y) ln((1−y)/(1−μ))]` at
/// the fitted probabilities `mu` — the number every standard binomial fit
/// reports, shared by the location-scale family, its link-wiggle form, and
/// the mean-wiggle family (#2786). For a 0/1 response it equals `−2·log L`;
/// for a grouped proportion with trial weights it does not, because the
/// saturated log-likelihood is then non-zero.
pub(crate) fn binomial_classical_deviance(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    mu: &Array1<f64>,
) -> Result<f64, String> {
    use gam_math::special::xlogy;
    if y.len() != weights.len() || y.len() != mu.len() {
        return Err(format!(
            "binomial classical deviance size mismatch: y={}, weights={}, mu={}",
            y.len(),
            weights.len(),
            mu.len()
        ));
    }
    let mut half = 0.0_f64;
    for i in 0..y.len() {
        let w = weights[i];
        if w == 0.0 {
            continue;
        }
        let (yi, mui) = (y[i], mu[i]);
        let unit = xlogy(yi, yi / mui) + xlogy(1.0 - yi, (1.0 - yi) / (1.0 - mui));
        half += w * unit;
        if !half.is_finite() {
            return Err(format!(
                "binomial classical deviance is non-finite at row {i}: y={yi}, mu={mui}, weight={w}"
            ));
        }
    }
    Ok(2.0 * half)
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
                gam_math::probability::log1mexp_positive(z)
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
    let SigmaJet1 { sigma, .. } = exp_sigma_jet1_scalar(eta_ls);
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

    // Write each row's six scalars directly into preallocated output buffers
    // in parallel, reducing the per-row log-likelihood alongside. The previous
    // path collected a `Vec<BinomialLocationScaleRow>` (its scalar fields plus
    // alignment) and then serially scattered into the `Array1`s, which at
    // large scale n=3e5 cost ~50 MB of transient allocation and a
    // single-threaded post-pass.
    let mut sigma = vec![0.0_f64; n];
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
        q0: Array1::from_vec(q0),
        mu: Array1::from_vec(mu),
        dmu_dq: Array1::from_vec(dmu_dq),
        d2mu_dq2: Array1::from_vec(d2mu_dq2),
        d3mu_dq3: Array1::from_vec(d3mu_dq3),
        log_likelihood: ll,
    })
}

// The binomial location-scale row negative log-likelihood, declared ONCE
// (#932). Every production derivative channel of this family's observed row —
// the score, the joint-Hessian row coefficients, and the first and second
// directional derivatives of that Hessian — is emitted from this declaration;
// no family code states a chain rule of its own.
//
// The declaration is written in local coordinates around the row's expansion
// point `(η_t, η_ls)`. With `δ_t, δ_ls` the offsets from that point and the
// exact σ link `σ = e^{η_ls}`,
//
//   q(δ) = −(η_t + δ_t)·e^{−(η_ls + δ_ls)} = (q0 − δ_t/σ)·e^{−δ_ls},  q0 = −η_t/σ,
//
// so the row core's `q0` and `1/σ` are the only map state the program reads,
// and `q` at the expansion point is exactly the `q0` at which the q-space loss
// stack `[−ℓ, m1, m2, m3, m4]` (`m_k = dᵏ(−ℓ)/dqᵏ`) was evaluated. Both
// derivative stacks are supplied: the exponential's at `δ_ls = 0` is all ones,
// and the loss stack comes from the row core. Every surface is valid at `δ = 0`
// only, which is the only point production evaluates it at, so callers pass
// literal zeros for both primaries.
//
// An all-zero loss stack is how a row says it contributes nothing (a zero
// weight, a saturated compatible tail), and the program skips it instead of
// composing it: composing a zero stack against a map whose `1/σ` overflowed
// forms `0·∞` in the derivative channels. The condition is also what makes
// every surface read every stack entry, so a caller passes zero for an entry a
// surface does not need and the entry folds away after inlining.
row_program! {
    pub(crate) fn binomial_ls_row_program(
        delta_eta_t,
        delta_eta_ls;
        q0,
        inv_sigma,
        neg_ll,
        m1,
        m2,
        m3,
        m4
    )
    emit [order2, third, fourth];
    leaves {
        unit_exponential => supplied,
        loss => supplied,
    }
    witnesses [];
    {
        let neg_delta_eta_ls = neg(delta_eta_ls);
        let scale_ratio = compose(unit_exponential, neg_delta_eta_ls, 1.0, 1.0, 1.0, 1.0, 1.0);
        let shifted_q = add_constant(scale(delta_eta_t, -inv_sigma), q0);
        let q = mul(shifted_q, scale_ratio);
        let mut nll = zero();
        if (neg_ll != 0.0 || m1 != 0.0 || m2 != 0.0 || m3 != 0.0 || m4 != 0.0) {
            nll = compose(loss, q, neg_ll, m1, m2, m3, m4);
        }
        return nll;
    }
}

/// Score `∂(−ℓ)/∂(η_t, η_ls)` of one row: the gradient channel of the
/// `binomial_ls_row_program` order-2 surface at the row's expansion point.
#[inline]
pub(crate) fn binomial_location_scale_row_score(
    y: f64,
    weight: f64,
    q0: f64,
    inv_sigma: f64,
    mu: f64,
    dmu_dq: f64,
    d2mu_dq2: f64,
    d3mu_dq3: f64,
    link_kind: &InverseLink,
) -> [f64; 2] {
    let (m1, _, _) = binomial_neglog_q_derivatives_dispatch(
        y, weight, q0, mu, dmu_dq, d2mu_dq2, d3mu_dq3, link_kind,
    );
    let (_, score, _, []) =
        binomial_ls_row_program_order2(0.0, 0.0, q0, inv_sigma, 0.0, m1, 0.0, 0.0, 0.0);
    score
}

/// Joint Hessian `∂²(−ℓ)/∂(η_t, η_ls)²` of one row: the Hessian channel of the
/// `binomial_ls_row_program` order-2 surface at the row's expansion point.
#[inline]
pub(crate) fn binomial_location_scale_row_hessian(
    y: f64,
    weight: f64,
    q0: f64,
    inv_sigma: f64,
    mu: f64,
    dmu_dq: f64,
    d2mu_dq2: f64,
    d3mu_dq3: f64,
    link_kind: &InverseLink,
) -> [[f64; 2]; 2] {
    let (m1, m2, _) = binomial_neglog_q_derivatives_dispatch(
        y, weight, q0, mu, dmu_dq, d2mu_dq2, d3mu_dq3, link_kind,
    );
    let (_, _, hessian, []) =
        binomial_ls_row_program_order2(0.0, 0.0, q0, inv_sigma, 0.0, m1, m2, 0.0, 0.0);
    hessian
}

/// Row coefficients of the joint directional derivative `D_β H_L[u]`: the
/// `binomial_ls_row_program` third surface contracted along the predictor
/// perturbation `(d_eta_t, d_eta_ls) = (X_t·u_t, X_ls·u_ls)`. Returns
/// `(c_tt, c_tl, c_ll)` such that the resulting matrix is
///
///   X_t^T diag(c_tt) X_t + X_t^T diag(c_tl) X_ls (+ symmetric)
///   + X_ls^T diag(c_ll) X_ls.
pub(crate) fn binomial_location_scale_first_directional_coefficients(
    y: &Array1<f64>,
    weights: &Array1<f64>,
    core: &BinomialLocationScaleCore,
    d_eta_t: &Array1<f64>,
    d_eta_ls: &Array1<f64>,
    link_kind: &InverseLink,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    let n = y.len();
    let mut coeff_tt = vec![0.0_f64; n];
    let mut coeff_tl = vec![0.0_f64; n];
    let mut coeff_ll = vec![0.0_f64; n];
    coeff_tt
        .par_iter_mut()
        .zip(coeff_tl.par_iter_mut())
        .zip(coeff_ll.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((c_tt, c_tl), c_ll))| {
            let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
                y[i],
                weights[i],
                core.q0[i],
                core.mu[i],
                core.dmu_dq[i],
                core.d2mu_dq2[i],
                core.d3mu_dq3[i],
                link_kind,
            );
            let third = binomial_ls_row_program_third_contracted(
                0.0,
                0.0,
                core.q0[i],
                core.sigma[i].recip(),
                0.0,
                m1,
                m2,
                m3,
                0.0,
                &[d_eta_t[i], d_eta_ls[i]],
            );
            *c_tt = third[0][0];
            *c_tl = third[0][1];
            *c_ll = third[1][1];
        });
    Ok((
        Array1::from_vec(coeff_tt),
        Array1::from_vec(coeff_tl),
        Array1::from_vec(coeff_ll),
    ))
}

/// Row coefficients of the joint second directional derivative
/// `D²_β H_L[u, v]`: the `binomial_ls_row_program` fourth surface contracted
/// along the predictor perturbations `(d_eta_t_u, d_eta_ls_u)` and
/// `(d_eta_t_v, d_eta_ls_v)`. Returns `(c_tt, c_tl, c_ll)` analogous to the
/// first-order builder.
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
    let n = y.len();
    let mut coeff_tt = vec![0.0_f64; n];
    let mut coeff_tl = vec![0.0_f64; n];
    let mut coeff_ll = vec![0.0_f64; n];
    coeff_tt
        .par_iter_mut()
        .zip(coeff_tl.par_iter_mut())
        .zip(coeff_ll.par_iter_mut())
        .enumerate()
        .try_for_each(|(i, ((c_tt, c_tl), c_ll))| -> Result<(), String> {
            let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
                y[i],
                weights[i],
                core.q0[i],
                core.mu[i],
                core.dmu_dq[i],
                core.d2mu_dq2[i],
                core.d3mu_dq3[i],
                link_kind,
            );
            let m4 = binomial_neglog_q_fourth_derivative_dispatch(
                y[i],
                weights[i],
                core.q0[i],
                core.mu[i],
                core.dmu_dq[i],
                core.d2mu_dq2[i],
                core.d3mu_dq3[i],
                link_kind,
            )?;
            let fourth = binomial_ls_row_program_fourth_contracted(
                0.0,
                0.0,
                core.q0[i],
                core.sigma[i].recip(),
                0.0,
                m1,
                m2,
                m3,
                m4,
                &[d_eta_t_u[i], d_eta_ls_u[i]],
                &[d_eta_t_v[i], d_eta_ls_v[i]],
            );
            *c_tt = fourth[0][0];
            *c_tl = fourth[0][1];
            *c_ll = fourth[1][1];
            Ok(())
        })?;
    Ok((
        Array1::from_vec(coeff_tt),
        Array1::from_vec(coeff_tl),
        Array1::from_vec(coeff_ll),
    ))
}

#[cfg(test)]
mod packed_scalar_oracle_tests {
    //! #932 oracle and speed gate for the score lowering of
    //! `binomial_ls_row_program`: the emitted order-2 surface's gradient against
    //! the dense `Tower4<2>` builder ([`binomial_location_scale_nll_tower`]),
    //! which spells the row NLL separately in predictor coordinates.
    use super::*;
    use crate::gamlss::test_support::binomial_location_scale_nll_tower;
    use gam_problem::{InverseLink, StandardLink};

    /// The emitted score must agree with the `Tower4` gradient (every build) and
    /// be faster than it (release profile, where the codegen layout is the
    /// shipped one -- `SpeedGate::open` documents why). The two spell `1/σ` as
    /// `1/e^{η_ls}` and `e^{−η_ls}`, so they agree within a few units in the last
    /// place of each factor rather than to the bit.
    #[test]
    fn measure_row_score_vs_tower4_932() {
        use gam_math::paired_timing::{SpeedGate, batched, paired_interleaved};

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
        let inv_sigma = sigma.recip();
        let q = binomial_location_scale_q0(eta_t, sigma);
        let mut gate = (!cfg!(debug_assertions)).then(|| SpeedGate::open("BINOMIAL-LS-GRAD-932"));
        for link in &links {
            let jet = inverse_link_jet_for_inverse_link(link, q).expect("inverse-link jet");
            let score = binomial_location_scale_row_score(
                y, weight, q, inv_sigma, jet.mu, jet.d1, jet.d2, jet.d3, link,
            );
            let tower = binomial_location_scale_nll_tower(
                y, weight, eta_t, eta_ls, q, jet.mu, jet.d1, jet.d2, jet.d3, link, false,
            )
            .expect("tower gradient baseline");
            for axis in 0..2 {
                let bound = 8.0 * f64::EPSILON * (score[axis].abs() + tower.g[axis].abs());
                assert!(
                    score[axis].is_finite() && (score[axis] - tower.g[axis]).abs() <= bound,
                    "{link:?}: emitted score[{axis}] {:+.17e} != tower {:+.17e} (bound {bound:.3e})",
                    score[axis],
                    tower.g[axis]
                );
            }
            let Some(gate) = gate.as_mut() else {
                continue;
            };
            // The nudge perturbs the threshold predictor, so consecutive
            // iterations cannot be folded; both arms score the same nudged row,
            // with the loss stack evaluated at the nudged `q` in both, and fold
            // the threshold channel back into the harness checksum.
            let timing = paired_interleaved(
                15,
                5_000,
                0x9320_B1A5,
                batched(64, |nudge| {
                    let nudged_q = -(eta_t + nudge) * inv_sigma;
                    binomial_location_scale_row_score(
                        y, weight, nudged_q, inv_sigma, jet.mu, jet.d1, jet.d2, jet.d3, link,
                    )[0]
                }),
                batched(64, |nudge| {
                    let nudged_q = -(eta_t + nudge) * inv_sigma;
                    binomial_location_scale_nll_tower(
                        y,
                        weight,
                        eta_t + nudge,
                        eta_ls,
                        nudged_q,
                        jet.mu,
                        jet.d1,
                        jet.d2,
                        jet.d3,
                        link,
                        false,
                    )
                    .expect("tower timing")
                    .g[0]
                }),
            );
            gate.faster(&format!("link={link:?}"), &timing, "row_program", "tower4");
        }
        if let Some(gate) = gate {
            gate.finish();
        }
    }
}
#[cfg(test)]
mod row_program_oracle_tests {
    //! #932: every surface of `binomial_ls_row_program` production reads — the
    //! row score and Hessian and the first and second directional Hessian
    //! coefficients — against the dense `Tower4<2>` builder
    //! ([`binomial_location_scale_nll_tower`]), which spells the row NLL
    //! separately in predictor coordinates, on every channel of ordinary and
    //! far-tail rows, both outcomes, and the three closed-form links.
    use super::*;
    use crate::gamlss::test_support::binomial_location_scale_nll_tower;
    use gam_problem::{InverseLink, StandardLink};

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

    /// Magnitude bound for the products one channel sums. Every channel is a
    /// sum, over the partitions of at most four derivative slots, of one loss
    /// derivative `m_k` times at most four q-map factors (each at most
    /// `1/σ + |q0|` in magnitude) times at most one component of each
    /// contraction direction; this majorises every such product.
    fn product_majorant(
        stack: [f64; 4],
        q0: f64,
        inv_sigma: f64,
        u: [f64; 2],
        v: [f64; 2],
    ) -> f64 {
        let loss = 1.0 + stack.iter().map(|m| m.abs()).sum::<f64>();
        let map = 1.0 + inv_sigma + q0.abs();
        let direction = |d: [f64; 2]| 1.0 + d[0].abs() + d[1].abs();
        loss * map.powi(4) * direction(u) * direction(v)
    }

    /// An emitted channel agrees with the tower channel when both are finite and
    /// their difference is inside the rounding of their two spellings: at most
    /// fifteen partition terms of at most four factors each, a few units of
    /// relative rounding per operation, and the `1/σ`-versus-`e^{−η_ls}` spelling
    /// of each map factor, all against the product majorant. A non-finite value
    /// on either side is a disagreement, never an agreement.
    fn channel_agrees(emitted: f64, oracle: f64, majorant: f64) -> bool {
        emitted.is_finite()
            && oracle.is_finite()
            && (emitted - oracle).abs() <= 256.0 * f64::EPSILON * majorant
    }

    #[test]
    fn emitted_surfaces_match_independent_tower_on_every_channel_932() {
        let links = [
            InverseLink::Standard(StandardLink::Logit),
            InverseLink::Standard(StandardLink::Probit),
            InverseLink::Standard(StandardLink::CLogLog),
        ];
        let mut rng = Lcg(0xD1B54A32D192ED03);
        let mut compared = 0usize;
        let mut expected = 0usize;
        let mut largest = 0.0_f64;
        for link in &links {
            let mut y = Vec::new();
            let mut weights = Vec::new();
            let mut eta_t = Vec::new();
            let mut eta_ls = Vec::new();
            for _ in 0..48 {
                y.push((rng.step() & 1) as f64);
                weights.push(rng.val().abs() + 0.3);
                eta_t.push(rng.val());
                eta_ls.push(rng.val() * 0.5);
            }
            // Far-tail rows: |q0| near 20 and near 7 against both outcomes, and a
            // large log-scale that shrinks the map.
            for (t, ls) in [(6.0, -1.2), (-6.0, -1.2), (9.0, 0.3), (-0.4, 2.5)] {
                for outcome in [0.0, 1.0] {
                    y.push(outcome);
                    weights.push(1.1);
                    eta_t.push(t);
                    eta_ls.push(ls);
                }
            }
            let n = y.len();
            let y = Array1::from_vec(y);
            let weights = Array1::from_vec(weights);
            let eta_t = Array1::from_vec(eta_t);
            let eta_ls = Array1::from_vec(eta_ls);
            let core = binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, link)
                .expect("core");
            let d_eta_t_u = Array1::from_iter((0..n).map(|_| rng.val()));
            let d_eta_ls_u = Array1::from_iter((0..n).map(|_| rng.val()));
            let d_eta_t_v = Array1::from_iter((0..n).map(|_| rng.val()));
            let d_eta_ls_v = Array1::from_iter((0..n).map(|_| rng.val()));
            let (first_tt, first_tl, first_ll) =
                binomial_location_scale_first_directional_coefficients(
                    &y, &weights, &core, &d_eta_t_u, &d_eta_ls_u, link,
                )
                .expect("first directional");
            let (second_tt, second_tl, second_ll) =
                binomial_location_scalesecond_directional_coefficients(
                    &y,
                    &weights,
                    &core,
                    &d_eta_t_u,
                    &d_eta_ls_u,
                    &d_eta_t_v,
                    &d_eta_ls_v,
                    link,
                )
                .expect("second directional");
            expected += 14 * n;
            for i in 0..n {
                let inv_sigma = core.sigma[i].recip();
                let score = binomial_location_scale_row_score(
                    y[i],
                    weights[i],
                    core.q0[i],
                    inv_sigma,
                    core.mu[i],
                    core.dmu_dq[i],
                    core.d2mu_dq2[i],
                    core.d3mu_dq3[i],
                    link,
                );
                let hessian = binomial_location_scale_row_hessian(
                    y[i],
                    weights[i],
                    core.q0[i],
                    inv_sigma,
                    core.mu[i],
                    core.dmu_dq[i],
                    core.d2mu_dq2[i],
                    core.d3mu_dq3[i],
                    link,
                );
                let tower = binomial_location_scale_nll_tower(
                    y[i],
                    weights[i],
                    eta_t[i],
                    eta_ls[i],
                    core.q0[i],
                    core.mu[i],
                    core.dmu_dq[i],
                    core.d2mu_dq2[i],
                    core.d3mu_dq3[i],
                    link,
                    true,
                )
                .expect("row tower");
                let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
                    y[i],
                    weights[i],
                    core.q0[i],
                    core.mu[i],
                    core.dmu_dq[i],
                    core.d2mu_dq2[i],
                    core.d3mu_dq3[i],
                    link,
                );
                let m4 = binomial_neglog_q_fourth_derivative_dispatch(
                    y[i],
                    weights[i],
                    core.q0[i],
                    core.mu[i],
                    core.dmu_dq[i],
                    core.d2mu_dq2[i],
                    core.d3mu_dq3[i],
                    link,
                )
                .expect("fourth loss derivative");
                let stack = [m1, m2, m3, m4];
                let u = [d_eta_t_u[i], d_eta_ls_u[i]];
                let v = [d_eta_t_v[i], d_eta_ls_v[i]];
                let mut tower_first = [[0.0_f64; 2]; 2];
                let mut tower_second = [[0.0_f64; 2]; 2];
                for a in 0..2 {
                    for b in 0..2 {
                        for c in 0..2 {
                            tower_first[a][b] += tower.t3[a][b][c] * u[c];
                            for d in 0..2 {
                                tower_second[a][b] += tower.t4[a][b][c][d] * u[c] * v[d];
                            }
                        }
                    }
                }
                let emitted_first = [[first_tt[i], first_tl[i]], [first_tl[i], first_ll[i]]];
                let emitted_second = [
                    [second_tt[i], second_tl[i]],
                    [second_tl[i], second_ll[i]],
                ];
                let order2_majorant =
                    product_majorant(stack, core.q0[i], inv_sigma, [0.0; 2], [0.0; 2]);
                let first_majorant = product_majorant(stack, core.q0[i], inv_sigma, u, [0.0; 2]);
                let second_majorant = product_majorant(stack, core.q0[i], inv_sigma, u, v);
                let mut check = |label: &str, emitted: f64, oracle: f64, majorant: f64| {
                    assert!(
                        channel_agrees(emitted, oracle, majorant),
                        "{link:?} row {i} ({label}): emitted {emitted:+.17e} vs tower \
                         {oracle:+.17e}, majorant {majorant:.3e}"
                    );
                    compared += 1;
                    largest = largest.max(oracle.abs());
                };
                for a in 0..2 {
                    check("score", score[a], tower.g[a], order2_majorant);
                    for b in 0..2 {
                        check("hessian", hessian[a][b], tower.h[a][b], order2_majorant);
                        check(
                            "first directional",
                            emitted_first[a][b],
                            tower_first[a][b],
                            first_majorant,
                        );
                        check(
                            "second directional",
                            emitted_second[a][b],
                            tower_second[a][b],
                            second_majorant,
                        );
                    }
                }
            }
        }
        assert_eq!(
            expected,
            links.len() * 56 * 14,
            "the fixture must hold 56 rows per link"
        );
        assert_eq!(compared, expected, "every channel of every row must be compared");
        assert!(
            largest >= 1.0,
            "no channel of magnitude one was reached (largest {largest:e})"
        );
    }

    /// The comparator the pin rests on refuses what it exists to catch: a
    /// one-part-in-a-million change of a real channel, a sign flip, and
    /// non-finite values on either side.
    #[test]
    fn channel_comparator_rejects_wrong_and_nonfinite_channels_932() {
        let link = InverseLink::Standard(StandardLink::Probit);
        let y = Array1::from_vec(vec![1.0]);
        let weights = Array1::from_vec(vec![1.3]);
        let eta_t = Array1::from_vec(vec![-0.7]);
        let eta_ls = Array1::from_vec(vec![0.5]);
        let core = binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &link)
            .expect("core");
        let inv_sigma = core.sigma[0].recip();
        let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
            y[0],
            weights[0],
            core.q0[0],
            core.mu[0],
            core.dmu_dq[0],
            core.d2mu_dq2[0],
            core.d3mu_dq3[0],
            &link,
        );
        let m4 = binomial_neglog_q_fourth_derivative_dispatch(
            y[0],
            weights[0],
            core.q0[0],
            core.mu[0],
            core.dmu_dq[0],
            core.d2mu_dq2[0],
            core.d3mu_dq3[0],
            &link,
        )
        .expect("fourth loss derivative");
        let hessian = binomial_location_scale_row_hessian(
            y[0],
            weights[0],
            core.q0[0],
            inv_sigma,
            core.mu[0],
            core.dmu_dq[0],
            core.d2mu_dq2[0],
            core.d3mu_dq3[0],
            &link,
        );
        let majorant =
            product_majorant([m1, m2, m3, m4], core.q0[0], inv_sigma, [0.0; 2], [0.0; 2]);
        let channel = hessian[0][0];
        assert!(
            channel.abs() > 1.0e6 * 256.0 * f64::EPSILON * majorant,
            "control channel {channel:e} is too close to its rounding bound to control anything"
        );
        assert!(channel_agrees(channel, channel, majorant));
        assert!(!channel_agrees(channel * (1.0 + 1.0e-6), channel, majorant));
        assert!(!channel_agrees(-channel, channel, majorant));
        assert!(!channel_agrees(f64::NAN, f64::NAN, majorant));
        assert!(!channel_agrees(f64::INFINITY, f64::INFINITY, majorant));
        assert!(!channel_agrees(f64::NEG_INFINITY, channel, majorant));
    }
}
