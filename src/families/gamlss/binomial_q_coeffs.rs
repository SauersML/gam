//! Pure scalar objective-derivative coefficient helpers for the binomial
//! location-scale family.
//!
//! Self-contained seam extracted from the gamlss monolith (issue #780): the
//! three closed-form `f64` arithmetic kernels that assemble the Hessian
//! coefficient, its first directional derivative, and its mixed second
//! directional derivative from the per-row objective derivative magnitudes
//! `m_k = F^(k)(q)` and the scalar `q`-map derivative terms. They operate
//! purely on `f64` scalars and depend on nothing else in the module.

#[inline]
pub(crate) fn hessian_coeff_fromobjective_q_terms(
    m1: f64,
    m2: f64,
    q_a: f64,
    q_b: f64,
    q_ab: f64,
) -> f64 {
    // F = -sum ell, scalar q:
    //   H_ab = m2 * q_a q_b + m1 * q_ab.
    m2 * q_a * q_b + m1 * q_ab
}

#[inline]
pub(crate) fn directionalhessian_coeff_fromobjective_q_terms(
    m1: f64,
    m2: f64,
    m3: f64,
    dq: f64,
    q_a: f64,
    q_b: f64,
    q_ab: f64,
    dq_a: f64,
    dq_b: f64,
    dq_ab: f64,
) -> f64 {
    // F = -sum ell, scalar q:
    //   dH_ab[u] = m3*dq*q_a*q_b + m2*(dq_a*q_b + q_a*dq_b + dq*q_ab) + m1*dq_ab.
    m3 * dq * q_a * q_b + m2 * (dq_a * q_b + q_a * dq_b + dq * q_ab) + m1 * dq_ab
}

#[inline]
pub(crate) fn second_directionalhessian_coeff_fromobjective_q_terms(
    m1: f64,
    m2: f64,
    m3: f64,
    m4: f64,
    dq_u: f64,
    dqv: f64,
    d2q_uv: f64,
    q_a: f64,
    q_b: f64,
    q_ab: f64,
    dq_a_u: f64,
    dq_av: f64,
    dq_b_u: f64,
    dq_bv: f64,
    d2q_a_uv: f64,
    d2q_b_uv: f64,
    dq_ab_u: f64,
    dq_abv: f64,
    d2q_ab_uv: f64,
) -> f64 {
    // F = -sum ell, scalar q:
    //   H_ab = m2 * q_a q_b + m1 * q_ab.
    // Exact mixed second directional derivative:
    //
    // Write
    //   A = q_a q_b,
    //   B = q_ab.
    //
    // Then
    //   H_ab = m2 * A + m1 * B,
    // where m_k = F^(k)(q).
    //
    // First directional derivative along u:
    //   D_u H_ab
    //   = m3 * dq_u * A
    //   + m2 * (D_u A + dq_u * B)
    //   + m1 * D_u B.
    //
    // Differentiate once more along v:
    //   D²H_ab[u,v] =
    //      m4*dq_u*dqv*q_a*q_b
    //    + m3*(d2q_uv*q_a*q_b
    //         + dq_u*(dq_av*q_b + q_a*dq_bv)
    //         + dqv*(dq_a_u*q_b + q_a*dq_b_u)
    //         + dq_u*dqv*q_ab)
    //    + m2*(d2q_a_uv*q_b + dq_a_u*dq_bv + dq_av*dq_b_u + q_a*d2q_b_uv
    //          + d2q_uv*q_ab + dq_u*dq_abv + dqv*dq_ab_u)
    //    + m1*d2q_ab_uv.
    //
    // The single dq_u*dqv*q_ab term is important. There is exactly one copy:
    //
    //   Dv[m2 * dq_u * B]
    //   = m3 * dqv * dq_u * B + m2 * (d2q_uv * B + dq_u * Dv B),
    //
    // and no second copy appears elsewhere. A previous version of this helper
    // accidentally counted this term twice by embedding `dqv * q_ab` in both
    // the `dq_u` and `dqv` product-rule branches.
    let d_qaqb_u = dq_a_u * q_b + q_a * dq_b_u;
    let d_qaqbv = dq_av * q_b + q_a * dq_bv;
    let d2_qaqb_uv = d2q_a_uv * q_b + dq_a_u * dq_bv + dq_av * dq_b_u + q_a * d2q_b_uv;
    m4 * dq_u * dqv * q_a * q_b
        + m3 * (d2q_uv * q_a * q_b + dq_u * d_qaqbv + dqv * d_qaqb_u + dq_u * dqv * q_ab)
        + m2 * (d2_qaqb_uv + d2q_uv * q_ab + dq_u * dq_abv + dqv * dq_ab_u)
        + m1 * d2q_ab_uv
}
