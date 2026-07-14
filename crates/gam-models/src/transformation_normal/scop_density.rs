use super::*;

pub fn transformation_normal_pit_score(
    h: f64,
    lower: f64,
    upper: f64,
    clip_eps: f64,
) -> Result<f64, String> {
    if !(clip_eps.is_finite() && clip_eps > 0.0 && clip_eps < 0.5) {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "transformation-normal PIT requires clip_eps in (0, 0.5), got {clip_eps}"
            ),
        }
        .into());
    }
    if !(h.is_finite() && lower.is_finite() && upper.is_finite()) {
        return Err(TransformationNormalError::InvalidInput { reason: format!(
            "transformation-normal PIT requires finite h/lower/upper, got h={h}, lower={lower}, upper={upper}"
        ) }.into());
    }
    if upper <= lower {
        return Err(TransformationNormalError::MonotonicityViolated { reason: format!(
            "transformation-normal PIT endpoint order violated: lower={lower:.6e}, upper={upper:.6e}"
        ) }.into());
    }

    // Extrapolation outside `[lower, upper]` is *not* a malformed input —
    // a test sample whose response sits at-or-beyond the training response
    // support will produce a finite `h` slightly below `lower` (or slightly
    // above `upper`) by exactly the amount the kernel reconstructs the
    // boundary. The PIT mapping is still well-defined: `u → 0` when
    // `h ≤ lower`, `u → 1` when `h ≥ upper`, and the `clip_eps` clamp on
    // the standard-normal quantile call at the end of this function turns
    // both into the extreme-quantile finite values that downstream
    // calibration code expects. Refusing here was surfacing routine
    // boundary roundoff at large-scale shape (`p_resp` coefficients × O(1)
    // basis evaluations introduce ~`p_resp·ε·scale` noise — 64·ε·scale
    // is below that floor) as a hard prediction failure.
    //
    // A debug-level log preserves visibility for genuinely far-out
    // inputs without aborting the prediction. Non-finite `h` is already
    // rejected above at the `is_finite()` guard.
    if h < lower || h > upper {
        log::debug!(
            "transformation-normal PIT extrapolation: h={h:.6e}, lower={lower:.6e}, upper={upper:.6e} — clamping to support and continuing"
        );
    }
    let h_inside = h.clamp(lower, upper);
    let u = if h_inside <= lower {
        0.0
    } else if h_inside >= upper {
        1.0
    } else {
        let log_num = log_normal_cdf_diff(h_inside, lower)?;
        let log_den = log_normal_cdf_diff(upper, lower)?;
        let ratio = (log_num - log_den).exp();
        if !(ratio.is_finite() && (-1.0e-12..=1.0 + 1.0e-12).contains(&ratio)) {
            return Err(TransformationNormalError::NumericalFailure { reason: format!(
                "transformation-normal PIT probability is not representable: h={h:.6e}, lower={lower:.6e}, upper={upper:.6e}, ratio={ratio}"
            ) }.into());
        }
        ratio.clamp(0.0, 1.0)
    };
    standard_normal_quantile(u.clamp(clip_eps, 1.0 - clip_eps))
        .map_err(|err| format!("transformation-normal PIT quantile failed: {err}"))
}

/// Accumulates the second-order monotone-transform quantities
/// `(h_i, h_j, h_ij, hp_i, hp_j, hp_ij)` for one row from the response value /
/// derivative bases and the per-response-knot ψ-directional derivatives of the
/// factored coordinates `α_k(x; ψ)`. With the direct-α chart (gam#2306) the
/// transform is LINEAR in the coordinates, so each accumulation is a plain
/// basis-weighted sum; the ψψ second derivative flows entirely through
/// `alpha_ij` (the covariate design is still nonlinear in ψ).
/// Shared verbatim across the SCOP Hessian/HVP/bilinear row loops.
pub(crate) fn scop_second_order_h(
    rv: ArrayView1<'_, f64>,
    rd: ArrayView1<'_, f64>,
    p_resp: usize,
    alpha_i: &[f64],
    alpha_j: &[f64],
    alpha_ij: &[f64],
) -> [f64; 6] {
    let mut h_i = 0.0;
    let mut h_j = 0.0;
    let mut h_ij = 0.0;
    let mut hp_i = 0.0;
    let mut hp_j = 0.0;
    let mut hp_ij = 0.0;
    for k in 0..p_resp {
        h_i += rv[k] * alpha_i[k];
        h_j += rv[k] * alpha_j[k];
        h_ij += rv[k] * alpha_ij[k];
        hp_i += rd[k] * alpha_i[k];
        hp_j += rd[k] * alpha_j[k];
        hp_ij += rd[k] * alpha_ij[k];
    }
    [h_i, h_j, h_ij, hp_i, hp_j, hp_ij]
}

/// Accumulates the second-order endpoint-normalizer chain inputs
/// `(endpoint_i, endpoint_j, endpoint_ij)` for one row. Shared verbatim across
/// the SCOP Hessian/HVP/bilinear row loops.
pub(crate) fn scop_second_order_endpoints(
    endpoint_basis: [&[f64]; 2],
    p_resp: usize,
    alpha_i: &[f64],
    alpha_j: &[f64],
    alpha_ij: &[f64],
) -> ([f64; 2], [f64; 2], [f64; 2]) {
    let mut endpoint_i = [0.0; 2];
    let mut endpoint_j = [0.0; 2];
    let mut endpoint_ij = [0.0; 2];
    for e in 0..2 {
        let basis = endpoint_basis[e];
        for k in 0..p_resp {
            endpoint_i[e] += basis[k] * alpha_i[k];
            endpoint_j[e] += basis[k] * alpha_j[k];
            endpoint_ij[e] += basis[k] * alpha_ij[k];
        }
    }
    (endpoint_i, endpoint_j, endpoint_ij)
}

/// Accumulates the psi-direction transform quantities `(h_psi, hp_psi,
/// endpoint_psi)` for one row from the response bases and the per-knot psi
/// directional derivatives of α. Shared verbatim across the SCOP psi setup
/// loops.
pub(crate) fn scop_psi_marginal(
    rv: ArrayView1<'_, f64>,
    rd: ArrayView1<'_, f64>,
    p_resp: usize,
    endpoint_basis: [&[f64]; 2],
    alpha_psi: &[f64],
) -> (f64, f64, [f64; 2]) {
    let mut h_psi = 0.0;
    let mut hp_psi = 0.0;
    for k in 0..p_resp {
        h_psi += rv[k] * alpha_psi[k];
        hp_psi += rd[k] * alpha_psi[k];
    }

    let mut endpoint_psi = [0.0; 2];
    for e in 0..2 {
        let basis = endpoint_basis[e];
        for k in 0..p_resp {
            endpoint_psi[e] += basis[k] * alpha_psi[k];
        }
    }
    (h_psi, hp_psi, endpoint_psi)
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPS: f64 = 1.0e-12;

    // ---- transformation_normal_pit_score: error semantics ----

    #[test]
    fn pit_rejects_clip_eps_outside_open_half_interval() {
        // clip_eps must satisfy 0 < clip_eps < 0.5.
        for bad in [0.0, -1.0e-3, 0.5, 0.6, f64::NAN, f64::INFINITY] {
            assert!(transformation_normal_pit_score(0.0, -1.0, 1.0, bad).is_err());
        }
        // A value strictly inside (0, 0.5) is accepted.
        assert!(transformation_normal_pit_score(0.0, -1.0, 1.0, 0.25).is_ok());
    }

    #[test]
    fn pit_rejects_nonfinite_h_lower_upper() {
        assert!(transformation_normal_pit_score(f64::NAN, -1.0, 1.0, EPS).is_err());
        assert!(transformation_normal_pit_score(0.0, f64::NEG_INFINITY, 1.0, EPS).is_err());
        assert!(transformation_normal_pit_score(0.0, -1.0, f64::INFINITY, EPS).is_err());
    }

    #[test]
    fn pit_rejects_endpoint_order_violation() {
        // upper <= lower is a monotonicity violation.
        assert!(transformation_normal_pit_score(0.0, 1.0, 1.0, EPS).is_err());
        assert!(transformation_normal_pit_score(0.0, 1.0, 0.5, EPS).is_err());
    }

    // ---- transformation_normal_pit_score: closed-form values ----

    #[test]
    fn pit_symmetric_midpoint_maps_to_zero() {
        // lower = -upper and h = 0: by normal symmetry the conditional CDF is
        // exactly 0.5, and Phi^{-1}(0.5) = 0.
        let u = transformation_normal_pit_score(0.0, -2.0, 2.0, EPS).unwrap();
        assert!(u.abs() < 1e-9, "expected ~0, got {u}");
    }

    #[test]
    fn pit_at_or_below_lower_clamps_to_low_quantile() {
        // h <= lower -> u = 0 -> clamped to clip_eps -> Phi^{-1}(clip_eps).
        let clip = 1e-6;
        let expected = standard_normal_quantile(clip).unwrap();
        let at = transformation_normal_pit_score(-1.0, -1.0, 1.0, clip).unwrap();
        let below = transformation_normal_pit_score(-1.5, -1.0, 1.0, clip).unwrap();
        assert!((at - expected).abs() < 1e-12);
        assert!((below - expected).abs() < 1e-12);
        // Low quantile is a large negative number.
        assert!(expected < -3.0);
    }

    #[test]
    fn pit_at_or_above_upper_clamps_to_high_quantile() {
        // h >= upper -> u = 1 -> clamped to 1 - clip_eps -> Phi^{-1}(1 - clip_eps).
        let clip = 1e-6;
        let expected = standard_normal_quantile(1.0 - clip).unwrap();
        let at = transformation_normal_pit_score(1.0, -1.0, 1.0, clip).unwrap();
        let above = transformation_normal_pit_score(2.0, -1.0, 1.0, clip).unwrap();
        assert!((at - expected).abs() < 1e-12);
        assert!((above - expected).abs() < 1e-12);
        assert!(expected > 3.0);
    }

    #[test]
    fn pit_is_monotone_increasing_in_h() {
        // Strictly inside the support the PIT score increases with h.
        let clip = 1e-9;
        let a = transformation_normal_pit_score(-0.5, -2.0, 2.0, clip).unwrap();
        let b = transformation_normal_pit_score(0.0, -2.0, 2.0, clip).unwrap();
        let c = transformation_normal_pit_score(0.5, -2.0, 2.0, clip).unwrap();
        assert!(a < b && b < c, "not monotone: {a} {b} {c}");
    }

    // ---- scop_second_order_h: pure accumulator closed forms ----

    #[test]
    fn scop_second_order_h_is_linear_in_the_directional_coordinates() {
        // Direct-α chart: every output is a plain basis-weighted sum of the
        // matching α directional-derivative slots — no coordinate factor.
        let rv = array![3.0];
        let rd = array![5.0];
        let ai = [2.0];
        let aj = [7.0];
        let aij = [11.0];
        let out = scop_second_order_h(rv.view(), rd.view(), 1, &ai, &aj, &aij);
        assert_eq!(
            out,
            [
                3.0 * 2.0,
                3.0 * 7.0,
                3.0 * 11.0,
                5.0 * 2.0,
                5.0 * 7.0,
                5.0 * 11.0
            ]
        );
    }

    #[test]
    fn scop_second_order_h_p_resp_two_matches_hand_formula() {
        let rv = array![1.0, 4.0];
        let rd = array![1.0, 6.0];
        let ai = [1.0, 3.0];
        let aj = [1.0, 5.0];
        let aij = [1.0, 7.0];
        let out = scop_second_order_h(rv.view(), rd.view(), 2, &ai, &aj, &aij);
        // h_i = rv0*ai0 + rv1*ai1 = 1 + 12 = 13; hp_i = 1 + 18 = 19; etc.
        assert_eq!(
            out,
            [
                1.0 + 4.0 * 3.0,
                1.0 + 4.0 * 5.0,
                1.0 + 4.0 * 7.0,
                1.0 + 6.0 * 3.0,
                1.0 + 6.0 * 5.0,
                1.0 + 6.0 * 7.0
            ]
        );
    }

    // ---- scop_second_order_endpoints ----

    #[test]
    fn scop_second_order_endpoints_matches_hand_formula() {
        let lower = [1.0, 2.0];
        let upper = [3.0, 4.0];
        let ai = [1.0, 6.0];
        let aj = [1.0, 7.0];
        let aij = [1.0, 8.0];
        let (ei, ej, eij) = scop_second_order_endpoints([&lower, &upper], 2, &ai, &aj, &aij);
        assert_eq!(ei[0], 1.0 + 2.0 * 6.0);
        assert_eq!(ei[1], 3.0 + 4.0 * 6.0);
        assert_eq!(ej[0], 1.0 + 2.0 * 7.0);
        assert_eq!(ej[1], 3.0 + 4.0 * 7.0);
        assert_eq!(eij[0], 1.0 + 2.0 * 8.0);
        assert_eq!(eij[1], 3.0 + 4.0 * 8.0);
    }

    // ---- scop_psi_marginal ----

    #[test]
    fn scop_psi_marginal_matches_hand_formula() {
        let rv = array![1.0, 4.0];
        let rd = array![1.0, 6.0];
        let lower = [1.0, 2.0];
        let upper = [3.0, 4.0];
        let alpha_psi = [9.0, 10.0];
        let (h_psi, hp_psi, endpoint_psi) =
            scop_psi_marginal(rv.view(), rd.view(), 2, [&lower, &upper], &alpha_psi);
        assert_eq!(h_psi, 9.0 + 4.0 * 10.0);
        assert_eq!(hp_psi, 9.0 + 6.0 * 10.0);
        assert_eq!(endpoint_psi[0], 1.0 * 9.0 + 2.0 * 10.0);
        assert_eq!(endpoint_psi[1], 3.0 * 9.0 + 4.0 * 10.0);
    }
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPS: f64 = 1.0e-12;

    // ---- transformation_normal_pit_score: error semantics ----

    #[test]
    fn pit_rejects_clip_eps_outside_open_half_interval() {
        // clip_eps must satisfy 0 < clip_eps < 0.5.
        for bad in [0.0, -1.0e-3, 0.5, 0.6, f64::NAN, f64::INFINITY] {
            assert!(transformation_normal_pit_score(0.0, -1.0, 1.0, bad).is_err());
        }
        // A value strictly inside (0, 0.5) is accepted.
        assert!(transformation_normal_pit_score(0.0, -1.0, 1.0, 0.25).is_ok());
    }

    #[test]
    fn pit_rejects_nonfinite_h_lower_upper() {
        assert!(transformation_normal_pit_score(f64::NAN, -1.0, 1.0, EPS).is_err());
        assert!(transformation_normal_pit_score(0.0, f64::NEG_INFINITY, 1.0, EPS).is_err());
        assert!(transformation_normal_pit_score(0.0, -1.0, f64::INFINITY, EPS).is_err());
    }

    #[test]
    fn pit_rejects_endpoint_order_violation() {
        // upper <= lower is a monotonicity violation.
        assert!(transformation_normal_pit_score(0.0, 1.0, 1.0, EPS).is_err());
        assert!(transformation_normal_pit_score(0.0, 1.0, 0.5, EPS).is_err());
    }

    // ---- transformation_normal_pit_score: closed-form values ----

    #[test]
    fn pit_symmetric_midpoint_maps_to_zero() {
        // lower = -upper and h = 0: by normal symmetry the conditional CDF is
        // exactly 0.5, and Phi^{-1}(0.5) = 0.
        let u = transformation_normal_pit_score(0.0, -2.0, 2.0, EPS).unwrap();
        assert!(u.abs() < 1e-9, "expected ~0, got {u}");
    }

    #[test]
    fn pit_at_or_below_lower_clamps_to_low_quantile() {
        // h <= lower -> u = 0 -> clamped to clip_eps -> Phi^{-1}(clip_eps).
        let clip = 1e-6;
        let expected = standard_normal_quantile(clip).unwrap();
        let at = transformation_normal_pit_score(-1.0, -1.0, 1.0, clip).unwrap();
        let below = transformation_normal_pit_score(-1.5, -1.0, 1.0, clip).unwrap();
        assert!((at - expected).abs() < 1e-12);
        assert!((below - expected).abs() < 1e-12);
        // Low quantile is a large negative number.
        assert!(expected < -3.0);
    }

    #[test]
    fn pit_at_or_above_upper_clamps_to_high_quantile() {
        // h >= upper -> u = 1 -> clamped to 1 - clip_eps -> Phi^{-1}(1 - clip_eps).
        let clip = 1e-6;
        let expected = standard_normal_quantile(1.0 - clip).unwrap();
        let at = transformation_normal_pit_score(1.0, -1.0, 1.0, clip).unwrap();
        let above = transformation_normal_pit_score(2.0, -1.0, 1.0, clip).unwrap();
        assert!((at - expected).abs() < 1e-12);
        assert!((above - expected).abs() < 1e-12);
        assert!(expected > 3.0);
    }

    #[test]
    fn pit_is_monotone_increasing_in_h() {
        // Strictly inside the support the PIT score increases with h.
        let clip = 1e-9;
        let a = transformation_normal_pit_score(-0.5, -2.0, 2.0, clip).unwrap();
        let b = transformation_normal_pit_score(0.0, -2.0, 2.0, clip).unwrap();
        let c = transformation_normal_pit_score(0.5, -2.0, 2.0, clip).unwrap();
        assert!(a < b && b < c, "not monotone: {a} {b} {c}");
    }

    // ---- scop_second_order_h: pure accumulator closed forms ----

    #[test]
    fn scop_second_order_h_p_resp_one_is_location_only() {
        // With p_resp = 1 the shape loop never runs: every output is the location
        // (column 0) basis value times the matching gamma directional derivative.
        let rv = array![3.0];
        let rd = array![5.0];
        let gamma = [0.0];
        let gi = [2.0];
        let gj = [7.0];
        let gij = [11.0];
        let out = scop_second_order_h(rv.view(), rd.view(), 1, &gamma, &gi, &gj, &gij);
        // [h_i, h_j, h_ij, hp_i, hp_j, hp_ij]
        assert_eq!(
            out,
            [
                3.0 * 2.0,
                3.0 * 7.0,
                3.0 * 11.0,
                5.0 * 2.0,
                5.0 * 7.0,
                5.0 * 11.0
            ]
        );
    }

    #[test]
    fn scop_second_order_h_p_resp_two_matches_hand_formula() {
        // p_resp = 2: one shape column (k = 1) contributes the squared-coefficient
        // chain terms. Hand-derive each accumulator from the source.
        let rv = array![1.0, 4.0];
        let rd = array![1.0, 6.0];
        let gamma = [0.0, 2.0];
        let gi = [1.0, 3.0];
        let gj = [1.0, 5.0];
        let gij = [1.0, 7.0];
        let out = scop_second_order_h(rv.view(), rd.view(), 2, &gamma, &gi, &gj, &gij);
        // h_i = rv0*gi0 + 2*rv1*g1*gi1 = 1 + 2*4*2*3 = 49
        let h_i = 1.0 + 2.0 * 4.0 * 2.0 * 3.0;
        // h_j = rv0*gj0 + 2*rv1*g1*gj1 = 1 + 2*4*2*5 = 81
        let h_j = 1.0 + 2.0 * 4.0 * 2.0 * 5.0;
        // h_ij = rv0*gij0 + 2*rv1*(gj1*gi1 + g1*gij1) = 1 + 2*4*(5*3 + 2*7) = 1 + 8*29 = 233
        let h_ij = 1.0 + 2.0 * 4.0 * (5.0 * 3.0 + 2.0 * 7.0);
        // hp_i = rd0*gi0 + 2*rd1*g1*gi1 = 1 + 2*6*2*3 = 73
        let hp_i = 1.0 + 2.0 * 6.0 * 2.0 * 3.0;
        let hp_j = 1.0 + 2.0 * 6.0 * 2.0 * 5.0;
        let hp_ij = 1.0 + 2.0 * 6.0 * (5.0 * 3.0 + 2.0 * 7.0);
        assert_eq!(out, [h_i, h_j, h_ij, hp_i, hp_j, hp_ij]);
    }

    // ---- scop_second_order_endpoints ----

    #[test]
    fn scop_second_order_endpoints_matches_hand_formula() {
        // Two endpoints (lower/upper bases). p_resp = 2 so one shape term per endpoint.
        let lower = [1.0, 2.0];
        let upper = [3.0, 4.0];
        let gamma = [0.0, 5.0];
        let gi = [1.0, 6.0];
        let gj = [1.0, 7.0];
        let gij = [1.0, 8.0];
        let (ei, ej, eij) =
            scop_second_order_endpoints([&lower, &upper], 2, &gamma, &gi, &gj, &gij);
        // endpoint_i[e] = basis0*gi0 + 2*basis1*g1*gi1
        assert_eq!(ei[0], 1.0 * 1.0 + 2.0 * 2.0 * 5.0 * 6.0); // lower: 1 + 120 = 121
        assert_eq!(ei[1], 3.0 * 1.0 + 2.0 * 4.0 * 5.0 * 6.0); // upper: 3 + 240 = 243
        assert_eq!(ej[0], 1.0 * 1.0 + 2.0 * 2.0 * 5.0 * 7.0); // 1 + 140 = 141
        assert_eq!(ej[1], 3.0 * 1.0 + 2.0 * 4.0 * 5.0 * 7.0); // 3 + 280 = 283
        // endpoint_ij[e] = basis0*gij0 + 2*basis1*(gj1*gi1 + g1*gij1)
        assert_eq!(eij[0], 1.0 * 1.0 + 2.0 * 2.0 * (7.0 * 6.0 + 5.0 * 8.0)); // 1 + 4*82 = 329
        assert_eq!(eij[1], 3.0 * 1.0 + 2.0 * 4.0 * (7.0 * 6.0 + 5.0 * 8.0)); // 3 + 8*82 = 659
    }

    // ---- scop_psi_marginal ----

    #[test]
    fn scop_psi_marginal_matches_hand_formula() {
        let rv = array![1.0, 4.0];
        let rd = array![1.0, 6.0];
        let lower = [1.0, 2.0];
        let upper = [3.0, 4.0];
        let gamma = [0.0, 5.0];
        let gamma_psi = [9.0, 10.0];
        let (h_psi, hp_psi, endpoint_psi) = scop_psi_marginal(
            rv.view(),
            rd.view(),
            2,
            [&lower, &upper],
            &gamma,
            &gamma_psi,
        );
        // h_psi = rv0*gpsi0 + 2*rv1*g1*gpsi1 = 9 + 2*4*5*10 = 409
        assert_eq!(h_psi, 9.0 + 2.0 * 4.0 * 5.0 * 10.0);
        // hp_psi = rd0*gpsi0 + 2*rd1*g1*gpsi1 = 9 + 2*6*5*10 = 609
        assert_eq!(hp_psi, 9.0 + 2.0 * 6.0 * 5.0 * 10.0);
        // endpoint_psi[e] = basis0*gpsi0 + 2*basis1*g1*gpsi1
        assert_eq!(endpoint_psi[0], 1.0 * 9.0 + 2.0 * 2.0 * 5.0 * 10.0); // 9 + 200 = 209
        assert_eq!(endpoint_psi[1], 3.0 * 9.0 + 2.0 * 4.0 * 5.0 * 10.0); // 27 + 400 = 427
    }
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
