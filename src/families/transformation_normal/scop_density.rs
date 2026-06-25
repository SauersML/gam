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
/// derivative bases and the per-response-knot gamma directional derivatives.
/// Shared verbatim across the SCOP Hessian/HVP/bilinear row loops.
pub(crate) fn scop_second_order_h(
    rv: ArrayView1<'_, f64>,
    rd: ArrayView1<'_, f64>,
    p_resp: usize,
    gamma: &[f64],
    gamma_i: &[f64],
    gamma_j: &[f64],
    gamma_ij: &[f64],
) -> [f64; 6] {
    let mut h_i = rv[0] * gamma_i[0];
    let mut h_j = rv[0] * gamma_j[0];
    let mut h_ij = rv[0] * gamma_ij[0];
    let mut hp_i = rd[0] * gamma_i[0];
    let mut hp_j = rd[0] * gamma_j[0];
    let mut hp_ij = rd[0] * gamma_ij[0];
    for k in 1..p_resp {
        let g = gamma[k];
        let gi = gamma_i[k];
        let gj = gamma_j[k];
        let gij = gamma_ij[k];
        h_i += 2.0 * rv[k] * g * gi;
        h_j += 2.0 * rv[k] * g * gj;
        h_ij += 2.0 * rv[k] * (gj * gi + g * gij);
        hp_i += 2.0 * rd[k] * g * gi;
        hp_j += 2.0 * rd[k] * g * gj;
        hp_ij += 2.0 * rd[k] * (gj * gi + g * gij);
    }
    [h_i, h_j, h_ij, hp_i, hp_j, hp_ij]
}

/// Accumulates the second-order endpoint-normalizer chain inputs
/// `(endpoint_i, endpoint_j, endpoint_ij)` for one row. Shared verbatim across
/// the SCOP Hessian/HVP/bilinear row loops.
pub(crate) fn scop_second_order_endpoints(
    endpoint_basis: [&[f64]; 2],
    p_resp: usize,
    gamma: &[f64],
    gamma_i: &[f64],
    gamma_j: &[f64],
    gamma_ij: &[f64],
) -> ([f64; 2], [f64; 2], [f64; 2]) {
    let mut endpoint_i = [0.0; 2];
    let mut endpoint_j = [0.0; 2];
    let mut endpoint_ij = [0.0; 2];
    for e in 0..2 {
        let basis = endpoint_basis[e];
        endpoint_i[e] = basis[0] * gamma_i[0];
        endpoint_j[e] = basis[0] * gamma_j[0];
        endpoint_ij[e] = basis[0] * gamma_ij[0];
        for k in 1..p_resp {
            endpoint_i[e] += 2.0 * basis[k] * gamma[k] * gamma_i[k];
            endpoint_j[e] += 2.0 * basis[k] * gamma[k] * gamma_j[k];
            endpoint_ij[e] += 2.0 * basis[k] * (gamma_j[k] * gamma_i[k] + gamma[k] * gamma_ij[k]);
        }
    }
    (endpoint_i, endpoint_j, endpoint_ij)
}

/// Accumulates the psi-direction transform quantities `(h_psi, hp_psi,
/// endpoint_psi)` for one row from the response bases and the per-knot psi
/// directional derivatives. Shared verbatim across the SCOP psi setup loops.
pub(crate) fn scop_psi_marginal(
    rv: ArrayView1<'_, f64>,
    rd: ArrayView1<'_, f64>,
    p_resp: usize,
    endpoint_basis: [&[f64]; 2],
    gamma: &[f64],
    gamma_psi: &[f64],
) -> (f64, f64, [f64; 2]) {
    let mut h_psi = rv[0] * gamma_psi[0];
    let mut hp_psi = rd[0] * gamma_psi[0];
    for k in 1..p_resp {
        h_psi += 2.0 * rv[k] * gamma[k] * gamma_psi[k];
        hp_psi += 2.0 * rd[k] * gamma[k] * gamma_psi[k];
    }

    let mut endpoint_psi = [0.0; 2];
    for e in 0..2 {
        let basis = endpoint_basis[e];
        endpoint_psi[e] = basis[0] * gamma_psi[0];
        for k in 1..p_resp {
            endpoint_psi[e] += 2.0 * basis[k] * gamma[k] * gamma_psi[k];
        }
    }
    (h_psi, hp_psi, endpoint_psi)
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
