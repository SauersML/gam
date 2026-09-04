use super::WorkingModelPirlsOptions;

/// The rounding band of a penalized objective `½·(s·D + βᵀSβ)` of magnitude
/// `magnitude`, evaluated over `n` rows and `p` coefficients.
///
/// The deviance accumulates `n` per-row terms and the penalty `p²` products, so
/// two evaluations that differ by less than `γ_{n+p²}·u·magnitude` differ by
/// arithmetic, not by progress (`γ_k` is Wilkinson's growth factor for a
/// `k`-term accumulation, `u` the unit roundoff). Exact when the accumulated
/// terms share a sign — a GLM deviance and a PSD penalty — and a LOWER bound
/// when a likelihood's per-row terms cancel (a survival log-density can be
/// positive), which is the safe direction: every certificate built on this band
/// can refuse more, never certify more. A Jeffreys term is a `p`-term log-det
/// on top and is covered by the `p²` count.
#[inline]
pub(super) fn objective_rounding_band(n: usize, p: usize, magnitude: f64) -> f64 {
    gam_linalg::roundoff::accumulation_growth(n + p * p)
        * gam_linalg::roundoff::UNIT_ROUNDOFF
        * magnitude.abs()
}

/// Compute the effective KKT convergence tolerance, honouring the optional
/// adaptive schedule when its parameters are all finite and ordered.
#[inline]
pub(super) fn effective_kkt_tolerance(options: &WorkingModelPirlsOptions) -> f64 {
    match options.adaptive_kkt_tolerance {
        Some(adaptive)
            if adaptive.eta.is_finite()
                && adaptive.floor.is_finite()
                && adaptive.ceiling.is_finite()
                && adaptive.outer_grad_norm.is_finite()
                && adaptive.eta >= 0.0
                && adaptive.floor > 0.0
                && adaptive.ceiling >= adaptive.floor
                && adaptive.outer_grad_norm >= 0.0 =>
        {
            (adaptive.eta * adaptive.outer_grad_norm).clamp(adaptive.floor, adaptive.ceiling)
        }
        _ => options.convergence_tolerance,
    }
}
