use super::WorkingModelPirlsOptions;

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
