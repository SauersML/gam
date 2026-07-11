use super::*;

/// Apply basis parsimony to a *secondary* (distributional) predictor's smooths.
///
/// In a location-scale / GAMLSS fit the mean is identified directly by the
/// response and warrants the generous default basis, but the scale (log-sigma)
/// and other distributional predictors are identified only through (noisy)
/// squared residuals. Handing their radial spatial smooths a basis sized for the
/// mean lets REML over-fit them (#501). For each spatial smooth (thin-plate /
/// Matern / Duchon) the user did not size explicitly, cap the *default* center
/// count via the private [`SECONDARY_CENTER_CAP_OPTION`]. The cap lowers the
/// default while preserving the `Auto` center strategy, so the basis is still
/// softly reduced when the data can't support the count (rather than erroring
/// like an explicit count would). Smooths the user sized explicitly, and the
/// non-radial bases (B-spline, cyclic, tensor) which already default modestly
/// via knot counts, are left untouched by the *center* cap. Penalty topology is
/// likewise left to the ordinary term builder: secondary predictors retain the
/// project-wide null-recovery default, while an explicit `double_penalty=false`
/// remains the opt-out.
pub(super) fn apply_secondary_predictor_basis_parsimony(terms: &mut [ParsedTerm], n_rows: usize) {
    for term in terms.iter_mut() {
        if let ParsedTerm::Smooth {
            vars,
            kind,
            options,
            ..
        } = term
        {
            let canonical = resolve_smooth_type_name(*kind, vars.len(), options);

            if !smooth_type_uses_spatial_center_heuristic(&canonical)
                || has_explicit_countwith_basis_alias(options, "centers")
            {
                continue;
            }
            let cap = gam_terms::basis::conservative_secondary_centers(n_rows, vars.len());
            options.insert(SECONDARY_CENTER_CAP_OPTION.to_string(), cap.to_string());
        }
    }
}
