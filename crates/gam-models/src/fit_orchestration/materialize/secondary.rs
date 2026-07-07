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
/// via knot counts, are left untouched by the *center* cap.
///
/// Separately (#1561), this also defaults the Marra & Wood null-space "double"
/// penalty OFF for a secondary smooth. That penalty is a term-*selection* device:
/// it shrinks a smooth toward its constant null space so a whole term can be
/// penalized out of the model. Defaulting it ON for a distributional predictor —
/// whose entire purpose is to model variation in that parameter — biases the fit
/// toward homoscedasticity and collapses the recovered log-sigma surface (the
/// #1561 over-smoothing). mgcv's `gaulss` defaults `select=FALSE` for exactly this
/// reason, and gam already defaults it off for the `sz` deviation smooth (gam#700);
/// this extends the same principle to the scale block's ordinary smooths. Only the
/// bases that DEFAULT the Marra & Wood penalty on (`bspline`/`tps`/`matern`) are
/// affected; `duchon` is excluded because it has no such penalty to disable and its
/// builder rejects a `double_penalty=` key outright. An explicit user
/// `double_penalty=` always wins, as does the mgcv shrinkage spline alias
/// `bs="cs"`, because that alias is itself an explicit request for null-space
/// shrinkage rather than a default inherited from `s()`.
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

            // #1561: drop the null-space double penalty by default on the scale /
            // distributional block so REML can resolve genuine parameter variation
            // instead of over-shrinking it toward the homoscedastic null space.
            // `duchon` is intentionally excluded: it carries no Marra & Wood
            // null-space double penalty to turn off (it ships its own
            // reproducing-norm penalty plus a null-space ridge) and its builder
            // rejects a `double_penalty=` key outright, so injecting one here
            // would abort an otherwise-valid scale-block Duchon fit.
            if secondary_smooth_should_disable_default_double_penalty(&canonical, options) {
                options.insert("double_penalty".to_string(), "false".to_string());
            }

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


fn secondary_smooth_should_disable_default_double_penalty(
    canonical: &str,
    options: &BTreeMap<String, String>,
) -> bool {
    if options.contains_key("double_penalty") {
        return false;
    }

    let selector = options
        .get("bs")
        .or_else(|| options.get("type"))
        .map(|raw| raw.trim().to_ascii_lowercase());
    if matches!(selector.as_deref(), Some("cs")) {
        // `cs` is mgcv's shrinkage cubic-regression spline. It is not a silent
        // default double penalty; it is the user's basis choice, equivalent to
        // selecting the cubic-regression span plus null-space shrinkage.
        return false;
    }

    matches!(canonical, "bspline" | "tps" | "matern")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn secondary_double_penalty_default_off_preserves_explicit_cs_shrinkage() {
        assert!(secondary_smooth_should_disable_default_double_penalty(
            "bspline",
            &opts(&[])
        ));
        assert!(!secondary_smooth_should_disable_default_double_penalty(
            "bspline",
            &opts(&[("bs", "cs")])
        ));
        assert!(!secondary_smooth_should_disable_default_double_penalty(
            "bspline",
            &opts(&[("double_penalty", "true")])
        ));
        assert!(!secondary_smooth_should_disable_default_double_penalty(
            "duchon",
            &opts(&[])
        ));
    }
}
