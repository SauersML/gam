use super::*;

pub(crate) fn spatial_basiswarning_family_and_cols(
    term: &SmoothTermSpec,
) -> Option<(&'static str, &[usize])> {
    spatial_basiswarning_family_and_cols_basis(&term.basis)
}

pub(crate) fn spatial_basiswarning_family_and_cols_basis(
    basis: &SmoothBasisSpec,
) -> Option<(&'static str, &[usize])> {
    match basis {
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            spatial_basiswarning_family_and_cols_basis(inner)
        }
        SmoothBasisSpec::BySmooth { smooth, .. } => {
            spatial_basiswarning_family_and_cols_basis(smooth)
        }
        SmoothBasisSpec::ThinPlate { feature_cols, .. } => Some(("thinplate/tps", feature_cols)),
        SmoothBasisSpec::Sphere { feature_cols, .. } => Some(("sphere/sos", feature_cols)),
        SmoothBasisSpec::ConstantCurvature { feature_cols, .. } => {
            Some(("constant_curvature", feature_cols))
        }
        SmoothBasisSpec::Matern { feature_cols, .. } => Some(("matern", feature_cols)),
        SmoothBasisSpec::MeasureJet { feature_cols, .. } => Some(("measurejet", feature_cols)),
        SmoothBasisSpec::Duchon { feature_cols, .. } => Some(("duchon", feature_cols)),
        SmoothBasisSpec::BSpline1D { .. }
        | SmoothBasisSpec::Pca { .. }
        | SmoothBasisSpec::TensorBSpline { .. }
        | SmoothBasisSpec::FactorSmooth { .. } => None,
    }
}

pub(crate) fn collect_spatial_smooth_usagewarnings(
    spec: &TermCollectionSpec,
    headers: &[String],
    label: &str,
) -> Vec<String> {
    let mut grouped: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
    for term in &spec.smooth_terms {
        let Some((family, feature_cols)) = spatial_basiswarning_family_and_cols(term) else {
            continue;
        };
        if feature_cols.len() != 1 {
            continue;
        }
        let col = feature_cols[0];
        let featurename = headers
            .get(col)
            .cloned()
            .unwrap_or_else(|| format!("#{col}"));
        grouped.entry(family).or_default().push(featurename);
    }

    grouped
        .into_iter()
        .filter_map(|(family, cols)| {
            if cols.len() < 2 {
                return None;
            }
            // `spatial_basiswarning_family_and_cols` returns one of these four
            // family strings; any other value is filtered out by returning None.
            let example = match family {
                "thinplate/tps" => format!("thinplate({})", cols.join(", ")),
                "matern" => format!("matern({})", cols.join(", ")),
                "duchon" => format!("duchon({})", cols.join(", ")),
                "sphere/sos" => format!("sphere({})", cols.join(", ")),
                _ => return None,
            };
            let bad_example = match family {
                "thinplate/tps" => cols
                    .iter()
                    .map(|col| format!("s({col}, type=tps)"))
                    .collect::<Vec<_>>()
                    .join(" + "),
                "matern" => cols
                    .iter()
                    .map(|col| format!("s({col}, type=matern)"))
                    .collect::<Vec<_>>()
                    .join(" + "),
                "duchon" => cols
                    .iter()
                    .map(|col| format!("s({col}, type=duchon)"))
                    .collect::<Vec<_>>()
                    .join(" + "),
                "sphere/sos" => cols
                    .iter()
                    .map(|col| format!("s({col}, type=sphere)"))
                    .collect::<Vec<_>>()
                    .join(" + "),
                _ => return None,
            };
            Some(format!(
                "{label}: detected {} separate 1D {family} spatial smooths over [{}]. These build unrelated additive 1D smooths, not one shared spatial manifold. TIP: if you intended one spatial surface, replace `{bad_example}` with one multivariate term such as `{example}`.",
                cols.len(),
                cols.join(", "),
            ))
        })
        .collect()
}

pub(crate) fn collect_linear_smooth_overlapwarnings(
    spec: &TermCollectionSpec,
    headers: &[String],
    label: &str,
) -> Vec<String> {
    let linear_by_col = spec
        .linear_terms
        .iter()
        .map(|term| (term.feature_col, term.name.as_str()))
        .collect::<BTreeMap<_, _>>();
    let mut warnings = Vec::new();
    for smooth in &spec.smooth_terms {
        let overlaps = smooth_term_feature_cols(smooth)
            .into_iter()
            .filter_map(|col| {
                linear_by_col.get(&col).map(|linearname| {
                    let featurename = headers
                        .get(col)
                        .cloned()
                        .unwrap_or_else(|| format!("#{col}"));
                    (featurename, (*linearname).to_string())
                })
            })
            .collect::<Vec<_>>();
        if overlaps.is_empty() {
            continue;
        }
        let overlap_features = overlaps
            .iter()
            .map(|(featurename, _)| featurename.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let linear_terms = overlaps
            .iter()
            .map(|(_, linearname)| format!("linear({linearname})"))
            .collect::<Vec<_>>()
            .join(" + ");
        warnings.push(format!(
            "{label}: feature(s) [{overlap_features}] appear both in smooth term `{}` and explicit linear term(s) `{linear_terms}`. The fit now residualizes the smooth against the intercept and those overlapping linear columns, so the smooth contributes only the nonlinear remainder on those variables. This changes the term decomposition and interpretation.",
            smooth.name
        ));
    }
    warnings
}

pub(crate) fn collect_hierarchical_smooth_overlapwarnings(
    spec: &TermCollectionSpec,
    headers: &[String],
    label: &str,
) -> Vec<String> {
    let feature_label = |col: usize| {
        headers
            .get(col)
            .cloned()
            .unwrap_or_else(|| format!("#{col}"))
    };
    let join_feature_labels = |cols: &[usize]| {
        cols.iter()
            .map(|&col| feature_label(col))
            .collect::<Vec<_>>()
            .join(", ")
    };

    let SmoothStructureAnalysis {
        ownership_order,
        term_feature_cols,
        term_owners,
        ..
    } = analyze_smooth_ownership(&spec.smooth_terms);

    let mut warnings = Vec::new();
    for &target_idx in &ownership_order {
        let owners = &term_owners[target_idx];
        if owners.is_empty() {
            continue;
        }
        let target = &spec.smooth_terms[target_idx];
        let target_features = join_feature_labels(&term_feature_cols[target_idx]);
        let owner_descriptions = owners
            .iter()
            .map(|&owner_idx| {
                format!(
                    "`{}` over [{}]",
                    spec.smooth_terms[owner_idx].name,
                    join_feature_labels(&term_feature_cols[owner_idx]),
                )
            })
            .collect::<Vec<_>>()
            .join(", ");

        warnings.push(format!(
            "{label}: smooth term `{}` over [{target_features}] overlaps nested or duplicate smooth term(s) {}. The fit uses automatic hierarchical ownership: those higher-priority smooth term(s) keep any shared realized subspace, and `{}` is residualized against that overlap before fitting.",
            target.name,
            owner_descriptions,
            target.name,
        ));
    }
    warnings
}

pub(crate) fn collect_smooth_structure_warnings(
    spec: &TermCollectionSpec,
    headers: &[String],
    label: &str,
) -> Vec<String> {
    let mut warnings = collect_spatial_smooth_usagewarnings(spec, headers, label);
    warnings.extend(collect_linear_smooth_overlapwarnings(spec, headers, label));
    warnings.extend(collect_hierarchical_smooth_overlapwarnings(
        spec, headers, label,
    ));
    warnings
}

pub(crate) fn emit_smooth_structure_warnings(stage: &str, warnings: &[String]) {
    for warning in warnings {
        cli_err!("WARNING [{stage}]: {warning}");
    }
}

/// Build anisotropic spatial-geometry report rows from an optional resolved spec.
pub(crate) fn build_anisotropic_scales_rows(
    spec: Option<&TermCollectionSpec>,
) -> Vec<report::AnisotropicScalesRow> {
    use gam::smooth::get_spatial_aniso_log_scales;
    use gam::terms::smooth::get_spatial_length_scale;
    let Some(spec) = spec else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for (term_idx, term) in spec.smooth_terms.iter().enumerate() {
        let Some(eta) = get_spatial_aniso_log_scales(spec, term_idx) else {
            continue;
        };
        if eta.is_empty() {
            continue;
        }
        let ls = get_spatial_length_scale(spec, term_idx);
        let axes = eta
            .iter()
            .enumerate()
            .map(|(a, &eta_a)| {
                let (length_a, kappa_a) = if let Some(ls) = ls {
                    (Some(ls * (-eta_a).exp()), Some((1.0 / ls) * eta_a.exp()))
                } else {
                    (None, None)
                };
                (a, eta_a, length_a, kappa_a)
            })
            .collect();
        rows.push(report::AnisotropicScalesRow {
            term_name: term.name.clone(),
            global_length_scale: ls,
            axes,
        });
    }
    rows
}

/// Build measure-jet spectrum report rows from a saved (frozen) spec alone:
/// realized band + spec order, no per-scale λ̂s (those need the rebuilt
/// design's penalty layout). Used when the report runs without a dataset.
pub(crate) fn measure_jet_spectrum_rows_from_spec(
    spec: Option<&TermCollectionSpec>,
) -> Vec<report::MeasureJetSpectrumRow> {
    let Some(spec) = spec else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for term in &spec.smooth_terms {
        let SmoothBasisSpec::MeasureJet { spec: mj, .. } = &term.basis else {
            continue;
        };
        let Some(frozen) = mj.frozen_quadrature.as_ref() else {
            continue;
        };
        let (Some(&eps_min), Some(&eps_max)) = (frozen.eps_band.first(), frozen.eps_band.last())
        else {
            continue;
        };
        rows.push(report::MeasureJetSpectrumRow {
            term_name: term.name.clone(),
            eps_min,
            eps_max,
            n_scales: frozen.eps_band.len(),
            length_scale: mj.length_scale,
            spec_order_s: mj.order_s,
            per_scale: Vec::new(),
            implied_order: None,
        });
    }
    rows
}

/// Implied continuous order from a measure-jet raw-form per-scale λ spectrum:
/// ŝ = −½ · (least-squares slope of ln λ̂_ℓ on ln ε_ℓ). `None` unless at
/// least two scales carry finite positive (ε_ℓ, λ̂_ℓ) and the band has
/// nonzero log-spread.
pub(crate) fn measure_jet_implied_order(per_scale: &[(f64, f64)]) -> Option<f64> {
    let pts: Vec<(f64, f64)> = per_scale
        .iter()
        .filter(|&&(eps, lam)| eps.is_finite() && eps > 0.0 && lam.is_finite() && lam > 0.0)
        .map(|&(eps, lam)| (eps.ln(), lam.ln()))
        .collect();
    if pts.len() < 2 {
        return None;
    }
    let n = pts.len() as f64;
    let xbar = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let ybar = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx = pts.iter().map(|p| (p.0 - xbar).powi(2)).sum::<f64>();
    if sxx <= 0.0 {
        return None;
    }
    let sxy = pts.iter().map(|p| (p.0 - xbar) * (p.1 - ybar)).sum::<f64>();
    let s_hat = -0.5 * (sxy / sxx);
    s_hat.is_finite().then_some(s_hat)
}

/// Print learned per-axis spatial anisotropy for spatial terms to stdout.
pub(crate) fn print_spatial_aniso_scales(spec: &TermCollectionSpec) {
    use gam::smooth::get_spatial_aniso_log_scales;
    use gam::terms::smooth::get_spatial_length_scale;
    for (term_idx, term) in spec.smooth_terms.iter().enumerate() {
        let Some(eta) = get_spatial_aniso_log_scales(spec, term_idx) else {
            continue;
        };
        if eta.is_empty() {
            continue;
        }
        let ls = get_spatial_length_scale(spec, term_idx);
        match ls {
            Some(ls) => cli_out!(
                "[spatial-kappa] term {} (\"{}\"): anisotropic length scales (global length_scale={:.4})",
                term_idx,
                term.name,
                ls
            ),
            None => cli_out!(
                "[spatial-kappa] term {} (\"{}\"): pure Duchon shape anisotropy",
                term_idx,
                term.name
            ),
        }
        for (a, &eta_a) in eta.iter().enumerate() {
            if let Some(ls) = ls {
                let length_a = ls * (-eta_a).exp();
                let kappa_a = (1.0 / ls) * eta_a.exp();
                cli_out!(
                    "  axis {}: eta={:+.4}, length={:.4}, kappa={:.4}",
                    a,
                    eta_a,
                    length_a,
                    kappa_a
                );
            } else {
                cli_out!("  axis {}: eta={:+.4}", a, eta_a);
            }
        }
    }
}
