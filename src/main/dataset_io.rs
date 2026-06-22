use super::*;

pub(crate) fn collect_term_column_names(terms: &[ParsedTerm], out: &mut BTreeSet<String>) {
    // Delegate to the single shared authority on the formula→columns walk
    // (`s(x, by=g)`'s `by` column is included there) so the fit-time required
    // columns, the predict-time required columns, and the PyFFI surface all
    // agree.
    parsed_term_column_names(terms, out);
}

pub(crate) fn required_columns_for_formula(parsed: &ParsedFormula) -> Result<Vec<String>, String> {
    let mut out = BTreeSet::<String>::new();
    if let Some((entry, exit, event)) = parse_surv_response(&parsed.response)? {
        if let Some(entry) = entry {
            out.insert(entry);
        }
        out.insert(exit);
        out.insert(event);
    } else if let Some((left, right, event)) = parse_surv_interval_response(&parsed.response)? {
        out.insert(left);
        out.insert(right);
        out.insert(event);
    } else {
        out.insert(parsed.response.clone());
    }
    collect_term_column_names(&parsed.terms, &mut out);
    for surface in &parsed.logslope_surfaces {
        out.insert(surface.z_column.clone());
        collect_term_column_names(&surface.terms, &mut out);
    }
    Ok(out.into_iter().collect())
}

pub(crate) fn merge_required_columns(target: &mut BTreeSet<String>, cols: Vec<String>) {
    target.extend(cols);
}

pub(crate) fn required_columns_for_fit(
    args: &FitArgs,
    parsed: &ParsedFormula,
) -> Result<Vec<String>, String> {
    let mut required = BTreeSet::<String>::new();
    merge_required_columns(&mut required, required_columns_for_formula(parsed)?);

    if let Some(noise_formula_raw) = args.predict_noise.as_deref() {
        let (_, parsed_noise) = parse_matching_auxiliary_formula(
            noise_formula_raw,
            &parsed.response,
            "--predict-noise",
        )?;
        merge_required_columns(&mut required, required_columns_for_formula(&parsed_noise)?);
    }

    if let Some(logslope_formula_raw) = args.logslope_formula.as_deref() {
        let (_, parsed_logslope) = parse_matching_auxiliary_formula(
            logslope_formula_raw,
            &parsed.response,
            "--logslope-formula",
        )?;
        merge_required_columns(
            &mut required,
            required_columns_for_formula(&parsed_logslope)?,
        );
    }

    if let Some(z_column) = args.z_column.as_ref() {
        required.insert(z_column.clone());
    }
    if let Some(weights_column) = args.weights_column.as_ref() {
        required.insert(weights_column.clone());
    }
    if let Some(offset_column) = args.offset_column.as_ref() {
        required.insert(offset_column.clone());
    }
    if let Some(noise_offset_column) = args.noise_offset_column.as_ref() {
        required.insert(noise_offset_column.clone());
    }
    Ok(required.into_iter().collect())
}

/// Format a `Surv(...)` response expression, omitting the entry argument
/// when the right-censored shorthand `Surv(time, event)` is in use.
pub(crate) fn surv_response_expr(entry: Option<&str>, exit: &str, event: &str) -> String {
    match entry {
        Some(entry) => format!("Surv({entry}, {exit}, {event})"),
        None => format!("Surv({exit}, {event})"),
    }
}

pub(crate) fn required_columns_for_survival(
    args: &SurvivalArgs,
    parsed: &ParsedFormula,
) -> Result<Vec<String>, String> {
    let mut required = BTreeSet::<String>::new();
    if let Some(entry) = args.entry.as_deref() {
        required.insert(entry.to_string());
    }
    required.insert(args.exit.clone());
    required.insert(args.event.clone());
    merge_required_columns(&mut required, required_columns_for_formula(parsed)?);

    if let Some(noise_formula_raw) = args.predict_noise.as_deref() {
        let response_expr = surv_response_expr(args.entry.as_deref(), &args.exit, &args.event);
        let (_, parsed_noise) =
            parse_matching_auxiliary_formula(noise_formula_raw, &response_expr, "--predict-noise")?;
        merge_required_columns(&mut required, required_columns_for_formula(&parsed_noise)?);
    }

    if let Some(z_column) = args.z_column.as_ref() {
        required.insert(z_column.clone());
    }
    if let Some(weights_column) = args.weights_column.as_ref() {
        required.insert(weights_column.clone());
    }
    if let Some(offset_column) = args.offset_column.as_ref() {
        required.insert(offset_column.clone());
    }
    if let Some(noise_offset_column) = args.noise_offset_column.as_ref() {
        required.insert(noise_offset_column.clone());
    }
    Ok(required.into_iter().collect())
}

pub(crate) fn load_dataset_projected(
    path: &Path,
    requested_columns: &[String],
) -> Result<Dataset, gam::inference::data::DataError> {
    load_dataset_auto_projected(path, requested_columns)
}

/// Collect the columns a parsed formula uses in a *factor-by-construction*
/// role, so the untyped CSV/TSV/parquet-numeric loader can force them to a
/// categorical encoding (the role-based analogue of the typed-frame
/// categorical sentinel).
///
/// Only roles that are factors *regardless of the data's values* are included:
///
/// * `group(g)` / `factor(g)` / `re(g)` random-effect terms
///   ([`ParsedTerm::RandomEffect`]) — a grouping factor by construction.
/// * a categorical / multinomial **response** column, when `response_is_categorical`.
///
/// Deliberately EXCLUDED so a genuinely-continuous integer covariate is never
/// wrongly factorized:
///
/// * bare `+ x` linear terms ([`ParsedTerm::Linear`]) — role-ambiguous between a
///   continuous slope and a factor main effect; the column kind alone
///   disambiguates, exactly as today. A user who means a numeric-coded factor
///   writes `factor(x)`.
/// * smooth arguments `s(x)` / `te(x, z)` and a smooth's `by=` column — a smooth
///   variable is numeric by construction, and `by=` may be a numeric
///   varying-coefficient.
fn collect_categorical_role_columns(terms: &[ParsedTerm], out: &mut BTreeSet<String>) {
    for term in terms {
        match term {
            ParsedTerm::RandomEffect { name } => {
                out.insert(name.clone());
            }
            ParsedTerm::LogSlopeSurface { terms, .. } => {
                collect_categorical_role_columns(terms, out);
            }
            _ => {}
        }
    }
}

/// Build the categorical-role column set for a fit (random-effect grouping
/// columns plus a categorical/multinomial response) and load the dataset with
/// those columns forced to a factor encoding. See
/// [`collect_categorical_role_columns`] and
/// `gam::inference::data::load_dataset_projected_with_categorical_roles`.
pub(crate) fn load_fit_dataset_with_roles(
    path: &Path,
    requested_columns: &[String],
    parsed: &ParsedFormula,
    response_is_categorical: bool,
) -> Result<Dataset, gam::inference::data::DataError> {
    let mut roles = BTreeSet::<String>::new();
    collect_categorical_role_columns(&parsed.terms, &mut roles);
    if response_is_categorical {
        roles.insert(parsed.response.clone());
    }
    let role_refs: std::collections::HashSet<&str> = roles.iter().map(String::as_str).collect();
    load_dataset_auto_projected_with_categorical_roles(path, requested_columns, &role_refs)
}

pub(crate) fn load_datasetwith_model_schema(
    path: &Path,
    model: &SavedModel,
) -> Result<Dataset, String> {
    load_datasetwith_model_schema_extra(path, model, &[])
}

/// Load a dataset for a *post-fit diagnostic* command (diagnose / sample /
/// report) against a fitted model's schema.
///
/// Unlike prediction, diagnostics need the observed response column: residuals,
/// R², posterior likelihoods, and leave-one-out are all statements *about* it.
/// The prediction loader deliberately drops a standard GAM's bare response
/// (#840 / #864), so this variant folds the model's diagnostic-required
/// response back in via [`SavedModel::diagnostic_extra_columns`]. Routing every
/// diagnostic command through here makes it structurally impossible to silently
/// drop the response — the #864 / #882 / #883 failure mode — rather than relying
/// on each command to remember an `extra_required` argument.
pub(crate) fn load_datasetwith_model_schema_for_diagnostics(
    path: &Path,
    model: &SavedModel,
) -> Result<Dataset, String> {
    let extras = model.diagnostic_extra_columns()?;
    load_datasetwith_model_schema_extra(path, model, &extras)
}

/// Load a new-data file against a fitted model's schema, keeping only the
/// columns the model references (plus any `extra_required` ones a caller knows
/// it will resolve by name, e.g. a `--offset-column` override that differs from
/// the model's saved offset).
///
/// A prediction file commonly carries extra ID / label / grouping columns the
/// formula never names; encoding those against the training schema would
/// strict-validate an unrelated categorical and abort on a held-out level
/// (#840). The projected loader selects just the model's input columns (and the
/// extras), erroring only when a genuinely required one is absent and ignoring
/// the rest — matching mgcv / glm semantics and the PyFFI predict path.
pub(crate) fn load_datasetwith_model_schema_extra(
    path: &Path,
    model: &SavedModel,
    extra_required: &[String],
) -> Result<Dataset, String> {
    let schema = model.require_data_schema()?;
    let policy =
        UnseenCategoryPolicy::encode_unknown_for_columns(model.random_effect_group_columns());
    let mut requested: Vec<String> = model
        .prediction_required_columns()?
        .into_iter()
        .collect::<Vec<_>>();
    requested.extend(extra_required.iter().cloned());
    load_dataset_auto_with_schema_projected(path, schema, policy, &requested).map_err(String::from)
}
