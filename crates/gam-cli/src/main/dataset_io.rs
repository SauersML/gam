use super::*;

pub(crate) fn load_dataset_projected(
    path: &Path,
    requested_columns: &[String],
) -> Result<Dataset, gam::data::DataError> {
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
            ParsedTerm::RandomEffect { name, .. } => {
                out.insert(name.clone());
            }
            ParsedTerm::SlopeSurface { terms, .. } => {
                collect_categorical_role_columns(terms, out);
            }
            // Deliberately not categorical-by-role — see the doc comment above
            // for why bare linear terms and smooth arguments are excluded.
            ParsedTerm::Linear { .. }
            | ParsedTerm::BoundedLinear { .. }
            | ParsedTerm::Smooth { .. }
            | ParsedTerm::Interaction { .. }
            | ParsedTerm::LinkWiggle { .. }
            | ParsedTerm::TimeWiggle { .. }
            | ParsedTerm::LinkConfig { .. }
            | ParsedTerm::SurvivalConfig { .. }
            | ParsedTerm::NoIntercept => {}
        }
    }
}

/// Build the categorical-role column set for a fit (random-effect grouping
/// columns plus a categorical/multinomial response) and load the dataset with
/// those columns forced to a factor encoding. See
/// [`collect_categorical_role_columns`] and
/// `gam::data::load_dataset_projected_with_categorical_roles`.
pub(crate) fn load_fit_dataset_with_roles(
    path: &Path,
    requested_columns: &[String],
    parsed: &ParsedFormula,
    response_is_categorical: bool,
) -> Result<Dataset, gam::data::DataError> {
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
