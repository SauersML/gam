//! Native composition of a frozen conditional score law and an outcome model.
//! Cross-fitting stores O(n) scores, never an influence Jacobian. The outcome
//! uses ordinary penalized estimation; no orthogonality claim is implied.
use std::collections::{BTreeMap, BTreeSet, HashMap};

use gam_data::{ColumnKindTag, DataSchema, EncodedDataset, SchemaColumn};
use gam_terms::inference::formula_dsl::{parse_formula, parse_surv_response, parsed_term_column_names};
use ndarray::{Array1, Array2, Axis};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};

use crate::fit_orchestration::{CtnStage1Recipe, FitConfig};
use super::model::{FittedModel, FittedModelPayload, PredictModelClass};
use super::model_payload_builders::fit_formula_to_payload;
use super::predict_input::build_transformation_normal_observed_scores;

const SCORE: &str = "__gamfit_ctn_score";

/// A validated native CTN supplied by a caller instead of fitted on these rows.
#[derive(Clone)]
pub struct FrozenCtn(pub Box<FittedModelPayload>);

impl std::fmt::Debug for FrozenCtn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("FrozenCtn").field(&self.0.formula).finish()
    }
}

/// Formula-aware CTN input schema, shared by native fits and both front ends.
pub fn recipe_columns(recipe: &CtnStage1Recipe) -> Result<BTreeSet<String>, String> {
    let parsed = parse_formula(&format!("{} ~ {}", recipe.response_column, recipe.covariate_formula_rhs))
        .map_err(|error| error.to_string())?;
    let mut names = BTreeSet::from([recipe.response_column.clone()]);
    parsed_term_column_names(&parsed.terms, &mut names);
    names.extend(recipe.weight_column.iter().cloned());
    names.extend(recipe.offset_column.iter().cloned());
    Ok(names)
}

/// Required input columns for CTN composition, resolved by the actual DSL parser.
pub fn required_fit_columns(formula: &str, config: &FitConfig) -> Result<BTreeSet<String>, String> {
    let parsed = parse_formula(formula).map_err(|error| error.to_string())?;
    let mut names = BTreeSet::new();
    parsed_term_column_names(&parsed.terms, &mut names);
    if let Some((entry, exit, event)) = parse_surv_response(&parsed.response)? {
        names.extend(entry); names.insert(exit); names.insert(event);
    } else { names.insert(parsed.response.clone()); }
    for rhs in [config.slope_formula.as_ref(), config.noise_formula.as_ref()].into_iter().flatten() {
        if rhs == "same-as-main" { continue; }
        let text = if rhs.contains('~') { rhs.clone() } else { format!("{} ~ {rhs}", parsed.response) };
        let auxiliary = parse_formula(&text).map_err(|error| error.to_string())?;
        parsed_term_column_names(&auxiliary.terms, &mut names);
    }
    names.extend(config.weight_column.iter().cloned());
    names.extend(config.offset_column.iter().cloned());
    names.extend(config.noise_offset_column.iter().cloned());
    if let Some(recipe) = config.ctn_stage1.as_ref() {
        names.extend(recipe_columns(recipe)?);
        names.extend(recipe.fold_column.iter().cloned());
        names.extend(recipe.group_column.iter().cloned());
    }
    if let Some(frozen) = config.frozen_ctn.as_ref() {
        let transform = FittedModel::from_payload((*frozen.0).clone());
        names.extend(transform.prediction_required_columns()?);
        names.insert(parse_formula(&transform.formula).map_err(|error| error.to_string())?.response);
    }
    Ok(names)
}

fn labels(data: &EncodedDataset, column: &str) -> Result<Vec<String>, String> {
    let index = data.column_map().get(column).copied()
        .ok_or_else(|| format!("missing CTN fold/group column '{column}'"))?;
    let schema = &data.schema.columns[index];
    data.values.column(index).iter().map(|&value| {
        if !value.is_finite() { return Err(format!("nonfinite CTN fold/group label in '{column}'")); }
        if schema.kind == ColumnKindTag::Categorical {
            if value < 0.0 || value.fract() != 0.0 { return Err("invalid categorical fold label".into()); }
            let label = schema.levels.get(value as usize).ok_or("invalid categorical fold code")?;
            if label.is_empty() { return Err("empty CTN fold/group label".into()); }
            Ok(format!("s:{label}"))
        } else {
            Ok(format!("n:{value}"))
        }
    }).collect()
}

/// Explicit folds or seeded whole-group assignment, invariant to row ordering.
pub fn crossfit_assignment(data: &EncodedDataset, recipe: &CtnStage1Recipe) -> Result<Vec<usize>, String> {
    let groups = recipe.group_column.as_deref().map(|name| labels(data, name)).transpose()?;
    let folds = if let Some(name) = recipe.fold_column.as_deref() {
        let values = labels(data, name)?;
        let levels: BTreeSet<_> = values.iter().cloned().collect();
        if levels.len() < 2 { return Err("CTN needs at least two nonempty folds".into()); }
        let codes: BTreeMap<_, _> = levels.into_iter().enumerate().map(|(i, label)| (label, i)).collect();
        values.iter().map(|label| codes[label]).collect::<Vec<_>>()
    } else {
        let groups = groups.as_ref().ok_or("CTN requires fold_column or group_column")?;
        let mut levels: Vec<_> = groups.iter().cloned().collect::<BTreeSet<_>>().into_iter().collect();
        if recipe.folds < 2 || levels.len() < recipe.folds {
            return Err("CTN requires at least two folds and enough independent groups".into());
        }
        levels.shuffle(&mut SmallRng::seed_from_u64(recipe.seed));
        let codes: BTreeMap<_, _> = levels.into_iter().enumerate()
            .map(|(i, label)| (label, i % recipe.folds)).collect();
        groups.iter().map(|label| codes[label]).collect()
    };
    if let Some(groups) = groups {
        let mut assigned = BTreeMap::new();
        for (group, &fold) in groups.iter().zip(&folds) {
            if assigned.insert(group, fold).is_some_and(|prior| prior != fold) {
                return Err("a group crosses CTN fold boundaries".into());
            }
        }
    }
    for fold in folds.iter().copied().collect::<BTreeSet<_>>() {
        if folds.iter().filter(|&&value| value != fold).count() < 2 {
            return Err("every CTN fold requires at least two training rows".into());
        }
    }
    Ok(folds)
}

fn subset(data: &EncodedDataset, rows: &[usize]) -> EncodedDataset {
    EncodedDataset { values: data.values.select(Axis(0), rows), headers: data.headers.clone(),
        schema: data.schema.clone(), column_kinds: data.column_kinds.clone() }
}

fn project(data: &EncodedDataset, names: &BTreeSet<String>) -> Result<EncodedDataset, String> {
    let map = data.column_map();
    let indices = names.iter().map(|name| map.get(name).copied().ok_or_else(|| format!("missing CTN column '{name}'")))
        .collect::<Result<Vec<_>, _>>()?;
    let mut schema = data.schema.clone();
    schema.columns = indices.iter().map(|&i| data.schema.columns[i].clone()).collect();
    Ok(EncodedDataset { headers: names.iter().cloned().collect(), values: data.values.select(Axis(1), &indices),
        schema, column_kinds: indices.iter().map(|&i| data.column_kinds[i]).collect() })
}

/// Evaluate the saved score transform with the native observed-score evaluator.
pub fn observed_scores(model: &FittedModel, data: ndarray::ArrayView2<'_, f64>,
                       columns: &HashMap<String, usize>) -> Result<Array1<f64>, String> {
    if model.predict_model_class() != PredictModelClass::TransformationNormal || model.score_transform.is_some() {
        return Err("score transformation requires one standalone fitted CTN".into());
    }
    let parsed = parse_formula(&model.formula).map_err(|error| error.to_string())?;
    let response = columns.get(&parsed.response).ok_or_else(|| format!("missing CTN response '{}'", parsed.response))?;
    let offset = match model.offset_column.as_ref() {
        Some(name) => data.column(*columns.get(name).ok_or_else(|| format!("missing CTN offset '{name}'"))?).to_owned(),
        None => Array1::zeros(data.nrows()),
    };
    build_transformation_normal_observed_scores(model, data, columns, model.training_headers.as_ref(),
                                               &data.column(*response).to_owned(), &offset)
}

/// One authoritative score read for both marginal-slope prediction families.
fn scores_from_schema(transform: &FittedModel, data: ndarray::ArrayView2<'_, f64>,
                      columns: &HashMap<String, usize>, source: &DataSchema) -> Result<Array1<f64>, String> {
    let target = transform.require_data_schema().map_err(|error| error.to_string())?;
    let mut recoded = None;
    for saved in &target.columns {
        let Some(&index) = columns.get(&saved.name) else { continue; };
        let current = source.columns.iter().find(|column| column.name == saved.name)
            .ok_or_else(|| format!("missing source schema for CTN column '{}'", saved.name))?;
        if current.kind != saved.kind {
            return Err(format!("CTN column '{}' has incompatible source and fitted types", saved.name));
        }
        if saved.kind != ColumnKindTag::Categorical || saved.levels == current.levels { continue; }
        let values = recoded.get_or_insert_with(|| data.to_owned());
        for row in 0..data.nrows() {
            let code = data[[row, index]];
            if !code.is_finite() || code < 0.0 || code.fract() != 0.0 {
                return Err(format!("invalid category code for CTN column '{}'", saved.name));
            }
            let label = current.levels.get(code as usize).ok_or("invalid CTN source category")?;
            let target_code = saved.levels.iter().position(|level| level == label)
                .ok_or_else(|| format!("CTN column '{}' contains an unseen category", saved.name))?;
            values[[row, index]] = target_code as f64;
        }
    }
    observed_scores(transform, recoded.as_ref().map_or(data, |values| values.view()), columns)
}

/// One authoritative score read for both marginal-slope prediction families.
pub fn latent_scores(model: &FittedModel, data: ndarray::ArrayView2<'_, f64>,
                     columns: &HashMap<String, usize>) -> Result<Array1<f64>, String> {
    if let Some(payload) = model.score_transform.as_ref() {
        let transform = FittedModel::from_payload((**payload).clone());
        return scores_from_schema(&transform, data, columns,
                                  model.require_data_schema().map_err(|error| error.to_string())?);
    }
    let name = model.z_column.as_ref().ok_or("marginal-slope model lacks score column")?;
    Ok(data.column(*columns.get(name).ok_or_else(|| format!("missing score column '{name}'"))?).to_owned())
}

/// Fit a shared native CTN/outcome payload, or attach an externally fitted CTN.
fn validate_chain_inputs(dataset: &EncodedDataset, config: &FitConfig) -> Result<(), String> {
    if config.z_column.is_some() || config.frozen_score || (config.ctn_stage1.is_some() && config.frozen_ctn.is_some()) {
        return Err("CTN owns the latent score; external z_column/frozen_score and competing transforms are invalid".into());
    }
    if config.family.as_deref() != Some("bernoulli-marginal-slope")
        && config.survival_likelihood.as_deref() != Some("marginal-slope") {
        return Err("CTN composition requires a marginal-slope outcome".into());
    }
    if dataset.headers.iter().any(|name| name == SCORE) { return Err("reserved CTN score column already exists".into()); }
    Ok(())
}

fn outcome_inputs(dataset: &EncodedDataset, config: &FitConfig, z: &Array1<f64>) -> (EncodedDataset, FitConfig) {
    let mut outcome_data = dataset.clone();
    let p = dataset.values.ncols();
    let mut values = Array2::zeros((dataset.values.nrows(), p + 1));
    values.slice_mut(ndarray::s![.., ..p]).assign(&dataset.values);
    values.column_mut(p).assign(z);
    outcome_data.values = values;
    outcome_data.headers.push(SCORE.into());
    outcome_data.column_kinds.push(ColumnKindTag::Continuous);
    outcome_data.schema.columns.push(SchemaColumn { name: SCORE.into(), kind: ColumnKindTag::Continuous, levels: vec![] });
    let mut outcome_config = config.clone();
    outcome_config.ctn_stage1 = None;
    outcome_config.frozen_ctn = None;
    outcome_config.z_column = Some(SCORE.into());
    outcome_config.frozen_score = true;
    (outcome_data, outcome_config)
}

/// Prepare structural validation without fitting a score distribution. Recipe
/// folds and all source columns are checked; the placeholder only describes
/// the generated score's column geometry and is never used for a fitted model.
pub fn structural_inputs(formula: &str, dataset: &EncodedDataset, config: &FitConfig)
    -> Result<(EncodedDataset, FitConfig), String> {
    validate_chain_inputs(dataset, config)?;
    project(dataset, &required_fit_columns(formula, config)?)?;
    let z = if let Some(frozen) = config.frozen_ctn.as_ref() {
        let model = FittedModel::from_payload((*frozen.0).clone());
        model.validate_for_persistence().map_err(|error| error.to_string())?;
        scores_from_schema(&model, dataset.values.view(), &dataset.column_map(), &dataset.schema)?
    } else {
        let recipe = config.ctn_stage1.as_ref().ok_or("missing CTN input")?;
        crossfit_assignment(dataset, recipe)?;
        let n = dataset.values.nrows();
        Array1::from_iter((0..n).map(|i| (i as f64 + 0.5) / n as f64 - 0.5))
    };
    Ok(outcome_inputs(dataset, config, &z))
}

/// Fit a shared native CTN/outcome payload, or attach an externally fitted CTN.
pub fn fit_chain(formula: String, dataset: &EncodedDataset, config: &FitConfig) -> Result<FittedModelPayload, String> {
    validate_chain_inputs(dataset, config)?;
    let columns = dataset.column_map();
    let (transform, z, folds) = if let Some(frozen) = config.frozen_ctn.as_ref() {
        let model = FittedModel::from_payload((*frozen.0).clone());
        model.validate_for_persistence().map_err(|error| error.to_string())?;
        let z = scores_from_schema(&model, dataset.values.view(), &columns, &dataset.schema)?;
        ((*frozen.0).clone(), z, None)
    } else {
        let recipe = config.ctn_stage1.as_ref().ok_or("missing CTN input")?;
        let folds = crossfit_assignment(dataset, recipe)?;
        let stage_data = project(dataset, &recipe_columns(recipe)?)?;
        let stage_columns = stage_data.column_map();
        let mut z = Array1::zeros(dataset.values.nrows());
        let stage = FitConfig { transformation_normal: true,
            transformation_normal_config: Some(recipe.config.clone()),
            weight_column: recipe.weight_column.clone(), offset_column: recipe.offset_column.clone(),
            scale_dimensions: config.scale_dimensions,
            spatial_optimization: config.spatial_optimization.clone(),
            persistent_warm_start_store: config.persistent_warm_start_store.clone(),
            ..FitConfig::default() };
        let stage_formula = format!("{} ~ {}", recipe.response_column, recipe.covariate_formula_rhs);
        for fold in folds.iter().copied().collect::<BTreeSet<_>>() {
            let train: Vec<_> = folds.iter().enumerate().filter(|(_, f)| **f != fold).map(|(i, _)| i).collect();
            let held: Vec<_> = folds.iter().enumerate().filter(|(_, f)| **f == fold).map(|(i, _)| i).collect();
            let payload = fit_formula_to_payload(stage_formula.clone(), &subset(&stage_data, &train), &stage)
                .map_err(|error| format!("CTN fold {fold} failed: {error}"))?;
            let model = FittedModel::from_payload(payload);
            let values = observed_scores(&model, subset(&stage_data, &held).values.view(), &stage_columns)?;
            for (local, &row) in held.iter().enumerate() { z[row] = values[local]; }
        }
        let transform = fit_formula_to_payload(stage_formula, &stage_data, &stage).map_err(|error| error.to_string())?;
        (transform, z, Some(folds))
    };
    if !z.iter().all(|value| value.is_finite()) { return Err("CTN produced nonfinite scores".into()); }
    let (outcome_data, outcome_config) = outcome_inputs(dataset, config, &z);
    let mut payload = fit_formula_to_payload(formula, &outcome_data, &outcome_config).map_err(|error| error.to_string())?;
    payload.score_transform = Some(Box::new(transform));
    payload.score_crossfit_folds = folds;
    payload.inference_notes.push("CTN is frozen at prediction. Uncertainty is conditional on that fitted transform; standard normality is assumed, not certified.".into());
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_data::DataSchema;
    use crate::transformation_normal::TransformationNormalConfig;

    fn fixture() -> (EncodedDataset, CtnStage1Recipe) {
        let headers = vec!["pgs".into(), "x".into(), "group".into(), "fold".into()];
        let values = Array2::from_shape_fn((12, 4), |(row, column)| match column {
            0 => row as f64, 1 => row as f64 / 12.0, 2 => (row / 2) as f64, _ => (row / 4) as f64,
        });
        let schema = DataSchema { columns: headers.iter().map(|name: &String| SchemaColumn {
            name: name.clone(), kind: ColumnKindTag::Continuous, levels: vec![],
        }).collect() };
        let data = EncodedDataset { headers, values, schema, column_kinds: vec![ColumnKindTag::Continuous; 4] };
        let mut recipe = CtnStage1Recipe::new("pgs", "x", TransformationNormalConfig::default(), None, None).unwrap();
        recipe.group_column = Some("group".into());
        recipe.folds = 3;
        (data, recipe)
    }

    #[test]
    fn native_folds_keep_groups_together_and_ignore_row_order() {
        let (data, recipe) = fixture();
        let folds = crossfit_assignment(&data, &recipe).unwrap();
        assert!(folds.chunks_exact(2).all(|pair| pair[0] == pair[1]));
        let reverse: Vec<_> = (0..12).rev().collect();
        assert_eq!(crossfit_assignment(&subset(&data, &reverse), &recipe).unwrap(), folds.into_iter().rev().collect::<Vec<_>>());
    }

    #[test]
    fn native_folds_reject_family_leakage_and_missing_labels() {
        let (mut data, mut recipe) = fixture();
        recipe.fold_column = Some("fold".into());
        data.values[[0, 3]] = 1.0;
        assert!(crossfit_assignment(&data, &recipe).unwrap_err().contains("crosses"));
        recipe.fold_column = None;
        recipe.group_column = None;
        assert!(crossfit_assignment(&data, &recipe).unwrap_err().contains("requires fold_column"));
    }

    #[test]
    fn native_schema_includes_all_roles_but_not_irrelevant_metadata() {
        let (_, mut recipe) = fixture();
        recipe.weight_column = Some("score_weight".into());
        recipe.offset_column = Some("score_offset".into());
        let config = FitConfig { ctn_stage1: Some(recipe), slope_formula: Some("1 + x".into()),
            weight_column: Some("outcome_weight".into()), ..FitConfig::default() };
        let names = required_fit_columns("Surv(entry, exit, event) ~ x", &config).unwrap();
        assert_eq!(names, ["entry", "exit", "event", "x", "pgs", "group", "score_weight", "score_offset", "outcome_weight"]
            .into_iter().map(String::from).collect());
    }
}
