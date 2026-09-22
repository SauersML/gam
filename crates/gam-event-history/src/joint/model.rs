//! The joint event model's production path (#2961, #2966): fit a cohort,
//! condition on a new history, forecast, save and reload. gam-cli, gam-pyffi
//! and gamfit call these functions, and every one of them hands over long
//! tables that `data.rs` encodes through one frozen schema.
//!
//! A saved model is the shared event-model envelope of kind `joint`, holding
//! the frozen encoding schema and the posterior that forecasting integrates,
//! never the training records. Save → reload → forecast reproduces the
//! in-memory forecast bit for bit.

use super::constant_rate_inference::ConstantRatePosterior;
use super::data::{
    EventDateState, FrozenJointSchema, JointDeclarations, JointTables, NodeResolution,
    VisitProcess,
};
use super::law::{JointLikelihood, JointSpecification, invalid};
use crate::{EventHistoryError, MarkKind};
use gam_model_api::saved_model::{
    SavedModelError, read_saved_model_file, read_saved_model_text, saved_model_text,
    write_saved_model,
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Kind of the saved joint event model in the shared saved-model envelope.
const JOINT_EVENT_MODEL_KIND: &str = "joint";

/// Version of the saved joint event model. Version 3 saves the frozen encoding
/// schema, so earlier payloads are refused.
const JOINT_EVENT_MODEL_VERSION: u64 = 3;

/// The posterior a model integrates when it forecasts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum JointModelLaw {
    /// No latent signatures: independent exact Gamma rate posteriors.
    RankZero(ConstantRatePosterior),
}

/// A fitted joint event model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointEventModel {
    schema: FrozenJointSchema,
    specification: JointSpecification,
    law: JointModelLaw,
}

/// A model conditioned on one history: the posterior given the training cohort
/// and that history, and the risk set the history leaves open at its exit.
pub struct ConditionedJointModel<'m> {
    model: &'m JointEventModel,
    id: String,
    exit: f64,
    law: JointModelLaw,
    at_risk: Vec<bool>,
}

/// Posterior-predictive absolute risks after a history's exit `s`.
#[derive(Clone, Debug, PartialEq)]
pub struct JointForecast {
    /// Offsets `u` after the exit.
    pub horizons: Vec<f64>,
    /// `P(T_dagger > s+u | H_s)`: no terminal mark fires by each horizon.
    pub survival: Vec<f64>,
    /// `P(s < T_d <= s+u, T_d < T_dagger | H_s)` for every mark's next
    /// occurrence (horizons × marks); zero for a once-only mark that has fired.
    pub incidence: Array2<f64>,
    /// A bound on each incidence's numerical error; zero where it is closed form.
    pub incidence_error: Array2<f64>,
}

/// What the rank-zero model declares. Every mark has a constant rate and no
/// covariate terms, measurement channels or genetic scores, so the only data are
/// subjects and their events. Visits are an exogenous schedule, and no
/// measurement observes an event date. A constant rate is integrated exactly by
/// one Gauss-Legendre point per cell, and without latent signatures there is no
/// state to resolve between dated nodes.
pub(super) fn rank_zero_declarations(marks: Option<Vec<(String, MarkKind)>>) -> JointDeclarations {
    JointDeclarations {
        marks,
        channels: Vec::new(),
        score_names: Vec::new(),
        visit_process: VisitProcess::Exogenous {
            event_date_state: EventDateState::BeforeEvents,
        },
        baseline_formula: "1".to_string(),
        population_formula: "1".to_string(),
        drive_formula: None,
        entry_formula: None,
        resolution: NodeResolution {
            quadrature_order: 1,
            state_width: None,
        },
    }
}

/// Refuse every table the rank-zero likelihood would silently ignore. It has no
/// covariate terms, measurement channels, visit process or genetic scores, so a
/// column or row of any of those tables is refused by name, never dropped.
fn refuse_unused_tables(tables: &JointTables) -> Result<(), EventHistoryError> {
    let c = &tables.covariates;
    let g = &tables.genetics;
    let unused = [
        ("covariates", c.id.len() + c.names.len()),
        ("measurements", tables.measurements.id.len()),
        ("visits", tables.visits.id.len()),
        ("genetics", g.id.len() + g.score_names.len()),
    ];
    match unused.iter().find(|(_, size)| *size > 0) {
        Some((table, _)) => Err(invalid(format!(
            "the rank-zero joint model has no covariate terms, measurement channels, visits or genetic scores, so it takes no {table} table"
        ))),
        None => Ok(()),
    }
}

/// Refuse a saved schema that differs from what the rank-zero declarations
/// freeze, so a tampered document can never encode differently when a model
/// conditions: distinct mark names matching the specification's kinds, no
/// covariate columns, channels or genetic scores, the rank-zero visit process
/// and resolution, and intercept-only baseline and population bases.
fn validate_schema(
    schema: &FrozenJointSchema,
    specification: &JointSpecification,
) -> Result<(), EventHistoryError> {
    let mut names: Vec<&String> = schema.mark_names.iter().collect();
    names.sort();
    if let Some(pair) = names.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(invalid(format!("duplicate mark name {:?}", pair[0])));
    }
    if schema.mark_names.len() != specification.marks.len()
        || schema.mark_kinds != specification.marks
    {
        return Err(invalid(format!(
            "{} mark names with kinds {:?} for the mark kinds {:?}",
            schema.mark_names.len(),
            schema.mark_kinds,
            specification.marks
        )));
    }
    let rank_zero = rank_zero_declarations(None);
    let intercept_only = |basis: &super::data::FrozenBasis| {
        basis.spec.linear_terms.is_empty()
            && basis.spec.random_effect_terms.is_empty()
            && basis.spec.smooth_terms.is_empty()
            && basis.penalties.is_empty()
    };
    let beyond = if !schema.covariate_names.is_empty() || !schema.covariate_levels.is_empty() {
        Some("covariate columns")
    } else if !schema.channels.is_empty() {
        Some("measurement channels")
    } else if !schema.score_names.is_empty() {
        Some("genetic scores")
    } else if schema.visit_process != rank_zero.visit_process {
        Some("a visit process beyond the exogenous schedule")
    } else if schema.resolution != rank_zero.resolution {
        Some("a node resolution beyond one point per cell")
    } else if schema.drive.is_some()
        || schema.entry.is_some()
        || !intercept_only(&schema.baseline)
        || !intercept_only(&schema.population)
    {
        Some("covariate bases")
    } else {
        None
    };
    match beyond {
        Some(what) => Err(invalid(format!(
            "a rank-zero joint model's schema holds no {what}"
        ))),
        None => Ok(()),
    }
}

/// Fit the joint event model to a cohort's tables: its subjects and their
/// events, with the declared mark vocabulary (without one, the observed marks,
/// all recurrent).
pub fn fit_joint_event_model(
    marks: Option<Vec<(String, MarkKind)>>,
    tables: &JointTables,
) -> Result<JointEventModel, EventHistoryError> {
    refuse_unused_tables(tables)?;
    let (schema, subjects) = FrozenJointSchema::fit(&rank_zero_declarations(marks), tables)?;
    let specification = JointSpecification::new(schema.mark_kinds.clone())?;
    let validated = JointLikelihood::new(specification.clone())?;
    let posterior = ConstantRatePosterior::infer(
        &specification.marks,
        subjects.into_iter().map(|subject| {
            validated
                .validate_history(&subject.history)
                .map(|()| subject.history)
        }),
    )?;
    Ok(JointEventModel {
        schema,
        specification,
        law: JointModelLaw::RankZero(posterior),
    })
}

impl JointEventModel {
    pub fn mark_names(&self) -> &[String] {
        &self.schema.mark_names
    }

    pub fn mark_kinds(&self) -> &[MarkKind] {
        &self.specification.marks
    }

    /// Condition on every history in `tables`, each ending at its exit `s`,
    /// encoded through the frozen schema the model was fitted with.
    pub fn condition(
        &self,
        tables: &JointTables,
    ) -> Result<Vec<ConditionedJointModel<'_>>, EventHistoryError> {
        refuse_unused_tables(tables)?;
        let marks = &self.specification.marks;
        let validated = JointLikelihood::new(self.specification.clone())?;
        self.schema
            .encode(tables)?
            .into_iter()
            .map(|subject| {
                validated.validate_history(&subject.history)?;
                let exit = subject.history.times.last().copied().ok_or_else(|| {
                    invalid(format!(
                        "subject {:?}: an encoded history has no nodes",
                        subject.id
                    ))
                })?;
                let at_risk = subject.history.open_risk_set(marks).ok_or_else(|| {
                    invalid(format!(
                        "subject {:?}: a terminal event ended this history, so there is nothing to forecast",
                        subject.id
                    ))
                })?;
                let law = match &self.law {
                    JointModelLaw::RankZero(posterior) => {
                        JointModelLaw::RankZero(posterior.condition(marks, &subject.history)?)
                    }
                };
                Ok(ConditionedJointModel {
                    model: self,
                    id: subject.id,
                    exit,
                    law,
                    at_risk,
                })
            })
            .collect()
    }

    /// Save the model to `path`, atomically.
    pub fn save(&self, path: &Path) -> Result<(), SavedModelError> {
        write_saved_model(path, self.saved_text()?.as_bytes())
    }

    /// Load a saved model, refusing another kind or version, or a state its
    /// law cannot hold.
    pub fn load(path: &Path) -> Result<Self, SavedModelError> {
        Self::from_saved_text(&read_saved_model_file(path)?)
    }

    fn saved_text(&self) -> Result<String, SavedModelError> {
        saved_model_text(JOINT_EVENT_MODEL_KIND, JOINT_EVENT_MODEL_VERSION, self)
    }

    /// The model in a saved document's text, refusing another kind or
    /// version, or a state its law cannot hold.
    pub fn from_saved_text(text: &str) -> Result<Self, SavedModelError> {
        let model: Self =
            read_saved_model_text(text, JOINT_EVENT_MODEL_KIND, JOINT_EVENT_MODEL_VERSION)?;
        let inconsistent =
            |reason: EventHistoryError| SavedModelError::Inconsistent { reason: Box::new(reason) };
        // A rank-zero law holds exactly the rank-zero specification of its marks.
        let specification =
            JointSpecification::new(model.specification.marks.clone()).map_err(inconsistent)?;
        if model.specification != specification {
            return Err(inconsistent(invalid(
                "a rank-zero joint model holds only the rank-zero specification of its marks",
            )));
        }
        validate_schema(&model.schema, &specification).map_err(inconsistent)?;
        match &model.law {
            JointModelLaw::RankZero(posterior) => {
                posterior.validate(&specification.marks).map_err(inconsistent)?
            }
        }
        Ok(model)
    }
}

impl ConditionedJointModel<'_> {
    /// The identifier of the history this model is conditioned on.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The history's exit `s`, where its forecasts open.
    pub fn exit(&self) -> f64 {
        self.exit
    }

    /// Forecast `horizons` after the history's exit.
    pub fn forecast(&self, horizons: &[f64]) -> Result<JointForecast, EventHistoryError> {
        match &self.law {
            JointModelLaw::RankZero(posterior) => {
                posterior.forecast(&self.model.specification.marks, &self.at_risk, horizons)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CovariateCells;
    use crate::joint::data::{EventTable, SubjectTable};

    fn tables(subjects: &[(&str, f64, f64)], events: &[(&str, f64, &str)]) -> JointTables {
        JointTables {
            subjects: SubjectTable {
                id: subjects.iter().map(|s| s.0.to_string()).collect(),
                entry: subjects.iter().map(|s| s.1).collect(),
                exit: subjects.iter().map(|s| s.2).collect(),
            },
            events: EventTable {
                id: events.iter().map(|e| e.0.to_string()).collect(),
                time: events.iter().map(|e| e.1).collect(),
                mark: events.iter().map(|e| e.2.to_string()).collect(),
            },
            ..JointTables::default()
        }
    }

    fn marks() -> Option<Vec<(String, MarkKind)>> {
        Some(vec![
            ("diagnosis".to_string(), MarkKind::Once),
            ("cvd_death".to_string(), MarkKind::Terminal),
            ("other_death".to_string(), MarkKind::Terminal),
            ("visit".to_string(), MarkKind::Recurrent),
        ])
    }

    /// A once-only diagnosis, two terminal causes of death and a recurrent visit.
    fn competing_model() -> JointEventModel {
        fit_joint_event_model(
            marks(),
            &tables(
                &[("a", 0.0, 4.0), ("b", 0.0, 6.0)],
                &[
                    ("a", 4.0, "cvd_death"),
                    ("b", 0.0, "diagnosis"),
                    ("b", 1.0, "visit"),
                    ("b", 5.0, "visit"),
                ],
            ),
        )
        .unwrap()
    }

    fn forecast_one(model: &JointEventModel, history: &JointTables, horizons: &[f64]) -> JointForecast {
        let conditioned = model.condition(history).unwrap();
        assert_eq!(conditioned.len(), 1);
        conditioned[0].forecast(horizons).unwrap()
    }

    fn bits(forecast: &JointForecast) -> Vec<u64> {
        forecast
            .horizons
            .iter()
            .chain(&forecast.survival)
            .chain(forecast.incidence.iter())
            .chain(forecast.incidence_error.iter())
            .map(|v| v.to_bits())
            .collect()
    }

    /// One JSON field as the shared envelope WRITES it, `"name":value`, taken
    /// from the writer rather than assumed.
    ///
    /// `gam_model_api::saved_model` writes a document with
    /// `serde_json::to_string`, which is compact. The mutants below are text
    /// edits of that document, so each one has to be spelled the way the
    /// writer spells it; spelling them with a space after the colon made every
    /// `replace` a no-op and the document's own assertions false. Rendering a
    /// one-field object and stripping its braces gives the writer's spelling
    /// whatever the writer later becomes.
    fn saved_field(name: &str, value: serde_json::Value) -> String {
        let rendered = serde_json::to_string(&serde_json::json!({ name: value }))
            .expect("a one-field object renders");
        rendered[1..rendered.len() - 1].to_string()
    }

    #[test]
    fn save_reload_forecast_is_bit_identical_and_other_versions_are_refused() {
        let model = competing_model();
        let history = tables(&[("new", 0.5, 2.5)], &[("new", 1.0, "visit")]);
        let horizons = [0.25, 3.0, 40.0];
        let before = forecast_one(&model, &history, &horizons);
        assert!(before.incidence_error.iter().any(|&e| e > 0.0));
        let path = std::env::temp_dir().join(format!("gam-joint-model-{}.json", std::process::id()));
        model.save(&path).unwrap();
        let reloaded = JointEventModel::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let after = forecast_one(&reloaded, &history, &horizons);
        assert_eq!(bits(&after), bits(&before));

        let text = model.saved_text().unwrap();
        assert_eq!(reloaded.saved_text().unwrap(), text);
        assert!(!text.contains("\"a\"") && !text.contains("\"b\""));
        let version = saved_field("version", serde_json::json!(JOINT_EVENT_MODEL_VERSION));
        let kind = saved_field("kind", serde_json::json!("joint"));
        assert!(text.contains(&version), "{version} in {text}");
        assert!(text.contains(&kind), "{kind}");
        assert!(matches!(
            JointEventModel::from_saved_text(
                &text.replace(&version, &saved_field("version", serde_json::json!(0)))
            ),
            Err(SavedModelError::Version { found: Some(0), expected: JOINT_EVENT_MODEL_VERSION, .. })
        ));
        assert!(matches!(
            JointEventModel::from_saved_text(&text.replace(&format!("{version},"), "")),
            Err(SavedModelError::Version { found: None, .. })
        ));
        // The previous payload version, which held no encoding schema, is refused: it
        // cannot rebuild the encoder a forecast needs, so the refusal names the remedy.
        let previous = saved_field(
            "version",
            serde_json::json!(JOINT_EVENT_MODEL_VERSION - 1),
        );
        let refused = JointEventModel::from_saved_text(&text.replace(&version, &previous));
        assert!(matches!(
            &refused,
            Err(SavedModelError::Version { found: Some(found), expected: JOINT_EVENT_MODEL_VERSION, .. })
                if *found == JOINT_EVENT_MODEL_VERSION - 1
        ));
        let reason = refused.err().map(|error| error.to_string()).unwrap_or_default();
        assert!(reason.contains("refit the model"), "{reason}");
        assert!(matches!(
            JointEventModel::from_saved_text(
                &text.replace(&kind, &saved_field("kind", serde_json::json!("event_history")))
            ),
            Err(SavedModelError::Kind { .. })
        ));
        let shape = saved_field("shape", serde_json::json!(1.0));
        assert!(text.contains(&shape), "{shape}");
        assert!(matches!(
            JointEventModel::from_saved_text(
                &text.replacen(&shape, &saved_field("shape", serde_json::json!(0.5)), 1)
            ),
            Err(SavedModelError::Inconsistent { .. })
        ));
        // A rank-zero law refuses any specification beyond its marks' rank-zero one.
        let columns = saved_field("population_columns", serde_json::json!(1));
        assert!(text.contains(&columns), "{columns}");
        assert!(matches!(
            JointEventModel::from_saved_text(
                &text.replacen(&columns, &saved_field("population_columns", serde_json::json!(2)), 1)
            ),
            Err(SavedModelError::Inconsistent { .. })
        ));
        // A tampered schema is refused on load, one mutant per refused field class:
        // a duplicated mark name, the resolution, covariate columns, genetic scores
        // and measurement channels. Each original field is present first, so no
        // mutant passes by changing nothing.
        for (original, tampered) in [
            ("\"cvd_death\"".to_string(), "\"diagnosis\"".to_string()),
            (
                saved_field("quadrature_order", serde_json::json!(1)),
                saved_field("quadrature_order", serde_json::json!(5)),
            ),
            (
                saved_field("state_width", serde_json::Value::Null),
                saved_field("state_width", serde_json::json!(1.0)),
            ),
            (
                saved_field("covariate_names", serde_json::json!([])),
                saved_field("covariate_names", serde_json::json!(["bmi"])),
            ),
            (
                saved_field("score_names", serde_json::json!([])),
                saved_field("score_names", serde_json::json!(["prs"])),
            ),
            (
                saved_field("channels", serde_json::json!([])),
                saved_field(
                    "channels",
                    serde_json::json!([{"name": "hba1c", "family": "StudentT"}]),
                ),
            ),
        ] {
            assert!(text.contains(&original), "{original}");
            assert!(
                matches!(
                    JointEventModel::from_saved_text(&text.replacen(&original, &tampered, 1)),
                    Err(SavedModelError::Inconsistent { .. })
                ),
                "{tampered}"
            );
        }
        // The classes a text replace cannot express, mutated as JSON values and reserialized: the visit
        // process, a drive or entry basis, a population penalty block, a mark kind the specification
        // does not hold, one mark name too few, and covariate levels. The originals are asserted first:
        // exogenous visits, no drive or entry basis, no penalties, the specification's kinds, one name
        // per kind and no levels.
        let saved: serde_json::Value = serde_json::from_str(&text).unwrap();
        let schema = &saved["model"]["schema"];
        assert!(schema["visit_process"]["Exogenous"].is_object());
        assert!(schema["drive"].is_null() && schema["entry"].is_null());
        assert!(schema["population"]["penalties"].as_array().is_some_and(|p| p.is_empty()));
        assert_eq!(schema["mark_kinds"][0], serde_json::json!("Once"));
        assert_eq!(schema["mark_names"].as_array().map(Vec::len), Some(model.mark_kinds().len()));
        assert!(schema["covariate_levels"].as_array().is_some_and(|l| l.is_empty()));
        let tamper = |edit: &dyn Fn(&mut serde_json::Value)| {
            let mut document = saved.clone();
            edit(&mut document["model"]["schema"]);
            JointEventModel::from_saved_text(&serde_json::to_string_pretty(&document).unwrap())
        };
        let edits: [(&str, Box<dyn Fn(&mut serde_json::Value)>); 7] = [
            (
                "visit process",
                Box::new(|s| {
                    s["visit_process"] = serde_json::json!({
                        "Informative": {"mark": "visit", "event_date_state": "BeforeEvents"}
                    })
                }),
            ),
            (
                "drive basis",
                Box::new(|s| {
                    let basis = s["baseline"].clone();
                    s["drive"] = basis;
                }),
            ),
            (
                "entry basis",
                Box::new(|s| {
                    let basis = s["baseline"].clone();
                    s["entry"] = basis;
                }),
            ),
            (
                "population penalty",
                Box::new(|s| {
                    s["population"]["penalties"] = serde_json::json!([{
                        "columns": {"start": 0, "end": 1},
                        "local": {"v": 1, "dim": [1, 1], "data": [1.0]},
                        "rank": 1
                    }])
                }),
            ),
            (
                "mark kind",
                Box::new(|s| s["mark_kinds"][0] = serde_json::json!("Recurrent")),
            ),
            (
                "mark names",
                Box::new(|s| {
                    if let Some(names) = s["mark_names"].as_array_mut() {
                        names.pop();
                    }
                }),
            ),
            (
                "covariate levels",
                Box::new(|s| s["covariate_levels"] = serde_json::json!([["x"]])),
            ),
        ];
        for (class, edit) in &edits {
            let result = tamper(edit.as_ref());
            assert!(
                matches!(&result, Err(SavedModelError::Inconsistent { .. })),
                "{class}: {:?}",
                result.as_ref().err()
            );
        }
    }

    #[test]
    fn conditioning_follows_the_history_risk_set() {
        let model = competing_model();
        // A fired diagnosis has no next occurrence to forecast; death still does.
        let diagnosed = forecast_one(
            &model,
            &tables(&[("dx", 0.0, 2.0)], &[("dx", 1.0, "diagnosis")]),
            &[3.0],
        );
        assert_eq!(diagnosed.incidence[[0, 0]], 0.0);
        assert!(diagnosed.incidence[[0, 1]] > 0.0 && diagnosed.incidence[[0, 3]] > 0.0);
        let error = model
            .condition(&tables(&[("dead", 0.0, 2.0)], &[("dead", 2.0, "other_death")]))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("terminal event ended"), "{error}");
        let conditioned = model
            .condition(&tables(&[("late", 0.0, 2.0), ("early", 0.0, 1.0)], &[]))
            .unwrap();
        assert_eq!(
            conditioned.iter().map(ConditionedJointModel::id).collect::<Vec<_>>(),
            ["late", "early"]
        );
        let error = conditioned[0].forecast(&[-1.0]).err().unwrap().to_string();
        assert!(error.contains("nonnegative"), "{error}");
    }

    #[test]
    fn fits_refuse_what_the_rank_zero_model_cannot_hold() {
        let cohort = tables(&[("a", 0.0, 1.0)], &[]);
        let duplicate = Some(vec![
            ("x".to_string(), MarkKind::Once),
            ("x".to_string(), MarkKind::Terminal),
        ]);
        // Each arm asserts on the whole result, so a check that stops refusing fails
        // its own assertion rather than an unwrap.
        let result = fit_joint_event_model(duplicate, &cohort);
        assert!(
            matches!(&result, Err(e) if e.to_string().contains("duplicate mark name")),
            "{:?}",
            result.as_ref().err()
        );
        let result = fit_joint_event_model(marks(), &tables(&[], &[]));
        assert!(
            matches!(&result, Err(e) if e.to_string().contains("at least one subject")),
            "{:?}",
            result.as_ref().err()
        );
        let mut covariates = cohort.clone();
        covariates.covariates.names = vec!["bmi".to_string()];
        covariates.covariates.columns = vec![CovariateCells::Numbers(vec![25.0])];
        covariates.covariates.id = vec!["a".to_string()];
        covariates.covariates.start = vec![0.0];
        let result = fit_joint_event_model(marks(), &covariates);
        assert!(
            matches!(&result, Err(e) if e.to_string().contains("takes no covariates table")),
            "{:?}",
            result.as_ref().err()
        );
        // Every other table the rank-zero likelihood would ignore is refused by name,
        // at fit and at conditioning.
        let mut visits = cohort.clone();
        visits.visits.id = vec!["a".to_string()];
        let mut genetics = cohort.clone();
        genetics.genetics.id = vec!["a".to_string()];
        let mut measurements = cohort.clone();
        measurements.measurements.id = vec!["a".to_string()];
        let model = competing_model();
        for (table, tables) in [
            ("visits", &visits),
            ("genetics", &genetics),
            ("measurements", &measurements),
        ] {
            let refused = format!("takes no {table} table");
            let result = fit_joint_event_model(marks(), tables);
            assert!(
                matches!(&result, Err(e) if e.to_string().contains(&refused)),
                "{:?}",
                result.as_ref().err()
            );
            let result = model.condition(tables);
            assert!(
                matches!(&result, Err(e) if e.to_string().contains(&refused)),
                "{:?}",
                result.as_ref().err()
            );
        }
        assert!(fit_joint_event_model(marks(), &cohort).is_ok());
        assert!(model.condition(&cohort).is_ok());
    }

    #[test]
    fn a_shared_horizon_forecast_is_bit_identical_whatever_other_horizons_are_requested() {
        // #2963 (A8): horizons are evaluation points, never accuracy controls.
        let model = competing_model();
        let conditioned = model.condition(&tables(&[("new", 0.0, 2.0)], &[])).unwrap();
        let conditioned = &conditioned[0];
        let single = conditioned.forecast(&[7.0]).unwrap();
        let several = conditioned.forecast(&[0.5, 7.0, 40.0, 400.0]).unwrap();
        assert_eq!(single.survival[0].to_bits(), several.survival[1].to_bits());
        for d in 0..model.mark_kinds().len() {
            assert_eq!(single.incidence[[0, d]].to_bits(), several.incidence[[1, d]].to_bits());
            assert_eq!(
                single.incidence_error[[0, d]].to_bits(),
                several.incidence_error[[1, d]].to_bits()
            );
        }
        // The diagnosis goes through the bracket, so its error term is live in the comparison.
        assert!(single.incidence_error[[0, 0]] > 0.0);
        // Positive control: a different horizon changes the values this test compares.
        assert_ne!(single.survival[0].to_bits(), several.survival[2].to_bits());
        assert_ne!(single.incidence[[0, 0]].to_bits(), several.incidence[[2, 0]].to_bits());
    }

    #[test]
    fn extending_a_history_after_the_assessment_time_never_changes_its_forecast() {
        // #2962 (A7): a forecast made at s sees exactly what was known at s. A history ends at its exit s,
        // and production guarantees that no record after s can reach the forecast: the encoder refuses an
        // event row past its subject's exit instead of reading it.
        let model = competing_model();
        let horizons = [0.5, 4.0, 30.0];
        // Alive, undiagnosed, with one visit, at the assessment time 3.
        let known = tables(&[("p", 0.0, 3.0)], &[("p", 1.5, "visit")]);
        let at_cutoff = forecast_one(&model, &known, &horizons);
        // The same history still carrying records from after s is refused by name, never conditioned on;
        // its neighbour without those rows is the accepted history above.
        let extended = tables(
            &[("p", 0.0, 3.0)],
            &[("p", 1.5, "visit"), ("p", 4.0, "diagnosis"), ("p", 4.5, "visit")],
        );
        let result = model.condition(&extended);
        assert!(
            matches!(&result, Err(e) if e.to_string().contains("after its exit")),
            "{:?}",
            result.as_ref().err()
        );
        assert!(model.condition(&known).is_ok());
        // Positive controls. A diagnosis AT the cutoff is part of H_s and closes that mark's risk.
        let closed = forecast_one(
            &model,
            &tables(&[("p", 0.0, 3.0)], &[("p", 1.5, "visit"), ("p", 3.0, "diagnosis")]),
            &horizons,
        );
        assert!(closed.incidence.column(0).iter().all(|&p| p == 0.0));
        assert_ne!(bits(&closed), bits(&at_cutoff));
        // A later exit adds exposure, so the forecast does see conditioning.
        let moved = forecast_one(&model, &tables(&[("p", 0.0, 3.5)], &[("p", 1.5, "visit")]), &horizons);
        assert_ne!(bits(&moved), bits(&at_cutoff));
    }
}
