//! The data contract of the joint latent-signature model: long tables in, joint
//! histories out. This is the only encoder; the CLI, the Python binding and
//! gamfit hand it raw columns, and fitting, conditioning on a new history and
//! serving all encode through the same [`FrozenJointSchema`].
//!
//! # Tables
//!
//! * subjects `(id, entry, exit)`: one row per subject, `entry < exit`.
//! * events `(id, time, mark)`: marks resolved against the declared vocabulary
//!   by the cohort's own resolver. An event at or before `entry` is entry
//!   information, never a count node, and it opens no exposure before entry.
//!   Events recorded at one date share one node as an unordered multiset. Every
//!   row is an event: two rows of a recurrent mark on one date are two events.
//!   The encoder cannot tell a duplicated record from a second event, so it
//!   never merges rows; removing duplicate records belongs to the caller.
//! * covariates `(id, start, columns...)`: predictable covariate segments, typed
//!   by the cohort's covariate encoder. A model with no covariate columns takes
//!   an empty table: every subject then has one empty segment from its entry.
//! * measurements `(id, time, channel, value, exposure)`: one row per channel at
//!   an attended visit. `value` is a recorded value, or `None` when the channel
//!   was not measured at that visit. A count channel carries its positive
//!   exposure; no other channel carries one.
//! * visits `(id, time, attended)`: the observation schedule.
//! * genetics `(id, scores...)`: at most one row per subject. A missing or NaN
//!   score is missing and is never filled.
//!
//! Rows may come in any order. Every record carries its own time, and each kind
//! of record is put in canonical order at one site, so a table's row order cannot
//! change an encoding:
//! * events: each goes to the node at its date, found by search over the sorted
//!   node times, and `encode_cohort` lists the marks at a node by mark index,
//!   whatever their kinds. The node's multiset is the date's whole record, so a
//!   terminal event and another event on its date need no order between them.
//! * covariate segments: `EventHistoryCohort::validate` sorts them by start and
//!   refuses two of one subject starting at one time.
//! * measurements: `encode_cohort` sorts each subject's records by time, channel,
//!   value bits and exposure bits. The key is total, so rows that compare equal
//!   are the same record.
//! * visits and genetics: read by subject and time, never by position.
//!
//! # Observations
//!
//! Three different observations are kept apart:
//! * nonattendance: a visit row with `attended = false`. It adds no
//!   measurement record.
//! * an unmeasured channel at an attended visit: a measurement row with no
//!   value. It contributes its integrated likelihood, one, and is never
//!   imputed.
//! * a recorded negative answer: a measurement row with the value `0`. It
//!   contributes the channel's likelihood of that value.
//!
//! A measurement row on a visit that was not attended, or on no visit at all,
//! contradicts the schedule and is refused.
//!
//! Visits follow a declared process. [`VisitProcess::Exogenous`] visits are a
//! schedule the model conditions on. [`VisitProcess::Informative`] visits are
//! observations of a modelled mark: every attended visit becomes an event of
//! that declared recurrent mark, so the visit intensity enters the likelihood.
//!
//! # Entry and recorded dates
//!
//! Entry conditions on what was recorded at entry. The context row in force at
//! entry, and the once-only marks recorded at or before entry, feed the entry
//! state law. No event-free exposure is invented before entry.
//!
//! An event time is the date a record was made, such as a first recorded
//! diagnosis. It is not the biological onset, and nothing here says when onset
//! happened.

use super::law::{BasisPenalty, CompensatorPoint, JointHistory, MeasurementFamily, MeasurementRecord};
use crate::cohort::{
    CovariateCells, CovariateSegment, CovariateValue, Event, EventHistoryCohort,
    EventHistoryError, MarkKind, SubjectHistory, cell_rule, code_covariate_value, design_rows,
    mark_index_of, resolve_mark_vocabulary,
};
use crate::formula::covariate_spec_from_formula;
use gam_terms::smooth::{
    TermCollectionSpec, build_term_collection_design, freeze_term_collection_from_design,
};
use ndarray::{Array2, ArrayView2, s};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// One declared measurement channel with its parsed family.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct ChannelDeclaration {
    pub(super) name: String,
    pub(super) family: MeasurementFamily,
}

/// The state a measurement observes when its date also records events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventDateState {
    /// The state before that date's events, the one their intensities use.
    BeforeEvents,
    /// The state after every event recorded on that date has applied its jump.
    AfterEvents,
}

/// The visit contract: how visit times relate to the latent state, and which
/// state a measurement taken on an event date observes. Both are declared,
/// never defaulted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisitProcess {
    /// A schedule that does not depend on the state; it is conditioned on.
    Exogenous { event_date_state: EventDateState },
    /// Attendance is an observation of a declared recurrent mark.
    Informative {
        mark: String,
        event_date_state: EventDateState,
    },
}

impl VisitProcess {
    fn event_date_state(&self) -> EventDateState {
        match self {
            Self::Exogenous { event_date_state } | Self::Informative { event_date_state, .. } => {
                *event_date_state
            }
        }
    }
}

/// The resolution histories are expanded at. `state_width` bounds the gap
/// between consecutive latent nodes, and is `None` for a model with no latent
/// state to resolve, which places only the dated nodes. `quadrature_order` is
/// the Gauss-Legendre order of every compensator cell. The fit's refinement
/// controller sets both at fixed coefficients; the encoder derives nothing from
/// coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeResolution {
    pub quadrature_order: usize,
    pub state_width: Option<f64>,
}

/// Subjects `(id, entry, exit)`.
#[derive(Clone, Debug, Default)]
pub struct SubjectTable {
    pub id: Vec<String>,
    pub entry: Vec<f64>,
    pub exit: Vec<f64>,
}

/// Events `(id, time, mark)`.
#[derive(Clone, Debug, Default)]
pub struct EventTable {
    pub id: Vec<String>,
    pub time: Vec<f64>,
    pub mark: Vec<String>,
}

/// Covariate segments `(id, start, columns...)`.
#[derive(Clone, Debug, Default)]
pub struct CovariateTable {
    pub id: Vec<String>,
    pub start: Vec<f64>,
    pub names: Vec<String>,
    pub columns: Vec<CovariateCells>,
}

/// Measurements `(id, time, channel, value, exposure)`. A `None` value is a
/// channel not measured at an attended visit. `exposure` is the positive
/// exposure multiplying a count channel's mean; only count channels carry one.
#[derive(Clone, Debug, Default)]
pub struct MeasurementTable {
    pub id: Vec<String>,
    pub time: Vec<f64>,
    pub channel: Vec<String>,
    pub value: Vec<Option<f64>>,
    pub exposure: Vec<Option<f64>>,
}

/// Visits `(id, time, attended)`.
#[derive(Clone, Debug, Default)]
pub struct VisitTable {
    pub id: Vec<String>,
    pub time: Vec<f64>,
    pub attended: Vec<bool>,
}

/// Genetic scores `(id, scores...)`, at most one row per subject. A `None` or
/// NaN score, or a subject with no row, is missing: it is integrated under the
/// score law, never filled. Scores pass as recorded, with no per-score
/// standardization, which would treat correlated scores as independent.
#[derive(Clone, Debug, Default)]
pub struct GeneticTable {
    pub id: Vec<String>,
    pub score_names: Vec<String>,
    pub scores: Vec<Vec<Option<f64>>>,
}

/// The long tables of one or many subjects.
#[derive(Clone, Debug, Default)]
pub struct JointTables {
    pub subjects: SubjectTable,
    pub events: EventTable,
    pub covariates: CovariateTable,
    pub measurements: MeasurementTable,
    pub visits: VisitTable,
    pub genetics: GeneticTable,
}

/// What a fit declares before any data is read.
#[derive(Clone, Debug)]
pub struct JointDeclarations {
    /// Mark names and kinds; `None` takes the observed marks, all recurrent.
    pub marks: Option<Vec<(String, MarkKind)>>,
    /// Channel names with their families as the surfaces spell them:
    /// `student_t`, `probit:<categories>` or `negative_binomial`.
    pub channels: Vec<(String, String)>,
    pub score_names: Vec<String>,
    pub visit_process: VisitProcess,
    /// Right-hand side of the baseline log-rate formula, e.g. `"x + s(time)"`.
    pub baseline_formula: String,
    /// Right-hand side of the population basis over time and context, which
    /// carries the decoder weights and measurement coefficients. Its first
    /// column must be the unit constant.
    pub population_formula: String,
    /// Right-hand side of the state drive basis; `None` at rank zero.
    pub drive_formula: Option<String>,
    /// Right-hand side of the entry context basis; `None` for no context.
    pub entry_formula: Option<String>,
    pub resolution: NodeResolution,
}

/// Everything a fit learned about encoding, frozen. It is saved with the
/// model, and conditioning and serving encode through it, so a reloaded model
/// sees exactly the columns, levels, channels and bases it was fitted with.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct FrozenJointSchema {
    pub mark_names: Vec<String>,
    pub mark_kinds: Vec<MarkKind>,
    pub covariate_names: Vec<String>,
    pub covariate_levels: Vec<Vec<String>>,
    pub channels: Vec<ChannelDeclaration>,
    pub score_names: Vec<String>,
    pub visit_process: VisitProcess,
    pub baseline: FrozenBasis,
    pub population: FrozenBasis,
    pub drive: Option<FrozenBasis>,
    pub entry: Option<FrozenBasis>,
    pub resolution: NodeResolution,
}

/// A formula's basis frozen on the fit's design rows, together with its
/// penalties exactly as gam-terms built them, in the law's `BasisPenalty` shape.
/// Several blocks may share a range, as a smooth's wiggliness penalty and its
/// double-penalty ridge do, and `rank` is gam-terms' declared structural rank.
/// Nothing here reparametrizes, reorders or drops a block.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct FrozenBasis {
    pub(super) spec: TermCollectionSpec,
    pub(super) penalties: Vec<BasisPenalty>,
}

/// One subject's encoded history.
#[derive(Clone, Debug)]
pub(super) struct EncodedSubject {
    pub id: String,
    pub history: JointHistory,
}

/// A refusal of the tables, naming what is wrong and where.
#[derive(Debug, Error, Clone)]
pub enum JointDataError {
    #[error("{table} column {column:?} has {found} rows, expected {expected}")]
    ColumnLength {
        table: &'static str,
        column: String,
        expected: usize,
        found: usize,
    },
    #[error("{table} table: subject {id:?} is not in the subjects table")]
    UnknownSubject { table: &'static str, id: String },
    #[error("{table} table: subject {id:?} has two rows")]
    DuplicateSubject { table: &'static str, id: String },
    #[error("measurement channel {channel:?} is not declared; channels: {channels:?}")]
    UnknownChannel {
        channel: String,
        channels: Vec<String>,
    },
    #[error("channel {channel:?} names two declarations")]
    DuplicateChannel { channel: String },
    #[error("channel {channel:?} ({family}) cannot take the value {value}")]
    OutsideSupport {
        channel: String,
        family: String,
        value: f64,
    },
    #[error("covariate columns {found:?} do not match the declared columns {expected:?}")]
    CovariateColumns {
        expected: Vec<String>,
        found: Vec<String>,
    },
    #[error("genetic score {name:?} is not declared; scores: {names:?}")]
    UnknownGeneticScore { name: String, names: Vec<String> },
    #[error("{table} table: subject {subject:?} has time {time} outside its follow-up [{entry}, {exit}]")]
    OutsideFollowUp {
        table: &'static str,
        subject: String,
        time: f64,
        entry: f64,
        exit: f64,
    },
    #[error("subject {subject:?}: a measurement at {time} is not on an attended visit")]
    MeasurementWithoutVisit { subject: String, time: f64 },
    #[error("subject {subject:?}: a measurement at {time} is declared to observe the state after that date's events, but follow-up terminated then")]
    MeasurementAfterTermination { subject: String, time: f64 },
    #[error("informative visit mark {mark:?}: {reason}")]
    VisitMark { mark: String, reason: String },
    #[error("the {basis} basis has a nonzero affine offset, which the joint law has no term for")]
    AffineOffset { basis: &'static str },
    #[error("subject {subject:?}: channel {channel:?} at {time}: {reason}")]
    Exposure {
        subject: String,
        channel: String,
        time: f64,
        reason: &'static str,
    },
    #[error("the {basis} basis does not carry the unit constant in column zero")]
    PopulationConstant { basis: &'static str },
    #[error("the {order}-point Gauss-Legendre rule has no certified weights: relative weight error {weight_relative_error}")]
    QuadratureCertificate {
        order: usize,
        weight_relative_error: f64,
    },
    #[error("unknown measurement family {text:?}; expected student_t, probit:<categories ≥ 2> or negative_binomial")]
    UnknownFamily { text: String },
    #[error(transparent)]
    Cohort(#[from] EventHistoryError),
}

impl From<JointDataError> for EventHistoryError {
    fn from(error: JointDataError) -> Self {
        match error {
            JointDataError::Cohort(inner) => inner,
            other => EventHistoryError::InvalidInput {
                reason: other.to_string(),
            },
        }
    }
}

/// A channel's family as every surface spells it: `student_t`,
/// `probit:<categories>` (a binary item is `probit:2`) or `negative_binomial`.
pub(super) fn parse_measurement_family(text: &str) -> Result<MeasurementFamily, JointDataError> {
    let spelled = text.trim().to_ascii_lowercase();
    let family = match spelled.split_once(':') {
        None if spelled == "student_t" => Some(MeasurementFamily::StudentT),
        None if spelled == "negative_binomial" => Some(MeasurementFamily::NegativeBinomial),
        Some(("probit", categories)) => categories
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|categories| *categories >= 2)
            .map(|categories| MeasurementFamily::Probit { categories }),
        _ => None,
    };
    family.ok_or_else(|| JointDataError::UnknownFamily {
        text: text.to_string(),
    })
}

fn invalid(reason: impl Into<String>) -> JointDataError {
    JointDataError::Cohort(EventHistoryError::InvalidInput {
        reason: reason.into(),
    })
}

fn same_length(
    table: &'static str,
    column: &str,
    expected: usize,
    found: usize,
) -> Result<(), JointDataError> {
    if expected == found {
        Ok(())
    } else {
        Err(JointDataError::ColumnLength {
            table,
            column: column.to_string(),
            expected,
            found,
        })
    }
}

fn cells_len(cells: &CovariateCells) -> usize {
    match cells {
        CovariateCells::Numbers(values) => values.len(),
        CovariateCells::Labels(values) => values.len(),
    }
}

impl JointTables {
    fn check_lengths(&self) -> Result<(), JointDataError> {
        let s = &self.subjects;
        same_length("subjects", "entry", s.id.len(), s.entry.len())?;
        same_length("subjects", "exit", s.id.len(), s.exit.len())?;
        let e = &self.events;
        same_length("events", "time", e.id.len(), e.time.len())?;
        same_length("events", "mark", e.id.len(), e.mark.len())?;
        let c = &self.covariates;
        same_length("covariates", "start", c.id.len(), c.start.len())?;
        if c.names.len() != c.columns.len() {
            return Err(JointDataError::CovariateColumns {
                expected: c.names.clone(),
                found: vec![format!("{} columns", c.columns.len())],
            });
        }
        for (name, cells) in c.names.iter().zip(&c.columns) {
            same_length("covariates", name, c.id.len(), cells_len(cells))?;
        }
        let m = &self.measurements;
        same_length("measurements", "time", m.id.len(), m.time.len())?;
        same_length("measurements", "channel", m.id.len(), m.channel.len())?;
        same_length("measurements", "value", m.id.len(), m.value.len())?;
        same_length("measurements", "exposure", m.id.len(), m.exposure.len())?;
        let v = &self.visits;
        same_length("visits", "time", v.id.len(), v.time.len())?;
        same_length("visits", "attended", v.id.len(), v.attended.len())?;
        let g = &self.genetics;
        same_length("genetics", "score names", g.score_names.len(), g.scores.len())?;
        for (name, column) in g.score_names.iter().zip(&g.scores) {
            same_length("genetics", name, g.id.len(), column.len())?;
        }
        Ok(())
    }

    fn subject_index(&self) -> Result<HashMap<&str, usize>, JointDataError> {
        let mut index = HashMap::with_capacity(self.subjects.id.len());
        for (i, id) in self.subjects.id.iter().enumerate() {
            if index.insert(id.as_str(), i).is_some() {
                return Err(JointDataError::DuplicateSubject {
                    table: "subjects",
                    id: id.clone(),
                });
            }
        }
        Ok(index)
    }
}

fn subject_of(
    index: &HashMap<&str, usize>,
    table: &'static str,
    id: &str,
) -> Result<usize, JointDataError> {
    index
        .get(id)
        .copied()
        .ok_or_else(|| JointDataError::UnknownSubject {
            table,
            id: id.to_string(),
        })
}

/// Code every covariate column against the schema's levels with the cohort's
/// own coder. At a fit the levels come from [`CovariateCells::encode`], so the
/// codes equal that encoder's; at serving an unknown level is refused.
fn coded_covariates(
    names: &[String],
    levels: &[Vec<String>],
    table: &CovariateTable,
) -> Result<Array2<f64>, JointDataError> {
    if table.names != names {
        return Err(JointDataError::CovariateColumns {
            expected: names.to_vec(),
            found: table.names.clone(),
        });
    }
    let mut coded = Array2::<f64>::zeros((table.id.len(), names.len()));
    for (j, ((name, levels), cells)) in names.iter().zip(levels).zip(&table.columns).enumerate() {
        let values: Vec<CovariateValue> = match cells {
            CovariateCells::Numbers(values) => {
                values.iter().map(|v| CovariateValue::Number(*v)).collect()
            }
            CovariateCells::Labels(values) => {
                values.iter().map(|v| CovariateValue::Label(v.clone())).collect()
            }
        };
        for (i, value) in values.into_iter().enumerate() {
            coded[[i, j]] = code_covariate_value(name, levels, value)?;
        }
    }
    Ok(coded)
}

/// The informative visit mark's index, refusing a mark that is undeclared or
/// cannot recur.
fn visit_mark(
    process: &VisitProcess,
    mark_names: &[String],
    mark_kinds: &[MarkKind],
) -> Result<Option<usize>, JointDataError> {
    let VisitProcess::Informative { mark, .. } = process else {
        return Ok(None);
    };
    let index = mark_names
        .iter()
        .position(|name| name == mark)
        .ok_or_else(|| JointDataError::VisitMark {
            mark: mark.clone(),
            reason: format!("it is not in the declared mark vocabulary {mark_names:?}"),
        })?;
    if mark_kinds[index] != MarkKind::Recurrent {
        return Err(JointDataError::VisitMark {
            mark: mark.clone(),
            reason: format!(
                "it is declared {}, but attendance can recur",
                mark_kinds[index].name()
            ),
        });
    }
    Ok(Some(index))
}

/// The cohort the tables describe, under a fixed vocabulary: subjects, their
/// events (attended informative visits included) and covariate segments,
/// validated by the cohort's own rules.
fn build_cohort(
    mark_names: &[String],
    mark_kinds: &[MarkKind],
    covariate_names: &[String],
    covariate_levels: &[Vec<String>],
    process: &VisitProcess,
    tables: &JointTables,
) -> Result<EventHistoryCohort, JointDataError> {
    tables.check_lengths()?;
    let index = tables.subject_index()?;
    let mut subjects: Vec<SubjectHistory> = tables
        .subjects
        .id
        .iter()
        .zip(tables.subjects.entry.iter().zip(&tables.subjects.exit))
        .map(|(id, (&entry, &exit))| SubjectHistory {
            id: id.clone(),
            entry,
            exit,
            events: Vec::new(),
            segments: Vec::new(),
        })
        .collect();
    let visit = visit_mark(process, mark_names, mark_kinds)?;
    let events = &tables.events;
    for ((id, &time), label) in events.id.iter().zip(&events.time).zip(&events.mark) {
        let subject = subject_of(&index, "events", id)?;
        let mark = mark_index_of(mark_names, label)?;
        if Some(mark) == visit {
            return Err(JointDataError::VisitMark {
                mark: label.clone(),
                reason: "the events table records it too; attendance comes from the visits table alone"
                    .to_string(),
            });
        }
        subjects[subject].events.push(Event { time, mark });
    }
    let visits = &tables.visits;
    for ((id, &time), &attended) in visits.id.iter().zip(&visits.time).zip(&visits.attended) {
        let subject = subject_of(&index, "visits", id)?;
        within_follow_up("visits", &subjects[subject], time)?;
        if let Some(mark) = visit
            && attended
        {
            subjects[subject].events.push(Event { time, mark });
        }
    }
    let covariates = &tables.covariates;
    let mut coded = coded_covariates(covariate_names, covariate_levels, covariates)?;
    if covariate_names.is_empty() && covariates.id.is_empty() {
        // No covariate columns and no rows: every subject has one empty segment
        // from its entry, so a model without covariates needs no covariate table.
        coded = Array2::zeros((1, 0));
        for subject in &mut subjects {
            subject.segments.push(CovariateSegment {
                start: subject.entry,
                row: 0,
            });
        }
    } else {
        for (row, (id, &start)) in covariates.id.iter().zip(&covariates.start).enumerate() {
            let subject = subject_of(&index, "covariates", id)?;
            subjects[subject].segments.push(CovariateSegment { start, row });
        }
    }
    let mut cohort = EventHistoryCohort {
        mark_names: mark_names.to_vec(),
        mark_kinds: mark_kinds.to_vec(),
        covariate_names: covariate_names.to_vec(),
        covariate_levels: covariate_levels.to_vec(),
        covariates: coded,
        subjects,
    };
    cohort.validate()?;
    Ok(cohort)
}

fn frozen_basis(
    formula: &str,
    rows: ArrayView2<'_, f64>,
    cohort: &EventHistoryCohort,
    basis: &'static str,
) -> Result<FrozenBasis, JointDataError> {
    let spec = covariate_spec_from_formula(formula, rows, cohort)?;
    let design = build_term_collection_design(rows, &spec)
        .map_err(|error| invalid(format!("{basis} design: {error}")))?;
    let frozen = freeze_term_collection_from_design(&spec, &design)
        .map_err(|error| invalid(format!("freezing the {basis} basis: {error}")))?;
    // gam-terms keeps `penaltyinfo` in lockstep with `penalties`; each entry
    // carries the block's declared rank.
    if design.penaltyinfo.len() != design.penalties.len() {
        return Err(invalid(format!(
            "the {basis} basis has {} penalty blocks but {} penalty descriptions",
            design.penalties.len(),
            design.penaltyinfo.len()
        )));
    }
    let penalties = design
        .penalties
        .iter()
        .zip(&design.penaltyinfo)
        .map(|(block, info)| BasisPenalty {
            columns: block.col_range.clone(),
            local: block.local.clone(),
            rank: info.penalty.effective_rank,
        })
        .collect();
    Ok(FrozenBasis {
        spec: frozen,
        penalties,
    })
}

impl FrozenJointSchema {
    /// Learn the vocabulary, the covariate levels and the frozen bases from the
    /// tables, then encode every subject through the learned schema. The bases
    /// are built on the cohort's outcome-free design rows, as the current engine
    /// builds its own.
    pub(super) fn fit(
        declarations: &JointDeclarations,
        tables: &JointTables,
    ) -> Result<(Self, Vec<EncodedSubject>), JointDataError> {
        tables.check_lengths()?;
        let channels = declarations
            .channels
            .iter()
            .map(|(name, family)| {
                Ok(ChannelDeclaration {
                    name: name.clone(),
                    family: parse_measurement_family(family)?,
                })
            })
            .collect::<Result<Vec<_>, JointDataError>>()?;
        {
            let mut names: Vec<&str> = channels.iter().map(|c| c.name.as_str()).collect();
            names.sort_unstable();
            if let Some(w) = names.windows(2).find(|w| w[0] == w[1]) {
                return Err(JointDataError::DuplicateChannel {
                    channel: w[0].to_string(),
                });
            }
            let mut scores: Vec<&str> = declarations.score_names.iter().map(String::as_str).collect();
            scores.sort_unstable();
            if let Some(w) = scores.windows(2).find(|w| w[0] == w[1]) {
                return Err(invalid(format!("genetic score {:?} is declared twice", w[0])));
            }
        }
        let labels: Vec<&str> = tables.events.mark.iter().map(String::as_str).collect();
        let (mark_names, mark_kinds, _) = resolve_mark_vocabulary(declarations.marks.clone(), &labels)?;
        let covariate_levels: Vec<Vec<String>> = tables
            .covariates
            .columns
            .iter()
            .map(|cells| cells.clone().encode().1)
            .collect();
        let cohort = build_cohort(
            &mark_names,
            &mark_kinds,
            &tables.covariates.names,
            &covariate_levels,
            &declarations.visit_process,
            tables,
        )?;
        let rows = design_rows(&cohort, declarations.resolution.quadrature_order)?;
        let baseline = frozen_basis(&declarations.baseline_formula, rows.view(), &cohort, "baseline")?;
        let population =
            frozen_basis(&declarations.population_formula, rows.view(), &cohort, "population")?;
        let drive = declarations
            .drive_formula
            .as_deref()
            .map(|formula| frozen_basis(formula, rows.view(), &cohort, "drive"))
            .transpose()?;
        let entry = declarations
            .entry_formula
            .as_deref()
            .map(|formula| frozen_basis(formula, rows.view(), &cohort, "entry"))
            .transpose()?;
        let schema = Self {
            mark_names,
            mark_kinds,
            covariate_names: tables.covariates.names.clone(),
            covariate_levels,
            channels,
            score_names: declarations.score_names.clone(),
            visit_process: declarations.visit_process.clone(),
            baseline,
            population,
            drive,
            entry,
            resolution: declarations.resolution,
        };
        let subjects = schema.encode_cohort(&cohort, tables)?;
        Ok((schema, subjects))
    }

    /// Encode tables of subjects that were not in the fit, or a training
    /// subject's later records, through the frozen schema. Nothing is relearned:
    /// an unknown mark, level, channel or genetic score is refused.
    pub(super) fn encode(
        &self,
        tables: &JointTables,
    ) -> Result<Vec<EncodedSubject>, JointDataError> {
        let cohort = build_cohort(
            &self.mark_names,
            &self.mark_kinds,
            &self.covariate_names,
            &self.covariate_levels,
            &self.visit_process,
            tables,
        )?;
        self.encode_cohort(&cohort, tables)
    }

    /// Expand every validated subject into a joint history at the schema's
    /// resolution. Latent nodes sit at entry, exit, every event date after
    /// entry, every measurement date, every drive-row change, and in between so
    /// that no gap exceeds the state width. Node `n` owns the compensator cell
    /// `(t_{n-1}, t_n]` and reads the state before its own jumps. Its points are
    /// a zero-weight anchor, then Gauss-Legendre points on the cell split at
    /// covariate changes. The entry node owns only its anchor.
    fn encode_cohort(
        &self,
        cohort: &EventHistoryCohort,
        tables: &JointTables,
    ) -> Result<Vec<EncodedSubject>, JointDataError> {
        let quadrature_order = self.resolution.quadrature_order;
        if quadrature_order == 0 {
            return Err(invalid("quadrature order must be positive"));
        }
        // One certified rule for every cell, so the declared weight error belongs
        // to the weights the histories actually carry.
        let rule = gam_math::special::gauss_legendre_certified(quadrature_order);
        let weight_error = declared_weight_error(&rule)?;
        let index = tables.subject_index()?;
        let n = cohort.subjects.len();
        let channel_names: Vec<String> = self.channels.iter().map(|c| c.name.clone()).collect();

        let mut attended: Vec<Vec<f64>> = vec![Vec::new(); n];
        for ((id, &time), &came) in tables
            .visits
            .id
            .iter()
            .zip(&tables.visits.time)
            .zip(&tables.visits.attended)
        {
            let subject = subject_of(&index, "visits", id)?;
            within_follow_up("visits", &cohort.subjects[subject], time)?;
            if came {
                attended[subject].push(time);
            }
        }

        // Scores map by name; a declared score the table lacks is missing for
        // every subject.
        let g = &tables.genetics;
        let mut positions = Vec::with_capacity(g.score_names.len());
        for name in &g.score_names {
            let position = self
                .score_names
                .iter()
                .position(|declared| declared == name)
                .ok_or_else(|| JointDataError::UnknownGeneticScore {
                    name: name.clone(),
                    names: self.score_names.clone(),
                })?;
            if positions.contains(&position) {
                return Err(invalid(format!(
                    "the genetics table has two {name:?} columns"
                )));
            }
            positions.push(position);
        }
        let mut genetics: Vec<Vec<Option<f64>>> = vec![vec![None; self.score_names.len()]; n];
        let mut has_row = vec![false; n];
        for (row, id) in g.id.iter().enumerate() {
            let subject = subject_of(&index, "genetics", id)?;
            if std::mem::replace(&mut has_row[subject], true) {
                return Err(JointDataError::DuplicateSubject {
                    table: "genetics",
                    id: id.clone(),
                });
            }
            for (column, &position) in g.scores.iter().zip(&positions) {
                genetics[subject][position] = match column[row] {
                    // A NaN cell is how array surfaces spell a missing score.
                    Some(value) if value.is_nan() => None,
                    Some(value) if !value.is_finite() => {
                        return Err(invalid(format!(
                            "subject {id:?}: genetic score {:?} is infinite",
                            self.score_names[position]
                        )));
                    }
                    other => other,
                };
            }
        }

        let m = &tables.measurements;
        let mut measured: Vec<Vec<(f64, usize, Option<f64>, Option<f64>)>> = vec![Vec::new(); n];
        for ((((id, &time), channel), &value), &exposure) in m
            .id
            .iter()
            .zip(&m.time)
            .zip(&m.channel)
            .zip(&m.value)
            .zip(&m.exposure)
        {
            let subject = subject_of(&index, "measurements", id)?;
            let history = &cohort.subjects[subject];
            within_follow_up("measurements", history, time)?;
            let c = channel_names
                .iter()
                .position(|name| name == channel)
                .ok_or_else(|| JointDataError::UnknownChannel {
                    channel: channel.clone(),
                    channels: channel_names.clone(),
                })?;
            if let Some(value) = value
                && super::emission::validate_value(&self.channels[c].family, value).is_err()
            {
                return Err(JointDataError::OutsideSupport {
                    channel: channel.clone(),
                    family: format!("{:?}", self.channels[c].family),
                    value,
                });
            }
            let reason = match (&self.channels[c].family, exposure) {
                (MeasurementFamily::NegativeBinomial, Some(e)) if e.is_finite() && e > 0.0 => None,
                (MeasurementFamily::NegativeBinomial, _) => {
                    Some("a count channel needs a finite positive exposure")
                }
                (_, Some(_)) => Some("only count channels carry an exposure"),
                (_, None) => None,
            };
            if let Some(reason) = reason {
                return Err(JointDataError::Exposure {
                    subject: history.id.clone(),
                    channel: channel.clone(),
                    time,
                    reason,
                });
            }
            if !attended[subject].contains(&time) {
                return Err(JointDataError::MeasurementWithoutVisit {
                    subject: history.id.clone(),
                    time,
                });
            }
            measured[subject].push((time, c, value, exposure));
        }

        // Rows may come in any order: each subject's records are ordered by time,
        // channel, value bits and exposure bits. The key is total, so records that
        // compare equal are the same record, and a table's row order cannot change
        // an encoding.
        let bits = |value: Option<f64>| value.map(f64::to_bits);
        for records in &mut measured {
            records.sort_by(|a, b| {
                a.0.total_cmp(&b.0)
                    .then(a.1.cmp(&b.1))
                    .then(bits(a.2).cmp(&bits(b.2)))
                    .then(bits(a.3).cmp(&bits(b.3)))
            });
        }
        let marks = cohort.marks();
        let mut encoded = Vec::with_capacity(n);
        for (s, subject) in cohort.subjects.iter().enumerate() {
            // Covariate changes inside the follow-up split compensator cells,
            // and they are dated nodes when a drive basis reads the covariates.
            let changes: Vec<f64> = subject
                .segments
                .iter()
                .map(|segment| segment.start)
                .filter(|&t| t > subject.entry && t < subject.exit)
                .collect();
            let drive_changes: &[f64] = if self.drive.is_some() { &changes } else { &[] };
            // Events at or before entry are entry information, not count nodes.
            let after_entry: Vec<&Event> = subject
                .events
                .iter()
                .filter(|e| e.time > subject.entry)
                .collect();
            let dated = after_entry
                .iter()
                .map(|e| e.time)
                .chain(measured[s].iter().map(|&(time, _, _, _)| time))
                .chain(drive_changes.iter().copied());
            let times =
                latent_node_times(subject.entry, subject.exit, dated, self.resolution.state_width)?;
            // Events recorded on one date share its node as an unordered
            // multiset, listed by mark index so the table's row order cannot
            // matter; every row is an event.
            let mut events: Vec<Vec<usize>> = vec![Vec::new(); times.len()];
            for event in &after_entry {
                events[times.partition_point(|&t| t < event.time)].push(event.mark);
            }
            for fired in &mut events {
                fired.sort_unstable();
            }
            let nodes = times.len();
            let placed = placed_points(&times, &events, &changes, &rule);
            // An anchor on an event date reads the left limit of the covariates.
            let point_rows = covariate_rows(
                cohort,
                subject,
                &placed
                    .iter()
                    .map(|p| (p.time, p.left_limit))
                    .collect::<Vec<_>>(),
            );
            let points: Vec<CompensatorPoint> = placed
                .iter()
                .map(|p| CompensatorPoint {
                    node: p.node,
                    time: p.time,
                    weight: p.weight,
                })
                .collect();
            let baseline_design = basis_rows(&self.baseline, point_rows.view(), "baseline")?;
            let population_design =
                basis_rows(&self.population, point_rows.view(), "population")?;
            if population_design.ncols() == 0
                || population_design.column(0).iter().any(|&v| v != 1.0)
            {
                return Err(JointDataError::PopulationConstant {
                    basis: "population",
                });
            }
            // The drive holds the row in force on each gap, read at its left end.
            let drive_design = match &self.drive {
                Some(spec) => {
                    let gap_points: Vec<(f64, bool)> =
                        times[..nodes - 1].iter().map(|&t| (t, false)).collect();
                    basis_rows(spec, covariate_rows(cohort, subject, &gap_points).view(), "drive")?
                }
                None => Array2::zeros((nodes - 1, 0)),
            };
            let entry_design = match &self.entry {
                Some(spec) => basis_rows(
                    spec,
                    covariate_rows(cohort, subject, &[(subject.entry, false)]).view(),
                    "entry",
                )?
                .row(0)
                .to_vec(),
                None => Vec::new(),
            };
            let initially_at_risk: Vec<bool> = (0..marks)
                .map(|d| match cohort.mark_kinds[d] {
                    MarkKind::Once => !subject
                        .events
                        .iter()
                        .any(|e| e.mark == d && e.time <= subject.entry),
                    MarkKind::Recurrent | MarkKind::Terminal => true,
                })
                .collect();
            // A measurement on an event date observes the state the visit
            // contract declares. After termination there is no state to observe.
            let after_events =
                self.visit_process.event_date_state() == EventDateState::AfterEvents;
            let mut measurements: Vec<MeasurementRecord> = Vec::with_capacity(measured[s].len());
            for &(time, channel, value, exposure) in &measured[s] {
                let node = times.partition_point(|&t| t < time);
                let after_event = after_events && !events[node].is_empty();
                if after_event
                    && events[node]
                        .iter()
                        .any(|&d| cohort.mark_kinds[d] == MarkKind::Terminal)
                {
                    return Err(JointDataError::MeasurementAfterTermination {
                        subject: subject.id.clone(),
                        time,
                    });
                }
                measurements.push(MeasurementRecord {
                    node,
                    channel,
                    value,
                    exposure,
                    after_event,
                });
            }
            encoded.push(EncodedSubject {
                id: subject.id.clone(),
                history: JointHistory {
                    times,
                    points,
                    weight_error,
                    events,
                    initially_at_risk,
                    baseline_design,
                    population_design,
                    drive_design,
                    entry_design,
                    genetics: std::mem::take(&mut genetics[s]),
                    measurements,
                },
            });
        }
        Ok(encoded)
    }
}

fn within_follow_up(
    table: &'static str,
    subject: &SubjectHistory,
    time: f64,
) -> Result<(), JointDataError> {
    if time >= subject.entry && time <= subject.exit {
        Ok(())
    } else {
        Err(JointDataError::OutsideFollowUp {
            table,
            subject: subject.id.clone(),
            time,
            entry: subject.entry,
            exit: subject.exit,
        })
    }
}

/// Covariate rows then time, at each `(time, left_limit)` point of a subject.
fn covariate_rows(
    cohort: &EventHistoryCohort,
    subject: &SubjectHistory,
    points: &[(f64, bool)],
) -> Array2<f64> {
    let columns = cohort.covariates.ncols();
    let mut rows = Array2::<f64>::zeros((points.len(), columns + 1));
    for (i, &(t, left_limit)) in points.iter().enumerate() {
        let row = subject.covariate_row_at(t, left_limit);
        rows.slice_mut(s![i, ..columns])
            .assign(&cohort.covariates.row(row));
        rows[[i, columns]] = t;
    }
    rows
}

/// A frozen basis evaluated on rows. The joint law has no fixed-coefficient
/// offset term, so a basis that produces one is refused, not dropped.
fn basis_rows(
    frozen: &FrozenBasis,
    rows: ArrayView2<'_, f64>,
    basis: &'static str,
) -> Result<Array2<f64>, JointDataError> {
    let design = build_term_collection_design(rows, &frozen.spec)
        .map_err(|error| invalid(format!("{basis} design: {error}")))?;
    if design.affine_offset.iter().any(|v| *v != 0.0) {
        return Err(JointDataError::AffineOffset { basis });
    }
    let dense = design
        .design
        .try_to_dense_arc("joint history design")
        .map_err(|error| invalid(error.to_string()))?;
    Ok(dense.as_ref().clone())
}

/// Latent node times of one subject: entry, exit, every dated time strictly
/// inside the follow-up (event dates, measurement dates, drive changes), and
/// interior times splitting each gap between dated times evenly, so that no gap
/// is wider than `state_width`. Dated times never merge, and each gap is split
/// into as few equal pieces as the width allows.
fn latent_node_times(
    entry: f64,
    exit: f64,
    dated: impl IntoIterator<Item = f64>,
    state_width: Option<f64>,
) -> Result<Vec<f64>, JointDataError> {
    if let Some(width) = state_width
        && !(width.is_finite() && width > 0.0)
    {
        return Err(invalid(format!(
            "the latent state width must be finite and positive, got {width}"
        )));
    }
    let mut anchors = vec![entry, exit];
    anchors.extend(dated.into_iter().filter(|&t| t > entry && t < exit));
    anchors.sort_by(f64::total_cmp);
    anchors.dedup();
    let mut times = Vec::with_capacity(anchors.len());
    for pair in anchors.windows(2) {
        let (left, right) = (pair[0], pair[1]);
        // Without a latent state to resolve, only the dated nodes are placed.
        let pieces = match state_width {
            Some(width) => ((right - left) / width).ceil().max(1.0),
            None => 1.0,
        };
        if !(pieces.is_finite() && pieces < (usize::MAX / 2) as f64) {
            return Err(invalid(format!(
                "a state width of {state_width:?} cannot split the gap ({left}, {right})"
            )));
        }
        let pieces = pieces as usize;
        let width = (right - left) / pieces as f64;
        times.push(left);
        times.extend((1..pieces).map(|p| left + width * p as f64));
    }
    times.push(exit);
    if times.windows(2).any(|w| !(w[0] < w[1])) {
        return Err(invalid(format!(
            "a state width of {state_width:?} places latent nodes closer than the times can represent"
        )));
    }
    Ok(times)
}

/// One placed compensator point: the latent node whose pre-jump state it
/// reads, its time, its Gauss-Legendre weight (zero for a node's anchor), and
/// whether its covariate row is the left limit (the anchor of a node with
/// events).
#[derive(Clone, Copy, Debug, PartialEq)]
struct PlacedPoint {
    node: usize,
    time: f64,
    weight: f64,
    left_limit: bool,
}

/// The encoder's declared bound on every cell weight's relative error against its
/// exact share of the cell width. Each weight is `half·ŵ` with `half = 0.5·(b − a)`
/// over its piece: the subtraction and the product each round by at most half an
/// ulp relative to their results, on top of the certified relative error of `ŵ`.
/// The pieces' exact widths telescope to the cell's, so the bound holds per cell.
/// A declined certificate is refused, never clamped.
fn declared_weight_error(
    rule: &gam_math::special::CertifiedGaussLegendreRule,
) -> Result<f64, JointDataError> {
    let declared = rule.weight_relative_error + f64::EPSILON;
    if declared.is_finite() && declared < 1.0 {
        Ok(declared)
    } else {
        Err(JointDataError::QuadratureCertificate {
            order: rule.nodes.len(),
            weight_relative_error: rule.weight_relative_error,
        })
    }
}

/// The compensator points of one subject, sorted by node. Each node's first
/// point is its zero-weight anchor at the node time. The cell `(t_{n-1}, t_n]`
/// belongs to node `n`: it is split again at every covariate change inside it,
/// with the certified rule's points on each piece. The entry node owns no cell,
/// only its anchor. `changes` must be sorted.
fn placed_points(
    node_times: &[f64],
    events: &[Vec<usize>],
    changes: &[f64],
    rule: &gam_math::special::CertifiedGaussLegendreRule,
) -> Vec<PlacedPoint> {
    let mut points = Vec::with_capacity(node_times.len() * (rule.nodes.len() + 1));
    for (n, &right) in node_times.iter().enumerate() {
        points.push(PlacedPoint {
            node: n,
            time: right,
            weight: 0.0,
            left_limit: !events[n].is_empty(),
        });
        if n == 0 {
            continue;
        }
        let left = node_times[n - 1];
        let mut cuts = vec![left];
        cuts.extend(changes.iter().copied().filter(|&c| c > left && c < right));
        cuts.push(right);
        for piece in cuts.windows(2) {
            points.extend(
                cell_rule(piece[0], piece[1], &rule.nodes, &rule.weights).map(|(time, weight)| {
                    PlacedPoint {
                        node: n,
                        time,
                        weight,
                        left_limit: false,
                    }
                }),
            );
        }
    }
    points
}

#[cfg(test)]
mod data_tests;
